############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# registry.py: Backend registry for discovery, health, and telemetry
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Backend registry - manages backend discovery, health, and telemetry."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.telemetry.adapters.ollama import OllamaAdapter
from backend.app.core.telemetry.adapters.vllm import VLLMAdapter
from backend.app.core.telemetry.latency_tracker import LatencyTracker
from backend.app.core.telemetry.models import (
    BackendCapabilities,
    BackendHealth,
    CircuitBreakerState,
    TelemetrySnapshot,
)
from backend.app.db import crud
from backend.app.db.models import Backend, BackendEngine, BackendStatus, Model, Modality
from backend.app.db.session import get_async_db_context
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)


class BackendRegistry:
    """
    Central registry for backend management.

    Responsibilities:
    - Maintain list of registered backends
    - Periodically poll backends for health and capabilities
    - Store telemetry snapshots
    - Provide backend/model lookup for scheduler
    """

    def __init__(self):
        self._settings = get_settings()
        self._adapters: Dict[int, OllamaAdapter | VLLMAdapter] = {}
        self._capabilities: Dict[int, BackendCapabilities] = {}
        self._telemetry: Dict[int, TelemetrySnapshot] = {}
        self._lock = asyncio.Lock()
        self._poll_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None

        # Circuit breaker state per backend
        self._circuit_breakers: Dict[int, CircuitBreakerState] = {}

        # Fast-poll set: backend_id -> fast_poll_until timestamp
        self._fast_poll_backends: Dict[int, datetime] = {}

        # Latency tracker
        self._latency_tracker = LatencyTracker(
            alpha=self._settings.latency_ema_alpha
        )

    @property
    def latency_tracker(self) -> LatencyTracker:
        """Access the latency tracker."""
        return self._latency_tracker

    async def start(self) -> None:
        """Start the registry and begin polling."""
        # Load existing backends from database
        await self._load_backends_from_db()

        # Load persisted latency EMAs
        async with get_async_db_context() as db:
            all_backends = await crud.get_all_backends(db=db)
        await self._latency_tracker.load_from_db(all_backends)

        # Load persisted circuit breaker state
        for b in all_backends:
            cb = CircuitBreakerState(
                live_failure_count=getattr(b, "live_failure_count", 0) or 0,
                circuit_open_until=getattr(b, "circuit_open_until", None),
            )
            self._circuit_breakers[b.id] = cb

        # Start background tasks
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._persist_task = asyncio.create_task(self._persist_latency_loop())
        logger.info("Backend registry started")

    async def stop(self) -> None:
        """Stop the registry."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._persist_task:
            self._persist_task.cancel()
            try:
                await self._persist_task
            except asyncio.CancelledError:
                pass

        # Final persist of latency data before shutdown
        try:
            await self._persist_latency_data()
        except Exception as e:
            logger.warning("shutdown_persist_error", error=str(e))

        # Close all adapters
        for adapter in self._adapters.values():
            await adapter.close()

        logger.info("Backend registry stopped")

    async def register_backend(
        self,
        name: str,
        url: str,
        engine: BackendEngine,
        max_concurrent: int = 4,
        gpu_memory_gb: Optional[float] = None,
        gpu_type: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Backend:
        """
        Register a new backend.

        Args:
            name: Unique backend name
            url: Backend URL
            engine: Backend engine type
            max_concurrent: Max concurrent requests
            gpu_memory_gb: GPU memory in GB
            gpu_type: GPU type description
            db: Optional database session

        Returns:
            Created Backend record
        """
        async with self._lock:
            # Create database record
            if db:
                backend = await crud.create_backend(
                    db=db,
                    name=name,
                    url=url,
                    engine=engine,
                    max_concurrent=max_concurrent,
                    gpu_memory_gb=gpu_memory_gb,
                    gpu_type=gpu_type,
                )
            else:
                async with get_async_db_context() as db_session:
                    backend = await crud.create_backend(
                        db=db_session,
                        name=name,
                        url=url,
                        engine=engine,
                        max_concurrent=max_concurrent,
                        gpu_memory_gb=gpu_memory_gb,
                        gpu_type=gpu_type,
                    )

            # Create adapter
            adapter = self._create_adapter(backend)
            self._adapters[backend.id] = adapter

        # Run discovery outside the lock to avoid blocking telemetry reads
        await self._discover_backend(backend.id)

        logger.info(
            "backend_registered",
            backend_id=backend.id,
            name=name,
            engine=engine.value,
        )

        return backend

    async def refresh_backend(self, backend_id: int) -> bool:
        """
        Force refresh capabilities for a backend.

        Args:
            backend_id: Backend ID to refresh

        Returns:
            True if refresh succeeded
        """
        if backend_id not in self._adapters:
            return False

        await self._discover_backend(backend_id)
        return True

    async def disable_backend(self, backend_id: int) -> bool:
        """Disable a backend."""
        async with get_async_db_context() as db:
            backend = await crud.update_backend_status(
                db=db,
                backend_id=backend_id,
                status=BackendStatus.DISABLED,
            )
            return backend is not None

    async def remove_backend(self, backend_id: int) -> bool:
        """Remove a backend entirely (unregister)."""
        async with self._lock:
            # Close and remove adapter
            adapter = self._adapters.pop(backend_id, None)
            if adapter:
                await adapter.close()

            # Remove cached capabilities and telemetry
            self._capabilities.pop(backend_id, None)
            self._telemetry.pop(backend_id, None)

        # Delete from database
        async with get_async_db_context() as db:
            deleted = await crud.delete_backend(db=db, backend_id=backend_id)

        if deleted:
            logger.info("backend_removed", backend_id=backend_id)
        return deleted

    async def enable_backend(self, backend_id: int) -> bool:
        """Enable a previously disabled backend."""
        async with get_async_db_context() as db:
            backend = await crud.update_backend_status(
                db=db,
                backend_id=backend_id,
                status=BackendStatus.UNKNOWN,
            )

        # Trigger immediate health check
        if backend_id in self._adapters:
            await self._check_backend_health(backend_id)

        return True

    async def get_healthy_backends(
        self,
        engine: Optional[BackendEngine] = None,
    ) -> List[Backend]:
        """Get all healthy backends."""
        async with get_async_db_context() as db:
            return await crud.get_healthy_backends(db=db, engine=engine)

    async def get_all_backends(self) -> List[Backend]:
        """Get all backends."""
        async with get_async_db_context() as db:
            return await crud.get_all_backends(db=db)

    async def get_backend_models(self, backend_id: int) -> List[Model]:
        """Get models for a backend."""
        async with get_async_db_context() as db:
            return await crud.get_models_for_backend(db=db, backend_id=backend_id)

    async def get_backends_with_model(
        self,
        model_name: str,
        modality: Optional[Modality] = None,
    ) -> List[Backend]:
        """Get backends that have a specific model."""
        async with get_async_db_context() as db:
            return await crud.get_backends_with_model(
                db=db,
                model_name=model_name,
                modality=modality,
            )

    async def get_gpu_utilizations(self) -> Dict[int, Optional[float]]:
        """Get current GPU utilization for all backends."""
        async with self._lock:
            return {
                bid: t.gpu_utilization
                for bid, t in self._telemetry.items()
            }

    async def get_telemetry(self, backend_id: int) -> Optional[TelemetrySnapshot]:
        """Get latest telemetry for a backend."""
        async with self._lock:
            return self._telemetry.get(backend_id)

    async def get_capabilities(self, backend_id: int) -> Optional[BackendCapabilities]:
        """Get discovered capabilities for a backend."""
        async with self._lock:
            return self._capabilities.get(backend_id)

    # ------------------------------------------------------------------
    # Circuit breaker & reactive health
    # ------------------------------------------------------------------

    async def report_live_failure(self, backend_id: int) -> None:
        """Called when a live request to a backend fails.

        Increments failure count. If threshold reached, opens circuit and
        marks backend UNHEALTHY immediately. Activates fast-polling.
        """
        cb = self._circuit_breakers.setdefault(backend_id, CircuitBreakerState())
        cb.live_failure_count += 1
        cb.last_failure_time = datetime.now(timezone.utc)

        threshold = self._settings.backend_circuit_breaker_threshold
        if cb.live_failure_count >= threshold and not cb.is_open:
            recovery = self._settings.backend_circuit_breaker_recovery_seconds
            cb.circuit_open_until = datetime.now(timezone.utc) + timedelta(seconds=recovery)

            # Mark UNHEALTHY in DB immediately
            try:
                async with get_async_db_context() as db:
                    await crud.update_backend_status(
                        db=db,
                        backend_id=backend_id,
                        status=BackendStatus.UNHEALTHY,
                    )
                    await crud.update_backend_circuit_breaker(
                        db=db,
                        backend_id=backend_id,
                        live_failure_count=cb.live_failure_count,
                        circuit_open_until=cb.circuit_open_until,
                    )
            except Exception as e:
                logger.warning("circuit_breaker_db_error", backend_id=backend_id, error=str(e))

            # Activate fast polling
            fast_duration = self._settings.backend_adaptive_poll_fast_duration
            self._fast_poll_backends[backend_id] = (
                datetime.now(timezone.utc) + timedelta(seconds=fast_duration)
            )

            logger.warning(
                "circuit_breaker_opened",
                backend_id=backend_id,
                failures=cb.live_failure_count,
                recovery_seconds=recovery,
            )

    async def report_live_success(self, backend_id: int) -> None:
        """Called when a live request to a backend succeeds.

        Resets failure count. If circuit was half-open, closes it and
        marks backend HEALTHY.
        """
        cb = self._circuit_breakers.get(backend_id)
        if cb is None:
            return

        was_half_open = cb.is_half_open
        cb.live_failure_count = 0
        cb.circuit_open_until = None
        cb.last_failure_time = None

        if was_half_open:
            # Circuit recovered — mark healthy
            try:
                async with get_async_db_context() as db:
                    await crud.update_backend_status(
                        db=db,
                        backend_id=backend_id,
                        status=BackendStatus.HEALTHY,
                    )
                    await crud.update_backend_circuit_breaker(
                        db=db,
                        backend_id=backend_id,
                        live_failure_count=0,
                        circuit_open_until=None,
                    )
            except Exception as e:
                logger.warning("circuit_close_db_error", backend_id=backend_id, error=str(e))

            # Remove from fast-poll set
            self._fast_poll_backends.pop(backend_id, None)

            logger.info("circuit_breaker_closed", backend_id=backend_id)

    async def is_backend_available(self, backend_id: int) -> bool:
        """Check if backend is available (circuit not open).

        Half-open circuits ARE available (they allow a single probe request).
        """
        cb = self._circuit_breakers.get(backend_id)
        if cb is None:
            return True
        return not cb.is_open

    # ------------------------------------------------------------------
    # Latency persistence
    # ------------------------------------------------------------------

    async def _persist_latency_loop(self) -> None:
        """Periodically persist latency EMA values to DB."""
        while True:
            try:
                await asyncio.sleep(self._settings.latency_ema_persist_interval)
                await self._persist_latency_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("persist_latency_error", error=str(e))

    async def _persist_latency_data(self) -> None:
        """Write current latency EMA and throughput scores to DB."""
        all_latencies = await self._latency_tracker.get_all_latencies()
        if not all_latencies:
            return

        async with get_async_db_context() as db:
            for backend_id, latency_ema in all_latencies.items():
                ttft_ema = await self._latency_tracker.get_ttft_ema(backend_id)
                throughput = self._latency_tracker.compute_throughput_score(backend_id)
                await crud.update_backend_latency_ema(
                    db=db,
                    backend_id=backend_id,
                    latency_ema_ms=latency_ema,
                    ttft_ema_ms=ttft_ema,
                    throughput_score=throughput,
                )

    def _create_adapter(self, backend: Backend) -> OllamaAdapter | VLLMAdapter:
        """Create the appropriate adapter for a backend."""
        timeout = self._settings.backend_health_timeout

        if backend.engine == BackendEngine.OLLAMA:
            return OllamaAdapter(backend.url, timeout=timeout)
        else:
            return VLLMAdapter(backend.url, timeout=timeout)

    async def _load_backends_from_db(self) -> None:
        """Load existing backends from database on startup."""
        async with get_async_db_context() as db:
            backends = await crud.get_all_backends(db=db)

        for backend in backends:
            adapter = self._create_adapter(backend)
            self._adapters[backend.id] = adapter

        logger.info("loaded_backends", count=len(backends))

    async def _poll_loop(self) -> None:
        """Background polling loop with adaptive intervals for troubled backends."""
        last_full_poll = 0.0
        while True:
            try:
                now = datetime.now(timezone.utc)

                # Clean up expired fast-poll entries
                self._fast_poll_backends = {
                    bid: until
                    for bid, until in self._fast_poll_backends.items()
                    if now < until
                }

                # Fast-poll troubled backends at the fast interval
                if self._fast_poll_backends:
                    fast_tasks = [
                        self._check_backend_health(bid)
                        for bid in self._fast_poll_backends
                    ]
                    await asyncio.gather(*fast_tasks, return_exceptions=True)

                # Determine sleep interval
                if self._fast_poll_backends:
                    interval = self._settings.backend_adaptive_poll_fast_interval
                else:
                    interval = self._settings.backend_poll_interval

                await asyncio.sleep(interval)

                # Full poll on normal interval
                import time as _time
                elapsed = _time.monotonic() - last_full_poll
                if elapsed >= self._settings.backend_poll_interval:
                    await self._poll_all_backends()
                    last_full_poll = _time.monotonic()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("poll_loop_error", error=str(e))

    async def _poll_all_backends(self) -> None:
        """Poll all backends for health and telemetry."""
        async with self._lock:
            backend_ids = list(self._adapters.keys())

        # Run health checks in parallel
        tasks = [
            self._check_backend_health(bid)
            for bid in backend_ids
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_backend_health(self, backend_id: int) -> None:
        """Check health of a single backend."""
        adapter = self._adapters.get(backend_id)
        if not adapter:
            return

        try:
            health = await adapter.health_check()

            # Update database
            async with get_async_db_context() as db:
                if health.is_healthy:
                    await crud.update_backend_status(
                        db=db,
                        backend_id=backend_id,
                        status=BackendStatus.HEALTHY,
                    )

                    # If circuit was open/half-open, close it on successful health check
                    cb = self._circuit_breakers.get(backend_id)
                    if cb and (cb.is_open or cb.is_half_open):
                        cb.live_failure_count = 0
                        cb.circuit_open_until = None
                        cb.last_failure_time = None
                        await crud.update_backend_circuit_breaker(
                            db=db,
                            backend_id=backend_id,
                            live_failure_count=0,
                            circuit_open_until=None,
                        )
                        self._fast_poll_backends.pop(backend_id, None)
                        logger.info("circuit_breaker_closed_by_health_check", backend_id=backend_id)
                else:
                    # Check consecutive failures
                    backend = await crud.get_backend_by_id(db, backend_id)
                    if backend:
                        failures = backend.consecutive_failures + 1
                        if failures >= self._settings.backend_unhealthy_threshold:
                            await crud.update_backend_status(
                                db=db,
                                backend_id=backend_id,
                                status=BackendStatus.UNHEALTHY,
                            )
                        else:
                            # Increment failure count but don't mark unhealthy yet
                            backend.consecutive_failures = failures
                            await db.flush()

            # Get telemetry if healthy
            if health.is_healthy:
                # Auto-discover if no capabilities stored yet
                if backend_id not in self._capabilities:
                    await self._discover_backend(backend_id)
                await self._collect_telemetry(backend_id)

        except Exception as e:
            logger.warning(
                "health_check_error",
                backend_id=backend_id,
                error=str(e),
            )

    async def _discover_backend(self, backend_id: int) -> None:
        """Discover capabilities for a backend."""
        adapter = self._adapters.get(backend_id)
        if not adapter:
            return

        try:
            caps = await adapter.discover_capabilities()

            async with self._lock:
                self._capabilities[backend_id] = caps

            # Update database with discovered info
            async with get_async_db_context() as db:
                backend = await crud.get_backend_by_id(db, backend_id)
                if backend:
                    # Update backend capabilities
                    backend.supports_vision = caps.supports_vision
                    backend.supports_embeddings = caps.supports_embeddings
                    backend.supports_structured_output = caps.supports_structured_output
                    backend.version = caps.engine_version

                    if caps.is_healthy:
                        backend.status = BackendStatus.HEALTHY
                    else:
                        backend.status = BackendStatus.UNHEALTHY

                    await db.flush()

                    # Update models
                    for model_info in caps.models:
                        modality = Modality.CHAT
                        if model_info.supports_vision:
                            modality = Modality.VISION
                        elif "embed" in model_info.name.lower():
                            modality = Modality.EMBEDDING

                        await crud.upsert_model(
                            db=db,
                            backend_id=backend_id,
                            name=model_info.name,
                            modality=modality,
                            context_length=model_info.context_length,
                            supports_vision=model_info.supports_vision,
                            supports_structured_output=model_info.supports_structured_output,
                            is_loaded=model_info.is_loaded,
                        )

            logger.info(
                "backend_discovered",
                backend_id=backend_id,
                models=len(caps.models),
                version=caps.engine_version,
            )

        except Exception as e:
            logger.warning(
                "discovery_error",
                backend_id=backend_id,
                error=str(e),
            )

    async def _collect_telemetry(self, backend_id: int) -> None:
        """Collect telemetry from a backend."""
        adapter = self._adapters.get(backend_id)
        if not adapter:
            return

        try:
            snapshot = await adapter.get_telemetry(backend_id)

            async with self._lock:
                self._telemetry[backend_id] = snapshot

            # Store in database
            async with get_async_db_context() as db:
                await crud.create_telemetry_snapshot(
                    db=db,
                    backend_id=backend_id,
                    gpu_utilization=snapshot.gpu_utilization,
                    gpu_memory_used_gb=snapshot.gpu_memory_used_gb,
                    gpu_memory_total_gb=snapshot.gpu_memory_total_gb,
                    active_requests=snapshot.active_requests,
                    queued_requests=snapshot.queued_requests,
                    loaded_models=snapshot.loaded_models,
                )

        except Exception as e:
            logger.debug(
                "telemetry_error",
                backend_id=backend_id,
                error=str(e),
            )


# Global registry instance
_registry: Optional[BackendRegistry] = None


def get_registry() -> BackendRegistry:
    """Get the global registry instance."""
    global _registry
    if _registry is None:
        _registry = BackendRegistry()
    return _registry


async def init_registry() -> BackendRegistry:
    """Initialize and start the global registry."""
    registry = get_registry()
    await registry.start()
    return registry


async def shutdown_registry() -> None:
    """Shutdown the global registry."""
    global _registry
    if _registry:
        await _registry.stop()
        _registry = None
