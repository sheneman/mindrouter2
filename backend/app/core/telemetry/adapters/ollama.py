############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# ollama.py: Ollama backend adapter for telemetry collection
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Ollama backend adapter for capability discovery and telemetry."""

import asyncio
from typing import Optional
import time

import httpx

from backend.app.core.telemetry.models import (
    BackendCapabilities,
    BackendHealth,
    GPUInfo,
    ModelInfo,
    TelemetrySnapshot,
)
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)


class OllamaAdapter:
    """
    Adapter for Ollama backend telemetry and capability discovery.

    Ollama API endpoints used:
    - GET /api/version - Engine version
    - GET /api/tags - List available models
    - POST /api/ps - List running/loaded models
    - GET /api/show - Get model details (optional)
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> BackendHealth:
        """
        Perform a health check on the Ollama backend.

        Returns:
            BackendHealth result
        """
        start_time = time.monotonic()
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")

            latency_ms = (time.monotonic() - start_time) * 1000

            if response.status_code == 200:
                return BackendHealth(
                    is_healthy=True,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                )
            else:
                return BackendHealth(
                    is_healthy=False,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    error_message=f"HTTP {response.status_code}",
                )

        except httpx.TimeoutException:
            latency_ms = (time.monotonic() - start_time) * 1000
            return BackendHealth(
                is_healthy=False,
                latency_ms=latency_ms,
                error_message="Connection timeout",
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            return BackendHealth(
                is_healthy=False,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    async def discover_capabilities(self) -> BackendCapabilities:
        """
        Discover backend capabilities.

        Returns:
            BackendCapabilities with models, version, and features
        """
        caps = BackendCapabilities()

        try:
            client = await self._get_client()

            # Get version
            version = await self._get_version(client)
            caps.engine_version = version

            # Get models
            models = await self._get_models(client)
            caps.models = models

            # Get loaded models
            loaded = await self._get_loaded_models(client)
            caps.loaded_models = loaded

            # Mark which models are loaded
            for model in caps.models:
                model.is_loaded = model.name in loaded

            # Determine capabilities based on models
            for model in caps.models:
                if model.supports_vision:
                    caps.supports_vision = True
                # Check for embedding models
                if "embed" in model.name.lower() or "embedding" in model.name.lower():
                    caps.supports_embeddings = True

            caps.is_healthy = True

        except Exception as e:
            logger.warning("ollama_capability_discovery_failed", error=str(e))
            caps.is_healthy = False
            caps.error_message = str(e)

        return caps

    async def get_telemetry(self, backend_id: int) -> TelemetrySnapshot:
        """
        Get current telemetry from the backend.

        Args:
            backend_id: Backend ID for the snapshot

        Returns:
            TelemetrySnapshot with current metrics
        """
        snapshot = TelemetrySnapshot(backend_id=backend_id)

        try:
            client = await self._get_client()

            # Get loaded models
            loaded = await self._get_loaded_models(client)
            snapshot.loaded_models = loaded

            # Try to get GPU info if available
            # Ollama doesn't expose this directly, but some setups have a sidecar
            gpu_info = await self._try_get_gpu_info(client)
            if gpu_info:
                snapshot.gpu_utilization = gpu_info.utilization
                snapshot.gpu_memory_used_gb = gpu_info.memory_used_gb
                snapshot.gpu_memory_total_gb = gpu_info.memory_total_gb
                snapshot.gpu_temperature = gpu_info.temperature

            snapshot.is_healthy = True

        except Exception as e:
            logger.debug("ollama_telemetry_failed", error=str(e))
            snapshot.is_healthy = False

        return snapshot

    async def _get_version(self, client: httpx.AsyncClient) -> Optional[str]:
        """Get Ollama version."""
        try:
            response = await client.get("/api/version")
            if response.status_code == 200:
                data = response.json()
                return data.get("version")
        except Exception:
            pass

        # Try parsing from headers
        try:
            response = await client.get("/api/tags")
            version = response.headers.get("x-ollama-version")
            return version
        except Exception:
            pass

        return None

    async def _get_models(self, client: httpx.AsyncClient) -> list[ModelInfo]:
        """Get list of available models."""
        models = []

        try:
            response = await client.get("/api/tags")
            if response.status_code != 200:
                return models

            data = response.json()
            for model_data in data.get("models", []):
                name = model_data.get("name", "")
                if not name:
                    continue

                # Parse model details
                details = model_data.get("details", {})

                # Determine if model supports vision
                supports_vision = False
                family = details.get("family", "").lower()
                name_lower = name.lower()
                if any(x in name_lower for x in ["llava", "vision", "-vl-", "-vl:"]):
                    supports_vision = True

                # Estimate parameter count from name or details
                param_count = details.get("parameter_size")
                if not param_count:
                    # Try to extract from name (e.g., "llama3.2:7b")
                    for size in ["70b", "34b", "13b", "7b", "3b", "1b"]:
                        if size in name.lower():
                            param_count = size.upper()
                            break

                models.append(
                    ModelInfo(
                        name=name,
                        family=details.get("family"),
                        parameter_count=param_count,
                        context_length=details.get("context_length"),
                        supports_vision=supports_vision,
                        supports_structured_output=True,  # Most Ollama models do
                    )
                )

        except Exception as e:
            logger.warning("ollama_get_models_failed", error=str(e))

        return models

    async def _get_loaded_models(self, client: httpx.AsyncClient) -> list[str]:
        """Get list of currently loaded models."""
        loaded = []

        try:
            # Ollama 0.1.24+ supports /api/ps
            response = await client.post("/api/ps", json={})
            if response.status_code == 200:
                data = response.json()
                for model in data.get("models", []):
                    name = model.get("name")
                    if name:
                        loaded.append(name)
        except Exception:
            # Older Ollama versions don't have /api/ps
            pass

        return loaded

    async def _try_get_gpu_info(self, client: httpx.AsyncClient) -> Optional[GPUInfo]:
        """
        Try to get GPU info from a sidecar or custom endpoint.

        Some Ollama deployments have a GPU metrics sidecar at /gpu-info.
        """
        try:
            response = await client.get("/gpu-info")
            if response.status_code == 200:
                data = response.json()
                return GPUInfo(
                    utilization=data.get("utilization"),
                    memory_used_gb=data.get("memory_used_gb"),
                    memory_total_gb=data.get("memory_total_gb"),
                    temperature=data.get("temperature"),
                    name=data.get("name"),
                )
        except Exception:
            pass

        return None
