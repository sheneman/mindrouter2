############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# health.py: Health check and Prometheus metrics endpoints
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Health check and metrics endpoints."""

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.telemetry.registry import get_registry
from backend.app.db.session import AsyncSessionLocal
from backend.app.settings import get_settings

router = APIRouter(tags=["health"])

# Prometheus metrics
REQUEST_COUNT = Counter(
    "mindrouter_requests_total",
    "Total number of requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "mindrouter_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
QUEUE_SIZE = Gauge(
    "mindrouter_queue_size",
    "Current queue size",
)
ACTIVE_BACKENDS = Gauge(
    "mindrouter_active_backends",
    "Number of healthy backends",
)
TOKENS_PROCESSED = Counter(
    "mindrouter_tokens_total",
    "Total tokens processed",
    ["type"],  # prompt, completion
)


@router.get("/healthz")
async def liveness_probe() -> Dict[str, str]:
    """
    Liveness probe - checks if the application is running.

    Returns 200 if the application is alive.
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/readyz")
async def readiness_probe() -> Dict[str, Any]:
    """
    Readiness probe - checks if the application is ready to serve traffic.

    Checks:
    - Database connectivity
    - At least one healthy backend available
    """
    checks = {
        "database": False,
        "backends": False,
    }

    # Check database
    try:
        async with AsyncSessionLocal() as db:
            await db.execute("SELECT 1")
            checks["database"] = True
    except Exception:
        pass

    # Check backends
    try:
        registry = get_registry()
        backends = await registry.get_healthy_backends()
        checks["backends"] = len(backends) > 0
    except Exception:
        pass

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    try:
        # Update dynamic metrics
        scheduler = get_scheduler()
        stats = await scheduler.get_stats()
        QUEUE_SIZE.set(stats["queue"]["total"])

        registry = get_registry()
        backends = await registry.get_healthy_backends()
        ACTIVE_BACKENDS.set(len(backends))

    except Exception:
        pass

    # Generate metrics
    metrics = generate_latest()
    return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)


@router.get("/status")
async def cluster_status() -> Dict[str, Any]:
    """
    Get cluster status summary.

    Returns high-level status information about the cluster.
    """
    settings = get_settings()

    # Get scheduler stats
    try:
        scheduler = get_scheduler()
        scheduler_stats = await scheduler.get_stats()
    except Exception:
        scheduler_stats = {"error": "unavailable"}

    # Get backend info
    try:
        registry = get_registry()
        all_backends = await registry.get_all_backends()
        healthy_backends = await registry.get_healthy_backends()
    except Exception:
        all_backends = []
        healthy_backends = []

    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backends": {
            "total": len(all_backends),
            "healthy": len(healthy_backends),
        },
        "queue": scheduler_stats.get("queue", {}),
        "fair_share": {
            "total_users": scheduler_stats.get("fair_share", {}).get("total_users", 0),
        },
    }
