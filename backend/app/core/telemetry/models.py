############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# models.py: Telemetry data models for backends and GPUs
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Telemetry data models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class ModelInfo:
    """Information about a model on a backend."""

    name: str
    family: Optional[str] = None
    parameter_count: Optional[str] = None  # "7B", "70B", etc.
    context_length: Optional[int] = None
    supports_vision: bool = False
    supports_structured_output: bool = True
    is_loaded: bool = False
    vram_required_gb: Optional[float] = None


@dataclass
class GPUInfo:
    """GPU information from a backend."""

    utilization: Optional[float] = None  # 0-100
    memory_used_gb: Optional[float] = None
    memory_total_gb: Optional[float] = None
    temperature: Optional[float] = None
    name: Optional[str] = None


@dataclass
class BackendCapabilities:
    """Capabilities discovered from a backend."""

    engine_version: Optional[str] = None
    models: List[ModelInfo] = field(default_factory=list)
    loaded_models: List[str] = field(default_factory=list)
    gpu_info: Optional[GPUInfo] = None

    supports_vision: bool = False
    supports_embeddings: bool = False
    supports_structured_output: bool = True

    max_concurrent: int = 4
    is_healthy: bool = False
    error_message: Optional[str] = None

    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TelemetrySnapshot:
    """Point-in-time telemetry snapshot from a backend."""

    backend_id: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # GPU metrics
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None

    # Request metrics
    active_requests: int = 0
    queued_requests: int = 0
    requests_per_second: Optional[float] = None

    # Model state
    loaded_models: List[str] = field(default_factory=list)

    # Health
    is_healthy: bool = True
    latency_ms: Optional[float] = None


@dataclass
class BackendHealth:
    """Health check result for a backend."""

    is_healthy: bool
    status_code: Optional[int] = None
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
