############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# gpu_agent.py: GPU metrics sidecar agent using pynvml
#
# Runs on each GPU inference node to expose per-GPU hardware
# metrics via a lightweight HTTP API. MindRouter2's backend
# registry polls this endpoint to collect telemetry.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""GPU metrics sidecar agent using NVIDIA Management Library (pynvml)."""

import os
import secrets
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse


def _read_sidecar_version() -> str:
    """Read version from VERSION file in the sidecar directory."""
    try:
        version_file = Path(__file__).resolve().parent / "VERSION"
        return version_file.read_text().strip()
    except Exception:
        return "0.0.0"


SIDECAR_VERSION = _read_sidecar_version()

# Require SIDECAR_SECRET_KEY at startup
SIDECAR_SECRET_KEY = os.environ.get("SIDECAR_SECRET_KEY", "").strip()
if not SIDECAR_SECRET_KEY:
    print(
        "FATAL: SIDECAR_SECRET_KEY environment variable is required but not set.\n"
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\"",
        file=sys.stderr,
    )
    sys.exit(1)


async def verify_sidecar_key(x_sidecar_key: Optional[str] = Header(None)) -> None:
    """Validate the X-Sidecar-Key header against the configured secret."""
    if x_sidecar_key is None or not secrets.compare_digest(
        x_sidecar_key, SIDECAR_SECRET_KEY
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing sidecar key")


app = FastAPI(title="MindRouter2 GPU Sidecar Agent", version=SIDECAR_VERSION)

# GPU state cached at startup
_initialized = False
_init_error: Optional[str] = None
_driver_version: Optional[str] = None
_cuda_version: Optional[str] = None
_device_count: int = 0


def _init_nvml() -> None:
    """Initialize NVML library and cache static info."""
    global _initialized, _init_error, _driver_version, _cuda_version, _device_count

    try:
        import pynvml
        pynvml.nvmlInit()
        _driver_version = pynvml.nvmlSystemGetDriverVersion()
        _cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        # Convert CUDA version from int (e.g., 12040) to string (e.g., "12.4")
        if isinstance(_cuda_version, int):
            major = _cuda_version // 1000
            minor = (_cuda_version % 1000) // 10
            _cuda_version = f"{major}.{minor}"
        _device_count = pynvml.nvmlDeviceGetCount()
        _initialized = True
    except Exception as e:
        _init_error = str(e)
        _initialized = False


def _get_gpu_info(index: int) -> Dict[str, Any]:
    """Collect metrics for a single GPU device."""
    import pynvml

    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    info: Dict[str, Any] = {"index": index}

    # Device name
    try:
        info["name"] = pynvml.nvmlDeviceGetName(handle)
    except Exception:
        info["name"] = None

    # UUID
    try:
        info["uuid"] = pynvml.nvmlDeviceGetUUID(handle)
    except Exception:
        info["uuid"] = None

    # PCI bus ID
    try:
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        info["pci_bus_id"] = pci_info.busId.decode() if isinstance(pci_info.busId, bytes) else pci_info.busId
    except Exception:
        info["pci_bus_id"] = None

    # Compute capability
    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        info["compute_capability"] = f"{major}.{minor}"
    except Exception:
        info["compute_capability"] = None

    # Memory
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info["memory_total_gb"] = round(mem.total / (1024**3), 2)
        info["memory_used_gb"] = round(mem.used / (1024**3), 2)
        info["memory_free_gb"] = round(mem.free / (1024**3), 2)
    except Exception:
        info["memory_total_gb"] = None
        info["memory_used_gb"] = None
        info["memory_free_gb"] = None

    # Utilization
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        info["utilization_gpu"] = util.gpu
        info["utilization_memory"] = util.memory
    except Exception:
        info["utilization_gpu"] = None
        info["utilization_memory"] = None

    # Temperature
    try:
        info["temperature_gpu"] = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
    except Exception:
        info["temperature_gpu"] = None

    # Memory temperature (not available on all GPUs)
    try:
        info["temperature_memory"] = pynvml.nvmlDeviceGetTemperature(
            handle, 2  # NVML_TEMPERATURE_MEM = 2 (not always in pynvml constants)
        )
    except Exception:
        info["temperature_memory"] = None

    # Power
    try:
        info["power_draw_watts"] = round(
            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1
        )  # milliwatts -> watts
    except Exception:
        info["power_draw_watts"] = None

    try:
        info["power_limit_watts"] = round(
            pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0, 1
        )
    except Exception:
        info["power_limit_watts"] = None

    # Fan speed
    try:
        info["fan_speed_percent"] = pynvml.nvmlDeviceGetFanSpeed(handle)
    except Exception:
        info["fan_speed_percent"] = None

    # Clocks
    try:
        info["clock_sm_mhz"] = pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_SM
        )
    except Exception:
        info["clock_sm_mhz"] = None

    try:
        info["clock_memory_mhz"] = pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_MEM
        )
    except Exception:
        info["clock_memory_mhz"] = None

    # Running processes
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        info["processes"] = [
            {
                "pid": p.pid,
                "gpu_memory_used_mb": round(p.usedGpuMemory / (1024**2), 1)
                if p.usedGpuMemory
                else None,
            }
            for p in procs
        ]
    except Exception:
        info["processes"] = []

    return info


@app.on_event("startup")
async def startup():
    """Initialize NVML on startup."""
    _init_nvml()


@app.get("/health")
async def health(_: None = Depends(verify_sidecar_key)):
    """Health check endpoint."""
    if _initialized:
        return {"status": "ok", "gpu_count": _device_count, "sidecar_version": SIDECAR_VERSION}
    return JSONResponse(
        status_code=503,
        content={
            "status": "error",
            "error": _init_error or "NVML not initialized",
            "sidecar_version": SIDECAR_VERSION,
        },
    )


@app.get("/gpu-info")
async def gpu_info(_: None = Depends(verify_sidecar_key)):
    """Return detailed GPU metrics for all devices on this node."""
    if not _initialized:
        return JSONResponse(
            status_code=503,
            content={
                "error": _init_error or "NVML not initialized",
                "hostname": socket.gethostname(),
                "gpu_count": 0,
                "gpus": [],
            },
        )

    gpus: List[Dict[str, Any]] = []
    for i in range(_device_count):
        try:
            gpus.append(_get_gpu_info(i))
        except Exception as e:
            gpus.append({"index": i, "error": str(e)})

    return {
        "hostname": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "driver_version": _driver_version,
        "cuda_version": _cuda_version,
        "gpu_count": _device_count,
        "gpus": gpus,
        "sidecar_version": SIDECAR_VERSION,
    }


def _get_pid_to_gpus() -> Dict[int, List[int]]:
    """Build PID→GPU index mapping from NVML process info."""
    if not _initialized:
        return {}

    import pynvml

    pid_to_gpus: Dict[int, List[int]] = {}
    for idx in range(_device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for p in procs:
                pid_to_gpus.setdefault(p.pid, []).append(idx)
        except Exception:
            continue
    return pid_to_gpus


def _get_listening_ports() -> Dict[int, List[int]]:
    """Parse /proc/net/tcp{,6} to find PID→listening ports mapping.

    Returns dict of pid -> list of ports for PIDs that have LISTEN sockets.
    Only works on Linux where /proc is available.
    """
    # Step 1: Find all listening socket inodes from /proc/net/tcp and tcp6
    listen_inodes: Dict[int, int] = {}  # inode -> port
    for tcp_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(tcp_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 10 or parts[0] == "sl":
                        continue
                    # State 0A = LISTEN
                    if parts[3] == "0A":
                        # local_address is hex ip:port
                        port = int(parts[1].split(":")[1], 16)
                        inode = int(parts[9])
                        if port > 0:
                            listen_inodes[inode] = port
        except FileNotFoundError:
            continue

    if not listen_inodes:
        return {}

    # Step 2: For each PID, scan /proc/{pid}/fd/ to find socket inodes
    pid_to_ports: Dict[int, List[int]] = {}
    try:
        proc_entries = os.listdir("/proc")
    except OSError:
        return {}

    for entry in proc_entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    link = os.readlink(f"{fd_dir}/{fd}")
                    if link.startswith("socket:["):
                        inode = int(link[8:-1])
                        if inode in listen_inodes:
                            port = listen_inodes[inode]
                            pid_to_ports.setdefault(pid, []).append(port)
                except (OSError, ValueError):
                    continue
        except OSError:
            continue

    return pid_to_ports


async def _probe_endpoint(port: int) -> Optional[Dict[str, Any]]:
    """Probe a local port to detect vLLM or Ollama inference endpoint.

    Returns dict with engine, models if detected, or None.
    """
    import httpx

    async with httpx.AsyncClient(timeout=3.0) as client:
        # Try vLLM: GET /v1/models
        try:
            resp = await client.get(f"http://localhost:{port}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and isinstance(data["data"], list):
                    models = [
                        {"id": m.get("id", "unknown")}
                        for m in data["data"]
                    ]
                    return {"engine": "vllm", "models": models}
        except Exception:
            pass

        # Try Ollama: GET /api/tags
        try:
            resp = await client.get(f"http://localhost:{port}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                if "models" in data and isinstance(data["models"], list):
                    models = [
                        {"id": m.get("name", "unknown")}
                        for m in data["models"]
                    ]
                    return {"engine": "ollama", "models": models}
        except Exception:
            pass

    return None


@app.get("/discover")
async def discover_endpoints(_: None = Depends(verify_sidecar_key)):
    """Discover inference endpoints running on this node.

    Uses PID→GPU mapping (from NVML) and PID→port mapping (from /proc/net/tcp)
    to find local inference servers, then probes them to identify engine type.
    """
    if not _initialized:
        return JSONResponse(
            status_code=503,
            content={"error": "NVML not initialized", "sidecar_version": SIDECAR_VERSION},
        )

    hostname = socket.gethostname()
    pid_to_gpus = _get_pid_to_gpus()
    pid_to_ports = _get_listening_ports()

    # Find PIDs that are both using GPU and listening on a port
    gpu_pids = set(pid_to_gpus.keys())
    listening_pids = set(pid_to_ports.keys())

    # Also check child processes: a GPU process might be a child of the listener
    # Build parent→child mapping
    pid_to_children: Dict[int, List[int]] = {}
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as f:
                    stat = f.read().split()
                    ppid = int(stat[3])
                    pid_to_children.setdefault(ppid, []).append(int(entry))
            except (OSError, IndexError, ValueError):
                continue
    except OSError:
        pass

    # For each listening PID, collect GPU indices from itself and its descendants
    def _collect_gpu_indices(pid: int, visited: set) -> List[int]:
        if pid in visited:
            return []
        visited.add(pid)
        indices = list(pid_to_gpus.get(pid, []))
        for child in pid_to_children.get(pid, []):
            indices.extend(_collect_gpu_indices(child, visited))
        return indices

    # Build candidate endpoints: listening PIDs that have GPU processes
    candidates: Dict[int, Dict[str, Any]] = {}  # port -> info

    for pid in listening_pids:
        gpu_indices = _collect_gpu_indices(pid, set())
        if not gpu_indices:
            continue
        for port in pid_to_ports[pid]:
            if port in candidates:
                continue
            candidates[port] = {
                "port": port,
                "pid": pid,
                "gpu_indices": sorted(set(gpu_indices)),
            }

    # Also check: GPU PIDs whose parent is a listener
    for gpu_pid in gpu_pids:
        if gpu_pid in listening_pids:
            continue
        # Check if parent is a listener
        try:
            with open(f"/proc/{gpu_pid}/stat") as f:
                stat = f.read().split()
                ppid = int(stat[3])
        except (OSError, IndexError, ValueError):
            continue
        if ppid in listening_pids:
            for port in pid_to_ports[ppid]:
                if port not in candidates:
                    gpu_indices = _collect_gpu_indices(ppid, set())
                    candidates[port] = {
                        "port": port,
                        "pid": ppid,
                        "gpu_indices": sorted(set(gpu_indices)),
                    }

    # Fallback: scan /proc for processes whose command line matches known
    # inference engines (ollama, vllm).  This catches idle servers that have
    # no model loaded in VRAM and therefore no NVML GPU process.
    _ENGINE_PATTERNS = ["ollama", "vllm"]
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().decode("utf-8", errors="replace").replace("\x00", " ").lower()
            except (OSError, PermissionError):
                continue
            if not any(pat in cmdline for pat in _ENGINE_PATTERNS):
                continue
            # This PID looks like an inference engine — check if it listens on any port
            if pid not in pid_to_ports:
                continue
            for port in pid_to_ports[pid]:
                if port in candidates:
                    continue  # already found via GPU mapping
                candidates[port] = {
                    "port": port,
                    "pid": pid,
                    "gpu_indices": [],  # unknown — not detected via NVML
                }
    except OSError:
        pass

    # Probe each candidate endpoint
    endpoints = []
    for port, info in sorted(candidates.items()):
        probe_result = await _probe_endpoint(port)
        if probe_result:
            endpoints.append({
                "port": info["port"],
                "pid": info["pid"],
                "engine": probe_result["engine"],
                "gpu_indices": info["gpu_indices"],
                "models": probe_result["models"],
                "url": f"http://{hostname}:{info['port']}",
            })

    return {
        "endpoints": endpoints,
        "sidecar_version": SIDECAR_VERSION,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GPU_AGENT_PORT", "8007"))
    host = os.environ.get("GPU_AGENT_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
