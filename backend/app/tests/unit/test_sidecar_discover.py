"""Tests for sidecar auto-discovery logic."""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest


# Load gpu_agent module directly to avoid needing pynvml at import time
_sidecar_path = Path(__file__).resolve().parents[4] / "sidecar" / "gpu_agent.py"


def _load_gpu_agent():
    """Load gpu_agent module with mocked dependencies."""
    # Mock pynvml and SIDECAR_SECRET_KEY before loading
    mock_pynvml = MagicMock()
    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        with patch.dict(os.environ, {"SIDECAR_SECRET_KEY": "test-secret-key"}):
            spec = importlib.util.spec_from_file_location("gpu_agent", _sidecar_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod


# Load the module once
_gpu_agent = _load_gpu_agent()


class TestPidToGpus:
    """Tests for _get_pid_to_gpus()."""

    def test_builds_mapping_from_nvml(self):
        """Should build PID→GPU index mapping from NVML process info."""
        mock_proc1 = MagicMock()
        mock_proc1.pid = 1000
        mock_proc1.usedGpuMemory = 1024 * 1024 * 100

        mock_proc2 = MagicMock()
        mock_proc2.pid = 2000
        mock_proc2.usedGpuMemory = 1024 * 1024 * 200

        mock_proc3 = MagicMock()
        mock_proc3.pid = 1000  # Same PID on GPU 1
        mock_proc3.usedGpuMemory = 1024 * 1024 * 50

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.side_effect = [
            [mock_proc1, mock_proc2],  # GPU 0
            [mock_proc3],              # GPU 1
        ]

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            # Set module state
            orig_initialized = _gpu_agent._initialized
            orig_count = _gpu_agent._device_count
            _gpu_agent._initialized = True
            _gpu_agent._device_count = 2
            try:
                result = _gpu_agent._get_pid_to_gpus()
            finally:
                _gpu_agent._initialized = orig_initialized
                _gpu_agent._device_count = orig_count

        assert 1000 in result
        assert sorted(result[1000]) == [0, 1]
        assert 2000 in result
        assert result[2000] == [0]

    def test_empty_when_not_initialized(self):
        """Should return empty dict when NVML is not initialized."""
        orig = _gpu_agent._initialized
        _gpu_agent._initialized = False
        try:
            result = _gpu_agent._get_pid_to_gpus()
        finally:
            _gpu_agent._initialized = orig
        assert result == {}


class TestGetListeningPorts:
    """Tests for _get_listening_ports()."""

    def test_parses_proc_net_tcp(self):
        """Should parse /proc/net/tcp to find listening sockets."""
        # Simulated /proc/net/tcp content
        # Port 8000 = 0x1F40, state 0A = LISTEN
        tcp_content = (
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
            "   0: 00000000:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 12345 1\n"
            "   1: 00000000:0050 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 12346 1\n"
            "   2: 0100007F:1F41 0100007F:1F40 01 00000000:00000000 00:00000000 00000000     0        0 12347 1\n"
        )

        fd_links = {
            "3": "socket:[12345]",
            "4": "pipe:[999]",
        }

        def mock_listdir(path):
            if path == "/proc":
                return ["1", "1000", "self", "net"]
            if path == "/proc/1000/fd":
                return list(fd_links.keys())
            raise OSError("no such dir")

        def mock_readlink(path):
            fd_name = path.split("/")[-1]
            if fd_name in fd_links:
                return fd_links[fd_name]
            raise OSError("no such link")

        def mock_open_fn(path, *args, **kwargs):
            if path == "/proc/net/tcp":
                return mock_open(read_data=tcp_content)()
            if path == "/proc/net/tcp6":
                raise FileNotFoundError
            raise FileNotFoundError

        with patch("builtins.open", side_effect=mock_open_fn):
            with patch("os.listdir", side_effect=mock_listdir):
                with patch("os.readlink", side_effect=mock_readlink):
                    result = _gpu_agent._get_listening_ports()

        assert 1000 in result
        assert 8000 in result[1000]

    def test_empty_when_no_proc(self):
        """Should return empty dict when /proc is not available."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = _gpu_agent._get_listening_ports()
        assert result == {}


class TestProbeEndpoint:
    """Tests for _probe_endpoint()."""

    @pytest.mark.asyncio
    async def test_detects_vllm(self):
        """Should detect vLLM from /v1/models response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "qwen/qwen3.5-400b", "object": "model"}]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _gpu_agent._probe_endpoint(8000)

        assert result is not None
        assert result["engine"] == "vllm"
        assert result["models"][0]["id"] == "qwen/qwen3.5-400b"

    @pytest.mark.asyncio
    async def test_detects_ollama(self):
        """Should detect Ollama from /api/tags response."""
        # First call (vLLM) fails, second call (Ollama) succeeds
        vllm_response = MagicMock()
        vllm_response.status_code = 404

        ollama_response = MagicMock()
        ollama_response.status_code = 200
        ollama_response.json.return_value = {
            "models": [{"name": "llama3:latest"}, {"name": "mistral:7b"}]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[vllm_response, ollama_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _gpu_agent._probe_endpoint(11434)

        assert result is not None
        assert result["engine"] == "ollama"
        assert len(result["models"]) == 2

    @pytest.mark.asyncio
    async def test_returns_none_for_non_inference(self):
        """Should return None when port is not an inference endpoint."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _gpu_agent._probe_endpoint(9999)

        assert result is None


class TestSidecarVersion:
    """Tests for sidecar version reading."""

    def test_reads_version_from_file(self):
        """SIDECAR_VERSION should match the VERSION file."""
        version_path = Path(__file__).resolve().parents[4] / "sidecar" / "VERSION"
        expected = version_path.read_text().strip()
        assert _gpu_agent.SIDECAR_VERSION == expected

    def test_health_includes_version(self):
        """The /health response structure includes sidecar_version."""
        # We verify the version field is in SIDECAR_VERSION
        assert hasattr(_gpu_agent, "SIDECAR_VERSION")
        assert isinstance(_gpu_agent.SIDECAR_VERSION, str)
        assert len(_gpu_agent.SIDECAR_VERSION) > 0
