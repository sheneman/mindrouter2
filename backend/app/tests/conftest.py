############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# conftest.py: Pytest configuration and shared test fixtures
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Pytest configuration and shared fixtures for MindRouter2 tests."""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()

    # Database settings
    settings.database_url = "sqlite+aiosqlite:///:memory:"

    # Scheduler settings
    settings.scheduler_fairness_window = 300
    settings.scheduler_deprioritize_threshold = 0.5
    settings.scheduler_score_model_loaded = 100
    settings.scheduler_score_low_utilization = 50
    settings.scheduler_score_short_queue = 30
    settings.scheduler_score_high_throughput = 20

    # Role weights
    settings.get_scheduler_weight = MagicMock(
        side_effect=lambda role: {
            "student": 1,
            "staff": 2,
            "faculty": 3,
            "admin": 10,
        }.get(role, 1)
    )

    # Quota defaults
    settings.default_quota_student = 100000
    settings.default_quota_staff = 500000
    settings.default_quota_faculty = 1000000
    settings.default_quota_admin = 10000000

    settings.default_rpm_student = 20
    settings.default_rpm_staff = 30
    settings.default_rpm_faculty = 60
    settings.default_rpm_admin = 120

    return settings


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "What's the weather like?"},
    ]


@pytest.fixture
def sample_openai_request():
    """Sample OpenAI-format request for testing."""
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False,
    }


@pytest.fixture
def sample_ollama_request():
    """Sample Ollama-format request for testing."""
    return {
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "Hello!"},
        ],
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_predict": 100,
        },
    }


@pytest.fixture
def sample_multimodal_request():
    """Sample multimodal request with image for testing."""
    return {
        "model": "gpt-4-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                ],
            }
        ],
    }


@pytest.fixture
def sample_json_schema():
    """Sample JSON schema for structured output testing."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "email"],
    }


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    from backend.app.db.models import Backend, BackendEngine, BackendStatus

    backend = MagicMock(spec=Backend)
    backend.id = 1
    backend.name = "test-backend"
    backend.url = "http://localhost:11434"
    backend.engine = BackendEngine.OLLAMA
    backend.status = BackendStatus.HEALTHY
    backend.supports_vision = True
    backend.supports_embeddings = True
    backend.supports_structured_output = True
    backend.current_concurrent = 0
    backend.max_concurrent = 10
    backend.gpu_memory_gb = 24.0
    backend.gpu_type = "NVIDIA RTX 4090"
    backend.throughput_score = 1.0
    backend.priority = 0
    backend.consecutive_failures = 0

    return backend


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    from backend.app.db.models import Model, Modality

    model = MagicMock(spec=Model)
    model.id = 1
    model.backend_id = 1
    model.name = "llama3.2"
    model.modality = Modality.CHAT
    model.context_length = 4096
    model.supports_vision = True
    model.supports_structured_output = True
    model.is_loaded = True
    model.vram_required_gb = 8.0

    return model


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    from backend.app.db.models import User, UserRole

    user = MagicMock(spec=User)
    user.id = 1
    user.uuid = "test-user-uuid"
    user.username = "testuser"
    user.email = "test@example.com"
    user.role = UserRole.FACULTY
    user.is_active = True
    user.created_at = datetime.now(timezone.utc)

    return user


@pytest.fixture
def mock_api_key():
    """Create a mock API key for testing."""
    from backend.app.db.models import ApiKey, ApiKeyStatus

    api_key = MagicMock(spec=ApiKey)
    api_key.id = 1
    api_key.user_id = 1
    api_key.key_prefix = "mr_test"
    api_key.name = "Test Key"
    api_key.status = ApiKeyStatus.ACTIVE
    api_key.expires_at = None
    api_key.usage_count = 0

    return api_key


@pytest.fixture
def mock_quota():
    """Create a mock quota for testing."""
    from backend.app.db.models import Quota

    quota = MagicMock(spec=Quota)
    quota.id = 1
    quota.user_id = 1
    quota.token_budget = 1000000
    quota.tokens_used = 0
    quota.budget_period_start = datetime.now(timezone.utc)
    quota.budget_period_days = 30
    quota.rpm_limit = 60
    quota.max_concurrent = 5
    quota.weight_override = None

    return quota
