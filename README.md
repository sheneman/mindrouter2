# MindRouter2

A production-ready **LLM inference load balancer** that fronts a heterogeneous backend cluster of **Ollama** and **vLLM** inference nodes, providing a unified OpenAI-compatible API surface with native Ollama compatibility.

## Features

- **Unified API Gateway**: OpenAI-compatible `/v1/*` endpoints + Ollama `/api/*` passthrough
- **API Dialect Translation**: Automatic translation between Ollama and vLLM formats
- **Fair-Share Scheduling**: Weighted Deficit Round Robin with burst credits
- **Multi-Modal Support**: Text, embeddings, and vision-language models
- **Structured Outputs**: JSON schema validation across all backends
- **Quota Management**: Per-user token budgets with role-based weights
- **Real-Time Telemetry**: GPU/memory/utilization monitoring per backend
- **Full Audit Logging**: All prompts, responses, and artifacts stored for review
- **Dual Dashboards**: Public status + authenticated user/admin interfaces

## Quickstart

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Running with Docker Compose

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (database passwords, secret key, etc.)
nano .env

# Start all services
docker compose up --build

# In another terminal, run migrations
docker compose exec app alembic upgrade head

# Seed development data
docker compose exec app python scripts/seed_dev_data.py
```

### API Access

The gateway runs on `http://localhost:8000` by default.

#### OpenAI-Compatible Endpoints

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "Hello world"
  }'

# List models
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-api-key"
```

#### Ollama-Compatible Endpoints

```bash
# Chat via Ollama API
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Generate via Ollama API
curl -X POST http://localhost:8000/api/generate \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "prompt": "Why is the sky blue?"
  }'
```

### Dashboards

- **Public Dashboard**: `http://localhost:8000/` - Cluster status, request API key
- **User Dashboard**: `http://localhost:8000/dashboard` - Usage, keys, quota requests
- **Admin Dashboard**: `http://localhost:8000/admin` - Full system control

### Default Development Credentials

After running the seed script:

| User | Password | Role |
|------|----------|------|
| admin | admin123 | admin |
| faculty1 | faculty123 | faculty |
| staff1 | staff123 | staff |
| student1 | student123 | student |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MindRouter2 Gateway                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ OpenAI   │  │ Ollama   │  │ Admin    │  │ Dashboard        │ │
│  │ /v1/*    │  │ /api/*   │  │ API      │  │ (Bootstrap)      │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│       │             │             │                  │           │
│       └─────────────┴─────────────┴──────────────────┘           │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │                    Translation Layer                       │  │
│  │  OpenAI ←→ Canonical ←→ Ollama/vLLM                       │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │              Fair-Share Scheduler (WDRR)                   │  │
│  │  • Per-user queues with deficit counters                  │  │
│  │  • Role-based weights (faculty > staff > student)         │  │
│  │  • Burst credits for idle cluster utilization             │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │                  Backend Registry                          │  │
│  │  • Capability polling (models, GPU, memory)               │  │
│  │  • Health monitoring                                       │  │
│  │  • Model residency tracking                                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   ┌─────────┐           ┌─────────┐           ┌─────────┐
   │ Ollama  │           │  vLLM   │           │ Ollama  │
   │ Node 1  │           │ Node 1  │           │ Node 2  │
   └─────────┘           └─────────┘           └─────────┘
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | MariaDB connection string | Required |
| `SECRET_KEY` | Session/JWT signing key | Required |
| `REDIS_URL` | Redis for rate limiting (optional) | None |
| `ARTIFACT_STORAGE_PATH` | Path for uploaded files | `/data/artifacts` |
| `DEFAULT_TOKEN_BUDGET` | Monthly token allowance | 100000 |
| `SCHEDULER_FAIRNESS_WINDOW` | Rolling window for usage tracking | 300 (5 min) |

### Backend Registration

Register backends via admin API or dashboard:

```bash
# Register an Ollama backend
curl -X POST http://localhost:8000/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ollama-gpu-1",
    "url": "http://ollama-host:11434",
    "engine": "ollama",
    "max_concurrent": 4,
    "gpu_memory_gb": 24
  }'

# Register a vLLM backend
curl -X POST http://localhost:8000/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-a100-1",
    "url": "http://vllm-host:8000",
    "engine": "vllm",
    "max_concurrent": 16,
    "gpu_memory_gb": 80
  }'
```

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Start MariaDB (via docker)
docker compose up -d mariadb redis

# Run migrations
alembic upgrade head

# Seed data
python scripts/seed_dev_data.py

# Start development server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Unit tests
pytest backend/app/tests/unit -v

# Integration tests (requires docker)
pytest backend/app/tests/integration -v

# End-to-end tests
pytest backend/app/tests/e2e -v

# All tests with coverage
pytest --cov=backend/app --cov-report=html
```

### Makefile Commands

```bash
make dev          # Start development server
make test         # Run all tests
make test-unit    # Run unit tests only
make lint         # Run linters
make format       # Format code
make migrate      # Run database migrations
make seed         # Seed development data
make docker-up    # Start docker compose stack
make docker-down  # Stop docker compose stack
```

## API Documentation

### Endpoints

#### Inference Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat |
| POST | `/v1/completions` | OpenAI-compatible completion |
| POST | `/v1/embeddings` | OpenAI-compatible embeddings |
| GET | `/v1/models` | List available models |
| POST | `/api/chat` | Ollama-compatible chat |
| POST | `/api/generate` | Ollama-compatible generate |
| GET | `/api/tags` | Ollama-compatible model list |

#### Health & Metrics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Liveness probe |
| GET | `/readyz` | Readiness probe |
| GET | `/metrics` | Prometheus metrics |

#### Admin Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/admin/backends/register` | Register new backend |
| POST | `/admin/backends/{id}/disable` | Disable backend |
| POST | `/admin/backends/{id}/enable` | Enable backend |
| POST | `/admin/backends/{id}/refresh` | Refresh capabilities |
| GET | `/admin/queue` | View scheduler queue |
| GET | `/admin/audit/search` | Search audit logs |

## Scheduler Algorithm

MindRouter2 implements **Weighted Deficit Round Robin (WDRR)** for fair resource allocation:

1. **Share Weights**: faculty=3, staff=2, student=1, admin=10
2. **Deficit Counters**: Track service debt per user
3. **Burst Credits**: Allow full cluster use when idle
4. **Backend Scoring**: Model residency, GPU utilization, queue depth

See [docs/scheduler.md](docs/scheduler.md) for detailed algorithm specification.

## Security

- API keys are stored hashed (Argon2)
- Role-based access control (RBAC)
- Rate limiting per key (RPM) and per user (concurrency)
- All admin actions are audited
- Request/response content logged for compliance review

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

Please follow conventional commit messages.
