# MindRouter2 Architecture

## Overview

MindRouter2 is a production-ready LLM inference load balancer designed to front heterogeneous backend clusters of Ollama and vLLM inference nodes. It provides a unified OpenAI-compatible API surface while implementing fair-share scheduling, quota management, and comprehensive audit logging.

## System Components

### 1. API Gateway

The API Gateway is the entry point for all client requests.

**Key Responsibilities:**
- Request authentication via API keys
- Request validation and normalization
- Protocol translation (OpenAI ↔ Ollama ↔ vLLM)
- Response streaming with low latency
- Error handling and retry logic

**Endpoints:**
- `/v1/chat/completions` - OpenAI-compatible chat
- `/v1/embeddings` - OpenAI-compatible embeddings
- `/v1/models` - List available models
- `/api/chat` - Ollama-compatible chat
- `/api/generate` - Ollama-compatible generation
- `/api/tags` - Ollama-compatible model list

### 2. Translation Layer

The translation layer converts between different API formats.

```
┌─────────────────┐
│ Client Request  │
└────────┬────────┘
         │
    ┌────▼────┐
    │OpenAI In│  or  │Ollama In│
    └────┬────┘      └────┬────┘
         │                │
         └───────┬────────┘
                 │
         ┌───────▼───────┐
         │   Canonical   │
         │    Schema     │
         └───────┬───────┘
                 │
         ┌───────┴────────┐
         │                │
    ┌────▼────┐      ┌────▼────┐
    │vLLM Out │      │Ollama Out│
    └────┬────┘      └────┬────┘
         │                │
         └───────┬────────┘
                 │
    ┌────────────▼────────────┐
    │     Backend Request     │
    └─────────────────────────┘
```

**Canonical Schemas:**
- `CanonicalChatRequest` - Unified chat format
- `CanonicalEmbeddingRequest` - Unified embedding format
- `CanonicalStreamChunk` - Unified streaming format

### 3. Fair-Share Scheduler

Implements Weighted Deficit Round Robin (WDRR) for fair resource allocation.

**Key Concepts:**
- **Share Weights**: faculty=3, staff=2, student=1, admin=10
- **Deficit Counters**: Track service debt per user
- **Burst Credits**: Allow full cluster use when idle
- **Deprioritization**: Reduce priority for heavy users

**Scheduling Flow:**
1. Job arrives, compute initial priority
2. Add to per-user queue
3. When backend available:
   - Select user with highest (deficit + burst_credits) / weight
   - Score eligible backends
   - Route to best backend
4. On completion, update deficit counter

### 4. Backend Registry

Manages backend discovery, health, and telemetry.

**Capabilities:**
- Automatic model discovery
- Health check polling
- GPU utilization tracking
- Model residency tracking

**Backend Adapters:**
- `OllamaAdapter` - Polls `/api/tags`, `/api/ps`, `/api/version`
- `VLLMAdapter` - Polls `/v1/models`, `/health`, `/metrics`

### 5. Backend Scorer

Multi-factor scoring for backend selection.

**Hard Constraints (must pass):**
- Model availability
- Modality support (vision, embeddings)
- Memory fit
- Capacity available

**Soft Scores (higher = better):**
- Model already loaded: +100
- Low GPU utilization: +50
- Short queue: +30
- High throughput GPU: +20

### 6. Quota Management

Token-based quota system with role-based defaults.

**Features:**
- Monthly token budgets
- Requests per minute (RPM) limits
- Max concurrent requests
- Automatic quota reset

### 7. Audit Logging

Complete request/response logging for compliance.

**Logged Data:**
- Full request content (prompts, messages)
- Full response content
- Token usage (actual or estimated)
- Timing metrics (queue delay, processing time)
- Scheduling decisions

## Data Model

```
┌─────────────┐     ┌─────────────┐
│    User     │────<│   ApiKey    │
└──────┬──────┘     └─────────────┘
       │
       │     ┌─────────────┐
       ├────<│    Quota    │
       │     └─────────────┘
       │
       │     ┌─────────────┐     ┌─────────────┐
       └────<│   Request   │────<│  Response   │
             └──────┬──────┘     └─────────────┘
                    │
                    │     ┌─────────────┐
                    ├────<│  Artifact   │
                    │     └─────────────┘
                    │
                    │     ┌─────────────┐
                    └────<│ Scheduler   │
                          │  Decision   │
                          └─────────────┘

┌─────────────┐     ┌─────────────┐
│   Backend   │────<│    Model    │
└──────┬──────┘     └─────────────┘
       │
       │     ┌─────────────┐
       └────<│  Telemetry  │
             └─────────────┘
```

## Request Flow

```
Client Request
       │
       ▼
┌─────────────────┐
│  Authentication │
│   (API Key)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quota Check     │
│ Rate Limit      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Translation    │
│  (→ Canonical)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Create Job     │
│  Compute Priority│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Route to       │
│  Backend        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Proxy Request  │
│  (Stream if     │
│   applicable)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Record Audit   │
│  Update Usage   │
└────────┬────────┘
         │
         ▼
    Response
```

## Deployment

### Development
```bash
docker compose up --build
```

### Production Considerations
- Use external MariaDB cluster
- Configure Redis for distributed rate limiting
- Mount persistent volume for artifacts
- Set up monitoring (Prometheus/Grafana)
- Configure TLS termination
- Set secure SECRET_KEY
