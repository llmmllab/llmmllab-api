# llmmllab-api

Python FastAPI inference service with OpenAI- and Anthropic-compatible endpoints, backed by `llama.cpp` and LangGraph agent orchestration.

## Quick Start

```bash
# 1. Clone and set up
git clone <repo> && cd llmmllab-api
cp .env.example .env
# Edit .env with your database, auth, and model configuration

# 2. Install dependencies (requires uv)
uv sync

# 3. Run
make start
```

The API server will start on `http://localhost:8000`. Docs at `/docs`.

## Commands

```bash
make start          # Start API server (uvicorn + reload)
make test           # Run all tests
make validate       # Python syntax check
make clean          # Remove build artifacts

make docker-build   # Build Docker image
make docker-push    # Build and push Docker image
make deploy         # Full deploy: build, push, apply k8s manifests
make sync-watch     # Watch mode: sync code to k8s node on changes
```

## Configuration

Copy `.env.example` to `.env` and set the required values. See `config.py` for defaults.

### Database

| Variable | Description |
|----------|-------------|
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_SSLMODE` | PostgreSQL (TimescaleDB) connection |
| `DB_CONNECTION_STRING` | Full connection string (overrides individual DB vars) |

### Redis (Optional)

| Variable | Description |
|----------|-------------|
| `REDIS_ENABLED` | Enable Redis cache (default: true in k8s, false locally) |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` | Redis connection |
| `REDIS_PASSWORD` | Redis password (optional) |
| `REDIS_CONVERSATION_TTL`, `REDIS_MESSAGE_TTL`, `REDIS_SUMMARY_TTL` | TTL in minutes for cached data |
| `REDIS_POOL_SIZE`, `REDIS_MIN_IDLE_CONNECTIONS`, `REDIS_CONNECT_TIMEOUT` | Connection pool settings |

### Authentication

| Variable | Description |
|----------|-------------|
| `AUTH_ISSUER`, `AUTH_AUDIENCE`, `AUTH_JWKS_URI` | JWT auth via JWKS |
| `AUTH_CLIENT_ID`, `AUTH_CLIENT_SECRET` | OAuth client credentials |
| `TEST_USER_ID` | Seed a local dev user + API key on startup (saved to `.env.local`) |
| `INTERNAL_API_KEY` | Internal service-to-service API key |
| `INTERNAL_ALLOWED_IPS` | Comma-separated CIDR list for internal API access |

### Runner / Inference

| Variable | Description |
|----------|-------------|
| `RUNNER_ENDPOINTS` | Comma-separated runner service URLs |
| `RUNNER_RETRIES` | Runner acquisition retries (default: 2) |
| `RUNNER_RETRY_BACKOFF_BASE` | Runner retry backoff base (default: 1) |
| `RUNNER_HEALTH_TIMEOUT_SEC` | Health check timeout (default: 5.0) |
| `RUNNER_FAST_TIMEOUT_SEC` | Fast request timeout (default: 10.0) |
| `RUNNER_ACQUIRE_TIMEOUT_SEC` | Server acquisition timeout (default: 150.0) |
| `RUNNER_MAX_ACQUIRE_FAILURES` | Circuit breaker threshold (default: 3) |
| `RUNNER_UNHEALTHY_WINDOW_SEC` | Circuit breaker reset window (default: 60.0) |
| `RUNNER_ACQUIRE_RETRIES` | Per-endpoint connection retries (default: 2) |
| `MODEL_CACHE_REFRESH_SEC` | Model map refresh interval (default: 60) |
| `CONTEXT_USAGE_SAFETY_MARGIN` | Fraction of `num_ctx` reserved for conversation input (default: 0.85) |
| `CONTEXT_MINIMUM_RATIO` | Min ratio of actual to requested context before rejecting a server (default: 0.80) |

### Priority Queue

| Variable | Description |
|----------|-------------|
| `PRIORITY_QUEUE_ENABLED` | Enable request priority queue (default: true) |
| `PRIORITY_QUEUE_MAX_SIZE` | Max queued requests (default: 100) |
| `PRIORITY_QUEUE_TIMEOUT_SEC` | Queue timeout in seconds (default: 300) |
| `PRIORITY_QUEUE_AGE_THRESHOLD_SEC` | Age threshold for priority bumping (default: 60) |
| `PRIORITY_QUEUE_MAX_WAIT_MIN_SEC` | Min wait before priority bump (default: 1) |
| `PRIORITY_QUEUE_MAX_WAIT_MAX_SEC` | Max wait before priority bump (default: 3600) |

### Chat / LLM

| Variable | Description |
|----------|-------------|
| `CHAT_OPENAI_MAX_RETRIES` | Max retries for OpenAI-compatible chat completions (default: 2) |
| `ENABLE_TOOL_CONTINUATION` | Force tool call if model describes but doesn't invoke (default: true) |
| `OPENAI_API_KEY` | OpenAI API key (for remote models) |
| `ANTHROPIC_API_KEY` | Anthropic API key (for remote models) |

### Summarization

| Variable | Description |
|----------|-------------|
| `MESSAGES_BEFORE_SUMMARY` | Messages before triggering summary (default: 6) |
| `SUMMARIES_BEFORE_CONSOLIDATION` | Summaries before consolidation (default: 3) |
| `SUMMARY_MODEL` | Model used for summarization (default: qwen3:0.6b) |
| `SUMMARY_SYSTEM_PROMPT` | System prompt for summarization |
| `MAX_SUMMARY_LEVELS` | Max nested summary levels (default: 3) |
| `SUMMARY_WEIGHT_COEFFICIENT` | Weight for summary importance (default: 1) |

### Images

| Variable | Description |
|----------|-------------|
| `IMAGE_GENERATION_ENABLED` | Enable image generation (default: true) |
| `IMAGE_DIR` | Local image storage path |
| `MAX_IMAGE_SIZE` | Max image dimension (default: 2048) |
| `IMAGE_RETENTION_HOURS` | Image cleanup retention (default: 24) |

### General

| Variable | Description |
|----------|-------------|
| `PORT` | Server port (default: 8000) |
| `API_VERSION` | API version prefix (default: v1) |
| `LOG_LEVEL` | Logging verbosity (debug, info, warning, error) |
| `LOG_FORMAT` | Log format (console or json) |
| `HF_TOKEN` | HuggingFace token for model downloads |
| `SEARX_HOST` | SearXNG instance URL for web search |
| `CUDA_VISIBLE_DEVICES` | GPU devices for inference |

## Project Structure

- `app.py` — FastAPI entry point
- `routers/` — API routes (openai/, anthropic/, common/)
- `middleware/` — Auth, DB init, message validation
- `services/` — Business logic (completion, token, tool)
- `runner/` — Model execution pipelines
- `composer_init.py` — Workflow orchestration API
- `agents/` — Agent implementations (see [agents/README.md](agents/README.md))
- `core/` — Core composer components
- `graph/` — LangGraph workflow builder, nodes, state
- `tools/` — Tool registry and static tools
- `db/` — Multi-tier storage (PostgreSQL + Redis)
- `models/` — Pydantic data models
- `utils/` — Shared helpers
- `k8s/` — Kubernetes deployment manifests
- `test/` — Tests

## Docker

```bash
make docker-build DOCKER_IMAGE=llmmllab-api DOCKER_TAG=latest
```

The `Dockerfile` builds a CUDA-enabled image with `llama.cpp` compiled from source.

## Kubernetes

```bash
make deploy DOCKER_TAG=main
```

Manifests in `k8s/` include deployment, service, PVC, and secrets setup.

## CI/CD

Deployments are automated via GitHub Actions on merges to `main`. Images are tagged with the commit SHA and `latest`.


<!-- trigger deploy -->


<!-- trigger self-hosted deploy -->

