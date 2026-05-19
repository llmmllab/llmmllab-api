# llmmllab-api

Python FastAPI inference service with OpenAI- and Anthropic-compatible endpoints, backed by `llama.cpp` (via a separate runner service) and LangGraph agent orchestration.

The Ollama-compatible router was removed; only the OpenAI (`/v1/chat/completions`, `/v1/embeddings`, ...) and Anthropic (`/v1/messages`) wire protocols are exposed.

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
| `REDIS_ENABLED` | Enable the multi-tier (memory + Redis + DB) user-config cache (default: `true`; set to `false`/`0`/`no`/`off` to opt out and run with memory + DB only). |
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
| `STALE_SERVER_RETRIES` | Retries on stale server handle (default: 1, set 0 to disable) |

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
- `routers/` — API routes (`openai/`, `anthropic/`, `common/`)
- `middleware/` — Auth, DB init, message validation
- `services/` — Business logic. The completion path is split across `completion_service.py` (orchestrator), `completion_state.py`, `prompt_templates.py`, `truncation.py`, `response_handlers.py`, `session_tracking.py`, `retry_policies.py`, and `continuation_logic.py`. The runner client (`runner_client.py`) tracks runner restart epochs.
- `runner/` — Model execution pipelines
- `composer_init.py` — Workflow orchestration API (module-level singleton; exposes `compose_workflow`, `invalidate_workflow`, `clear_workflow_cache`, `execute_workflow`)
- `agents/` — Agent implementations
- `core/` — Core composer components
- `graph/` — LangGraph workflow system. Both IDE and Dialog builders subclass the shared `GraphBuilder` in `graph/workflows/base.py`. `graph/executor.py` converts runner-restart 404s into `StaleServerError`.
- `tools/` — Tool registry and static tools
- `db/` — Multi-tier storage (PostgreSQL + optional Redis)
- `models/` — Pydantic data models
- `utils/` — Shared helpers (message conversion, logging, token estimation, ...)
- `k8s/` — Kubernetes deployment manifests
- `test/` — Tests

## Runner Restart Recovery

The api recovers transparently from runner process restarts:

- `RunnerClient` tracks a per-endpoint `startup_epoch` returned by `GET /v1/status`. On `acquire_server` and opportunistically on 503 responses, it re-probes status. An epoch bump purges all active handles for that endpoint and invalidates the model map.
- 404 responses against `/v1/server/<id>/...` for a known handle, and 404-shaped errors raised during LangGraph workflow execution, are converted into `StaleServerError`.
- Empty SSE streams from chat completions trigger `revalidate_runner_handles()` (chat streams bypass the proxy that would otherwise see the 404 directly).
- The completion service catches `StaleServerError`, invalidates the cached workflow for `(user_id, model_name)` via `composer_init.invalidate_workflow`, refreshes the model map, and retries with a fresh handle. Retries are capped by `STALE_SERVER_RETRIES` (default 1, set 0 to disable).
- Empty-after-all-retries now closes the stream cleanly with no content. The api no longer injects a `[Model returned empty response...]` diagnostic into the assistant response — that text was being echoed back into history by clients and eventually exhausted output token budgets.

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

## Priority Queue Rules

- items must be sorted in order of priority
- items with the SAME priority are sorted by active session (tracking session-id) such that an session which has been active longer takes priority over a new session with the same model
- items in queue will be elevated in priority if they remain in queue for long enough
- items may be dequeued if the model has a server which has ANY idle slots. for example if there are 4 total slots available for a server (denoted by the `parallel` flag), and three are processing, but one is idle, an item for that server may be dequeued so long as it meets the rest of this criteria
- SYSTEM or SCHEDULED items may only take up all but 1 available slots. i.e. - if there are 4 slots, 3 processing and 1 idle, a SYSTEM or SCHEDULED item must remain in queue until there are at least 2 idle slots on that server
- if resources allow, an additional server may be spun up for SAME model if there is a backlog of requests for that model in the queue, but only if there are no idle servers for that model. i.e. - if there are 4 slots available across 2 servers for a model, and all 4 are processing, and there are 2 items in the queue for that model, an additional server may be spun up to accommodate the backlog, but if there is even 1 idle slot across those servers, no additional server may be spun up until that slot is filled