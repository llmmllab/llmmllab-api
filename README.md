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

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_USER` | `postgres` | PostgreSQL username |
| `DB_PASSWORD` | *(empty)* | PostgreSQL password |
| `DB_NAME` | `llmmllab` | Database name |
| `DB_SSLMODE` | `disable` | SSL mode (`disable`, `require`, etc.) |
| `DB_CONNECTION_STRING` | *(auto-built from above)* | Full connection string; overrides individual DB vars if set |
| `DB_MAINTENANCE_INTERVAL_HOURS` | `24` | Hours between automated DB maintenance runs (VACUUM ANALYZE) |
| `DB_MAINTENANCE_INITIAL_DELAY_SECONDS` | `300` | Seconds to wait before the first maintenance run |
| `DB_REINDEX_ON_MAINTENANCE` | `false` | Whether to run `REINDEX` during maintenance (`true`/`false`) |

### Redis (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_ENABLED` | `true` | Enable Redis caching (`true`/`false`) |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_PASSWORD` | *(empty)* | Redis password |
| `REDIS_CONVERSATION_TTL` | `360` | Conversation cache TTL (seconds) |
| `REDIS_MESSAGE_TTL` | `180` | Message cache TTL (seconds) |
| `REDIS_SUMMARY_TTL` | `720` | Summary cache TTL (seconds) |
| `REDIS_POOL_SIZE` | `10` | Connection pool size |
| `REDIS_MIN_IDLE_CONNECTIONS` | `2` | Minimum idle connections in pool |
| `REDIS_CONNECT_TIMEOUT` | `5` | Connection timeout (seconds) |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_ISSUER` | `https://auth.longstorymedia.com` | JWT issuer URL |
| `AUTH_AUDIENCE` | `lsm-client` | Expected JWT audience claim |
| `AUTH_JWKS_URI` | `https://auth.longstorymedia.com/keys` | JWKS endpoint for key discovery |
| `AUTH_CLIENT_ID` | `lsm-client` | OAuth client ID |
| `AUTH_CLIENT_SECRET` | *(empty)* | OAuth client secret |
| `TEST_USER_ID` | *(empty)* | If set, seeds a local dev user + API key on startup (saved to `.env.local`) |

### Runner / Inference

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNNER_ENDPOINTS` | `http://localhost:9000` | Comma-separated runner service URLs |
| `RUNNER_RETRIES` | `2` | Number of retries for runner requests |
| `RUNNER_RETRY_BACKOFF_BASE` | `1` | Base seconds for exponential backoff between retries |
| `RUNNER_HEALTH_TIMEOUT_SEC` | `5.0` | Timeout for runner health checks (seconds) |
| `RUNNER_FAST_TIMEOUT_SEC` | `10.0` | Timeout for fast runner requests (status, release, etc.) |
| `RUNNER_ACQUIRE_TIMEOUT_SEC` | `150.0` | Timeout for server acquisition (seconds) |
| `RUNNER_MAX_ACQUIRE_FAILURES` | `3` | Failures before marking a runner unhealthy (circuit breaker) |
| `RUNNER_UNHEALTHY_WINDOW_SEC` | `60.0` | Seconds a runner stays unhealthy after tripping circuit breaker |
| `RUNNER_ACQUIRE_RETRIES` | `2` | Per-endpoint retries during server acquisition |
| `MODEL_CACHE_REFRESH_SEC` | `60` | Seconds between model list cache refreshes |
| `STALE_SERVER_RETRIES` | `1` | Retries on stale server handle (set `0` to disable) |

### Priority Queue

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIORITY_QUEUE_ENABLED` | `true` | Enable request priority queuing (`true`/`false`) |
| `PRIORITY_QUEUE_MAX_SIZE` | `100` | Maximum queued requests |
| `PRIORITY_QUEUE_TIMEOUT_SEC` | `300` | Max time a request waits in queue (seconds) |
| `PRIORITY_QUEUE_AGE_THRESHOLD_SEC` | `60` | Seconds before a queued request is considered "aging" |
| `PRIORITY_QUEUE_MAX_WAIT_MIN_SEC` | `1` | Minimum wait time before aging bump (seconds) |
| `PRIORITY_QUEUE_MAX_WAIT_MAX_SEC` | `3600` | Maximum wait time before aging bump (seconds) |

### Chat / LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_OPENAI_MAX_RETRIES` | `2` | Max retries for OpenAI-compatible chat completions |
| `ENABLE_TOOL_CONTINUATION` | `true` | Allow tool-call continuation in agent loops (`true`/`false`) |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key (for external model calls) |
| `ANTHROPIC_API_KEY` | *(empty)* | Anthropic API key |
| `HF_TOKEN` | *(empty)* | HuggingFace token for model downloads |
| `SEARX_HOST` | *(empty)* | SearXNG instance URL for web search tool |

### Images

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_DIR` | `/root/images` | Directory for generated images |
| `IMAGE_RETENTION_HOURS` | `24` | Hours to retain generated images before cleanup |
| `CONFIG_DIR` | `/app/config` | Directory for runtime config files |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace cache directory |

### General

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `WARNING` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `console` | Log format (`console` for human-readable, `json` for structured) |
| `FORCE_COLOR` | `0` | Force colored output even without TTY (`1` to enable) |
| `TEMPO_ENDPOINT` | `http://tempo.llmmllab.svc.cluster.local:4317` | Jaeger/Tempo OTLP endpoint for distributed tracing |
| `CUDA_VISIBLE_DEVICES` | *(unset)* | GPU device IDs visible to the process |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | CUDA device ordering |
| `PYTHONMALLOC` | `malloc` | Python memory allocator |
| `MALLOC_ARENA_MAX` | `2` | glibc malloc arena limit (reduces memory fragmentation) |
| `GGML_LOG_LEVEL` | `2` | GGML (llama.cpp backend) log verbosity |

## Project Structure

- `app.py` — FastAPI entry point
- `routers/` — API routes (openai/, anthropic/, common/)
- `middleware/` — Auth, DB init, message validation
- `services/` — Business logic (completion, token, tool)
- `runner/` — Model execution pipelines
- `composer_init.py` — Workflow orchestration API
- `agents/` — Agent implementations
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

## Priority Queue Rules

- items must be sorted in order of priority
- items with the SAME priority are sorted by active session (tracking session-id) such that an session which has been active longer takes priority over a new session with the same model
- items in queue will be elevated in priority if they remain in queue for long enough
- items may be dequeued if the model has a server which has ANY idle slots. for example if there are 4 total slots available for a server (denoted by the `parallel` flag), and three are processing, but one is idle, an item for that server may be dequeued so long as it meets the rest of this criteria
- SYSTEM or SCHEDULED items may only take up all but 1 available slots. i.e. - if there are 4 slots, 3 processing and 1 idle, a SYSTEM or SCHEDULED item must remain in queue until there are at least 2 idle slots on that server
- if resources allow, an additional server may be spun up for SAME model if there is a backlog of requests for that model in the queue, but only if there are no idle servers for that model. i.e. - if there are 4 slots available across 2 servers for a model, and all 4 are processing, and there are 2 items in the queue for that model, an additional server may be spun up to accommodate the backlog, but if there is even 1 idle slot across those servers, no additional server may be spun up until that slot is filled