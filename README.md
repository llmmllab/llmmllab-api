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

Copy `.env.example` to `.env` and set the required values:

| Variable | Description |
|----------|-------------|
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | PostgreSQL connection |
| `REDIS_ENABLED`, `REDIS_HOST`, `REDIS_PORT` | Redis cache (optional)
| `AUTH_ISSUER`, `AUTH_AUDIENCE`, `AUTH_JWKS_URI` | JWT auth (JWT + API key) |
| `HF_TOKEN` | HuggingFace token for model downloads |
| `PORT` | Server port (default: 8000) |
| `LOG_LEVEL` | Logging verbosity (debug, info, warning, error) |
| `CUDA_VISIBLE_DEVICES` | GPU devices for inference |
| `STALE_SERVER_RETRIES` | Retries on stale server handle (default: 1, set 0 to disable) |

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

