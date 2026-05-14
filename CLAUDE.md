# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastAPI inference service with OpenAI-, Anthropic-, and Ollama-compatible endpoints. Backed by `llama.cpp` (local), `llama.cpp` server management, Flux (image generation), and LangGraph agent orchestration. Python 3.12+, managed via `uv`.

## Commands

```bash
uv sync              # Install dependencies (requires uv)
make start           # Start API server (uvicorn + reload, port 8000)
make start PORT=9000 # Start on custom port
make test            # Run all tests (unit + integration)
make test-unit       # Run unit tests only
make validate        # Python syntax check
make clean           # Remove __pycache__ and .pyc files
make docker-build    # Build Docker image (CUDA 12.8, llama.cpp from source)
make deploy          # Build, push, and apply k8s manifests
make sync-watch      # Watch mode: sync code to k8s node on changes
```

### Running Individual Tests

```bash
uv run pytest test/unit/test_composer_validation.py -v
uv run pytest test/unit/test_composer_validation.py::test_specific_name -v
uv run pytest test/integration/test_api_key_storage.py -v
```

Integration tests spin up a TimescaleDB container via testcontainers, run Alembic migrations, and use transactional sessions (rollback after each test). Docker must be running.

## Architecture

### Request Flow

```
HTTP Request → Auth Middleware → Router → CompletionService → Composer → LangGraph Workflow → Agent → Pipeline → Model
```

1. **Auth** (`middleware/auth.py`) validates JWT tokens via JWKS. API key auth also supported. `TEST_USER_ID` seeds a local dev user + API key on startup.
2. **Routers** (`routers/`) handle provider-compatible endpoints. OpenAI (`/v1/chat/completions`), Anthropic (`/v1/messages`), and Ollama all delegate to the same service layer.
3. **CompletionService** (`services/completion_service.py`) is the shared execution layer. It owns workflow building, retry on empty responses, nudge prompts, tool-continuation checks, and filtering server tool calls. Routers only format the wire protocol (SSE chunks).
4. **Composer** (`composer_init.py`) is the public API boundary for LangGraph workflow orchestration. It manages the `ComposerService` singleton, composes workflows, creates initial state, and executes with streaming.
5. **LangGraph Workflows** (`graph/workflows/`) — two types: `IDE` (full agent with tools) and `Dialog` (simpler conversation). The factory (`graph/workflows/factory.py`) selects based on `WorkFlowType` enum.
6. **Agents** (`agents/base.py`) wrap LangChain `create_agent()` with node metadata injection, logging, and error handling. They bridge between LangGraph nodes and the pipeline layer.
7. **Pipelines** (`runner/pipelines/`) execute inference: `llamacpp/chat.py` (text), `llamacpp/embed.py` (embeddings), `txt2img/flux.py` (image gen), `imgtxt2txt/qwen3_vl.py` (multimodal). The factory (`runner/pipeline_factory.py`) routes by `ModelProvider` and `ModelTask`. Local providers get cached instances; remote providers (OpenAI, Anthropic) are transient.

### Key Entry Points

| Component | File |
|-----------|------|
| FastAPI app | `app.py` |
| OpenAI chat | `routers/openai/chat.py` |
| Anthropic messages | `routers/anthropic/messages.py` |
| Ollama compat | `routers/ollama.py` |
| Shared completion logic | `services/completion_service.py` |
| Composer API | `composer_init.py` |
| Pipeline factory | `runner/pipeline_factory.py` |
| Pipeline cache | `runner/pipeline_cache.py` |
| Workflow factory | `graph/workflows/factory.py` |
| Base agent | `agents/base.py` |
| DB storage singleton | `db/__init__.py` |

### Key Patterns

**Pipeline System**: `runner/pipeline_factory.py` creates pipelines by model task (text, image, embeddings, multimodal). `runner/pipeline_cache.py` manages instances with memory-based eviction. Local providers (llama.cpp, Stable Diffusion) are cached; remote API providers are created per-call.

**Provider Compatibility**: OpenAI (`routers/openai/`), Anthropic (`routers/anthropic/`), and Ollama (`routers/ollama.py`) endpoints all share the same `CompletionService` → `Composer` → `Pipeline` execution chain.

**LangGraph Workflows**: `graph/workflows/ide/builder.py` and `graph/workflows/dialog/builder.py` build StateGraphs with nodes for agent execution, tool calling, memory search/store, and web search. `graph/executor.py` streams workflow events.

**Context Overflow Guard**: `agents/base.py::_ensure_context_fits()` estimates token usage and trims old messages when the conversation exceeds the model's context window (with a configurable safety margin). If trimming can't help (e.g., runner auto-reduced `n_ctx` due to memory pressure), a `ContextOverflowError` is raised and converted to a user-friendly message. The `num_ctx` is passed to the runner on `acquire_server()` so the runner can refuse to start undersized servers.

**Tool System**: `tools/registry.py` manages tool discovery. `tools/static/` contains built-in tools (web search, web reader, memory retrieval, todo). `graph/nodes/server_tools.py` handles server-side tool execution within the workflow.

**Multi-Tier Storage**: `db/__init__.py` exposes a `storage` singleton with 15+ storage components (conversation, message, memory, search, summary, thought, tool_call, document, todo, image, model, api_key, checkpoint, user_config, cache). Schema managed by Alembic, runs automatically on startup. Raw SQL stored in `db/sql/`.

**Multi-Tier Caching**: User config flows memory → Redis → PostgreSQL via `db/multi_tier_cache.py`. Redis is optional (`REDIS_ENABLED=false`).

**Auth**: JWT-based via JWKS URI. API key auth also supported (`routers/api_key.py`). Public paths: `/health`, `/docs`, `/redoc`, `/openapi.json`, `/static/images/view/`.

### Configuration

All config is environment-based (`config.py`). Key variables:

- `DB_CONNECTION_STRING` — PostgreSQL (TimescaleDB) connection
- `REDIS_ENABLED`, `REDIS_HOST`, `REDIS_PORT` — Redis cache (optional)
- `AUTH_ISSUER`, `AUTH_AUDIENCE`, `AUTH_JWKS_URI` — JWT auth
- `TEST_USER_ID` — seeds a local dev user + API key on startup (saved to .env.local)
- `HF_TOKEN` — HuggingFace model downloads
- Cache eviction is controlled by the runner via `CACHE_TIMEOUT_MIN` and `EVICTION_TIMEOUT_MIN` env vars
- `ENABLE_TOOL_CONTINUATION` — force tool call if model describes but doesn't invoke (default true)
- `CONTEXT_USAGE_SAFETY_MARGIN` — fraction of num_ctx reserved for conversation input (default 0.85)
- `CONTEXT_MINIMUM_RATIO` — reject runner servers with less than this fraction of requested context (default 0.80)

### Project Structure

```
app.py                  FastAPI entry point + lifespan
config.py               Environment-based configuration
composer_init.py        Composer public API (workflow orchestration)
routers/                API routes
  openai/               OpenAI-compatible endpoints (chat, embeddings, audio, etc.)
  anthropic/            Anthropic-compatible endpoints (messages, completions)
  common/               Shared endpoints (models, files)
  chat.py, images.py, conversation.py, ...  Direct endpoints
middleware/             Auth, DB init, message validation
services/               Business logic (completion, token, tool)
runner/                 Model execution
  pipeline_factory.py   Routes local vs remote, delegates to cache
  pipeline_cache.py     Memory-based eviction cache
  pipelines/            Pipeline implementations (llamacpp, flux, qwen3_vl)
  server_manager/       llama.cpp server lifecycle
  utils/                Hardware manager, model loader
agents/                 Agent implementations (base, chat, embed)
core/                   Core composer components (service, errors)
graph/                  LangGraph workflow system
  workflows/            IDE and Dialog workflow builders
  nodes/                Agent, tool, memory, web search nodes
  state.py              Workflow state definition
  executor.py           Stream-enabled workflow execution
tools/                  Tool registry and static tools
db/                     Database layer
  __init__.py           Storage singleton (15+ components)
  engine.py             SQLAlchemy async engine
  models.py             ORM models
  sql/                  Raw SQL scripts
  multi_tier_cache.py   Memory → Redis → PostgreSQL cache
  maintenance.py        Scheduled DB maintenance
models/                 Pydantic data models
utils/                  Shared helpers (logging, message conversion, token estimation)
k8s/                    Kubernetes deployment manifests
test/                   Tests (unit + integration with testcontainers)
alembic/                Database migrations
```

### Data Models

Pydantic models in `models/` define the domain: `Message`, `ChatResponse`, `ToolCall`, `MemoryFragment`, `Conversation`, `UserConfig`, `Model`, etc. The `models/anthropic/` and `models/openai/` subdirectories contain provider-specific request/response types. Edit directly — there is no separate schema generation step.

### Database

TimescaleDB (PostgreSQL + timeseries). Alembic migrations run automatically on app startup. The `db/__init__.py` `storage` singleton initializes all storage components from a shared async session factory. Integration tests use `testcontainers` with TimescaleDB PG16 image.
