# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastAPI inference service with OpenAI- and Anthropic-compatible endpoints. Backed by a runner service that hosts `llama.cpp` (text), `stable-diffusion.cpp` (image generation, e.g. Qwen-Image-2512 GGUF), and an in-process TRELLIS pipeline (image-to-3D). LangGraph agent orchestration for chat. Python 3.12+, managed via `uv`.

The Ollama-compatible router was removed; only OpenAI and Anthropic wire protocols are exposed.

### Image / 3D endpoints

| Endpoint | Backend | Notes |
|----------|---------|-------|
| `POST /v1/images/generations` | runner sd-server (stable-diffusion.cpp) | OpenAI-compatible `CreateImageRequest`. Returns `b64_json`. |
| `POST /v1/images/edits` | runner sd-server (img2img, e.g. Qwen-Image-Edit-2511) | Custom JSON body (`prompt`, `image` base64, `denoising_strength`). Returns `b64_json`. |
| `POST /v1/images/3d` | runner pipeline `img23d` (TRELLIS) | Returns `mesh_path` (`.glb`) and/or `gaussian_path` (`.ply`) plus `mesh_url`/`gaussian_url` for download. Long-running. |
| `GET  /v1/images/3d/{filename}` | runner pipeline `img23d/files/{filename}` | Streams `.glb` / `.ply` / `.png` back through the api so clients don't need pod access. |

`services/image_service.py` is the single bridge. `generate_image`, `edit_image` acquire a runner server (text/image model handled by `SDCppServerManager`) and hit `/sdapi/v1/txt2img` or `/sdapi/v1/img2img`. `generate_3d` hits `/v1/pipelines/img23d/run` (no server acquisition, the pipeline is in-process on the runner). `stream_3d_artifact` proxies `GET /v1/pipelines/img23d/files/{filename}` for downloads, with a path-traversal-safe regex on filename. Tests live in `test/unit/test_image_service.py` and `test/unit/test_images_router.py`. CLI test scripts under `scripts/` (see `scripts/README.md`).

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
2. **Routers** (`routers/`) handle provider-compatible endpoints. OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) both delegate to the same service layer.
3. **CompletionService** (`services/completion_service.py`) is the shared execution layer. It owns workflow building and orchestrates retry policies, continuation logic, and tool-call filtering by composing the helper modules listed under "CompletionService modules" below. Routers only format the wire protocol (SSE chunks).
4. **Composer** (`composer_init.py`) is the public API boundary for LangGraph workflow orchestration. It manages a module-level `ComposerService` (no manager class), composes workflows, creates initial state, and executes with streaming.
5. **LangGraph Workflows** (`graph/workflows/`) — two types: `IDE` (full agent with tools) and `Dialog` (simpler conversation). The factory (`graph/workflows/factory.py`) selects based on `WorkFlowType` enum. Both builders inherit from `GraphBuilder` (`graph/workflows/base.py`), which hosts shared model resolution + `runner_client.acquire_server` + `ChatAgent` construction.
6. **Agents** (`agents/base.py`) wrap LangChain `create_agent()` with node metadata injection, logging, and error handling. They bridge between LangGraph nodes and the pipeline layer.
7. **Pipelines** (`runner/pipelines/`) execute inference: `llamacpp/chat.py` (text), `llamacpp/embed.py` (embeddings), `txt2img/flux.py` (image gen), `imgtxt2txt/qwen3_vl.py` (multimodal). The factory (`runner/pipeline_factory.py`) routes by `ModelProvider` and `ModelTask`. Local providers get cached instances; remote providers (OpenAI, Anthropic) are transient.

### Key Entry Points

| Component | File |
|-----------|------|
| FastAPI app | `app.py` |
| OpenAI chat | `routers/openai/chat.py` |
| Anthropic messages | `routers/anthropic/messages.py` |
| Shared completion logic | `services/completion_service.py` |
| Composer API | `composer_init.py` |
| Runner client | `services/runner_client.py` |
| Pipeline factory | `runner/pipeline_factory.py` |
| Pipeline cache | `runner/pipeline_cache.py` |
| Workflow factory | `graph/workflows/factory.py` |
| Workflow builder base | `graph/workflows/base.py` |
| Base agent | `agents/base.py` |
| DB storage singleton | `db/__init__.py` |

### CompletionService modules

`services/completion_service.py` was split into focused helpers. The public surface — `CompletionService.stream_completion`, `run_completion`, `build_workflow`, `cancel_session` — is unchanged; all remain `@staticmethod`. Routers and tests required no changes.

| Module | Responsibility |
|--------|----------------|
| `services/completion_service.py` | Orchestrator (~720 lines). Owns `_build_and_run`, which holds the stale-server retry to preserve test patches. |
| `services/completion_state.py` | `CompletionResult`, `StreamAccumulator` dataclasses. Re-exported from `services/__init__.py` and `services.completion_service` for backward compat. |
| `services/prompt_templates.py` | `CONTINUATION_PROMPT`, `EMPTY_RESPONSE_NUDGE`, `TRUNCATION_CONTINUATION_PROMPT`, plus length thresholds. |
| `services/truncation.py` | Pure functions `is_truncated`, `is_context_overflow`. |
| `services/response_handlers.py` | `extract_text`, `build_followup_messages`, `filter_tool_calls` (drops server-side tool calls), `update_stream_delta`, `update_stream_final`. |
| `services/session_tracking.py` | In-flight session task registry: `register_session_task`, `unregister_session_task`, `cancel_session`. |
| `services/retry_policies.py` | `stream_with_connection_retry` (connection-error retry with exponential backoff). Stale-server retry stays in `_build_and_run`. |
| `services/continuation_logic.py` | `maybe_continue_on_truncation`, `maybe_continue_on_missing_tool_call`, `maybe_retry_on_empty`, `stream_secondary_pass`, `collect_response`. Hosts the `revalidate_runner_handles()` trigger fired when an LLM returns an empty stream. |

### Key Patterns

**Pipeline System**: `runner/pipeline_factory.py` creates pipelines by model task (text, image, embeddings, multimodal). `runner/pipeline_cache.py` manages instances with memory-based eviction. Local providers (llama.cpp, Stable Diffusion) are cached; remote API providers are created per-call.

**Provider Compatibility**: OpenAI (`routers/openai/`) and Anthropic (`routers/anthropic/`) endpoints share the same `CompletionService` → `Composer` → `Pipeline` execution chain.

**LangGraph Workflows**: `graph/workflows/ide/builder.py` and `graph/workflows/dialog/builder.py` build StateGraphs with nodes for agent execution, tool calling, memory search/store, and web search. Both subclass `GraphBuilder` (`graph/workflows/base.py`), which centralises model resolution, server acquisition, and agent construction. `graph/executor.py` streams workflow events and converts 404-shaped errors into `StaleServerError` (see "Runner restart recovery" below).

**Context Overflow Guard**: `agents/base.py::_ensure_context_fits()` estimates token usage and trims old messages when the conversation exceeds the model's context window (with a configurable safety margin). If trimming can't help (e.g., runner auto-reduced `n_ctx` due to memory pressure), a `ContextOverflowError` is raised and converted to a user-friendly message. The `num_ctx` is passed to the runner on `acquire_server()` so the runner can refuse to start undersized servers.

**Tool System**: `tools/registry.py` manages tool discovery. `tools/static/` contains built-in tools (web search, web reader, memory retrieval, todo). `graph/nodes/server_tools.py` handles server-side tool execution within the workflow.

**Multi-Tier Storage**: `db/__init__.py` exposes a `storage` singleton with 15+ storage components (conversation, message, memory, search, summary, thought, tool_call, document, todo, image, model, api_key, checkpoint, user_config, cache). Schema managed by Alembic, runs automatically on startup. Raw SQL stored in `db/sql/`.

**Multi-Tier Caching**: User config flows memory → Redis → PostgreSQL via `db/multi_tier_cache.py`. **`REDIS_ENABLED` defaults to `true`**; the parser accepts `{"1","true","yes","on"}` (case-insensitive). Set it to `false`/`0`/`no`/`off` to opt out and run with memory + DB only. Redis reads fall through to the DB on `ConnectionError` (do not raise). Covered by `test/unit/test_multi_tier_cache.py` (Redis-enabled and Redis-disabled branches).

**Auth**: JWT-based via JWKS URI. API key auth also supported (`routers/api_key.py`). Public paths: `/health`, `/docs`, `/redoc`, `/openapi.json`, `/static/images/view/`.

### Runner restart recovery

When the runner process restarts, in-flight `server_id` handles in `services/runner_client.py` become invalid. The api detects and recovers from this transparently:

1. **`startup_epoch`** — the runner emits a monotonic `startup_epoch` value on `GET /v1/status`. `RunnerClient` tracks `_runner_epochs: Dict[endpoint, int]`. On every `acquire_server` call, and opportunistically when `proxy_request` sees a 503, it probes `/v1/status`. If the epoch changed, it purges all `_active_handles` for that endpoint and invalidates the model map.
2. **404 → `StaleServerError`** — a 404 from `/v1/server/<id>/...` for a known `server_id` is converted to `StaleServerError` in `services/runner_client.py`. `graph/executor.py` likewise converts 404-shaped workflow errors (`openai.NotFoundError`, `"Not Found"` substring) into `StaleServerError` after calling `runner_client.revalidate_runner_handles()`.
3. **Empty SSE stream** — chat completions stream directly without going through `proxy_request`, so 404 → `StaleServerError` cannot fire there. `services/continuation_logic.py::maybe_retry_on_empty` therefore calls `runner_client.revalidate_runner_handles()` when the LLM yields an empty stream. The epoch poll surfaces a dead runner even when no HTTP error did.
4. **Workflow-cache invalidation** — the cached `CompiledStateGraph` for a `(user_id, model_name)` key has its `ChatOpenAI(base_url=...)` baked with the now-dead handle. When stale-server retry fires (in `services/completion_service.py::_build_and_run`, which also exposes the retry to test patches), it calls `composer_init.invalidate_workflow(user_id, model_name)` to drop the cached workflow, refreshes the model map, and recursively retries with a fresh handle. Backed by `STALE_SERVER_RETRIES` (default 1; set 0 to disable).
5. **Lease discipline** — `acquire_server` / `release_server` follow a strict try/finally lease pattern. Handles are released on the success path too — previously they only released on error. `_in_flight_tasks` registration guards against `current_task()` returning `None`.
6. **Routers no longer pollute history** — `routers/openai/chat.py` and `routers/anthropic/messages.py` previously injected a `[Model returned empty response after all retries…]` diagnostic into the assistant stream when retries failed. Clients echoed it back as user-visible assistant content on subsequent turns, eventually exhausting output token budgets. Empty-after-all-retries now closes the stream cleanly with zero content and a normal `stop_reason`.

Invariant: when `StaleServerError` propagates out of `_build_and_run`, the cached workflow for that `(user_id, model_name)` **must** have been invalidated before the retry executes. If you add a new path that surfaces 404 to the service layer, route it through `StaleServerError` and let the existing retry handle invalidation.

### Composer public API

`composer_init.py` exposes:

- `compose_workflow(user_id, model_name, ...) -> CompiledStateGraph`
- `invalidate_workflow(user_id, model_name) -> bool` — drop one cached workflow (used by stale-handle retry)
- `clear_workflow_cache() -> None`
- `create_initial_state(...)`
- `execute_workflow(...)`
- `shutdown_composer()`
- `get_composer_service(builder)` — accessor for the module-level singleton

There is **no** `ComposerServiceManager` class. The previous singleton wrapper was inlined into the module (`_composer_service: Optional[ComposerService]`).

### Configuration

All config is environment-based (`config.py`). Key variables:

- `DB_CONNECTION_STRING` — PostgreSQL (TimescaleDB) connection
- `REDIS_ENABLED` (default `true`), `REDIS_HOST`, `REDIS_PORT` — Redis cache
- `AUTH_ISSUER`, `AUTH_AUDIENCE`, `AUTH_JWKS_URI` — JWT auth
- `TEST_USER_ID` — seeds a local dev user + API key on startup (saved to .env.local)
- `HF_TOKEN` — HuggingFace model downloads
- Cache eviction is controlled by the runner via `CACHE_TIMEOUT_MIN` and `EVICTION_TIMEOUT_MIN` env vars
- `ENABLE_TOOL_CONTINUATION` — force tool call if model describes but doesn't invoke (default true)
- `CONTEXT_USAGE_SAFETY_MARGIN` — fraction of num_ctx reserved for conversation input (default 0.85)
- `CONTEXT_MINIMUM_RATIO` — reject runner servers with less than this fraction of requested context (default 0.80)
- `STALE_SERVER_RETRIES` — retries on stale-handle recovery (default 1; set 0 to disable)

### Project Structure

```
app.py                  FastAPI entry point + lifespan
config.py               Environment-based configuration
composer_init.py        Composer public API (module-level singleton)
routers/                API routes
  openai/               OpenAI-compatible endpoints (chat, embeddings, audio, etc.)
  anthropic/            Anthropic-compatible endpoints (messages, completions)
  common/               Shared endpoints (models, files)
  chat.py, images.py, conversation.py, ...  Direct endpoints
middleware/             Auth, DB init, message validation
services/               Business logic
  completion_service.py     Orchestrator (~720 lines)
  completion_state.py       CompletionResult, StreamAccumulator
  prompt_templates.py       Nudge/continuation/truncation prompts
  truncation.py             is_truncated, is_context_overflow
  response_handlers.py      Text extraction, tool-call filtering, stream updates
  session_tracking.py       In-flight task registry, cancel_session
  retry_policies.py         Connection-error retry with backoff
  continuation_logic.py     Truncation/empty/missing-tool-call continuation
  runner_client.py          Runner HTTP client with epoch tracking
  priority_queue.py, ...    Queue, services, etc.
runner/                 Model execution
  pipeline_factory.py
  pipeline_cache.py
  pipelines/            llamacpp, flux, qwen3_vl
  server_manager/
  utils/
agents/                 Agent implementations (base, chat, embed)
core/                   Core composer components (service, errors)
graph/                  LangGraph workflow system
  workflows/
    base.py             Shared GraphBuilder (model resolve, acquire, agent)
    factory.py          IDE vs Dialog selector
    ide/                IDE workflow builder
    dialog/             Dialog workflow builder
  nodes/                Agent, tool, memory, web search nodes
  state.py              Workflow state definition
  executor.py           Stream-enabled workflow execution; 404→StaleServerError
tools/                  Tool registry and static tools
db/                     Database layer
  __init__.py           Storage singleton (15+ components)
  engine.py             SQLAlchemy async engine
  models.py             ORM models
  sql/                  Raw SQL scripts
  multi_tier_cache.py   Memory → Redis → PostgreSQL cache
  maintenance.py        Scheduled DB maintenance
models/                 Pydantic data models
utils/                  Shared helpers
  message_conversion.py     Consolidated provider/internal message mapping
  logging.py
  token_estimation.py
  ...
k8s/                    Kubernetes deployment manifests
test/                   Tests (unit + integration with testcontainers)
alembic/                Database migrations
```

Notes on removed modules:

- `routers/ollama.py` — deleted. The Ollama-compatible router was unused (contained a literal `"urmomu"` typo proving it was never exercised) and is no longer registered in `app.py`.
- `models/learned_limits.py`, `models/technical_domain.py` — deleted along with their re-exports in `models/__init__.py`.
- `utils/message_transformation.py` — merged into `utils/message_conversion.py`. Import from `utils.message_conversion`.

### Data Models

Pydantic models in `models/` define the domain: `Message`, `ChatResponse`, `ToolCall`, `MemoryFragment`, `Conversation`, `UserConfig`, `Model`, etc. The `models/anthropic/` and `models/openai/` subdirectories contain provider-specific request/response types. Edit directly — there is no separate schema generation step.

### Database

TimescaleDB (PostgreSQL + timeseries). Alembic migrations run automatically on app startup. The `db/__init__.py` `storage` singleton initializes all storage components from a shared async session factory. Integration tests use `testcontainers` with TimescaleDB PG16 image.

### Tests

`test/unit/` has 314 passing tests. Recent additions and updates:

- `test/unit/test_multi_tier_cache.py` — 10 new tests covering Redis-enabled and Redis-disabled branches; verifies the `ConnectionError` fallthrough.
- `test/unit/test_runner_client.py`, `test/unit/test_proxy_request.py` — updated to model the new `startup_epoch` poll and the active-handle gate.
