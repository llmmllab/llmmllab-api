# llmmllab-api

Python FastAPI inference service with OpenAI- and Anthropic-compatible endpoints. Backed by a separate runner service that hosts:

- **`llama.cpp`** — text completion + embeddings
- **`stable-diffusion.cpp`** — text-to-image (`POST /v1/images/generations`) and image-to-image (`POST /v1/images/edits`, Qwen-Image-Edit-2511)
- **Hunyuan3D-2.1** — image-to-3D (`POST /v1/images/3d`, download via `GET /v1/images/3d/{filename}`)

Plus LangGraph agent orchestration. The Ollama-compatible router was removed; only the OpenAI (`/v1/chat/completions`, `/v1/embeddings`, `/v1/images/generations`, …) and Anthropic (`/v1/messages`) wire protocols are exposed.

### Image generation

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a teacup with steam","model":"qwen-image-2512","size":"1024x1024"}'
# -> {"created": ..., "data": [{"b64_json": "iVBORw0K..."}], "output_format": "png"}
```

The `model` field is forwarded to the runner; any model registered as `provider: stable_diffusion_cpp` is eligible. Default sampling parameters (40 steps, cfg 2.5, sampler `euler`) are tuned for Qwen-Image-2512 Q4_K_M; override in the runner's `.models.yaml` to target SDXL/SD3.

### Image-to-3D

```bash
curl http://localhost:8000/v1/images/3d \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"<base64 PNG>","formats":["mesh","gaussian"]}'
# -> {"id":"abc123","elapsed_sec":48.2,"mesh_path":"/data/sd-out/3d/abc123.glb", ...}
```

Backed by Tencent Hunyuan3D-2.1 (shape-only path, ~6 GB VRAM) running in-process on the runner. Returns a `.glb` mesh; gaussian-splat output is not supported by this backbone (the response's `gaussian_url` will be `null`). The response includes `mesh_url` pointing at `GET /v1/images/3d/{filename}`, which streams the binary back through the api without requiring pod access.

### Mesh-to-parts decomposition

```bash
curl http://localhost:8000/v1/images/3d/parts \
  -H "Content-Type: application/json" \
  -d '{"mesh_b64":"<base64 .glb>","octree_resolution":256,"split":true}'
# -> {"id":"abc123","elapsed_sec":92.4,"mesh_url":"...decomposed.glb",
#     "part_urls":["...part_00.glb","...part_01.glb",...]}
```

Backed by Tencent Hunyuan3D-Part (P3-SAM + XPart) running in-process
on the runner. Decomposes a whole mesh (typically the output of
`/v1/images/3d`) into semantically meaningful parts. Optional `aabb`
field (`[K, 2, 3]` bounding boxes) bypasses P3-SAM's auto-segmentation
when you already know the part layout.

### Image-pipeline tuning

Every image endpoint exposes per-request sampling knobs via the
request body. Unset fields fall through to per-model defaults from
the runner's `.models.yaml`. The most commonly used:

- `negative_prompt` — strongly recommended for object-specific gens
- `cfg_scale` — prompt-faithfulness (default 4.0; bump to 5-7 for stubborn-geometry mechanical objects)
- `steps` — diffusion sampling steps
- `sampler_name` — `dpm++_2m` (default), `euler`, `dpm++_sde`, ...
- `num_inference_steps`, `guidance_scale`, `octree_resolution`, `mc_level`, `box_v`, `num_chunks` — img23d-side knobs
- `max_parts`, `aabb` — mesh2parts-side knobs

See [`scripts/README.md`](scripts/README.md) for the env-var
equivalents on the shell scripts, and
[`.claude/skills/generate-3d/generate_3d_models.md`](.claude/skills/generate-3d/generate_3d_models.md)
for full parameter reference + tuning advice per use case
(industrial vs organic, low-fidelity iteration vs final, etc.).

### Test scripts

See [`scripts/README.md`](scripts/README.md) for ready-made curl + jq harnesses (`txt2img.sh`, `img2img.sh`, `img2-3d.sh`, `mesh2parts.sh`) that exercise each endpoint and decode the responses.

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

All variables below are read through `config.py` unless noted otherwise.

### Database (including maintenance)

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_USER` | `postgres` | PostgreSQL username |
| `DB_PASSWORD` | *(empty)* | PostgreSQL password |
| `DB_NAME` | `llmmllab` | Database name |
| `DB_SSLMODE` | `disable` | SSL mode (`disable`, `require`, etc.) |
| `DB_CONNECTION_STRING` | *(auto-built from above)* | Full connection string; overrides individual DB vars if set |
| `DB_MAINTENANCE_INTERVAL_HOURS` | `24` | Hours between automated DB maintenance runs (VACUUM ANALYZE + sequence align) |
| `DB_MAINTENANCE_INITIAL_DELAY_SECONDS` | `300` | Seconds to wait before the first maintenance run after pod startup |
| `DB_REINDEX_ON_MAINTENANCE` | `false` | Whether to run `REINDEX CONCURRENTLY` during maintenance — off by default as it can trip stale-OID plan errors under live TimescaleDB traffic |

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
| `STALE_SERVER_RETRIES` | `2` | Retries on stale server handle (set `0` to disable) |
| `CACHE_TIMEOUT_MIN` | `10` | Minutes before an idle loaded server is considered "abandoned" (safe to commandeer). Should match the runner's value. |

### Cold-Start Retry

When a fresh model server cold-starts, the runner returns HTTP 503 while loading (~45–90 s). These knobs control how long the api waits and retries:

| Variable | Default | Description |
|----------|---------|-------------|
| `COLD_START_RETRIES` | `4` | Extra retry attempts for cold-start 503s (set `0` to disable) |
| `COLD_START_BACKOFF_SEC` | `20.0` | Fixed wait (seconds) between cold-start retries; 4 × 20 ≈ 80 s covers a typical load |

### Server-Side Tools

Controls execution of tools like `web_search` / `web_fetch` on the server side:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_SIDE_TOOLS_ENABLED` | `true` | Master switch for server-side tool execution (`true`/`false`). Per-request override via `X-Server-Side-Tools: false` header (Anthropic). |
| `MCP_WEB_TOOLS_URL` | `http://mcp-server-web.llmmllab.svc.cluster.local:8000` | MCP server URL for `web_search` / `web_fetch`. Set to empty to fall back to inline SearxNG + Playwright. |
| `SERVER_TOOL_MAX_ITERATIONS` | `4` | Hard cap on Agent ↔ ServerToolNode loops per completion; hitting the cap routes the graph to END. |

### Priority Queue

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIORITY_QUEUE_ENABLED` | `true` | Enable request priority queuing (`true`/`false`) |
| `PRIORITY_QUEUE_MAX_SIZE` | `100` | Maximum queued requests |
| `PRIORITY_QUEUE_TIMEOUT_SEC` | `300` | Max time a request waits in queue (seconds) |
| `PRIORITY_QUEUE_AGE_THRESHOLD_SEC` | `60` | Seconds before a queued request is considered "aging" |
| `PRIORITY_QUEUE_RECHECK_SEC` | `2.0` | Seconds between safety-net poll checks for resource-aware scheduling |
| `PRIORITY_QUEUE_MAX_WAIT_MIN_SEC` | `1` | Minimum wait time before aging bump (seconds) |
| `PRIORITY_QUEUE_MAX_WAIT_MAX_SEC` | `3600` | Maximum wait time before aging bump (seconds) |

### Chat / LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_OPENAI_MAX_RETRIES` | `2` | Max retries for OpenAI-compatible chat completions |
| `AGENT_MAX_RETRY_ATTEMPTS` | `11` | Max transient-error retries per agent turn (connection errors, 5xx). Set lower to fail-fast when clients disconnect. |
| `ENABLE_TOOL_CONTINUATION` | `true` | Allow tool-call continuation in agent loops (`true`/`false`) |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key (for external model calls) |
| `ANTHROPIC_API_KEY` | *(empty)* | Anthropic API key |
| `HF_TOKEN` | *(empty)* | HuggingFace token for model downloads |
| `SEARX_HOST` | *(empty)* | SearXNG instance URL for web search tool |

### Storage / Images

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_DIR` | `/root/images` | Directory for generated images |
| `IMAGE_RETENTION_HOURS` | `24` | Hours to retain generated images before cleanup |
| `CONFIG_DIR` | `/app/config` | Directory for runtime config files |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace cache directory |
| `IMG_SERVER_AUTO_SHUTDOWN` | `true` | Tear down sd-server / qwen-image servers after each request rather than holding them warm. Image servers are 4–12 GB resident; set `false` for benchmarking or batched generation. |

### Vision Token Accounting

llama.cpp's text `/tokenize` endpoint can't see image blocks in multimodal
messages, but the model side (via `clip_model_path` / mmproj) produces
real vision tokens per image. Without accounting for those, the api's
pre-trim guard (`agents/base.py::_ensure_context_fits`) under-counts and
the runner refuses requests that have grown beyond `n_ctx`. The
`services/token_counter.py` helpers estimate image tokens with the
Qwen2/3-VL formula `⌈W / patch⌉ × ⌈H / patch⌉` after resizing the long
edge down to a cap.

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_TOKENS_DEFAULT` | `1500` | Per-image fallback when dimensions can't be decoded (HTTP URLs, missing PIL, malformed base64) |
| `VISION_PATCH_PX` | `28` | Vision-tower patch size in pixels (Qwen-VL family default) |
| `VISION_MAX_LONG_EDGE_PX` | `1280` | Max long-edge the vision tower processes before patchification |

### Raw Token Debug

When enabled, writes every raw model token (no stripping/modification) plus all user messages to `<RAW_TOKEN_DEBUG_DIR>/<session_id>.tokens`. Useful for diagnosing premature stops and context overflow patterns.

| Variable | Default | Description |
|----------|---------|-------------|
| `RAW_TOKEN_DEBUG` | `false` | Enable raw-token debug logging (`true`/`false`) |
| `RAW_TOKEN_DEBUG_DIR` | `/tmp/llmmllab_debug` | Output directory for debug token files |

### Tracing

| Variable | Default | Description |
|----------|---------|-------------|
| `TEMPO_ENDPOINT` | `http://tempo.llmmllab.svc.cluster.local:4317` | OTLP gRPC endpoint for OpenTelemetry trace export. Set to empty to disable tracing. |

### General

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `9999` | Server port (passed via Makefile to uvicorn) |
| `API_VERSION` | `v1` | API version prefix |
| `LOG_LEVEL` | `WARNING` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Read directly by `utils/logging.py` because logging is bootstrapped before `config.py` loads. |
| `LOG_FORMAT` | `console` | Log format (`console` for human-readable, `json` for structured). Same bootstrap-order constraint as `LOG_LEVEL`. |
| `FORCE_COLOR` | `0` | Force ANSI colors in console log output even without TTY (`1` to enable) |
| `TEMPO_ENDPOINT` | `http://tempo.llmmllab.svc.cluster.local:4317` | Jaeger/Tempo OTLP endpoint for distributed tracing |

### Container / Runner Environment

These variables are not consumed by the Python API directly but are set in the deployment environment (e.g. `k8s/env.yaml`, `.env.example`) for the container and llama.cpp runner process.

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_LAUNCH_BLOCKING` | `0` | Force synchronous CUDA kernel launches (`1` to enable, set by the API at startup) |
| `CUDA_VISIBLE_DEVICES` | *(unset)* | GPU device IDs visible to the process |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | CUDA device ordering |
| `PYTHONMALLOC` | `malloc` | Python memory allocator |
| `MALLOC_ARENA_MAX` | `2` | glibc malloc arena limit (reduces memory fragmentation) |
| `GGML_LOG_LEVEL` | `2` | GGML (llama.cpp backend) log verbosity |

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
- The completion service catches `StaleServerError`, invalidates the cached workflow for `(user_id, model_name)` via `composer_init.invalidate_workflow`, refreshes the model map, and retries with a fresh handle. Retries are capped by `STALE_SERVER_RETRIES` (default 2, set 0 to disable).
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
