# Stale Server Recovery

## Problem

When a llama.cpp server running on a llmmllab-runner is evicted (e.g., due to OOM, runner restart, or slot recycling), the `ServerHandle` held by the API service becomes stale. Subsequent requests to that server return **HTTP 404 "Server X not found"**, causing user-facing failures.

## Solution

The stale server recovery mechanism detects these 404 errors, releases the stale handle, refreshes the model map, and retries the workflow with a fresh server.

### How It Works

1. **Detection** — Two detection points exist:

   - **`agents/base.py`**: When the agent node calls the llama.cpp server and receives a 404 containing "server" and "not found", it extracts the server ID and raises `StaleServerError`.
   - **`RunnerClient._is_stale_server_error()`**: A static helper that checks if an `httpx.Response` is a 404 with "server not found" in the body. Used for proactive validation.

2. **Recovery** — `CompletionService._build_and_run()` catches `StaleServerError`:

   ```
   ┌─────────────────────────────────────────────────────┐
   │  CompletionService._build_and_run()                 │
   │                                                     │
   │  1. Build workflow (acquires server handle)         │
   │  2. Run workflow                                    │
   │  3. If StaleServerError:                            │
   │     a. Release stale server handle                  │
   │     b. Refresh model map (clear stale endpoints)    │
   │     c. Retry from step 1 (with fresh handle)        │
   │  4. If retries exhausted → propagate error          │
   └─────────────────────────────────────────────────────┘
   ```

3. **Configuration** — Controlled by `STALE_SERVER_RETRIES` (env: `STALE_SERVER_RETRIES`, default: `1`):

   - `0` = no retry, error propagates immediately
   - `1` = one retry (default — handles the common case of a single eviction)
   - `N` = up to N retries

### Key Files

| File | Role |
|------|------|
| `graph/errors.py` | `StaleServerError` exception class |
| `services/completion_service.py` | Retry logic in `_build_and_run()` |
| `services/runner_client.py` | `_is_stale_server_error()`, `validate_server_handle()` |
| `agents/base.py` | Detection and re-raise as `StaleServerError` |
| `config.py` | `STALE_SERVER_RETRIES` configuration |

### Testing

- `test/unit/test_stale_server_recovery.py` — Tests the retry flow in `CompletionService._build_and_run()`
- `test/unit/test_runner_client.py` — Tests `_is_stale_server_error()` and `validate_server_handle()`
- `test/unit/test_agent_stale_detection.py` — Tests the detection pattern in `agents/base.py`

Run: `uv run pytest test/unit/`
