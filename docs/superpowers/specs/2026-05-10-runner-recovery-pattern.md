# Runner Recovery Pattern

**Added:** 2026-05-09 (PR #49)
**Related Issues:** #44

## Problem

When a llama.cpp runner restarts mid-request (OOM, crash, deployment), the API
holds stale `ServerHandle` objects pointing to dead servers. Subsequent requests
fail with connection errors instead of gracefully routing to healthy runners.

## Solution

Three layers of recovery:

### 1. Immediate Model Map Invalidation

When a runner fails its health check, `_invalidate_model_map_for_endpoint()`
immediately removes it from the cached model-to-runner map. This prevents
`acquire_server()` from routing new requests to the dead runner, without
waiting for the next scheduled refresh (default 60 s).

### 2. Connection-Error Retry with Server Handle Refresh

`CompletionService._build_and_run_with_retry()` wraps the primary workflow
execution. On connection-level errors (`ConnectError`, `RemoteProtocolError`,
`APIConnectionError`), it:

1. Logs the error with attempt count
2. Sleeps with linear backoff (`RUNNER_RETRY_BACKOFF_BASE * attempt`)
3. Calls `runner_client.refresh_model_map()` to discover healthy runners
4. Retries with a fresh server handle

This cycle repeats up to `RUNNER_RETRIES` times (default: 2).

### 3. Graceful Shutdown Handle Cleanup

`RunnerClient.aclose()` now tracks all active server handles in
`_active_handles`. On shutdown, it releases each handle back to its runner
before closing the HTTP client, preventing orphaned llama.cpp servers.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNNER_RETRIES` | `2` | Max connection-error retries per request |
| `RUNNER_RETRY_BACKOFF_BASE` | `1` | Base delay (seconds) for linear backoff |

## Router-Level Error Handling

The OpenAI, Anthropic, and chat routers catch connection-level exceptions
from `CompletionService` and return HTTP 503 with a retry-friendly message:

```
Runner service is temporarily unavailable. Please retry.
```

This prevents stack traces from leaking to clients and enables proper
retry behavior for API consumers.

## Startup Readiness Check

`app.py` now waits up to 120 seconds for at least one runner to report
models before accepting requests. This prevents early 503s during
deployment rollouts.

## Testing

- `test/unit/test_runner_client_recovery.py` — handle validation, model map
  invalidation, active handle tracking, graceful shutdown
- `test/unit/test_completion_service_retry.py` — retry logic, backoff,
  error classification, retry exhaustion
