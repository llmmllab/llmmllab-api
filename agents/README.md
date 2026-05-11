# Agents

Agent implementations that wrap LangChain `create_agent()` with node metadata
injection, logging, error handling, and retry logic.

## Base Agent (`base.py`)

`BaseAgent` is the foundation for all agent types (chat, embeddings, etc.).

### Key Features

- **Node Metadata Injection** — Every agent carries `NodeMetadata` for workflow tracking.
- **Structured Logging** — Uses `llmmllogger` with component binding for structured log output.
- **Tool & Middleware Deduplication** — Combines instance-level and call-level tools/middleware, deduplicating by name/class.
- **Transient Error Retry** — Automatically retries on `openai.APIConnectionError` with exponential backoff.

### Transient Error Retry

When `BaseAgent.run()` encounters a transient connection error
(`openai.APIConnectionError`), it retries up to **10 times** (11 total attempts)
with exponential backoff:

| Attempt | Backoff |
|---------|---------|
| 1 → 2   | 2 s     |
| 2 → 3   | 4 s     |
| 3 → 4   | 8 s     |
| 4 → 5   | 16 s    |
| 5 → 6   | 32 s    |
| 6 → 7   | 60 s    |
| 7 → 8   | 60 s    |
| 8 → 9   | 60 s    |
| 9 → 10  | 60 s    |
| 10 → 11 | 60 s    |

The backoff formula is `min(2^(attempt+1), 60)`, capped at 60 seconds.

**Only** `APIConnectionError` triggers retries. Other exceptions (e.g.,
`ValueError`, `APITimeoutError`) propagate immediately. After exhausting all
retries, the last `APIConnectionError` is re-raised and caught by the outer
error handler, which returns an error `ChatResponse`.

Each retry emits a `WARNING` log entry with the attempt number, backoff
duration, and error message.

### Tests

Unit tests for the retry logic live in `test/unit/test_agent_retry.py`:

```bash
uv run pytest test/unit/test_agent_retry.py -v
```

Tests cover:
- Success on first attempt (no retry)
- Retry on transient error, success on a later attempt
- Retry warning logs
- Exhaustion of all retries (error response returned)
- Non-transient errors propagate immediately
- Backoff schedule correctness
