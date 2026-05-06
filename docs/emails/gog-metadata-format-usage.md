# Using `gog gmail get --format=metadata` in Cron Jobs

## Problem

When cron jobs (like the daily email cleanup) use `gog gmail get <messageId> -j` without `--format=metadata`, the command returns the full email body (~150KB of HTML). This exceeds the OpenClaw trajectory field size limit of 32,768 characters (`TRAJECTORY_RUNTIME_DATA_STRING_MAX_CHARS` in the runtime). The tool result gets truncated to a marker object:

```json
{
  "truncated": true,
  "reason": "trajectory-field-size-limit",
  "originalChars": 148894,
  "limitChars": 32768
}
```

The model then receives this truncated marker instead of the actual email content, and typically responds with `stopReason=stop` and zero tool calls — effectively giving up.

## Solution

Always use `--format=metadata` when fetching email data in cron jobs:

```bash
gog gmail get <messageId> --format=metadata -j
```

This returns only ~2.5KB of structured data including:
- `headers` — standard email headers (From, To, Subject, Date)
- `message.payload.headers` — raw headers including `List-Unsubscribe`
- `message.labelIds` — Gmail labels
- `message.snippet` — preview text
- `unsubscribe` — **top-level field with the direct unsubscribe URL** (extracted from List-Unsubscribe header)

## Example

```bash
# BAD — returns ~150KB, causes truncation
gog gmail get 19dfd5b1e9308901 -j

# GOOD — returns ~2.5KB, includes unsubscribe link
gog gmail get 19dfd5b1e9308901 --format=metadata -j
```

## Affected Cron Jobs

- `email-cleanup-daily` — fixed in `~/.openclaw/cron.json`

## References

- GitHub Issue: https://github.com/llmmllab/llmmllab-api/issues/18
- Runtime constant: `TRAJECTORY_RUNTIME_DATA_STRING_MAX_CHARS = 32768` (in OpenClaw gateway runtime)
