---
name: api-wrong-model-fallback
description: llmmllab-api silently serves a DIFFERENT model when the requested one is mid-cold-load
metadata:
  type: project
---

When the requested model has no warm llama-server (e.g. right after a runner pod
roll, during the ~minutes-long cold load), llmmllab-api's `resolve_default_model`
**falls back to the user's `default_model`** and serves the request on a
*different model entirely*. Observed 2026-06-10: a `Qwen3_6_27B` session was
silently answered by `Gemma4_12B` on the small runner (api logged
`model: Qwen3_6_27B` but the RunnerRequestFingerprint went to the small runner's
Gemma server `8743ccb1fd7e`; responses were generic/empty because Gemma had none
of the conversation context). A runner restart cleared it.

**Why:** rolling the main runner leaves it with no server loaded (servers spawn
on-demand). The first `Qwen3_6_27B` request during that window can't acquire the
27B, so the api falls back to the default model rather than cold-starting/waiting
for the requested one.

**How to apply:** fix later — the api should **cold-start / wait for the requested
model** (or hard-error) instead of silently serving a different model. A request
for model X must never be answered by model Y. Also: batch runner config changes
into ONE deploy and coordinate timing — each push rolls the main runner, cuts the
active session, and opens this fallback window. Related: [[deploy-cuts-active-sessions]].
