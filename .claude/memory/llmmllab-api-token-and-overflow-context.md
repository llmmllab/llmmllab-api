---
name: llmmllab-api-token-and-overflow-context
description: Runner reports bogus original_ctx=2048 for Qwen3.6-27B; use runtime num_ctx for the real window
metadata:
  type: reference
---

In llmmllab-api, the runner reports `model.details.original_ctx` as a garbage **2048** for Qwen3.6-27B (a bare default, not the GGUF's real context). The real runtime window is `model.parameters.num_ctx` (200000; the user's 220000 runner bump never deployed). Any overflow/window logic must use `num_ctx`, NOT `original_ctx`.

Concrete bite (fixed 2026-06-10): `is_context_overflow` compared `prompt_tokens` against `original_ctx`=2048, so it flagged *every* conversation over 2048 tokens as overflow and skipped the empty-response rescue-retry → live Claude Code sessions dead-ended mid-conversation at ~170k tokens. It was masked while token counts were under-reported (~0 → `total < 100k threshold` short-circuited to "no overflow"); fixing the token undercount (capturing streaming `usage_metadata` in `graph/executor.py`, not the empty `response_metadata.token_usage`) UNMASKED it. Lesson: a correctness fix can surface a latent bug that a wrong value was hiding — check downstream consumers.

The model emitting empty turns at ~170k is now non-fatal (auto-rescued by retry) but is a real model-behavior signal at high context. Verify kubectl-via-`ssh lsm@lsnode-0.local`; see [[openclaw-deploy-topology]], [[deploy-cuts-active-sessions]], [[runner-deploy-guard-and-qwen-degeneration]].
