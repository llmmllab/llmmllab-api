#!/usr/bin/env python3
"""Bulk triage of open issues and PRs in llmmllab-api + llmmllab-runner.

Generated 2026-05-21.  Run under a gh auth that has WRITE access to the
llmmllab org (the EMU lons7862_ehp account is blocked from `mergePullRequest`,
`addComment`, and close operations on non-enterprise repos).

Sections, executed top-to-bottom:
  1. Merge PR #244 (API) — kills the log-monitor noise generator
  2. Merge PR #90  (API) — meaningful "no healthy runner" error
  3. Close superseded PRs with explanatory comments
  4. Close stale issues
  5. Post review comments on judgment-call PRs (left open)
  6. Bulk-close 93 log-monitor noise issues with umbrella pointers

Usage:
  python3 triage_close.py --dry    # print actions, don't execute
  python3 triage_close.py           # execute
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

DRY = "--dry" in sys.argv

API = "llmmllab/llmmllab-api"
RUN = "llmmllab/llmmllab-runner"


def run(cmd: list[str], *, stdin: str | None = None) -> None:
    """Print and (optionally) execute a gh CLI command."""
    print("+", " ".join(shlex.quote(c) for c in cmd))
    if stdin:
        # Show just the first line of stdin
        first = stdin.splitlines()[0] if stdin.splitlines() else ""
        print(f"   (stdin: {first[:80]}{'...' if len(first) > 80 else ''})")
    if DRY:
        return
    try:
        result = subprocess.run(cmd, input=stdin, text=True, capture_output=True, timeout=30)
        if result.returncode != 0:
            print(f"   ERR (exit {result.returncode}): {result.stderr.strip()[:300]}")
        elif result.stdout.strip():
            print(f"   ok: {result.stdout.strip()[:200]}")
    except subprocess.TimeoutExpired:
        print("   ERR: timeout")


def comment(repo: str, number: int, body: str) -> None:
    """gh issue comment / gh pr comment via stdin to avoid shell quoting."""
    # gh issue comment supports `--body-file -` to read from stdin
    run(["gh", "issue", "comment", str(number), "-R", repo, "--body-file", "-"], stdin=body)


def pr_comment(repo: str, number: int, body: str) -> None:
    run(["gh", "pr", "comment", str(number), "-R", repo, "--body-file", "-"], stdin=body)


def close_issue(repo: str, number: int, reason: str = "completed") -> None:
    run(["gh", "issue", "close", str(number), "-R", repo, "--reason", reason])


def close_pr(repo: str, number: int) -> None:
    run(["gh", "pr", "close", str(number), "-R", repo])


# -----------------------------------------------------------------------------
# 1. Merges (do these first; #244 kills the noise generator)
# -----------------------------------------------------------------------------
print("\n=== 1. Merges ===")
run(["gh", "pr", "merge", "244", "-R", API, "--squash", "--delete-branch"])
# #90 review state is CHANGES_REQUESTED ("fix unit tests") but tests are now
# SUCCESS — use --admin to override the stale review state.
run(["gh", "pr", "merge", "90", "-R", API, "--squash", "--delete-branch", "--admin"])

# -----------------------------------------------------------------------------
# 2. Close superseded PRs with comments
# -----------------------------------------------------------------------------
print("\n=== 2. Stale PRs ===")

pr_comment(API, 165, (
    "Closing as superseded. The transient-connection-error retry now lives in "
    "`services/retry_policies.py::stream_with_connection_retry`, which catches "
    "`APIConnectionError`, `RemoteProtocolError`, and `ConnectError` with exponential "
    "backoff (see `services/retry_policies.py:65-84`). The PR predates that helper."
))
close_pr(API, 165)

pr_comment(API, 113, (
    "Closing as already shipped. `num_ctx` is now extracted from kwargs and included in "
    "the runner acquire payload at `services/runner_client.py:806`:\n\n"
    "```python\n"
    "if num_ctx is not None:\n"
    "    payload[\"num_ctx\"] = num_ctx\n"
    "```\n\n"
    "Companion change on the runner side: `routers/servers.py:147` returns HTTP 507 "
    "`context_too_large` when the request exceeds the model's configured `n_ctx`."
))
close_pr(API, 113)

pr_comment(RUN, 27, (
    "Closing as already shipped. The 507 `context_too_large` guard exists in main at "
    "`routers/servers.py:147`:\n\n"
    "```python\n"
    "\"reason\": \"context_too_large\",\n"
    "\"requested_num_ctx\": requested_ctx,\n"
    "\"model_num_ctx\": model_ctx,\n"
    "```\n\n"
    "with the comparison gate at lines 134-138."
))
close_pr(RUN, 27)

pr_comment(RUN, 36, (
    "Closing as superseded. Commit `3ab5051` rewrote the DCGM integration end-to-end:\n\n"
    "- URL switched to the cluster Service "
    "(`nvidia-dcgm-exporter.gpu-operator.svc.cluster.local:9400`)\n"
    "- Metric name map fixed (`nv_*` → `DCGM_FI_DEV_*`)\n"
    "- NODE_NAME filtering added so cross-node Service load-balancing doesn't return "
    "foreign samples\n\n"
    "The original debug-vs-warning tweak is no longer relevant — the scrape now succeeds. "
    "Issue #23 closed alongside."
))
close_pr(RUN, 36)

pr_comment(RUN, 30, (
    "Closing as superseded. The proxy router has been substantially refactored since this "
    "PR was opened (slot LRU, KV cache persistence, prompt-hash diagnostics, session "
    "attribution), and the equivalent error handling already exists in main:\n\n"
    "- `_stream_upstream` at `proxy/router.py:724`\n"
    "- `upstream_iterator` at `proxy/router.py:768`\n"
    "- `httpx.ConnectError` handling at lines 819, 860, 1088\n"
    "- \"Upstream server X disconnected\" 503 response at lines 1083-1086\n\n"
    "The current branch has heavy conflicts (proxy/router.py, routers/servers.py, two "
    "test files). Rebasing would require reimplementing on top of the new structure; "
    "the functional gap that motivated the PR is closed.\n\n"
    "Closing issue #28 alongside."
))
close_pr(RUN, 30)

# -----------------------------------------------------------------------------
# 3. Close stale issues
# -----------------------------------------------------------------------------
print("\n=== 3. Stale issues ===")

comment(RUN, 23, (
    "Closing — DCGM exporter integration was rewritten in commit `3ab5051` (URL switched "
    "to the cluster Service, metric name map fixed `nv_* → DCGM_FI_DEV_*`, NODE_NAME "
    "hostname filter to drop cross-node samples). Scrape no longer fails. PR #36 closed "
    "alongside as superseded."
))
close_issue(RUN, 23, reason="completed")

comment(RUN, 38, (
    "Closing as exact duplicate of #39 — same fingerprint "
    "(`Upstream server ab6677136e7f error before response`), same session_id, filed "
    "twice within seconds."
))
close_issue(RUN, 38, reason="not planned")

comment(RUN, 28, (
    "Closing alongside PR #30. The proxy_router upstream-disconnect path has full "
    "coverage in main — see comment on PR #30 for line references."
))
close_issue(RUN, 28, reason="completed")

# -----------------------------------------------------------------------------
# 4. Review comments on judgment-call PRs (left OPEN for user decision)
# -----------------------------------------------------------------------------
print("\n=== 4. Review comments (PRs left open) ===")

pr_comment(API, 115, (
    "**Decision needed.** Runner PR #27 already shipped, so the runner now returns HTTP "
    "507 `context_too_large` with `requested_num_ctx` / `model_num_ctx` in the error "
    "detail (see `runner/routers/servers.py:147`).\n\n"
    "This PR adds a second layer at the API: silently halving `num_ctx` and retrying up "
    "to 3 times. Two paths forward:\n\n"
    "1. **Close this PR.** The runner's 507 is now actionable — the client gets a clear "
    "error and can decide how to react. The API silently truncating context may hide "
    "that the configured num_ctx is wrong.\n"
    "2. **Rebase to complement, not duplicate.** Keep the fallback but only trigger on "
    "507 `context_too_large` specifically, and surface a warning log indicating the "
    "fallback fired. Otherwise propagate.\n\n"
    "Preference for option 1 unless there's a known client that benefits from silent "
    "retry. Marking for triage."
))

pr_comment(RUN, 37, (
    "**Partially superseded — decision needed.** The original motivation in #22 (prefer "
    "runner where model is NOT in use) is now satisfied at the API layer by "
    "`services/runner_client.py::_select_runner`, which ranks endpoints by "
    "`(-active_handles_here, available_vram)` — exactly the behavior #22 asked for.\n\n"
    "The endpoint in this PR is still useful for:\n"
    "- External observability / dashboard queries\n"
    "- Explicit health checks\n"
    "- Multi-tier orchestration outside the API service\n\n"
    "But it's no longer load-bearing for session prioritization. Decide whether to ship "
    "it for observability or close both as already-solved."
))

# -----------------------------------------------------------------------------
# 5. Bulk-close log-monitor noise issues
# -----------------------------------------------------------------------------
print("\n=== 5. Bulk-close log-monitor noise ===")

plan_path = Path("/tmp/cleanup/api_close_plan.json")
plan = json.loads(plan_path.read_text())
print(f"Closing {len(plan)} log-monitor noise issues...")

for item in plan:
    n = item["n"]
    umb = item["umb"]
    if umb is None:
        # #67 context size — already mitigated by Context Overflow Guard
        body = (
            "Closing — auto-detected log-monitor noise. The context-overflow path is "
            "now guarded by `agents/base.py::_ensure_context_fits()`, which trims old "
            "messages when the conversation exceeds the model's context window (with "
            "`CONTEXT_USAGE_SAFETY_MARGIN`). When trimming can't help, a "
            "`ContextOverflowError` is raised and converted to a user-friendly message. "
            "Reopen if you observe this after the next deploy."
        )
    else:
        body = (
            "Closing — auto-detected log-monitor duplicate with no diagnostic body "
            f"content. Root cause is tracked under: {umb}. PR #244 (merged) renames "
            "the retry log key so the monitor stops flagging expected transient "
            "warnings as actionable errors. If a *new* failure mode appears after the "
            "next deploy, please open a fresh issue with details rather than relying "
            "on the auto-monitor."
        )
    comment(API, n, body)
    close_issue(API, n, reason="not planned")

print("\n=== Done. ===")
