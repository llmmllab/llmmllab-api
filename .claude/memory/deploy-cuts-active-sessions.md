---
name: deploy-cuts-active-sessions
description: "Deploying llmmllab-api rolls the pod and cuts the user's in-flight Claude Code session"
metadata:
  type: feedback
---

Pushing to `llmmllab-api` main triggers "Deploy API", which rolls the single api pod. Any in-flight Claude Code session talking to the api (the user often has one open against this stack) gets its streaming request cut when the old pod terminates (uvicorn graceful shutdown is 30s). The user hit this as a session "stopping while processing prefill" right after a deploy.

**Why:** The user frequently runs a live Claude Code session against llmmllab-api while I'm working on it; an uncoordinated deploy interrupts their actual work.

**How to apply:** Before pushing an api fix, tell the user it will roll the pod (~3–4 min build, then the in-flight request is cut) and let them choose timing — unless it's the fix *for* a bug actively killing their session, in which case deploy promptly but say so. The build delay usually lets their current turn finish. Related: [[llmmllab-api-token-and-overflow-context]].
