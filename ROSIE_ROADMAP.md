# Rosie Roadmap — from homelab stack to self-hostable product

> **Rosie** = Radically Open-Source Self-hosted Intelligence Engine.
> Migration of the `llmmllab` org from a single-operator homelab deployment into an easy-to-install, self-hostable, multi-backend product — then a coordinated rebrand to **rosie**.

Generated from a full audit of all 6 org repos (112 findings) + a cross-repo dependency map.
Owner's stated sequence: **(1) make it self-hostable → (2) rebrand to rosie → (3) support non-CUDA backends.**

---

## 1. The system at a glance

| Repo | Role | Lang | Visibility | Buildable image? |
|------|------|------|-----------|------------------|
| `llmmllab-api` | Inference gateway (OpenAI/Anthropic wire, LangGraph, DB) — **the hub** | Python | **public** | yes |
| `llmmllab-runner` | llama.cpp / SD.cpp / Hunyuan3D host — **the inference leaf** | Python | **public** | yes (CUDA) |
| `mcp-server-web` | Web-search/fetch MCP (SearXNG + Playwright) | Python | **public** | yes |
| `llmmllab-gateway` | k8s Gateway-API routing (NGINX Gateway Fabric) | YAML | **public** | no (config only) |
| `llmmllab-ui` | React/TS SPA front-end | TS | **public** | no (static, rsync) |
| `openclaw-k8s` | Deploys upstream **OpenClaw** agent (personal assistant) | mixed | private | no (upstream image) |

**Runtime topology:**
```
ui ──https──▶ api ──http──▶ runner (inference, GPU leaf)
                  └─http──▶ mcp-server-web ──▶ SearXNG
openclaw ─▶ api (LLM/embeddings)  +  openclaw ─▶ mcp-server-web (/mcp)
gateway fronts: api (Service `llmmllab`:9999) + mcp-server-web (/web/:8000)
externals: auth.longstorymedia.com (OIDC), 192.168.0.71:31500 (registry),
           HuggingFace, RabbitMQ, Tempo/Loki/Prometheus/DCGM, Nextcloud/Slack/Telegram (openclaw)
```

**Critical naming gotcha:** the api's k8s **Service is literally `llmmllab`** (not `llmmllab-api`). The repo, image, Deployment, and label are `llmmllab-api`; the Service that everyone resolves (`llmmllab.llmmllab.svc:9999`) is `llmmllab`. **Repo/image rename is independent of the Service rename** — they are separate cutovers, and the Service rename is the riskiest one.

---

## 2. 🚨 Phase 0 — Stop the bleeding (do this now, independent of everything)

Five of six repos are **public** and contain **live committed secrets**. This is exposed today.

| Item | Where | Action |
|------|-------|--------|
| DB / auth / RabbitMQ / internal-API secrets | `api k8s/env.yaml:106-110` | **Rotate now**, move to generated per-install secrets, delete file from working tree |
| Private-registry basic-auth creds | `api Makefile:74-75`, `api scripts/push_manifest.py:9-10,43` | Rotate registry creds; remove from source |
| SPA confidential `client_secret` | `ui src/config/index.ts:7,26,32` | Move IdP to public/PKCE client (no secret in a browser bundle) |
| `RAW_TOKEN_DEBUG=true` + `LOG_LEVEL=debug` + raw bearer logging | `api k8s/deployment.yaml:116,119`, `env.yaml:95`, `middleware/auth.py:355` | Default **off**; these write user content + tokens to disk/logs |
| Personal account creds (Google/Slack/Telegram), API keys | `openclaw-k8s` (private, lower urgency) | Rotate, template |

**Then scrub git history** (`git filter-repo`/BFG) on the public repos and force-push — coordinate because it rewrites history (breaks existing clones/forks).

**Exit criteria:** no valid credential exists in any public repo's working tree *or* history; debug data-leak flags off by default.

---

## 3. Prioritization — impact × complexity

```
                    LOW complexity                 HIGH complexity
            ┌─────────────────────────────┬──────────────────────────────┐
   HIGH     │ QUICK WINS (Phase 1)        │ STRATEGIC (Phases 2–3)        │
  impact    │ • public-registry default   │ • Helm chart / compose stack  │
            │ • untrack runner .models.*  │ • weights bootstrap           │
            │ • fix .env.example mismatch  │ • auth decoupling (api + ui)  │
            │ • ui prod-URL fallback bug  │ • private-infra templating    │
            │ • neutral DB user           │ • UI containerize + runtime cfg│
            │ • secret rotation (Phase 0) │                               │
            ├─────────────────────────────┼──────────────────────────────┤
   LOW      │ FILL-IN (during Phase 4)    │ DEFER / LATER (Phase 5)       │
  impact    │ • doc fixes (TRELLIS, CUDA) │ • multi-backend build matrix  │
            │ • logo / header strings     │ • HardwareManager abstraction │
            │ • CODEOWNERS, URLs          │ • fp16/dtype guards           │
            │ • remove owner dev scripts  │ • Service/DB/namespace rename │
            └─────────────────────────────┴──────────────────────────────┘
```

---

## 4. Phased roadmap

### Phase 1 — Quick wins + your stated near-term priorities
*Goal: kill the silent footguns and land the two changes you already called out (public registry, user-owned models.yaml). Low effort, high signal.*

- **Default to a public registry** (`ghcr.io/<org>` or `docker.io/<org>`):
  - flip `REGISTRY` default in `api/Makefile:5`, `runner/Makefile:10`, `mcp/Makefile:2`; the k8s `image:` fields are **literal** (`api/k8s/deployment.yaml:21`, `runner/k8s/deployment.yaml:31,207`, `mcp/k8s/deployment.yaml:20`) → templatize.
  - public HTTPS registry supports `docker buildx --push` directly → **delete** the insecure-HTTP workaround (`api/scripts/push_manifest.py`, the `--load`+`docker push` dance) and the bespoke registry-cleanup subsystem (`runner/scripts/registry_cleanup.py`, `runner/k8s/registry-cleanup-cronjob.yaml`).
  - converge multi-arch: CI builds **arm64-only** (`mcp deploy.yml`) and runner **amd64-only** — pick a coherent multi-arch + immutable-tag strategy.
  - **publish images** so users don't build the (heavy, CUDA) images locally.
- **Make `.models(.*)?.yaml` a user concern** (runner):
  - `.gitignore:11-15` currently *force-tracks* `.models.yaml` + `.models.small.yaml` despite README/CLAUDE claiming they're ignored — **git-untrack them**, ship only `.models.example.yaml`.
  - introduce a single configurable `MODELS_ROOT` and **relative** paths (today every entry is absolute `/models/...`; `Dockerfile:228,292` and `generate_models_yaml.py:622` hardcode `/models`).
  - make `generate_models_yaml.py` a first-boot scaffolding step.
- **Fix first-run footguns:**
  - **ui `.env.example` documents the wrong variable names** — code reads `VITE_BASE_URL/VITE_ISSUER/VITE_CLIENT_ID` but the example/README say `VITE_API_BASE_URL/VITE_AUTH_*` (`ui src/config/index.ts:5-8` vs `.env.example`). A user who follows the docs silently falls back to **your** backend `https://ai.longstorymedia.com` (`src/config/index.ts:19`). High-impact, trivial fix.
  - api `.env.example` is missing keys it actually reads (`RUNNER_ENDPOINTS`, `MCP_WEB_TOOLS_URL`, `CONFIG_DIR`, …); mcp ships none. Add complete templates.
  - neutral DB defaults: `DB_USER=lsm` appears in `api .env.example:4`, `docker/init.sql:5`, `k8s/postgres-statefulset.yaml:62` — pick one neutral default (`rosie`) and a single source of truth.
- **Repo hygiene that MUST precede any rename `sed`:**
  - `ui src/types/openai/` has corrupted artifacts from a prior botched rename (`$1$2.ts`, `.!76501!*.ts`, `U…U…` dupes) — these dominate the 1273-file count. Clean first.
  - stale docs: api README claims a CUDA Dockerfile (it's `python:3.12-slim`); runner README says **TRELLIS** but code is **Hunyuan3D-2.1**, and claims "no test suite" (there are ~9); align the `.gitignore` vs README models.yaml story.
- **Decouple CI from owner infra:** every `deploy.yml` is `runs-on: self-hosted` on `lsnode-3` pushing to the private cluster — gate/replace with GitHub-hosted runners pushing to the public registry (or drop auto-deploy).

**Exit criteria:** images pull from a public registry; a fresh clone has no force-tracked model configs; `.env.example` matches code in all repos; CI doesn't require your cluster.

---

### Phase 2 — Make a stranger able to deploy it (the heart of "self-hostable")
*Goal: one documented command yields a working stack on hardware that is not yours. This is the highest-impact, highest-complexity phase.*

- **Two install paths, parameterized:**
  - **Single-box `docker compose up`** (the most likely stranger target): api + runner + postgres + redis + mcp-server-web + **SearXNG** + ui, wired together. Today compose only starts Postgres (`api/docker-compose.yml`); mcp/ui have none.
  - **Helm chart (or Kustomize base+overlays)** for k8s, with a `values.yaml` covering everything currently hardcoded:
    - LAN IPs `192.168.0.71` (DB/Redis/gateway), NodePorts `32345/32346/30081`, `INTERNAL_ALLOWED_IPS` CIDRs (`api`).
    - node pinning `lsnode-3/4` (`runner nodeSelector` + `pvc nodeAffinity`), `hostPath /models /slots /data`, `storageClassName: ""` → **default StorageClass / dynamic provisioning**.
    - `privileged: true` + `SYS_ADMIN/SYS_RAWIO` on the runner → **non-privileged default**, GPU power-capping made opt-in.
    - in-cluster Service FQDNs (`*.llmmllab.svc`) as overridable values with sane localhost defaults.
    - templated secret generation (the `apply.sh` currently pulls secrets from *foreign namespaces* — `api/k8s/apply.sh:17-19` — so it fails on a fresh cluster).
- **Model-weights bootstrap:** the runner expects tens-to-hundreds of GB of (partly **HF-gated**) weights pre-staged on the host with `HF_HUB_OFFLINE=1` (nothing auto-downloads). Add a download/init job + documented gated-model flow, or ship a small default model set.
- **Bundle SearXNG:** every web tool needs it; it's an undocumented hard dependency today (`mcp k8s/deployment.yaml:38`). Ship a compose service / k8s manifest.
- **Containerize the UI** with **runtime** config injection (Vite bakes env at build time → a prebuilt image can't be repointed without a rebuild; add a `/config.json` fetched at boot).
- **Make hardware tuning configurable, not baked:** per-model `main_gpu`/`tensor_split` pinned to specific 3090/3060/2060 indices (`runner .models.yaml`), VRAM floors (`min_vram_gb` hardcoded in pipelines, not yaml fields), memory caps tuned to a 32GiB node, and gateway `9999s/600s` timeouts.
- **Gateway portability:** assumes NGINX Gateway Fabric (`gatewayClassName: nginx` + `gateway.nginx.org` CRDs) pre-installed; the catch-all `/` route defaults to a **Gmail/OAuth** backend (`gateway routes.yaml:92`); the `snippetsfilter.yaml` is orphaned (never applied). Add a README, fix the default route, document the controller prerequisite (or ship a portable ingress option).

**Exit criteria:** a documented `docker compose up` (single box) **and** `helm install` (k8s) both bring up a working chat + image + web-search stack on non-owner hardware, with no edits to manifests and no foreign-namespace secrets.

---

### Phase 3 — Auth that works for someone who isn't you
*Goal: a stranger can run single-user/no-auth out of the box, OR bring their own OIDC — without your IdP. Can overlap Phase 2.*

- **api:** `AUTH_ISSUER/JWKS/AUDIENCE` default to `auth.longstorymedia.com` and `app.py` **raises `ValueError` if no JWKS URI** (`middleware/auth.py:535`) — there is *no* auth-less path. Add: configurable issuer/audience, an **API-key-only** mode, and an **optional/dev no-auth single-user** mode that doesn't require a JWKS endpoint.
- **ui:** OIDC issuer hardcoded, auth **mandatory** (forces `signinRedirect`), `isAdmin` keyed off a literal `'admins'` group, and a whole **LDAP-style user-manager API** assumed at `issuer + /api` (`ui src/api/usrmgr.ts`) that generic OIDC providers don't have. Add a single-user/no-auth mode; make user-management an optional feature; move to public PKCE client (ties into Phase 0).
- **runner & mcp:** both bind `0.0.0.0` with **no auth** (trust = "only reachable via the api on a private network"). For a product where users may flatten the network, document the trust model and add optional token/allowlist (mcp is an SSRF-capable fetch endpoint).
- Keep **both** auth modes (JWKS + API-key) supported on the api throughout — openclaw and other clients depend on the API-key path.

**Exit criteria:** `docker compose up` yields a usable single-user instance with zero IdP setup; BYO-OIDC is documented and works against a stock provider (e.g. Keycloak/Authentik) without the `/api` user-manager.

---

### Phase 4 — Rebrand to Rosie
*Goal: nothing user-facing says `llmmllab`/`LongStoryMedia`; clean rosie identity. Only after the stack is self-hostable.*

- **Org & repo cutover:** consolidate the **two-org split** (`LongStoryMedia` ⇄ `llmmllab` — note the api's working clone pushes to `LongStoryMedia/llmmllab-api`) under one `rosie` org; rename repos; rely on GitHub redirects; re-point self-hosted runner registrations + deploy keys.
- **Image renames in lockstep** with k8s `image:` refs + CI (per §5): push new-named image → update `deployment.yaml image:` → update Makefile/CI.
- **Identifier refactors** (not string swaps): api `llmmllogger`/`LlmmlLogger` (imported in ~77 files), `pyproject` names, OTEL `service.name`; ui `ls-ai-ui`, `LllabUser`/`getLllabUsers`, `lsm-client`.
- **Logo & branding:** replace **Nurturebot/beaker/leaf** assets (`ui public/nurturebot*.png`, `Title.tsx`, `index.html` favicon) with the **Rosie-from-the-Jetsons** logo; header strings (`TopBar.tsx:44`, `index.html:8`); maintainer email `scott@llmmllab.com` (`mcp Dockerfile:6`); CODEOWNERS `@longstoryscott`.
- **Gateway** resource names must change **atomically** across `gateway.yaml`+`routes.yaml` (parentRefs/sectionName cross-reference).
- **Decide the deep-identifier renames separately** (high risk, low user value — see Decisions): the k8s **Service `llmmllab`**, the **namespace `llmmllab`**, and the **DB name `llmmllab`**. Recommendation: keep these stable internal names and rebrand only user-facing + repo/image, to avoid data migration. If renamed, do the Service/namespace/DB together with a migration and update **every** consumer FQDN (openclaw, api env, gateway, runner) in one rollout.
- **openclaw:** rename only the `llmmllab` coupling; **do NOT** rename OpenClaw's own identifiers (image, config keys, `/openclaw` basePath, plugin names) — they're wire-compat with the upstream project.

**Exit criteria:** public docs, UI, image names, and repo/org URLs are all "rosie"; runtime wiring unbroken; OpenClaw identifiers untouched.

---

### Phase 5 — Multi-backend (CUDA → ROCm / Apple Metal / CPU)
*Goal: rosie runs on non-NVIDIA hardware. Highest complexity, lowest urgency — but stage cheap foundations earlier.*

Almost all backend coupling lives in **`llmmllab-runner`** (the api/ui are HTTP-layer agnostic):
- **Build matrix:** all 4 Dockerfile stages are `nvidia/cuda:12.8.1`, `CMAKE_CUDA_ARCHITECTURES="75;86"`, cu121 torch wheels, flash-attn cu12 wheel, spconv-cu124 → parameterized base images + per-backend wheel sets + build flags (this also excludes Ada/Hopper/Blackwell and pre-Turing today).
- **Device abstraction:** `utils/hardware_manager.py` (nvsmi/nvidia-smi power/thermal), `utils/dcgm_metrics.py`, `pipelines/_gpu_select.py` (torch.cuda only) → implement the existing `ThermalThrottler`/`HardwareManager` Protocol for ROCm/Metal/CPU; VRAM/selection paths aren't abstracted yet.
- **dtype guards:** fp16-only casts gated on `cuda` (`pipelines/rembg/rmbg.py`), XPart spconv/bbox fp32-only — test/guard for Metal/CPU/ROCm.
- **Brittle build patches:** the Dockerfile `sed`-rewrites vendored Hunyuan3D-Part source (`Dockerfile:225-263`) including a `/models/sonata` hardcode — proper packaging before multi-backend.
- **api/ui:** make the `CUDA_*` env passthrough backend-conditional; parameterize vision-token estimation (tuned for Qwen-VL); let GPU-health UI panels degrade gracefully when the backend reports no GPU.

**Stage early (cheap, in Phases 1–2):** add a **backend tag** to `.models.yaml` entries, expose VRAM floors as yaml fields, and keep the model schema stable — so the multi-backend work is additive later.

**Exit criteria:** documented, tested install on at least one non-CUDA backend (recommend **Apple Silicon** first for laptop self-hosters, or **CPU** as the universal baseline).

---

## 5. Cross-cutting sequencing constraints (don't break the wiring)

1. **Service name `llmmllab` is load-bearing and decoupled from the repo/image name.** Renaming the api repo/image is runtime-safe as long as the Service stays `llmmllab`. Renaming the Service requires lockstep edits to `openclaw deployment.yaml` + `openclaw.json` + `init-config.mjs` + `gateway routes.yaml`.
2. **Image rename order (per image):** push new-named image to registry **first** → update `deployment.yaml image:` → update Makefile/CI.
3. **Runner ↔ api:** the api hardcodes `RUNNER_ENDPOINTS` to Services `llmmllab-runner(-small)`. Rename those Services and the api env in the **same rollout**; keep the shared **Model JSON schema** (`models/model_parameters.py`: `num_ctx`, `tensor_split`, `parallel`) unchanged across the boundary or runner-selection breaks silently.
4. **mcp-server-web Service rename** must update 3 consumers in lockstep: api `MCP_WEB_TOOLS_URL`, `openclaw.json mcp.servers.web.url`, gateway `/web/` backendRef. The `/mcp` path suffix is also load-bearing.
5. **Namespace `llmmllab` rename** ripples through every `*.svc.cluster.local` FQDN in openclaw, api, runner, gateway + the ReferenceGrant + `INTERNAL_ALLOWED_IPS` pod-CIDR assumptions.
6. **Org/repo rename** breaks openclaw's self-modifying `git push` (`init-config.mjs` hardcodes the 5-repo list + safe.directory + SSH deploy key scope) and self-hosted runner registration — update those together.
7. **Shared wire contracts to preserve:** OpenAI/Anthropic protocol shapes (ui generates types from `/openapi.json`); runner HTTP API incl. `/v1/status startup_epoch` (StaleServerError recovery); model-ID names (openclaw references `llmmllab/<id>` verbatim); the search envelope `contents` key the api consumes.

---

## 6. Proposed naming scheme (for your approval)

| Old | New (proposed) |
|-----|----------------|
| org `llmmllab` (+ `LongStoryMedia`) | `rosie` (GitHub org; fall back to `rosie-ai`/`getrosie` if taken) |
| `llmmllab-api` | `rosie-api` (or `rosie-engine`) |
| `llmmllab-runner` | `rosie-runner` |
| `mcp-server-web` | `rosie-web` (or `rosie-mcp-web`) |
| `llmmllab-gateway` | `rosie-gateway` |
| `llmmllab-ui` | `rosie-ui` |
| `openclaw-k8s` | keep / `rosie-assistant` *(see decision)* |
| images | `ghcr.io/rosie/<name>` |
| internal Service/namespace/DB `llmmllab` | **keep stable** unless you accept a migration |

---

## 7. Decisions

**Resolved (2026-06-02):**
1. **OpenClaw scope → keep separate.** `openclaw-k8s` is your personal assistant that *consumes* Rosie, not part of the product. Only de-personalize its `llmmllab` coupling (URLs, namespace, your accounts); do **not** rename OpenClaw's own identifiers (image, config keys, `/openclaw` path, plugins) — wire-compat with upstream.
2. **Headline install → compose-first.** Single-box `docker compose up` is the primary path; Helm/k8s is the advanced option. (Reflected in Phase 2.)
3. **Auth → no-auth single-user by default + optional BYO-OIDC + API-key.** Remove the hard JWKS requirement; add a no-auth path. (Reflected in Phase 3.)
4. **First non-CUDA backend → Apple Silicon / Metal.** (Reflected in Phase 5; CPU baseline is the natural companion since it's nearly free once the device abstraction exists.)

**Still open:**
5. **Target public registry:** GHCR vs Docker Hub vs both? (GHCR pairs naturally with the GitHub org rename.)
6. **Deep-identifier rename:** keep internal Service/namespace/DB name `llmmllab` stable (recommended — avoids a data migration), or rename it and accept the migration + multi-repo lockstep?
7. **Final names:** confirm the §6 naming scheme (esp. `rosie-api` vs `rosie-engine`, and the GitHub org handle if `rosie` is taken).

## 8. Top risks

- Already-public secrets — history rewrite breaks existing clones/forks; coordinate.
- Service/namespace/DB rename = data migration + multi-repo lockstep; easy to half-do and silently break runner selection.
- openclaw's self-modifying git push + self-hosted runners break on org/repo rename.
- UI build-time env baking blocks reconfigurable prebuilt images until the runtime-config shim lands.
- Model schema drift between api and runner during refactors fails *silently* (server allocation).
