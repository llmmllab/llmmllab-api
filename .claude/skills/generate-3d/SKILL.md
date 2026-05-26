---
name: generate-3d
description: |
  Walk a user from a text description to a final 3D mesh (and optional
  per-part decomposition).  Conducts a short interview to clarify
  intent, formulates a prompt per ./generate_3d_models.md, generates
  an image, removes its background, lifts it to 3D, and (optionally)
  decomposes into parts.  Force-evicts any resident LLM session before
  the heavy 3D steps so XPart's ~17 GB transient load doesn't OOM the
  pod (the live runner is capped at 26 GiB of system RAM).
argument-hint: [description] [out-dir] [auto] [evict] [parts]
allowed-tools:
  - Bash
  - Read
  - Edit
  - AskUserQuestion
---

# Skill: `/generate-3d`

Drive the full text-to-3D workflow described in
`./generate_3d_models.md` end-to-end from a single conversation turn.
When the user invokes this skill, you become a co-author of their 3D
asset: interview them about the subject, write the prompt, run the
four-step (or five-step, with parts) pipeline, and show them the
result.

## Resource model — read first

The runner pod (`lsnode-3`) is capped at **26 GiB of system RAM**.
That ceiling matters because:

- **Hunyuan3D-2.1 (img23d)** uses ~8 GB transient during weight load.
- **Hunyuan3D-Part / XPart (img23d_part)** uses ~17 GB transient
  during weight load.
- An **active Qwen-27B chat session** on the runner holds ~5–8 GB of
  system RAM (KV-offload + context checkpoints).

So a Qwen session + XPart load = ~25 GB peak, which is at the cap
with no headroom.  In practice that combination OOMKills the pod
(exit 137), losing every in-flight request.

The cluster has two automatic cleanup hooks but neither is enough on
its own:

- `IN_PROCESS_AUTO_UNLOAD=1` (live on the cluster) — the runner
  unloads in-process pipelines after each request.  Covers
  back-to-back pipeline calls, **does not** cover a concurrent chat
  session.
- `IMG_SERVER_AUTO_SHUTDOWN` — referenced in the api code but **not
  currently set** on the cluster deployment.  Treat as off.

**Therefore: evict any resident LLM session BEFORE the heavy steps
(img23d in Step 6, img23d_part in Step 7).**  This is the only thing
that prevents a Qwen + XPart collision.  The skill defaults to
evict-before-heavy-step.  The user can opt out only by passing
`evict=false`, and only if they know no chat session is running.

## Arguments

The user can pass arguments via skill invocation:

```
/generate-3d "ceramic mug" /tmp/model true false true
```

Arguments are positional:

| Position | Description | Default |
|----------|-------------|---------|
| `$1` | Short text description of the 3D object.  Skips the Step 1 interview. | (ask user) |
| `$2` | Output directory for all generated files.  Skips the output-dir question. | `/tmp/generate-3d-<descriptor>` |
| `$3` | `auto` — auto-accept all recommendations.  If `$1` is empty, still asks for the subject brief once, then auto-accepts everything else. | false |
| `$4` | `evict` — controls eviction before the heavy steps (img23d, img23d_part).  Pass `false` only if you know no LLM session is running. | true |
| `$5` | `parts` — run parts decomposition (Step 7) automatically after 3D generation. | false (overridden by interview signal — see Step 7) |

Boolean args accept `true`/`false`/`1`/`0`/`yes`/`no`.  An empty
string means "use the default", not "false".

## Required environment

```bash
API_BASE          # e.g. http://192.168.0.71:9999  (the test scripts default to this)
LLMMLL_AUTH_TOKEN # admin-capable bearer token
```

If `API_BASE` isn't set, use `http://192.168.0.71:9999` (matches the
scripts' built-in default).  All `curl` calls and all
`./scripts/test_*.sh` invocations must see the same `API_BASE` — do
not introduce a second variable name; the scripts ignore anything
else.

`OUT_DIR` comes from `$2`, the Step 1 interview, or a sensible
default derived from the subject description.

## Workflow

### Step 1 — Interview

Ask the user what they want to create AND where to output files.
Use a single `AskUserQuestion` with both questions where possible.

**Argument overrides:**

- If `$1` is set, skip the subject brief and use it directly.
- If `$2` is set, skip the output-dir question; create it with
  `mkdir -p`.
- If `$3` (auto) is true and `$1` is set, skip all questions in this
  step.  Derive `OUT_DIR` from `$2` or default to
  `/tmp/generate-3d-<short-descriptor>`.

**What to create** (interactive only): restate your understanding
in one or two sentences.  Iterate via `AskUserQuestion`
(header "Subject brief", options ``"Yes, that's it"`` /
``"Adjust: ..."`` / ``"Start over"``) until confirmed.  If `$3` is
true but `$1` is empty, ask the brief once and accept the first
answer without iteration.

Beyond "what's the object", listen for:

- **Material** — ceramic / metal / wood / plastic / glass.  Drives
  prompt vocabulary.  **Glass / transparent subjects are
  XPart-hostile** (no SDF surface to mesh) — warn the user before
  proceeding and offer to switch the material.
- **Style** — photorealistic / stylized / cartoon.  Default to
  photorealistic product photography (what Hunyuan3D-2.1 was trained
  on; what Blender expects downstream).
- **Use case** — game asset / 3D printing / just to see.  This
  drives the Step 7 parts default (see below).

**Where to output files** (interactive only): ask via
`AskUserQuestion` (header "Output dir") with default
`/tmp/generate-3d-<short-descriptor>` (e.g.
`/tmp/generate-3d-ceramic-mug`).  Create with `mkdir -p` before
Step 3.

Save the brief, the inferred use case, and `OUT_DIR` for later
steps.

### Step 1.5 — Pre-flight evict (always runs unless `$4=false`)

Before any pipeline work, clear any resident LLM session so the
first heavy step starts on a clean pod.  This is the cheap insurance
that prevents the Qwen + XPart collision described in the resource
model above.

```bash
curl -sS -X POST "$API_BASE/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $LLMMLL_AUTH_TOKEN"
```

Skip only if `$4=false` AND the user has explicitly said they want
to preserve a chat session.  In interactive mode with `$4` unset,
ask once:

> "About to evict any running LLM/image servers on the runner so the
> 3D pipeline has the full 26 GiB pod budget.  Skip eviction?"

Default answer: "No, evict it" (recommended).

### Step 2 — Prompt formulation

Read `./generate_3d_models.md` (sections "Step 1 — txt2img" and
"Prompting strategy") for the canonical guidance.  Construct a
prompt that satisfies:

1. **One subject, centered** (no scenes, no environments)
2. **Soft studio lighting, light gray seamless background**
3. **Explicit material vocabulary** matching the brief
4. **Photorealistic product photography, sharp focus, high detail**

Avoid: scenes ("on a table"), strong shadows, transparent / glass
subjects, text or UI elements.

If `$3` is true: skip confirmation, show the prompt as a one-liner,
proceed.
Otherwise: confirm via `AskUserQuestion` (header "Prompt", options
``"Run it"`` / ``"Refine: ..."`` / ``"Start over"``).  Loop until
confirmed.

### Step 3 — Run txt2img

Eviction is **not** needed here — sd-server's footprint is small
(~2 GB system RAM) and doesn't collide with anything.

```bash
API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
  ./scripts/test_txt2img.sh "<prompt>"
```

The script writes ``<OUT_DIR>/txt2img_<ts>.png`` and prints the path
on its last line.  Capture it.

### Step 4 — Show and confirm the image

Read the generated PNG with the `Read` tool.

If `$3` is true: skip confirmation, proceed to Step 5.

Otherwise ask:

- ``"Continue to background removal"`` → Step 5
- ``"Refine via img2img"`` → Step 4b
- ``"Regenerate from scratch"`` → Step 2 with a new prompt
- ``"Abort"`` → exit

### Step 4b — Optional img2img edit

Only if the user picks "Refine via img2img".  Construct an edit
prompt following `./generate_3d_models.md`'s rule: **additive edits
are most reliable; full-surface material/colour changes are flaky.**
Warn the user if their requested change is the latter.

No eviction needed (still sd-server territory):

```bash
API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
  ./scripts/test_img2img.sh <last_image.png> "<edit prompt>" \
  qwen-image-edit-2511 0.75
```

Show the result, loop back to Step 4.

### Step 5 — Background removal

No eviction needed — rembg is ~1.6 GB and `IN_PROCESS_AUTO_UNLOAD`
cleans it up after the request:

```bash
API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
  ./scripts/test_rembg.sh <last_image.png>
```

The script writes ``<OUT_DIR>/rembg_<ts>_cutout.png``.  Read it to
show the user.  Confirm it looks right (skip confirmation if `$3`);
if it's wrong, the right fix is usually to refine the input image
(Step 4b), not to retry rembg.

### Step 6 — 3D generation (heavy step)

**Eviction matters here.**  Hunyuan3D-2.1 needs ~8 GB transient.
If `$4` defaulted true (and Step 1.5 ran) the pod is already clear
and you can skip a second evict.  If the user has had a long
interactive session that could have re-spawned an LLM (e.g. they
asked you questions in another window between Step 1 and now), evict
again just before this step.

When in doubt, evict.  It's cheap (~50 ms when nothing is loaded):

```bash
curl -sS -X POST "$API_BASE/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $LLMMLL_AUTH_TOKEN"
```

Then run:

```bash
API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
  ./scripts/test_img2-3d.sh <cutout.png>
```

The script writes ``<OUT_DIR>/<id>.glb``.  **Tell the user this
takes ~3 minutes** before kicking it off.

Capture the `id` from the JSON response — every subsequent open /
parts call needs it substituted explicitly (do not pass a literal
`<id>`):

```bash
open "$OUT_DIR/${ID}.glb"   # macOS — xdg-open on Linux
```

### Step 7 — Optional parts decomposition (heaviest step)

Whether to run this defaults from the use-case signal captured in
Step 1:

- "game asset" → default `parts=true`
- "3D printing" → default `parts=true` (each shell prints better
  separated)
- "just to see" → default `parts=false`

`$5` overrides the default explicitly.  In interactive mode, confirm
via `AskUserQuestion`.

**This is the highest-memory step in the pipeline (~17 GB transient
during XPart load).  Evict unconditionally before it runs**, even if
you evicted earlier — long-running interactive sessions are exactly
when an LLM tends to come back:

```bash
curl -sS -X POST "$API_BASE/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $LLMMLL_AUTH_TOKEN"
```

Default `octree_resolution=256` on a 26 GiB pod.  Bump to 512 only
if the user explicitly asks for higher-detail decomposition AND
confirms no chat session is running.  The skill passes 256 + seed 42
+ split=1 by default:

```bash
API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
  ./scripts/test_img2-3d-parts.sh <mesh.glb> 256 42 1
```

`split=1` (4th positional) gives one `.glb` per detected part.
Outputs:

- `${ID}_decomposed.glb` — assembled mesh, each part as a named
  primitive (drop-in for Blender → separate objects)
- `${ID}_exploded.glb` — parts spatially separated for inspection
- `${ID}_part_NN.glb` — one file per part
- `${ID}_bbox.glb` / `${ID}_gt_bbox.glb` — segmentation debug

Open the exploded view (most informative):

```bash
open "$OUT_DIR/${ID}_exploded.glb"
```

### Step 8 — Final eviction + summary

Final eviction so nothing's left resident:

```bash
curl -sS -X POST "$API_BASE/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $LLMMLL_AUTH_TOKEN"
```

Print a summary listing every produced file:

```
✓ Generated 3D asset for "<one-line brief>"

  base:      $OUT_DIR/txt2img_*.png
  edited:    $OUT_DIR/img2img_*.png       (if step 4b ran)
  cutout:    $OUT_DIR/rembg_*_cutout.png
  mesh:      $OUT_DIR/${ID}.glb
  parts:     $OUT_DIR/${ID}_part_*.glb    (if step 7 ran)
  exploded:  $OUT_DIR/${ID}_exploded.glb  (if step 7 ran)

Total wall-clock: <N> seconds
```

## Troubleshooting

- **Pod OOMKilled (exit 137) mid-pipeline** — almost certainly a
  concurrent LLM session collided with XPart.  Wait for the pod to
  restart (`kubectl rollout status deploy/llmmllab-runner -n llmmllab`),
  then rerun with `$4=true` (the default) so Step 1.5 evicts up
  front.
- **`503 Service Unavailable` on /v1/images/3d/parts** — usually a
  missing XPart dep in the runner image.  Check `kubectl logs` for
  the runner pod.
- **Empty `.glb` outputs from img23d_part** — XPart's diffusion
  produced no valid SDF surfaces.  Rare with Hunyuan3D-2.1 inputs;
  re-run with a different seed (5th positional to
  `test_img2-3d-parts.sh`) or refine the input image first.
- **Ctrl-C mid-pipeline** — the script returns and the api will
  surface a 499.  Run `evict-all` afterwards to free VRAM/RAM in
  case the runner is still holding the model.
