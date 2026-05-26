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
| `$4` | `evict` — controls eviction before the heavy steps (img23d, mesh2parts).  Pass `false` only if you know no LLM session is running. | true |
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
`scripts/*.sh` invocations must see the same `API_BASE` — do
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

IF `$3`, RUN EXACTLY THIS:

```bash
curl -sS -X POST "$API_BASE/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $LLMMLL_AUTH_TOKEN" && \
    sleep 5 && \  # give the runner a moment to recover if eviction happened
    API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
    .claude/skills/generate-3d/scripts/txt2img.sh "<prompt>"
```

ELSE:

```bash
API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
  .claude/skills/generate-3d/scripts/txt2img.sh "<prompt>"
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

IF `$3`, RUN EXACTLY THIS:

```bash
API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
  .claude/skills/generate-3d/scripts/img2img.sh <last_image.png> "<edit prompt>" \
  qwen-image-edit-2511 0.75
```

ELSE:

```bash
curl -sS -X POST "$API_BASE/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $LLMMLL_AUTH_TOKEN" && \
    sleep 5 && \  # give the runner a moment to recover if eviction happened
    API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
    .claude/skills/generate-3d/scripts/img2img.sh <last_image.png> "<edit prompt>" \
    qwen-image-edit-2511 0.75
```

Show the result, loop back to Step 4.

### Step 5 — Background removal

No eviction needed — rembg is ~1.6 GB and `IN_PROCESS_AUTO_UNLOAD`
cleans it up after the request:

```bash
API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
  .claude/skills/generate-3d/scripts/rembg.sh <last_image.png>
```

The script writes ``<OUT_DIR>/rembg_<ts>_cutout.png``.  Read it to
show the user.  Confirm it looks right (skip confirmation if `$3`);
if it's wrong, the right fix is usually to refine the input image
(Step 4b), not to retry rembg.

### Step 6 — 3D generation (heavy step)

IF `$3`, RUN EXACTLY THIS:

```bash
curl -sS -X POST "$API_BASE/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $LLMMLL_AUTH_TOKEN" && \
    sleep 5 && \  # give the runner a moment to recover if eviction happened
    API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
    .claude/skills/generate-3d/scripts/img2-3d.sh <cutout.png>
```

ELSE:

```bash
API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
    .claude/skills/generate-3d/scripts/img2-3d.sh <cutout.png>
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


```bash
curl -sS -X POST "$API_BASE/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $LLMMLL_AUTH_TOKEN" && \
    sleep 5 && \  # give the runner a moment to recover if eviction happened
    API_BASE="$API_BASE" OUT_DIR="$OUT_DIR" \
  .claude/skills/generate-3d/scripts/mesh2parts.sh <mesh.glb> 256 42
```

The script always emits per-part files; no separate ``split`` toggle.
Outputs:

- `${ID}_decomposed.glb` — assembled mesh, each part as a named
  primitive (drop-in for Blender → separate objects)
- `${ID}_exploded.glb` — parts spatially separated for inspection
- `${ID}_part_NN.glb` — one file per part
- `${ID}_bbox.glb` / `${ID}_gt_bbox.glb` — segmentation debug

Open the exploded view (most informative):

```bash
xdg-open "$OUT_DIR/${ID}_exploded.glb"
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
- **Empty `.glb` outputs from mesh2parts** — XPart's diffusion
  produced no valid SDF surfaces.  Rare with Hunyuan3D-2.1 inputs;
  re-run with a different seed (3rd positional to
  `mesh2parts.sh`) or refine the input image first.
- **Ctrl-C mid-pipeline** — the script returns and the api will
  surface a 499.  Run `evict-all` afterwards to free VRAM/RAM in
  case the runner is still holding the model.
