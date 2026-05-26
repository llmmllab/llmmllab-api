---
name: generate-3d
description: |
  Walk a user from a text description to a final 3D mesh (and optional
  per-part decomposition).  Conducts a short interview to clarify
  intent, formulates a prompt per docs/generate_3d_models.md, generates
  an image, removes its background, lifts it to 3D, and (optionally)
  decomposes into parts.  Force-evicts each pipeline's runner-side
  server between steps so no image / 3D model stays resident in VRAM
  longer than the request that needed it — leaves the cluster's GPU
  free for whatever the user does next.
argument-hint: [description] [out-dir] [auto] [evict] [parts]
allowed-tools:
  - Bash
  - Read
  - Edit
  - AskUserQuestion
---

# Skill: `/generate-3d`

Drive the full text-to-3D workflow described in
`./generate_3d_models.md` end-to-end from a single conversation
turn.  When the user invokes this skill, you become a co-author of
their 3D asset: interview them about the subject, write the prompt,
run the four-step (or five-step, with parts) pipeline, and show them
the result.

Before each pipeline step, the runner may have other models
resident (LLM sessions, previous pipeline models).  The api
auto-shuts the image / 3D server after each request via
``IMG_SERVER_AUTO_SHUTDOWN`` and ``IN_PROCESS_AUTO_UNLOAD``
(default ``true`` on the cluster), so eviction is often
unnecessary.  **Ask the user whether to evict** — they may have a
running LLM session they want to keep.  Only evict if they
confirm it's needed or if the pipeline returns an OOM / resource
error.

## Arguments

The user can pass arguments via skill invocation:

```
/generate-3d "ceramic mug" /tmp/model true false true
```

Arguments are injected through positional string substitutions:

| Position | Description | Default |
|----------|-------------|---------|
| `$0` | Short text description of the 3D object. Skips the Step 1 interview; use this description to formulate the prompt directly. | (ask user) |
| `$1` | Output directory for all generated files. Skips the output dir question in Step 1. | `/tmp/generate-3d-<descriptor>` |
| `$2` | `auto` — auto-accept all recommendations. Skips confirmation prompts. If `$0` is empty, still asks for the subject brief once, then auto-accepts everything else. | false |
| `$3` | `evict` — force-evict runner servers before each pipeline step. Implied by `auto` unless set to false. | false |
| `$4` | `parts` — run parts decomposition (Step 7) automatically after 3D generation. | false |

### Argument resolution

1. If all positions are empty, run the full interactive flow.
2. Provided arguments override the corresponding questions.
3. Boolean args (`$2`, `$3`, `$4`) accept `true`/`false`/`1`/`0`/`yes`/`no`. Empty means use interactive default.
4. If `auto` is true, `evict` is implied true unless `$3` is explicitly false.

## Required environment

Read these from the shell at the start of the skill.  If any are
missing, ask the user before continuing (unless `$2` is true, then
use defaults):

```bash
ANTHROPIC_BASE_URL  # e.g. http://192.168.0.122:9999
LLMMLL_AUTH_TOKEN   # admin-capable bearer token
```

If `ANTHROPIC_BASE_URL` isn't set, default to `http://192.168.0.122:9999` (the
on-cluster gateway).

`OUT_DIR` can come from `$1`, the Step 1 interview,
or a sensible default based on the subject description.

## Workflow

### Step 1 — Interview

Ask the user what they want to create AND where to output files.
You can ask both in a single `AskUserQuestion` call with two
questions, or sequentially if you prefer.

**Argument overrides:**
- If `$0` is set, skip the subject brief question and use it
  directly as the description.
- If `$1` is set, skip the output dir question and use it. Create
  the directory with `mkdir -p`.
- If `$2` is true and `$0` is set, skip all questions in this
  step. Derive `OUT_DIR` from `$1` or default to
  `/tmp/generate-3d-<short-descriptor>`.

**What to create** (interactive only): restate your understanding
in one or two sentences.  Iterate using `AskUserQuestion` (header
"Subject brief", options ``"Yes, that's it"`` / ``"Adjust: ..."`` /
``"Start over"``) until they confirm.  If `$2` is true but `$0` is
empty, ask the brief once and accept the first answer without
iteration.

What you're listening for, beyond the obvious "what's the object":

- **Material** — ceramic, metal, wood, plastic, glass.  Material
  changes what prompt vocabulary works and whether the subject is
  realistically 3D-extractable (transparent glass is XPart-hostile,
  per the docs).
- **Style** — photorealistic / stylized / cartoon.  Default to
  photorealistic product photography since it's what
  Hunyuan3D-2.1 was trained on and what downstream tools (Blender)
  expect.
- **Use case** — for game asset / for 3D printing / just to see.
  Drives whether the user cares about per-part decomposition (asset
  use cases benefit; "just to see" doesn't).

**Where to output files** (interactive only): ask via
`AskUserQuestion` (header "Output dir").  Suggest a default of
`/tmp/generate-3d-<short-descriptor>` (e.g.
`/tmp/generate-3d-ceramic-mug`) based on what they described.  The
user can accept the default or type a custom path.  Create the
directory with `mkdir -p` before the first pipeline step.

Save the brief and `OUT_DIR` for the remaining steps.

### Step 2 — Prompt formulation

Read `docs/generate_3d_models.md` (sections "Step 1 — txt2img" and
"Prompting strategy") for the canonical guidance.  Construct a
prompt that satisfies:

1. **One subject, centered** (no scenes, no environments)
2. **Soft studio lighting, light gray seamless background**
3. **Explicit material vocabulary** matching the user's brief
4. **Photorealistic product photography, sharp focus, high detail**

Avoid: scenes ("on a table"), strong shadows, transparent / glass
subjects (XPart can't mesh those), text or UI elements.

If `$2` is true: skip confirmation. Show the prompt to the user
as a one-liner and proceed immediately.
Otherwise: show the constructed prompt via `AskUserQuestion`
(header "Prompt", options ``"Run it"`` / ``"Refine: ..."`` /
``"Start over"``).  If they ask for refinement, edit the prompt and
re-ask.  Repeat until confirmed.

### Step 3 — Evict (optional) and run txt2img

**If `$2` is true:** skip the "Proceed?" confirmation. Proceed
directly to the eviction check below.

**If `$3` is true:** skip the eviction question and evict.

Otherwise, ask the user two questions via `AskUserQuestion`:

**1. Proceed?** (header "Generate image")

- ``"Yes, generate it"`` — proceed to eviction check below
- ``"Not yet, I want to adjust the prompt"`` — go back to step 2
- ``"Abort"`` — exit the skill

**2. Evict running servers?** (header "Evict servers") — only ask
if they confirmed generation.  They may have a running LLM
session they'd rather keep:

- ``"Yes, evict — I need the GPU"`` — run the eviction curl below
- ``"No, skip eviction"`` — skip straight to the txt2img script
- ``"I'm not sure"`` — recommend skipping; the auto-shutdown
  mechanism usually handles cleanup.  Only evict if they insist.

If evicting (either via `$3` or user confirmation), force-evict
any servers currently resident on the runner, then run the image
generation script inline with a `&&` after eviction to avoid immediately restarting the llm
session they just evicted.  

**IMPORTANT**: if `$3`, you MUST run this **exact** command:

```bash
curl -sS -X POST "$ANTHROPIC_BASE_URL/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $LLMMLL_AUTH_TOKEN" && \
  OUT_DIR="$OUT_DIR" \
  ./scripts/test_txt2img.sh "<prompt>"
```

else:

```bash
OUT_DIR="$OUT_DIR" \
  ./scripts/test_txt2img.sh "<OUT_DIR="$OUT_DIR" \
prompt>"
```

The script writes ``<OUT_DIR>/txt2img_<ts>.png`` and prints the
path on its last line.  Capture it.

### Step 4 — Show and confirm the image

Read the generated PNG with the `Read` tool so the user sees it
inline.

**If `$2` is true:** skip the confirmation. Show the image and
proceed to step 5 (background removal).

Otherwise ask whether to:

- ``"Continue to background removal"`` — proceed to step 5
- ``"Refine via img2img"`` — go to step 4b (edit step)
- ``"Regenerate from scratch"`` — go to step 2 (new prompt)
- ``"Abort"`` — exit the skill

### Step 4b — Optional img2img edit

Only if the user picks "Refine via img2img" (not available when
`$2` is true).  Ask them what to change.  Construct an edit prompt
following the ``./generate_3d_models.md`` rule "additive edits
are most reliable; full-surface changes are flaky".

Evict if `$3` is true or user confirms (see step 3 eviction
check).  Then run:

```bash
OUT_DIR="$OUT_DIR" \
  ./scripts/test_img2img.sh <last_image.png> "<edit prompt>" \
  qwen-image-edit-2511 0.75
```

Show the result and loop back to step 4's choice menu.

### Step 5 — Background removal

Run rembg:

```bash
OUT_DIR="$OUT_DIR" \
  ./scripts/test_rembg.sh <last_image.png>
```

The script writes ``<OUT_DIR>/rembg_<ts>_cutout.png``.  Read it to
show the user.

**If `$2` is true:** skip confirmation, proceed to step 6.
**Otherwise:** confirm it looks right; if not, go back to step 4
(usually you want to refine the input image first since rembg has
no tuning knobs).

### Step 6 — 3D generation

Evict if `$3` is true or user confirms (see step 3 eviction
check).  Run img2-3d:

```bash
OUT_DIR="$OUT_DIR" \
  ./scripts/test_img2-3d.sh <cutout.png>
```

The script writes ``<OUT_DIR>/<id>.glb`` and prints the path.
**Note: this takes ~3 minutes.**  Tell the user before kicking it
off.

Use ``open`` on macOS / ``xdg-open`` on Linux to launch the .glb
in the system viewer:

```bash
open "$OUT_DIR/<id>.glb"
```

### Step 7 — Optional parts decomposition

**If `$4` is true:** skip the question and proceed to
decomposition.

Otherwise, ask the user via `AskUserQuestion`:

- ``"Yes, decompose into parts"`` — proceed
- ``"No, the assembled mesh is enough"`` — go to summary

Evict if `$3` is true or user confirms (see step 3 eviction
check).  Run the parts pipeline with split mode enabled so each
part comes back as a standalone ``.glb`` (ready for Blender import
as separate objects):

```bash
OUT_DIR="$OUT_DIR" \
  ./scripts/test_img2-3d-parts.sh <mesh.glb> 512 42 1
```

The 4th positional arg ``1`` enables ``split=true``.  Outputs:

- ``<id>_decomposed.glb`` — assembled mesh with each part as a
  named primitive (drag-into-Blender gives separate objects)
- ``<id>_exploded.glb`` — same parts spatially separated for
  visualization
- ``<id>_part_NN.glb`` — one file per part, ready to import
  individually
- ``<id>_bbox.glb`` + ``<id>_gt_bbox.glb`` — segmentation debug

Open the exploded view (it's the most visually informative):

```bash
open "$OUT_DIR/<id>_exploded.glb"
```

Optionally open each per-part .glb too if the user wants to inspect
them individually.

### Step 8 — Final eviction + summary

Evict if `$3` is true or user confirms (see step 3 eviction
check).  One last eviction so nothing's left resident:

```bash
curl -sS -X POST "$ANTHROPIC_BASE_URL/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $LLMMLL_AUTH_TOKEN"
```

Print a summary listing every file produced, with their paths, in
this format:

```
✓ Generated 3D asset for "<one-line brief>"

  base:      $OUT_DIR/txt2img_*.png
  edited:    $OUT_DIR/img2img_*.png       (if step 4b ran)
  cutout:    $OUT_DIR/rembg_*_cutout.png
  mesh:      $OUT_DIR/<id>.glb
  parts:     $OUT_DIR/<id>_part_*.glb     (if step 7 ran)
  exploded:  $OUT_DIR/<id>_exploded.glb   (if step 7 ran)

Total wall-clock: <N> seconds
```

## Notes on the cluster auto-shutdown story

You're not the only mechanism keeping VRAM free:

- **sd-server** (txt2img, img2img) — the api's
  ``services/image_service.py`` calls ``shutdown_server`` after
  every request when ``IMG_SERVER_AUTO_SHUTDOWN=1`` (default on
  the cluster).  Each call boots a fresh sd-server.
- **In-process pipelines** (rembg, img23d, img23d_part) — the
  runner's ``InProcessPipeline.run`` calls ``self.unload()`` after
  every request when ``IN_PROCESS_AUTO_UNLOAD=1`` (default on the
  cluster).  Each call cold-loads the model again
  (~5-30 s depending on pipeline).
- **Your explicit ``/v1/runner/servers/evict-all`` calls** —
  optional, user-confirmed.  Ask before evicting; the user may
  have a running LLM session they want to keep.

Default to **skipping** eviction unless the user confirms or the
pipeline fails with an OOM / resource error.  The auto-shutdown
mechanisms above usually handle cleanup, so eviction is a
fallback, not a prerequisite.

## Troubleshooting

- **`503 Service Unavailable` on /v1/images/3d/parts** — usually
  means the runner image is missing one of the XPart deps.  Check
  the per-part export errors via `kubectl logs` for the runner pod.
- **Empty `.glb` outputs from img23d_part** — XPart's diffusion
  produced no valid SDF surfaces; rare with Hunyuan3D-2.1 inputs
  but possible.  Re-run with a different seed (5th positional) or
  go back and refine the input image.
- **The user wants to interrupt mid-pipeline** — pipeline steps
  are individually killable via Ctrl-C on the script; the api will
  return a 499.  Evict to free VRAM, then report what was completed.
