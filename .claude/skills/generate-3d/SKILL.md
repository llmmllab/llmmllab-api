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
allowed-tools:
  - Bash
  - Read
  - Edit
  - AskUserQuestion
---

# Skill: `/generate-3d`

Drive the full text-to-3D workflow described in
`docs/generate_3d_models.md` end-to-end from a single conversation
turn.  When the user invokes this skill, you become a co-author of
their 3D asset: interview them about the subject, write the prompt,
run the four-step (or five-step, with parts) pipeline, and show them
the result.

Before each pipeline step, **force-evict everything else on the
runner** so the active model has the full GPU to itself.  After
each pipeline step, the api already auto-shuts the image / 3D
server back down via the ``IMG_SERVER_AUTO_SHUTDOWN`` and
``IN_PROCESS_AUTO_UNLOAD`` env vars (set ``true`` by default on the
cluster), but you should still issue an explicit eviction before
the *next* step in case anything else (a Qwen LLM session on
another agent) spawned in the interim.  Belt-and-braces.

## Required environment

Read these from the shell at the start of the skill.  If any are
missing, ask the user before continuing:

```bash
API_BASE  # e.g. http://192.168.0.122:9999
API_KEY   # admin-capable bearer token
OUT_DIR   # local dir for downloads, e.g. /tmp/generate-3d-session-XXXX
```

If `API_BASE` isn't set, default to `http://192.168.0.122:9999` (the
on-cluster gateway) and confirm with the user.

## Workflow

### Step 1 — Interview

Ask the user what they want to create.  Restate your understanding
in one or two sentences.  Iterate using `AskUserQuestion` (header
"Subject brief", options ``"Yes, that's it"`` / ``"Adjust: ..."`` /
``"Start over"``) until they confirm.

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

Save the brief as a one-paragraph summary for the prompt step.

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

Show the constructed prompt to the user via `AskUserQuestion`
(header "Prompt", options ``"Run it"`` / ``"Refine: ..."`` /
``"Start over"``).  If they ask for refinement, edit the prompt and
re-ask.  Repeat until confirmed.

### Step 3 — Evict everything, run txt2img

Force-evict any image / 3D / LLM servers currently resident on the
runner:

```bash
curl -sS -X POST "$API_BASE/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $API_KEY"
```

Then run the image generation script.  Use the prompt from step 2
verbatim:

```bash
cd ~/workspace/llmmllab-api
OUT_DIR="$OUT_DIR" \
  ./scripts/test_txt2img.sh "<prompt>"
```

The script writes ``<OUT_DIR>/txt2img_<ts>.png`` and prints the
path on its last line.  Capture it.

### Step 4 — Show and confirm the image

Read the generated PNG with the `Read` tool so the user sees it
inline.  Ask whether to:

- ``"Continue to background removal"`` — proceed to step 5
- ``"Refine via img2img"`` — go to step 4b (edit step)
- ``"Regenerate from scratch"`` — go to step 2 (new prompt)
- ``"Abort"`` — exit the skill

### Step 4b — Optional img2img edit

Only if the user picks "Refine via img2img".  Ask them what to
change.  Construct an edit prompt following the
``docs/generate_3d_models.md`` rule "additive edits are most
reliable; full-surface changes are flaky".

Force-evict everything (the txt2img sd-server has already
auto-shut, but be safe).  Then run:

```bash
OUT_DIR="$OUT_DIR" \
  ./scripts/test_img2img.sh <last_image.png> "<edit prompt>" \
  qwen-image-edit-2511 0.75
```

Show the result and loop back to step 4's choice menu.

### Step 5 — Background removal

Evict everything.  Run rembg:

```bash
OUT_DIR="$OUT_DIR" \
  ./scripts/test_rembg.sh <last_image.png>
```

The script writes ``<OUT_DIR>/rembg_<ts>_cutout.png``.  Read it to
show the user.  Confirm it looks right; if not, go back to step 4
(usually you want to refine the input image first since rembg has
no tuning knobs).

### Step 6 — 3D generation

Evict everything.  Run img2-3d:

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

Ask the user via `AskUserQuestion`:

- ``"Yes, decompose into parts"`` — proceed
- ``"No, the assembled mesh is enough"`` — go to summary

If yes: evict everything, run the parts pipeline with split mode
enabled so each part comes back as a standalone ``.glb`` (ready for
Blender import as separate objects):

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

One last eviction so nothing's left resident:

```bash
curl -sS -X POST "$API_BASE/v1/runner/servers/evict-all" \
    -H "Authorization: Bearer $API_KEY"
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
  belt-and-braces in case another agent started something in the
  meantime.

You should still issue the eviction explicitly between steps in
this skill since the user may have parallel work happening that
spawned a Qwen LLM or other model on the cluster.  The cost of
eviction is near-zero (~1 RTT to the api) compared to the seconds
of a single image gen.

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
