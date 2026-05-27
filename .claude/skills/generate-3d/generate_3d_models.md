# Generating 3D models from text — end-to-end pipeline

## Contents

- [Quick start: validated end-to-end run](#quick-start-validated-end-to-end-run-2026-05-24)
- [Prompting strategy](#prompting-strategy) — per-step (txt2img, img2img, rembg, img23d)
- [Automated pipelining](#automated-pipelining) — shell / Python / future server-side
- [Optional 5th step: part decomposition](#optional-5th-step-part-decomposition-hunyuan3d-part)
  - CLI, endpoint shape, supported input meshes
  - [Parameter tuning](#parameter-tuning) — all env-var knobs for every script
    - [txt2img / img2img knobs + multi-image edits](#txt2imgsh--img2imgsh-image-generation)
    - [img2-3d knobs](#img2-3dsh-image--3d-mesh)
    - [mesh2parts knobs](#mesh2partssh-mesh--per-part-meshes)
    - [AABB / AABB_FILE — caller-driven decomposition](#aabb--aabb_file--caller-driven-part-decomposition)
- [Operational notes](#operational-notes)
- [Troubleshooting](#troubleshooting)
- [See also](#see-also)

A four-step pipeline that takes a text prompt and produces a textured 3D
mesh suitable for use in Blender, Unity, three.js, or any glTF-capable
viewer. Each step is a single HTTP request to llmmllab-api and is
independently scriptable.

```
text prompt
   │
   ▼  Step 1 — txt2img (qwen-image-2512)
        runner: llmmllab-runner (lsnode-3, big GPUs)
        ~40 s, returns 1024×1024 PNG
   │
   ▼  Step 2 — img2img edit (qwen-image-edit-2511)
        runner: llmmllab-runner (lsnode-3)
        ~60 s, refines composition / adds details
   │
   ▼  Step 3 — rembg (briaai/RMBG-2.0, BiRefNet)
        runner: llmmllab-runner-small (lsnode-4)
        ~1.5 s on GPU, returns alpha-cut PNG
   │
   ▼  Step 4 — img23d (Hunyuan3D-2.1, shape-only)
        runner: llmmllab-runner (lsnode-3)
        ~3 min, returns .glb mesh
   │
   ▼
.glb file — ready for any 3D tool
```

Endpoint routing happens automatically — the api reads each runner's
`/v1/models` (which mirrors its `.models.yaml`) and routes pipeline
requests to whichever runner advertises the right model id. The split
between the big and small runner is yaml-driven, not code-driven.

## Quick start: validated end-to-end run (2026-05-24)

```bash
export API_BASE=http://192.168.0.122:9999
export API_KEY=<your-api-key>
export OUT_DIR=/tmp/pipeline_demo
mkdir -p $OUT_DIR

# 1. Generate the base image
./scripts/txt2img.sh \
  "a single glossy red ceramic mug, centered, soft studio lighting, light gray seamless background, photorealistic product photography"
ln -sf $OUT_DIR/txt2img_*.png $OUT_DIR/step1_base.png

# 2. Edit it — add steam (additive edits are reliable)
./scripts/img2img.sh $OUT_DIR/step1_base.png \
  "add visible steam rising from the top of the mug. Wispy white steam curls drifting upward. Keep the mug, handle, lighting, and background exactly the same." \
  qwen-image-edit-2511 0.75
ln -sf $OUT_DIR/img2img_*.png $OUT_DIR/step2_edited.png

# 3. Remove the background
./scripts/rembg.sh $OUT_DIR/step2_edited.png
ln -sf $OUT_DIR/rembg_*_cutout.png $OUT_DIR/step3_cutout.png

# 4. Generate the 3D mesh from the cutout
./scripts/img2-3d.sh $OUT_DIR/step3_cutout.png
ln -sf $OUT_DIR/*.glb $OUT_DIR/step4_mesh.glb

open $OUT_DIR/step4_mesh.glb   # macOS Quick Look renders .glb natively
```

Wall-clock for the full chain in the last validated run: **~4 min**
(40 s + 60 s + 1.5 s + 195 s).

## Prompting strategy

Each step has different prompt-following characteristics. What works in
one model can fight the next one.

### Step 1 — txt2img (qwen-image-2512)

Goal: produce an image that's easy for Hunyuan3D to mesh later. That
means **one subject, centered, on a clean background**. Hunyuan3D
infers depth from a single view, so cluttered scenes or partial
occlusions produce confused geometry.

**What works:**

- "a single \<object\>, centered, soft studio lighting, light gray
  seamless background, photorealistic product photography, sharp focus"
- Single-subject prompts. Avoid "two ...", scenes, environments.
- Explicit material vocabulary: *ceramic*, *brushed steel*, *polished
  wood*, *matte plastic*. Qwen-Image-2512 has strong material priors —
  use them.
- Light gray (#cccccc-ish) or white seamless background. Pure white can
  blow out highlights on light-colored subjects; pure black hides dark
  surfaces. Light gray is the safe default.

**What doesn't:**

- "On a table in a kitchen ..." — table geometry leaks into the 3D mesh
  in step 4. Hunyuan3D will model the table.
- Strong shadows. The cast shadow becomes a flat plane in 3D.
- Glass / transparent subjects. RMBG-2.0 cuts them out as solid
  silhouettes; Hunyuan3D meshes them as opaque.

**Defaults baked into the runner** (Qwen-Image-2512 tuning):
40 inference steps, `cfg_scale=2.5`, sampler `euler`, 1024×1024.
Override per-call via `.models.yaml` if you change models.

### Step 2 — img2img edit (qwen-image-edit-2511)

This is the trickiest stage. Qwen-Image-Edit-2511 is conservative —
it has strong priors toward preserving the input image, which is
useful for keeping geometry stable but punishing when you ask for big
changes.

**Edit types ranked by reliability** (validated empirically):

| Edit | Reliability | Notes |
|------|------------|-------|
| Additive ("add steam", "add petals scattered around") | High | Adds the new element without disturbing existing geometry. Best choice for pipeline edits. |
| Local material swap ("the rim is now gold") | Medium | Works for localized regions but can bleed into adjacent surfaces. |
| Pose / orientation ("rotate 45°") | Medium-Low | Often interpreted as composition shift rather than geometric rotation. |
| Full-surface color change ("the mug is now blue") | Low | Often partial — only some surfaces change. Use cfg_scale ≥ 4.0 + denoising_strength ≥ 0.85 and accept ~60% success rate. |
| "Remove X" / "make it transparent" | Very low | Don't try. Use rembg (step 3) for background removal. |
| Style transfer ("oil painting style") | Medium | Works but degrades fidelity for step 4. |

**Prompt patterns that work:**

- `"add <thing>. <details>. Keep the <subject>, lighting, and background exactly the same."` — explicit preservation clause anchors the rest of the image.
- `"<edit>. Preserve the exact shape, composition, lighting, and background."` — same idea, terser.
- Use `denoising_strength` around `0.75` for additive edits, `0.85` for material swaps. `1.0` ignores the input entirely and degrades to plain txt2img — never useful here.

**Prompt patterns that don't:**

- Negation-only prompts: `"remove the steam"`, `"without the handle"`. The model doesn't have reliable subtraction semantics for img2img. If you need removal, generate a fresh image without the unwanted element.
- Multiple simultaneous edits: `"make it blue and add a saucer"`. Split into two passes.

**Hard-won discovery** (see `services/image_service.py::edit_image`): sd-server's img2img reads the source image from two different body fields. `init_images` populates the legacy noise-img2img path; `extra_images` populates `ref_images` for the QwenImageEditPlusPipeline (which is what gives Qwen-Image-Edit its instruction-following behavior). The api sends the source in **both** so the same endpoint works whether or not the underlying model uses the edit-aware path. If you're hitting sd-server directly, send both.

### Step 3 — rembg (briaai/RMBG-2.0)

No prompt. Pure segmentation.

- Subject is whatever the model identifies as the salient foreground.
- Works extraordinarily well on product-photography-style images
  (single subject, neutral background) — which is exactly what step 1
  produced.
- Outputs both a grayscale alpha mask (`mask_b64`) and an
  alpha-composited RGBA PNG (`transparent_b64`).
- The cutout is also served as a hosted file at
  `GET /v1/images/remove-bg/<id>.png` so you can skip the b64
  round-trip when chaining.

**Knobs:**

- `mask_only=true` if you want just the mask (e.g. for compositing
  elsewhere). Skips the RGBA composite.
- `size` (default 1024) — square edge for the model's internal resize.
  The mask is always upsampled back to source resolution. Higher
  `size` improves fine-detail edges (hair, fur) at the cost of
  inference time. Default is usually fine for product photography.

### Step 4 — img23d (Hunyuan3D-2.1, shape-only path)

The cutout from step 3 is the conditioning image. Hunyuan3D infers a
3D mesh from a single 2D view — there is no prompt.

**Inputs that produce clean meshes:**

- Single subject, alpha-cut to transparent background (which is
  exactly what rembg produces).
- Subject roughly fills the frame — don't have the cutout occupy <30%
  of the canvas.
- Avoid wispy or translucent elements in the cutout (steam, smoke,
  glass, transparent water). They become weird tendril geometry in the
  mesh. If step 2 added steam (great for the 2D image), consider
  re-rembg-ing or hand-editing the cutout to remove it before step 4.
- Front-facing or 3/4 view. Pure side or top-down views give Hunyuan3D
  much less geometric information.

**Parameters** (defaults baked into the api router):

- `ss_steps=12`, `slat_steps=12` — sparse-structure + SLAT sampler
  steps. Higher = finer detail, longer runtime. 12 is the sweet spot.
- `ss_cfg_strength=7.5`, `slat_cfg_strength=3.0` — classifier-free
  guidance strengths. Defaults are tuned; the SLAT pass uses lower
  cfg because it operates on already-structured latents.
- `formats=["mesh"]` — Hunyuan3D-2.1's shape-only path emits a
  textured `.glb` mesh. `formats=["gaussian"]` would request 3D
  gaussians but **this backbone doesn't produce them** — the response
  field will be `null`. Stable Fast 3D or TRELLIS would, but they're
  not currently deployed.

Wall-clock: ~3 min per image on the big runner. The pipeline is
synchronous; clients should set HTTP timeouts to at least 5 min.

## Automated pipelining

There are three reasonable ways to chain the four steps.

### Option A: shell script (simplest)

The four bash scripts under `scripts/` are designed to compose. Each
writes a timestamped JSON response next to a decoded PNG/GLB. Wrap them
in a parent script:

```bash
#!/usr/bin/env bash
# pipeline.sh — text -> 3D
set -euo pipefail

PROMPT="${1:?usage: pipeline.sh '<prompt>' [edit_prompt]}"
EDIT_PROMPT="${2:-}"
export API_BASE="${API_BASE:-http://192.168.0.122:9999}"
export API_KEY="${API_KEY:?API_KEY env var required}"
export OUT_DIR="${OUT_DIR:-/tmp/pipeline_$$}"
mkdir -p "$OUT_DIR"

echo "=== 1/4 txt2img ==="
./scripts/txt2img.sh "$PROMPT"
ln -sf "$(ls -1t "$OUT_DIR"/txt2img_*.png | head -1)" "$OUT_DIR/step1.png"

if [[ -n "$EDIT_PROMPT" ]]; then
    echo "=== 2/4 img2img ==="
    ./scripts/img2img.sh "$OUT_DIR/step1.png" "$EDIT_PROMPT"
    ln -sf "$(ls -1t "$OUT_DIR"/img2img_*.png | head -1)" "$OUT_DIR/step2.png"
else
    ln -sf "$OUT_DIR/step1.png" "$OUT_DIR/step2.png"
fi

echo "=== 3/4 rembg ==="
./scripts/rembg.sh "$OUT_DIR/step2.png"
ln -sf "$(ls -1t "$OUT_DIR"/rembg_*_cutout.png | head -1)" "$OUT_DIR/step3.png"

echo "=== 4/4 img23d (this takes ~3 min) ==="
./scripts/img2-3d.sh "$OUT_DIR/step3.png"
ln -sf "$(ls -1t "$OUT_DIR"/*.glb | head -1)" "$OUT_DIR/step4.glb"

echo
echo "✓ pipeline complete"
echo "  base:   $OUT_DIR/step1.png"
echo "  edited: $OUT_DIR/step2.png"
echo "  cutout: $OUT_DIR/step3.png"
echo "  mesh:   $OUT_DIR/step4.glb"
```

### Option B: single Python script (better error handling)

```python
import base64, httpx, pathlib, time

API = "http://192.168.0.122:9999"
KEY = "..."
H = {"Authorization": f"Bearer {KEY}"}
OUT = pathlib.Path("/tmp/pipeline"); OUT.mkdir(exist_ok=True)

def b64(p):  return base64.b64encode(p.read_bytes()).decode()
def save(p, b): p.write_bytes(base64.b64decode(b))

with httpx.Client(base_url=API, headers=H, timeout=900) as c:
    # 1 - txt2img
    r = c.post("/v1/images/generations", json={
        "prompt": "a single porcelain teapot, centered, soft studio lighting, light gray background, photorealistic",
        "model": "qwen-image-2512", "size": "1024x1024",
    }).json()
    save(OUT/"step1.png", r["data"][0]["b64_json"])

    # 2 - edit (optional)
    r = c.post("/v1/images/edits", json={
        "prompt": "add a delicate floral pattern in cobalt blue. Keep the teapot, lighting, background unchanged.",
        "image": b64(OUT/"step1.png"),
        "model": "qwen-image-edit-2511", "denoising_strength": 0.75,
    }).json()
    save(OUT/"step2.png", r["data"][0]["b64_json"])

    # 3 - rembg
    r = c.post("/v1/images/remove-bg", json={
        "image": b64(OUT/"step2.png"),
    }).json()
    save(OUT/"step3.png", r["transparent_b64"])

    # 4 - img23d (long-running)
    r = c.post("/v1/images/3d", json={
        "image_b64": b64(OUT/"step3.png"),
        "formats": ["mesh"],
    }, timeout=1200).json()
    mesh = c.get(r["mesh_url"], timeout=120).content
    (OUT/"step4.glb").write_bytes(mesh)

print(f"done: {OUT}")
```

### Option C: server-side composition (future)

These four endpoints already share infrastructure
(`services/image_service.py`, priority_queue, runner endpoint
selection). A `/v1/pipelines/text-to-3d` endpoint that runs all four
steps server-side and streams progress events to the client would
remove four round-trips of base64 over HTTP. Not currently implemented
— file an issue if you want it.

## Optional 5th step: part decomposition (Hunyuan3D-Part)

After step 4 produces a holistic `.glb`, an optional 5th step takes
that mesh and decomposes it into semantically meaningful parts using
**Tencent's Hunyuan3D-Part** (P3-SAM + XPart):

```
.glb (holistic)
   │
   ▼  Step 5 — mesh-to-parts (hunyuan3d-part)
        runner: llmmllab-runner (lsnode-3, big GPUs)
        ~2-5 min, returns 4 .glb files
   │
   ▼
{decomposed, exploded, bbox, gt_bbox}.glb
```

Two stages inside one pipeline call:
1. **P3-SAM** predicts part bounding boxes from the input mesh.
2. **XPart** regenerates each detected part as standalone high-fidelity
   geometry and emits both an assembled view (parts joined) and an
   exploded view (parts spatially separated).

### CLI

```bash
./scripts/mesh2parts.sh /tmp/pipeline_demo/step4_mesh.glb
ln -sf $OUT_DIR/*_decomposed.glb $OUT_DIR/step5_decomposed.glb
ln -sf $OUT_DIR/*_exploded.glb   $OUT_DIR/step5_exploded.glb
open $OUT_DIR/step5_exploded.glb   # the exploded view is the headline output
```

### Endpoint shape

`POST /v1/images/3d/parts` accepts base64-encoded mesh bytes:

```json
{
  "mesh_b64": "<base64 .glb>",
  "octree_resolution": 512,
  "seed": 42
}
```

Returns four download URLs (`mesh_url` / `exploded_url` / `bbox_url` /
`gt_bbox_url`), each pointing at `GET /v1/images/3d/parts/{filename}`.

### What input meshes work

XPart was trained on AI-generated and scanned meshes. The upstream
README is explicit: **"For X-Part, we recommend using scanned or
AI-generated meshes (e.g., from Hunyuan3D V2.5 or V3.0) as input."**
Our V2.1 outputs work but aren't the sweet spot.

| Input source | Reliability |
|--------------|-------------|
| `img2-3d.sh` output (Hunyuan3D-2.1) | Good — the chained workflow |
| Scanned meshes (photogrammetry, lidar) | Good |
| Hand-modeled CAD geometry | Mixed — P3-SAM's part priors may produce odd segmentations on overly clean procedural geometry |
| Heavily edited / Blender-built meshes | Mixed — same reason |

### Why it's not a step in the default pipeline

- **Cost**: adds 2-5 minutes on top of the existing ~4 min pipeline.
- **Opt-in by nature**: many use cases want one mesh, not a part-decomposed
  scene graph. Rigging, retopo, and texture authoring per part want it;
  rendering and quick previews don't.
- **Output shape**: four files instead of one. Clients have to decide
  which they want.

### Parameter tuning

Every script accepts env vars that map directly to per-request body
fields the api forwards to the runner pipeline.  Unset → the script
omits the field → api falls through to the per-model defaults in the
runner's `.models.yaml`.  Override per-call as needed.

#### txt2img.sh / img2img.sh (image generation)

| Env | Body field | Yaml default | When to bump |
|---|---|---|---|
| `NEGATIVE_PROMPT` | `negative_prompt` | (empty) | **Almost always set** for object-specific gens.  E.g. `"G-clamp, vise, multiple objects"` when prompting a C-clamp.  Models confuse similar classes constantly. |
| `CFG_SCALE` | `cfg_scale` | 4.0 (qwen-image) | Higher = sticks closer to the prompt.  5-7 for stubborn-geometry industrial objects; 8+ if the model still resists.  Too high washes out aesthetics. |
| `STEPS` | `steps` | 50 (qwen-image) | More steps = finer detail at linear cost.  60-80 for fine industrial / mechanical scenes. |
| `SAMPLER` | `sampler_name` | `dpm++_2m` (qwen-image) | `dpm++_2m` is sharpest on geometry.  `euler` is fastest.  Also: `dpm++_sde`, `unipc`, `dpmpp_2m_sde`. |
| `SEED` | `seed` | -1 (random) | Set an int for reproducible regenerations of the same prompt. |
| `EXTRA_IMAGES` (img2img only) | `image[1:]` | (none) | Comma-separated paths to **additional reference images**.  The primary image (positional arg 1) is the one being edited; extras are visual context Qwen-Image-Edit-2511 uses as conditioning (style donor, subject reference, palette, etc.).  Up to ~16.  See "Multi-image edits" below. |

img2img also accepts a 4th positional arg `denoising_strength` (0-1):
0.0 reproduces the input, 1.0 ignores it.  0.65-0.8 is the
prompt-guided-edit sweet spot.

##### Multi-image edits (img2img.sh)

Qwen-Image-Edit-2511 conditions on **multiple** reference images
when more than one is supplied.  The first image is the canvas being
edited; subsequent images steer style, subject identity, palette, or
composition.  The api exposes this via the polymorphic ``image``
field on ``POST /v1/images/edits`` — a single base64 string for the
classic single-reference case, or a JSON array
``[primary_b64, ref1_b64, ref2_b64, ...]`` for multi-reference.

```bash
# Edit photo.png using style.png and subject.png as additional refs
EXTRA_IMAGES=./style.png,./subject.png \
  ./scripts/img2img.sh ./photo.png \
    "blend the subject of photo.png with the style of style.png; keep the pose from subject.png"
```

When to reach for it:
- **Style transfer with anchor** — prompt + style-donor image is more
  reliable than text-only style prompts, which Qwen-Image-Edit handles
  poorly.
- **Identity preservation** — include a clean reference shot of the
  subject when editing a low-quality original.
- **Palette matching** — drop in a swatch image and prompt "match
  the color palette of <ref>".
- **Compositional borrowing** — borrow framing or lighting from one
  image while keeping the subject of another.

The denoising_strength still applies to the primary image only.

#### img2-3d.sh (image → 3D mesh)

| Env | Body field | Yaml default | When to bump |
|---|---|---|---|
| `SEED` | `seed` | 42 | Lock for reproducible meshes. |
| `STEPS` | `num_inference_steps` | 50 | 75-100 for finer geometry.  Linear cost in time. |
| `GUIDANCE_SCALE` | `guidance_scale` | 7.5 | Lower (4-7) for cleaner geometry; higher (8-10) chases the image harder but introduces spikes / floaters. |
| `OCTREE_RESOLUTION` | `octree_resolution` | 384 | 256 = fast iteration, 512 = high-fidelity.  Quadratic memory cost. |
| `MC_LEVEL` | `mc_level` | -1/512 | Marching-cubes iso-level.  More negative thickens output; positive thins and risks holes.  Tweak by ±0.001 increments. |
| `BOX_V` | `box_v` | 1.01 | SDF bounding-box scale.  Rarely needs tuning. |
| `NUM_CHUNKS` | `num_chunks` | 8000 | Bump to 400000 if you have VRAM headroom for faster eval. |

#### mesh2parts.sh (mesh → per-part meshes)

| Env | Body field | Default | When to set |
|---|---|---|---|
| `STEPS` | `num_inference_steps` | 50 | Higher → finer per-part geometry. |
| `GUIDANCE_SCALE` | `guidance_scale` | (XPart default) | Bump if you see merged or over-smoothed parts. |
| `MAX_PARTS` | `max_parts` | 0 (no cap) | P3-SAM can detect 20-50+ parts on dense meshes, OOMing the conditioner (~7-8 GB activation per K=25).  Set 8-15 for safety.  **Ignored when `AABB`/`AABB_FILE` is set.** |
| `AABB` | `aabb` (inline JSON) | (auto-segment) | **Caller-specified bounding boxes.**  Bypasses P3-SAM auto-segmentation entirely — XPart uses your boxes directly.  Shape `[K, 2, 3]` (K parts × min/max corners × xyz).  Mesh coords are normalised to [-1, 1] around the centroid.  See examples below. |
| `AABB_FILE` | `aabb` (from file) | (auto-segment) | Same as `AABB` but read from a JSON file.  Use when the box list is too long for env / argv.  `AABB` wins if both are set. |

#### `AABB` / `AABB_FILE` — caller-driven part decomposition

When you already know which regions of the mesh should be separate
parts, P3-SAM's auto-segmentation is just a guess that you have to
clean up.  Feeding `aabb` directly skips that step and forces XPart
to decompose along exactly the boundaries you specify.

Common use cases:
- **CAD geometry** — you designed the parts, you know where they are
- **Hand-marked reference** — Blender bounding-box selections,
  exported to JSON
- **Repeatable workflows** — same shape decomposed the same way
  every time
- **Recovery from bad auto-seg** — P3-SAM merged two parts that
  should be separate, or split one part into nonsense.  Re-run with
  explicit boxes.

Inline JSON form:

```bash
AABB='[[[-1,-1,-1],[-0.2,1,1]], [[-0.2,-1,-1],[0.2,1,1]], [[0.2,-1,-1],[1,1,1]]]' \
  ./scripts/mesh2parts.sh /tmp/mesh.glb 256
# → mesh split into 3 parts along the X axis
```

File form (for larger box lists):

```bash
cat > /tmp/parts.json <<JSON
[
  [[-1.0, -1.0, -1.0], [-0.2,  1.0,  1.0]],
  [[-0.2, -1.0, -1.0], [ 0.2,  1.0,  1.0]],
  [[ 0.2, -1.0, -1.0], [ 1.0,  1.0,  1.0]]
]
JSON
AABB_FILE=/tmp/parts.json ./scripts/mesh2parts.sh /tmp/mesh.glb 256
```

The shape is `[K, 2, 3]`: a list of K parts, each part is
`[min-corner, max-corner]`, each corner is `[x, y, z]`.  The mesh
is internally normalised to a unit cube around its centroid before
P3-SAM operates, so coordinates in the range `[-1, 1]` typically
work.  If the model produces wildly off output, check whether your
input mesh is centred / scaled in a non-standard space.

### Why some legacy fields are gone

The earlier api signature included TRELLIS-era params (`ss_steps`,
`slat_steps`, `ss_cfg_strength`, `slat_cfg_strength`) — those were
left over from a previous backbone and the Hunyuan3D-2.1 pipeline
ignored them entirely (no matching kwargs in `_pick`).  They've
been removed; the native fields above are the actual knobs.

### Notes on the four outputs

- **`decomposed.glb`** — usually the one you actually want. Parts are
  joined back together but each is a separate primitive, so glTF
  viewers and downstream tools (Blender, Three.js, Unity glTF
  importer) can address them individually.
- **`exploded.glb`** — best demo output. Parts spatially separated
  for visualisation; not useful as final geometry but excellent for
  validating the segmentation.
- **`bbox.glb`** — just the P3-SAM bbox wireframes. Debug-only.
- **`gt_bbox.glb`** — input mesh with bboxes overlaid. Debug-only;
  shows what P3-SAM saw before XPart regenerated.

## Operational notes

- **Routing is yaml-driven.** Each runner reads its own
  `.models.yaml`; the api builds a `pipeline_name → endpoints` index
  from each runner's `/v1/models`. To move a pipeline between runners,
  edit one yaml. No code change, no env var.
- **Image requests share the priority queue with chat.** All four
  image endpoints enqueue at `Priority.MEDIUM` (same as chat). When
  capacity is tight, chat and image requests age fairly against each
  other. Set `PRIORITY_QUEUE_ENABLED=false` to opt out.
- **No artifact garbage collection yet.** rembg cutout PNGs and
  Hunyuan3D `.glb` files persist on the runner pod's
  `/data/sd-out/{rembg,3d}/`. Run
  `scripts/registry_cleanup.py` (in the runner repo) periodically, or
  add a sidecar CronJob.
- **HF gating.** RMBG-2.0 is a gated repo on HuggingFace. Weights are
  pre-downloaded to `/models/rmbg-2.0` on lsnode-4 out-of-band; the
  pod doesn't need an HF token at runtime. Same pattern for any future
  gated model.

## Troubleshooting

- **Step 1: blurry / "soft" image.** `qwen-image-2512` defaults to
  cfg_scale 2.5 which is intentionally low. Bump to 4.0 in the body
  if you want sharper output; values >6 produce over-saturated artifacts.
- **Step 2: edit ignored or partial.** Qwen-Image-Edit-2511 is
  conservative. See the "ranked by reliability" table above. Try a
  different edit pattern; for full-surface changes accept partial
  success.
- **Step 3: cutout has halo / fringing.** Increase `size` to 2048;
  RMBG resizes internally and finer input means finer mask edges.
  Default 1024 is fine for product photography.
- **Step 4: mesh has weird tendrils.** Wispy elements in the cutout
  (steam, smoke, hair, glass) confuse Hunyuan3D's depth inference.
  Regenerate the base image without them, or use an image editor to
  clean the cutout before step 4.
- **Step 4: 503 with "TRELLIS not installed".** Hunyuan3D's CUDA
  extensions (custom_rasterizer, differentiable_renderer) didn't
  build in the runtime image. Rebuild the runner image; the relevant
  Dockerfile stages are tagged `hunyuan-builder`.
- **End-to-end too slow.** Step 4 dominates (~3 min). Step 1 takes
  ~40 s. If iterating on prompts, skip step 4 until you're happy with
  the cutout from step 3.
- **Step 5: 503 with "Hunyuan3D-Part dependencies are missing".**
  XPart's `spconv-cu124` + `torch_cluster` / `torch_scatter` wheels
  failed to install in the runtime image, or the `chamfer3D` CUDA
  extension didn't build. Rebuild — the Dockerfile pulls them into
  the `hunyuan-builder` stage. If `chamfer3D` fails on a specific
  CUDA arch, set `TORCH_CUDA_ARCH_LIST` to include your GPU's compute
  capability.
- **Step 5: parts segmentation looks wrong.** P3-SAM was trained on
  AI-generated and scanned meshes. Hand-modeled CAD geometry (or
  meshes with degenerate triangles / non-manifold edges) confuses
  it. Either pre-clean the mesh (`trimesh.repair`, MeshLab) or
  regenerate via `img2-3d.sh` first.
- **Step 5: only one or two parts detected.** P3-SAM's confidence
  threshold is baked in. If your subject is geometrically uniform
  (e.g., a smooth sphere), there genuinely aren't part-like features
  to detect — XPart will return one large part. Pick subjects with
  obvious structural divisions (chair: seat + legs + back; teapot:
  body + handle + spout + lid).

## See also

- `scripts/README.md` — script-level docs for the individual curl
  helpers.
- `CLAUDE.md` — overall architecture; the "Image / 3D endpoints" table
  is the source of truth for endpoint shapes.
- llmmllab-runner `.models.yaml` / `.models.small.yaml` — declares
  which model each runner serves. Editing these is how you re-point a
  pipeline.
