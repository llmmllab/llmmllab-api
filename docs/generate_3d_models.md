# Generating 3D models from text — end-to-end pipeline

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
./scripts/test_txt2img.sh \
  "a single glossy red ceramic mug, centered, soft studio lighting, light gray seamless background, photorealistic product photography"
ln -sf $OUT_DIR/txt2img_*.png $OUT_DIR/step1_base.png

# 2. Edit it — add steam (additive edits are reliable)
./scripts/test_img2img.sh $OUT_DIR/step1_base.png \
  "add visible steam rising from the top of the mug. Wispy white steam curls drifting upward. Keep the mug, handle, lighting, and background exactly the same." \
  qwen-image-edit-2511 0.75
ln -sf $OUT_DIR/img2img_*.png $OUT_DIR/step2_edited.png

# 3. Remove the background
./scripts/test_rembg.sh $OUT_DIR/step2_edited.png
ln -sf $OUT_DIR/rembg_*_cutout.png $OUT_DIR/step3_cutout.png

# 4. Generate the 3D mesh from the cutout
./scripts/test_img2-3d.sh $OUT_DIR/step3_cutout.png
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
./scripts/test_txt2img.sh "$PROMPT"
ln -sf "$(ls -1t "$OUT_DIR"/txt2img_*.png | head -1)" "$OUT_DIR/step1.png"

if [[ -n "$EDIT_PROMPT" ]]; then
    echo "=== 2/4 img2img ==="
    ./scripts/test_img2img.sh "$OUT_DIR/step1.png" "$EDIT_PROMPT"
    ln -sf "$(ls -1t "$OUT_DIR"/img2img_*.png | head -1)" "$OUT_DIR/step2.png"
else
    ln -sf "$OUT_DIR/step1.png" "$OUT_DIR/step2.png"
fi

echo "=== 3/4 rembg ==="
./scripts/test_rembg.sh "$OUT_DIR/step2.png"
ln -sf "$(ls -1t "$OUT_DIR"/rembg_*_cutout.png | head -1)" "$OUT_DIR/step3.png"

echo "=== 4/4 img23d (this takes ~3 min) ==="
./scripts/test_img2-3d.sh "$OUT_DIR/step3.png"
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

## See also

- `scripts/README.md` — script-level docs for the individual curl
  helpers.
- `CLAUDE.md` — overall architecture; the "Image / 3D endpoints" table
  is the source of truth for endpoint shapes.
- llmmllab-runner `.models.yaml` / `.models.small.yaml` — declares
  which model each runner serves. Editing these is how you re-point a
  pipeline.
