# Image API test scripts

Five curl-based shell scripts for exercising the image + 3D endpoints
from the command line. No Python deps — just `bash`, `curl`, and `jq`.

> The actual script bodies live under
> `.claude/skills/generate-3d/scripts/` (so the `/generate-3d` skill
> and ad-hoc CLI runs share one canonical copy).  The files under
> `scripts/` are symlinks that drop the `test_` prefix — invoke
> either path; they're literally the same files.

| Script | Endpoint | Backend | Output |
|--------|----------|---------|--------|
| [`txt2img.sh`](#txt2imgsh) | `POST /v1/images/generations` | stable-diffusion.cpp `sd-server` | base64 PNG inline in response |
| [`img2img.sh`](#img2imgsh) | `POST /v1/images/edits` | stable-diffusion.cpp `sd-server` (img2img) | base64 PNG inline in response |
| [`rembg.sh`](#rembgsh) | `POST /v1/images/remove-bg` + `GET /v1/images/remove-bg/{file}` | briaai/RMBG-2.0 in-process pipeline | alpha mask PNG + transparent cutout PNG |
| [`img2-3d.sh`](#img2-3dsh) | `POST /v1/images/3d` + `GET /v1/images/3d/{file}` | Hunyuan3D-2.1 in-process pipeline | `.glb` mesh + `.ply` gaussian, streamed back through api |
| [`mesh2parts.sh`](#mesh2partssh) | `POST /v1/images/3d/parts` + `GET /v1/images/3d/parts/{file}` | Hunyuan3D-Part (P3-SAM + XPart) in-process pipeline | per-part `.glb` files + decomposed/exploded/bbox/gt_bbox views |

> `runner_shutdown.sh` (ops tool to free VRAM by force-evicting runner
> servers) lives in the **llmmllab-runner** repo's `scripts/` directory.
> It uses the same `API_BASE` + `API_KEY` env vars as the scripts here,
> so an admin token reaches it without any port-forward.

## Common configuration

All scripts honour the same connectivity env vars:

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_BASE` | `http://localhost:8000` | Base URL of llmmllab-api |
| `API_KEY` | *(unset)* | If set, sent as `Authorization: Bearer ...` |
| `OUT_DIR` | `./out` | Where decoded images / artefacts / raw JSON responses land |

Each invocation writes:

- A `<endpoint>_<unix-ts>.json` file with the raw response, useful for
  debugging or replaying.
- A decoded `.png` (or `.glb`/`.ply`) sibling next to it.

## Tuning per-request (override yaml defaults)

Each script also exposes the underlying sampling / geometry knobs as
env vars.  Leave any unset → script omits the field → api falls
through to the per-model defaults in the runner's `.models.yaml`.
Set them to override per-request.

| Env var | Scripts | Purpose |
|---|---|---|
| `NEGATIVE_PROMPT` | txt2img, img2img | Things to exclude (e.g. `"G-clamp, blurry"` when prompting a C-clamp). **Almost always set this** for object-specific gens. |
| `CFG_SCALE` | txt2img, img2img | Classifier-free guidance.  Yaml default 4.0; bump to 5-7 for stubborn-geometry industrial objects, 8+ if the model still resists.  Too high washes out aesthetics. |
| `STEPS` | txt2img, img2img, img2-3d, mesh2parts | Diffusion sampling steps.  Higher = finer detail, linear cost.  Yaml defaults: 50 (qwen-image), 50 (Hunyuan3D DiT), 50 (XPart DiT). |
| `SAMPLER` | txt2img, img2img | Sampler name.  Yaml default `dpm++_2m`.  Also: `euler`, `dpm++_sde`, `unipc`, `dpmpp_2m_sde`. |
| `SEED` | all | Integer seed for reproducible runs.  -1 = random. |
| `EXTRA_IMAGES` | img2img | Comma-separated paths to **additional reference images**.  Primary image (positional arg) is still the one being edited; extras are visual context Qwen-Image-Edit conditions on (style donors, subject refs, palette).  Sent as `image: [primary, ref1, ...]` to the api. |
| `GUIDANCE_SCALE` | img2-3d, mesh2parts | CFG for the 3D pipelines.  img23d default 7.5; bump if image fidelity is low; lower if you see spikes/floaters. |
| `OCTREE_RESOLUTION` | img2-3d | Marching-cubes resolution.  Yaml default 384.  256 = fast iteration, 512 = high-fidelity.  Quadratic memory. |
| `MC_LEVEL`, `BOX_V`, `NUM_CHUNKS` | img2-3d | Advanced MC tuning — see [generate_3d_models.md](../docs/generate_3d_models.md) for full descriptions. |
| `MAX_PARTS` | mesh2parts | Cap on K parts.  Pipeline default 0 (no cap).  Set 8-15 if P3-SAM is detecting too many parts and OOMing the conditioner.  Ignored when `AABB_FILE` is set. |
| `AABB` | mesh2parts | Inline JSON literal with shape `[K, 2, 3]` (K parts × min-corner + max-corner × xyz).  Bypasses P3-SAM auto-segmentation entirely — XPart decomposes exactly along your boundaries.  Coords in normalised mesh space (typically `[-1, 1]`). |
| `AABB_FILE` | mesh2parts | Same as `AABB` but read from a file (useful for large box lists).  `AABB` wins if both are set. |

**Worked example — stubborn mechanical object:**

```bash
NEGATIVE_PROMPT="G-clamp, bar clamp, vise, multiple objects, distorted, deformed, blurry" \
CFG_SCALE=5.5 \
STEPS=70 \
./scripts/txt2img.sh "single black cast iron C-clamp on white seamless background, deep throat, threaded screw spindle with T-bar at bottom, swivel pad at tip, isolated centered, 4k industrial catalog"
```

**Worked example — caller-driven mesh decomposition (inline):**

```bash
AABB='[[[-1,-1,-1],[-0.2,1,1]], [[-0.2,-1,-1],[0.2,1,1]], [[0.2,-1,-1],[1,1,1]]]' \
  ./scripts/mesh2parts.sh /tmp/mesh.glb 256
# → 3 parts along the X axis, P3-SAM auto-segmentation skipped
```

**Worked example — same thing from a file:**

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

## `txt2img.sh`

Text-to-image via `POST /v1/images/generations` (OpenAI-compatible
wire shape).

```bash
./scripts/txt2img.sh "a teacup with steam"
./scripts/txt2img.sh "a teacup with steam" qwen-image-2512 1024x1024
```

**Positional args:**

1. prompt (required)
2. model id (default `qwen-image-2512` — must match a runner `.models.yaml` entry with `provider: stable_diffusion_cpp`)
3. size as `WIDTHxHEIGHT` (default `1024x1024`; allowed values are the OpenAI Literal set — see `CreateImageRequest.size`)

**Response shape:**

```json
{
  "created": 1700000000,
  "output_format": "png",
  "data": [{"b64_json": "iVBORw0KGgo..."}]
}
```

No URL, no separate blob fetch — the PNG bytes are inlined as base64
in `data[0].b64_json`. The script decodes and saves it next to the JSON
response.

**Defaults baked into the api** (Qwen-Image-2512 tuning, override in
`.models.yaml` to change globally): 40 inference steps, `cfg_scale=2.5`,
sampler `euler`.

## `img2img.sh`

Image-to-image / instruction edit via `POST /v1/images/edits` (Qwen-Image-Edit-2511 backed).

```bash
./scripts/img2img.sh path/to/photo.png "make it autumn"
./scripts/img2img.sh path/to/photo.png "make it autumn" qwen-image-edit-2511 0.75
```

**Positional args:**

1. input image path (required — base64-encoded inline; PNG or JPEG)
2. edit prompt (required)
3. model id (default `qwen-image-edit-2511`)
4. `denoising_strength` (default `0.75`; range 0.0–1.0)

**Denoising strength** is the key knob:

- `0.0` reproduces the input exactly (no edit)
- `0.5` keeps strong structural fidelity, lets prompt nudge colors / style
- `0.75` (default) — useful sweet spot for prompt-guided edits
- `1.0` ignores the input image entirely (degrades to txt2img)

**Response shape** is identical to txt2img: a JSON envelope with one
or more `b64_json` PNGs in `data[]`.

**Request shape** departs from OpenAI's multipart `image-edits` API —
we use JSON with `image` carrying base64 because every other endpoint
in this api is JSON-with-base64. Keeps the wire surface uniform.

**Multiple reference images (`EXTRA_IMAGES`):** Qwen-Image-Edit-2511
conditions on additional reference images alongside the primary
image. Set `EXTRA_IMAGES` to a comma-separated list of paths and
each one gets base64-encoded and added to the request as part of the
polymorphic `image` field — a JSON array `[primary, ref1, ref2, ...]`
when extras are present, a plain string otherwise. The primary
(positional arg 1) is still the canvas being edited; extras are
style donors / subject anchors / palette refs / composition borrows.

```bash
EXTRA_IMAGES=./style.png,./subject.png \
  ./scripts/img2img.sh ./photo.png \
    "blend photo with the style of style.png; keep the pose from subject.png"
```

Practical wins:
- Style transfer with anchor — text-only style prompts are flaky;
  a style-donor image is more reliable.
- Identity preservation — include a clean reference of the subject
  when editing a low-quality original.
- Palette matching — `match the color palette of <ref>`.
- Compositional borrowing — borrow framing/lighting from one image
  while keeping the subject of another.

## `rembg.sh`

Background removal via briaai/RMBG-2.0 (BiRefNet) — purpose-built
segmentation that picks up where Qwen-Image-Edit's instruction-following
pipeline tops out. No prompt required.

```bash
./scripts/rembg.sh path/to/photo.png
./scripts/rembg.sh path/to/photo.png 1            # mask-only
SIZE=2048 ./scripts/rembg.sh path/to/photo.png    # higher-res internal resize
```

**Positional args:**

1. input image path (required — base64-encoded inline; PNG or JPEG)
2. `mask_only` (default `0`; pass `1` to skip the cutout composite and
   return only the grayscale alpha mask)

**Env overrides:**

- `SIZE` — square edge for the model's internal resize (default `1024`,
  the RMBG-2.0 recipe). The returned mask is always upsampled back to
  the source resolution, so this only affects model fidelity, not
  output shape.

**Response shape:**

```json
{
  "id": "8d4e2c1f9ab3",
  "created": 1700000000,
  "elapsed_sec": 1.4,
  "width": 1024,
  "height": 1024,
  "mask_b64": "iVBORw0K...",
  "transparent_b64": "iVBORw0K...",
  "cutout_url": "/v1/images/remove-bg/8d4e2c1f9ab3.png"
}
```

- `mask_b64` — single-channel PNG of the alpha mask (white = subject,
  black = background)
- `transparent_b64` — the source image with the mask applied as alpha;
  `null` when `mask_only=true`
- `cutout_url` — the same transparent PNG served as a hosted file via
  `GET /v1/images/remove-bg/{id}.png`; skips the base64 round-trip
  when you just want the file (also `null` when `mask_only=true`)

The script decodes `mask_b64` to `rembg_<ts>_mask.png` and
`transparent_b64` to `rembg_<ts>_cutout.png` next to the JSON
response. Generation is fast (~1-3 s on GPU); the heavy cost is the
first-request weight load (~5 s).

## `img2-3d.sh`

Image-to-3D via Hunyuan3D-2.1 (shape-only path). Two-step interaction:

1. `POST /v1/images/3d` — submit the conditioning image, get back paths + download URLs
2. `GET /v1/images/3d/{filename}` — stream the `.glb` mesh back through the api

```bash
./scripts/img2-3d.sh path/to/photo.png
./scripts/img2-3d.sh path/to/photo.png "mesh,gaussian"
```

**Positional args:**

1. input image path (required)
2. comma-separated formats (default `mesh`; valid values `mesh`, `gaussian`)

**Response shape:**

```json
{
  "id": "abc123def456",
  "created": 1700000000,
  "elapsed_sec": 48.2,
  "mesh_path": "/data/sd-out/3d/abc123def456.glb",
  "gaussian_path": "/data/sd-out/3d/abc123def456.ply",
  "mesh_url":     "/v1/images/3d/abc123def456.glb",
  "gaussian_url": "/v1/images/3d/abc123def456.ply",
  "preview_b64": "iVBORw0K..."
}
```

- `*_path` fields are the absolute paths on the runner pod — debug-only
- `*_url` fields are the relative URLs you actually `GET` to download
- `gaussian_path` / `gaussian_url` will be `null`: Hunyuan3D-2.1's
  shape-only path doesn't produce gaussian splats, only meshes
- `preview_b64` is also `null` on this backbone (no auto-render)

The script automatically follows `mesh_url` to download the `.glb` into
`$OUT_DIR`. Total wall-time includes the Hunyuan3D run itself (typically
~30–60 seconds per image on a 3060; no streaming).

### Multi-runner caveat

The `.glb` file exists on whichever runner ran the generation. The api
currently sends both the generation and the download to
`RunnerClient._endpoints[0]`, so the download always finds the file.
When Hunyuan3D gets deployed across multiple runners, the download
proxy will need to fan a HEAD out to each endpoint to locate the
artefact. That refactor lives in
`services/image_service.py::stream_3d_artifact`.

## `mesh2parts.sh`

Mesh-to-parts decomposition via Hunyuan3D-Part (P3-SAM + XPart).
**Input is a mesh, not an image** — typically the `.glb` from a prior
`img2-3d.sh` run.

```bash
./scripts/mesh2parts.sh path/to/mesh.glb
./scripts/mesh2parts.sh path/to/mesh.glb 256       # lower octree res, faster
./scripts/mesh2parts.sh path/to/mesh.glb 512 42    # with seed
```

**Positional args:**

1. input `.glb` mesh (required — base64-encoded inline)
2. `octree_resolution` (default `512`, allowed `128`+; lower = faster, blockier output)
3. `seed` (optional; empty string skips)

The script always requests `split: true` — every detected part is
emitted as its own `<id>_part_NN.glb` alongside the combined views.
There used to be a `split` toggle (4th positional / `SPLIT` env);
that's gone, because callers always want the per-part files in
practice and the combined `_decomposed.glb` is still produced.

**Response shape:**

```json
{
  "id": "abc123def456",
  "created": 1700000000,
  "elapsed_sec": 92.4,
  "mesh_path":     "/data/sd-out/3d_parts/abc123def456_decomposed.glb",
  "exploded_path": "/data/sd-out/3d_parts/abc123def456_exploded.glb",
  "bbox_path":     "/data/sd-out/3d_parts/abc123def456_bbox.glb",
  "gt_bbox_path":  "/data/sd-out/3d_parts/abc123def456_gt_bbox.glb",
  "mesh_url":      "/v1/images/3d/parts/abc123def456_decomposed.glb",
  "exploded_url":  "/v1/images/3d/parts/abc123def456_exploded.glb",
  "bbox_url":      "/v1/images/3d/parts/abc123def456_bbox.glb",
  "gt_bbox_url":   "/v1/images/3d/parts/abc123def456_gt_bbox.glb",
  "part_urls": [
    "/v1/images/3d/parts/abc123def456_part_00.glb",
    "/v1/images/3d/parts/abc123def456_part_01.glb"
  ]
}
```

**Outputs:**

| Suffix | What it is |
|--------|------------|
| `_decomposed.glb` | Assembled mesh with parts re-joined as one (each part is a separate primitive — viewable in glTF viewers as named groups) |
| `_exploded.glb` | Parts spatially separated — useful for visualisation, presentation, or debugging segmentation |
| `_bbox.glb` | Bounding-box wireframe only — shows what P3-SAM detected before X-Part regenerated the geometry |
| `_gt_bbox.glb` | Input mesh + bbox overlay — debug view to compare predicted boxes against the input |
| `_part_NN.glb` | Each detected part as a standalone `.glb`. Drag-and-drop into Blender / three.js / Unity. Same geometry as the corresponding primitive inside `_decomposed.glb`, just packaged individually. |

Ordering is whatever order XPart emitted the part latents — there's no
semantic guarantee about which index is which body part. The script
downloads them all and prints each path.

XPart is **fp32 only** (spconv kernels lack lower-precision paths),
so each request takes a few minutes — the script sets
`--max-time 1800` (30 min) accordingly.

**Best input meshes**: outputs of `img2-3d.sh` (i.e. Hunyuan3D-2.1
results) or scanned meshes. Hand-modeled CAD geometry can confuse
P3-SAM's part priors — the upstream README explicitly recommends
AI-generated or scanned input.

**Chaining the full text-to-parts pipeline:**

```bash
./scripts/txt2img.sh   "a single porcelain teapot, white background"
./scripts/rembg.sh     $OUT_DIR/txt2img_*.png        # → cutout PNG
./scripts/img2-3d.sh   $OUT_DIR/rembg_*_cutout.png   # → <id>.glb
./scripts/mesh2parts.sh $OUT_DIR/*.glb               # → per-part .glb files
```

## Implementation notes

- `img2img.sh`, `rembg.sh`, `img2-3d.sh`, and `mesh2parts.sh` write
  the base64-encoded image/mesh to a temp file and feed it to `jq`
  via `--rawfile`, then post the resulting JSON via
  `curl --data-binary @<file>`. Passing a multi-MB base64 blob as a
  `jq --arg` (or `curl -d`) overruns the OS argv limit at ~128 KB and
  fails with `Argument list too long`.
- All scripts honour the same `${AUTH_HEADER[@]+...}` safe expansion
  so an empty array doesn't trip `set -u`.

## Troubleshooting

- **`✘ no b64_json in response`** — the runner failed mid-generation.
  Open the `<endpoint>_<ts>.json` file and check the `detail` /
  `parameters` keys for the upstream error.
- **`502 Bad Gateway`** from `/edits` or `/generations` — sd-server
  returned non-200; almost always means the model files at the paths
  declared in `.models.yaml` aren't actually present on the runner pod.
- **`503 Service Unavailable`** from `/3d` — Hunyuan3D's CUDA
  extensions or the `hy3dgen` package aren't installed in the runner
  image. The response body names the missing package.
- **`404` from `/v1/images/3d/{file}`** — the runner the api targeted
  doesn't have that artefact. If you have multiple Hunyuan3D-equipped
  runners, check whether the file ended up on a different pod.
- **`502` from `/remove-bg`** with `"trust_remote_code"` in the body —
  the runtime image's transformers version isn't picking up RMBG-2.0's
  custom BiRefNet code. Verify `transformers>=4.40` is present and the
  weights downloaded with `huggingface-cli download briaai/RMBG-2.0`
  (not a shallow clone) so the custom code files are co-located.
- **`Argument list too long`** during script run — you're on an older
  copy that hadn't migrated to `jq --rawfile`. Pull main.
