# Image API test scripts

Five curl-based shell scripts for exercising the image + 3D endpoints
from the command line. No Python deps — just `bash`, `curl`, and `jq`.

| Script | Endpoint | Backend | Output |
|--------|----------|---------|--------|
| [`test_txt2img.sh`](#test_txt2imgsh) | `POST /v1/images/generations` | stable-diffusion.cpp `sd-server` | base64 PNG inline in response |
| [`test_img2img.sh`](#test_img2imgsh) | `POST /v1/images/edits` | stable-diffusion.cpp `sd-server` (img2img) | base64 PNG inline in response |
| [`test_rembg.sh`](#test_rembgsh) | `POST /v1/images/remove-bg` + `GET /v1/images/remove-bg/{file}` | briaai/RMBG-2.0 in-process pipeline | alpha mask PNG + transparent cutout PNG |
| [`test_img2-3d.sh`](#test_img2-3dsh) | `POST /v1/images/3d` + `GET /v1/images/3d/{file}` | Hunyuan3D-2.1 in-process pipeline | `.glb` mesh + `.ply` gaussian, streamed back through api |
| [`test_img2-3d-parts.sh`](#test_img2-3d-partssh) | `POST /v1/images/3d/parts` + `GET /v1/images/3d/parts/{file}` | Hunyuan3D-Part (P3-SAM + XPart) in-process pipeline | 4 `.glb` files: decomposed, exploded, bbox, gt_bbox |

> `runner_shutdown.sh` (ops tool to free VRAM by force-evicting runner
> servers) lives in the **llmmllab-runner** repo's `scripts/` directory.
> It uses the same `API_BASE` + `API_KEY` env vars as the scripts here,
> so an admin token reaches it without any port-forward.

## Common configuration

All three scripts honour the same environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_BASE` | `http://localhost:8000` | Base URL of llmmllab-api |
| `API_KEY` | *(unset)* | If set, sent as `Authorization: Bearer ...` |
| `OUT_DIR` | `./out` | Where decoded images / artefacts / raw JSON responses land |

Each invocation writes:

- A `<endpoint>_<unix-ts>.json` file with the raw response, useful for
  debugging or replaying.
- A decoded `.png` (or `.glb`/`.ply`) sibling next to it.

## `test_txt2img.sh`

Text-to-image via `POST /v1/images/generations` (OpenAI-compatible
wire shape).

```bash
./scripts/test_txt2img.sh "a teacup with steam"
./scripts/test_txt2img.sh "a teacup with steam" qwen-image-2512 1024x1024
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

## `test_img2img.sh`

Image-to-image / instruction edit via `POST /v1/images/edits` (Qwen-Image-Edit-2511 backed).

```bash
./scripts/test_img2img.sh path/to/photo.png "make it autumn"
./scripts/test_img2img.sh path/to/photo.png "make it autumn" qwen-image-edit-2511 0.75
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

## `test_rembg.sh`

Background removal via briaai/RMBG-2.0 (BiRefNet) — purpose-built
segmentation that picks up where Qwen-Image-Edit's instruction-following
pipeline tops out. No prompt required.

```bash
./scripts/test_rembg.sh path/to/photo.png
./scripts/test_rembg.sh path/to/photo.png 1            # mask-only
SIZE=2048 ./scripts/test_rembg.sh path/to/photo.png    # higher-res internal resize
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

## `test_img2-3d.sh`

Image-to-3D via Hunyuan3D-2.1 (shape-only path). Two-step interaction:

1. `POST /v1/images/3d` — submit the conditioning image, get back paths + download URLs
2. `GET /v1/images/3d/{filename}` — stream the `.glb` mesh back through the api

```bash
./scripts/test_img2-3d.sh path/to/photo.png
./scripts/test_img2-3d.sh path/to/photo.png "mesh,gaussian"
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

## `test_img2-3d-parts.sh`

Mesh-to-parts decomposition via Hunyuan3D-Part (P3-SAM + XPart).
**Input is a mesh, not an image** — typically the `.glb` from a prior
`test_img2-3d.sh` run.

```bash
./scripts/test_img2-3d-parts.sh path/to/mesh.glb
./scripts/test_img2-3d-parts.sh path/to/mesh.glb 256      # lower octree res, faster
./scripts/test_img2-3d-parts.sh path/to/mesh.glb 512 42   # with seed
./scripts/test_img2-3d-parts.sh path/to/mesh.glb 512 42 1 # split parts (4th positional)
SPLIT=1 ./scripts/test_img2-3d-parts.sh path/to/mesh.glb  # same via env
```

**Positional args:**

1. input `.glb` mesh (required — base64-encoded inline)
2. `octree_resolution` (default `512`, allowed `128`+; lower = faster, blockier output)
3. `seed` (optional; empty string skips)
4. `split` (optional; pass `1`/`true`/`yes` to also export each detected part as its own `<id>_part_NN.glb` for direct import into Blender / three.js / Unity. Default `false` returns only the combined `_decomposed.glb`.)

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
  "gt_bbox_url":   "/v1/images/3d/parts/abc123def456_gt_bbox.glb"
}
```

**The four base outputs:**

| Suffix | What it is |
|--------|------------|
| `_decomposed.glb` | Assembled mesh with parts re-joined as one (each part is a separate primitive — viewable in glTF viewers as named groups) |
| `_exploded.glb` | Parts spatially separated — useful for visualisation, presentation, or debugging segmentation |
| `_bbox.glb` | Bounding-box wireframe only — shows what P3-SAM detected before X-Part regenerated the geometry |
| `_gt_bbox.glb` | Input mesh + bbox overlay — debug view to compare predicted boxes against the input |

**Plus, when `split=true`:**

| Suffix | What it is |
|--------|------------|
| `_part_00.glb`, `_part_01.glb`, … | Each detected part as a standalone `.glb`. Drag-and-drop into Blender / three.js / Unity. Same geometry as the corresponding primitive inside `_decomposed.glb`, just packaged individually. |

Ordering is whatever order XPart emitted the part latents — there's no semantic guarantee about which index is which body part. The script downloads them all and prints each path.

The script downloads all four to `$OUT_DIR/<id>_<role>.glb` and prints
the wall-clock. XPart is **fp32 only** (spconv kernels lack lower-precision
paths), so each request takes a few minutes — the script bumps
`--max-time 1800` (30 min) accordingly.

**Best input meshes**: outputs of `test_img2-3d.sh` (i.e.
Hunyuan3D-2.1 results) or scanned meshes. Hand-modeled CAD geometry
can confuse P3-SAM's part priors — the upstream README explicitly
recommends AI-generated or scanned input.

**Chaining the full text-to-parts pipeline:**

```bash
./scripts/test_txt2img.sh "a single porcelain teapot, white background"
./scripts/test_img2-3d.sh $OUT_DIR/txt2img_*.png        # → <id>.glb
./scripts/test_img2-3d-parts.sh $OUT_DIR/*.glb         # → 4 part .glb files
```

## Implementation notes

- `test_img2img.sh`, `test_rembg.sh`, and `test_img2-3d.sh` write the
  base64-encoded image to a temp file and feed it to `jq` via
  `--rawfile`, then post the resulting JSON via
  `curl --data-binary @<file>`. Passing a multi-MB base64 blob as a
  `jq --arg` (or `curl -d`) overruns the OS argv limit at ~128 KB and
  fails with `Argument list too long`.
- All four scripts honour the same `${AUTH_HEADER[@]+...}` safe
  expansion so an empty array doesn't trip `set -u`.

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

## Verified end-to-end (2026-05-24)

| Script | Model | Resolution | Output size | Notes |
|--------|-------|------------|-------------|-------|
| `test_txt2img.sh` | `qwen-image-2512` | 1024×1024 | 1.89 MB PNG | clean run, ~40s |
| `test_img2img.sh` | `qwen-image-edit-2511` | 1024×1024 | 1.80 MB PNG | edited a prior txt2img output with `denoising_strength=0.75` |
| `test_img2-3d.sh` | Hunyuan3D-2.1 (shape-only) | n/a | 38.5 MB `.glb` mesh | conditioning image was the txt2img teacup; ~195s generation; mesh streamed back through `/v1/images/3d/{id}.glb` |
