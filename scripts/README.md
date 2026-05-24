# Image API test scripts

Three curl-based shell scripts for exercising the image generation
endpoints from the command line. No Python deps — just `bash`, `curl`,
and `jq`.

| Script | Endpoint | Backend | Output |
|--------|----------|---------|--------|
| [`test_txt2img.sh`](#test_txt2imgsh) | `POST /v1/images/generations` | stable-diffusion.cpp `sd-server` | base64 PNG inline in response |
| [`test_img2img.sh`](#test_img2imgsh) | `POST /v1/images/edits` | stable-diffusion.cpp `sd-server` (img2img) | base64 PNG inline in response |
| [`test_img2-3d.sh`](#test_img2-3dsh) | `POST /v1/images/3d` + `GET /v1/images/3d/{file}` | TRELLIS in-process pipeline | `.glb` mesh + `.ply` gaussian, streamed back through api |

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

## `test_img2-3d.sh`

Image-to-3D via TRELLIS. Two-step interaction:

1. `POST /v1/images/3d` — submit the conditioning image, get back paths + download URLs
2. `GET /v1/images/3d/{filename}` — stream the `.glb` mesh (and `.ply` splat) back through the api

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
- `preview_b64` is an optional small preview frame the script writes to
  disk if present

The script automatically follows the `mesh_url` and `gaussian_url`
links to download the binary artefacts into `$OUT_DIR`. Total wall-time
includes TRELLIS itself (minutes per image — there's no streaming).

### Multi-runner caveat

The `.glb` / `.ply` files exist on whichever runner ran the generation.
The api currently sends both the generation and the download to
`RunnerClient._endpoints[0]`, so the download always finds the file.
When TRELLIS gets deployed across multiple runners, the download proxy
will need to fan a HEAD out to each endpoint to locate the artefact.
That refactor lives in `services/image_service.py::stream_3d_artifact`.

## Implementation notes

- `test_img2img.sh` and `test_img2-3d.sh` write the base64-encoded
  image to a temp file and feed it to `jq` via `--rawfile`, then post
  the resulting JSON via `curl --data-binary @<file>`. Passing a
  multi-MB base64 blob as a `jq --arg` (or `curl -d`) overruns the OS
  argv limit at ~128 KB and fails with `Argument list too long`.
- All three scripts honour the same `${AUTH_HEADER[@]+...}` safe
  expansion so an empty array doesn't trip `set -u`.

## Troubleshooting

- **`✘ no b64_json in response`** — the runner failed mid-generation.
  Open the `<endpoint>_<ts>.json` file and check the `detail` /
  `parameters` keys for the upstream error.
- **`502 Bad Gateway`** from `/edits` or `/generations` — sd-server
  returned non-200; almost always means the model files at the paths
  declared in `.models.yaml` aren't actually present on the runner pod.
- **`503 Service Unavailable`** from `/3d` — TRELLIS' CUDA extensions
  aren't installed in the runner image. The response body names the
  missing package.
- **`404` from `/v1/images/3d/{file}`** — the runner the api targeted
  doesn't have that artefact. If you have multiple TRELLIS runners,
  check whether file ended up on a different pod.
- **`Argument list too long`** during script run — you're on an older
  copy that hadn't migrated to `jq --rawfile`. Pull main.

## Verified end-to-end (2026-05-24)

| Script | Model | Resolution | Output size | Notes |
|--------|-------|------------|-------------|-------|
| `test_txt2img.sh` | `qwen-image-2512` | 1024×1024 | 1.89 MB PNG | clean run, ~40s |
| `test_img2img.sh` | `qwen-image-edit-2511` | 1024×1024 | 1.80 MB PNG | edited a prior txt2img output with `denoising_strength=0.75` |
| `test_img2-3d.sh` | TRELLIS | n/a | not yet validated | weights / CUDA extensions still need installation on the runner |
