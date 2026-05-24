"""Image generation + image-to-3D orchestration.

Bridges the API's wire-protocol routers (``/v1/images/generations``,
``/v1/images/3d``) to the runner's native backends:

  * **txt2img** — stable-diffusion.cpp's ``sd-server`` exposes
    ``/sdapi/v1/txt2img`` (WebUI-compatible).  We acquire a runner
    server for the requested model, POST the prompt + sampling params,
    and unwrap the base64 PNG from ``response.images[0]``.

  * **img2-3d** — Hunyuan3D-2.1 lives in-process inside the runner
    under ``/v1/pipelines/img23d/run``.  No server acquisition is
    needed; we POST the conditioning image and parameters directly.

Both helpers are coroutines and may raise:

  * ``ImageServiceError`` — wraps any non-200 from the runner with a
    descriptive message and the original status code.

The functions accept a pre-built :class:`RunnerClient` so tests can pass
in a stubbed client without monkey-patching the global singleton.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.runner_client import RunnerClient, runner_client as _default_client
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="image_service")


class ImageServiceError(RuntimeError):
    """Raised when the runner returns a non-success response."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class GeneratedImage:
    """One image returned by the runner.

    Holds the raw base64 PNG so the router can either inline it as
    ``b64_json`` or persist + return a URL, depending on what the wire
    protocol asks for.
    """

    b64_png: str


@dataclass(frozen=True)
class TxtToImageResult:
    images: List[GeneratedImage]
    created: int  # unix seconds — matches OpenAI's ImagesResponse.created
    parameters: Dict[str, Any]  # echo of the sd-server parameter dict


async def _resolve_sd_defaults(
    cli: RunnerClient, model_id: str
) -> Dict[str, Any]:
    """Pull per-model SD defaults (steps, cfg_scale, sampler_name, …) off
    the runner's model registry.

    The runner exposes ``GET /v1/models`` which returns each entry's
    full ``parameters`` block.  We map the SD-relevant fields back into
    a flat dict that callers merge into their request body.  Returns an
    empty dict on any failure — callers fall back to their hardcoded
    defaults, so a transient runner blip doesn't break image generation.
    """
    try:
        # ``list_models`` returns a flat ``List[Model]``-shaped dict per
        # entry; we look up by id.  The cost is amortised across the
        # generate_image call (one extra GET to a fast endpoint).
        models = await cli.list_models()
    except Exception:  # noqa: BLE001
        return {}

    target = None
    for entry in models or []:
        # Either Pydantic Model object or dict — handle both.
        entry_id = getattr(entry, "id", None) if not isinstance(entry, dict) else entry.get("id")
        if entry_id == model_id:
            target = entry
            break
    if target is None:
        return {}

    params = getattr(target, "parameters", None) if not isinstance(target, dict) else target.get("parameters")
    if params is None:
        return {}

    out: Dict[str, Any] = {}
    for field in ("steps", "cfg_scale", "sampler_name", "width", "height", "denoising_strength"):
        value = getattr(params, field, None) if not isinstance(params, dict) else params.get(field)
        if value is not None:
            out[field] = value
    return out


async def generate_image(
    *,
    prompt: str,
    model_id: str,
    negative_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    sampler_name: Optional[str] = None,
    seed: int = -1,
    batch_size: int = 1,
    client: Optional[RunnerClient] = None,
) -> TxtToImageResult:
    """Run a text-to-image generation against the runner's sd-server.

    Resolution order for each sampling parameter:

      1. Explicit kwarg from the caller (router or test).
      2. Per-model default from the runner's ``.models.yaml``
         ``parameters`` block (resolved via :func:`_resolve_sd_defaults`).
      3. Hardcoded fallback (40 steps, cfg_scale 2.5, sampler ``euler``,
         1024×1024 — tuned for Qwen-Image-2512 Q4_K_M).

    Parameters
    ----------
    prompt:
        Required positive prompt.
    model_id:
        ID of the SD model registered in the runner's ``.models.yaml``.
        The runner picks the right diffusion + VAE + text encoder files
        from the entry's ``details.diffusion_model_path`` etc.
    client:
        Override for the global :data:`runner_client` singleton.  Tests
        pass in a stub.
    """
    cli = client or _default_client

    # Layer the resolution chain so any value the caller didn't pass
    # falls back to model defaults, then global Qwen-Image-tuned defaults.
    defaults = await _resolve_sd_defaults(cli, model_id)
    resolved_steps = steps if steps is not None else defaults.get("steps", 40)
    resolved_cfg = cfg_scale if cfg_scale is not None else defaults.get("cfg_scale", 2.5)
    resolved_sampler = sampler_name if sampler_name is not None else defaults.get("sampler_name", "euler")
    resolved_width = width if width is not None else defaults.get("width", 1024)
    resolved_height = height if height is not None else defaults.get("height", 1024)

    handle = await cli.acquire_server(model_id=model_id)
    try:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "width": resolved_width,
            "height": resolved_height,
            "steps": resolved_steps,
            "cfg_scale": resolved_cfg,
            "sampler_name": resolved_sampler,
            "seed": seed,
            "batch_size": batch_size,
        }
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt

        logger.info(
            "Submitting txt2img request",
            extra={"model_id": model_id, "size": f"{width}x{height}", "steps": steps},
        )
        # Long timeout: image diffusion at 40 steps on a low-end card
        # can take well over a minute, and we don't want backoff to
        # hide a still-progressing job as a 503.
        response = await cli.proxy_request(
            handle,
            method="POST",
            path="sdapi/v1/txt2img",
            json=payload,
            timeout=600.0,
        )

        if response.status_code != 200:
            raise ImageServiceError(
                f"sd-server returned {response.status_code}: {response.text[:512]}",
                status_code=response.status_code,
            )

        body = response.json()
        raw_images = body.get("images") or []
        if not raw_images:
            raise ImageServiceError("sd-server returned no images", status_code=200)

        return TxtToImageResult(
            images=[GeneratedImage(b64_png=img) for img in raw_images],
            created=int(time.time()),
            parameters=body.get("parameters") or {},
        )
    finally:
        # Best-effort release — ``release_server`` swallows its own errors.
        try:
            await cli.release_server(handle)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"release_server failed: {e}")


@dataclass(frozen=True)
class ImageToImageResult:
    images: List[GeneratedImage]
    created: int
    parameters: Dict[str, Any]


async def edit_image(
    *,
    prompt: str,
    image_b64: str,
    model_id: str,
    negative_prompt: Optional[str] = None,
    denoising_strength: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    sampler_name: Optional[str] = None,
    seed: int = -1,
    client: Optional[RunnerClient] = None,
) -> ImageToImageResult:
    """Run an image edit (img2img) against the runner's sd-server.

    Same resolution chain as :func:`generate_image`: explicit kwargs win,
    otherwise model-level defaults from the runner's ``.models.yaml``
    ``parameters`` block, otherwise the Qwen-Image-Edit-2511 tuned
    fallbacks (40 steps, cfg 2.5, sampler ``euler``, 1024×1024,
    denoising_strength 0.75).

    ``denoising_strength`` controls how much the model deviates from
    the input image: 0.0 reproduces the input, 1.0 ignores it.  0.65–0.8
    is the useful range for prompt-guided edits.
    """
    cli = client or _default_client

    defaults = await _resolve_sd_defaults(cli, model_id)
    resolved_steps = steps if steps is not None else defaults.get("steps", 40)
    resolved_cfg = cfg_scale if cfg_scale is not None else defaults.get("cfg_scale", 2.5)
    resolved_sampler = sampler_name if sampler_name is not None else defaults.get("sampler_name", "euler")
    resolved_width = width if width is not None else defaults.get("width", 1024)
    resolved_height = height if height is not None else defaults.get("height", 1024)
    resolved_denoise = denoising_strength if denoising_strength is not None else defaults.get("denoising_strength", 0.75)

    handle = await cli.acquire_server(model_id=model_id)
    try:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            # sd-server's /sdapi/v1/img2img reads TWO different image
            # fields off the body, and they wire to DIFFERENT internal
            # attributes:
            #
            #   init_images:  legacy img2img noise-and-denoise path
            #                 (populates ``gen_params.init_image``)
            #   extra_images: QwenImageEditPlusPipeline ref-image
            #                 conditioning (populates
            #                 ``gen_params.ref_images``)
            #
            # The Qwen-Image-Edit pipeline only fires when ``ref_images``
            # is non-empty (see ``src/conditioner.hpp`` —
            # ``if (llm->enable_vision && conditioner_params.ref_images
            # != nullptr && !conditioner_params.ref_images->empty())``);
            # missing this is exactly why "remove the background" was
            # producing wildly different images — sd-server was falling
            # back to plain Qwen-Image txt2img on the prompt alone, with
            # the source image only used as a noise seed.
            #
            # We send the source image in BOTH fields so the same
            # endpoint works for edit-aware models (Qwen-Image-Edit-2511,
            # use ref_images path) and any legacy img2img model that
            # falls through (uses init_image path).
            "init_images": [image_b64],
            "extra_images": [image_b64],
            "denoising_strength": resolved_denoise,
            "width": resolved_width,
            "height": resolved_height,
            "steps": resolved_steps,
            "cfg_scale": resolved_cfg,
            "sampler_name": resolved_sampler,
            "seed": seed,
            "batch_size": 1,
        }
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt

        logger.info(
            "Submitting img2img request",
            extra={
                "model_id": model_id,
                "size": f"{width}x{height}",
                "steps": steps,
                "denoising_strength": denoising_strength,
            },
        )
        response = await cli.proxy_request(
            handle,
            method="POST",
            path="sdapi/v1/img2img",
            json=payload,
            timeout=600.0,
        )

        if response.status_code != 200:
            raise ImageServiceError(
                f"sd-server img2img returned {response.status_code}: {response.text[:512]}",
                status_code=response.status_code,
            )

        body = response.json()
        raw_images = body.get("images") or []
        if not raw_images:
            raise ImageServiceError("sd-server returned no images", status_code=200)

        return ImageToImageResult(
            images=[GeneratedImage(b64_png=img) for img in raw_images],
            created=int(time.time()),
            parameters=body.get("parameters") or {},
        )
    finally:
        try:
            await cli.release_server(handle)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"release_server failed: {e}")


@dataclass(frozen=True)
class ImageTo3DResult:
    id: str
    elapsed_sec: float
    mesh_path: Optional[str]
    gaussian_path: Optional[str]
    preview_b64: Optional[str]
    # The runner endpoint that holds the .glb / .ply on disk.  Stashed so
    # ``stream_3d_artifact`` can re-target the same runner instead of
    # round-robining (the file only exists on one pod).
    runner_endpoint: Optional[str] = None


async def generate_3d(
    *,
    image_b64: str,
    formats: Optional[List[str]] = None,
    seed: int = 42,
    ss_steps: int = 12,
    slat_steps: int = 12,
    ss_cfg_strength: float = 7.5,
    slat_cfg_strength: float = 3.0,
    client: Optional[RunnerClient] = None,
) -> ImageTo3DResult:
    """Submit an image to the runner's Hunyuan3D-2.1-based pipeline.

    The pipeline is in-process on the runner, so we bypass
    ``acquire_server`` and hit ``/v1/pipelines/img23d/run`` directly on
    whichever runner currently advertises the pipeline.  We pick the
    first endpoint from :attr:`RunnerClient._endpoints`; if/when we
    deploy multiple runners with Hunyuan3D, replace this with a
    capability query against ``GET /v1/pipelines``.
    """
    cli = client or _default_client

    if not cli._endpoints:
        raise ImageServiceError("No runner endpoints configured", status_code=503)

    endpoint = cli._endpoints[0]
    payload: Dict[str, Any] = {
        "image_b64": image_b64,
        "seed": seed,
        "ss_steps": ss_steps,
        "slat_steps": slat_steps,
        "ss_cfg_strength": ss_cfg_strength,
        "slat_cfg_strength": slat_cfg_strength,
        "formats": formats or ["mesh"],
    }

    logger.info(
        "Submitting img23d request",
        extra={"endpoint": endpoint, "formats": payload["formats"]},
    )

    http_client = cli._get_client()
    response = await http_client.post(
        f"{endpoint}/v1/pipelines/img23d/run",
        json=payload,
        timeout=1200.0,  # Hunyuan3D can run for minutes per image
    )
    if response.status_code != 200:
        raise ImageServiceError(
            f"runner img23d returned {response.status_code}: {response.text[:512]}",
            status_code=response.status_code,
        )

    body = response.json()
    return ImageTo3DResult(
        id=body.get("id", ""),
        elapsed_sec=float(body.get("elapsed_sec", 0.0)),
        mesh_path=body.get("mesh_path"),
        gaussian_path=body.get("gaussian_path"),
        preview_b64=body.get("preview_b64"),
        runner_endpoint=endpoint,
    )


# ---------------------------------------------------------------------------
# Background removal — briaai/RMBG-2.0 via the runner's in-process
# ``rembg`` pipeline.  Unlike SD txt2img / img2img we don't acquire a
# server: the model is loaded inside the runner process and exposed at
# ``/v1/pipelines/rembg/run``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RembgResult:
    id: str
    mask_b64: str
    transparent_b64: Optional[str]
    cutout_url: Optional[str]
    width: int
    height: int
    elapsed_sec: float
    runner_endpoint: Optional[str] = None


async def remove_image_background(
    *,
    image_b64: str,
    mask_only: bool = False,
    size: Optional[int] = None,
    client: Optional[RunnerClient] = None,
) -> RembgResult:
    """Remove the background of an image via briaai/RMBG-2.0.

    Always returns ``mask_b64`` (the grayscale alpha mask) and, unless
    ``mask_only=True``, also returns ``transparent_b64`` (the source
    image with the mask applied as alpha).  The cutout is also
    persisted on the runner and accessible via the proxy GET endpoint
    ``/v1/images/remove-bg/{id}.png``.
    """
    cli = client or _default_client
    if not cli._endpoints:
        raise ImageServiceError("No runner endpoints configured", status_code=503)

    endpoint = cli._endpoints[0]
    payload: Dict[str, Any] = {"image_b64": image_b64, "mask_only": bool(mask_only)}
    if size is not None:
        payload["size"] = int(size)

    logger.info("Submitting rembg request", extra={"endpoint": endpoint})

    http_client = cli._get_client()
    response = await http_client.post(
        f"{endpoint}/v1/pipelines/rembg/run",
        json=payload,
        timeout=120.0,  # RMBG-2.0 is a single forward pass; ~1-3 s on GPU
    )
    if response.status_code != 200:
        raise ImageServiceError(
            f"runner rembg returned {response.status_code}: {response.text[:512]}",
            status_code=response.status_code,
        )

    body = response.json()
    cutout_path = body.get("cutout_path")
    cutout_url: Optional[str] = None
    if cutout_path:
        import os as _os
        cutout_url = f"/v1/images/remove-bg/{_os.path.basename(cutout_path)}"

    return RembgResult(
        id=body.get("id", ""),
        mask_b64=body.get("mask_b64") or "",
        transparent_b64=body.get("transparent_b64"),
        cutout_url=cutout_url,
        width=int(body.get("width", 0) or 0),
        height=int(body.get("height", 0) or 0),
        elapsed_sec=float(body.get("elapsed_sec", 0.0)),
        runner_endpoint=endpoint,
    )


async def stream_rembg_artifact(
    filename: str,
    *,
    client: Optional[RunnerClient] = None,
):
    """Stream a rembg cutout PNG from the runner.

    Mirrors :func:`stream_3d_artifact`.  Filename is validated by the
    runner side (regex on basename), so we just forward the GET.
    """
    if not filename or "/" in filename or "\\" in filename:
        raise ImageServiceError(
            f"Invalid filename '{filename}'",
            status_code=400,
        )

    cli = client or _default_client
    if not cli._endpoints:
        raise ImageServiceError("No runner endpoints configured", status_code=503)

    endpoint = cli._endpoints[0]
    url = f"{endpoint}/v1/pipelines/rembg/files/{filename}"
    http_client = cli._get_client()

    stream_ctx = http_client.stream("GET", url, timeout=30.0)
    resp = await stream_ctx.__aenter__()
    try:
        if resp.status_code == 404:
            raise ImageServiceError(
                f"Artefact '{filename}' not found on runner", status_code=404,
            )
        if resp.status_code == 400:
            raise ImageServiceError(
                f"Runner rejected filename '{filename}'", status_code=400,
            )
        if resp.status_code >= 400:
            raise ImageServiceError(
                f"Runner returned {resp.status_code}", status_code=502,
            )
    except BaseException:
        await stream_ctx.__aexit__(None, None, None)
        raise

    async def _iter_bytes():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await stream_ctx.__aexit__(None, None, None)

    return "image/png", _iter_bytes()


# ---------------------------------------------------------------------------
# Artefact download — proxies through to whichever runner holds the file.
# ---------------------------------------------------------------------------


_IMG23D_FILENAME_RE = __import__("re").compile(r"^[A-Za-z0-9_-]{1,64}\.(glb|ply|png)$")


async def stream_3d_artifact(
    filename: str,
    *,
    client: Optional[RunnerClient] = None,
):
    """Stream a generated 3D artefact (.glb / .ply / .png) from the runner.

    Returns an ``(media_type, byte_iterator)`` tuple that the api router
    wraps in a :class:`StreamingResponse`.  We don't load the whole file
    into memory — .glb meshes from Hunyuan3D can be 10+ MiB and we'd
    rather pass them through transparently.

    Multi-runner caveat: the file lives on the runner that ran the
    generation; we currently only target ``RunnerClient._endpoints[0]``
    (the same one ``generate_3d`` uses).  When we scale Hunyuan3D
    across multiple runners, fan out a HEAD request to each and return
    the 200 responder.
    """
    if not _IMG23D_FILENAME_RE.match(filename):
        raise ImageServiceError(
            f"Invalid filename '{filename}'. Expected <id>.{{glb,ply,png}}.",
            status_code=400,
        )

    cli = client or _default_client

    if not cli._endpoints:
        raise ImageServiceError("No runner endpoints configured", status_code=503)

    endpoint = cli._endpoints[0]
    url = f"{endpoint}/v1/pipelines/img23d/files/{filename}"

    media_types = {
        ".glb": "model/gltf-binary",
        ".ply": "application/octet-stream",
        ".png": "image/png",
    }
    import os as _os
    media_type = media_types.get(_os.path.splitext(filename)[1].lower(), "application/octet-stream")

    http_client = cli._get_client()

    # Opening the streaming GET inside the context manager and reading the
    # initial response headers before we hand the body iterator to the
    # caller lets us surface 4xx/5xx as clean ``ImageServiceError`` values
    # without partially-streamed output.  The runner doesn't register a
    # HEAD route, so we skip the pre-check that would have made this
    # simpler.
    stream_ctx = http_client.stream("GET", url, timeout=120.0)
    resp = await stream_ctx.__aenter__()
    try:
        if resp.status_code == 404:
            raise ImageServiceError(
                f"Artefact '{filename}' not found on runner",
                status_code=404,
            )
        if resp.status_code == 400:
            raise ImageServiceError(
                f"Runner rejected filename '{filename}'", status_code=400
            )
        if resp.status_code >= 400:
            raise ImageServiceError(
                f"Runner returned {resp.status_code} fetching '{filename}'",
                status_code=502,
            )
    except BaseException:
        await stream_ctx.__aexit__(None, None, None)
        raise

    async def _iter_bytes():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await stream_ctx.__aexit__(None, None, None)

    return media_type, _iter_bytes()
