"""Image generation + image-to-3D orchestration.

Bridges the API's wire-protocol routers (``/v1/images/generations``,
``/v1/images/3d``) to the runner's native backends:

  * **txt2img** — stable-diffusion.cpp's ``sd-server`` exposes
    ``/sdapi/v1/txt2img`` (WebUI-compatible).  We acquire a runner
    server for the requested model, POST the prompt + sampling params,
    and unwrap the base64 PNG from ``response.images[0]``.

  * **img2-3d** — TRELLIS lives in-process inside the runner under
    ``/v1/pipelines/img23d/run``.  No server acquisition is needed; we
    POST the conditioning image and parameters directly.

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


async def generate_image(
    *,
    prompt: str,
    model_id: str,
    negative_prompt: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 40,
    cfg_scale: float = 2.5,
    sampler_name: str = "euler",
    seed: int = -1,
    batch_size: int = 1,
    client: Optional[RunnerClient] = None,
) -> TxtToImageResult:
    """Run a text-to-image generation against the runner's sd-server.

    The defaults match the Qwen-Image-2512 tutorial (Q4_K_M):
    40 steps, cfg_scale 2.5, sampler ``euler``, 1024×1024.  Callers
    that target SDXL or SD3 should override these.

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

    handle = await cli.acquire_server(model_id=model_id)
    try:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": sampler_name,
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
class ImageTo3DResult:
    id: str
    elapsed_sec: float
    mesh_path: Optional[str]
    gaussian_path: Optional[str]
    preview_b64: Optional[str]


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
    """Submit an image to the runner's TRELLIS-based pipeline.

    The pipeline is in-process on the runner, so we bypass
    ``acquire_server`` and hit ``/v1/pipelines/img23d/run`` directly on
    whichever runner currently advertises the pipeline.  We pick the
    first endpoint from :attr:`RunnerClient._endpoints`; if/when we
    deploy multiple runners with TRELLIS, replace this with a capability
    query against ``GET /v1/pipelines``.
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
        timeout=1200.0,  # TRELLIS can run for minutes per image
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
    )
