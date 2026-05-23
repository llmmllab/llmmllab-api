"""OpenAI-compatible image generation endpoints.

Surfaces stable-diffusion.cpp (via the runner) on the OpenAI
``/v1/images/generations`` wire protocol so existing OpenAI clients can
target Qwen-Image / SDXL / SD3 without any code changes.

Mapping
-------
``CreateImageRequest`` fields are translated to ``sd-server`` params:

==========================  =====================================
OpenAI field                 stable-diffusion.cpp param
==========================  =====================================
``prompt``                   ``prompt``
``size`` (``WIDTHxHEIGHT``)  ``width`` + ``height``
``n``                        ``batch_size``
``model``                    runner ``model_id`` for acquire_server
==========================  =====================================

OpenAI's ``response_format`` controls whether we hand the caller a
``b64_json`` blob (the runner's native form, zero copy) or a hosted
``url`` (not implemented — would require persisting to ``IMAGE_DIR``).

The ``edits`` and ``variations`` endpoints remain ``NotImplementedError``
stubs — they require img2img which sd-server supports but we haven't
wired through.
"""

import logging
import re
import time
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from models.openai.create_image_request import CreateImageRequest
from models.openai.image import Image
from models.openai.images_response import ImagesResponse
from services.image_service import (
    ImageServiceError,
    generate_3d,
    generate_image,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["Images"])


_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")


def _parse_size(size: Optional[str]) -> Tuple[int, int]:
    """Convert ``"1024x1024"`` into ``(1024, 1024)``; ``"auto"`` -> default."""
    if not size or size == "auto":
        return 1024, 1024
    m = _SIZE_RE.match(size)
    if not m:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size '{size}'. Use WIDTHxHEIGHT (e.g. '1024x1024').",
        )
    return int(m.group(1)), int(m.group(2))


@router.post("/edits")
async def createImageEdit() -> ImagesResponse:
    """Operation ID: createImageEdit"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/generations")
async def createImage(body: CreateImageRequest) -> ImagesResponse:
    """Generate an image from a prompt using a runner-hosted SD model.

    Defaults are tuned for the Qwen-Image-2512-GGUF Q4_K_M tutorial:
    40 inference steps, cfg_scale 2.5, sampler ``euler``, 1024×1024.
    """
    if not body.model:
        raise HTTPException(
            status_code=400,
            detail="`model` is required — point it at a runner-registered SD model ID.",
        )

    width, height = _parse_size(body.size)

    try:
        result = await generate_image(
            prompt=body.prompt,
            model_id=body.model,
            width=width,
            height=height,
            batch_size=body.n or 1,
        )
    except ImageServiceError as e:
        logger.error("Image generation failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e

    return ImagesResponse(
        created=result.created,
        data=[Image(b64_json=img.b64_png) for img in result.images],
        output_format="png",
    )


@router.post("/variations")
async def createImageVariation() -> ImagesResponse:
    """Operation ID: createImageVariation"""
    raise NotImplementedError("Endpoint not yet implemented")


# ---------------------------------------------------------------------------
# img2-3D — outside the OpenAI spec, exposed as a sibling endpoint.
# ---------------------------------------------------------------------------


class CreateImageTo3DRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded conditioning image (PNG or JPEG)")
    seed: Optional[int] = Field(42, description="RNG seed for reproducible runs")
    ss_steps: Optional[int] = Field(12, description="Sparse-structure sampler steps")
    slat_steps: Optional[int] = Field(12, description="SLAT sampler steps")
    ss_cfg_strength: Optional[float] = Field(7.5, description="Sparse-structure CFG strength")
    slat_cfg_strength: Optional[float] = Field(3.0, description="SLAT CFG strength")
    formats: Optional[List[str]] = Field(
        default_factory=lambda: ["mesh"],
        description="Outputs to materialise. Allowed: 'mesh' (.glb), 'gaussian' (.ply)",
    )


class CreateImageTo3DResponse(BaseModel):
    id: str = Field(..., description="Generation ID — also the stem of the persisted file(s)")
    created: int = Field(..., description="Unix timestamp (seconds)")
    elapsed_sec: float = Field(..., description="Server-reported wall-clock time")
    mesh_path: Optional[str] = Field(None, description="Path to the .glb mesh on the runner filesystem")
    gaussian_path: Optional[str] = Field(None, description="Path to the .ply gaussian-splat on the runner filesystem")
    preview_b64: Optional[str] = Field(None, description="Optional rendered preview frame (base64 PNG)")


@router.post("/3d", response_model=CreateImageTo3DResponse)
async def createImageTo3D(body: CreateImageTo3DRequest) -> CreateImageTo3DResponse:
    """Convert a 2D image to a 3D mesh + gaussian-splat representation.

    Backed by TRELLIS running in-process on the runner.  The generation
    is synchronous and can take minutes — clients should either set
    long HTTP timeouts or move long-running calls behind their own
    job queue.
    """
    try:
        result = await generate_3d(
            image_b64=body.image_b64,
            seed=body.seed or 42,
            ss_steps=body.ss_steps or 12,
            slat_steps=body.slat_steps or 12,
            ss_cfg_strength=body.ss_cfg_strength or 7.5,
            slat_cfg_strength=body.slat_cfg_strength or 3.0,
            formats=body.formats or ["mesh"],
        )
    except ImageServiceError as e:
        logger.error("img23d failed: %s", e)
        if e.status_code == 503:
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise HTTPException(status_code=502, detail=str(e)) from e

    return CreateImageTo3DResponse(
        id=result.id,
        created=int(time.time()),
        elapsed_sec=result.elapsed_sec,
        mesh_path=result.mesh_path,
        gaussian_path=result.gaussian_path,
        preview_b64=result.preview_b64,
    )
