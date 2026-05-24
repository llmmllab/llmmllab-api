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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from models.openai.create_image_request import CreateImageRequest
from models.openai.image import Image
from models.openai.images_response import ImagesResponse
from services.image_service import (
    ImageServiceError,
    edit_image,
    generate_3d,
    generate_image,
    remove_image_background,
    stream_3d_artifact,
    stream_rembg_artifact,
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


class CreateImageEditRequest(BaseModel):
    """Img2img / instruction-edit request body.

    Departs from OpenAI's multipart/form-data ``image-edits`` shape on
    purpose — every other image endpoint in this api is JSON with
    base64 inline, and matching that style keeps the test scripts
    consistent.  ``image`` accepts a base64-encoded PNG or JPEG.
    """

    prompt: str = Field(..., description="Edit instruction / new prompt")
    image: str = Field(..., description="Base64-encoded source image (PNG or JPEG)")
    model: str = Field(..., description="Runner model_id (e.g. 'qwen-image-edit-2511')")
    negative_prompt: Optional[str] = Field(None)
    size: Optional[str] = Field("1024x1024", description="WIDTHxHEIGHT")
    denoising_strength: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "0.0 reproduces input, 1.0 ignores it. 0.65–0.8 is the sweet "
            "spot for prompt-guided edits.  Largely a no-op on Qwen-Image-"
            "Edit since the edit pipeline uses ref-image conditioning "
            "rather than noise/denoise.  None → defer to model defaults."
        ),
    )
    cfg_scale: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Classifier-free guidance scale (sd-server ``cfg_scale`` → "
            "``txt_cfg``).  Higher = stronger adherence to the prompt at "
            "the cost of fidelity.  Qwen-Image-Edit-2511 ships with 4.0 "
            "as its default (matching diffusers); push to 6-8 for edits "
            "the model resists ('remove the background')."
        ),
    )
    steps: Optional[int] = Field(
        None, ge=1, description="Diffusion sampling steps (model default applies when None)."
    )
    sampler_name: Optional[str] = Field(
        None, description="Sampler name (model default applies when None)."
    )
    seed: Optional[int] = Field(-1, description="-1 for random")


@router.post("/edits")
async def createImageEdit(body: CreateImageEditRequest) -> ImagesResponse:
    """Edit an image with stable-diffusion.cpp's img2img endpoint.

    Backed by Qwen-Image-Edit-2511-GGUF when ``model=qwen-image-edit-2511``;
    any sd-server model registered in the runner's ``.models.yaml`` with
    ``task: ImageToImage`` is eligible.
    """
    width, height = _parse_size(body.size)

    try:
        result = await edit_image(
            prompt=body.prompt,
            image_b64=body.image,
            model_id=body.model,
            negative_prompt=body.negative_prompt,
            denoising_strength=body.denoising_strength,
            width=width,
            height=height,
            steps=body.steps,
            cfg_scale=body.cfg_scale,
            sampler_name=body.sampler_name,
            seed=body.seed if body.seed is not None else -1,
        )
    except ImageServiceError as e:
        logger.error("Image edit failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e

    return ImagesResponse(
        created=result.created,
        data=[Image(b64_json=img.b64_png) for img in result.images],
        output_format="png",
    )


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
    mesh_path: Optional[str] = Field(None, description="Path to the .glb mesh on the runner filesystem (debug-only — use mesh_url to download)")
    gaussian_path: Optional[str] = Field(None, description="Path to the .ply gaussian-splat on the runner filesystem (debug-only — use gaussian_url to download)")
    mesh_url: Optional[str] = Field(None, description="Relative URL to download the .glb via /v1/images/3d/{id}.glb")
    gaussian_url: Optional[str] = Field(None, description="Relative URL to download the .ply via /v1/images/3d/{id}.ply")
    preview_b64: Optional[str] = Field(None, description="Optional rendered preview frame (base64 PNG)")


@router.post("/3d", response_model=CreateImageTo3DResponse)
async def createImageTo3D(body: CreateImageTo3DRequest) -> CreateImageTo3DResponse:
    """Convert a 2D image to a 3D mesh + gaussian-splat representation.

    Backed by Hunyuan3D-2.1 (shape-only) running in-process on the
    runner.  Returns a ``.glb`` mesh; ``gaussian_url`` is always
    ``null`` since Hunyuan3D-2.1 doesn't produce gaussian splats.  The
    generation is synchronous and can take a couple of minutes per
    image — clients should set long HTTP timeouts.
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

    # Build relative download URLs so clients don't need pod access.  We
    # derive them from the artefact filename (basename of mesh_path /
    # gaussian_path) rather than rebuilding from result.id because that
    # leaves a single source of truth — whatever the pipeline named the
    # file is what we serve.
    import os as _os

    mesh_url = None
    if result.mesh_path:
        mesh_url = f"/v1/images/3d/{_os.path.basename(result.mesh_path)}"
    gaussian_url = None
    if result.gaussian_path:
        gaussian_url = f"/v1/images/3d/{_os.path.basename(result.gaussian_path)}"

    return CreateImageTo3DResponse(
        id=result.id,
        created=int(time.time()),
        elapsed_sec=result.elapsed_sec,
        mesh_path=result.mesh_path,
        gaussian_path=result.gaussian_path,
        mesh_url=mesh_url,
        gaussian_url=gaussian_url,
        preview_b64=result.preview_b64,
    )


# ---------------------------------------------------------------------------
# Background removal — briaai/RMBG-2.0 (in-process pipeline on the runner)
# ---------------------------------------------------------------------------


class RemoveBackgroundRequest(BaseModel):
    """Input for ``POST /v1/images/remove-bg`` — base64 image in, cutout out."""

    image: str = Field(..., description="Base64-encoded source image (PNG or JPEG)")
    mask_only: Optional[bool] = Field(
        False,
        description=(
            "If true, the response only includes the alpha mask (grayscale "
            "PNG) without the alpha-composited cutout.  Useful when the "
            "caller wants to do their own compositing downstream."
        ),
    )
    size: Optional[int] = Field(
        None,
        ge=64,
        description=(
            "Square edge in pixels for the model's internal resize.  "
            "Defaults to 1024 (the RMBG-2.0 recipe).  The mask is "
            "upsampled back to the source resolution regardless."
        ),
    )


class RemoveBackgroundResponse(BaseModel):
    id: str = Field(..., description="Generation ID — also the stem of the cutout file")
    created: int = Field(..., description="Unix timestamp (seconds)")
    elapsed_sec: float = Field(..., description="Server-reported wall-clock time")
    width: int = Field(..., description="Output width (matches source)")
    height: int = Field(..., description="Output height (matches source)")
    mask_b64: str = Field(..., description="Base64 grayscale PNG of the alpha mask")
    transparent_b64: Optional[str] = Field(
        None,
        description=(
            "Base64 PNG of the source image with the mask applied as alpha "
            "(transparent background).  ``null`` when ``mask_only=true``."
        ),
    )
    cutout_url: Optional[str] = Field(
        None,
        description=(
            "Relative URL to download the cutout PNG via "
            "``GET /v1/images/remove-bg/{id}.png`` — avoids the b64 "
            "round-trip for callers that just want the file."
        ),
    )


@router.post("/remove-bg", response_model=RemoveBackgroundResponse)
async def createBackgroundRemoval(body: RemoveBackgroundRequest) -> RemoveBackgroundResponse:
    """Remove the background of an image with briaai/RMBG-2.0.

    Purpose-built segmentation model — picks up where Qwen-Image-Edit's
    instruction-following pipeline tops out.  Always returns the alpha
    mask; also returns an alpha-composited transparent PNG unless
    ``mask_only=true``.  Generation is fast (~1-3 s on GPU).
    """
    try:
        result = await remove_image_background(
            image_b64=body.image,
            mask_only=bool(body.mask_only),
            size=body.size,
        )
    except ImageServiceError as e:
        logger.error("rembg failed: %s", e)
        if e.status_code == 503:
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise HTTPException(status_code=502, detail=str(e)) from e

    return RemoveBackgroundResponse(
        id=result.id,
        created=int(time.time()),
        elapsed_sec=result.elapsed_sec,
        width=result.width,
        height=result.height,
        mask_b64=result.mask_b64,
        transparent_b64=result.transparent_b64,
        cutout_url=result.cutout_url,
    )


@router.get("/remove-bg/{filename}")
async def downloadBackgroundRemoval(filename: str):
    """Stream a rembg cutout PNG back through the api."""
    try:
        media_type, body = await stream_rembg_artifact(filename)
    except ImageServiceError as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail=str(e)) from e
        if e.status_code == 400:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if e.status_code == 503:
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise HTTPException(status_code=502, detail=str(e)) from e

    return StreamingResponse(
        body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/3d/{filename}")
async def downloadImageTo3D(filename: str):
    """Download a generated 3D artefact (mesh or gaussian-splat) by filename.

    The :class:`CreateImageTo3DResponse` returned by ``POST /images/3d``
    includes ``mesh_url`` and ``gaussian_url`` fields pointing here — use
    those rather than reconstructing the path manually.

    Streams the file from whichever runner ran the generation.  Returns
    ``model/gltf-binary`` for ``.glb``, ``image/png`` for ``.png``, and
    ``application/octet-stream`` for ``.ply``.
    """
    try:
        media_type, body = await stream_3d_artifact(filename)
    except ImageServiceError as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail=str(e)) from e
        if e.status_code == 400:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if e.status_code == 503:
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise HTTPException(status_code=502, detail=str(e)) from e

    return StreamingResponse(
        body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
