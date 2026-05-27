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
from typing import List, Optional, Tuple, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from middleware.auth import get_user_id

from models.openai.create_image_request import CreateImageRequest
from models.openai.image import Image
from models.openai.images_response import ImagesResponse
from services.image_service import (
    ImageServiceError,
    edit_image,
    generate_3d,
    generate_3d_parts,
    generate_image,
    remove_image_background,
    stream_3d_artifact,
    stream_3d_parts_artifact,
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
    consistent.  ``image`` accepts either a single base64-encoded
    PNG / JPEG or a list of them (Qwen-Image-Edit-2511 supports
    multi-image conditioning so the model can e.g. blend / restyle /
    combine subjects across reference images).
    """

    prompt: str = Field(..., description="Edit instruction / new prompt")
    image: Union[str, List[str]] = Field(
        ...,
        description=(
            "Base64-encoded source image(s).  Single string for plain "
            "img2img.  List of strings for multi-image conditioning "
            "(Qwen-Image-Edit-2511 takes the first as the primary "
            "image being edited and the rest as visual-context "
            "reference images).  Each entry is a PNG or JPEG base64 "
            "blob; OpenAI accepts up to 16 references."
        ),
    )
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
async def createImageEdit(body: CreateImageEditRequest, request: Request) -> ImagesResponse:
    """Edit an image with stable-diffusion.cpp's img2img endpoint.

    Backed by Qwen-Image-Edit-2511-GGUF when ``model=qwen-image-edit-2511``;
    any sd-server model registered in the runner's ``.models.yaml`` with
    ``task: ImageToImage`` is eligible.
    """
    width, height = _parse_size(body.size)

    # Normalise the polymorphic ``image`` field into a primary
    # (the image being edited) + an optional list of additional
    # reference images that condition the edit without being the
    # noise seed.
    if isinstance(body.image, list):
        if not body.image:
            raise HTTPException(
                status_code=400,
                detail="`image` was provided as an empty list; need at least one base64 string.",
            )
        primary_image = body.image[0]
        extra_images: Optional[List[str]] = body.image[1:] or None
    else:
        primary_image = body.image
        extra_images = None

    try:
        result = await edit_image(
            prompt=body.prompt,
            image_b64=primary_image,
            extra_images_b64=extra_images,
            model_id=body.model,
            negative_prompt=body.negative_prompt,
            denoising_strength=body.denoising_strength,
            width=width,
            height=height,
            steps=body.steps,
            cfg_scale=body.cfg_scale,
            sampler_name=body.sampler_name,
            seed=body.seed if body.seed is not None else -1,
            user_id=get_user_id(request),
        )
    except ImageServiceError as e:
        logger.error("Image edit failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e

    return ImagesResponse(
        created=result.created,
        data=[Image(b64_json=img.b64_png) for img in result.images],
        output_format="png",
    )


class CreateImageRequestExtra(CreateImageRequest):
    """OpenAI's ``CreateImageRequest`` + sd-server sampling knobs.

    OpenAI's canonical request omits the diffusion-side controls
    (negative_prompt, cfg_scale, steps, sampler) because their
    hosted models don't expose them.  Our backend (stable-
    diffusion.cpp via sd-server) does, so we extend the schema
    here.  Fields default to ``None`` → falls through to per-model
    defaults in ``.models.yaml`` (resolved by
    ``_resolve_sd_defaults`` in image_service).  Pass any of them
    in the body to override per-request — same shape the
    ``/edits`` endpoint already accepts.
    """

    negative_prompt: Optional[str] = Field(None)
    cfg_scale: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Classifier-free guidance scale.  Higher = stronger adherence "
            "to the prompt at the cost of aesthetic fidelity.  For "
            "qwen-image the model default is 4.0; bump to 5-7 for "
            "stubborn-geometry objects (mechanical parts, technical "
            "illustrations).  None → use the model's yaml default."
        ),
    )
    steps: Optional[int] = Field(
        None,
        ge=1,
        description="Diffusion sampling steps.  None → model default.",
    )
    sampler_name: Optional[str] = Field(
        None,
        description=(
            "Sampler name.  Valid options depend on the sd-server build "
            "but typically include ``euler``, ``dpm++_2m``, ``dpm++_sde``, "
            "``unipc``, ``dpmpp_2m_sde``.  None → model default."
        ),
    )
    seed: Optional[int] = Field(
        -1, description="Integer seed.  -1 = random (default)."
    )


@router.post("/generations")
async def createImage(body: CreateImageRequestExtra, request: Request) -> ImagesResponse:
    """Generate an image from a prompt using a runner-hosted SD model.

    Defaults come from the runner's ``.models.yaml`` entry; per-request
    overrides via ``negative_prompt`` / ``cfg_scale`` / ``steps`` /
    ``sampler_name`` / ``seed`` win when present.
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
            negative_prompt=body.negative_prompt,
            width=width,
            height=height,
            steps=body.steps,
            cfg_scale=body.cfg_scale,
            sampler_name=body.sampler_name,
            seed=body.seed if body.seed is not None else -1,
            batch_size=body.n or 1,
            user_id=get_user_id(request),
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
    """Input for ``POST /v1/images/3d`` — image-to-3D via Hunyuan3D-2.1.

    Sampling/geometry knobs all default to ``None``, which means
    "fall through to per-model defaults from ``.models.yaml``".
    Pass any field to override per-request.

    The legacy ``ss_*`` / ``slat_*`` fields were TRELLIS-era params
    that the Hunyuan3D-2.1 pipeline ignored anyway; they're gone.
    """

    image_b64: str = Field(..., description="Base64-encoded conditioning image (PNG or JPEG)")
    seed: Optional[int] = Field(42, description="RNG seed for reproducible runs")
    formats: Optional[List[str]] = Field(
        default_factory=lambda: ["mesh"],
        description="Outputs to materialise. Allowed: 'mesh' (.glb), 'gaussian' (.ply)",
    )
    num_inference_steps: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Diffusion sampling steps for Hunyuan3D-2.1's DiT.  "
            "Pipeline default 50; bump to 75-100 for finer detail "
            "at the cost of wall-clock."
        ),
    )
    guidance_scale: Optional[float] = Field(
        None,
        ge=0.0,
        description=(
            "Classifier-free guidance scale.  Pipeline default 7.5; "
            "higher = stronger prompt-image fidelity but more prone "
            "to over-extrusion artefacts."
        ),
    )
    octree_resolution: Optional[int] = Field(
        None,
        ge=128,
        description=(
            "Marching-cubes octree resolution.  Pipeline default 384.  "
            "Higher = finer mesh detail (quadratic memory cost); "
            "256 is fast iteration, 512 is high-fidelity output."
        ),
    )
    mc_level: Optional[float] = Field(
        None,
        description=(
            "Marching-cubes iso-level.  Default is ``-1/512`` (slightly "
            "below 0) which captures the SDF surface tightly.  Pushing "
            "this more negative (e.g. ``-0.005``) thickens output "
            "geometry; pushing positive thins it (and risks holes)."
        ),
    )
    box_v: Optional[float] = Field(
        None,
        description=(
            "Bounding-box scale around the SDF.  Default 1.01 — slight "
            "expansion past the unit cube so marching cubes can close "
            "off the edges cleanly.  Rarely needs tuning."
        ),
    )
    num_chunks: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Marching-cubes evaluation chunk size — how many SDF "
            "samples to evaluate per GPU call.  Default 8000; bump to "
            "400000+ if you have VRAM headroom and want faster output."
        ),
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
async def createImageTo3D(body: CreateImageTo3DRequest, request: Request) -> CreateImageTo3DResponse:
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
            formats=body.formats or ["mesh"],
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            octree_resolution=body.octree_resolution,
            mc_level=body.mc_level,
            box_v=body.box_v,
            num_chunks=body.num_chunks,
            user_id=get_user_id(request),
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
# 3D mesh-to-parts decomposition — tencent/Hunyuan3D-Part (P3-SAM + XPart)
# ---------------------------------------------------------------------------


class CreateImageTo3DPartsRequest(BaseModel):
    """Input for ``POST /v1/3d/parts`` — decompose a whole mesh into parts.

    The mesh comes in as base64-encoded ``.glb`` bytes.  Typical
    workflow: run ``POST /v1/images/3d`` first to get a holistic
    mesh, then feed its ``.glb`` here (e.g. download via ``mesh_url``,
    re-encode to base64, post).  Scratch-built CAD meshes also work
    but XPart was trained on AI-generated and scanned meshes — its
    part priors may produce odd segmentations on overly clean
    procedural geometry.
    """

    mesh_b64: str = Field(
        ..., description="Base64-encoded input ``.glb`` mesh"
    )
    octree_resolution: Optional[int] = Field(
        512,
        ge=128,
        description=(
            "Marching-cubes octree resolution for each output part.  "
            "Default 512 matches the upstream demo; 256 is faster "
            "for iteration."
        ),
    )
    seed: Optional[int] = Field(
        None, description="RNG seed for reproducible runs (default 42 in pipeline)"
    )
    split: Optional[bool] = Field(
        False,
        description=(
            "If true, also export each detected part as its own "
            "``.glb`` and return one download URL per part in "
            "``part_urls``.  Useful for importing parts as separate "
            "objects in Blender / three.js / Unity without manually "
            "splitting the combined ``decomposed.glb`` scene."
        ),
    )
    num_inference_steps: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "XPart DiT sampling steps.  Pipeline default 50.  Higher = "
            "finer per-part geometry at the cost of wall-clock."
        ),
    )
    guidance_scale: Optional[float] = Field(
        None,
        description=(
            "XPart classifier-free guidance scale.  Default depends on "
            "the partformer config; bumping helps when the model "
            "produces overly-smoothed or merged parts."
        ),
    )
    max_parts: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Cap on the number of parts the conditioner attends over.  "
            "P3-SAM can detect 20-50+ on dense fixture meshes, which "
            "overflows the conditioner's cross-attention activation "
            "(~7-8 GB per K=25).  Caller sets a tighter cap (8-15 is "
            "the safe range) or 0 to disable capping.  Ignored when "
            "``aabb`` is provided."
        ),
    )
    aabb: Optional[List[List[List[float]]]] = Field(
        None,
        description=(
            "OPTIONAL caller-specified bounding boxes — bypasses "
            "P3-SAM's auto-segmentation entirely.  Shape ``[K, 2, 3]``: "
            "K parts, each with min-corner ``[x, y, z]`` and max-corner "
            "``[x, y, z]`` in the mesh's coordinate system.  P3-SAM "
            "normalises the input mesh to a unit cube around the "
            "centroid, so feeding coords in ``[-1, 1]`` typically "
            "works.  Use this when auto-segmentation merges parts "
            "you want kept separate, or when you know the exact "
            "regions you want to isolate (e.g. wireframe-driven CAD "
            "where the geometry is known up-front)."
        ),
    )


class CreateImageTo3DPartsResponse(BaseModel):
    id: str = Field(..., description="Generation ID — also the stem of the persisted files")
    created: int = Field(..., description="Unix timestamp (seconds)")
    elapsed_sec: float = Field(..., description="Server-reported wall-clock time")
    # Filesystem paths on the runner — debug-only.  Use the ``_url``
    # fields below to actually fetch artefacts.
    mesh_path: Optional[str] = Field(None, description="Runner path to the decomposed mesh")
    exploded_path: Optional[str] = Field(None, description="Runner path to the exploded-view mesh")
    bbox_path: Optional[str] = Field(None, description="Runner path to the bounding-box wireframe")
    gt_bbox_path: Optional[str] = Field(None, description="Runner path to the input + bbox overlay")
    part_paths: List[str] = Field(
        default_factory=list,
        description=(
            "Runner paths to per-part ``.glb`` files when "
            "``split=true``.  Empty list otherwise.  Debug-only; use "
            "``part_urls`` to download."
        ),
    )
    # Public download URLs.  Each routes through
    # ``GET /v1/3d/parts/{filename}`` which streams the .glb back.
    mesh_url: Optional[str] = Field(None, description="Download URL for the decomposed mesh")
    exploded_url: Optional[str] = Field(None, description="Download URL for the exploded view")
    bbox_url: Optional[str] = Field(None, description="Download URL for the bbox wireframe")
    gt_bbox_url: Optional[str] = Field(None, description="Download URL for the input + bbox overlay")
    part_urls: List[str] = Field(
        default_factory=list,
        description=(
            "Per-part download URLs when ``split=true`` was requested.  "
            "One entry per detected part; ordering matches what XPart "
            "emitted (no semantic guarantee about which index is which "
            "part).  Empty list when ``split=false``."
        ),
    )


# Sibling endpoint to ``POST /v1/images/3d``.  Final URL is
# ``/v1/images/3d/parts`` since the router prefix is ``/images`` and
# the api mounts it at ``/v1``.  Conceptually it's mesh-in / mesh-out
# rather than image-in / mesh-out, but lives in the same module so
# all 3D-flavoured endpoints stay co-located.
@router.post("/3d/parts", response_model=CreateImageTo3DPartsResponse)
async def createImageTo3DParts(
    body: CreateImageTo3DPartsRequest, request: Request
) -> CreateImageTo3DPartsResponse:
    """Decompose a holistic mesh into part-by-part geometry.

    Backed by tencent/Hunyuan3D-Part (P3-SAM + XPart) running
    in-process on the runner.  Returns four ``.glb`` files:

    * ``mesh_url``       — the assembled decomposed mesh
    * ``exploded_url``   — parts spatially separated for visualisation
    * ``bbox_url``       — bounding-box wireframe only
    * ``gt_bbox_url``    — input mesh + bbox overlay (debug)

    The generation is synchronous and can take several minutes per
    mesh — clients should set HTTP timeouts of at least 30 minutes.
    """
    try:
        result = await generate_3d_parts(
            mesh_b64=body.mesh_b64,
            octree_resolution=body.octree_resolution,
            seed=body.seed,
            split=bool(body.split),
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            max_parts=body.max_parts,
            aabb=body.aabb,
            user_id=get_user_id(request),
        )
    except ImageServiceError as e:
        logger.error("mesh2parts failed: %s", e)
        if e.status_code == 503:
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise HTTPException(status_code=502, detail=str(e)) from e

    import os as _os

    def _url(path: Optional[str]) -> Optional[str]:
        return f"/v1/images/3d/parts/{_os.path.basename(path)}" if path else None

    part_urls = [
        u
        for p in (result.part_paths or [])
        for u in [_url(p)]
        if u is not None
    ]

    return CreateImageTo3DPartsResponse(
        id=result.id,
        created=int(time.time()),
        elapsed_sec=result.elapsed_sec,
        mesh_path=result.mesh_path,
        exploded_path=result.exploded_path,
        bbox_path=result.bbox_path,
        gt_bbox_path=result.gt_bbox_path,
        part_paths=list(result.part_paths or []),
        mesh_url=_url(result.mesh_path),
        exploded_url=_url(result.exploded_path),
        bbox_url=_url(result.bbox_path),
        gt_bbox_url=_url(result.gt_bbox_path),
        part_urls=part_urls,
    )


@router.get("/3d/parts/{filename}")
async def downloadImageTo3DParts(filename: str):
    """Stream a Hunyuan3D-Part output ``.glb`` from the runner."""
    try:
        media_type, body = await stream_3d_parts_artifact(filename)
    except ImageServiceError as e:
        if e.status_code == 404:
            raise HTTPException(status_code=404, detail=str(e)) from e
        if e.status_code == 400:
            raise HTTPException(status_code=400, detail=str(e)) from e
        raise HTTPException(status_code=502, detail=str(e)) from e
    return StreamingResponse(body, media_type=media_type)


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
async def createBackgroundRemoval(
    body: RemoveBackgroundRequest, request: Request
) -> RemoveBackgroundResponse:
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
            user_id=get_user_id(request),
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
