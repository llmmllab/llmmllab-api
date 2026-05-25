from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat


class ModelParameters(BaseModel):
    """Parameters for configuring a language model"""

    num_ctx: Annotated[
        Optional[int], Field(default=None, description="Size of the context window")
    ] = None
    """Size of the context window"""
    repeat_last_n: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Number of tokens to consider for repetition penalties",
        ),
    ] = None
    """Number of tokens to consider for repetition penalties"""
    repeat_penalty: Annotated[
        Optional[float], Field(default=None, description="Penalty for repetitions")
    ] = None
    """Penalty for repetitions"""
    temperature: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Sampling temperature; higher values produce more creative outputs",
        ),
    ] = None
    """Sampling temperature; higher values produce more creative outputs"""
    seed: Annotated[
        Optional[int],
        Field(default=None, description="Random seed for reproducibility"),
    ] = None
    """Random seed for reproducibility"""
    stop: Annotated[
        Optional[List[str]],
        Field(
            default=None, description="Sequences where the model should stop generating"
        ),
    ] = None
    """Sequences where the model should stop generating"""
    num_predict: Annotated[
        Optional[int],
        Field(default=None, description="Maximum number of tokens to predict"),
    ] = None
    """Maximum number of tokens to predict"""
    top_k: Annotated[
        Optional[int],
        Field(default=None, description="Limits next token selection to top K options"),
    ] = None
    """Limits next token selection to top K options"""
    top_p: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Limits next token selection to tokens comprising the top P probability mass (nucleus sampling)",
        ),
    ] = None
    """Limits next token selection to tokens comprising the top P probability mass (nucleus sampling)"""
    min_p: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Minimum probability threshold for token selection",
        ),
    ] = None
    """Minimum probability threshold for token selection"""
    think: Annotated[
        Optional[bool],
        Field(
            default=None, description='Whether to enable "thinking" mode for the model'
        ),
    ] = None
    """Whether to enable \"thinking\" mode for the model"""
    max_tokens: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Maximum number of tokens to generate in a single response",
        ),
    ] = None
    """Maximum number of tokens to generate in a single response"""
    n_parts: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Number of parts to split the model into. -1 means auto.",
        ),
    ] = None
    """Number of parts to split the model into. -1 means auto."""
    batch_size: Annotated[
        Optional[int],
        Field(default=None, description="Batch size for processing inputs"),
    ] = None
    """Batch size for processing inputs"""
    micro_batch_size: Annotated[
        Optional[int],
        Field(default=None, description="Micro batch size for processing inputs"),
    ] = None
    """Micro batch size for processing inputs"""
    n_gpu_layers: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Number of model layers to keep on GPU for performance optimization",
            ge=-1,
        ),
    ] = None
    """Number of model layers to keep on GPU for performance optimization"""
    main_gpu: Annotated[
        Optional[int],
        Field(
            default=-1,
            description="Main GPU device index (-1 for auto-selection)",
            ge=-1,
        ),
    ] = -1
    """Main GPU device index (-1 for auto-selection)"""
    tensor_split: Annotated[
        Optional[str],
        Field(
            default=None,
            description='Comma-separated fractions of model to put on each GPU (e.g. "2,5,5")',
        ),
    ] = None
    """Comma-separated fractions of model to put on each GPU (e.g. \"2,5,5\")"""
    split_mode: Annotated[
        Optional[Literal["none", "layer", "row"]],
        Field(default="layer", description="How to split model across devices"),
    ] = "layer"
    """How to split model across devices"""
    n_cpu_moe: Annotated[
        Optional[int],
        Field(
            default=0,
            description="Number of MoE (Mixture of Experts) layers to keep on CPU for memory optimization",
            ge=0,
        ),
    ] = 0
    """Number of MoE (Mixture of Experts) layers to keep on CPU for memory optimization"""
    kv_on_cpu: Annotated[
        Optional[bool],
        Field(
            default=False, description="Store key-value cache on CPU to save GPU memory"
        ),
    ] = False
    """Store key-value cache on CPU to save GPU memory"""
    reasoning_effort: Annotated[
        Optional[Literal["low", "medium", "high"]],
        Field(
            default="medium",
            description="Reasoning effort level for chain-of-thought processing",
        ),
    ] = "medium"
    """Reasoning effort level for chain-of-thought processing"""
    reasoning_budget: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Maximum tokens to spend on reasoning (chain-of-thought) before generating final answer",
        ),
    ] = None
    """Maximum tokens to spend on reasoning (chain-of-thought) before generating final answer"""
    flash_attention: Annotated[
        Optional[bool],
        Field(
            default=True,
            description="Enable flash attention optimization for memory efficiency",
        ),
    ] = True
    """Enable flash attention optimization for memory efficiency"""
    parallel: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Number of parallel request slots (llama.cpp --parallel). Defaults to 1 for large context models.",
            ge=1,
        ),
    ] = None
    """Number of parallel request slots (llama.cpp --parallel). Defaults to 1 for large context models."""
    ctx_size_reduction_limit: Annotated[
        Optional[float],
        Field(
            default=0.5,
            description="Minimum fraction of num_ctx that llama.cpp --fit is allowed to auto-reduce to (0.0-1.0). Default 0.5 means context can shrink to 50% of requested.",
            ge=0.0,
            le=1.0,
        ),
    ] = 0.5
    """Minimum fraction of num_ctx that llama.cpp --fit is allowed to auto-reduce to (0.0-1.0). Default 0.5 means context can shrink to 50% of requested."""
    slot_prompt_similarity: Annotated[
        Optional[float],
        Field(
            default=None,
            description="llama.cpp --slot-prompt-similarity threshold (0.0-1.0). Set to 0 to disable LCP slot matching; set to 1.0 to require exact match. None uses llama.cpp default (0.10).",
            ge=0.0,
            le=1.0,
        ),
    ] = None
    """llama.cpp --slot-prompt-similarity threshold (0.0-1.0). Set to 0 to disable LCP slot matching."""
    spec_type_mtp: Annotated[
        Optional[bool],
        Field(
            default=False,
            description="Enable MTP (Multi-Token Prediction) speculative decoding. Passes --spec-type draft-mtp to llama-server.",
        ),
    ] = False
    """Enable MTP speculative decoding for ~2x token generation speedup."""
    spec_draft_n_max: Annotated[
        Optional[int],
        Field(
            default=3,
            description="Number of draft tokens for MTP speculative decoding (llama.cpp --spec-draft-n-max).",
            ge=1,
            le=16,
        ),
    ] = 3
    """Number of draft tokens for MTP speculative decoding."""
    kv_unified: Annotated[
        bool,
        Field(
            default=True,
            description="Use unified key-value cache format (llama.cpp --kv-unified) for improved performance and compatibility with future features. Requires llama.cpp v1.3.0 or later.",
        ),
    ] = True
    """Use unified key-value cache format for improved performance and future compatibility."""

    cache_reuse: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Minimum chunk size (in tokens) that llama.cpp will attempt to reuse from "
            "the KV cache via KV-shifting for near-prefix matches (llama.cpp --cache-reuse, "
            "request param n_cache_reuse). Requires prompt caching to be enabled. "
            "0 disables reuse; a reasonable default is 256.",
            ge=0,
        ),
    ] = None
    """Minimum chunk size for KV-shifting prefix reuse (llama.cpp --cache-reuse). Default 256 when unset."""

    cache_ram: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Host-memory (system RAM) prompt cache size in MiB (llama.cpp --cache-ram). "
            "When set, idle slot KV data is cached in system RAM as a secondary tier "
            "behind VRAM, enabling --cache-idle-slots for automatic save/restore of idle slots. "
            "Default: 8192 (8 GiB). Set to 0 to disable, -1 for no limit.",
        ),
    ] = None
    """Host-memory prompt cache size in MiB. Active KV always lives in VRAM; this provides a RAM fallback for idle slots."""

    # ------------------------------------------------------------------
    # stable-diffusion.cpp / sd-server sampling defaults.
    #
    # These travel in the request body to ``/sdapi/v1/txt2img`` (or
    # ``/sdapi/v1/img2img``) — they are NOT CLI flags on sd-server.
    # The api reads them off ``model.parameters`` when the wire request
    # omits the corresponding field, so YAML defines per-model defaults
    # without forcing every caller to know them.
    # ------------------------------------------------------------------
    steps: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Default diffusion sampling steps for SD models (sd-server `steps`). "
            "Qwen-Image-2512 uses 40.",
            ge=1,
        ),
    ] = None
    """Default diffusion sampling steps for SD models."""

    cfg_scale: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Default classifier-free guidance scale (sd-server `cfg_scale`). "
            "Qwen-Image-2512 uses 2.5; SDXL typically 7.0.",
            ge=0.0,
        ),
    ] = None
    """Default classifier-free guidance scale for SD models."""

    sampler_name: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Default sampler for SD models (sd-server `sampler_name`). "
            "Qwen-Image-2512 uses `euler`.",
        ),
    ] = None
    """Default sampler for SD models."""

    width: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Default output width (sd-server `width`).",
            ge=64,
        ),
    ] = None
    """Default output image width in pixels."""

    height: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Default output height (sd-server `height`).",
            ge=64,
        ),
    ] = None
    """Default output image height in pixels."""

    denoising_strength: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Default denoising strength for img2img (sd-server `denoising_strength`, 0.0–1.0). "
            "Only meaningful for ``task: ImageToImage`` models.",
            ge=0.0,
            le=1.0,
        ),
    ] = None
    """Default denoising strength for img2img (sd-server)."""

    # ------------------------------------------------------------------
    # Multi-GPU layout for sd-server (per-component placement).
    #
    # stable-diffusion.cpp does NOT support llama.cpp-style tensor
    # splitting (a single layer's weights sharded across devices), but
    # it does let us place different sub-models onto different devices.
    # Use ``sd_backend`` to set ``--backend`` (compute placement) and
    # ``sd_params_backend`` to set ``--params-backend`` (weight
    # storage placement).  Example values:
    #
    #     sd_backend: "clip=cuda0,diffusion=cuda1,vae=cuda1"
    #     sd_params_backend: "diffusion=cpu"
    #
    # When ``sd_backend`` is set, the runner intentionally does NOT pin
    # ``CUDA_VISIBLE_DEVICES`` (otherwise sd-server would only see the
    # one GPU named by ``main_gpu`` and the multi-device names in the
    # backend string would fail to resolve).
    # ------------------------------------------------------------------
    sd_backend: Annotated[
        Optional[str],
        Field(
            default=None,
            description="sd-server `--backend` value: per-component compute placement. "
            'Example: ``"clip=cuda0,diffusion=cuda1,vae=cuda1"``. '
            "Mutually exclusive with single-GPU ``main_gpu`` pinning.",
        ),
    ] = None
    """sd-server --backend compute placement (multi-GPU layout)."""

    sd_params_backend: Annotated[
        Optional[str],
        Field(
            default=None,
            description="sd-server `--params-backend` value: per-component weight storage. "
            'Example: ``"diffusion=cpu"`` to keep diffusion weights in RAM and '
            "compute on GPU.  Trades ~30% throughput for VRAM headroom.",
        ),
    ] = None
    """sd-server --params-backend (weight storage placement)."""

    flow_shift: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Flow shift for flow-matching models like Qwen-Image (sd-server `--flow-shift`). "
            "Tutorial recommends 3 for Qwen-Image / Qwen-Image-Edit; default is auto if unset.",
            ge=0.0,
        ),
    ] = None
    """Flow shift for flow-matching schedulers (Qwen-Image / SD3.x / WAN)."""

    qwen_image_zero_cond_t: Annotated[
        Optional[bool],
        Field(
            default=None,
            description="Enable Qwen-Image's zero_cond_t conditioning (sd-server `--qwen-image-zero-cond-t`). "
            "REQUIRED for Qwen-Image-Edit-2511; the leejet/stable-diffusion.cpp docs note that "
            "without it, image editing quality degrades significantly.",
        ),
    ] = None
    """Enable Qwen-Image zero_cond_t conditioning — required for Qwen-Image-Edit-2511."""

    diffusion_fa: Annotated[
        Optional[bool],
        Field(
            default=None,
            description="Enable Flash Attention in the diffusion model only (sd-server `--diffusion-fa`). "
            "Used in the official sd.cpp Qwen-Image-Edit-2511 example for speed + memory. "
            "Distinct from --fa which enables FA everywhere (including text encoder).",
        ),
    ] = None
    """Enable flash attention for the diffusion model only (sd-server)."""

    diffusion_conv_direct: Annotated[
        Optional[bool],
        Field(
            default=None,
            description="Use ``ggml_conv2d_direct`` in the diffusion model (sd-server `--diffusion-conv-direct`). "
            "Memory-efficient convolution variant; small speedup on most configurations.",
        ),
    ] = None
    """Enable direct conv2d in the diffusion model."""

    vae_conv_direct: Annotated[
        Optional[bool],
        Field(
            default=None,
            description="Use ``ggml_conv2d_direct`` in the VAE (sd-server `--vae-conv-direct`). "
            "Pairs naturally with ``--vae-tiling`` for low-VRAM VAE decode.",
        ),
    ] = None
    """Enable direct conv2d in the VAE."""

    offload_to_cpu: Annotated[
        Optional[bool],
        Field(
            default=None,
            description="Keep SD weights in RAM and page to VRAM on demand (sd-server `--offload-to-cpu`). "
            "Trades ~30% throughput for VRAM headroom; used in the official sd.cpp examples on "
            "memory-constrained setups.",
        ),
    ] = None
    """Offload SD weights to system RAM (sd-server --offload-to-cpu)."""

    max_vram_gib: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Maximum VRAM budget in GiB for sd-server's graph-cut segmented execution "
            "(`--max-vram`).  0 disables.  Useful when the GPU is shared with other workloads.",
            ge=0.0,
        ),
    ] = None
    """sd-server --max-vram budget in GiB."""

    vae_on_cpu: Annotated[
        Optional[bool],
        Field(
            default=None,
            description="Keep VAE on CPU (sd-server `--vae-on-cpu`).  Drops ~250 MiB of VRAM at the "
            "cost of a slow VAE decode round-trip; rarely needed alongside --vae-tiling.",
        ),
    ] = None
    """Keep VAE on CPU (sd-server)."""

    clip_on_cpu: Annotated[
        Optional[bool],
        Field(
            default=None,
            description="Keep the text encoder / CLIP on CPU (sd-server `--clip-on-cpu`).  Drops the "
            "text-encode weights out of VRAM; slow but useful when the VRAM budget is tight.",
        ),
    ] = None
    """Keep text encoder on CPU (sd-server)."""

    img_cfg_scale: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Image-conditioning guidance scale for sd-server's edit / instruct-pix2pix "
            "pipelines (`--img-cfg-scale`).  Controls how strongly the model preserves the input "
            "image vs. follows the text prompt.  sd-server's /sdapi/v1/img2img request body has "
            "no field for this, so it MUST be pinned at sd-server launch.  Defaults to the value "
            "of cfg_scale when unset, which on Qwen-Image-Edit causes the model to over-preserve "
            "the input and ignore structural prompts ('remove the background' produces a near-"
            "identical image).  Set to 1.0–1.5 to let txt_cfg dominate.",
            ge=0.0,
        ),
    ] = None
    """sd-server --img-cfg-scale (image-preservation pressure for Qwen-Image-Edit etc.)."""

    # ------------------------------------------------------------------
    # Hunyuan3D-2.1 / img23d + img23d_part inference knobs.
    #
    # These travel in the runner's pipeline payload (``POST
    # /v1/pipelines/img23d/run``) — they are NOT CLI flags.  Defined
    # here so per-model defaults can live in ``.models.yaml`` and the
    # api layer doesn't have to thread every knob through its
    # request/response schemas.  Per-request overrides win over
    # yaml defaults via the standard precedence
    # (explicit kwarg > model.parameters > pipeline-baked default).
    # ------------------------------------------------------------------
    num_inference_steps: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Diffusion sampling steps for Hunyuan3D-2.1 (and XPart).  "
            "Hunyuan3D-2.1 default 30 in the pipeline; XPart hardcodes 50 "
            "internally and ignores this field for now.  More steps = "
            "sharper SDF surface but linearly more wall-clock time.",
            ge=1,
        ),
    ] = None
    """Hunyuan3D diffusion steps."""

    guidance_scale: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Classifier-free guidance scale for diffusion-based 3D models "
            "(Hunyuan3D-2.1, XPart, and future shape-diffusion backbones).  Distinct "
            "from SD's ``cfg_scale`` field — that one maps to sd-server's wire-protocol "
            "name; this one matches the diffusers / Hunyuan3D convention.  Pipeline "
            "default 5.5.  Lower values (3-5) give cleaner geometry but less prompt "
            "fidelity; higher (7-10) sharpens features at the cost of artefacts.",
            ge=0.0,
        ),
    ] = None
    """Classifier-free guidance scale for 3D diffusion models (distinct from SD's cfg_scale)."""

    octree_resolution: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Marching-cubes octree resolution for Hunyuan3D-2.1 mesh extraction.  "
            "Pipeline default 512.  256 is ~4× faster and noticeably blockier; 1024 is "
            "~4× slower and rarely yields visible improvement.  Bigger numbers also need "
            "linearly more VRAM during the MC chunk sweep.",
            ge=64,
        ),
    ] = None
    """Hunyuan3D marching-cubes resolution."""

    mc_level: Annotated[
        Optional[float],
        Field(
            default=None,
            description="SDF threshold for marching cubes — the isosurface level where the "
            "implicit function is sampled.  Hunyuan3D-2.1 default ``-1/512 ≈ -0.00195``, "
            "which slightly biases the surface OUTSIDE the SDF zero-crossing and tends to "
            "include a flat 'ghost platform' below the subject when the input image had "
            "any non-transparent pixels in the lower region.  Set to ``0.0`` for the "
            "true zero-crossing (tighter geometry, no ghost platform, but small surface "
            "features can pinch off).  Negative values thicken the mesh; positive values "
            "carve into it.",
        ),
    ] = None
    """Hunyuan3D marching-cubes SDF threshold (tune to suppress ghost platforms)."""

    box_v: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Padding factor for the marching-cubes bounding volume.  Pipeline "
            "default 1.01 (1% padding around the subject's bbox).  Set to ``1.0`` for "
            "exact bbox — this is the other lever for removing ghost-platform artefacts "
            "since 1% padding sometimes catches a sliver of the floor plane.",
            ge=1.0,
            le=1.5,
        ),
    ] = None
    """Hunyuan3D bounding-volume padding factor."""

    num_chunks: Annotated[
        Optional[int],
        Field(
            default=None,
            description="Marching-cubes chunk size for Hunyuan3D-2.1.  Pipeline default "
            "400000.  Smaller = less VRAM during MC at the cost of more chunk overhead; "
            "200000 is a safe halving when the SD-server is competing for memory.",
            ge=10000,
        ),
    ] = None
    """Hunyuan3D MC chunk size."""

    model_config = ConfigDict(extra="ignore")
