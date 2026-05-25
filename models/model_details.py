from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat


class ModelDetails(BaseModel):
    """ModelDetails contains additional information about a model"""

    parent_model: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Identifier of the parent model if this is a derivative",
        ),
    ] = None
    """Identifier of the parent model if this is a derivative"""
    format: Annotated[
        str, Field(..., description="Format of the model file (e.g., gguf)")
    ]
    """Format of the model file (e.g., gguf)"""
    gguf_file: Annotated[
        Optional[str],
        Field(default=None, description="Path to the GGUF file if applicable"),
    ] = None
    """Path to the GGUF file if applicable"""
    clip_model_path: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Path to the CLIP model file for multimodal models (e.g., mmproj file)",
        ),
    ] = None
    """Path to the CLIP model file for multimodal models (e.g., mmproj file)"""
    family: Annotated[
        str,
        Field(
            ..., description="Primary model family this belongs to (e.g., llama, phi3)"
        ),
    ]
    """Primary model family this belongs to (e.g., llama, phi3)"""
    families: Annotated[
        List[str], Field(..., description="All model families this belongs to")
    ]
    """All model families this belongs to"""
    parameter_size: Annotated[
        str, Field(..., description="Size of model parameters (e.g., '7.2B')")
    ]
    """Size of model parameters (e.g., '7.2B')"""
    quantization_level: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Level of quantization applied to the model (e.g., 'Q4_0')",
        ),
    ] = None
    """Level of quantization applied to the model (e.g., 'Q4_0')"""
    dtype: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Data type of the model (e.g., 'float16', 'bfloat16')",
        ),
    ] = None
    """Data type of the model (e.g., 'float16', 'bfloat16')"""
    precision: Annotated[
        Optional[Literal["fp32", "fp16", "bf16", "int8", "int4", "int2", "int1"]],
        Field(
            default=None, description="Precision of the model (e.g., 'fp16', 'bf16')"
        ),
    ] = None
    """Precision of the model (e.g., 'fp16', 'bf16')"""
    specialization: Annotated[
        Optional[
            Literal["LoRA", "Embedding", "TextToImage", "ImageToImage", "Audio", "Text"]
        ],
        Field(
            default=None,
            description="Specialization of the model (e.g., 'LoRA', 'Embedding', 'TextToImage', 'Audio', 'Vision')",
        ),
    ] = None
    """Specialization of the model (e.g., 'LoRA', 'Embedding', 'TextToImage', 'Audio', 'Vision')"""
    description: Annotated[
        Optional[str], Field(default=None, description="Description of the model")
    ] = None
    """Description of the model"""
    weight: Annotated[
        Optional[float],
        Field(default=None, description="Weight of the model (applies to LoRA models)"),
    ] = None
    """Weight of the model (applies to LoRA models)"""
    size: Annotated[int, Field(..., description="Size of the model in bytes")]
    """Size of the model in bytes"""
    original_ctx: Annotated[
        int, Field(..., description="Original context window size of the model")
    ]
    """Original context window size of the model"""
    n_layers: Annotated[
        Optional[int], Field(default=None, description="Number of layers in the model")
    ] = None
    """Number of layers in the model"""
    hidden_size: Annotated[
        Optional[int], Field(default=None, description="Hidden size of the model")
    ] = None
    """Hidden size of the model"""
    n_heads: Annotated[
        Optional[int],
        Field(default=None, description="Number of attention heads in the model"),
    ] = None
    """Number of attention heads in the model"""
    n_kv_heads: Annotated[
        Optional[int],
        Field(default=None, description="Number of key-value heads in the model"),
    ] = None
    """Number of key-value heads in the model"""
    clip_model_size: Annotated[
        Optional[int],
        Field(
            default=None, description="Size of the CLIP model in bytes, if applicable"
        ),
    ] = None
    """Size of the CLIP model in bytes, if applicable"""

    # ---------------------------------------------------------------
    # stable-diffusion.cpp specific paths.
    #
    # SD models — unlike LLMs — typically ship as multiple files:
    #   * diffusion_model_path:  the main UNet/DiT (.gguf)
    #   * vae_path:              VAE weights (.safetensors)
    #   * text_encoder_path:     CLIP-L / T5 / Qwen-VL etc. text encoder
    #   * clip_g_path:           secondary text encoder (SDXL / SD3)
    #
    # All four are optional; only the ones the chosen model needs are
    # forwarded to sd-server's CLI as --diffusion-model, --vae, --llm
    # (or --clip_l / --t5xxl based on which slot is populated), and
    # --clip_g.
    # ---------------------------------------------------------------
    diffusion_model_path: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Path to the diffusion model (.gguf) for stable-diffusion.cpp",
        ),
    ] = None
    """Path to the diffusion model file for stable-diffusion.cpp"""
    vae_path: Annotated[
        Optional[str],
        Field(
            default=None, description="Path to the VAE weights for stable-diffusion.cpp"
        ),
    ] = None
    """Path to the VAE file for stable-diffusion.cpp"""
    text_encoder_path: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Path to the text encoder (LLM/CLIP-L/T5XXL) for stable-diffusion.cpp",
        ),
    ] = None
    """Path to the text encoder file"""
    text_encoder_kind: Annotated[
        Optional[Literal["llm", "clip_l", "t5xxl"]],
        Field(
            default="llm",
            description="Which sd-server flag receives text_encoder_path (--llm/--clip_l/--t5xxl)",
        ),
    ] = "llm"
    """Determines which sd-server flag receives the text encoder"""
    clip_g_path: Annotated[
        Optional[str],
        Field(default=None, description="Secondary CLIP-G text encoder (for SDXL/SD3)"),
    ] = None
    """Secondary CLIP-G text encoder path"""

    llm_vision_path: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Qwen2.5-VL visual tower / vision encoder for Qwen-Image-Edit 2509+ (sd-server --llm_vision). Required for instruction-following edits; without it sd-server falls back to the plain Qwen-Image txt2img pipeline and ignores the prompt's editing intent.",
        ),
    ] = None
    """Qwen2.5-VL visual encoder (sd-server --llm_vision) — required for Qwen-Image-Edit 2509+."""

    model_config = ConfigDict(extra="ignore")
