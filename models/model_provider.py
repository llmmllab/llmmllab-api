

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class ModelProvider(str, Enum):
    """Provider / runtime of the model (e.g., 'llama.cpp', 'hf', 'hugging face', 'openai', 'stable-diffusion.cpp', 'anthropic')"""
    LLAMA_CPP = 'llama_cpp'
    HF = 'hf'
    HUGGING_FACE = 'hugging_face'
    OPENAI = 'openai'
    STABLE_DIFFUSION_CPP = 'stable_diffusion_cpp'
    ANTHROPIC = 'anthropic'
    # In-process pipeline: not a subprocess-backed server (llama.cpp /
    # sd-server) but a Python pipeline running inside the runner.  Loads
    # lazily on first request and is reachable at
    # ``POST /v1/pipelines/<name>/run``.  /v1/server/create rejects
    # acquire-attempts for these because there's nothing to acquire.
    IN_PROCESS = 'in_process'
    OTHER = 'other'