"""Text-to-speech via Piper HTTP server.

Calls the Piper TTS service directly (not through the runner pool).
Piper is a standalone K8s deployment at PIPER_TTS_URL.
"""

from __future__ import annotations

from typing import Optional

import httpx

from config import PIPER_TTS_URL
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="tts_service")


class TTSError(Exception):
    """Raised when Piper returns a non-2xx response."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


async def synthesize(
    text: str,
    *,
    response_format: str = "wav",
    speed: float = 1.0,
) -> bytes:
    """Synthesize speech from text via Piper.

    Returns raw WAV audio bytes.
    Piper always outputs WAV; other response_format values are accepted
    but not transcoded (document as WAV-only for now).
    """
    if not text.strip():
        raise TTSError("Text input is empty", status_code=400)

    # Piper HTTP server accepts raw text body on POST /
    url = PIPER_TTS_URL.rstrip("/") + "/"
    logger.debug(f"Synthesizing TTS via Piper at {url}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                url,
                content=text.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )

        if resp.status_code != 200:
            raise TTSError(
                f"Piper returned {resp.status_code}: {resp.text[:512]}",
                status_code=resp.status_code,
            )

        return resp.content

    except httpx.TimeoutException:
        raise TTSError("Piper TTS request timed out", status_code=504)
    except httpx.ConnectError as e:
        raise TTSError(f"Failed to connect to Piper: {e}", status_code=503)
