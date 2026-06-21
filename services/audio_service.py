"""Audio transcription via whisper-server on a runner.

Acquires a whisper-server subprocess on an available runner, POSTs audio
files to its ``/inference`` endpoint, and returns the transcript.

Follows the same pattern as ``image_service.py``: acquire server →
proxy request → release (with optional auto-shutdown).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from services.runner_client import RunnerClient, ServerHandle, runner_client as _default_client
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="audio_service")


# ------------------------------------------------------------------- #
# Exceptions
# ------------------------------------------------------------------- #

class AudioServiceError(Exception):
    """Raised when whisper-server returns a non-2xx response."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


# ------------------------------------------------------------------- #
# Results
# ------------------------------------------------------------------- #

@dataclass
class TranscriptionResult:
    """Unpacked response from whisper-server /inference."""

    text: str
    language: str = ""
    duration: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)  # verbose_json only


# ------------------------------------------------------------------- #
# Service
# ------------------------------------------------------------------- #

class AudioService:
    """Bridge between the API routers and runner-side whisper-server."""

    def __init__(self, runner_client: Optional[RunnerClient] = None):
        self._client = runner_client or _default_client

    # -- public API --------------------------------------------------

    async def transcribe(
        self,
        file_content: bytes,
        filename: str,
        *,
        model_id: str = "whisper-small-en",
        language: Optional[str] = None,
        temperature: float = 0,
        translate: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio file via whisper-server on a runner.

        Parameters
        ----------
        file_content:
            Raw audio bytes (wav, mp3, flac, …).
        filename:
            Filename with extension (used to set Content-Type).
        model_id:
            Runner model id (default: ``whisper-small-en``).
        language:
            Force language (e.g. ``"en"``).  ``None`` → auto-detect.
        temperature:
            Sampling temperature (0 = greedy).
        translate:
            If True, return English translation instead of transcript.

        Returns
        -------
        TranscriptionResult
        """
        handle = await self._client.acquire_server(model_id)
        try:
            result = await self._do_transcribe(
                handle,
                file_content,
                filename,
                language=language,
                temperature=temperature,
                translate=translate,
            )
            return result
        finally:
            await self._release_server(handle)

    # -- internals ---------------------------------------------------

    async def _do_transcribe(
        self,
        handle: ServerHandle,
        file_content: bytes,
        filename: str,
        *,
        language: Optional[str],
        temperature: float,
        translate: bool,
    ) -> TranscriptionResult:
        """POST audio to whisper-server /inference endpoint."""

        url = f"{handle.base_url}/inference"

        # Build multipart form data — whisper-server expects:
        #   file: audio file
        #   language: (optional) forced language
        #   temperature: (optional) sampling temp
        #   translate: (optional) translate to English
        form_data: dict[str, str] = {"temperature": str(temperature)}
        if language:
            form_data["language"] = language
        if translate:
            form_data["translate"] = "true"

        async with httpx.AsyncClient(timeout=120.0) as http:
            response = await http.post(
                url,
                files={
                    "file": (filename, file_content, self._content_type(filename))
                },
                data=form_data,
            )

        if response.status_code != 200:
            raise AudioServiceError(
                f"whisper-server /inference returned {response.status_code}: "
                f"{response.text[:512]}",
                status_code=response.status_code,
            )

        body = response.json()
        return TranscriptionResult(
            text=body.get("text", "").strip(),
            language=body.get("language", ""),
            segments=body.get("segments", []),
        )

    async def _release_server(self, handle: ServerHandle) -> None:
        """Release the server handle back to the runner."""
        try:
            await self._client.release_server(handle)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"release_server failed: {e}")

    @staticmethod
    def _content_type(filename: str) -> str:
        """Guess MIME type from file extension."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "m4a": "audio/x-m4a",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
            "mp4": "video/mp4",
            "mpeg": "audio/mpeg",
            "mpga": "audio/mpeg",
        }
        return types.get(ext, "application/octet-stream")
