"""Audio transcription / translation / speech synthesis.

Transcription & translation are proxied to runner-side whisper-server.
Speech synthesis (TTS) calls the Piper TTS service directly.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse, Response
from typing import Optional

from models.openai.audio_response_format import AudioResponseFormat
from models.openai.create_speech_request import CreateSpeechRequest
from services.audio_service import AudioService, AudioServiceError
from services.runner_client import runner_client
from services.tts_service import synthesize, TTSError

router = APIRouter(prefix="/audio", tags=["Audio"])


def get_audio_service() -> AudioService:
    """Dependency: inject AudioService wired to the shared RunnerClient."""
    return AudioService(runner_client)


# -- helpers -----------------------------------------------------------

_ALLOWED_EXTENSIONS = frozenset(
    ["wav", "mp3", "flac", "m4a", "ogg", "webm", "mp4", "mpeg", "mpga"]
)


def _validate_file(filename: Optional[str]):
    """Raise 400 if the file extension is not supported."""
    if not filename:
        raise HTTPException(
            status_code=400, detail="Missing filename"
        )
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )


def _format_response(
    result,
    response_format: Optional[AudioResponseFormat] = None,
):
    """Convert TranscriptionResult → OpenAI-compatible response."""
    fmt = response_format or AudioResponseFormat.JSON

    if fmt == AudioResponseFormat.TEXT:
        return JSONResponse(content=result.text)

    if fmt == AudioResponseFormat.VERBOSE_JSON:
        return JSONResponse(
            content={
                "text": result.text,
                "language": result.language,
                "duration": result.duration,
                "segments": result.segments,
            }
        )

    # Default: json
    return JSONResponse(content={"text": result.text})


# -- endpoints ---------------------------------------------------------

@router.post("/transcriptions")
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[AudioResponseFormat] = Form(None),
    temperature: float = Form(0),
    audio_service: AudioService = Depends(get_audio_service),
):
    """Transcribe audio to text via runner-side whisper-server.

    Accepts audio files in wav, mp3, flac, m4a, ogg, webm, mp4, mpeg, mpga formats.
    Returns transcription in specified format (default: json).
    """
    try:
        _validate_file(file.filename)

        contents = await file.read()

        result = await audio_service.transcribe(
            file_content=contents,
            filename=file.filename or "audio.wav",
            model_id=model or "whisper-small-en",
            language=language,
            temperature=temperature,
            translate=False,
        )

        return _format_response(result, response_format)

    except HTTPException:
        raise
    except AudioServiceError as e:
        raise HTTPException(
            status_code=e.status_code or 502,
            detail=f"Transcription service error: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {e}"
        )


@router.post("/translations")
async def create_translation(
    file: UploadFile = File(..., description="Audio file to translate to English"),
    model: Optional[str] = Form(None),
    response_format: Optional[AudioResponseFormat] = Form(None),
    temperature: float = Form(0),
    audio_service: AudioService = Depends(get_audio_service),
):
    """Translate audio to English via runner-side whisper-server.

    Accepts audio files in any supported format, returns English transcription.
    """
    try:
        _validate_file(file.filename)

        contents = await file.read()

        result = await audio_service.transcribe(
            file_content=contents,
            filename=file.filename or "audio.wav",
            model_id=model or "whisper-small-en",
            language=None,
            temperature=temperature,
            translate=True,
        )

        return _format_response(result, response_format)

    except HTTPException:
        raise
    except AudioServiceError as e:
        raise HTTPException(
            status_code=e.status_code or 502,
            detail=f"Translation service error: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Translation failed: {e}"
        )


@router.post("/speech")
async def create_speech(
    request: CreateSpeechRequest,
):
    """Synthesize speech from text via Piper TTS.

    OpenAI-compatible endpoint. Accepts text input, returns audio.
    Piper outputs WAV format regardless of response_format request.
    """
    try:
        audio_bytes = await synthesize(
            text=request.input,
            response_format=request.response_format or "wav",
            speed=request.speed or 1.0,
        )
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
        )
    except TTSError as e:
        raise HTTPException(
            status_code=e.status_code or 502,
            detail=f"TTS service error: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Speech synthesis failed: {e}"
        )
