"""Admin endpoint to cancel sessions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from middleware.auth import is_admin

router = APIRouter(prefix="/internal/session", tags=["session-admin"])


class CancelResponse(BaseModel):
    queued_cancelled: int
    in_flight_cancelled: bool
    session_removed: bool


@router.post("/{session_id}/cancel", response_model=CancelResponse)
async def cancel_session(request: Request, session_id: str):
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")

    queued_cancelled = 0
    try:
        from services.priority_queue import priority_queue

        queued_cancelled = await priority_queue.cancel_by_session_id(session_id)
    except Exception:
        pass

    in_flight_cancelled = False
    try:
        from services.completion_service import cancel_session as cancel_in_flight

        in_flight_cancelled = await cancel_in_flight(session_id)
    except Exception:
        pass

    session_removed = False
    state = None
    try:
        from services.session_registry import remove_session

        state = remove_session(session_id)
        session_removed = state is not None
    except Exception:
        pass

    if session_removed and state:
        try:
            from middleware.api_metrics import active_sessions

            active_sessions.labels(
                model_id=state.model_id,
                source=state.source,
            ).dec()
        except Exception:
            pass

    return CancelResponse(
        queued_cancelled=queued_cancelled,
        in_flight_cancelled=in_flight_cancelled,
        session_removed=session_removed,
    )
