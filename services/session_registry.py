"""Central session state registry.

Shared between app.py callbacks and admin cancel endpoint.
"""

from __future__ import annotations

from typing import Optional


class SessionState:
    __slots__ = ("model_id", "source", "start_time", "turn_count")
    model_id: str
    source: str
    start_time: float
    turn_count: int


_session_state: dict[str, SessionState] = {}


def get_session_state() -> dict[str, SessionState]:
    return _session_state


def get_session(session_id: str) -> Optional[SessionState]:
    return _session_state.get(session_id)


def remove_session(session_id: str) -> Optional[SessionState]:
    return _session_state.pop(session_id, None)
