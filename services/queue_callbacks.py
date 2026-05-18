"""Priority queue callback wiring and session cleanup.

Extracted from app.py lifespan to improve readability and testability.
"""

import asyncio
import time
from collections import defaultdict

from services.runner_client import runner_client
from services.session_registry import (
    SessionState,
    get_session,
    get_session_state,
    remove_session,
)
from services.priority_queue import priority_queue
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="queue_callbacks")

_active_counts: dict[str, int] = defaultdict(int)
_parallel_cache: dict[str, int] = {}


async def _get_parallel_for_model(model_id: str) -> int:
    """Get the parallel slot count for a model, cached."""
    if model_id in _parallel_cache:
        return _parallel_cache[model_id]
    try:
        models = await runner_client.list_models()
        for m in models:
            if m.model == model_id or m.name == model_id:
                parallel = (m.parameters.parallel if m.parameters else None) or 4
                _parallel_cache[model_id] = parallel
                return parallel
    except Exception:
        pass
    _parallel_cache[model_id] = 4
    return 4


async def _can_proceed(metadata) -> bool:
    """Check whether a queued request can proceed (slot available)."""
    if not metadata.model_id:
        return True
    from models.request_priority_metadata import RequestSource

    if metadata.source in (RequestSource.SCHEDULED, RequestSource.SYSTEM):
        parallel = await _get_parallel_for_model(metadata.model_id)
        scheduled_cap = max(1, parallel - 1)
        if _active_counts[metadata.model_id] >= scheduled_cap:
            return False
    try:
        return await runner_client.check_slot_availability(metadata.model_id)
    except Exception:
        return True


def _on_release(metadata) -> None:
    """Called when a request is released from the queue to execute."""
    if metadata.model_id:
        _active_counts[metadata.model_id] += 1

    if metadata.session_id:
        state = get_session(metadata.session_id)
        if state is None:
            states = get_session_state()
            state = SessionState()
            state.model_id = metadata.model_id or "unknown"
            state.source = metadata.source.value if metadata.source else "user"
            state.start_time = time.monotonic()
            state.turn_count = 0
            states[metadata.session_id] = state
        state.turn_count += 1

        try:
            from middleware.api_metrics import active_sessions

            active_sessions.labels(
                model_id=state.model_id,
                source=state.source,
            ).inc()
        except ImportError:
            pass


def _on_complete(metadata) -> None:
    """Called when a request finishes and is dequeued."""
    if metadata.model_id:
        _active_counts[metadata.model_id] -= 1

    if metadata.session_id:
        state = get_session(metadata.session_id)
        if state:
            try:
                from middleware.api_metrics import active_sessions

                active_sessions.labels(
                    model_id=state.model_id,
                    source=state.source,
                ).dec()
            except ImportError:
                pass


async def _cleanup_stale_sessions(stale_timeout: float = 300.0) -> None:
    """Background task to detect completed sessions and observe metrics."""
    try:
        from middleware.api_metrics import session_duration_seconds, session_turns_total
    except ImportError:
        session_duration_seconds = None
        session_turns_total = None

    while True:
        await asyncio.sleep(30)
        now = time.monotonic()
        all_states = get_session_state()
        stale_ids = [
            sid
            for sid, state in all_states.items()
            if now - state.start_time > stale_timeout and state.turn_count > 0
        ]
        for sid in stale_ids:
            state = remove_session(sid)
            if state and session_duration_seconds:
                session_duration_seconds.labels(
                    model_id=state.model_id,
                    source=state.source,
                ).observe(now - state.start_time)
            if state and session_turns_total:
                session_turns_total.labels(
                    model_id=state.model_id,
                    source=state.source,
                ).observe(state.turn_count)


def wire_priority_queue() -> None:
    """Register all callbacks with the priority queue and start cleanup task."""
    priority_queue.set_can_proceed_callback(_can_proceed)
    priority_queue.set_session_callbacks(_on_release, _on_complete)
    asyncio.create_task(_cleanup_stale_sessions())
    logger.info("Resource-aware priority queue callbacks wired up")
