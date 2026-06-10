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


async def _can_proceed(metadata) -> bool:
    """Check whether a queued request can proceed (slot available).

    USER requests always proceed — the runner-side slot LRU pins each
    session to a slot and llama.cpp queues per-slot internally, so
    admitting a user request just routes it to its slot's local queue.
    Blocking here only adds latency.

    SCHEDULED/SYSTEM requests respect a per-model scheduled cap (one
    fewer than --parallel) so they can't starve interactive users, and
    additionally consult ``check_slot_availability`` to avoid kicking
    off a fresh batch when nothing is ready.
    """
    if not metadata.model_id:
        return True
    from models.request_priority_metadata import RequestSource

    if metadata.source in (RequestSource.SCHEDULED, RequestSource.SYSTEM):
        # Gate scheduled/system work on the REAL idle-slot state, not a fixed
        # reserve. The old `scheduled_cap = parallel - 1` reserved a slot for
        # users UNCONDITIONALLY — so a cron request blocked (and aged in the
        # queue) even when the server was idle and no user was waiting, wasting
        # capacity. Users don't need that reserve: USER requests are always
        # admitted (below) and the runner's per-session slot LRU pins each user
        # session to a slot regardless. check_slot_availability is the correct
        # gate — it admits when a slot is actually idle (or a runner has VRAM to
        # start a server) and self-limits to the real slot count, so scheduled
        # work uses idle servers instead of waiting on a phantom reserve.
        # (_active_counts is still tracked in _on_release/_on_complete for
        # observability but no longer gates admission.)
        try:
            return await runner_client.check_slot_availability(metadata.model_id)
        except Exception:
            return True

    # USER source: always admit (slot LRU + HIGH priority protect interactive turns).
    return True


def _on_release(metadata) -> None:
    """Called when a request is released from the queue to execute."""
    if metadata.model_id:
        _active_counts[metadata.model_id] += 1

    if metadata.session_id:
        state = get_session(metadata.session_id)
        is_new_session = state is None
        if is_new_session:
            states = get_session_state()
            state = SessionState()
            state.model_id = metadata.model_id or "unknown"
            state.source = metadata.source.value if metadata.source else "user"
            state.start_time = time.monotonic()
            state.turn_count = 0
            states[metadata.session_id] = state
        # Always update last_activity so the stale-cleanup uses the most
        # recent turn, not the first.  This keeps long-running sessions
        # alive in the registry across many turns.
        assert state is not None  # narrowing for type-checker
        state.last_activity = time.monotonic()
        state.turn_count += 1

        # Increment the gauge ONLY on first observation of a session_id.
        # Subsequent turns of the same session must not bump the count —
        # this was the source of the misleading "10 active sessions"
        # number that drifted upward whenever a turn aborted before
        # _on_complete fired (OOM crashes, etc.).  Counterpart dec() now
        # lives in the stale-cleanup, when the session is actually
        # removed from the registry.
        if is_new_session:
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
    # NOTE: active_sessions is no longer dec'd here.  It's a per-session
    # gauge driven by the stale-cleanup loop, not a per-turn counter.


async def _cleanup_stale_sessions(stale_timeout: float = 300.0) -> None:
    """Background task to detect inactive sessions and observe metrics.

    A session is considered stale when it has been inactive (no turns)
    for ``stale_timeout`` seconds.  Uses ``last_activity`` rather than
    ``start_time`` so long-running sessions stay tracked across many
    turns; only sustained silence triggers cleanup.

    On cleanup we observe the session-duration and session-turns
    histograms AND decrement the ``active_sessions`` gauge — this is
    the counterpart to the inc() in :func:`_on_release` so the gauge
    accurately reflects the number of unique session_ids currently
    tracked in the registry.
    """
    try:
        from middleware.api_metrics import session_duration_seconds, session_turns_total
    except ImportError:
        session_duration_seconds = None
        session_turns_total = None

    try:
        from middleware.api_metrics import active_sessions
    except ImportError:
        active_sessions = None

    while True:
        await asyncio.sleep(30)
        now = time.monotonic()
        all_states = get_session_state()
        # Inactive = no turn in the last ``stale_timeout`` seconds.  Fall
        # back to start_time if last_activity wasn't recorded (legacy
        # SessionState objects from before this field existed).
        stale_ids = [
            sid
            for sid, state in all_states.items()
            if now - getattr(state, "last_activity", state.start_time) > stale_timeout
            and state.turn_count > 0
        ]
        for sid in stale_ids:
            state = remove_session(sid)
            if not state:
                continue
            if session_duration_seconds:
                session_duration_seconds.labels(
                    model_id=state.model_id,
                    source=state.source,
                ).observe(now - state.start_time)
            if session_turns_total:
                session_turns_total.labels(
                    model_id=state.model_id,
                    source=state.source,
                ).observe(state.turn_count)
            if active_sessions:
                active_sessions.labels(
                    model_id=state.model_id,
                    source=state.source,
                ).dec()


def wire_priority_queue() -> None:
    """Register all callbacks with the priority queue and start cleanup task."""
    priority_queue.set_can_proceed_callback(_can_proceed)
    priority_queue.set_session_callbacks(_on_release, _on_complete)
    asyncio.create_task(_cleanup_stale_sessions())
    logger.info("Resource-aware priority queue callbacks wired up")
