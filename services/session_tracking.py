"""
In-flight session task registry used to cancel running completions by id.

The registry maps ``session_id`` to the ``asyncio.Task`` currently servicing
that session.  Routers register a session at the start of a completion (via
the priority-queue context manager) and unregister it when the request exits.
The admin router can call :func:`cancel_session` to interrupt an in-flight
session, typically when the client disconnects or the user hits "stop".
"""

import asyncio

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="session_tracking")

# Maps session_id -> asyncio.Task servicing that session.  Module-level state
# is appropriate here because the registry is global to the process and there
# is no per-instance configuration.
_in_flight_tasks: dict[str, asyncio.Task] = {}


async def register_session_task(session_id: str, task: asyncio.Task) -> None:
    """Register an in-flight task for ``session_id``.

    Overwrites any existing entry for the same session_id (the previous task
    is assumed to have already completed or to be no longer reachable).
    """
    _in_flight_tasks[session_id] = task


async def unregister_session_task(session_id: str) -> None:
    """Remove the in-flight task for ``session_id`` if present."""
    _in_flight_tasks.pop(session_id, None)


async def cancel_session(session_id: str) -> bool:
    """Cancel an in-flight task by session_id. Returns True if found."""
    task = _in_flight_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()
        return True
    return False
