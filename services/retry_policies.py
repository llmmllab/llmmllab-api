"""
Retry orchestration for completion workflow execution.

This module owns the outer-layer retry loops that wrap
``CompletionService._build_and_run``:

* :func:`stream_with_connection_retry` — catches connection-level errors
  (``APIConnectionError``, ``RemoteProtocolError``, ``ConnectError``) and
  retries with a refreshed model map and a fresh server handle.

The inner stale-server retry (which depends on the
``CompletionService._run_workflow`` / ``CompletionService.build_workflow``
static methods and the module-level ``create_initial_state`` symbol that
tests patch) remains in :mod:`services.completion_service` so those test
patches continue to work.

Both layers MUST let ``asyncio.CancelledError`` re-raise immediately — never
retry on cancellation.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Awaitable, Callable, Optional, Union

from graph.state import ServerToolEvent
from httpx import ConnectError, RemoteProtocolError
from models.chat_response import ChatResponse
from models.message import Message
from models.model_parameters import ModelParameters
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="retry_policies")

# Connection-level errors that indicate the runner/server is unreachable.
# These should trigger a server-handle refresh, not an empty-response retry.
_CONNECTION_ERRORS = (RemoteProtocolError, ConnectError)


def _looks_like_cold_start(exc: Exception) -> bool:
    """True if *exc* reads like a model-still-loading cold start.

    Belt-and-braces fallback for paths that surface the runner's 503 as a
    bare ``RuntimeError`` (e.g. ``acquire_server``'s "No healthy runner …
    Last error: …503… still loading" message) rather than the typed
    :class:`ColdStartError`.  Keeps the cold-start retry working even if a
    new code path forgets to raise the typed error.
    """
    body = str(exc).lower()
    return any(
        marker in body
        for marker in (
            "still loading",
            "runner busy",
            "busy starting the model",
            "starting the model",
            "model server is still loading",
        )
    )


async def stream_with_connection_retry(
    inner: Callable[..., AsyncIterator[Union[ChatResponse, ServerToolEvent]]],
    *,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    tool_choice: str | None,
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
    max_retries: int,
    backoff_base: float,
    refresh_model_map: Callable[[], Awaitable[None]],
    cold_start_retries: int | None = None,
    cold_start_backoff: float | None = None,
    disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
    """Run ``inner`` with retry on connection-level and cold-start errors.

    ``inner`` must accept the keyword arguments above and yield
    ``ChatResponse | ServerToolEvent``.

    Two independent retry budgets are tracked:

    * **Connection errors** (``APIConnectionError`` + the httpx connection
      errors): retried up to ``max_retries`` times with linear backoff
      (``backoff_base * attempt``).
    * **Cold-start errors** (:class:`graph.errors.ColdStartError`, or a
      RuntimeError whose text reads "still loading"/"Runner busy"): the
      target model's server is still loading (~45-90 s).  Retried up to
      ``cold_start_retries`` times with a FIXED, longer
      ``cold_start_backoff`` wait — model load takes far longer than a
      network blip, so a fixed ~20 s pace covers a cold start without the
      thrash a short exponential backoff would cause.  This is what stops a
      transient cold-start 503 from reaching the client; the wait gives the
      runner time to finish loading, then we re-acquire a (now-warm) server.

    The two budgets are counted separately so a slow cold start doesn't burn
    the connection-error retries, and vice versa.

    ``disconnected`` is the optional client-liveness predicate; it is forwarded
    to ``inner`` (``_build_and_run``) so the agent's own retry loop can abort
    promptly when the streaming client has hung up.  ``None`` (default) is a
    no-op.

    ``asyncio.CancelledError`` is never retried — it always re-raises
    immediately.
    """
    from openai import APIConnectionError
    from graph.errors import ColdStartError

    if cold_start_retries is None:
        from config import COLD_START_RETRIES

        cold_start_retries = COLD_START_RETRIES
    if cold_start_backoff is None:
        from config import COLD_START_BACKOFF_SEC

        cold_start_backoff = COLD_START_BACKOFF_SEC

    conn_attempts = 0
    cold_attempts = 0

    while True:
        try:
            async for event in inner(
                user_id,
                messages,
                model_name,
                workflow_type,
                conversation_id,
                client_tools,
                tool_choice,
                server_tool_names,
                model_parameters,
                disconnected=disconnected,
            ):
                yield event
            return
        except asyncio.CancelledError:
            raise  # Never retry on cancellation
        except ColdStartError as e:
            if cold_attempts < cold_start_retries:
                cold_attempts += 1
                logger.warning(
                    "Model still loading (cold start), waiting before retry "
                    "with fresh server handle",
                    extra={
                        "model_id": getattr(e, "model_id", model_name),
                        "attempt": cold_attempts,
                        "cold_start_retries": cold_start_retries,
                        "backoff": cold_start_backoff,
                    },
                )
                await asyncio.sleep(cold_start_backoff)
                await refresh_model_map()
                continue
            logger.error(
                "Cold-start retries exhausted — surfacing error",
                extra={
                    "model_id": getattr(e, "model_id", model_name),
                    "cold_start_retries": cold_start_retries,
                },
            )
            raise
        except (APIConnectionError, *_CONNECTION_ERRORS) as e:
            if conn_attempts < max_retries:
                conn_attempts += 1
                logger.warning(
                    "Connection error during workflow execution, "
                    "retrying with fresh server handle",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "attempt": conn_attempts,
                        "max_retries": max_retries,
                    },
                )
                await asyncio.sleep(backoff_base * conn_attempts)
                await refresh_model_map()
                continue
            raise
        except RuntimeError as e:
            # Fallback: a cold-start 503 that reached us as a bare
            # RuntimeError (e.g. acquire_server's "No healthy runner …
            # Last error: …still loading" message) instead of the typed
            # ColdStartError.  Treat it with the cold-start budget.
            if _looks_like_cold_start(e) and cold_attempts < cold_start_retries:
                cold_attempts += 1
                logger.warning(
                    "Cold-start-shaped RuntimeError, waiting before retry "
                    "with fresh server handle",
                    extra={
                        "error": str(e),
                        "attempt": cold_attempts,
                        "cold_start_retries": cold_start_retries,
                        "backoff": cold_start_backoff,
                    },
                )
                await asyncio.sleep(cold_start_backoff)
                await refresh_model_map()
                continue
            raise
