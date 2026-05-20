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
from typing import Awaitable, Callable, Union

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
) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
    """Run ``inner`` with retry on connection-level errors.

    ``inner`` must accept the keyword arguments above and yield
    ``ChatResponse | ServerToolEvent``.  On ``APIConnectionError`` or the
    httpx connection errors, the model map is refreshed and the call is
    retried up to ``max_retries`` times with linear backoff.

    ``asyncio.CancelledError`` is never retried — it always re-raises
    immediately.
    """
    from openai import APIConnectionError

    for attempt in range(max_retries + 1):
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
            ):
                yield event
            return
        except asyncio.CancelledError:
            raise  # Never retry on cancellation
        except (APIConnectionError, *_CONNECTION_ERRORS) as e:
            if attempt < max_retries:
                logger.warning(
                    "Connection error during workflow execution, "
                    "retrying with fresh server handle",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                    },
                )
                await asyncio.sleep(backoff_base * (attempt + 1))
                await refresh_model_map()
                continue
            raise
