"""
CompletionService — shared workflow execution, retry, and continuation logic.

This is the core service that all three LLM endpoints delegate to.  It owns:
  • Building and executing a composer workflow
  • Single-retry on empty responses
  • Nudge prompt when retries also fail
  • Tool-continuation check (model produced text but no tool calls)
  • Filtering server-tool calls from the final response

The service yields ``ChatResponse | ServerToolEvent`` objects — the routers
are responsible only for formatting those into the wire protocol (SSE chunks
in Anthropic/OpenAI format, raw JSON for llmmllab).
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional, Union

from composer_init import (
    compose_workflow,
    create_initial_state,
    execute_workflow,
    get_graph_builder,
)
from config import RUNNER_RETRIES, RUNNER_RETRY_BACKOFF_BASE
from graph.state import ServerToolEvent
from graph.workflows.factory import WorkFlowType
from httpx import ConnectError, RemoteProtocolError
from models.chat_response import ChatResponse
from models.message import Message, MessageContent, MessageContentType
from models.request_priority_metadata import (
    Priority,
    RequestPriorityMetadata,
    RequestSource,
)
from services.completion_state import CompletionResult, StreamAccumulator
from services.prompt_templates import (
    CONTINUATION_PROMPT,
    EMPTY_RESPONSE_NUDGE,
    TRUNCATION_CONTINUATION_PROMPT,
)
from services.response_handlers import (
    build_followup_messages,
    extract_text,
    set_result_response,
    update_stream_delta,
    update_stream_final,
)
from services.truncation import is_context_overflow, is_truncated
from utils.logging import llmmllogger

__all__ = ["CompletionService", "CompletionResult", "StreamAccumulator", "cancel_session"]

logger = llmmllogger.bind(component="completion_service")

# Connection-level errors that indicate the runner/server is unreachable.
# These should trigger a server-handle refresh, not an empty-response retry.
_CONNECTION_ERRORS = (RemoteProtocolError, ConnectError)

# In-flight task registry for session cancellation
_in_flight_tasks: dict[str, asyncio.Task] = {}


async def cancel_session(session_id: str) -> bool:
    """Cancel an in-flight task by session_id. Returns True if found."""
    task = _in_flight_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()
        return True
    return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

from middleware.api_metrics import (  # noqa: E402
    empty_response_retries_total,
    workflow_completions_total,
    workflow_duration_seconds,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from config import ENABLE_TOOL_CONTINUATION, STALE_SERVER_RETRIES

_CONTINUATION_ENABLED = ENABLE_TOOL_CONTINUATION


async def _resolve_model(model_name: str, user_id: str) -> str:
    """Resolve a model name, falling back to the user's default if unavailable.

    This centralises the fallback logic so every workflow gets consistent
    behaviour: if the requested model isn't on any runner, try the user's
    ``default_model`` before giving up.

    Returns
    -------
    str
        The resolved model ID (may be the original if no fallback exists).
    """
    try:
        from services.model_service import model_service

        return await model_service.resolve_default_model(model_name, user_id)
    except Exception:
        # If model_service is unavailable, return the original name
        return model_name


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CompletionService:
    """Shared workflow execution and resilience logic.

    All methods are static / stateless — the service exists as an
    organisational namespace, not a stateful singleton.
    """

    # ------------------------------------------------------------------
    # Core workflow helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_model_num_ctx(model_name: str) -> int | None:
        """Look up the model's context window (num_ctx) from the runner cache.

        Returns the ``original_ctx`` value from the model details, or ``None``
        if the model is not found or the value is unavailable.
        """
        try:
            from services import model_service  # noqa: F811

            model = await model_service.get_model_by_id(model_name)
            if model and model.details and model.details.original_ctx:
                return model.details.original_ctx
        except Exception as e:
            logger.debug(f"Failed to look up model num_ctx for {model_name}: {e}")
        return None

    @staticmethod
    async def build_workflow(
        user_id: str,
        model_name: str,
        workflow_type: WorkFlowType,
        client_tools: list | None = None,
        tool_choice: str | None = None,
        server_tool_names: set[str] | None = None,
    ):
        """Build a composer workflow and return (workflow, builder, server_url)."""
        builder = await get_graph_builder(workflow_type, user_id)
        workflow = await compose_workflow(
            user_id=user_id,
            builder=builder,
            model_name=model_name,
            client_tools=client_tools,
            tool_choice=tool_choice,
            server_tool_names=server_tool_names or None,
        )
        server_url = None
        if builder.server_handle:
            server_url = builder.server_handle.base_url
        return workflow, builder, server_url

    @staticmethod
    async def _run_workflow(
        initial_state,
        workflow,
        workflow_type: WorkFlowType,
    ) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
        """Execute a composed workflow and yield its events."""
        start = time.monotonic()
        finished = False
        async for event in execute_workflow(initial_state, workflow):
            yield event
            if not finished and isinstance(event, ChatResponse) and event.done:
                finished = True
                duration = time.monotonic() - start
                status = "success" if event.finish_reason != "error" else "error"
                workflow_completions_total.labels(
                    workflow_type=workflow_type.value, status=status
                ).inc()
                workflow_duration_seconds.labels(
                    workflow_type=workflow_type.value
                ).observe(duration)

    @staticmethod
    async def _build_and_run(
        user_id: str,
        messages: list[Message],
        model_name: str,
        workflow_type: WorkFlowType,
        conversation_id: int = 0,
        client_tools: list | None = None,
        tool_choice: str | None = None,
        server_tool_names: set[str] | None = None,
        _retry_count: int = 0,
    ) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
        """Build a composer workflow and yield its events.

        When a ``StaleServerError`` is raised (server handle was evicted by
        the runner), the stale handle is released, the model map is refreshed,
        and the workflow is retried with a fresh server. The number of retries
        is controlled by :data:`config.STALE_SERVER_RETRIES` (default: 1).

        Non-stale errors propagate immediately without retry.
        """
        from graph.errors import StaleServerError
        from services.runner_client import runner_client

        max_retries = STALE_SERVER_RETRIES
        model_name = await _resolve_model(model_name, user_id)
        builder = None
        stale_err: StaleServerError | None = None

        try:
            workflow, builder, _server_url = await CompletionService.build_workflow(
                user_id,
                model_name,
                workflow_type,
                client_tools,
                tool_choice,
                server_tool_names,
            )
            initial_state = await create_initial_state(
                user_id, conversation_id, builder, messages
            )
            async for event in CompletionService._run_workflow(
                initial_state, workflow, workflow_type
            ):
                yield event
        except asyncio.CancelledError:
            logger.info(
                "Workflow cancelled — releasing server handle in finally",
                extra={
                    "server_id": (
                        builder.server_handle.server_id
                        if builder and builder.server_handle
                        else None
                    ),
                },
            )
            raise
        except StaleServerError as e:
            # Capture the error and defer the retry decision until after
            # the finally block releases the (now stale) handle.  This
            # guarantees release fires on the stale path too, without
            # double-releasing.
            stale_err = e
        finally:
            # Always release the handle: success, error, or cancel.  The
            # underlying llama.cpp process is not killed — release is a
            # soft refcount decrement on the runner.  A follow-up request
            # for the same model will simply re-acquire the warm server.
            if builder and builder.server_handle:
                handle = builder.server_handle
                # Detach from the builder so any later code path can't
                # accidentally release twice.
                builder.server_handle = None
                try:
                    await runner_client.release_server(handle)
                    logger.debug(
                        "Released server handle on workflow exit",
                        extra={"server_id": handle.server_id},
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as release_err:
                    logger.warning(
                        "release_server failed in _build_and_run finally — handle discarded",
                        extra={
                            "server_id": handle.server_id,
                            "error": str(release_err),
                        },
                    )

        if stale_err is not None:
            if _retry_count >= max_retries:
                logger.error(
                    f"Stale server {stale_err.server_id} — retry exhausted, propagating error",
                )
                raise stale_err
            logger.warning(
                f"Stale server {stale_err.server_id} detected — re-acquiring "
                f"(attempt {_retry_count + 1}/{max_retries})",
            )
            # Purge the cached workflow whose ChatOpenAI(base_url=...) still
            # points at the dead server. Without this the next compose_workflow
            # call would return the same stale CompiledStateGraph and the
            # retry would re-raise StaleServerError forever.
            from composer_init import invalidate_workflow as _invalidate_workflow
            await _invalidate_workflow(user_id, model_name)
            await runner_client.refresh_model_map()
            async for event in CompletionService._build_and_run(
                user_id,
                messages,
                model_name,
                workflow_type,
                conversation_id,
                client_tools,
                tool_choice,
                server_tool_names,
                _retry_count=_retry_count + 1,
            ):
                yield event

    @staticmethod
    async def _build_and_run_with_retry(
        user_id: str,
        messages: list[Message],
        model_name: str,
        workflow_type: WorkFlowType,
        max_retries: int | None = None,
        conversation_id: int = 0,
        client_tools: list | None = None,
        tool_choice: str | None = None,
        server_tool_names: set[str] | None = None,
    ) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
        """Build and run with automatic server handle refresh on connection failure.

        When a request fails with a connection-level error (e.g. the runner
        restarted mid-request), this method catches the error, forces a model
        map refresh, and retries with a fresh server handle.  This prevents
        stale ``ServerHandle`` objects from pointing to dead servers.

        Parameters
        ----------
        max_retries:
            Maximum number of connection-error retries.
            Defaults to RUNNER_RETRIES from config (env: RUNNER_RETRIES).
        """
        from openai import APIConnectionError

        if max_retries is None:
            max_retries = RUNNER_RETRIES

        for attempt in range(max_retries + 1):
            try:
                async for event in CompletionService._build_and_run(
                    user_id,
                    messages,
                    model_name,
                    workflow_type,
                    conversation_id,
                    client_tools,
                    tool_choice,
                    server_tool_names,
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
                    await asyncio.sleep(RUNNER_RETRY_BACKOFF_BASE * (attempt + 1))
                    from services.runner_client import runner_client

                    await runner_client.refresh_model_map()
                    continue
                raise

    @staticmethod
    async def _collect_response(
        user_id: str,
        messages: list[Message],
        model_name: str,
        workflow_type: WorkFlowType,
        conversation_id: int,
        client_tools: list | None,
        tool_choice: str | None,
        server_tool_names: set[str] | None,
    ) -> Optional[ChatResponse]:
        """Run a workflow and return only the final ChatResponse."""
        async for event in CompletionService._build_and_run(
            user_id,
            messages,
            model_name,
            workflow_type,
            conversation_id,
            client_tools,
            tool_choice,
            server_tool_names,
        ):
            if isinstance(event, ServerToolEvent):
                continue
            if event.done and event.message:
                return event
        return None

    @staticmethod
    async def _stream_secondary_pass(
        acc: StreamAccumulator,
        user_id: str,
        messages: list[Message],
        model_name: str,
        workflow_type: WorkFlowType,
        conversation_id: int,
        client_tools: list | None,
        tool_choice: str | None,
        server_tool_names: set[str] | None,
        *,
        content_prefix: str = "",
    ) -> AsyncIterator[tuple[Union[ChatResponse, ServerToolEvent], StreamAccumulator]]:
        """Yield streaming events from a secondary pass, updating acc."""
        async for event in CompletionService._build_and_run(
            user_id,
            messages,
            model_name,
            workflow_type,
            conversation_id,
            client_tools,
            tool_choice,
            server_tool_names,
        ):
            if isinstance(event, ServerToolEvent):
                continue
            if event.done:
                update_stream_final(
                    acc,
                    event,
                    server_tool_names,
                    content_prefix=content_prefix,
                    accumulate_output_tokens=True,
                )
                continue
            update_stream_delta(acc, event)
            yield event, acc

    @staticmethod
    @asynccontextmanager
    async def _enqueue_and_wait(
        model_name: str,
        user_id: str,
        priority: Priority | None,
        max_queue_wait: float | None,
        source: RequestSource | None,
        session_id: str | None,
    ) -> AsyncIterator[str]:
        """Handle priority queue enqueue/dequeue lifecycle."""
        from config import PRIORITY_QUEUE_ENABLED
        from services.priority_queue import priority_queue

        queue_item = None
        effective_model = model_name
        effective_priority = priority if priority is not None else Priority.HIGH

        if PRIORITY_QUEUE_ENABLED:
            effective_model = await priority_queue.ensure_model_available(
                model_name, user_id
            )
            if effective_model != model_name:
                logger.info(
                    "Model resolved before enqueue",
                    extra={"original": model_name, "resolved": effective_model},
                )

            meta = RequestPriorityMetadata(
                source=source or RequestSource.USER,
                priority=effective_priority,
                user_id=user_id,
                model_id=effective_model,
                max_queue_wait=max_queue_wait,
                session_id=session_id,
            )
            queue_item, _ = await priority_queue.enqueue(meta)
            if session_id:
                try:
                    cur_task = asyncio.current_task()
                    if cur_task is not None:
                        _in_flight_tasks[session_id] = cur_task
                except RuntimeError:
                    pass

        try:
            yield effective_model
        finally:
            if session_id:
                _in_flight_tasks.pop(session_id, None)
            if queue_item is not None:
                await priority_queue.dequeue(queue_item)

    # ------------------------------------------------------------------
    # Streaming path  (yields raw events; router formats SSE)
    # ------------------------------------------------------------------

    @staticmethod
    async def stream_completion(
        user_id: str,
        messages: list[Message],
        model_name: str,
        workflow_type: WorkFlowType = WorkFlowType.IDE,
        conversation_id: int = 0,
        client_tools: list | None = None,
        tool_choice: str | None = None,
        server_tool_names: set[str] | None = None,
        priority: Priority | None = None,
        max_queue_wait: float | None = None,
        source: RequestSource | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[tuple[Union[ChatResponse, ServerToolEvent], StreamAccumulator]]:
        """Execute a workflow and yield ``(event, accumulator)`` pairs.

        The *accumulator* is mutated in-place as events arrive so that the
        router can inspect it at any time (e.g. to decide whether to start
        a text block).

        After the primary pass completes, the method transparently performs
        continuation and retry logic and continues yielding events from
        those secondary passes.
        """
        acc = StreamAccumulator()

        try:
            async with CompletionService._enqueue_and_wait(
                model_name,
                user_id,
                priority,
                max_queue_wait,
                source,
                session_id,
            ) as effective_model:
                model_name = effective_model

                async for event in CompletionService._build_and_run_with_retry(
                    user_id,
                    messages,
                    model_name,
                    workflow_type,
                    conversation_id=conversation_id,
                    client_tools=client_tools,
                    tool_choice=tool_choice,
                    server_tool_names=server_tool_names,
                ):
                    if isinstance(event, ServerToolEvent):
                        yield event, acc
                        continue
                    if event.done:
                        update_stream_final(
                            acc,
                            event,
                            server_tool_names,
                            include_prompt_tokens=True,
                        )
                        yield event, acc
                        continue
                    update_stream_delta(acc, event)
                    yield event, acc

                if (
                    is_truncated(acc.final_content or "", acc.finish_reason)
                    and not acc.has_tool_calls
                ):
                    accumulated_text = acc.final_content or ""
                    logger.info(
                        "Model response appears truncated — sending continuation prompt",
                        extra={
                            "content_len": len(accumulated_text),
                            "finish_reason": acc.finish_reason,
                            "content_preview": accumulated_text[-200:],
                        },
                    )
                    truncation_messages = build_followup_messages(
                        messages,
                        TRUNCATION_CONTINUATION_PROMPT,
                        assistant_text=accumulated_text,
                    )
                    async for event, acc in CompletionService._stream_secondary_pass(
                        acc,
                        user_id,
                        truncation_messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        "auto",
                        server_tool_names,
                        content_prefix=accumulated_text,
                    ):
                        yield event, acc

                if (
                    _CONTINUATION_ENABLED
                    and not acc.has_tool_calls
                    and client_tools
                    and (acc.has_content or acc.final_content)
                    and acc.finish_reason not in ("stop", "length")
                ):
                    accumulated_text = acc.final_content or ""
                    if accumulated_text:
                        logger.info(
                            "Model produced text without tool calls — sending single continuation check",
                            extra={
                                "content_len": len(accumulated_text),
                                "content_preview": accumulated_text[:200],
                            },
                        )
                        continuation_messages = (
                            build_followup_messages(
                                messages,
                                CONTINUATION_PROMPT,
                                assistant_text=accumulated_text,
                            )
                        )
                        async for event, acc in CompletionService._stream_secondary_pass(
                            acc,
                            user_id,
                            continuation_messages,
                            model_name,
                            workflow_type,
                            conversation_id,
                            client_tools,
                            "auto",
                            server_tool_names,
                        ):
                            yield event, acc

                if (
                    not acc.has_content
                    and not acc.has_tool_calls
                    and not acc.final_content
                    and not acc.is_error
                    and acc.finish_reason != "stop"
                ):
                    model_num_ctx = await CompletionService._get_model_num_ctx(model_name)
                    if is_context_overflow(
                        acc.input_tokens,
                        acc.finish_reason,
                        acc.output_tokens,
                        model_num_ctx=model_num_ctx,
                    ):
                        acc.context_overflow = True
                        logger.warning(
                            "Skipping retry — context likely exceeds model window",
                            extra={
                                "model": model_name,
                                "input_tokens": acc.input_tokens,
                                "output_tokens": acc.output_tokens,
                                "finish_reason": acc.finish_reason,
                            },
                        )
                    else:
                        # Before wasting a retry on the same (possibly stale)
                        # handle, probe every known runner's startup_epoch.
                        # Chat completions go through LangChain's ChatOpenAI
                        # directly, bypassing proxy_request's 404→StaleServerError
                        # path; a dead-handle scenario surfaces here as an
                        # empty response.  If any runner restarted, purge its
                        # handles and raise StaleServerError so the upper
                        # retry layer rebuilds the workflow with a fresh handle.
                        from graph.errors import StaleServerError
                        from services.runner_client import runner_client
                        purged = await runner_client.revalidate_runner_handles()
                        if purged > 0:
                            logger.warning(
                                "Empty response coincided with detected runner "
                                "restart — invalidating workflow and re-acquiring",
                                extra={
                                    "model": model_name,
                                    "purged_handles": purged,
                                },
                            )
                            raise StaleServerError(
                                f"runner restart detected via empty response ({purged} handles purged)"
                            )
                        logger.warning(
                            "Model produced empty response — retrying with same messages",
                            extra={"model": model_name},
                        )
                        empty_response_retries_total.inc()
                        async for event, acc in CompletionService._stream_secondary_pass(
                            acc,
                            user_id,
                            messages,
                            model_name,
                            workflow_type,
                            conversation_id,
                            client_tools,
                            tool_choice,
                            server_tool_names,
                        ):
                            yield event, acc

                        if (
                            not acc.has_content
                            and not acc.has_tool_calls
                            and not acc.final_content
                        ):
                            logger.warning(
                                "Retry also produced empty response — sending nudge prompt",
                                extra={"model": model_name},
                            )
                            nudge_messages = build_followup_messages(
                                messages,
                                EMPTY_RESPONSE_NUDGE,
                            )
                            async for event, acc in CompletionService._stream_secondary_pass(
                                acc,
                                user_id,
                                nudge_messages,
                                model_name,
                                workflow_type,
                                conversation_id,
                                client_tools,
                                "auto",
                                server_tool_names,
                            ):
                                yield event, acc
        except asyncio.CancelledError:
            logger.warning("Stream cancelled (client disconnect) — stopping retries")
            return

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    @staticmethod
    async def run_completion(
        user_id: str,
        messages: list[Message],
        model_name: str,
        workflow_type: WorkFlowType = WorkFlowType.IDE,
        conversation_id: int = 0,
        client_tools: list | None = None,
        tool_choice: str | None = None,
        server_tool_names: set[str] | None = None,
        priority: Priority | None = None,
        max_queue_wait: float | None = None,
        source: RequestSource | None = None,
        session_id: str | None = None,
    ) -> CompletionResult:
        """Execute a workflow and return the final accumulated result.

        Handles continuation, empty-response retry, and nudge logic
        identically to the streaming path.
        """
        result = CompletionResult()

        async with CompletionService._enqueue_and_wait(
            model_name,
            user_id,
            priority,
            max_queue_wait,
            source,
            session_id,
        ) as effective_model:
            model_name = effective_model

            async for event in CompletionService._build_and_run_with_retry(
                user_id,
                messages,
                model_name,
                workflow_type,
                conversation_id=conversation_id,
                client_tools=client_tools,
                tool_choice=tool_choice,
                server_tool_names=server_tool_names,
            ):
                if isinstance(event, ServerToolEvent):
                    continue
                if event.done and event.message:
                    set_result_response(
                        result,
                        event,
                        server_tool_names,
                    )

            if result.chat_response is None:
                return result

            if result.has_content and not result.has_tool_calls:
                accumulated_text = extract_text(
                    result.chat_response.message
                )
                if is_truncated(
                    accumulated_text,
                    result.chat_response.finish_reason or "",
                ):
                    logger.info(
                        "Non-streaming: model response appears truncated — sending continuation prompt",
                        extra={
                            "content_len": len(accumulated_text),
                            "finish_reason": result.chat_response.finish_reason,
                        },
                    )
                    truncation_messages = build_followup_messages(
                        messages,
                        TRUNCATION_CONTINUATION_PROMPT,
                        assistant_text=accumulated_text,
                    )
                    response = await CompletionService._collect_response(
                        user_id,
                        truncation_messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        "auto",
                        server_tool_names,
                    )
                    if response and response.message:
                        continuation_text = extract_text(
                            response.message
                        )
                        merged_text = accumulated_text + continuation_text
                        non_text_content = [
                            content
                            for content in (response.message.content or [])
                            if content.type != MessageContentType.TEXT
                        ]
                        response.message.content = [
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=merged_text,
                            ),
                            *non_text_content,
                        ]
                        if response.eval_count and result.chat_response:
                            response.eval_count = (
                                result.chat_response.eval_count or 0
                            ) + int(response.eval_count)
                        set_result_response(
                            result,
                            response,
                            server_tool_names,
                        )

            skip_continuation = (
                result.chat_response
                and result.chat_response.finish_reason in ("stop", "length")
            )
            if (
                _CONTINUATION_ENABLED
                and not result.has_tool_calls
                and client_tools
                and result.has_content
                and not skip_continuation
            ):
                accumulated_text = extract_text(
                    result.chat_response.message
                )
                if accumulated_text:
                    logger.info(
                        "Non-streaming: model produced text without tool calls — sending single continuation check",
                        extra={
                            "content_len": len(accumulated_text),
                            "content_preview": accumulated_text[:200],
                        },
                    )
                    continuation_messages = build_followup_messages(
                        messages,
                        CONTINUATION_PROMPT,
                        assistant_text=accumulated_text,
                    )
                    response = await CompletionService._collect_response(
                        user_id,
                        continuation_messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        "auto",
                        server_tool_names,
                    )
                    if response and response.message and response.message.tool_calls:
                        set_result_response(
                            result,
                            response,
                            server_tool_names,
                        )

            if (
                not result.has_content
                and not result.has_tool_calls
                and not result.is_error
                and (result.chat_response.finish_reason or "") != "stop"
            ):
                primary_prompt_tokens = int(
                    result.chat_response.prompt_eval_count or 0
                )
                primary_output_tokens = int(result.chat_response.eval_count or 0)
                primary_finish_reason = result.chat_response.finish_reason or ""
                model_num_ctx = await CompletionService._get_model_num_ctx(model_name)

                if is_context_overflow(
                    primary_prompt_tokens,
                    primary_finish_reason,
                    primary_output_tokens,
                    model_num_ctx=model_num_ctx,
                ):
                    result.context_overflow = True
                    logger.warning(
                        "Non-streaming: skipping retry — context likely exceeds model window",
                        extra={
                            "model": model_name,
                            "input_tokens": primary_prompt_tokens,
                            "output_tokens": primary_output_tokens,
                            "finish_reason": primary_finish_reason,
                        },
                    )
                else:
                    logger.warning(
                        "Non-streaming: model produced empty response — retrying",
                        extra={"model": model_name},
                    )
                    empty_response_retries_total.inc()
                    response = await CompletionService._collect_response(
                        user_id,
                        messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        tool_choice,
                        server_tool_names,
                    )
                    if response:
                        set_result_response(
                            result,
                            response,
                            server_tool_names,
                        )

                    if not result.has_content and not result.has_tool_calls:
                        logger.warning(
                            "Non-streaming: retry also empty — sending nudge prompt",
                            extra={"model": model_name},
                        )
                        nudge_messages = build_followup_messages(
                            messages,
                            EMPTY_RESPONSE_NUDGE,
                        )
                        response = await CompletionService._collect_response(
                            user_id,
                            nudge_messages,
                            model_name,
                            workflow_type,
                            conversation_id,
                            client_tools,
                            "auto",
                            server_tool_names,
                        )
                        if response:
                            set_result_response(
                                result,
                                response,
                                server_tool_names,
                            )

        return result
