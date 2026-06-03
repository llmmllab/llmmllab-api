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
from collections.abc import AsyncIterator, Awaitable, Callable
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
from models.chat_response import ChatResponse
from models.message import Message
from models.request_priority_metadata import (
    Priority,
    RequestPriorityMetadata,
    RequestSource,
)
from models.model_parameters import ModelParameters
from services.completion_state import CompletionResult, StreamAccumulator
from services.continuation_logic import (
    maybe_continue_on_hallucinated_tools,
    maybe_continue_on_hallucinated_tools_nonstream,
    maybe_continue_on_missing_tool_call,
    maybe_continue_on_missing_tool_call_nonstream,
    maybe_continue_on_truncation,
    maybe_continue_on_truncation_nonstream,
    maybe_retry_on_empty,
    maybe_retry_on_empty_nonstream,
)
from services.response_handlers import (
    extract_client_tool_names,
    extract_text,
    looks_like_anticipated_tool_call,
    looks_like_premature_stop,
    partition_tool_calls_by_validity,
    set_result_response,
    update_stream_delta,
    update_stream_final,
)
from services.retry_policies import stream_with_connection_retry
from services.session_tracking import (
    cancel_session as _cancel_session_impl,
    register_session_task,
    unregister_session_task,
)
from services.truncation import is_context_overflow, is_truncated
from utils.logging import llmmllogger

__all__ = ["CompletionService", "CompletionResult", "StreamAccumulator", "cancel_session"]

logger = llmmllogger.bind(component="completion_service")


async def cancel_session(session_id: str) -> bool:
    """Cancel an in-flight task by session_id. Returns True if found.

    Thin wrapper around :func:`services.session_tracking.cancel_session` kept
    here for backward compatibility — external callers (e.g.
    ``routers/session_admin.py``) import this symbol directly from
    ``services.completion_service``.
    """
    return await _cancel_session_impl(session_id)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

from middleware.api_metrics import (  # noqa: E402
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
        model_parameters: ModelParameters | None = None,
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
            model_parameters=model_parameters,
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
        disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    ) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
        """Execute a composed workflow and yield its events.

        ``disconnected`` is the optional client-liveness predicate threaded
        from the streaming router; it is forwarded to ``execute_workflow`` →
        the agent node so the in-flight run aborts on client disconnect.
        """
        start = time.monotonic()
        finished = False
        async for event in execute_workflow(
            initial_state, workflow, disconnected=disconnected
        ):
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
        model_parameters: ModelParameters | None = None,
        _retry_count: int = 0,
        disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
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
                model_parameters,
            )
            initial_state = await create_initial_state(
                user_id, conversation_id, builder, messages
            )
            async for event in CompletionService._run_workflow(
                initial_state, workflow, workflow_type, disconnected=disconnected
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
                model_parameters,
                _retry_count=_retry_count + 1,
                disconnected=disconnected,
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
        model_parameters: ModelParameters | None = None,
        disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    ) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
        """Build and run with automatic server handle refresh on connection failure.

        Delegates to :func:`services.retry_policies.stream_with_connection_retry`
        which catches connection-level errors, refreshes the model map, and
        retries with a fresh server handle.

        Parameters
        ----------
        max_retries:
            Maximum number of connection-error retries.
            Defaults to RUNNER_RETRIES from config (env: RUNNER_RETRIES).
        """
        if max_retries is None:
            max_retries = RUNNER_RETRIES

        async def _refresh_model_map() -> None:
            from services.runner_client import runner_client

            await runner_client.refresh_model_map()

        async for event in stream_with_connection_retry(
            CompletionService._build_and_run,
            user_id=user_id,
            messages=messages,
            model_name=model_name,
            workflow_type=workflow_type,
            conversation_id=conversation_id,
            client_tools=client_tools,
            tool_choice=tool_choice,
            server_tool_names=server_tool_names,
            model_parameters=model_parameters,
            max_retries=max_retries,
            backoff_base=RUNNER_RETRY_BACKOFF_BASE,
            refresh_model_map=_refresh_model_map,
            disconnected=disconnected,
        ):
            yield event

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
                        await register_session_task(session_id, cur_task)
                except RuntimeError:
                    pass

        try:
            yield effective_model
        finally:
            if session_id:
                await unregister_session_task(session_id)
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
        model_parameters: ModelParameters | None = None,
        priority: Priority | None = None,
        max_queue_wait: float | None = None,
        source: RequestSource | None = None,
        session_id: str | None = None,
        disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
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

        # Pre-bind the client-disconnect predicate onto _build_and_run so it
        # rides through EVERY re-dispatch path — not just the primary pass,
        # but the continuation / nudge / empty-retry helpers below, which each
        # take a ``build_and_run`` callable and re-run the workflow.  This is
        # what makes a zombie IDE session stop dead the moment its client
        # leaves, instead of grinding through a continuation/retry cascade on
        # the 27B runner.  With disconnected=None (non-streaming callers) the
        # partial is a transparent no-op.
        if disconnected is not None:
            from functools import partial

            run_fn = partial(
                CompletionService._build_and_run, disconnected=disconnected
            )
        else:
            run_fn = CompletionService._build_and_run

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
                    model_parameters=model_parameters,
                    disconnected=disconnected,
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
                    async for event, acc in maybe_continue_on_truncation(
                        run_fn,
                        acc,
                        user_id,
                        messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        server_tool_names,
                        model_parameters,
                    ):
                        yield event, acc

                # ----- Hallucinated tool name detection ------------
                # The model sometimes emits a tool_use block with a
                # name that isn't in the bound client_tools list
                # (commonly ``browser`` or ``memory_get`` — both
                # pretraining-frequent generic names that no MCP server
                # actually exposes here).  The client silently drops
                # those, no tool_result returns, and the conversation
                # loops with the model trying the same imagined name.
                # We:
                #   1. log the hallucination with the bad name(s)
                #   2. drop the invalid tool_calls from the response
                #      so the client doesn't even see them
                #   3. inject a synthetic user-message-style feedback
                #      ("Tool 'X' doesn't exist, available names: ...")
                #      and re-run the workflow so the model corrects
                _valid_tool_names = extract_client_tool_names(client_tools)
                _valid_calls, _invalid_calls = partition_tool_calls_by_validity(
                    acc.final_tool_calls, _valid_tool_names
                )
                if _invalid_calls:
                    logger.warning(
                        "Model emitted hallucinated tool names",
                        extra={
                            "invalid_tool_names": sorted(
                                {tc.name for tc in _invalid_calls if tc.name}
                            ),
                            "valid_emitted_count": len(_valid_calls),
                            "bound_tool_count": len(_valid_tool_names),
                        },
                    )
                    # Drop invalid calls from the accumulator before
                    # the downstream router emits tool_use SSE blocks
                    # from them.
                    acc.final_tool_calls = _valid_calls
                    acc.has_tool_calls = bool(_valid_calls)
                    async for event, acc in maybe_continue_on_hallucinated_tools(
                        run_fn,
                        acc,
                        user_id,
                        messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        _valid_tool_names,
                        _invalid_calls,
                        server_tool_names,
                        model_parameters,
                    ):
                        yield event, acc

                # On finish_reason=stop, nudge when the model declared
                # `[TOOL_INTENT:` in its text (explicit convention) OR
                # when the text ends with a tool-call precursor pattern
                # like "Now let me do X:" (heuristic — common
                # silent-stop right before a tool call the model
                # forgot to emit).  ``length`` is handled by
                # maybe_continue_on_truncation above.
                _declared_intent = "[TOOL_INTENT:" in (acc.final_content or "")
                _anticipated = looks_like_anticipated_tool_call(acc.final_content)
                # Degenerate premature stop: a tiny finish=stop answer
                # ("I", "Done") emitted mid-tool-loop — the model bailed
                # and the user would otherwise have to nudge by hand.
                _premature = looks_like_premature_stop(acc.final_content, messages)
                # Log every turn's tool-intent classification so we can
                # tell from logs whether the model is using the marker
                # convention or just stopping cleanly without ever
                # signalling tool use.  Three states matter:
                #   marker_yes_call_yes → working as designed (no nudge needed)
                #   marker_yes_call_no  → nudge fires (rescue path)
                #   marker_no_call_no_finish_stop  → accepted as final answer
                #   marker_no_call_no_finish_other → nudge fires (non-stop)
                # Log unconditionally for any turn that had tools bound
                # — including tool-only turns with zero text content,
                # which are the common case for direct tool invocations
                # and were missed by the original gate.
                if _CONTINUATION_ENABLED and client_tools:
                    _classification = (
                        "marker_yes_call_yes" if _declared_intent and acc.has_tool_calls
                        else "marker_yes_call_no" if _declared_intent
                        else "marker_no_call_yes" if acc.has_tool_calls
                        else f"marker_no_call_no_finish_{acc.finish_reason or 'unknown'}"
                    )
                    logger.info(
                        "Tool-intent classification",
                        extra={
                            "classification": _classification,
                            "finish_reason": acc.finish_reason,
                            "has_tool_calls": acc.has_tool_calls,
                            "has_content": acc.has_content,
                            "final_content_len": len(acc.final_content or ""),
                            "declared_intent": _declared_intent,
                            "anticipated_pattern": _anticipated,
                            "premature_stop": _premature,
                            "content_preview": (
                                acc.final_content[:200] if acc.final_content else ""
                            ),
                            "will_nudge": (
                                not acc.has_tool_calls
                                and (acc.has_content or acc.final_content)
                                and acc.finish_reason != "length"
                                and (
                                    acc.finish_reason != "stop"
                                    or _declared_intent
                                    or _anticipated
                                    or _premature
                                )
                            ),
                        },
                    )
                if (
                    _CONTINUATION_ENABLED
                    and not acc.has_tool_calls
                    and client_tools
                    and (acc.has_content or acc.final_content)
                    and acc.finish_reason != "length"
                    and (
                        acc.finish_reason != "stop"
                        or _declared_intent
                        or _anticipated
                        or _premature
                    )
                ):
                    async for event, acc in maybe_continue_on_missing_tool_call(
                        run_fn,
                        acc,
                        user_id,
                        messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        server_tool_names,
                        model_parameters,
                    ):
                        yield event, acc

                if (
                    not acc.has_content
                    and not acc.has_tool_calls
                    and not acc.final_content
                    and not acc.is_error
                ):
                    # Empty response with finish_reason="stop" is the
                    # exact failure mode that surfaces in a Claude Code
                    # session as "the model stopped mid-conversation":
                    # the model emits EOS without producing any content
                    # or tool calls, the api closes with stop_reason=
                    # end_turn, and the client has nothing to render.
                    # The retry below probes for stale runner handles
                    # and resends the same prompt with the nudge — that's
                    # exactly what's wanted here.  Previously this
                    # branch was gated by ``finish_reason != "stop"``,
                    # which silently skipped the retry for the most
                    # common case.
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
                        # Chat completions go through LangChain's ChatOpenAI
                        # directly, bypassing proxy_request's 404→StaleServerError
                        # path; a dead-handle scenario surfaces here as an
                        # empty response.  The probe + StaleServerError raise
                        # inside maybe_retry_on_empty bubbles up through the
                        # connection-retry layer and reaches the stale-server
                        # retry in _build_and_run on the next pass.
                        from services.runner_client import runner_client

                        async def _revalidate() -> int:
                            return await runner_client.revalidate_runner_handles()

                        async for event, acc in maybe_retry_on_empty(
                            run_fn,
                            acc,
                            user_id,
                            messages,
                            model_name,
                            workflow_type,
                            conversation_id,
                            client_tools,
                            tool_choice,
                            server_tool_names,
                            model_parameters,
                            revalidate_runner_handles=_revalidate,
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
        model_parameters: ModelParameters | None = None,
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
                model_parameters=model_parameters,
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
                accumulated_text = extract_text(result.chat_response.message)
                if is_truncated(
                    accumulated_text,
                    result.chat_response.finish_reason or "",
                ):
                    await maybe_continue_on_truncation_nonstream(
                        CompletionService._build_and_run,
                        result,
                        user_id,
                        messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        server_tool_names,
                        model_parameters,
                    )

            # See streaming-path comment above. On stop, only nudge
            # when the model declared `[TOOL_INTENT:` in its text but
            # didn't invoke a tool. ``length`` is handled by
            # maybe_continue_on_truncation_nonstream.
            _nonstream_text = ""
            if result.chat_response and result.chat_response.message:
                msg_content = result.chat_response.message.content
                if isinstance(msg_content, str):
                    _nonstream_text = msg_content
                elif isinstance(msg_content, list):
                    for block in msg_content:
                        if hasattr(block, "text") and block.text:
                            _nonstream_text += block.text
            _nonstream_declared_intent = "[TOOL_INTENT:" in _nonstream_text
            _ns_anticipated = looks_like_anticipated_tool_call(_nonstream_text)
            _ns_premature = looks_like_premature_stop(_nonstream_text, messages)
            _ns_finish = (
                result.chat_response.finish_reason if result.chat_response else None
            )

            # ----- Hallucinated tool name detection (non-stream) ------
            # Same logic as the streaming path: drop tool calls with
            # names not in client_tools, then re-run with synthetic
            # feedback so the model picks a real name.
            _ns_valid_names = extract_client_tool_names(client_tools)
            _ns_response_tool_calls = (
                result.chat_response.message.tool_calls
                if result.chat_response and result.chat_response.message
                else None
            )
            _ns_valid_calls, _ns_invalid_calls = partition_tool_calls_by_validity(
                _ns_response_tool_calls, _ns_valid_names
            )
            if _ns_invalid_calls:
                logger.warning(
                    "Model emitted hallucinated tool names (non-stream)",
                    extra={
                        "invalid_tool_names": sorted(
                            {tc.name for tc in _ns_invalid_calls if tc.name}
                        ),
                        "valid_emitted_count": len(_ns_valid_calls),
                        "bound_tool_count": len(_ns_valid_names),
                    },
                )
                if result.chat_response and result.chat_response.message:
                    result.chat_response.message.tool_calls = _ns_valid_calls
                await maybe_continue_on_hallucinated_tools_nonstream(
                    CompletionService._build_and_run,
                    result,
                    user_id,
                    messages,
                    model_name,
                    workflow_type,
                    conversation_id,
                    client_tools,
                    _ns_valid_names,
                    _ns_invalid_calls,
                    server_tool_names,
                    model_parameters,
                )
                # The recursive pass produced result.chat_response.
                # Refresh the cached text + classification view for
                # the gate logic below.
                _nonstream_text = ""
                if result.chat_response and result.chat_response.message:
                    msg_content = result.chat_response.message.content
                    if isinstance(msg_content, str):
                        _nonstream_text = msg_content
                    elif isinstance(msg_content, list):
                        for block in msg_content:
                            if hasattr(block, "text") and block.text:
                                _nonstream_text += block.text
                _nonstream_declared_intent = "[TOOL_INTENT:" in _nonstream_text
                _ns_anticipated = looks_like_anticipated_tool_call(_nonstream_text)
                _ns_premature = looks_like_premature_stop(_nonstream_text, messages)
                _ns_finish = (
                    result.chat_response.finish_reason
                    if result.chat_response else None
                )

            # Same broadened gate as the streaming path — log every
            # turn that had tools bound, regardless of whether it
            # produced content or just tool calls.
            if _CONTINUATION_ENABLED and client_tools:
                _ns_classification = (
                    "marker_yes_call_yes" if _nonstream_declared_intent and result.has_tool_calls
                    else "marker_yes_call_no" if _nonstream_declared_intent
                    else "marker_no_call_yes" if result.has_tool_calls
                    else f"marker_no_call_no_finish_{_ns_finish or 'unknown'}"
                )
                logger.info(
                    "Tool-intent classification (non-stream)",
                    extra={
                        "classification": _ns_classification,
                        "finish_reason": _ns_finish,
                        "has_tool_calls": result.has_tool_calls,
                        "has_content": result.has_content,
                        "final_content_len": len(_nonstream_text),
                        "declared_intent": _nonstream_declared_intent,
                        "anticipated_pattern": _ns_anticipated,
                        "premature_stop": _ns_premature,
                        "content_preview": _nonstream_text[:200],
                    },
                )
            skip_continuation = result.chat_response is not None and (
                result.chat_response.finish_reason == "length"
                or (
                    result.chat_response.finish_reason == "stop"
                    and not _nonstream_declared_intent
                    and not _ns_anticipated
                    and not _ns_premature
                )
            )
            if (
                _CONTINUATION_ENABLED
                and not result.has_tool_calls
                and client_tools
                and result.has_content
                and not skip_continuation
            ):
                await maybe_continue_on_missing_tool_call_nonstream(
                    CompletionService._build_and_run,
                    result,
                    user_id,
                    messages,
                    model_name,
                    workflow_type,
                    conversation_id,
                    client_tools,
                    server_tool_names,
                    model_parameters,
                )

            if (
                not result.has_content
                and not result.has_tool_calls
                and not result.is_error
            ):
                # See streaming-path note above: the previous
                # ``finish_reason != "stop"`` guard suppressed the
                # retry on the dominant empty-turn failure mode where
                # the model emits EOS with no content or tool calls.
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
                    await maybe_retry_on_empty_nonstream(
                        CompletionService._build_and_run,
                        result,
                        user_id,
                        messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        tool_choice,
                        server_tool_names,
                        model_parameters,
                    )

        return result
