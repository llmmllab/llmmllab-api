"""
Secondary-pass continuation and empty-response retry logic.

CompletionService orchestrates a single completion as a primary pass followed
(when needed) by one or more secondary passes:

* truncation continuation — model output was cut off mid-sentence
* tool continuation — model described a tool call but never invoked one
* empty-response retry — model produced nothing (with a runner-restart probe
  first, then a follow-up nudge prompt if the retry is also empty)
* nudge — final fallback prompt when retries don't produce output

This module owns the helpers for those passes.  ``CompletionService`` is the
thin orchestrator that decides which to invoke and in what order.

CRITICAL: the empty-response retry path calls
``runner_client.revalidate_runner_handles()`` before retrying.  If a runner
restart is detected, a ``StaleServerError`` is raised so the upper retry
layer rebuilds the workflow with a fresh handle.
"""

from collections.abc import AsyncIterator
from typing import Awaitable, Callable, Optional, Union

from graph.state import ServerToolEvent
from middleware.api_metrics import empty_response_retries_total
from models.chat_response import ChatResponse
from models.message import Message, MessageContent, MessageContentType
from models.model_parameters import ModelParameters
from services.completion_state import CompletionResult, StreamAccumulator
from models.tool_call import ToolCall
from services.prompt_templates import (
    CONTINUATION_PROMPT,
    EMPTY_RESPONSE_NUDGE,
    INCOMPLETE_TOOL_TURN_PROMPT,
    MISSING_SUMMARY_NUDGE,
    TRUNCATION_CONTINUATION_PROMPT,
    hallucinated_tool_feedback,
)
from services.response_handlers import (
    build_followup_messages,
    extract_text,
    set_result_response,
    update_stream_delta,
    update_stream_final,
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="continuation_logic")


# Type alias for the inner build-and-run callable.  Returns an async iterator
# of ``ChatResponse | ServerToolEvent``.
BuildAndRunFn = Callable[..., AsyncIterator[Union[ChatResponse, ServerToolEvent]]]


# ---------------------------------------------------------------------------
# Non-streaming helpers
# ---------------------------------------------------------------------------


async def collect_response(
    build_and_run: BuildAndRunFn,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    tool_choice: str | None,
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
) -> Optional[ChatResponse]:
    """Run a workflow and return only the final ChatResponse."""
    async for event in build_and_run(
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
        if isinstance(event, ServerToolEvent):
            continue
        if event.done and event.message:
            return event
    return None


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


async def stream_secondary_pass(
    build_and_run: BuildAndRunFn,
    acc: StreamAccumulator,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    tool_choice: str | None,
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
    *,
    content_prefix: str = "",
) -> AsyncIterator[tuple[Union[ChatResponse, ServerToolEvent], StreamAccumulator]]:
    """Yield streaming events from a secondary pass, updating ``acc``.

    The final event is consumed to update the accumulator's tool-calls,
    content, and token counts — it is not re-yielded because the primary
    pass already emitted a ``done`` event.
    """
    async for event in build_and_run(
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


async def maybe_continue_on_truncation(
    build_and_run: BuildAndRunFn,
    acc: StreamAccumulator,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
) -> AsyncIterator[tuple[Union[ChatResponse, ServerToolEvent], StreamAccumulator]]:
    """Send a truncation-continuation prompt and stream the follow-up."""
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
    async for event, acc in stream_secondary_pass(
        build_and_run,
        acc,
        user_id,
        truncation_messages,
        model_name,
        workflow_type,
        conversation_id,
        client_tools,
        "auto",
        server_tool_names,
        model_parameters,
        content_prefix=accumulated_text,
    ):
        yield event, acc


async def maybe_continue_on_missing_tool_call(
    build_and_run: BuildAndRunFn,
    acc: StreamAccumulator,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
) -> AsyncIterator[tuple[Union[ChatResponse, ServerToolEvent], StreamAccumulator]]:
    """Send a tool-continuation prompt and stream the follow-up."""
    accumulated_text = acc.final_content or ""
    if not accumulated_text:
        return
    logger.info(
        "Model produced text without tool calls — sending single continuation check",
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
    async for event, acc in stream_secondary_pass(
        build_and_run,
        acc,
        user_id,
        continuation_messages,
        model_name,
        workflow_type,
        conversation_id,
        client_tools,
        "auto",
        server_tool_names,
        model_parameters,
    ):
        yield event, acc


async def maybe_continue_on_hallucinated_tools(
    build_and_run: BuildAndRunFn,
    acc: StreamAccumulator,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    valid_tool_names: set[str],
    invalid_calls: list[ToolCall],
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
) -> AsyncIterator[tuple[Union[ChatResponse, ServerToolEvent], StreamAccumulator]]:
    """Re-run the workflow with feedback about hallucinated tool names.

    The primary pass emitted tool calls with names that aren't in the
    bound list — the client would silently drop them and the model
    would loop indefinitely calling the same nonexistent tool.  Inject
    a user-message-style error explaining which names were bad and
    what's actually available, then stream the secondary pass to the
    client so the corrected response replaces (well, *augments*) the
    dropped call.
    """
    bad_names = [tc.name for tc in invalid_calls if tc.name]
    if not bad_names:
        return
    logger.warning(
        "Model emitted tool calls with names not in bound list — "
        "dropping invalid calls and re-running with feedback",
        extra={
            "invalid_tool_names": bad_names,
            "valid_count": len(valid_tool_names),
            "valid_preview": sorted(valid_tool_names)[:10],
        },
    )
    feedback = hallucinated_tool_feedback(bad_names, valid_tool_names)
    # Preserve any text the model emitted alongside the bad tool call so
    # the secondary pass keeps the train-of-thought context.
    assistant_text = acc.final_content or ""
    correction_messages = build_followup_messages(
        messages,
        feedback,
        assistant_text=assistant_text or None,
    )
    async for event, acc in stream_secondary_pass(
        build_and_run,
        acc,
        user_id,
        correction_messages,
        model_name,
        workflow_type,
        conversation_id,
        client_tools,
        "auto",
        server_tool_names,
        model_parameters,
    ):
        yield event, acc


async def maybe_continue_on_hallucinated_tools_nonstream(
    build_and_run: BuildAndRunFn,
    result: CompletionResult,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    valid_tool_names: set[str],
    invalid_calls: list[ToolCall],
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
) -> None:
    """Non-streaming counterpart to :func:`maybe_continue_on_hallucinated_tools`.

    Replaces ``result.chat_response`` with the corrected pass's final
    response, the same way ``maybe_continue_on_missing_tool_call_nonstream``
    does for its nudge.
    """
    bad_names = [tc.name for tc in invalid_calls if tc.name]
    if not bad_names:
        return
    logger.warning(
        "Model emitted tool calls with names not in bound list "
        "(non-streaming) — dropping and re-running with feedback",
        extra={
            "invalid_tool_names": bad_names,
            "valid_count": len(valid_tool_names),
        },
    )
    feedback = hallucinated_tool_feedback(bad_names, valid_tool_names)
    assistant_text = (
        extract_text(result.chat_response.message)
        if result.chat_response and result.chat_response.message
        else ""
    )
    correction_messages = build_followup_messages(
        messages,
        feedback,
        assistant_text=assistant_text or None,
    )
    final = await collect_response(
        build_and_run,
        user_id,
        correction_messages,
        model_name,
        workflow_type,
        conversation_id,
        client_tools,
        "auto",
        server_tool_names,
        model_parameters,
    )
    if final is not None:
        set_result_response(result, final, server_tool_names)


async def maybe_nudge_on_missing_summary(
    build_and_run: BuildAndRunFn,
    acc: StreamAccumulator,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
) -> AsyncIterator[tuple[Union[ChatResponse, ServerToolEvent], StreamAccumulator]]:
    """Nudge the model when its response lacks the ``*-(o.o)-*`` marker."""
    accumulated_text = acc.final_content or ""
    logger.info(
        "Model response missing summary marker — sending nudge",
        extra={
            "content_len": len(accumulated_text),
            "content_tail": accumulated_text[-200:] if accumulated_text else "",
        },
    )
    nudge_messages = build_followup_messages(
        messages,
        MISSING_SUMMARY_NUDGE,
        assistant_text=accumulated_text,
    )
    async for event, acc in stream_secondary_pass(
        build_and_run,
        acc,
        user_id,
        nudge_messages,
        model_name,
        workflow_type,
        conversation_id,
        client_tools,
        "auto",
        server_tool_names,
        model_parameters,
    ):
        yield event, acc


async def maybe_retry_on_empty(
    build_and_run: BuildAndRunFn,
    acc: StreamAccumulator,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    tool_choice: str | None,
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
    *,
    revalidate_runner_handles: Callable[[], Awaitable[int]],
) -> AsyncIterator[tuple[Union[ChatResponse, ServerToolEvent], StreamAccumulator]]:
    """Retry on empty response with tool-result-aware continuation.

    Before retrying, probes runner startup epochs via
    ``revalidate_runner_handles``.  If any runner was found to have restarted,
    raises ``StaleServerError`` so the upper retry layer rebuilds the
    workflow with a fresh handle.

    When the empty response follows tool results (the model stopped after
    receiving tool output), re-sending the same messages is futile — the
    model already saw them and chose to stop.  Instead, inject a continuation
    prompt (``INCOMPLETE_TOOL_TURN_PROMPT``) to force the model to continue.

    After all retries are exhausted with zero content, marks the accumulator
    as ``incomplete_turn`` so the router can emit a diagnostic finish reason.
    """
    from graph.errors import StaleServerError
    from services.response_handlers import last_message_has_tool_results

    purged = await revalidate_runner_handles()
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

    is_tool_turn = last_message_has_tool_results(messages)

    logger.warning(
        "Model produced empty response — retrying",
        extra={
            "model": model_name,
            "is_tool_turn": is_tool_turn,
        },
    )
    empty_response_retries_total.inc()

    # First retry: if this is an incomplete tool turn, use a continuation
    # prompt instead of re-sending identical messages.
    if is_tool_turn:
        retry_messages = build_followup_messages(
            messages,
            INCOMPLETE_TOOL_TURN_PROMPT,
        )
    else:
        retry_messages = messages

    async for event, acc in stream_secondary_pass(
        build_and_run,
        acc,
        user_id,
        retry_messages,
        model_name,
        workflow_type,
        conversation_id,
        client_tools,
        tool_choice,
        server_tool_names,
        model_parameters,
    ):
        yield event, acc

    # Second pass: if still empty, try a different nudge
    if not acc.has_content and not acc.has_tool_calls and not acc.final_content:
        if is_tool_turn:
            # Force a summary — the model had tool results but refused to
            # produce any output across two retries.
            logger.warning(
                "Incomplete tool turn — retry also empty, sending summary nudge",
                extra={"model": model_name},
            )
            nudge_messages = build_followup_messages(
                messages,
                MISSING_SUMMARY_NUDGE,
            )
        else:
            logger.warning(
                "Retry also produced empty response — sending nudge prompt",
                extra={"model": model_name},
            )
            nudge_messages = build_followup_messages(
                messages,
                EMPTY_RESPONSE_NUDGE,
            )
        async for event, acc in stream_secondary_pass(
            build_and_run,
            acc,
            user_id,
            nudge_messages,
            model_name,
            workflow_type,
            conversation_id,
            client_tools,
            "auto",
            server_tool_names,
            model_parameters,
        ):
            yield event, acc

    # All retries exhausted — mark as incomplete turn so the router can
    # emit a diagnostic finish reason for clients like OpenClaw.
    if not acc.has_content and not acc.has_tool_calls and not acc.final_content:
        logger.warning(
            "Empty response retries exhausted — marking incomplete turn",
            extra={"model": model_name, "is_tool_turn": is_tool_turn},
        )
        acc.incomplete_turn = True


# ---------------------------------------------------------------------------
# Non-streaming continuation helpers
# ---------------------------------------------------------------------------


async def maybe_continue_on_truncation_nonstream(
    build_and_run: BuildAndRunFn,
    result: CompletionResult,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
) -> None:
    """Send a truncation-continuation prompt and merge the response into ``result``."""
    accumulated_text = extract_text(result.chat_response.message)
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
    response = await collect_response(
        build_and_run,
        user_id,
        truncation_messages,
        model_name,
        workflow_type,
        conversation_id,
        client_tools,
        "auto",
        server_tool_names,
        model_parameters,
    )
    if response and response.message:
        continuation_text = extract_text(response.message)
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
            response.eval_count = (result.chat_response.eval_count or 0) + int(
                response.eval_count
            )
        set_result_response(result, response, server_tool_names)


async def maybe_continue_on_missing_tool_call_nonstream(
    build_and_run: BuildAndRunFn,
    result: CompletionResult,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
) -> None:
    """Send a tool-continuation prompt and adopt the response if it has tool calls."""
    accumulated_text = extract_text(result.chat_response.message)
    if not accumulated_text:
        return
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
    response = await collect_response(
        build_and_run,
        user_id,
        continuation_messages,
        model_name,
        workflow_type,
        conversation_id,
        client_tools,
        "auto",
        server_tool_names,
        model_parameters,
    )
    if response and response.message and response.message.tool_calls:
        set_result_response(result, response, server_tool_names)


async def maybe_retry_on_empty_nonstream(
    build_and_run: BuildAndRunFn,
    result: CompletionResult,
    user_id: str,
    messages: list[Message],
    model_name: str,
    workflow_type,
    conversation_id: int,
    client_tools: list | None,
    tool_choice: str | None,
    server_tool_names: set[str] | None,
    model_parameters: ModelParameters | None = None,
) -> None:
    """Retry on empty response with tool-result-aware continuation.

    When the empty response follows tool results, use a continuation prompt
    instead of re-sending identical messages.  After all retries, mark as
    ``incomplete_turn`` if still empty.
    """
    from services.response_handlers import last_message_has_tool_results

    is_tool_turn = last_message_has_tool_results(messages)

    logger.warning(
        "Non-streaming: model produced empty response — retrying",
        extra={"model": model_name, "is_tool_turn": is_tool_turn},
    )
    empty_response_retries_total.inc()

    # First retry: continuation prompt for tool turns, same messages otherwise
    if is_tool_turn:
        retry_messages = build_followup_messages(
            messages,
            INCOMPLETE_TOOL_TURN_PROMPT,
        )
    else:
        retry_messages = messages

    response = await collect_response(
        build_and_run,
        user_id,
        retry_messages,
        model_name,
        workflow_type,
        conversation_id,
        client_tools,
        tool_choice,
        server_tool_names,
        model_parameters,
    )
    if response:
        set_result_response(result, response, server_tool_names)

    if not result.has_content and not result.has_tool_calls:
        # Second pass: summary nudge for tool turns, generic nudge otherwise
        if is_tool_turn:
            logger.warning(
                "Non-streaming: incomplete tool turn — retry also empty, sending summary nudge",
                extra={"model": model_name},
            )
            nudge_messages = build_followup_messages(
                messages,
                MISSING_SUMMARY_NUDGE,
            )
        else:
            logger.warning(
                "Non-streaming: retry also empty — sending nudge prompt",
                extra={"model": model_name},
            )
            nudge_messages = build_followup_messages(
                messages,
                EMPTY_RESPONSE_NUDGE,
            )
        response = await collect_response(
            build_and_run,
            user_id,
            nudge_messages,
            model_name,
            workflow_type,
            conversation_id,
            client_tools,
            "auto",
            server_tool_names,
            model_parameters,
        )
        if response:
            set_result_response(result, response, server_tool_names)

    # All retries exhausted — mark as incomplete turn
    if not result.has_content and not result.has_tool_calls:
        logger.warning(
            "Non-streaming: empty response retries exhausted — marking incomplete turn",
            extra={"model": model_name, "is_tool_turn": is_tool_turn},
        )
        result.incomplete_turn = True
