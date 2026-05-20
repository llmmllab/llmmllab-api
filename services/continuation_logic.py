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
    """Retry on empty response and follow up with a nudge if still empty.

    Before retrying, probes runner startup epochs via
    ``revalidate_runner_handles``.  If any runner was found to have restarted,
    raises ``StaleServerError`` so the upper retry layer rebuilds the
    workflow with a fresh handle.
    """
    from graph.errors import StaleServerError

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

    logger.warning(
        "Model produced empty response — retrying with same messages",
        extra={"model": model_name},
    )
    empty_response_retries_total.inc()
    async for event, acc in stream_secondary_pass(
        build_and_run,
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
            response.eval_count = (
                result.chat_response.eval_count or 0
            ) + int(response.eval_count)
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
    """Retry on empty response and follow up with a nudge if still empty."""
    logger.warning(
        "Non-streaming: model produced empty response — retrying",
        extra={"model": model_name},
    )
    empty_response_retries_total.inc()
    response = await collect_response(
        build_and_run,
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
    if response:
        set_result_response(result, response, server_tool_names)

    if not result.has_content and not result.has_tool_calls:
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
