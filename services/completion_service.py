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
from dataclasses import dataclass, field
from typing import Optional, Union

from composer_init import (
    compose_workflow,
    create_initial_state,
    execute_workflow,
    get_graph_builder,
)
from graph.state import ServerToolEvent
from graph.workflows.factory import WorkFlowType
from models.chat_response import ChatResponse
from models.message import Message, MessageContent, MessageContentType, MessageRole
from models.request_priority_metadata import (
    Priority,
    RequestPriorityMetadata,
    RequestSource,
)
from models.tool_call import ToolCall
from utils.logging import llmmllogger
from httpx import RemoteProtocolError, ConnectError
from config import RUNNER_RETRIES, RUNNER_RETRY_BACKOFF_BASE

logger = llmmllogger.bind(component="completion_service")

# Connection-level errors that indicate the runner/server is unreachable.
# These should trigger a server-handle refresh, not an empty-response retry.
_CONNECTION_ERRORS = (RemoteProtocolError, ConnectError)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

from middleware.api_metrics import (  # noqa: E402
    workflow_completions_total,
    workflow_duration_seconds,
    empty_response_retries_total,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from config import ENABLE_TOOL_CONTINUATION

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

_CONTINUATION_PROMPT = (
    "You described using a tool but did not actually call one. "
    "Call the appropriate tool now. Do not describe what you will do — invoke the tool directly."
)

_EMPTY_RESPONSE_NUDGE = (
    "Your response didn't produce any output. Did you mean to say something "
    "or use a tool? If so, continue. Otherwise, simply respond with 'done' "
    "and nothing else."
)

# Threshold (in tokens) above which we consider the prompt "large".
# When a large prompt produces an empty response, retrying is futile —
# the context is likely exceeding the model's window.
_CONTEXT_OVERFLOW_THRESHOLD = 100_000

_TRUNCATION_CONTINUATION_PROMPT = (
    "Your response was cut off. Continue from where you left off. "
    "If you were in the middle of a tool call, complete the tool call. "
    "If you were in the middle of text, continue the text."
)

_SENTENCE_TERMINATORS = frozenset(".!?)\n`]}\"'>,:")
_TRUNCATION_MIN_LEN = 40


def _is_context_overflow(
    prompt_tokens: int,
    finish_reason: str,
    output_tokens: int,
    model_num_ctx: int | None = None,
) -> bool:
    """Detect whether the model likely hit its context window limit.

    When *model_num_ctx* is provided, the total token budget
    (prompt tokens + output tokens) is compared against the model's
    context window to determine overflow.  When it is not provided,
    a fixed threshold is used as a fallback.

    Returns True when:
    - The total tokens exceed the model's context window (if known), OR
    - The prompt consumed a lot of tokens (above threshold), AND
    - The model produced no output (empty response), OR
    - The model was cut off immediately (finish_reason == 'length' with zero output).

    In these cases, retrying with the same (or larger) context is futile.
    """
    # Prefer a model-aware check when num_ctx is available.
    if model_num_ctx is not None:
        total_tokens = prompt_tokens + output_tokens
        if total_tokens >= model_num_ctx:
            return True
        # If we're well below the context window, no overflow.
        if total_tokens < _CONTEXT_OVERFLOW_THRESHOLD:
            return False
    else:
        if prompt_tokens < _CONTEXT_OVERFLOW_THRESHOLD:
            return False
    if output_tokens > 0:
        return False
    # Empty response with a large prompt — almost certainly context overflow.
    # Also catches finish_reason == 'length' with zero output.
    return True


def _is_truncated(text: str, finish_reason: str) -> bool:
    """Detect a response that should be continued.

    `finish_reason="length"` means the model hit the token limit — always
    truncated by definition, regardless of trailing punctuation.

    `finish_reason="stop"` means the model emitted EOS. This is usually
    intentional, but llama.cpp occasionally emits EOS mid-sentence (a
    "premature stop"). We apply a heuristic: if the response is non-trivial
    in length and ends without a sentence terminator, treat it as truncated.
    Short replies are excluded — single-word answers ("OK", "42", a URL)
    legitimately end without punctuation.
    """
    if finish_reason == "length":
        return bool(text and text.strip())
    if finish_reason != "stop":
        return False
    stripped = text.rstrip()
    if len(stripped) < _TRUNCATION_MIN_LEN:
        return False
    return stripped[-1] not in _SENTENCE_TERMINATORS


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CompletionResult:
    """Accumulated result from a non-streaming completion."""

    chat_response: Optional[ChatResponse] = None
    output_tokens: int = 0

    @property
    def has_content(self) -> bool:
        return bool(
            self.chat_response
            and self.chat_response.message
            and self.chat_response.message.content
            and any(
                c.text
                for c in self.chat_response.message.content
                if c.type == MessageContentType.TEXT and c.text
            )
        )

    @property
    def has_tool_calls(self) -> bool:
        return bool(
            self.chat_response
            and self.chat_response.message
            and self.chat_response.message.tool_calls
        )

    @property
    def is_error(self) -> bool:
        return bool(self.chat_response and self.chat_response.finish_reason == "error")


@dataclass
class StreamAccumulator:
    """Mutable state accumulated while streaming events to the router."""

    has_content: bool = False
    has_tool_calls: bool = False
    is_error: bool = False
    finish_reason: str = ""
    final_tool_calls: list[ToolCall] = field(default_factory=list)
    final_content: str = ""
    output_tokens: int = 0
    input_tokens: int = 0


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
    ) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
        """Build a composer workflow and yield its events."""
        # Resolve model name — fall back to user's default if unavailable
        model_name = await _resolve_model(model_name, user_id)

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
                return  # Success — all events yielded
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
                    await asyncio.sleep(
                        RUNNER_RETRY_BACKOFF_BASE * (attempt + 1)
                    )  # Linear backoff
                    # Force model map refresh so we get a healthy runner
                    from services.runner_client import runner_client

                    await runner_client.refresh_model_map()
                    continue
                # Exhausted retries — re-raise
                raise

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

        # Priority queue integration
        from config import PRIORITY_QUEUE_ENABLED
        from services.priority_queue import priority_queue

        _effective_priority = priority if priority is not None else Priority.HIGH
        _queue_ctx = None
        if PRIORITY_QUEUE_ENABLED:
            _meta = RequestPriorityMetadata(
                source=RequestSource.USER,
                priority=_effective_priority,
                user_id=user_id,
                max_queue_wait=max_queue_wait,
            )
            _queue_ctx = await priority_queue.enqueue(_meta)
        try:
            # ---------- primary pass (with connection-error retry) ----------
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
                    if event.finish_reason == "error":
                        acc.is_error = True
                    if event.finish_reason:
                        acc.finish_reason = event.finish_reason
                    if event.message and event.message.tool_calls:
                        acc.final_tool_calls = event.message.tool_calls
                        acc.has_tool_calls = True
                    if event.message and event.message.content:
                        parts = [
                            c.text
                            for c in event.message.content
                            if c.type == MessageContentType.TEXT and c.text
                        ]
                        acc.final_content = "".join(parts)
                    if event.prompt_eval_count:
                        acc.input_tokens = int(event.prompt_eval_count)
                    if event.eval_count:
                        acc.output_tokens = int(event.eval_count)
                    yield event, acc
                    continue

                # Live delta — let the router stream it
                if event.message and event.message.content:
                    for part in event.message.content:
                        if part.type == MessageContentType.TEXT and part.text:
                            acc.has_content = True
                yield event, acc

            # Filter server tool calls out of final_tool_calls
            if server_tool_names and acc.final_tool_calls:
                acc.final_tool_calls = [
                    tc
                    for tc in acc.final_tool_calls
                    if tc.name not in server_tool_names
                ]
                acc.has_tool_calls = bool(acc.final_tool_calls)

            # ---------- truncation continuation ----------
            if (
                _is_truncated(acc.final_content or "", acc.finish_reason)
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
                truncation_messages = list(messages) + [
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=accumulated_text
                            )
                        ],
                    ),
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=_TRUNCATION_CONTINUATION_PROMPT,
                            )
                        ],
                    ),
                ]
                async for event in CompletionService._build_and_run(
                    user_id,
                    truncation_messages,
                    model_name,
                    workflow_type,
                    conversation_id,
                    client_tools,
                    "auto",
                    server_tool_names,
                ):
                    if isinstance(event, ServerToolEvent):
                        continue
                    if event.done:
                        if event.message and event.message.tool_calls:
                            acc.final_tool_calls = event.message.tool_calls
                            acc.has_tool_calls = True
                        if event.message and event.message.content:
                            parts = [
                                c.text
                                for c in event.message.content
                                if c.type == MessageContentType.TEXT and c.text
                            ]
                            acc.final_content = accumulated_text + "".join(parts)
                        if event.eval_count:
                            acc.output_tokens += int(event.eval_count)
                        # Reflect the continuation's terminal state — without
                        # this the client would see the original "length" and
                        # think the response is still truncated.
                        if event.finish_reason:
                            acc.finish_reason = event.finish_reason
                        continue

                    # Live delta from truncation continuation
                    if event.message and event.message.content:
                        for part in event.message.content:
                            if part.type == MessageContentType.TEXT and part.text:
                                acc.has_content = True
                    yield event, acc

            # ---------- continuation check ----------
            # Skip when the model naturally stopped (finish_reason="stop") —
            # it intentionally chose not to call a tool and forcing one
            # creates an infinite loop where the model keeps saying "done"
            # but gets coerced into unnecessary tool calls.
            # Also skip on "length" — the model was cut off mid-response
            # by the token limit, not intentionally avoiding tool calls.
            if (
                _CONTINUATION_ENABLED
                and not acc.has_tool_calls
                and client_tools
                and (acc.has_content or acc.final_content)
                and acc.finish_reason not in ("stop", "length")
            ):
                accumulated_text = acc.final_content or ""
                logger.info(
                    "Model produced text without tool calls — sending single continuation check",
                    extra={
                        "content_len": len(accumulated_text),
                        "content_preview": accumulated_text[:200],
                    },
                )
                continuation_messages = list(messages) + [
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=accumulated_text
                            )
                        ],
                    ),
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=_CONTINUATION_PROMPT
                            )
                        ],
                    ),
                ]
                async for event in CompletionService._build_and_run(
                    user_id,
                    continuation_messages,
                    model_name,
                    workflow_type,
                    conversation_id,
                    client_tools,
                    "auto",
                    server_tool_names,
                ):
                    if isinstance(event, ServerToolEvent):
                        continue
                    if event.done:
                        if event.message and event.message.tool_calls:
                            acc.final_tool_calls = event.message.tool_calls
                            acc.has_tool_calls = True
                        if event.eval_count:
                            acc.output_tokens += int(event.eval_count)
                        continue

            # ---------- empty-response retry ----------
            if (
                not acc.has_content
                and not acc.has_tool_calls
                and not acc.final_content
                and not acc.is_error
            ):
                # Look up the model's context window for an accurate overflow check.
                _model_num_ctx = await CompletionService._get_model_num_ctx(model_name)

                # Skip retry if context is likely too large — retrying the
                # same oversized messages (or adding a nudge) is futile.
                if _is_context_overflow(
                    acc.input_tokens,
                    acc.finish_reason,
                    acc.output_tokens,
                    model_num_ctx=_model_num_ctx,
                ):
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
                    logger.warning(
                        "Model produced empty response — retrying with same messages",
                        extra={"model": model_name},
                    )
                    empty_response_retries_total.inc()
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
                            if event.message and event.message.tool_calls:
                                acc.final_tool_calls = event.message.tool_calls
                                acc.has_tool_calls = True
                            if event.message and event.message.content:
                                parts = [
                                    c.text
                                    for c in event.message.content
                                    if c.type == MessageContentType.TEXT and c.text
                                ]
                                acc.final_content = "".join(parts)
                            if event.eval_count:
                                acc.output_tokens += int(event.eval_count)
                            continue

                        # Live delta from retry
                        if event.message and event.message.content:
                            for part in event.message.content:
                                if part.type == MessageContentType.TEXT and part.text:
                                    acc.has_content = True
                        yield event, acc

                    # ---------- nudge if retry also empty ----------
                    if (
                        not acc.has_content
                        and not acc.has_tool_calls
                        and not acc.final_content
                    ):
                        logger.warning(
                            "Retry also produced empty response — sending nudge prompt",
                            extra={"model": model_name},
                        )
                        nudge_messages = list(messages) + [
                            Message(
                                role=MessageRole.USER,
                                content=[
                                    MessageContent(
                                        type=MessageContentType.TEXT,
                                        text=_EMPTY_RESPONSE_NUDGE,
                                    )
                                ],
                            ),
                        ]
                        async for event in CompletionService._build_and_run(
                            user_id,
                            nudge_messages,
                            model_name,
                            workflow_type,
                            conversation_id,
                            client_tools,
                            "auto",
                            server_tool_names,
                        ):
                            if isinstance(event, ServerToolEvent):
                                continue
                            if event.done:
                                if event.message and event.message.tool_calls:
                                    acc.final_tool_calls = event.message.tool_calls
                                    acc.has_tool_calls = True
                                if event.message and event.message.content:
                                    parts = [
                                        c.text
                                        for c in event.message.content
                                        if c.type == MessageContentType.TEXT and c.text
                                    ]
                                    acc.final_content = "".join(parts)
                                if event.eval_count:
                                    acc.output_tokens += int(event.eval_count)
                                continue

                            if event.message and event.message.content:
                                for part in event.message.content:
                                    if (
                                        part.type == MessageContentType.TEXT
                                        and part.text
                                    ):
                                        acc.has_content = True
                            yield event, acc

        except asyncio.CancelledError:
            logger.warning("Stream cancelled (client disconnect) — stopping retries")
            return
        finally:
            if _queue_ctx is not None:
                await priority_queue.dequeue()

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
    ) -> CompletionResult:
        """Execute a workflow and return the final accumulated result.

        Handles continuation, empty-response retry, and nudge logic
        identically to the streaming path.
        """
        result = CompletionResult()

        # Priority queue integration
        from config import PRIORITY_QUEUE_ENABLED
        from services.priority_queue import priority_queue

        _effective_priority = priority if priority is not None else Priority.HIGH
        _queue_ctx = None
        if PRIORITY_QUEUE_ENABLED:
            _meta = RequestPriorityMetadata(
                source=RequestSource.USER,
                priority=_effective_priority,
                user_id=user_id,
                max_queue_wait=max_queue_wait,
            )
            _queue_ctx = await priority_queue.enqueue(_meta)

        try:

            # ---------- primary pass (with connection-error retry) ----------
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
                    result.chat_response = event

            if result.chat_response is None:
                return result

            # Filter server tool calls
            if (
                server_tool_names
                and result.chat_response.message
                and result.chat_response.message.tool_calls
            ):
                result.chat_response.message.tool_calls = [
                    tc
                    for tc in result.chat_response.message.tool_calls
                    if tc.name not in server_tool_names
                ]

            # ---------- truncation continuation ----------
            if result.has_content and not result.has_tool_calls:
                accumulated_text = "".join(
                    c.text
                    for c in result.chat_response.message.content  # type: ignore
                    if c.type == MessageContentType.TEXT and c.text
                )
                if _is_truncated(
                    accumulated_text, result.chat_response.finish_reason or ""
                ):
                    logger.info(
                        "Non-streaming: model response appears truncated — sending continuation prompt",
                        extra={
                            "content_len": len(accumulated_text),
                            "finish_reason": result.chat_response.finish_reason,
                        },
                    )
                    truncation_messages = list(messages) + [
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT, text=accumulated_text
                                )
                            ],
                        ),
                        Message(
                            role=MessageRole.USER,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=_TRUNCATION_CONTINUATION_PROMPT,
                                )
                            ],
                        ),
                    ]
                    async for event in CompletionService._build_and_run(
                        user_id,
                        truncation_messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        "auto",
                        server_tool_names,
                    ):
                        if isinstance(event, ServerToolEvent):
                            continue
                        if event.done and event.message:
                            # Merge continuation onto the original text. Without this,
                            # `result.chat_response = event` would replace the truncated
                            # original with only the continuation fragment.
                            cont_text = "".join(
                                c.text
                                for c in (event.message.content or [])
                                if c.type == MessageContentType.TEXT and c.text
                            )
                            merged = accumulated_text + cont_text
                            if event.message.content:
                                event.message.content = [
                                    MessageContent(
                                        type=MessageContentType.TEXT, text=merged
                                    ),
                                    *[
                                        c
                                        for c in event.message.content
                                        if c.type != MessageContentType.TEXT
                                    ],
                                ]
                            else:
                                event.message.content = [
                                    MessageContent(
                                        type=MessageContentType.TEXT, text=merged
                                    )
                                ]
                            if event.eval_count and result.chat_response:
                                event.eval_count = (
                                    result.chat_response.eval_count or 0
                                ) + int(event.eval_count)
                            result.chat_response = event

            # ---------- continuation check ----------
            # Skip when the model naturally stopped or was cut off by token
            # limit — see streaming path comment.
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
                accumulated_text = "".join(
                    c.text
                    for c in result.chat_response.message.content  # type: ignore
                    if c.type == MessageContentType.TEXT and c.text
                )
                if accumulated_text:
                    logger.info(
                        "Non-streaming: model produced text without tool calls — sending single continuation check",
                        extra={
                            "content_len": len(accumulated_text),
                            "content_preview": accumulated_text[:200],
                        },
                    )
                    continuation_messages = list(messages) + [
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT, text=accumulated_text
                                )
                            ],
                        ),
                        Message(
                            role=MessageRole.USER,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=_CONTINUATION_PROMPT,
                                )
                            ],
                        ),
                    ]
                    async for event in CompletionService._build_and_run(
                        user_id,
                        continuation_messages,
                        model_name,
                        workflow_type,
                        conversation_id,
                        client_tools,
                        "auto",
                        server_tool_names,
                    ):
                        if isinstance(event, ServerToolEvent):
                            continue
                        if event.done and event.message:
                            if event.message.tool_calls:
                                result.chat_response = event

            # ---------- empty-response retry ----------
            if (
                not result.has_content
                and not result.has_tool_calls
                and not result.is_error
            ):
                # Gather token info from the primary response for overflow detection.
                _primary_prompt_tokens = 0
                _primary_output_tokens = 0
                _primary_finish_reason = ""
                if result.chat_response:
                    _primary_prompt_tokens = int(
                        result.chat_response.prompt_eval_count or 0
                    )
                    _primary_output_tokens = int(result.chat_response.eval_count or 0)
                    _primary_finish_reason = result.chat_response.finish_reason or ""

                # Look up the model's context window for an accurate overflow check.
                _model_num_ctx = await CompletionService._get_model_num_ctx(model_name)

                # Skip retry if context is likely too large — retrying the
                # same oversized messages (or adding a nudge) is futile.
                if _is_context_overflow(
                    _primary_prompt_tokens,
                    _primary_finish_reason,
                    _primary_output_tokens,
                    model_num_ctx=_model_num_ctx,
                ):
                    logger.warning(
                        "Non-streaming: skipping retry — context likely exceeds model window",
                        extra={
                            "model": model_name,
                            "input_tokens": _primary_prompt_tokens,
                            "output_tokens": _primary_output_tokens,
                            "finish_reason": _primary_finish_reason,
                        },
                    )
                else:
                    logger.warning(
                        "Non-streaming: model produced empty response — retrying",
                        extra={"model": model_name},
                    )
                    empty_response_retries_total.inc()
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
                            result.chat_response = event

                    # ---------- nudge ----------
                    if not result.has_content and not result.has_tool_calls:
                        logger.warning(
                            "Non-streaming: retry also empty — sending nudge prompt",
                            extra={"model": model_name},
                        )
                        nudge_messages = list(messages) + [
                            Message(
                                role=MessageRole.USER,
                                content=[
                                    MessageContent(
                                        type=MessageContentType.TEXT,
                                        text=_EMPTY_RESPONSE_NUDGE,
                                    )
                                ],
                            ),
                        ]
                        async for event in CompletionService._build_and_run(
                            user_id,
                            nudge_messages,
                            model_name,
                            workflow_type,
                            conversation_id,
                            client_tools,
                            "auto",
                            server_tool_names,
                        ):
                            if isinstance(event, ServerToolEvent):
                                continue
                            if event.done and event.message:
                                result.chat_response = event

            return result
        finally:
            if _queue_ctx is not None:
                await priority_queue.dequeue()
