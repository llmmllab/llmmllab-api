import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Dict, Union, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
import regex

from middleware.auth import get_user_id
from models.request_priority_metadata import Priority, RequestSource
from services import (
    CompletionService,
    StreamAccumulator,
    TokenService,
    ToolService,
    model_service,
)
from graph.workflows.factory import WorkFlowType
from graph.workflows.ide.builder import IDE_PRIMARY_SYSTEM_PROMPT
from models.anthropic.create_message_request import CreateMessageRequest
from models.anthropic.message_response import MessageResponse
from models.anthropic.count_tokens_request import CountTokensRequest
from models.anthropic.count_tokens_response import CountTokensResponse
from models.anthropic.output_content_block import OutputContentBlock
from models.anthropic.text_content_block import TextContentBlock
from models.anthropic.tool_reference_content_block import ToolReferenceContentBlock

from models.anthropic.tool_use_content_block import ToolUseContentBlock
from models.anthropic.thinking_content_block import ThinkingContentBlock
from models.anthropic.usage import Usage
from models.message import Message, MessageRole, MessageContent, MessageContentType
from models.tool_call import ToolCall
from models.chat_response import ChatResponse
from graph.state import ServerToolEvent
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="anthropic_messages_router")
router = APIRouter(prefix="/messages", tags=["Messages"])

# Content block types that we inject into the SSE stream for server-side tool
# execution.  When the client sends these back on subsequent turns they will
# fail Pydantic validation (the request models don't know them).  We strip
# them from incoming messages before validation.
_SERVER_TOOL_BLOCK_TYPES = frozenset(
    {
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    }
)

# Thinking blocks returned by the API that clients echo back on subsequent
# turns.  Local models don't use these so we strip them before validation.
_THINKING_BLOCK_TYPES = frozenset(
    {
        "thinking",
        "redacted_thinking",
    }
)


def _strip_server_tool_blocks(req_body: Dict[str, Any]) -> Dict[str, Any]:
    """Remove server-tool and thinking content blocks that the client echoed back."""
    messages = req_body.get("messages")
    if not messages:
        return req_body

    strip_types = _SERVER_TOOL_BLOCK_TYPES | _THINKING_BLOCK_TYPES

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        filtered = [
            block
            for block in content
            if not (isinstance(block, dict) and block.get("type") in strip_types)
        ]
        if not filtered:
            # Don't leave an empty content list — replace with placeholder text.
            filtered = [{"type": "text", "text": "(content omitted)"}]
        msg["content"] = filtered

    return req_body


def _sse(event_type: str, data: dict) -> str:
    """Format a server-sent event with the required Anthropic event/data structure."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def messages_from_anthropic(
    anthropic_messages: list,
    system: Any = None,
) -> list[Message]:
    """Convert Anthropic messages to internal Message format.

    Handles:
    - String content
    - Text and tool_use blocks in assistant messages
    - tool_result blocks in user messages (expanded to TOOL role messages)
    - System prompt (prepended as SYSTEM message)
    """
    messages: list[Message] = []

    # Prepend system message if present
    if system is not None:
        if isinstance(system, str):
            system_text = system
        else:
            # List of TextContentBlock
            system_text = "\n".join(
                block.text for block in system if hasattr(block, "text") and block.text
            )
        if system_text:
            messages.append(
                Message(
                    role=MessageRole.SYSTEM,
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text=system_text)
                    ],
                )
            )

    for msg in anthropic_messages:
        content = msg.content

        # Simple string content
        if isinstance(content, str):
            role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
            messages.append(
                Message(
                    role=role,
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text=content)
                    ],
                )
            )
            continue

        # List of content blocks
        if msg.role == "user":
            tool_result_blocks = [
                b for b in content if hasattr(b, "type") and b.type == "tool_result"
            ]
            if tool_result_blocks:
                # Each tool_result block becomes a separate TOOL message (mirrors OAI tool messages)
                for block in tool_result_blocks:
                    result_text = ""
                    if isinstance(block.content, str):
                        result_text = block.content
                    elif isinstance(block.content, list):
                        # Handle mixed content: text, tool_reference, etc.
                        parts = []
                        for item in block.content:
                            if hasattr(item, "text") and item.text:
                                parts.append(item.text)
                            elif isinstance(item, ToolReferenceContentBlock):
                                # Format tool reference as readable text
                                parts.append(f"[Tool: {item.tool_name}]")
                        result_text = "\n".join(parts)
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT, text=result_text
                                )
                            ],
                            tool_calls=[
                                ToolCall(
                                    name="tool_result",
                                    execution_id=block.tool_use_id,
                                    args={},
                                )
                            ],
                        )
                    )
                # Handle any non-tool_result text blocks in the same user message
                other_text = [
                    b.text
                    for b in content
                    if hasattr(b, "type")
                    and b.type == "text"
                    and hasattr(b, "text")
                    and b.text
                ]
                if other_text:
                    messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text="\n".join(other_text),
                                )
                            ],
                        )
                    )
                continue

        # Regular user or assistant message with text and/or tool_use blocks
        text_contents: list[MessageContent] = []
        tool_calls: list[ToolCall] | None = None

        for block in content:
            if not hasattr(block, "type"):
                continue
            if block.type == "text":
                text_contents.append(
                    MessageContent(type=MessageContentType.TEXT, text=block.text)
                )
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        execution_id=block.id,
                        name=block.name,
                        args=block.input if isinstance(block.input, dict) else {},
                    )
                )

        role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
        if text_contents or tool_calls:
            messages.append(
                Message(
                    role=role,
                    content=text_contents
                    or [MessageContent(type=MessageContentType.TEXT, text="")],
                    tool_calls=tool_calls,
                )
            )

    return messages


def anthropic_response_from_chat_response(
    chat_response: ChatResponse,
    model: str = "unknown",
    stop_reason: str = "end_turn",
) -> MessageResponse:
    """Convert internal ChatResponse to Anthropic MessageResponse format."""

    content_blocks: list[OutputContentBlock] = []

    # Thinking blocks first (per Anthropic spec ordering)
    if chat_response.message and chat_response.message.thoughts:
        for thought in chat_response.message.thoughts:
            content_blocks.append(
                ThinkingContentBlock(
                    type="thinking", thinking=thought.text if thought.text else ""
                )
            )

    # Text blocks
    if chat_response.message and chat_response.message.content:
        for part in chat_response.message.content:
            if part.type == MessageContentType.TEXT and part.text:
                content_blocks.append(TextContentBlock(type="text", text=part.text))

    # Tool use blocks
    if chat_response.message and chat_response.message.tool_calls:
        for tc in chat_response.message.tool_calls:
            content_blocks.append(
                ToolUseContentBlock(
                    type="tool_use",
                    id=tc.execution_id or f"toolu_{uuid.uuid4().hex[:24]}",
                    name=tc.name,
                    input=tc.args,
                )
            )

    usage = Usage(
        input_tokens=int(chat_response.prompt_eval_count or 0),
        output_tokens=int(chat_response.eval_count or 0),
    )

    valid_stop_reasons = [
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "pause_turn",
    ]
    actual_stop_reason = (
        stop_reason if stop_reason in valid_stop_reasons else "end_turn"
    )

    return MessageResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        type="message",
        role="assistant",
        content=content_blocks,
        model=model,
        stop_reason=actual_stop_reason,  # type: ignore
        usage=usage,
    )


def _server_tools_enabled_from_header(request: Request) -> bool | None:
    """Parse the per-request ``X-Server-Side-Tools`` override.

    Returns ``True`` / ``False`` when the client explicitly asks for or
    opts out of server-side tool execution; ``None`` when the header is
    absent (caller should fall back to the env default).
    """
    raw = request.headers.get("x-server-side-tools")
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in {"0", "false", "no", "off"}:
        return False
    if val in {"1", "true", "yes", "on"}:
        return True
    return None  # malformed → ignore, fall back to default


async def stream_message(
    user_id: str,
    messages: list[Message],
    model_name: str,
    client_tools: list | None = None,
    tool_choice: str | None = None,
    priority: Priority | None = None,
    max_queue_wait: float | None = None,
    source: RequestSource | None = None,
    session_id: str | None = None,
    server_tools_enabled: bool | None = None,
) -> AsyncIterator[str]:
    """Stream composer events as Anthropic SSE message chunks.

    Emits the full Anthropic streaming event sequence:
      message_start → ping → content_block_start → content_block_delta(s)
      → content_block_stop → message_delta → message_stop

    Retry, continuation, and nudge logic is handled by CompletionService.
    Server-tool separation is handled by ToolService.

    ``server_tools_enabled`` overrides ``config.SERVER_SIDE_TOOLS_ENABLED``
    for this request (driven by the ``X-Server-Side-Tools`` header).
    """
    # Prepare tools via shared service
    prepared = ToolService.prepare_tools(client_tools, enabled=server_tools_enabled)
    client_tools = prepared.client_tools
    server_tool_names = prepared.server_tool_names

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Resolve model name before building workflow to handle unavailable models.
    # This ensures we use a fallback model if the requested model is not found.
    resolved_model = await model_service.resolve_default_model(model_name, user_id)
    if resolved_model:
        model_name = resolved_model
        logger.info(
            "Resolved model name",
            extra={
                "original": model_name,
                "resolved": resolved_model,
            },
        )

    # Build workflow first so we have a server for accurate token counting.
    # The workflow is cached, so stream_completion's internal build will
    # hit the cache and return immediately.
    _, _, server_url = await CompletionService.build_workflow(
        user_id=user_id,
        model_name=model_name,
        workflow_type=WorkFlowType.IDE,
        client_tools=client_tools,
        tool_choice=tool_choice,
        server_tool_names=server_tool_names or None,
    )

    # Count tokens using the real llama.cpp tokenizer
    input_tokens = await TokenService.count_input_tokens(
        messages,
        client_tools,
        server_url=server_url,
        system_prompt=IDE_PRIMARY_SYSTEM_PROMPT,
    )

    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )
    yield _sse("ping", {"type": "ping"})

    text_block_started = False
    text_block_index = 0
    next_block_index = 0
    output_tokens = 0
    acc = StreamAccumulator()

    try:
        async for event, acc in CompletionService.stream_completion(
            user_id=user_id,
            messages=messages,
            model_name=model_name,
            client_tools=client_tools,
            tool_choice=tool_choice,
            server_tool_names=server_tool_names or None,
            priority=priority,
            max_queue_wait=max_queue_wait,
            source=source,
            session_id=session_id,
        ):
            # ---- ServerToolEvent → emit as standard text content blocks ----
            if isinstance(event, ServerToolEvent):
                if text_block_started:
                    yield _sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": text_block_index},
                    )
                    text_block_started = False

                tc = event.tool_call
                canonical = event.canonical_name
                result_text = event.result_text or ""

                if canonical == "web_search":
                    query = tc.args.get("query", "")
                    tool_summary = (
                        f"\n\n🔍 **Web Search:** {query}\n\n{result_text}\n\n"
                    )
                elif canonical == "web_fetch":
                    url = tc.args.get("url", "")
                    tool_summary = f"\n\n📄 **Web Fetch:** {url}\n\n{result_text}\n\n"
                else:
                    tool_summary = f"\n\n🔧 **{tc.name}**\n\n{result_text}\n\n"

                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": next_block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": next_block_index,
                        "delta": {"type": "text_delta", "text": tool_summary},
                    },
                )
                yield _sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": next_block_index},
                )
                next_block_index += 1
                continue

            # ---- ChatResponse events ----
            if event.done:
                if event.prompt_eval_count:
                    input_tokens = int(event.prompt_eval_count)
                if event.eval_count:
                    output_tokens = int(event.eval_count)
                continue

            # Stream live text deltas
            if event.message and event.message.content:
                for part in event.message.content:
                    if part.type == MessageContentType.TEXT and part.text:
                        if not text_block_started:
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": next_block_index,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                            text_block_index = next_block_index
                            next_block_index += 1
                            text_block_started = True
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_block_index,
                                "delta": {"type": "text_delta", "text": part.text},
                            },
                        )

    except asyncio.CancelledError:
        logger.warning("Client disconnected — stream_message cancelled")
        return

    # Use the accumulator for final state
    output_tokens = acc.output_tokens or output_tokens

    # Fallback: emit content from done event if nothing was streamed live
    if not acc.has_content and not acc.has_tool_calls and acc.final_content:
        if not text_block_started:
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": next_block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            text_block_index = next_block_index
            next_block_index += 1
            text_block_started = True
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": text_block_index,
                "delta": {"type": "text_delta", "text": acc.final_content},
            },
        )

    # Determine stop_reason for the message_delta event.
    stop_reason: str | None = None

    # Final fallback: all retries produced nothing.  Do NOT inject a
    # diagnostic string into the assistant's response stream — clients
    # echo that text back as the assistant's message on the next turn,
    # which (a) pollutes the conversation history with our error text,
    # (b) compounds when consecutive turns also produce empty (each new
    # turn appends another copy), and (c) eventually exhausts output
    # token budgets.  Instead, surface stop_reason="error" with empty
    # content so the caller can detect failure cleanly.
    if not acc.has_content and not acc.has_tool_calls and not acc.final_content and acc.finish_reason != "stop":
        logger.warning(
            "All retries produced empty response — returning error stop_reason",
            extra={
                "model": model_name,
                "input_tokens": input_tokens,
                "finish_reason": acc.finish_reason,
            },
        )
        # Anthropic protocol doesn't define an "error" stop_reason, but
        # "max_tokens" with zero output content is the closest signal
        # that the generation failed.  No text content is emitted.
        stop_reason = "max_tokens"

    # Close the text block
    if text_block_started:
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": text_block_index},
        )

    # Emit tool_use blocks
    tool_block_start = next_block_index
    for i, tc in enumerate(acc.final_tool_calls):
        block_index = tool_block_start + i
        tc_id = tc.execution_id or f"toolu_{uuid.uuid4().hex[:24]}"

        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tc_id,
                    "name": tc.name,
                    "input": {},
                },
            },
        )
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(tc.args),
                },
            },
        )
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )

    if stop_reason is None:
        if acc.has_tool_calls:
            stop_reason = "tool_use"
        elif acc.finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"
    logger.debug(
        "Stream complete",
        extra={
            "has_content": acc.has_content,
            "has_tool_calls": acc.has_tool_calls,
            "final_content_len": len(acc.final_content),
            "stop_reason": stop_reason,
            "finish_reason": acc.finish_reason,
            "text_block_started": text_block_started,
        },
    )
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


@router.post("", response_model=None)
async def createMessage(
    req_body: Dict[str, Any],
    request: Request,
) -> Union[MessageResponse, StreamingResponse]:
    """Operation ID: createMessage"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    logger.debug("Anthropic headers", extra={"headers": dict(request.headers)})

    try:
        req_body = _strip_server_tool_blocks(req_body)
        body = CreateMessageRequest.model_validate(req_body)
        internal_messages = messages_from_anthropic(body.messages, system=body.system)
        _priority_meta = getattr(request.state, "request_priority_metadata", None)
        priority = getattr(_priority_meta, "priority", None) if _priority_meta else None
        max_queue_wait = (
            getattr(_priority_meta, "max_queue_wait", None) if _priority_meta else None
        )
        req_source = getattr(_priority_meta, "source", None) if _priority_meta else None
        req_session_id = (
            getattr(_priority_meta, "session_id", None) if _priority_meta else None
        )

        # Resolve model: fall back to user's default_model if unavailable
        resolved_model = await model_service.resolve_default_model(body.model, user_id)
        if resolved_model:
            body.model = resolved_model

        client_tools = None
        tool_choice = None
        if body.tools:
            raw_client_tools = [
                tool.model_dump(exclude_none=True) for tool in body.tools
            ]
            client_tools = raw_client_tools

            if body.tool_choice:
                if body.tool_choice.type == "any":
                    tool_choice = "required"
                elif body.tool_choice.type == "auto":
                    tool_choice = "auto"
                elif body.tool_choice.type == "tool":
                    tool_choice = "tool"
            logger.info(
                "Anthropic request with tools",
                extra={
                    "tool_count": len(body.tools),
                    "tool_names": [t.name for t in body.tools],
                    "client_tools_created": len(client_tools),
                    "tool_choice": tool_choice,
                },
            )
        else:
            raw_client_tools = None

        server_tools_enabled = _server_tools_enabled_from_header(request)

        if body.stream:
            return StreamingResponse(
                stream_message(
                    user_id,
                    internal_messages,
                    body.model,
                    client_tools=raw_client_tools,
                    tool_choice=tool_choice,
                    priority=priority,
                    max_queue_wait=max_queue_wait,
                    source=req_source,
                    session_id=req_session_id,
                    server_tools_enabled=server_tools_enabled,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming path — delegate to CompletionService
        prepared = ToolService.prepare_tools(client_tools, enabled=server_tools_enabled)
        try:
            result = await CompletionService.run_completion(
                user_id=user_id,
                messages=internal_messages,
                model_name=body.model,
                client_tools=prepared.client_tools,
                tool_choice=tool_choice,
                server_tool_names=prepared.server_tool_names or None,
                priority=priority,
                max_queue_wait=max_queue_wait,
                source=req_source,
                session_id=req_session_id,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if any(
                kw in error_msg
                for kw in ("connection", "runner", "unavailable", "refused", "protocol")
            ):
                # Generally fires when the runner is loading a model for
                # the first time this session and the http client timed
                # out waiting (model load can take 45-90s on cold start).
                # Retry once the model is loaded.
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Runner busy starting the model. This usually means a "
                        "model server is still loading (~45-90s on cold start). "
                        "Please retry in 30-60 seconds."
                    ),
                ) from e
            if "model" in error_msg and "not found" in error_msg:
                raise HTTPException(
                    status_code=404,
                    detail=f"Requested model '{body.model}' is not available. Please use a model that is available on a runner.",
                ) from e
            raise

        if result.chat_response is None:
            raise HTTPException(
                status_code=503,
                detail="Model returned an empty response.",
            )

        if (
            not result.has_content
            and not result.has_tool_calls
            and result.chat_response.finish_reason != "stop"
        ):
            if getattr(result, "context_overflow", False):
                raise HTTPException(
                    status_code=507,
                    detail="Context window exceeded. Please reduce conversation length or use a model with larger context.",
                )
            raise HTTPException(
                status_code=503,
                detail="Model returned an empty response.",
            )

        stop_reason_map: dict[str | None, str] = {
            "stop": "end_turn",
            "complete": "end_turn",
            "length": "max_tokens",
            "tool_call": "tool_use",
        }
        stop_reason = stop_reason_map.get(
            result.chat_response.finish_reason, "end_turn"
        )

        return anthropic_response_from_chat_response(
            result.chat_response, model=body.model, stop_reason=stop_reason
        )
    except ValidationError as ve:
        logger.error(f"Validation error in createMessage request: {ve.json()}")
        raise HTTPException(status_code=422, detail=json.loads(ve.json())) from ve

    except HTTPException:
        # Inner code already raised an HTTPException with the correct status
        # code (e.g. 503 for runner-unavailable, 404 for unknown model).
        # Propagate it unchanged — DO NOT wrap it in a 400, which produces
        # the confusing "400 {detail: '503: ...'}" surface bug.
        raise

    except Exception as e:
        logger.error(f"Error processing createMessage request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/count_tokens")
async def countTokens(
    request: Request,
    body: CountTokensRequest,
) -> CountTokensResponse:
    """Operation ID: countTokens

    Estimates the token count for a message request by forwarding the
    rendered text to the running llama-server's /tokenize endpoint.
    """
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    try:
        internal_messages = messages_from_anthropic(body.messages, system=body.system)
        raw_count = await TokenService.count_input_tokens(internal_messages, body.tools)
        return CountTokensResponse(input_tokens=raw_count)

    except Exception as e:
        logger.error(f"Error in countTokens: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
