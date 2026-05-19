"""
Response-manipulation helpers used by CompletionService.

These functions are pure (apart from the in-place mutation of the dataclasses
in ``services.completion_state``). They handle:

* extracting text parts from a Message
* building follow-up message lists for continuation/nudge prompts
* filtering server-side tool calls out of a response
* applying a final ChatResponse to either a ``CompletionResult`` (non-streaming)
  or a ``StreamAccumulator`` (streaming)
"""

from models.chat_response import ChatResponse
from models.message import Message, MessageContent, MessageContentType, MessageRole
from models.tool_call import ToolCall
from services.completion_state import CompletionResult, StreamAccumulator


def extract_text(message: Message | None) -> str:
    """Return concatenated text parts from a message."""
    if not message or not message.content:
        return ""
    return "".join(
        part.text
        for part in message.content
        if part.type == MessageContentType.TEXT and part.text
    )


def build_followup_messages(
    messages: list[Message],
    prompt: str,
    assistant_text: str | None = None,
) -> list[Message]:
    """Create follow-up messages for continuation or nudge prompts."""
    followup_messages = list(messages)
    if assistant_text is not None:
        followup_messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=assistant_text,
                    )
                ],
            )
        )
    followup_messages.append(
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(type=MessageContentType.TEXT, text=prompt)
            ],
        )
    )
    return followup_messages


def filter_tool_calls(
    tool_calls: list[ToolCall] | None,
    server_tool_names: set[str] | None,
) -> list[ToolCall]:
    """Remove server-side tool calls from a tool call list."""
    if not tool_calls:
        return []
    if not server_tool_names:
        return list(tool_calls)
    return [tc for tc in tool_calls if tc.name not in server_tool_names]


def filter_response_tool_calls(
    response: ChatResponse | None,
    server_tool_names: set[str] | None,
) -> None:
    """Filter server-side tool calls in-place on a final response."""
    if not response or not response.message or not response.message.tool_calls:
        return
    response.message.tool_calls = filter_tool_calls(
        response.message.tool_calls,
        server_tool_names,
    )


def set_result_response(
    result: CompletionResult,
    response: ChatResponse,
    server_tool_names: set[str] | None,
) -> None:
    """Store a final response on the non-streaming result."""
    filter_response_tool_calls(response, server_tool_names)
    result.chat_response = response


def update_stream_delta(
    acc: StreamAccumulator,
    event: ChatResponse,
) -> None:
    """Update streaming accumulator state from a delta event."""
    if not event.message or not event.message.content:
        return
    for part in event.message.content:
        if part.type == MessageContentType.TEXT and part.text:
            acc.has_content = True
            return


def update_stream_final(
    acc: StreamAccumulator,
    event: ChatResponse,
    server_tool_names: set[str] | None,
    *,
    content_prefix: str = "",
    accumulate_output_tokens: bool = False,
    include_prompt_tokens: bool = False,
) -> None:
    """Update streaming accumulator state from a final event."""
    if event.finish_reason == "error":
        acc.is_error = True
    if event.finish_reason:
        acc.finish_reason = event.finish_reason
    if include_prompt_tokens and event.prompt_eval_count:
        acc.input_tokens = int(event.prompt_eval_count)
    if event.eval_count:
        if accumulate_output_tokens:
            acc.output_tokens += int(event.eval_count)
        else:
            acc.output_tokens = int(event.eval_count)
    if not event.message:
        return

    filtered_tool_calls = filter_tool_calls(
        event.message.tool_calls,
        server_tool_names,
    )
    acc.final_tool_calls = filtered_tool_calls
    acc.has_tool_calls = bool(filtered_tool_calls)

    final_text = extract_text(event.message)
    if final_text:
        acc.final_content = f"{content_prefix}{final_text}"
    elif content_prefix and not filtered_tool_calls:
        acc.final_content = content_prefix
