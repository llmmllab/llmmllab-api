"""
Response-manipulation helpers used by CompletionService.

These functions are pure (apart from the in-place mutation of the dataclasses
in ``services.completion_state``). They handle:

* extracting text parts from a Message
* building follow-up message lists for continuation/nudge prompts
* filtering server-side tool calls out of a response
* applying a final ChatResponse to either a ``CompletionResult`` (non-streaming)
  or a ``StreamAccumulator`` (streaming)
* detecting hallucinated tool names (names not in the bound tool list)
"""

import re

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


def extract_client_tool_names(client_tools: list | None) -> set[str]:
    """Return the set of bound tool names from a heterogeneous tool list.

    Tools can arrive in several shapes depending on the entry router:

    * **Anthropic flat**: ``{"name": "...", "description": "...", "input_schema": ...}``
    * **Anthropic server-tool**: ``{"type": "web_search_20250305", ...}`` —
      derive the name by stripping the version suffix.
    * **OpenAI function**: ``{"type": "function", "function": {"name": "...", ...}}``
    * **LangChain BaseTool**: any object with a ``.name`` attribute.

    Used by :func:`partition_tool_calls_by_validity` to detect when the
    model emits a tool call with a name we never offered it.  The empty
    set means "no tools bound" — partitioning is a no-op in that case.
    """
    names: set[str] = set()
    if not client_tools:
        return names
    for tool in client_tools:
        if isinstance(tool, dict):
            # OpenAI function form.
            fn = tool.get("function")
            if isinstance(fn, dict) and fn.get("name"):
                names.add(fn["name"])
                continue
            # Direct name (Anthropic flat).
            n = tool.get("name")
            if n:
                names.add(n)
                continue
            # Server-tool form: derive from type, strip ``_NNNN`` version.
            tool_type = tool.get("type", "")
            if tool_type and tool_type not in {"function", "custom", "tool"}:
                base = re.sub(r"_\d+$", "", tool_type)
                if base:
                    names.add(base)
            continue
        # LangChain BaseTool / pydantic model with .name.
        n = getattr(tool, "name", None)
        if n:
            names.add(n)
    return names


def partition_tool_calls_by_validity(
    tool_calls: list[ToolCall] | None,
    valid_names: set[str],
) -> tuple[list[ToolCall], list[ToolCall]]:
    """Split ``tool_calls`` into ``(valid, invalid)`` by ``valid_names``.

    Used to detect model hallucinations.  When the model emits a tool
    call with a name that wasn't in the bound list (commonly ``browser``
    or ``memory_get`` — both pretraining-frequent generic tool names),
    the client silently drops the call, the model never gets a
    tool_result, and the conversation degenerates into a "try again
    with the same imagined name" loop.

    Returns ``(all_calls, [])`` when ``valid_names`` is empty — caller
    didn't supply bound names, so we can't validate.  Otherwise: any
    call whose ``name`` isn't in ``valid_names`` lands in the invalid
    bucket.
    """
    if not tool_calls:
        return [], []
    if not valid_names:
        return list(tool_calls), []
    valid: list[ToolCall] = []
    invalid: list[ToolCall] = []
    for tc in tool_calls:
        if tc.name in valid_names:
            valid.append(tc)
        else:
            invalid.append(tc)
    return valid, invalid


# A heuristic for "the model wrote a tool-call precursor and then
# stopped before actually calling the tool."
#
# Two patterns trigger:
#   1. The tail of the content contains a setup verb phrase
#      (``Let me``, ``I'll``, ``Going to``, ``Now``, ``Next``, etc.)
#      AND ends with a colon, ellipsis, or no terminator at all.
#   2. The tail ends with an ellipsis (``...``) regardless of phrase.
#
# A complete sentence ending with ``.`` is NOT flagged — that would
# capture every final answer.  Only the colon / ellipsis / open-ended
# endings combined with a setup phrase are treated as anticipation.
_SETUP_PHRASE_RE = re.compile(
    r"\b(?:"
    r"now\s+let(?:'|’)?s|"
    r"now\s+let\s+me|"
    r"let\s+me|"
    r"let(?:'|’)?s|"
    r"i(?:'|’)?ll|"
    r"i\s+will|"
    r"i(?:'|’)?m\s+going\s+to|"
    r"going\s+to|"
    r"(?:^|[\.\n])\s*next\b|"
    r"(?:^|[\.\n])\s*now\b"
    r")\b",
    re.IGNORECASE,
)
_OPEN_ENDED_TAIL_RE = re.compile(r"(?:[:\(]|\.{2,})\s*$")
_TRAILING_ELLIPSIS_RE = re.compile(r"\.{3,}\s*$")


def looks_like_anticipated_tool_call(text: str | None) -> bool:
    """True if ``text`` reads like a setup phrase for a tool call.

    The model often writes "Now let me do X:" and then stops without
    actually calling the tool — usually a sampler hiccup, not a final
    answer.  Without the explicit ``[TOOL_INTENT:`` marker the system
    prompt asks for, the existing gate accepts those turns as terminal.
    This heuristic fills the gap.

    Triggers when:
      * the tail contains a setup verb phrase AND ends with ``:`` or
        ``...`` (or with no terminator at all), OR
      * the tail ends with a trailing ``...``

    Returns False for sentences ending in ``.`` — that's a normal
    final answer, not an anticipated tool call.  False positives are
    further bounded because the nudge fires once per turn.
    """
    if not text:
        return False
    stripped = text.rstrip()
    if not stripped:
        return False
    # Last 250 chars is plenty; longer responses are typically final
    # answers (precursors are short by nature).
    tail = stripped[-250:]

    # A bare ``...`` at the end is a clear continuation marker on its
    # own, regardless of verb phrase.
    if _TRAILING_ELLIPSIS_RE.search(tail):
        return True

    # Otherwise: require a setup verb phrase AND either a colon /
    # open paren / ellipsis ending, or no sentence terminator at all
    # (i.e. mid-thought stop).
    if not _SETUP_PHRASE_RE.search(tail):
        return False
    if _OPEN_ENDED_TAIL_RE.search(tail):
        return True
    # Trailing char other than a sentence terminator → mid-thought stop.
    last_char = tail[-1]
    return last_char not in ".!?\"')]}"


# Max stripped length of a finish=stop answer that still counts as a
# "premature" bail when the conversation is mid-tool-loop.  Deliberately
# tiny — catches degenerate one-word stops ("I", "Done", "Ok", "Let me")
# without re-prompting genuine short final answers.  A real answer to a
# user question is rarely this short AND mid-tool-loop at the same time.
_PREMATURE_STOP_MAX_CHARS = 16


def _last_turn_is_tool_result(messages: list | None) -> bool:
    """True when the most recent non-assistant message is a tool result.

    Handles both wire shapes:
      * OpenAI: a message with ``role == TOOL``
      * Anthropic: a ``USER`` message carrying ``TOOL_RESULT`` content
        blocks (tool results are folded into user turns there)

    Used to confirm we're mid-agentic-loop before treating a tiny
    finish=stop answer as a premature bail rather than a final reply.
    """
    if not messages:
        return False
    for msg in reversed(messages):
        role = getattr(msg, "role", None)
        # Skip the model's own just-emitted assistant turn(s).
        if role in (MessageRole.ASSISTANT, MessageRole.AGENT):
            continue
        if role == MessageRole.TOOL:
            return True
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for part in content:
                ptype = getattr(part, "type", None) or (
                    part.get("type") if isinstance(part, dict) else None
                )
                if ptype in (MessageContentType.TOOL_RESULT, "tool_result"):
                    return True
        # First non-assistant message decides; stop scanning.
        return False
    return False


def looks_like_premature_stop(
    final_content: str | None, messages: list | None
) -> bool:
    """True when a ``finish=stop`` turn looks like the model bailed mid-task.

    The degenerate failure mode (observed repeatedly on Qwen3.6-27B at
    deep context): after a tool result the model emits a one-token answer
    like ``"I"`` or ``"Done"`` and ends the turn cleanly, forcing the user
    to manually prompt it to continue.  This is NOT caught by the
    ``[TOOL_INTENT:`` marker (the model doesn't emit it) nor by
    :func:`looks_like_anticipated_tool_call` (no setup phrase).

    Tight gate to avoid re-prompting genuine short answers:
      * the visible answer is <= ``_PREMATURE_STOP_MAX_CHARS`` chars, AND
      * the conversation is mid-tool-loop (last non-assistant message is
        a tool result) — i.e. the model was clearly expected to keep
        working, not deliver a final reply.
    """
    text = (final_content or "").strip()
    if not text or len(text) > _PREMATURE_STOP_MAX_CHARS:
        return False
    return _last_turn_is_tool_result(messages)


_SUMMARY_MIN_CONTENT_LEN = 200


def looks_like_missing_summary(final_content: str | None) -> bool:
    """True when a finish=stop response lacks the ``## !SUMMARY!`` marker.

    Only flags responses long enough (> 200 chars) that a conclusion
    summary would be expected.  Short answers to simple questions pass
    through without a nudge.
    """
    from services.prompt_templates import SUMMARY_MARKER

    text = (final_content or "").strip()
    if not text or len(text) <= _SUMMARY_MIN_CONTENT_LEN:
        return False
    return SUMMARY_MARKER not in text


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
