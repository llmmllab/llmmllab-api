"""Monkey-patch langchain-openai to surface reasoning_content from streaming deltas.

langchain-openai's ``_convert_delta_to_message_chunk`` (chat completions path)
reads ``content``, ``tool_calls``, and ``function_call`` from the streaming delta
but never reads ``reasoning_content`` — the field llama.cpp emits when
``--reasoning on --reasoning-format deepseek`` is active.  This means thinking
tokens from Qwen3.6 (and gemma-4 when enabled) are silently dropped before
reaching the executor's ``chunk.additional_kwargs.get("reasoning_content")``
check.

This patch adds reasoning_content to additional_kwargs so the full chain works:
  llama.cpp (reasoning_content in SSE delta)
  → langchain ChatOpenAI (additional_kwargs["reasoning_content"])
  → executor (MessageContentType.THINKING)
  → Anthropic SSE router (thinking_delta block)
  → openclaw client (thinking display)

Import this module early (before any ChatOpenAI usage) to apply the patch.
"""

import logging
from collections.abc import Mapping
from typing import Any, cast

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.messages.tool import tool_call_chunk

import langchain_openai.chat_models.base as _lc_base

logger = logging.getLogger(__name__)

_original_convert_delta = _lc_base._convert_delta_to_message_chunk  # noqa: F841


def _convert_delta_to_message_chunk_patched(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: dict = {}

    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call

    # ---- PATCH: surface reasoning_content from llama.cpp streaming deltas ----
    reasoning = _dict.get("reasoning_content") or _dict.get("reasoning")
    if reasoning:
        additional_kwargs["reasoning_content"] = reasoning
    # ---- END PATCH ----

    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        try:
            tool_call_chunks = [
                tool_call_chunk(  # type: ignore[call-arg]
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,
        )
    if role in ("system", "developer") or default_class == SystemMessageChunk:
        if role == "developer":
            additional_kwargs = {"__openai_role__": "developer"}
        else:
            additional_kwargs = {}
        return SystemMessageChunk(
            content=content, id=id_, additional_kwargs=additional_kwargs
        )
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)


# Apply the patch
_lc_base._convert_delta_to_message_chunk = _convert_delta_to_message_chunk_patched
logger.info("langchain-openai patched: reasoning_content now surfaced in streaming deltas")
