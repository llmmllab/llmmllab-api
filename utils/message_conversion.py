"""
Unified message conversion utilities for converting between internal Message objects and LangChain BaseMessage types.

This module consolidates all message conversion logic to eliminate duplicate implementations
and provide a single source of truth for message format conversion.
"""

import json
import re
from typing import List, Optional, Union, Dict, Any
from datetime import datetime, timezone

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    AnyMessage,
)

from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    Document,
)
from .logging import llmmllogger
from .tool_call_types import is_langchain_tool_call
from .tool_call_extraction import extract_tool_calls_from_langchain_message
from .file_handler import (
    decode_and_save_image,
    extract_text_from_file,
    is_image_format,
)
from .data_uri_utils import (
    extract_base64_from_data_uri,
    extract_mime_type_from_data_uri,
    create_data_uri,
    is_data_uri,
    get_decoded_data,
)
from .text_extraction import extract_text_content

logger = llmmllogger.bind(component="message_conversion")

MessageInput = Union[str, Message, List[Union[str, Message]], List[str], List[Message]]


def _get_file_extras(content_item: MessageContent) -> Dict[str, Any]:
    """Extract extra metadata for file content blocks."""
    extras = {}
    if content_item.name:
        extras["filename"] = content_item.name
    return {"extras": extras} if extras else {}


_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


# How many of the most-recent ToolMessages to keep image data intact for.
# Older ones get their base64 image blobs replaced with size stubs so they
# stop occupying context window space.  Two is enough for the model to
# correlate a current screenshot with the immediately-preceding state.
_KEEP_RECENT_TOOL_IMAGES = 2

# Catches base64 inside a data URI (e.g. ``data:image/png;base64,iVBOR...``).
# We preserve the ``data:image/...;base64,`` prefix so the placeholder
# stays recognisable as having been an image.
_DATA_URI_IMAGE_RE = re.compile(
    r"(data:image/[a-zA-Z0-9.+-]+;base64,)([A-Za-z0-9+/=]{200,})"
)

# Catches bare base64 image-sized runs (no data URI) anywhere in the text.
# 2 KB lower bound on the base64 string excludes ordinary text and
# short fingerprints while reliably catching real screenshots (smallest
# practical PNG is well above this).
_BARE_B64_BLOB_RE = re.compile(r"[A-Za-z0-9+/]{2000,}={0,2}")


def _b64_stub(decoded_size_kb: int) -> str:
    return f"[image redacted from older turn: ~{decoded_size_kb} KB base64]"


def _redact_image_blobs_in_string(content: str) -> str:
    """Replace large base64 runs in a tool-result string with size stubs.

    Used to shrink older ``ToolMessage`` content so MCP servers that
    return inline screenshots (FreeCAD ``get_view``, Blender viewport
    captures, etc.) don't accumulate into context overflow over many
    turns.  Recent images are left alone — see
    :func:`_redact_old_tool_images`.
    """
    def _data_uri_repl(m: re.Match) -> str:
        size_kb = (len(m.group(2)) * 3) // 4 // 1024  # approx decoded size
        return f"{m.group(1)}{_b64_stub(size_kb)}"

    def _bare_repl(m: re.Match) -> str:
        size_kb = (len(m.group(0)) * 3) // 4 // 1024
        return _b64_stub(size_kb)

    content = _DATA_URI_IMAGE_RE.sub(_data_uri_repl, content)
    return _BARE_B64_BLOB_RE.sub(_bare_repl, content)


def _redact_image_blobs_in_content(
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """Apply :func:`_redact_image_blobs_in_string` to string or list content.

    For list content, drops Anthropic-style ``image`` blocks entirely
    (replacing each with a text block carrying the size stub) and
    scrubs any base64 runs that appear inside text blocks.
    """
    if isinstance(content, str):
        return _redact_image_blobs_in_string(content)

    cleaned: List[Union[str, Dict[str, Any]]] = []
    for item in content:
        if isinstance(item, dict):
            t = item.get("type")
            if t == "image":
                # Anthropic-style image block: {"type":"image","source":
                # {"type":"base64","media_type":"image/png","data":"..."}}
                src = item.get("source") or {}
                data = src.get("data") or ""
                size_kb = (len(data) * 3) // 4 // 1024
                cleaned.append({"type": "text", "text": _b64_stub(size_kb)})
            elif t == "image_url":
                # OpenAI-style image_url block.
                url = (item.get("image_url") or {}).get("url") or ""
                size_kb = (len(url) * 3) // 4 // 1024
                cleaned.append({"type": "text", "text": _b64_stub(size_kb)})
            elif t == "text" and isinstance(item.get("text"), str):
                cleaned.append(
                    {**item, "text": _redact_image_blobs_in_string(item["text"])}
                )
            else:
                cleaned.append(item)
        elif isinstance(item, str):
            cleaned.append(_redact_image_blobs_in_string(item))
        else:
            cleaned.append(item)
    return cleaned


def _redact_old_tool_images(messages: List[AnyMessage]) -> List[AnyMessage]:
    """Strip large base64 image blobs from all but the most recent N tool messages.

    MCP servers that return screenshots (FreeCAD's ``get_view``, Blender
    viewport captures, image-generation pipelines) embed base64 inline,
    and a single 1024×1024 PNG is ~85k tokens.  Three such calls
    accumulated in history burns the context window without any actual
    new information — the model only needs the *current* view, not
    every prior one.

    We keep the last :data:`_KEEP_RECENT_TOOL_IMAGES` tool messages
    untouched; older ones get their base64 replaced with size stubs.
    """
    tool_indices = [
        i for i, m in enumerate(messages) if isinstance(m, ToolMessage)
    ]
    if len(tool_indices) <= _KEEP_RECENT_TOOL_IMAGES:
        return messages
    to_scrub = tool_indices[: -_KEEP_RECENT_TOOL_IMAGES]
    redacted_count = 0
    for idx in to_scrub:
        msg = messages[idx]
        original = msg.content
        scrubbed = _redact_image_blobs_in_content(original)
        if scrubbed != original:
            msg.content = scrubbed  # type: ignore[assignment]
            redacted_count += 1
    if redacted_count:
        logger.info(
            "Redacted base64 image blobs from older tool messages",
            extra={
                "redacted_messages": redacted_count,
                "total_tool_messages": len(tool_indices),
                "kept_recent": _KEEP_RECENT_TOOL_IMAGES,
            },
        )
    return messages


def _strip_think_tags_from_content(
    content_data: Union[str, List[Union[str, Dict[str, Any]]]],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """Remove <think>/<​/think> tags from assistant message content.

    Handles both simple string content and multimodal list-of-dicts format.
    """
    if isinstance(content_data, str):
        return _THINK_TAG_RE.sub("", content_data).strip()

    cleaned: List[Union[str, Dict[str, Any]]] = []
    for item in content_data:
        if isinstance(item, dict) and item.get("type") == "text":
            text = _THINK_TAG_RE.sub("", item.get("text", "")).strip()
            if text:
                cleaned.append({**item, "text": text})
        elif isinstance(item, str):
            text = _THINK_TAG_RE.sub("", item).strip()
            if text:
                cleaned.append(text)
        else:
            cleaned.append(item)
    return cleaned


def message_to_lc_message(
    message: Message, use_llama_format: bool = False
) -> AnyMessage:
    """Convert a Message object to a LangChain BaseMessage, preserving multimodal content.

    Args:
        message: Message object to convert
        use_llama_format: If True, use llama.cpp compatible format (images as URLs, files as text)
                         If False, use OpenAI compatible format (base64 encoded content)
    """

    # Convert Message.content to the appropriate multimodal format
    if use_llama_format:
        content_data = convert_message_content_to_llama_format(message.content)
    else:
        content_data = convert_message_content_to_langchain_format(message.content)

    # Strip residual think tags from assistant content to prevent poisoning
    # the model's context (a prior turn may have leaked </think> as content).
    if message.role in (MessageRole.ASSISTANT, MessageRole.AGENT):
        content_data = _strip_think_tags_from_content(content_data)

    # For assistant messages, also parse XML tool calls from text content
    parsed_tool_calls = []
    lc_id = str(message.id) if message.id is not None else None
    if message.role == MessageRole.ASSISTANT or message.role == MessageRole.AGENT:
        # First, use structured tool_calls from the Message object.
        # These come from Copilot's conversation history where assistant
        # messages carry tool_calls (preserved by messages_from_openai).
        # They have proper IDs that must match subsequent ToolMessage entries.
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc.name and tc.name != "tool_result":
                    lc_tool_call = {
                        "name": tc.name,
                        "args": tc.args if tc.args else {},
                        "id": tc.execution_id
                        or f"call_{tc.name}_{len(parsed_tool_calls)}",
                    }
                    parsed_tool_calls.append(lc_tool_call)

        # Fall back to parsing XML-wrapped tool calls from text content
        # (for model outputs in GLM's native XML tool call format)
        if not parsed_tool_calls and isinstance(content_data, str):
            # Parse <tool_call>{"name": "func", "arguments": {...}}</tool_call> format
            tool_call_pattern = (
                r"<((tool|function)[-_])call>\s*({[^}]*(?:{[^}]*}[^}]*)*})\s*</\1call>"
            )
            matches = re.findall(
                tool_call_pattern,
                content_data,
                re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )

            for match in matches:
                try:
                    tool_call_data = json.loads(match)
                    if isinstance(tool_call_data, dict) and "name" in tool_call_data:
                        # Extract arguments - handle both dict and JSON string formats
                        args = tool_call_data.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Failed to parse arguments string: {args}"
                                )
                                args = {}

                        # Convert to LangChain tool call format
                        lc_tool_call = {
                            "name": tool_call_data["name"],
                            "args": args,
                            "id": f"call_{tool_call_data['name']}_{len(parsed_tool_calls)}",
                        }
                        parsed_tool_calls.append(lc_tool_call)
                        logger.info(
                            f"🔧 Parsed tool call: {tool_call_data['name']} with args: {args}"
                        )
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool call JSON: {e}")

            # Remove tool call XML from content to clean it up
            content_data = re.sub(
                tool_call_pattern, "", content_data, flags=re.DOTALL
            ).strip()

        # Create AIMessage with parsed tool calls
        # Convert int ID to string for LangChain compatibility
        ai_message = AIMessage(content=content_data, id=lc_id)
        if parsed_tool_calls:
            ai_message.tool_calls = parsed_tool_calls
        return ai_message

    elif message.role == MessageRole.USER:
        return HumanMessage(content=content_data, id=lc_id)
    elif message.role == MessageRole.SYSTEM:
        return SystemMessage(content=content_data, id=lc_id)
    elif message.role == MessageRole.TOOL:
        # For tool messages, we need both content and tool_call_id
        tool_call_id = None
        if message.tool_calls and len(message.tool_calls) > 0:
            tool_call_id = message.tool_calls[0].execution_id
        return ToolMessage(
            content=content_data, tool_call_id=tool_call_id or "unknown", id=lc_id
        )
    elif message.role == MessageRole.OBSERVER:
        # Treat observer messages as system messages
        return SystemMessage(content=content_data, id=lc_id)
    else:
        # Default to human message for unknown roles
        return HumanMessage(content=content_data, id=lc_id)


def lc_message_to_message(
    base_message: AnyMessage | BaseMessage,
    conversation_id: Optional[int] = None,
) -> Message:
    """Convert a LangChain BaseMessage to a Message object."""

    # Determine role based on message type
    if isinstance(base_message, (AIMessage)):
        role = MessageRole.ASSISTANT
    elif isinstance(base_message, HumanMessage):
        role = MessageRole.USER
    elif isinstance(base_message, SystemMessage):
        role = MessageRole.SYSTEM
    elif isinstance(base_message, ToolMessage):
        role = MessageRole.TOOL  # Tool messages are typically assistant responses
    else:
        # Default to user role for unknown types
        role = MessageRole.USER

    # Handle content - preserve multimodal structure or convert to MessageContent
    content = convert_lc_message_content_to_message_format(base_message.content)
    tool_calls = extract_tool_calls_from_langchain_message(base_message)

    # Validate that content is not empty - ensure at least empty text content
    if not content:
        content = [
            MessageContent(
                type=MessageContentType.TEXT,
                text="",
                url=None,
            )
        ]

    # Create message with explicit field validation
    try:
        # Convert string ID from LangChain back to int for our format
        msg_id = None
        if hasattr(base_message, "id") and base_message.id is not None:
            try:
                msg_id = int(base_message.id)
            except (ValueError, TypeError):
                logger.warning(
                    f"Failed to convert LangChain message ID '{base_message.id}' to int"
                )
                msg_id = None

        msg = Message(
            id=msg_id,
            role=role,
            content=content,
            conversation_id=conversation_id,
            created_at=datetime.now(timezone.utc),
            tool_calls=tool_calls,
        )
    except Exception as e:
        logger.error(f"Failed to create Message object: {e}")
        logger.error(f"Role: {role}, Content: {content}")
        raise

    logger.debug(
        f"Converted LC message to Message: role={msg.role}, content_count={len(msg.content)}"
    )

    return msg


_CLIENT_INTERRUPT_MARKERS = re.compile(
    r"\[(?:Tool use|Request)\s+(?:was\s+)?interrupted(?:\s+by\s+user)?\]",
    re.IGNORECASE,
)
"""Markers that some clients (Claude Code in particular) inject into
assistant turns when the user interrupts a streaming tool call.  We
strip these before sending history to the model so it doesn't learn
to emit them verbatim.  Without this scrub, after a session with
several interruptions the model starts producing
``[Tool use interrupted]`` as its own ~22-char response — its EOS
output rather than actual content.  The cancellation is already
represented in history by the absence of a tool-result message;
the marker text adds noise the model can pattern-match against."""


def messages_to_lc_messages(
    messages: List[Message], use_llama_format: bool = False
) -> List[AnyMessage]:
    """Convert a list of Message objects to LangChain BaseMessages.

    Drops empty assistant messages (no content AND no tool_calls) that would
    teach the model to produce empty responses, and merges any consecutive
    same-role messages that result from the removal.

    Client-injected interrupt markers (``[Tool use interrupted]``,
    ``[Request interrupted by user]``) are scrubbed from assistant
    content before conversion — see ``_CLIENT_INTERRUPT_MARKERS``.

    Args:
        messages: List of Message objects to convert
        use_llama_format: If True, use llama.cpp compatible format
    """
    converted: List[AnyMessage] = []
    for msg in messages:
        lc_msg = message_to_lc_message(msg, use_llama_format)

        # Scrub client-side interrupt markers from assistant content.
        # These are Claude-Code-isms, not model output — leaving them in
        # history teaches the model to mimic them.
        if isinstance(lc_msg, AIMessage) and isinstance(lc_msg.content, str):
            cleaned = _CLIENT_INTERRUPT_MARKERS.sub("", lc_msg.content).strip()
            if cleaned != lc_msg.content:
                lc_msg.content = cleaned
        elif isinstance(lc_msg, AIMessage) and isinstance(lc_msg.content, list):
            new_parts: List[Any] = []
            for part in lc_msg.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    cleaned = _CLIENT_INTERRUPT_MARKERS.sub(
                        "", part.get("text", "")
                    ).strip()
                    if cleaned:
                        new_parts.append({**part, "text": cleaned})
                else:
                    new_parts.append(part)
            lc_msg.content = new_parts

        # Drop empty AI messages (no content, no tool_calls) — these poison
        # the model into producing EOS immediately.
        if isinstance(lc_msg, AIMessage):
            content_empty = not lc_msg.content or (
                isinstance(lc_msg.content, str) and not lc_msg.content.strip()
            )
            has_tc = bool(getattr(lc_msg, "tool_calls", None))
            if content_empty and not has_tc:
                logger.debug(
                    "Dropping empty AIMessage from conversation history "
                    "(no content, no tool_calls)"
                )
                continue

        # Merge consecutive same-type messages (can happen after dropping
        # empty AI messages: user → [empty AI dropped] → user).
        if (
            converted
            and type(lc_msg) is type(converted[-1])
            and isinstance(lc_msg, HumanMessage)
        ):
            prev = converted[-1]
            prev_text = (
                prev.content if isinstance(prev.content, str) else str(prev.content)
            )
            cur_text = (
                lc_msg.content
                if isinstance(lc_msg.content, str)
                else str(lc_msg.content)
            )
            converted[-1] = HumanMessage(content=f"{prev_text}\n\n{cur_text}")
            logger.debug("Merged consecutive HumanMessages after empty AI removal")
            continue

        converted.append(lc_msg)

    # Post-pass: scrub base64 image blobs out of older tool messages.
    # See :func:`_redact_old_tool_images` for the rationale.
    return _redact_old_tool_images(converted)


def lc_messages_to_messages(
    base_messages: List[AnyMessage], conversation_id: Optional[int] = None
) -> List[Message]:
    """Convert a list of LangChain BaseMessages to Message objects."""
    return [lc_message_to_message(msg, conversation_id) for msg in base_messages]


def convert_message_content_to_langchain_format(
    content: List[MessageContent],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """
    Convert Message.content list to LangChain multimodal format.

    Returns:
        - str: For simple text-only messages
        - List[Union[str, Dict[str, Any]]]: For multimodal messages with text and/or images
    """
    if not content:
        return ""

    # If single text content, return as string for simplicity
    if len(content) == 1 and content[0].type == MessageContentType.TEXT:
        return content[0].text or ""

    # Multimodal content - return as list of dictionaries
    result = []
    for content_item in content:
        if content_item.type == MessageContentType.TEXT:
            result.append({"type": "text", "text": content_item.text or ""})
        elif content_item.type == MessageContentType.IMAGE:
            # Handle images using LangChain standard format - base64 required for OpenAI compatibility
            if content_item.url and is_data_uri(content_item.url):
                base64_data = extract_base64_from_data_uri(content_item.url)
                mime_type = extract_mime_type_from_data_uri(content_item.url)

                if base64_data and mime_type:
                    # Use LangChain standard base64 format (required for OpenAI models)
                    result.append(
                        {
                            "type": "image",
                            "base64": base64_data,
                            "mime_type": mime_type,
                            **(_get_file_extras(content_item)),
                        }
                    )
                else:
                    # Skip image content without valid data URI
                    logger.warning(
                        f"Image content has invalid data URI (required for OpenAI): {content_item}"
                    )
                    continue
            else:
                # Skip image content without data URI (OpenAI requires base64)
                logger.warning(
                    f"Image content missing data URI (required for OpenAI): {content_item}"
                )
                continue

        elif content_item.type == MessageContentType.AUDIO:
            # Handle audio files using LangChain standard format - base64 required for OpenAI compatibility
            if content_item.url and is_data_uri(content_item.url):
                base64_data = extract_base64_from_data_uri(content_item.url)
                mime_type = extract_mime_type_from_data_uri(content_item.url)

                if base64_data and mime_type:
                    # Use LangChain standard base64 format for audio (required for OpenAI models)
                    result.append(
                        {
                            "type": "audio",
                            "base64": base64_data,
                            "mime_type": mime_type,
                            **(_get_file_extras(content_item)),
                        }
                    )
                else:
                    # Skip audio content without valid data URI
                    logger.warning(
                        f"Audio content has invalid data URI (required for OpenAI): {content_item}"
                    )
                    continue
            else:
                # Skip audio content without data URI (OpenAI requires base64)
                logger.warning(
                    f"Audio content missing data URI (required for OpenAI): {content_item}"
                )
                continue

        elif content_item.type == MessageContentType.VIDEO:
            # Handle video files using LangChain standard format - base64 required for OpenAI compatibility
            if content_item.url and is_data_uri(content_item.url):
                base64_data = extract_base64_from_data_uri(content_item.url)
                mime_type = extract_mime_type_from_data_uri(content_item.url)

                if base64_data and mime_type:
                    # Use LangChain standard base64 format for video (required for OpenAI models)
                    result.append(
                        {
                            "type": "video",
                            "base64": base64_data,
                            "mime_type": mime_type,
                            **(_get_file_extras(content_item)),
                        }
                    )
                else:
                    # Skip video content without valid data URI
                    logger.warning(
                        f"Video content has invalid data URI (required for OpenAI): {content_item}"
                    )
                    continue
            else:
                # Skip video content without data URI (OpenAI requires base64)
                logger.warning(
                    f"Video content missing data URI (required for OpenAI): {content_item}"
                )
                continue

        elif content_item.type == MessageContentType.FILE:
            # Handle generic file attachments using LangChain standard format - base64 required for OpenAI compatibility
            base64_data = None
            mime_type = None

            # Try to get base64 data from URL field (data URI) or fallback to data field
            if content_item.url and is_data_uri(content_item.url):
                base64_data = extract_base64_from_data_uri(content_item.url)
                mime_type = extract_mime_type_from_data_uri(content_item.url)
            elif content_item.data and content_item.format:
                # Fallback to legacy data field (for backward compatibility)
                base64_data = content_item.data
                mime_type = content_item.format

            if base64_data and mime_type:
                # Use LangChain standard base64 format for files (required for OpenAI models)
                result.append(
                    {
                        "type": "file",
                        "base64": base64_data,
                        "mime_type": mime_type,
                        **(_get_file_extras(content_item)),
                    }
                )
            else:
                # Fallback to text description if no base64 data (OpenAI requires base64 for files)
                file_name = content_item.name or "attachment"
                file_info = f"[File: {file_name}"
                if content_item.format:
                    file_info += f" ({content_item.format})"
                file_info += "]"
                logger.warning(
                    f"File content missing base64 data, converting to text description: {file_name}"
                )
                result.append({"type": "text", "text": file_info})
        # Add other content types as needed

    return result


def convert_message_content_to_llama_format(
    content: List[MessageContent],
) -> Union[str, List[Union[str, Dict[str, Any]]]]:
    """
    Convert Message.content list to llama.cpp compatible format.
    - Images: Use data URI format with base64 encoded image data
    - Files: Extract text content or create text descriptions
    - Audio/Video: Convert to text descriptions (llama.cpp doesn't support these)

    Returns:
        - str: For simple text-only messages
        - List[Union[str, Dict[str, Any]]]: For multimodal messages with text and/or image data URIs
    """
    if not content:
        return ""

    # If single text content, return as string for simplicity
    if len(content) == 1 and content[0].type == MessageContentType.TEXT:
        return content[0].text or ""

    # Multimodal content - return as list of dictionaries
    result = []
    for content_item in content:
        if content_item.type == MessageContentType.TEXT:
            result.append({"type": "text", "text": content_item.text or ""})

        elif content_item.type == MessageContentType.IMAGE:
            # Handle images: use data URI with base64 for llama.cpp
            data_uri = None

            # Try to get data URI from URL field or create from data field
            if content_item.url and is_data_uri(content_item.url):
                # Already a data URI
                data_uri = content_item.url
            elif (
                content_item.data
                and content_item.format
                and is_image_format(content_item.format)
            ):
                # Create data URI from legacy data field
                data_uri = create_data_uri(content_item.format, content_item.data)

            if data_uri:
                try:
                    # Use image_url format with data URI for llama.cpp
                    result.append({"type": "image_url", "image_url": {"url": data_uri}})
                    logger.info(
                        f"Using data URI for llama.cpp image: {content_item.name or 'image'}"
                    )

                except Exception as e:
                    logger.error(f"Failed to process image: {e}")
                    # Fallback to text description
                    result.append(
                        {
                            "type": "text",
                            "text": f"[Image: {content_item.name or 'attachment'} - processing failed]",
                        }
                    )
            else:
                # Skip images without proper data or unsupported formats
                logger.warning(
                    f"Image content missing data or unsupported format: {content_item}"
                )
                result.append(
                    {
                        "type": "text",
                        "text": f"[Image: {content_item.name or 'attachment'} - unsupported format]",
                    }
                )

        elif content_item.type in (
            MessageContentType.AUDIO,
            MessageContentType.VIDEO,
            MessageContentType.FILE,
        ):
            # Handle non-image files: extract text or create descriptions
            base64_data = None
            mime_type = None

            # Try to get base64 data from URL field (data URI) or fallback to data field
            if content_item.url and is_data_uri(content_item.url):
                base64_data = extract_base64_from_data_uri(content_item.url)
                mime_type = extract_mime_type_from_data_uri(content_item.url)
            elif content_item.data and content_item.format:
                # Fallback to legacy data field
                base64_data = content_item.data
                mime_type = content_item.format

            if base64_data and mime_type:
                try:
                    # Extract text content from file
                    text_content = extract_text_from_file(
                        base64_data, mime_type, content_item.name
                    )

                    result.append({"type": "text", "text": text_content})
                    logger.info(f"Converted file to text: {content_item.name}")

                except Exception as e:
                    logger.error(f"Failed to extract text from file: {e}")
                    # Fallback to basic description
                    file_name = content_item.name or "attachment"
                    result.append(
                        {
                            "type": "text",
                            "text": f"[File: {file_name} - unable to process]",
                        }
                    )
            else:
                # No data available - create basic description
                file_name = content_item.name or "attachment"
                file_info = f"[File: {file_name}"
                if content_item.format:
                    file_info += f" ({content_item.format})"
                file_info += "]"
                result.append({"type": "text", "text": file_info})
        # Add other content types as needed

    return result


def convert_lc_message_content_to_message_format(
    lc_content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> List[MessageContent]:
    """
    Convert LangChain BaseMessage content to Message.content format.

    Args:
        lc_content: Content from LangChain BaseMessage (str or list)

    Returns:
        - List[MessageContent]: List of MessageContent objects
    """

    content = []
    if isinstance(lc_content, list):
        # Multimodal content - convert each item to MessageContent
        for item in lc_content:
            if isinstance(item, dict):
                if is_langchain_tool_call(item.get("content", {})):
                    try:
                        content.append(
                            MessageContent(
                                type=MessageContentType.TOOL_CALL,
                                text=json.dumps(item.get("content", {})),
                                url=None,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create TOOL_CALL MessageContent: {e}"
                        )
                if item.get("type") == "text":
                    try:
                        content.append(
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=item.get("text", ""),
                                url=None,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create TEXT MessageContent: {e}")
                elif item.get("type") == "image":
                    # Handle LangChain standard image format with base64
                    try:
                        base64_data = item.get("base64")
                        mime_type = item.get("mime_type")
                        url = item.get("url")
                        filename = item.get("extras", {}).get("filename")

                        # Create data URI if we have base64 data, otherwise use provided URL
                        if base64_data and mime_type:
                            data_uri = create_data_uri(mime_type, base64_data)
                        else:
                            data_uri = url

                        content.append(
                            MessageContent(
                                type=MessageContentType.IMAGE,
                                text=None,
                                url=data_uri,
                                format=mime_type,
                                name=filename,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create IMAGE MessageContent from LangChain format: {e}"
                        )
                elif item.get("type") == "image_url":
                    # Handle legacy OpenAI image_url format
                    try:
                        url = item.get("image_url", {}).get("url", "")
                        content.append(
                            MessageContent(
                                type=MessageContentType.IMAGE,
                                text=None,
                                url=url,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create IMAGE MessageContent from image_url format: {e}"
                        )
                elif item.get("type") == "audio":
                    # Handle LangChain standard audio format with base64
                    try:
                        base64_data = item.get("base64")
                        mime_type = item.get("mime_type")
                        url = item.get("url")
                        filename = item.get("extras", {}).get("filename")

                        # Create data URI if we have base64 data, otherwise use provided URL
                        if base64_data and mime_type:
                            data_uri = create_data_uri(mime_type, base64_data)
                        else:
                            data_uri = url

                        content.append(
                            MessageContent(
                                type=MessageContentType.AUDIO,
                                text=None,
                                url=data_uri,
                                format=mime_type,
                                name=filename,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create AUDIO MessageContent: {e}")
                elif item.get("type") == "video":
                    # Handle LangChain standard video format with base64
                    try:
                        base64_data = item.get("base64")
                        mime_type = item.get("mime_type")
                        url = item.get("url")
                        filename = item.get("extras", {}).get("filename")

                        # Create data URI if we have base64 data, otherwise use provided URL
                        if base64_data and mime_type:
                            data_uri = create_data_uri(mime_type, base64_data)
                        else:
                            data_uri = url

                        content.append(
                            MessageContent(
                                type=MessageContentType.VIDEO,
                                text=None,
                                url=data_uri,
                                format=mime_type,
                                name=filename,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create VIDEO MessageContent: {e}")
                elif item.get("type") == "file":
                    # Handle LangChain standard file format with base64
                    try:
                        base64_data = item.get("base64")
                        mime_type = item.get("mime_type")
                        url = item.get("url")
                        filename = item.get("extras", {}).get("filename")

                        # Create data URI if we have base64 data, otherwise use provided URL
                        if base64_data and mime_type:
                            data_uri = create_data_uri(mime_type, base64_data)
                        else:
                            data_uri = url

                        content.append(
                            MessageContent(
                                type=MessageContentType.FILE,
                                text=None,
                                url=data_uri,
                                format=mime_type,
                                name=filename,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create FILE MessageContent: {e}")
                else:
                    # Unknown content type, treat as text
                    try:
                        content.append(
                            MessageContent(
                                type=MessageContentType.TEXT, text=str(item), url=None
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create fallback TEXT MessageContent: {e}"
                        )
            else:
                # String content item
                try:
                    content.append(
                        MessageContent(
                            type=MessageContentType.TEXT, text=str(item), url=None
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to create string MessageContent: {e}")
    else:
        # Simple string content
        try:
            content = [
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=str(lc_content) if lc_content else "",
                    url=None,
                )
            ]
        except Exception as e:
            logger.error(f"Failed to create simple text MessageContent: {e}")
            # Fallback to empty list if all else fails
            content = []

    # Ensure we always return at least one content item
    if not content:
        logger.warning("No content items created, adding empty text content")
        content = [
            MessageContent(
                type=MessageContentType.TEXT,
                text="",
                url=None,
            )
        ]

    return content


def extract_text_from_message(message: Union[Message, BaseMessage]) -> str:
    """
    Extract text content from either a Message object or LangChain BaseMessage.

    This is the unified function that handles both message types.
    """
    if isinstance(message, BaseMessage):
        # Handle LangChain BaseMessage
        if not hasattr(message, "content"):
            return ""

        content = message.content

        # Handle multimodal content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(
                text_parts
            )  # Fixed: removed \n join that was causing newline issues

        # Handle simple string content
        return str(content) if content else ""

    else:
        # Handle Message object
        text_parts = []
        for content in message.content:
            # Handle both MessageContent objects and dictionaries
            if isinstance(content, dict):
                # Handle dictionary format: {'type': 'text', 'text': 'content'}
                if content.get("type") == "text" and content.get("text"):
                    text_parts.append(content["text"])
            else:
                # Handle MessageContent object format
                if hasattr(content, "type") and hasattr(content, "text"):
                    if content.type == MessageContentType.TEXT and content.text:
                        text_parts.append(content.text)
        # Fixed: use space join instead of newline to prevent character separation
        return "".join(text_parts)


def get_most_recent_user_message_text(messages: List[BaseMessage]) -> str:
    """Extract text from the most recent user message in a conversation."""
    if not messages:
        return ""

    # Look for the most recent user message
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return extract_text_from_message(msg)

    # Fallback: if no user message found, use the last message
    if messages:
        return extract_text_from_message(messages[-1])

    return ""


def create_text_message_content(text: str) -> List[MessageContent]:
    """
    Create a list containing a single MessageContent object with text content.

    This is the unified function for creating text message content.
    Replaces: convert_string_to_message_content
    """
    return [MessageContent(type=MessageContentType.TEXT, text=text, url=None)]


def normalize_message_input(
    input_data: Union[
        str, Message, List[Union[str, Message]], List[str], List[Message]
    ],
    role: MessageRole = MessageRole.USER,
) -> List[Message]:
    """
    Normalize various message input formats to a list of Message objects.

    Args:
        input_data: String, Message, or list of strings/messages
        role: Default role for string inputs

    Returns:
        List of normalized Message objects
    """
    if isinstance(input_data, str):
        return [
            Message(
                role=role,
                content=create_text_message_content(input_data),
                created_at=datetime.now(timezone.utc),
            )
        ]
    elif isinstance(input_data, Message):
        return [input_data]
    elif isinstance(input_data, list):
        normalized = []
        for item in input_data:
            if isinstance(item, str):
                normalized.append(
                    Message(
                        role=role,
                        content=create_text_message_content(item),
                        created_at=datetime.now(timezone.utc),
                    )
                )
            elif isinstance(item, Message):
                normalized.append(item)
            else:
                # Handle other types by converting to string
                normalized.append(
                    Message(
                        role=role,
                        content=create_text_message_content(str(item)),
                        created_at=datetime.now(timezone.utc),
                    )
                )
        return normalized
    else:
        # Fallback: convert to string and create message
        return [
            Message(
                role=role,
                content=create_text_message_content(str(input_data)),
                created_at=datetime.now(timezone.utc),
            )
        ]


# --- File and document transformation ---


async def transform_file_content_to_documents(message: Message, user_id: str) -> Message:
    """
    Transform message content of type 'file' into Document objects in message.documents.

    This function processes incoming messages from the UI and converts any content items
    with type="file" into proper Document objects that can be stored in the database.
    Uses existing utility functions to avoid code duplication.

    Args:
        message: The message to transform
        user_id: ID of the user sending the message

    Returns:
        Transformed message with file content moved to documents array
    """
    if not message.content:
        return message

    # Initialize documents list if not present
    if message.documents is None:
        message.documents = []

    # Process content items and extract files
    new_content = []

    for content_item in message.content:
        if content_item.type == MessageContentType.FILE:
            # Extract file information using existing utilities
            document = await create_document_from_content(
                content_item=content_item,
                user_id=user_id
            )

            if document:
                # Add to documents array
                message.documents.append(document)

                # Replace the file content with a text reference
                text_ref = MessageContent(
                    type=MessageContentType.TEXT,
                    text=f"[File: {document.filename}]"
                )
                new_content.append(text_ref)
            else:
                # Keep original content if document creation failed
                new_content.append(content_item)

        else:
            # Keep non-file content as-is
            new_content.append(content_item)

    # Update message content
    message.content = new_content

    return message


async def create_document_from_content(
    content_item: MessageContent, user_id: str
) -> Optional[Document]:
    """
    Create a Document object from a MessageContent item using existing file utilities.

    Args:
        content_item: MessageContent with type FILE
        user_id: ID of the user uploading the file

    Returns:
        Document object or None if creation fails
    """
    try:
        # Get filename from name field or generate default
        filename = content_item.name or "attachment"

        # Extract content and MIME type using existing utilities
        content_type = None
        base64_content = None

        # Try to extract from data URI first (preferred method)
        if content_item.url and is_data_uri(content_item.url):
            content_type = extract_mime_type_from_data_uri(content_item.url)
            base64_content = extract_base64_from_data_uri(content_item.url)

        # Fallback to format field for content type
        if not content_type:
            content_type = content_item.format or "application/octet-stream"

        # Fallback to text field for content
        if not base64_content:
            base64_content = content_item.text or ""

        # Calculate file size using utility
        try:
            decoded_data = get_decoded_data(content_item.url) if content_item.url and is_data_uri(content_item.url) else None
            if decoded_data:
                file_size = len(decoded_data)
            else:
                # Fallback calculation
                import base64
                file_size = len(base64.b64decode(base64_content))
        except Exception:
            # If base64 decode fails, use string length as approximation
            file_size = len(base64_content)

        # Extract text content for searchability using existing utility
        text_content = extract_text_content(base64_content, content_type, filename)

        # Create Document object
        document = Document(
            message_id=0,  # Temporary, will be set by message storage
            user_id=user_id,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            content=base64_content,
            text_content=text_content,
            created_at=datetime.now(timezone.utc),
        )

        logger.info(f"Created document from content: {filename} ({content_type}, {file_size} bytes)")
        return document

    except Exception as e:
        logger.warning(f"Failed to create document from content: {e}")
        return None
