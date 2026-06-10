"""Real token counts via llama.cpp's ``/tokenize`` endpoint.

The api previously relied on ``utils/token_estimation.estimate_tokens`` —
a ``len(content) // 3`` heuristic that over-counts dense JSON / tool-call
payloads by 2× or more.  Under heavy claude-cli traffic the estimator
reports 200 K tokens for a conversation the real tokenizer sees at ~93 K,
triggering a false context-overflow.

llama.cpp exposes ``POST /tokenize`` on every running server.  Through the
runner proxy that endpoint is reachable at
``{server_handle.base_url}/tokenize`` (the proxy rewrites
``/v1/server/{id}/tokenize`` → ``http://127.0.0.1:<port>/tokenize``).

This module wraps that call so the agent can plug it in wherever it used
to call ``estimate_tokens`` / ``estimate_message_tokens``.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any, Iterable, Optional, Tuple

from config import (
    IMAGE_TOKENS_DEFAULT as _DEFAULT_IMAGE_TOKENS,
    VISION_MAX_LONG_EDGE_PX as _VISION_MAX_LONG_EDGE_PX,
    VISION_PATCH_PX as _VISION_PATCH_PX,
)
from services.runner_client import runner_client as _default_client
from utils.logging import llmmllogger
from utils.message_conversion import extract_text_from_message

logger = llmmllogger.bind(component="token_counter")


def _estimate_image_tokens(width: int, height: int) -> int:
    """Qwen-VL-style patch count given image dimensions.

    Resizes the long edge down to ``_VISION_MAX_LONG_EDGE_PX`` if needed,
    then returns ``ceil(W / patch) * ceil(H / patch)``.  Same formula
    Qwen2/3-VL applies on the model side.
    """
    if width <= 0 or height <= 0:
        return _DEFAULT_IMAGE_TOKENS
    long_edge = max(width, height)
    if long_edge > _VISION_MAX_LONG_EDGE_PX:
        scale = _VISION_MAX_LONG_EDGE_PX / long_edge
        width = int(width * scale)
        height = int(height * scale)
    return max(
        1,
        math.ceil(width / _VISION_PATCH_PX) * math.ceil(height / _VISION_PATCH_PX),
    )


def _image_dims_from_b64(data: str) -> Optional[Tuple[int, int]]:
    """Decode just enough of a base64 image to get (width, height).

    Uses PIL if available; otherwise falls back to None and the caller
    uses the default per-image token cost.
    """
    if not data:
        return None
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None

    # Strip ``data:image/...;base64,`` prefix if present.
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception:  # noqa: BLE001
        return None
    try:
        with Image.open(io.BytesIO(raw)) as img:
            return img.size  # (width, height)
    except Exception:  # noqa: BLE001
        return None


def _iter_content_blocks(content: Any) -> Iterable[Any]:
    """Yield content blocks from any of the message shapes we accept.

    Anthropic, OpenAI, LangChain, and our own Message format all wrap
    content as either a string, a list of dicts/objects, or a single
    block.  Normalising here so the caller can just type-switch.
    """
    if content is None:
        return
    if isinstance(content, str):
        return
    if isinstance(content, list):
        for item in content:
            yield item
        return
    # Single block (dict or MessageContent object).
    yield content


def _looks_like_image_block(block: Any) -> bool:
    """True if *block* is one of the multimodal image shapes we count."""
    if isinstance(block, dict):
        t = block.get("type")
        return t in ("image", "image_url", "input_image")
    btype = getattr(block, "type", None)
    if btype is None:
        return False
    if hasattr(btype, "value"):  # enum
        btype = btype.value
    return str(btype).lower() in ("image", "image_url", "input_image")


def _block_image_tokens(block: Any) -> int:
    """Estimate vision tokens for one image-bearing content block.

    Tries to decode dimensions from common payload shapes:
      * Anthropic: ``{"type": "image", "source": {"type": "base64",
        "media_type": ..., "data": "<b64>"}}``
      * OpenAI:    ``{"type": "image_url", "image_url": {"url":
        "data:...;base64,..." | "https://..."}}``
      * Our own MessageContent object: ``.source.data`` / ``.image_url``

    Returns ``_DEFAULT_IMAGE_TOKENS`` when dimensions can't be cheaply
    extracted (HTTP URLs, or PIL unavailable).
    """
    # Pull base64 data out of the various shapes.
    b64: Optional[str] = None

    if isinstance(block, dict):
        # Anthropic shape: block.source.data
        src = block.get("source")
        if isinstance(src, dict) and src.get("type") == "base64":
            b64 = src.get("data")
        # OpenAI shape: block.image_url.url == "data:...;base64,..."
        if b64 is None:
            url_obj = block.get("image_url")
            url = (
                url_obj.get("url")
                if isinstance(url_obj, dict)
                else (url_obj if isinstance(url_obj, str) else None)
            )
            if isinstance(url, str) and url.startswith("data:"):
                b64 = url
    else:
        # Object-shaped MessageContent.
        src = getattr(block, "source", None)
        if src is not None:
            data = getattr(src, "data", None)
            if isinstance(data, str):
                b64 = data
        if b64 is None:
            url_obj = getattr(block, "image_url", None)
            url = url_obj if isinstance(url_obj, str) else getattr(url_obj, "url", None)
            if isinstance(url, str) and url.startswith("data:"):
                b64 = url

    if b64:
        dims = _image_dims_from_b64(b64)
        if dims is not None:
            return _estimate_image_tokens(*dims)
    return _DEFAULT_IMAGE_TOKENS


def _coerce_to_text(content: Any) -> str:
    """Best-effort serialise an arbitrary Message.content to a single string.

    llama.cpp's ``/tokenize`` takes a single ``content`` string.  Structured
    content blocks (list of ``{type, text}`` dicts) get flattened to their
    text; everything else falls back to ``str(...)``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
    return str(content)


async def count_tokens(
    text: str,
    *,
    base_url: str,
    client: Optional[Any] = None,
    timeout: float = 5.0,
) -> Optional[int]:
    """Return the exact token count for *text* using llama.cpp's tokenizer.

    Returns ``None`` if the tokenizer call fails (network error, non-200,
    malformed JSON).  Callers must handle the fallback themselves; we
    explicitly do *not* return a fabricated estimate, because the whole
    point of this helper is to escape the misleading char-based heuristic.
    """
    if not text:
        return 0
    if not base_url:
        return None

    cli = client or _default_client
    try:
        http_client = cli._get_client()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"runner client unavailable for tokenize: {e}")
        return None

    url = f"{base_url.rstrip('/')}/tokenize"
    try:
        response = await http_client.post(
            url,
            json={"content": text},
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"tokenize POST failed: {e}", extra={"url": url})
        return None

    if response.status_code != 200:
        logger.debug(
            "tokenize returned non-200",
            extra={"status": response.status_code, "url": url},
        )
        return None

    try:
        body = response.json()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"tokenize JSON decode failed: {e}")
        return None

    tokens = body.get("tokens")
    if not isinstance(tokens, list):
        return None
    return len(tokens)


async def count_message_tokens(
    message: Any,
    *,
    base_url: str,
    client: Optional[Any] = None,
) -> Optional[int]:
    """Tokenize a :class:`Message` object via llama.cpp.

    Counts both:
      * Text content via llama.cpp's ``/tokenize`` endpoint (real
        per-token count using the model's own tokenizer).
      * Image content via Qwen-VL's patch-count formula (see
        :func:`_estimate_image_tokens`), since llama.cpp's text
        tokenizer can't see vision tokens — they're produced by
        the mmproj projector at inference time.  Without this,
        a session full of screenshots looks "small" to the api's
        pre-trim while llama-server actually receives 1500+
        tokens per image, blowing past ``n_ctx`` mid-stream.

    Returns ``None`` only if the *text* tokenize call itself fails
    (network error, non-200) — that mirrors the original contract
    so callers can choose their fallback.  Image-token estimation
    is best-effort; on missing dimensions / PIL we substitute a
    conservative per-image default.
    """
    # 1. Image-block tokens — walked directly off the message's
    # content so the multimodal blocks aren't lost when we flatten
    # to text below.
    image_tokens = 0
    content = getattr(message, "content", message)
    try:
        for block in _iter_content_blocks(content):
            if _looks_like_image_block(block):
                image_tokens += _block_image_tokens(block)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"image token walk failed: {e}")

    # 2. Text-block tokens via the real tokenizer.
    try:
        text = extract_text_from_message(message)
    except Exception:
        text = _coerce_to_text(content)
    text_tokens = await count_tokens(text, base_url=base_url, client=client)
    if text_tokens is None:
        return None

    return text_tokens + image_tokens


def _estimate_tokens(text: str) -> int:
    """Last-resort char-based estimate, used ONLY when the real tokenizer is
    unreachable. ~4 chars/token tracks mixed English/JSON far better than the
    legacy ``len // 3`` (which over-counted dense payloads ~2x and tripped
    false context-overflows — the very bug this module exists to kill)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


async def _count_text(
    text: str, base_url: Optional[str], client: Optional[Any] = None
) -> int:
    """Real ``/tokenize`` count for *text*, falling back to the char estimate
    only when no server is reachable. Always returns a usable number."""
    if text and base_url:
        n = await count_tokens(text, base_url=base_url, client=client)
        if n is not None:
            return n
    return _estimate_tokens(text)


def _tools_to_openai(tools: Optional[list]) -> Optional[list]:
    """Normalize tool schemas to OpenAI *function* format for /apply-template.

    Accepts already-OpenAI tools (``{"type":"function","function":{...}}``),
    Anthropic tools (``{"name","description","input_schema"}``), or pydantic
    models exposing ``model_dump()``. Returns None if nothing usable, so the
    template is rendered without tools rather than with a malformed list.
    """
    if not tools:
        return None
    out: list = []
    for tool in tools:
        if isinstance(tool, dict):
            d: Any = tool
        elif hasattr(tool, "model_dump"):
            d = tool.model_dump(exclude_none=True)
        else:
            continue
        if not isinstance(d, dict):
            continue
        if d.get("type") == "function" and isinstance(d.get("function"), dict):
            out.append(d)  # already OpenAI shape
        elif d.get("name"):
            out.append({
                "type": "function",
                "function": {
                    "name": d.get("name"),
                    "description": d.get("description", "") or "",
                    "parameters": d.get("input_schema")
                    or d.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            })
    return out or None


def _sum_image_tokens(messages: Iterable[Any]) -> int:
    """Qwen-VL image-patch tokens across all messages. The chat-templated text
    (/apply-template's output) never contains image pixels, so these are added
    on top to match what the mmproj projector actually feeds the model."""
    image_tokens = 0
    for message in messages or []:
        content = getattr(message, "content", message)
        try:
            for block in _iter_content_blocks(content):
                if _looks_like_image_block(block):
                    image_tokens += _block_image_tokens(block)
        except Exception:  # noqa: BLE001
            continue
    return image_tokens


async def _count_templated_input_tokens(
    messages: list,
    tools: Optional[list],
    *,
    base_url: str,
    system_prompt: Optional[str],
    client: Optional[Any],
) -> Optional[int]:
    """Tokenize the FULL chat-templated prompt — system + messages + tools,
    rendered by the model's own chat template via llama.cpp's ``/apply-template``
    — then add image-patch tokens.

    This matches the model's real ``prompt_eval_count`` to ~0.5% (measured),
    whereas a per-message text sum undercounts ~30%: it omits the chat-template
    role headers and tool-call markup the model actually processes. message_start
    reports this number as the conversation's context size, so it must reflect
    the templated prompt — otherwise the client thinks it has far more headroom
    than the model does. Reuses the SAME converter the generation path uses
    (``messages_to_lc_messages`` → ``convert_to_openai_messages``) so the
    rendered prompt equals what ChatOpenAI sends to the model.

    Returns None (caller falls back) if conversion, ``/apply-template``, or
    ``/tokenize`` is unavailable.
    """
    try:
        from langchain_core.messages import convert_to_openai_messages

        from utils.message_conversion import messages_to_lc_messages

        oai_messages = list(convert_to_openai_messages(messages_to_lc_messages(messages)))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"templated count: message conversion failed: {e}")
        return None

    if system_prompt:
        oai_messages = [{"role": "system", "content": system_prompt}] + oai_messages

    payload: dict[str, Any] = {"messages": oai_messages}
    oai_tools = _tools_to_openai(tools)
    if oai_tools:
        payload["tools"] = oai_tools

    cli = client or _default_client
    try:
        http_client = cli._get_client()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"templated count: runner client unavailable: {e}")
        return None

    url = f"{base_url.rstrip('/')}/apply-template"
    try:
        resp = await http_client.post(url, json=payload, timeout=15.0)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"templated count: /apply-template POST failed: {e}")
        return None
    if resp.status_code != 200:
        logger.debug(f"templated count: /apply-template returned {resp.status_code}")
        return None
    try:
        prompt = resp.json().get("prompt") or resp.json().get("formatted")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(prompt, str) or not prompt:
        return None

    text_tokens = await count_tokens(prompt, base_url=base_url, client=client)
    if text_tokens is None:
        return None
    return text_tokens + _sum_image_tokens(messages)


async def count_input_tokens(
    messages: Iterable[Any],
    tools: Optional[list] = None,
    *,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    client: Optional[Any] = None,
) -> int:
    """Prompt-token count for an Anthropic-style request (system + messages + tools).

    Preferred path: tokenize the full chat-templated prompt via
    :func:`_count_templated_input_tokens` (~0.5% off the model's real
    prompt_eval_count). message_start reports this as the conversation's context
    size, so it must reflect the templated prompt.

    Fallback (no server reachable, or ``/apply-template`` unavailable): sum
    per-message counts via :func:`count_message_tokens` (real ``/tokenize`` +
    image patches) plus system and serialized tools, with a char/4 estimate only
    when the tokenizer itself is down — so message_start always carries a number.
    This path undercounts by the chat-template overhead, but it's a last resort.

    Replaces the old ``TokenService.count_input_tokens``, whose ``_combine_text``
    path dropped image blocks and silently fell back to a coarse ``len // 3``.
    """
    materialized = list(messages) if messages is not None else []

    if base_url:
        templated = await _count_templated_input_tokens(
            materialized, tools,
            base_url=base_url, system_prompt=system_prompt, client=client,
        )
        if templated is not None:
            return max(1, templated)

    import json as _json

    total = 0
    if system_prompt:
        total += await _count_text(system_prompt, base_url, client)

    for message in materialized:
        n: Optional[int] = None
        if base_url:
            n = await count_message_tokens(message, base_url=base_url, client=client)
        if n is None:
            content = getattr(message, "content", message)
            try:
                text = extract_text_from_message(message)
            except Exception:  # noqa: BLE001
                text = _coerce_to_text(content)
            n = _estimate_tokens(text)
        total += n

    if tools:
        for tool in tools:
            if isinstance(tool, dict):
                td: Any = tool
            elif hasattr(tool, "model_dump"):
                td = tool.model_dump(exclude_none=True)
            else:
                td = tool
            try:
                serialized = _json.dumps(td, default=str)
            except (TypeError, ValueError):
                serialized = str(td)
            total += await _count_text(serialized, base_url, client)

    return max(1, total)
