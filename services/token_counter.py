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

from typing import Any, Optional

from services.runner_client import runner_client as _default_client
from utils.logging import llmmllogger
from utils.message_conversion import extract_text_from_message

logger = llmmllogger.bind(component="token_counter")


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

    Flattens the message's content with ``extract_text_from_message`` so
    tool-call blocks, structured payloads, and plain strings are all
    counted on the same footing.  Returns ``None`` on tokenizer failure
    (caller decides the fallback policy).
    """
    try:
        text = extract_text_from_message(message)
    except Exception:
        text = _coerce_to_text(getattr(message, "content", message))
    return await count_tokens(text, base_url=base_url, client=client)
