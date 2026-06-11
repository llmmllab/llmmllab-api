"""Reactive context-overflow recovery: summarize older history, then retry.

When llama.cpp rejects a prompt with ``exceed_context_size_error`` (the prompt is
larger than the model's KV window — usually because the api's pre-send token
count was slightly UNDER the model's real tokenizer count), the request would
otherwise fail with a 400 and surface to the client as an error.

Instead, we consolidate the OLDER portion of the conversation into a summary —
via the existing :class:`SummarizationMiddleware`, on the SAME model the session
uses — and retry with ``[summary, *recent_messages]``. This is a safety net that
complements (does not replace) accurate token counting; configurable via env
(see ``config.OVERFLOW_SUMMARY_*``).

Kept deliberately self-contained: the only coupling to the agent architecture is
building one ``ChatAgent`` on the session's model for the middleware. When that
architecture is simplified/removed, this module is the single call site to update.
"""

from __future__ import annotations

from typing import Optional

from models import Message, MessageContent, MessageContentType, MessageRole
from utils.logging import llmmllogger
from utils.message_conversion import (
    lc_messages_to_messages,
    messages_to_lc_messages,
)

logger = llmmllogger.bind(component="overflow_recovery")

# Substrings llama.cpp / the runner use for "prompt larger than the KV window".
_OVERFLOW_MARKERS = ("exceed_context_size", "exceeds the available context size")

_SUMMARY_SYSTEM_PROMPT = (
    "You summarize prior conversation faithfully and concisely. Preserve "
    "decisions made, concrete facts, file paths, identifiers, numbers, and any "
    "open threads or unfinished tasks. Do not invent details."
)


def is_overflow_error(exc: BaseException) -> bool:
    """True if *exc* is llama.cpp's context-window-exceeded error.

    This is the hard 400 raised on PREFILL when the prompt is larger than the
    model's ``n_ctx`` — distinct from the post-response empty/truncation
    heuristic in ``services.truncation.is_context_overflow``.
    """
    s = str(exc).lower()
    return any(m in s for m in _OVERFLOW_MARKERS)


async def summarize_older_history(
    messages: list[Message],
    *,
    server_url: str,
    model_name: str,
    conversation_id: int,
    model_num_ctx: Optional[int],
    keep_percent: int,
) -> Optional[list[Message]]:
    """Consolidate the OLDER portion of *messages* into a summary and return
    ``[summary, *recent_messages]`` — or ``None`` if there's nothing safe to
    summarize (too few messages / no safe cutoff), in which case the caller
    should give up and re-raise the original overflow.

    The summary is produced by :class:`SummarizationMiddleware` (its safe-cutoff
    logic keeps AI/Tool message pairs together) using a ``ChatAgent`` on the
    SAME model the session uses. ``keep_percent`` is the percent of recent
    context to KEEP; the remainder (the "back"/older part) is summarized.
    """
    if not server_url:
        logger.warning("overflow recovery: no server_url available; cannot summarize")
        return None

    # --- the only agent-architecture touchpoint (isolated on purpose) -------
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    from agents.chat import ChatAgent
    from graph.middleware.summarization_middleware import SummarizationMiddleware
    from graph.workflows.base import _make_runner_http_client

    summary_model = ChatOpenAI(
        base_url=server_url,
        api_key=SecretStr("none"),
        model=model_name,
        http_async_client=_make_runner_http_client(),
    )
    summary_agent = ChatAgent(
        model=summary_model,
        system_prompt=_SUMMARY_SYSTEM_PROMPT,
        num_ctx=model_num_ctx or 90000,
        component_name="OverflowSummary",
    )
    # ------------------------------------------------------------------------

    middleware = SummarizationMiddleware(
        agent=summary_agent,
        conversation_id=conversation_id,
        # Force summarization regardless of the (approximate) token counter —
        # we already KNOW the real prompt overflowed, so we call the partition
        # logic directly rather than abefore_model's threshold gate (which uses
        # the same kind of approximate count that under-reported in the first
        # place). max_tokens_before_summary only scales the keep-target below.
        max_tokens_before_summary=model_num_ctx or 49152,
        percent_to_keep=keep_percent,
    )

    lc = messages_to_lc_messages(messages)
    middleware._ensure_message_ids(lc)
    cutoff = middleware._find_safe_cutoff(lc)
    if cutoff <= 0:
        logger.warning(
            "overflow recovery: no safe cutoff (too few messages to summarize)",
            extra={"n_messages": len(messages)},
        )
        return None

    to_summarize, preserved = middleware._partition_messages(lc, cutoff)
    summary_text = await middleware._create_summary(to_summarize)

    summary_msg = Message(
        role=MessageRole.USER,
        content=[MessageContent(type=MessageContentType.TEXT, text=summary_text)],
        conversation_id=conversation_id or None,
    )
    preserved_msgs = lc_messages_to_messages(preserved, conversation_id)

    logger.info(
        "overflow recovery: summarized older history",
        extra={
            "summarized_messages": len(to_summarize),
            "preserved_messages": len(preserved_msgs),
            "keep_percent": keep_percent,
            "model": model_name,
        },
    )
    return [summary_msg, *preserved_msgs]
