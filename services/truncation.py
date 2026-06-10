"""
Pure detection functions for response truncation and context-window overflow.

These helpers are shared between the streaming and non-streaming retry paths
in CompletionService. They depend only on token counts, finish reasons, and
the model's context window — no I/O, no side effects.
"""

from services.prompt_templates import (
    CONTEXT_OVERFLOW_THRESHOLD,
    SENTENCE_TERMINATORS,
    TRUNCATION_MIN_LEN,
)


def is_context_overflow(
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
    # When the real window is known, the ONLY reliable overflow signal is
    # exceeding it. An empty/short response that still fits inside the window is
    # a transient stop (the model occasionally emits EOS early) and IS
    # retryable — flagging it as overflow skipped the rescue-retry and
    # dead-ended sessions that recovered fine on the very next attempt.
    if model_num_ctx is not None:
        return (prompt_tokens + output_tokens) >= model_num_ctx

    # Window unknown: fall back to the prompt-size + empty-response heuristic.
    if prompt_tokens < CONTEXT_OVERFLOW_THRESHOLD:
        return False
    if output_tokens > 0:
        return False
    # Empty response with a large prompt — likely context overflow.
    # Also catches finish_reason == 'length' with zero output.
    return True


def is_truncated(text: str, finish_reason: str) -> bool:
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
    if len(stripped) < TRUNCATION_MIN_LEN:
        return False
    return stripped[-1] not in SENTENCE_TERMINATORS
