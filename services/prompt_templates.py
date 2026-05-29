"""
Prompt templates and truncation/overflow thresholds used by CompletionService.

These constants drive the secondary-pass behaviour: continuation prompts when
the model truncates or skips a tool call, the nudge prompt when retries fail,
and the heuristics for detecting mid-sentence truncation.
"""


CONTINUATION_PROMPT = (
    "You described using a tool but did not actually call one. "
    "Call the appropriate tool now. Do not describe what you will do — invoke the tool directly."
)

EMPTY_RESPONSE_NUDGE = (
    "Your response didn't produce any output. Did you mean to say something "
    "or use a tool? If so, continue. Otherwise, simply respond with 'done' "
    "and nothing else."
)

TRUNCATION_CONTINUATION_PROMPT = (
    "Your response was cut off. Continue from where you left off. "
    "If you were in the middle of a tool call, complete the tool call. "
    "If you were in the middle of text, continue the text."
)


def hallucinated_tool_feedback(
    invalid_tool_names: list[str],
    valid_names: set[str],
    *,
    max_listed: int = 25,
) -> str:
    """Build a user-message-style feedback string for a hallucinated tool call.

    The model emitted a tool call with a name that wasn't in the bound
    list (e.g. ``browser`` or ``memory_get`` — both pretraining-frequent
    generic names that no MCP server actually exposes here).  We feed
    this string back as the next user turn so the model gets explicit
    feedback in conversation context and picks a real name on its
    retry.

    ``max_listed`` caps the number of valid names included — with 50+
    tools we can't dump them all into context without blowing the
    window.  The listed slice is alphabetical for stability across
    requests.
    """
    listed = sorted(valid_names)[:max_listed]
    remainder = max(0, len(valid_names) - max_listed)
    suffix = f" (and {remainder} more)" if remainder else ""
    bad = ", ".join(sorted(set(invalid_tool_names)))
    return (
        f"ERROR: You tried to call tool(s) [{bad}] but none of those "
        f"names are in the bound tool list — your tool call was dropped "
        f"and never executed.  Pick from the actual tool names: "
        f"{', '.join(listed)}{suffix}.  Retry with a valid tool name."
    )

# Threshold (in tokens) above which we consider the prompt "large".
# When a large prompt produces an empty response, retrying is futile —
# the context is likely exceeding the model's window.
CONTEXT_OVERFLOW_THRESHOLD = 100_000

SENTENCE_TERMINATORS = frozenset('.!?)\n`]}"\'>,:')
TRUNCATION_MIN_LEN = 40
