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

# Threshold (in tokens) above which we consider the prompt "large".
# When a large prompt produces an empty response, retrying is futile —
# the context is likely exceeding the model's window.
CONTEXT_OVERFLOW_THRESHOLD = 100_000

SENTENCE_TERMINATORS = frozenset('.!?)\n`]}"\'>,:')
TRUNCATION_MIN_LEN = 40
