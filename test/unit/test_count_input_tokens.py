"""Tests for count_input_tokens' templated-prompt path + tool normalization.

The message_start input_tokens a client displays must reflect the FULL
chat-templated prompt (system + messages + tools), which the model's own
template renders via /apply-template. A per-message text sum undercounts ~30%
(it misses chat-template/tool-call markup); validated live, apply-template+
tokenize matches the real prompt_eval_count to ~0.5%.
"""

from unittest.mock import AsyncMock, patch

import pytest

from services import token_counter as tc


# --- _tools_to_openai ------------------------------------------------------

def test_tools_to_openai_from_anthropic():
    out = tc._tools_to_openai([
        {"name": "bash", "description": "run", "input_schema": {"type": "object"}},
    ])
    assert out == [{"type": "function", "function": {
        "name": "bash", "description": "run", "parameters": {"type": "object"}}}]


def test_tools_to_openai_passthrough_openai_shape():
    oai = {"type": "function", "function": {"name": "x", "parameters": {}}}
    assert tc._tools_to_openai([oai]) == [oai]


def test_tools_to_openai_empty_is_none():
    assert tc._tools_to_openai(None) is None
    assert tc._tools_to_openai([]) is None
    assert tc._tools_to_openai([object()]) is None  # nothing usable


# --- count_input_tokens prefers the templated path -------------------------

@pytest.mark.asyncio
async def test_prefers_templated_path_when_available():
    with patch.object(tc, "_count_templated_input_tokens",
                      AsyncMock(return_value=12345)):
        n = await tc.count_input_tokens(
            [{"role": "user", "content": "hi"}], base_url="http://r")
    assert n == 12345


@pytest.mark.asyncio
async def test_falls_back_when_templated_returns_none():
    # /apply-template unavailable → None → per-message fallback (char/4 here,
    # since base_url is set but count_message_tokens is mocked to None).
    with patch.object(tc, "_count_templated_input_tokens",
                      AsyncMock(return_value=None)), \
         patch.object(tc, "count_message_tokens", AsyncMock(return_value=None)):
        n = await tc.count_input_tokens(
            [{"role": "user", "content": "a" * 400}],
            base_url="http://r", system_prompt="sys")
    assert n > 1  # produced a real fallback estimate, never zero


@pytest.mark.asyncio
async def test_no_base_url_skips_templated_path():
    # Without a server the templated path can't run; must not even be attempted.
    with patch.object(tc, "_count_templated_input_tokens",
                      AsyncMock(side_effect=AssertionError("should not be called"))):
        n = await tc.count_input_tokens([{"role": "user", "content": "hello"}],
                                        base_url=None)
    assert n >= 1
