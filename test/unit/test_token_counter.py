"""Tests for the llama.cpp ``/tokenize``-backed token counter.

These verify that we hit the right URL, deserialise the response correctly,
and surface failures as ``None`` (so callers can skip proactive trimming
without falling back to the broken char-based estimate).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from services.token_counter import count_tokens, _coerce_to_text


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_resp(status: int = 200, body=None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value=body if body is not None else {})
    return resp


def _make_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    http = MagicMock()
    http.post = AsyncMock(return_value=response)
    client._get_client = MagicMock(return_value=http)
    return client


def test_count_tokens_returns_token_array_length():
    client = _make_client(_mock_resp(200, {"tokens": [12, 34, 56, 78, 9]}))

    n = _run(count_tokens(
        "hello world",
        base_url="http://runner-1:8000/v1/server/abc",
        client=client,
    ))

    assert n == 5


def test_count_tokens_posts_to_tokenize_endpoint():
    """Verify the path is suffixed correctly to the server handle base_url."""
    client = _make_client(_mock_resp(200, {"tokens": []}))

    _run(count_tokens(
        "abc", base_url="http://runner-1:8000/v1/server/xyz", client=client,
    ))

    http = client._get_client.return_value
    args, kwargs = http.post.call_args
    assert args[0] == "http://runner-1:8000/v1/server/xyz/tokenize"
    assert kwargs["json"] == {"content": "abc"}


def test_count_tokens_handles_trailing_slash_in_base_url():
    client = _make_client(_mock_resp(200, {"tokens": [1]}))

    _run(count_tokens(
        "abc", base_url="http://runner-1:8000/v1/server/xyz/", client=client,
    ))

    http = client._get_client.return_value
    args, _ = http.post.call_args
    # Must not double-slash
    assert "//tokenize" not in args[0].split("://", 1)[1]


def test_count_tokens_returns_zero_for_empty_string():
    client = MagicMock()
    n = _run(count_tokens("", base_url="http://x", client=client))
    assert n == 0


def test_count_tokens_returns_none_on_http_error():
    client = _make_client(_mock_resp(500))

    n = _run(count_tokens(
        "abc", base_url="http://runner-1:8000/v1/server/xyz", client=client,
    ))

    assert n is None


def test_count_tokens_returns_none_on_network_failure():
    client = MagicMock()
    http = MagicMock()
    http.post = AsyncMock(side_effect=ConnectionError("boom"))
    client._get_client = MagicMock(return_value=http)

    n = _run(count_tokens(
        "abc", base_url="http://runner-1:8000/v1/server/xyz", client=client,
    ))

    assert n is None


def test_count_tokens_returns_none_when_body_missing_tokens():
    """If llama-server changes its response shape we don't want a fake
    zero-count slipping through as a falsy success."""
    client = _make_client(_mock_resp(200, {"not_tokens": "something else"}))

    n = _run(count_tokens(
        "abc", base_url="http://runner-1:8000/v1/server/xyz", client=client,
    ))

    assert n is None


def test_count_tokens_returns_none_when_base_url_missing():
    client = MagicMock()
    n = _run(count_tokens("abc", base_url="", client=client))
    assert n is None


def test_coerce_to_text_flattens_structured_blocks():
    """Mixed string + dict + nested content collapses to a single string we
    can hand to /tokenize without losing the actual text."""
    blocks = [
        "first part",
        {"type": "text", "text": "second part"},
        {"type": "tool_use", "content": "third part"},
        {"type": "unknown"},
    ]
    out = _coerce_to_text(blocks)
    assert "first part" in out
    assert "second part" in out
    assert "third part" in out
