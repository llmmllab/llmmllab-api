"""Unit tests for reactive context-overflow recovery."""

import pytest

from services.overflow_recovery import is_overflow_error, summarize_older_history


class TestIsOverflowError:
    def test_llamacpp_exceed_context_error(self):
        e = Exception(
            "Error code: 400 - {'error': {'code': 400, 'message': 'request "
            "(33247 tokens) exceeds the available context size (32768 tokens), "
            "try increasing it', 'type': 'exceed_context_size_error'}}"
        )
        assert is_overflow_error(e) is True

    def test_type_marker_alone(self):
        assert is_overflow_error(Exception("exceed_context_size_error")) is True

    def test_message_marker_alone(self):
        assert is_overflow_error(Exception("exceeds the available context size")) is True

    def test_unrelated_error_is_not_overflow(self):
        assert is_overflow_error(Exception("connection reset by peer")) is False
        assert is_overflow_error(ValueError("bad request: missing field")) is False


class TestSummarizeOlderHistory:
    @pytest.mark.asyncio
    async def test_no_server_url_returns_none(self):
        # Returns before importing/instantiating any agent — the caller then
        # re-raises the original overflow (honest failure, no crash).
        out = await summarize_older_history(
            [],
            server_url="",
            model_name="Gemma4_12B",
            conversation_id=0,
            model_num_ctx=49152,
            keep_percent=50,
        )
        assert out is None
