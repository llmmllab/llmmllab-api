"""Tests for early context-size guard: ContextExceededError and _check_context_fits.

Verifies that the gateway refuses to start a server when the estimated input
token count exceeds the model's context window (with safety margin applied).
This is the implementation of the review feedback for PR #71:
  "the runner should refuse to even start a server for a request where the
   context is larger than the context window."
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graph.errors import ContextExceededError
from models import ModelDetails, ModelParameters
from models.model import Model
from services.completion_service import (
    _check_context_fits,
    _estimate_message_token_count,
    _get_effective_context_window,
)
from models.message import Message, MessageContent, MessageContentType, MessageRole


# ---------------------------------------------------------------------------
# ContextExceededError
# ---------------------------------------------------------------------------


class TestContextExceededError:
    """ContextExceededError carries the right attributes and message."""

    def test_error_attributes(self):
        err = ContextExceededError(
            estimated_tokens=100_000,
            model_context_window=8192,
            model_name="llama-3-8b",
        )
        assert err.estimated_tokens == 100_000
        assert err.model_context_window == 8192
        assert err.model_name == "llama-3-8b"
        assert "llama-3-8b" in err.message
        assert "100,000" in err.message
        assert "8,192" in err.message
        assert err.details["estimated_tokens"] == 100_000
        assert err.details["model_context_window"] == 8192

    def test_error_is_composer_error(self):
        from graph.errors import ComposerError
        err = ContextExceededError(100, 50, "m")
        assert isinstance(err, ComposerError)


# ---------------------------------------------------------------------------
# _estimate_message_token_count
# ---------------------------------------------------------------------------


class TestEstimateMessageTokenCount:
    """Token estimation uses ~3 chars per token heuristic."""

    def test_empty_messages(self):
        assert _estimate_message_token_count([]) == 0

    def test_single_message(self):
        msg = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="Hello world")],
        )
        count = _estimate_message_token_count([msg])
        # "Hello world" = 11 chars // 3 = 3 tokens + 15 overhead = 18
        assert count == 18

    def test_system_prompt_included(self):
        msg = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="hi")],
        )
        count = _estimate_message_token_count([msg], system_text="You are a helpful assistant.")
        # system: 28 chars // 3 = 9, msg: 2//3=0 + 15 = 15, total = 24
        assert count == 24

    def test_multiple_messages(self):
        msgs = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="A" * 300)],
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="B" * 300)],
            ),
        ]
        count = _estimate_message_token_count(msgs)
        # Each: 300//3=100 + 15 = 115, total = 230
        assert count == 230

    def test_large_conversation(self):
        msgs = [
            Message(
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="X" * 1000)],
            )
            for i in range(10)
        ]
        count = _estimate_message_token_count(msgs)
        # Each: 1000//3=333 + 15 = 348, total = 3480
        assert count == 3480


# ---------------------------------------------------------------------------
# _get_effective_context_window
# ---------------------------------------------------------------------------


class TestGetEffectiveContextWindow:
    """Safety margin is applied to raw num_ctx."""

    def test_normal_value(self):
        with patch("config.CONTEXT_USAGE_SAFETY_MARGIN", 0.85):
            result = _get_effective_context_window(10_000)
            assert result == 8500

    def test_none_input(self):
        assert _get_effective_context_window(None) is None

    def test_zero_input(self):
        assert _get_effective_context_window(0) is None

    def test_negative_input(self):
        assert _get_effective_context_window(-1) is None


# ---------------------------------------------------------------------------
# _check_context_fits — integration
# ---------------------------------------------------------------------------


class TestCheckContextFits:
    """_check_context_fits raises ContextExceededError when input exceeds window."""

    def _make_model(self, original_ctx: int):
        """Build a minimal Model with the given original_ctx."""
        return Model(
            id="test-model",
            name="Test Model",
            model="test-org/test-model",
            task="TextToText",
            provider="llama_cpp",
            modified_at="2025-01-01",
            digest="abc123",
            details=ModelDetails(
                format="gguf",
                family="llama",
                families=["llama"],
                parameter_size="8B",
                size=4_000_000_000,
                original_ctx=original_ctx,
            ),
            parameters=ModelParameters(),
        )

    @pytest.mark.asyncio
    async def test_raises_when_exceeds(self):
        """Small context window + large messages → raises."""
        model = self._make_model(original_ctx=1000)
        msgs = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="X" * 5000)],
            )
        ]
        mock_model_service = AsyncMock()
        mock_model_service.get_model_by_id = AsyncMock(return_value=model)

        with patch("services.model_service", mock_model_service):
            with pytest.raises(ContextExceededError) as exc_info:
                await _check_context_fits(msgs, "test-model")

        err = exc_info.value
        assert err.model_name == "test-model"
        assert err.model_context_window == 1000
        assert err.estimated_tokens > 0

    @pytest.mark.asyncio
    async def test_passes_when_fits(self):
        """Large context window + small messages → no error."""
        model = self._make_model(original_ctx=100_000)
        msgs = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="Hello")],
            )
        ]
        mock_model_service = AsyncMock()
        mock_model_service.get_model_by_id = AsyncMock(return_value=model)

        with patch("services.model_service", mock_model_service):
            await _check_context_fits(msgs, "test-model")  # should not raise

    @pytest.mark.asyncio
    async def test_passes_when_model_not_found(self):
        """Model lookup failure → silently passes (runner will refuse)."""
        msgs = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="X" * 50000)],
            )
        ]
        mock_model_service = AsyncMock()
        mock_model_service.get_model_by_id = AsyncMock(return_value=None)

        with patch("services.model_service", mock_model_service):
            await _check_context_fits(msgs, "unknown-model")  # should not raise

    @pytest.mark.asyncio
    async def test_passes_when_model_lookup_raises(self):
        """Model lookup exception → silently passes."""
        msgs = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="X" * 50000)],
            )
        ]
        mock_model_service = AsyncMock()
        mock_model_service.get_model_by_id = AsyncMock(side_effect=Exception("cache miss"))

        with patch("services.model_service", mock_model_service):
            await _check_context_fits(msgs, "test-model")  # should not raise

    @pytest.mark.asyncio
    async def test_passes_when_no_original_ctx(self):
        """Model with no original_ctx → silently passes."""
        model = self._make_model(original_ctx=0)
        msgs = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="X" * 50000)],
            )
        ]
        mock_model_service = AsyncMock()
        mock_model_service.get_model_by_id = AsyncMock(return_value=model)

        with patch("services.model_service", mock_model_service):
            await _check_context_fits(msgs, "test-model")  # should not raise

    @pytest.mark.asyncio
    async def test_safety_margin_applied(self):
        """Safety margin reduces effective window; messages near raw ctx may be rejected."""
        model = self._make_model(original_ctx=1000)
        # 1000 * 0.85 = 850 effective. Message of ~2550 chars = 850 tokens + 15 overhead = 865 > 850
        msgs = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="X" * 2550)],
            )
        ]
        mock_model_service = AsyncMock()
        mock_model_service.get_model_by_id = AsyncMock(return_value=model)

        with patch("services.model_service", mock_model_service):
            with patch("config.CONTEXT_USAGE_SAFETY_MARGIN", 0.85):
                with pytest.raises(ContextExceededError):
                    await _check_context_fits(msgs, "test-model")
