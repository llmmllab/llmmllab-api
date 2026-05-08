"""Tests for agents/base.py — context overflow guard and config-driven margins."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agents.base import BaseAgent, ContextOverflowError
from langchain_core.messages import HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """Return a minimal mock BaseChatModel."""
    m = AsyncMock()
    m.bind_tools = lambda *a, **k: m
    return m


@pytest.fixture
def agent(mock_model):
    """BaseAgent with a generous context window (90 k tokens)."""
    return BaseAgent(model=mock_model, num_ctx=90_000)


@pytest.fixture
def agent_small_ctx(mock_model):
    """BaseAgent with a tiny context window for overflow tests."""
    return BaseAgent(model=mock_model, num_ctx=500)


# ---------------------------------------------------------------------------
# _ensure_context_fits — happy path
# ---------------------------------------------------------------------------

class TestEnsureContextFits:
    """_ensure_context_fits trims or passes through correctly."""

    @pytest.mark.asyncio
    async def test_no_trim_when_fits(self, agent):
        """Messages well within budget are returned unchanged."""
        msgs = [HumanMessage(content="hello")]
        result = await agent._ensure_context_fits(msgs, "You are helpful.")
        assert result is msgs

    @pytest.mark.asyncio
    async def test_trims_old_messages(self, agent_small_ctx):
        """Old messages are dropped when total exceeds budget."""
        # Each message is ~100 chars ≈ 33 tokens.
        # num_ctx=500, safety_margin=0.85 → max_convo ≈ 425 - system_tokens.
        # 20 messages × 33 = 660 tokens > budget → should trim.
        msgs = [
            HumanMessage(content=f"message number {i} " + "x" * 90)
            for i in range(20)
        ]
        system = "You are a test agent. " + "y" * 50
        result = await agent_small_ctx._ensure_context_fits(msgs, system)
        assert len(result) < len(msgs), "Should have trimmed some messages"
        assert len(result) >= 2, "Should keep at least min_keep=2"

    @pytest.mark.asyncio
    async def test_keeps_last_message(self, agent_small_ctx):
        """The most recent user message is always preserved."""
        msgs = [
            HumanMessage(content=f"old message {i} " + "x" * 90)
            for i in range(20)
        ]
        msgs.append(HumanMessage(content="THIS IS THE LAST MESSAGE"))
        system = "You are a test agent. " + "y" * 50
        result = await agent_small_ctx._ensure_context_fits(msgs, system)
        last_content = result[-1].content
        assert "THIS IS THE LAST MESSAGE" in last_content

    @pytest.mark.asyncio
    async def test_zero_num_ctx_returns_unchanged(self, mock_model):
        """num_ctx=0 skips the check entirely."""
        a = BaseAgent(model=mock_model, num_ctx=0)
        msgs = [HumanMessage(content="hi")]
        result = await a._ensure_context_fits(msgs, "system")
        assert result is msgs

    @pytest.mark.asyncio
    async def test_none_num_ctx_returns_unchanged(self, mock_model):
        """num_ctx=None skips the check entirely."""
        a = BaseAgent(model=mock_model, num_ctx=None)
        msgs = [HumanMessage(content="hi")]
        result = await a._ensure_context_fits(msgs, "system")
        assert result is msgs

    @pytest.mark.asyncio
    async def test_empty_messages(self, agent):
        """Empty message list returns empty."""
        result = await agent._ensure_context_fits([], "system")
        assert result == []


# ---------------------------------------------------------------------------
# ContextOverflowError — raised when trimming can't help
# ---------------------------------------------------------------------------

class TestContextOverflowError:
    """ContextOverflowError is raised when context can't be reduced enough."""

    @pytest.mark.asyncio
    async def test_overflow_raised_when_trim_exhausted(self, mock_model):
        """When even min_keep messages exceed budget, raise error."""
        # Create an agent with a tiny context and a huge system prompt
        # so that max_convo_tokens is very small.
        a = BaseAgent(model=mock_model, num_ctx=100)
        # System prompt alone consumes most of the budget
        system = "S" * 200  # ~67 tokens
        # Messages that can't be trimmed enough
        msgs = [
            HumanMessage(content="M" * 200),  # ~67 tokens
            AIMessage(content="A" * 200),     # ~67 tokens
        ]
        with patch("agents.base.CONTEXT_USAGE_SAFETY_MARGIN", 0.85):
            with pytest.raises(ContextOverflowError) as exc_info:
                await a._ensure_context_fits(msgs, system)

        err = exc_info.value
        assert "context" in err.message.lower()
        assert err.suggested_context_size is not None
        assert err.suggested_context_size > 0

    @pytest.mark.asyncio
    async def test_error_attributes(self):
        """Error carries message and suggested_context_size."""
        err = ContextOverflowError(
            message="Too big",
            suggested_context_size=120_000,
        )
        assert err.message == "Too big"
        assert err.suggested_context_size == 120_000
        assert str(err) == "Too big"


# ---------------------------------------------------------------------------
# Config-driven safety margin
# ---------------------------------------------------------------------------

class TestConfigDrivenMargin:
    """CONTEXT_USAGE_SAFETY_MARGIN is read from config, not hardcoded."""

    @pytest.mark.asyncio
    async def test_uses_config_margin(self, mock_model):
        """Changing the config constant changes the budget."""
        a = BaseAgent(model=mock_model, num_ctx=10_000)
        # With margin=0.5, budget is halved → more aggressive trimming
        with patch("agents.base.CONTEXT_USAGE_SAFETY_MARGIN", 0.5):
            msgs = [HumanMessage(content="X" * 300) for _ in range(10)]
            result_half = await a._ensure_context_fits(msgs, "sys")

        # With margin=0.95, budget is generous → less trimming
        with patch("agents.base.CONTEXT_USAGE_SAFETY_MARGIN", 0.95):
            msgs = [HumanMessage(content="X" * 300) for _ in range(10)]
            result_high = await a._ensure_context_fits(msgs, "sys")

        # Higher margin keeps more messages
        assert len(result_high) >= len(result_half)

    @pytest.mark.asyncio
    async def test_config_import_available(self):
        """The config module exports the new constants."""
        from config import CONTEXT_USAGE_SAFETY_MARGIN, CONTEXT_MINIMUM_RATIO
        assert isinstance(CONTEXT_USAGE_SAFETY_MARGIN, float)
        assert 0 < CONTEXT_USAGE_SAFETY_MARGIN <= 1
        assert isinstance(CONTEXT_MINIMUM_RATIO, float)
        assert 0 <= CONTEXT_MINIMUM_RATIO <= 1


# ---------------------------------------------------------------------------
# run() — ContextOverflowError returns user-friendly response
# ---------------------------------------------------------------------------

class TestRunContextOverflow:
    """run() handles ContextOverflowError gracefully."""

    @pytest.mark.asyncio
    async def test_run_returns_overflow_message(self, mock_model):
        """When _ensure_context_fits raises, run() returns a ChatResponse."""
        a = BaseAgent(model=mock_model, num_ctx=100)

        # Patch _ensure_context_fits to always raise
        async def fake_ensure(*args, **kwargs):
            raise ContextOverflowError(
                message="Context too large.",
                suggested_context_size=200_000,
            )

        with patch.object(a, "_ensure_context_fits", fake_ensure):
            response = await a.run(messages="test message")

        assert response.done is True
        text = response.message.content[0].text
        assert "context" in text.lower() or "too large" in text.lower()
