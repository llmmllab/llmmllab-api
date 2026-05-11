"""
Unit tests for agents/base.py — proactive context window truncation.

Before invoking the LLM, the agent estimates total tokens (system prompt +
conversation messages) and removes oldest messages if the estimate exceeds
the context window (with a safety margin).

This prevents wasteful 400 exceed_context_size_error responses.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.base import BaseAgent
from models import Message, MessageRole, MessageContent, MessageContentType


def _make_base_agent(num_ctx: int = 50176, system_prompt: str = "You are helpful."):
    """Create a BaseAgent with a mocked model."""
    model = MagicMock()
    return BaseAgent(
        model=model,
        system_prompt=system_prompt,
        num_ctx=num_ctx,
    )


def _make_message(role: MessageRole, text: str) -> Message:
    """Create a simple text Message."""
    return Message(
        role=role,
        content=[MessageContent(type=MessageContentType.TEXT, text=text)],
    )


class TestProactiveContextTruncation:
    """Proactive truncation fires before the first LLM invocation."""

    @pytest.mark.asyncio
    async def test_truncates_when_estimated_tokens_exceed_context(self):
        """When estimated tokens exceed num_ctx, oldest messages are trimmed."""
        agent = _make_base_agent(num_ctx=5000)

        # Build a conversation whose estimated tokens exceed 5000.
        # estimate_tokens uses ~3 chars/token, so 5000 tokens ≈ 15000 chars.
        # With safety factor 1.15, we need > 5000/1.15 ≈ 4348 estimated tokens.
        # Each message has ~15 overhead tokens.
        big_text = "x" * 10000  # ~3333 tokens per message
        messages = [
            _make_message(MessageRole.USER, big_text),
            _make_message(MessageRole.ASSISTANT, big_text),
            _make_message(MessageRole.USER, "final question"),
        ]

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.return_value = {
            "messages": [MagicMock(role="ai", content="OK")]
        }
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        response = await agent.run(messages)

        assert response.done is True
        # The agent should have been invoked (truncation happens before invoke)
        assert mock_lc_agent.ainvoke.call_count == 1
        # Check that the messages passed to ainvoke were trimmed
        call_args = mock_lc_agent.ainvoke.call_args
        passed_messages = call_args[0][0]["messages"]
        # Should have fewer messages than the original 3
        assert len(passed_messages) < len(messages)

    @pytest.mark.asyncio
    async def test_no_truncation_when_tokens_fit(self):
        """When estimated tokens fit within num_ctx, no truncation occurs."""
        agent = _make_base_agent(num_ctx=50000)

        short_messages = [
            _make_message(MessageRole.USER, "Hello"),
            _make_message(MessageRole.ASSISTANT, "Hi there!"),
            _make_message(MessageRole.USER, "How are you?"),
        ]

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.return_value = {
            "messages": [MagicMock(role="ai", content="I'm fine!")]
        }
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        response = await agent.run(short_messages)

        assert response.done is True
        assert mock_lc_agent.ainvoke.call_count == 1
        # All messages should be passed through
        call_args = mock_lc_agent.ainvoke.call_args
        passed_messages = call_args[0][0]["messages"]
        assert len(passed_messages) == len(short_messages)

    @pytest.mark.asyncio
    async def test_no_truncation_when_num_ctx_is_none(self):
        """When num_ctx is None, no proactive truncation occurs."""
        model = MagicMock()
        agent = BaseAgent(
            model=model,
            system_prompt="You are helpful.",
            num_ctx=None,
        )

        big_text = "x" * 100000
        messages = [
            _make_message(MessageRole.USER, big_text),
            _make_message(MessageRole.ASSISTANT, big_text),
            _make_message(MessageRole.USER, "final"),
        ]

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.return_value = {
            "messages": [MagicMock(role="ai", content="OK")]
        }
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        response = await agent.run(messages)

        assert response.done is True
        # All messages should be passed through (no truncation)
        call_args = mock_lc_agent.ainvoke.call_args
        passed_messages = call_args[0][0]["messages"]
        assert len(passed_messages) == len(messages)

    @pytest.mark.asyncio
    async def test_keeps_at_least_one_message(self):
        """Truncation never removes all messages — at least the last one is kept."""
        agent = _make_base_agent(num_ctx=100)

        # A single huge message that alone exceeds the context
        big_text = "x" * 100000
        messages = [_make_message(MessageRole.USER, big_text)]

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.return_value = {
            "messages": [MagicMock(role="ai", content="OK")]
        }
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        response = await agent.run(messages)

        assert response.done is True
        # With only 1 message, truncation stops (len(convo) > 1 guard)
        call_args = mock_lc_agent.ainvoke.call_args
        passed_messages = call_args[0][0]["messages"]
        assert len(passed_messages) == 1

    @pytest.mark.asyncio
    async def test_truncation_preserves_last_message(self):
        """The most recent (last) message is always preserved."""
        agent = _make_base_agent(num_ctx=5000)

        big_text = "x" * 10000
        final_text = "This is my final question"
        messages = [
            _make_message(MessageRole.USER, big_text),
            _make_message(MessageRole.ASSISTANT, big_text),
            _make_message(MessageRole.USER, final_text),
        ]

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.return_value = {
            "messages": [MagicMock(role="ai", content="OK")]
        }
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        await agent.run(messages)

        call_args = mock_lc_agent.ainvoke.call_args
        passed_messages = call_args[0][0]["messages"]
        # The last message should still be present
        last_msg = passed_messages[-1]
        assert final_text in str(last_msg.content)
