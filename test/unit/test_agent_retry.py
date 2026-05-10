"""
Unit tests for agents/base.py — retry logic for transient APIConnectionError.

The retry block in BaseAgent.run() catches openai.APIConnectionError and retries
up to 10 times (11 total attempts) with exponential backoff capped at 60 s.

These tests mock the agent's ainvoke() to control when errors are raised and
when success occurs, verifying:
- Success on first attempt (no retry)
- Retry on transient error, success on a later attempt
- Exhaustion of all retries re-raises the last error
- Non-transient errors propagate immediately (no retry)
- Backoff values follow the expected exponential schedule
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openai import APIConnectionError

from agents.base import BaseAgent


def _make_base_agent() -> BaseAgent:
    """Create a BaseAgent with a mocked model."""
    model = MagicMock()
    return BaseAgent(model=model, system_prompt="You are helpful.")


class TestAgentRetrySuccessFirstAttempt:
    """Agent succeeds on the first ainvoke call — no retry needed."""

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        agent = _make_base_agent()

        # Mock _get_or_create_agent to return an agent whose ainvoke succeeds
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.return_value = MagicMock(
            content="Hello!", role="ai"
        )
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        response = await agent.run("Hi there")

        assert response.done is True
        # ainvoke should have been called exactly once
        mock_lc_agent.ainvoke.assert_called_once()


class TestAgentRetryTransientError:
    """Agent encounters APIConnectionError and retries."""

    @pytest.mark.asyncio
    async def test_retry_on_api_connection_error(self):
        agent = _make_base_agent()

        call_count = [0]

        async def flaky_ainvoke(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise APIConnectionError(message="Connection refused", request=MagicMock())
            return MagicMock(content="Recovered!", role="ai")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky_ainvoke
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        # Patch asyncio.sleep to avoid actual delays
        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await agent.run("Hi there")

        assert response.done is True
        # Should have been called 3 times (2 failures + 1 success)
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_retry_logs_warning(self, caplog):
        """Each retry emits a WARNING log with attempt info."""
        agent = _make_base_agent()

        call_count = [0]

        async def flaky_ainvoke(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 2:
                raise APIConnectionError(message="Connection refused", request=MagicMock())
            return MagicMock(content="OK", role="ai")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky_ainvoke
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await agent.run("Hi there")

        # There should be at least one WARNING about retrying
        warning_messages = [r.msg for r in caplog.records if r.levelname == "WARNING"]
        assert any("retrying" in str(m).lower() for m in warning_messages)


class TestAgentRetryExhaustion:
    """All retries fail — error is returned as ChatResponse (outer except catches it)."""

    @pytest.mark.asyncio
    async def test_returns_error_response_after_max_attempts(self):
        agent = _make_base_agent()

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = APIConnectionError(
            message="Connection refused", request=MagicMock()
        )
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await agent.run("Hi there")

        # The outer try/except in run() catches APIConnectionError and
        # returns an error ChatResponse (it only re-raises TimeoutError).
        assert response.done is True
        text = response.message.content[0].text
        assert "Connection refused" in text
        # max_attempts = 11, so ainvoke should be called 11 times
        assert mock_lc_agent.ainvoke.call_count == 11


class TestAgentRetryNonTransientError:
    """Non-APIConnectionError exceptions propagate immediately."""

    @pytest.mark.asyncio
    async def test_value_error_propagates_immediately(self):
        agent = _make_base_base_agent = _make_base_agent()

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = ValueError("bad model")
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        # The run() method catches generic exceptions and returns an error ChatResponse
        response = await agent.run("Hi there")

        assert response.done is True
        # ainvoke called only once — no retry for ValueError
        assert mock_lc_agent.ainvoke.call_count == 1


class TestAgentRetryBackoffSchedule:
    """Verify backoff delays follow exponential schedule capped at 60s."""

    @pytest.mark.asyncio
    async def test_backoff_values(self):
        agent = _make_base_agent()

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        call_count = [0]

        async def always_fail(*a, **kw):
            call_count[0] += 1
            raise APIConnectionError(message="fail", request=MagicMock())

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = always_fail
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            response = await agent.run("Hi there")

        # The outer except catches the error and returns a ChatResponse,
        # but the retry loop still ran all 11 attempts with 10 sleeps.
        assert response.done is True
        assert call_count[0] == 11

        # 11 attempts → 10 sleeps
        assert len(sleep_calls) == 10

        # Expected backoffs: 2, 4, 8, 16, 32, 60, 60, 60, 60, 60
        expected = [2, 4, 8, 16, 32, 60, 60, 60, 60, 60]
        assert sleep_calls == expected
