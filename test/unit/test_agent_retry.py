"""
Unit tests for agents/base.py — retry logic for transient API errors.

The retry block in BaseAgent.run() catches:
- openai.APIConnectionError (connection drops)
- openai.APIStatusError with status 502/503/504 (server busy/gateway errors)

and retries up to 10 times (11 total attempts) with exponential backoff capped at 60 s.

Non-transient errors (e.g. ValueError, 400 Bad Request) propagate immediately.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openai import APIConnectionError, APIStatusError

from agents.base import BaseAgent


def _make_base_agent() -> BaseAgent:
    """Create a BaseAgent with a mocked model."""
    model = MagicMock()
    return BaseAgent(model=model, system_prompt="You are helpful.")


def _make_mock_response(status_code: int) -> MagicMock:
    """Build a minimal httpx-like response for APIStatusError."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {}
    return resp


def _make_api_status_error(status_code: int, message: str = "error") -> APIStatusError:
    """Create an APIStatusError with a given status code."""
    resp = _make_mock_response(status_code)
    return APIStatusError(message, response=resp, body={"detail": message})


# ── Success on first attempt ──────────────────────────────────────────────

class TestAgentRetrySuccessFirstAttempt:
    """Agent succeeds on the first ainvoke call — no retry needed."""

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.return_value = MagicMock(content="Hello!", role="ai")
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        response = await agent.run("Hi there")

        assert response.done is True
        mock_lc_agent.ainvoke.assert_called_once()


# ── Retry on APIConnectionError ───────────────────────────────────────────

class TestAgentRetryConnectionError:
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

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await agent.run("Hi there")

        assert response.done is True
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

        warning_messages = [r.msg for r in caplog.records if r.levelname == "WARNING"]
        assert any("retrying" in str(m).lower() for m in warning_messages)


# ── Retry on 503 Service Unavailable ──────────────────────────────────────

class TestAgentRetry503:
    """503 'All inference slots busy' should be retried."""

    @pytest.mark.asyncio
    async def test_retry_on_503(self):
        agent = _make_base_agent()
        call_count = [0]

        async def flaky_ainvoke(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise _make_api_status_error(503, "All inference slots are busy")
            return MagicMock(content="Recovered!", role="ai")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky_ainvoke
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await agent.run("Hi there")

        assert response.done is True
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_retry_on_502(self):
        agent = _make_base_agent()
        call_count = [0]

        async def flaky_ainvoke(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 2:
                raise _make_api_status_error(502, "Bad Gateway")
            return MagicMock(content="OK", role="ai")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky_ainvoke
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await agent.run("Hi there")

        assert response.done is True
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_retry_on_504(self):
        agent = _make_base_agent()
        call_count = [0]

        async def flaky_ainvoke(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 2:
                raise _make_api_status_error(504, "Gateway Timeout")
            return MagicMock(content="OK", role="ai")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky_ainvoke
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await agent.run("Hi there")

        assert response.done is True
        assert call_count[0] == 2


# ── Non-transient errors propagate immediately ────────────────────────────

class TestAgentRetryNonTransientError:
    """Non-transient exceptions propagate immediately (no retry)."""

    @pytest.mark.asyncio
    async def test_value_error_propagates_immediately(self):
        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = ValueError("bad model")
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        response = await agent.run("Hi there")

        assert response.done is True
        assert mock_lc_agent.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_400_not_retried(self):
        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = _make_api_status_error(400, "Bad Request")
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        response = await agent.run("Hi there")

        assert response.done is True
        assert mock_lc_agent.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_500_not_retried(self):
        """500 Internal Server Error is NOT in the transient set (only 502/503/504)."""
        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = _make_api_status_error(500, "Internal Server Error")
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        response = await agent.run("Hi there")

        assert response.done is True
        assert mock_lc_agent.ainvoke.call_count == 1


# ── Retry exhaustion ──────────────────────────────────────────────────────

class TestAgentRetryExhaustion:
    """All retries fail — error is returned as ChatResponse."""

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

        assert response.done is True
        text = response.message.content[0].text
        assert "Connection refused" in text
        assert mock_lc_agent.ainvoke.call_count == 11

    @pytest.mark.asyncio
    async def test_503_exhaustion(self):
        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = _make_api_status_error(503, "All inference slots are busy")
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await agent.run("Hi there")

        assert response.done is True
        text = response.message.content[0].text
        assert "All inference slots are busy" in text
        assert mock_lc_agent.ainvoke.call_count == 11


# ── Backoff schedule ──────────────────────────────────────────────────────

class TestAgentRetryBackoffSchedule:
    """Verify backoff delays follow exponential schedule capped at 60s."""

    @pytest.mark.asyncio
    async def test_backoff_values(self):
        agent = _make_base_agent()
        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = APIConnectionError(
            message="fail", request=MagicMock()
        )
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await agent.run("Hi there")

        assert len(sleep_calls) == 10
        expected = [2, 4, 8, 16, 32, 60, 60, 60, 60, 60]
        assert sleep_calls == expected


# ── Node metadata ─────────────────────────────────────────────────────────

class TestAgentNodeMetadata:
    """BaseAgent.bind_node_metadata() correctly updates metadata and logger."""

    def test_bind_node_metadata(self):
        from models import NodeMetadata

        agent = _make_base_agent()
        meta = NodeMetadata(
            node_name="test-node",
            node_id="n-123",
            node_type="TestAgent",
            user_id="u-456",
        )
        result = agent.bind_node_metadata(meta)

        assert result is agent
        assert agent._node_metadata.node_name == "test-node"
        assert agent._node_metadata.node_id == "n-123"
        assert agent._node_metadata.user_id == "u-456"


# ── run_structured retry ──────────────────────────────────────────────────

class TestAgentRunStructured:
    """BaseAgent.run_structured() returns parsed grammar output with retry."""

    @pytest.mark.asyncio
    async def test_run_structured_returns_parsed_model(self):
        from pydantic import BaseModel as PydanticModel
        from langchain_core.messages import AIMessage

        class Greeting(PydanticModel):
            name: str

        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.return_value = AIMessage(content='{"name": "World"}')
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        result = await agent.run_structured("Say hi", grammar=Greeting)

        assert isinstance(result, Greeting)
        assert result.name == "World"

    @pytest.mark.asyncio
    async def test_run_structured_raises_on_error(self):
        from pydantic import BaseModel as PydanticModel

        class Greeting(PydanticModel):
            name: str

        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = RuntimeError("model crash")
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with pytest.raises(RuntimeError, match="Structured agent execution failed"):
            await agent.run_structured("Say hi", grammar=Greeting)

    @pytest.mark.asyncio
    async def test_run_structured_retries_on_503(self):
        from pydantic import BaseModel as PydanticModel
        from langchain_core.messages import AIMessage

        class Greeting(PydanticModel):
            name: str

        agent = _make_base_agent()
        call_count = [0]

        async def flaky_ainvoke(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 2:
                raise _make_api_status_error(503, "All inference slots are busy")
            return AIMessage(content='{"name": "World"}')

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky_ainvoke
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await agent.run_structured("Say hi", grammar=Greeting)

        assert isinstance(result, Greeting)
        assert result.name == "World"
        assert call_count[0] == 2
