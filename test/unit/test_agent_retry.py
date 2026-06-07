"""
Unit tests for agents/base.py — retry logic for transient API errors.

The retry block in BaseAgent.run() catches only:
- openai.APIStatusError with status 502/503/504 (server busy/gateway errors)

Connection-level errors (``APIConnectionError``) are NOT retried at the agent
level — they propagate to the outer ``stream_with_connection_retry`` layer which
refreshes the model map and acquires a fresh server handle before retrying (#267).

Non-transient errors (e.g. ValueError, 400 Bad Request, APIConnectionError)
propagate immediately.
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


# ── APIConnectionError propagates (not retried at agent level) ─────────────

class TestAgentRetryConnectionError:
    """APIConnectionError is NOT transient at the agent level.

    Connection errors propagate immediately so the outer retry layer
    (``stream_with_connection_retry``) can refresh the model map and re-acquire
    a fresh server handle before retrying (#267).
    """

    @pytest.mark.asyncio
    async def test_no_retry_on_api_connection_error(self):
        """A single connection error -> error response, no retries."""
        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = APIConnectionError(
            message="Connection refused", request=MagicMock()
        )
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", new_callable=AsyncMock) as msleep:
            response = await agent.run("Hi there")

        assert response.done is True
        text = response.message.content[0].text
        assert "Connection refused" in text
        # Only one attempt — no retries at the agent level for connection errors.
        assert mock_lc_agent.ainvoke.call_count == 1
        # No backoff sleep was scheduled.
        msleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_connection_error_propagates_after_one_attempt(self):
        """Even if ainvoke eventually succeeds, it never gets a second chance."""
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

        response = await agent.run("Hi there")

        assert response.done is True
        text = response.message.content[0].text
        assert "Connection refused" in text
        # Stopped after first failure — did not retry to reach success path.
        assert call_count[0] == 1


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
        mock_lc_agent.ainvoke.side_effect = _make_api_status_error(
            503, "All inference slots are busy"
        )
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await agent.run("Hi there")

        assert response.done is True
        text = response.message.content[0].text
        assert "All inference slots are busy" in text
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
        """Backoff uses transient 5xx errors (not connection errors)."""
        agent = _make_base_agent()
        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = _make_api_status_error(503, "busy")
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


class TestStaleServerDetection:
    """A 404 'Server <id> not found' must convert to StaleServerError so the
    CompletionService can transparently re-acquire a fresh server.

    Regression: the structured/grammar path (run_structured) wrapped every
    such 404 in a bare RuntimeError, leaving the stale handle unrecoverable
    and surfacing the failure as 'Chat Agent failed'.  Both run() and
    run_structured() must now raise StaleServerError on a stale-handle 404.
    """

    @staticmethod
    def _stale_404(server_id: str = "e728eac96b6f") -> APIStatusError:
        return _make_api_status_error(
            404, f"Error code: 404 - {{'detail': 'Server {server_id} not found'}}"
        )

    @pytest.mark.asyncio
    async def test_run_raises_stale_server_on_404(self):
        from graph.errors import StaleServerError

        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = self._stale_404("e728eac96b6f")
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with pytest.raises(StaleServerError) as exc_info:
            await agent.run("Hi there")

        assert exc_info.value.server_id == "e728eac96b6f"
        # 404 is not in the transient set — no retry, immediate raise.
        assert mock_lc_agent.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_run_structured_raises_stale_server_on_404(self):
        from pydantic import BaseModel as PydanticModel
        from graph.errors import StaleServerError

        class Greeting(PydanticModel):
            name: str

        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = self._stale_404("7b87809e26bf")
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with pytest.raises(StaleServerError) as exc_info:
            await agent.run_structured("Say hi", grammar=Greeting)

        assert exc_info.value.server_id == "7b87809e26bf"
        assert mock_lc_agent.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_run_structured_non_stale_error_still_runtimeerror(self):
        """A genuine non-stale failure must still raise the wrapped RuntimeError."""
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
    async def test_run_non_stale_404_without_server_word_not_stale(self):
        """A 404 that is not a 'server not found' (e.g. unknown llama.cpp
        sub-path) must NOT be misclassified as a stale handle."""
        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = _make_api_status_error(
            404, "Error code: 404 - {'detail': 'Not Found'}"
        )
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        # run() returns an error ChatResponse rather than raising StaleServerError.
        response = await agent.run("Hi there")
        assert response.done is True
        assert mock_lc_agent.ainvoke.call_count == 1


class TestToolIntentMarkerGating:
    """The [TOOL_INTENT:] instruction must be emitted whenever tools are
    available for the request — including the dynamic-tool case where
    tools arrive per-call and self.tools is empty (Claude Code / MCP).

    Regression: gating on self.tools alone silently disabled the marker
    for every dynamic-tool session, so the model never signalled tool
    intent and the nudge gate was inert.
    """

    def test_no_tools_no_marker(self):
        agent = _make_base_agent()  # self.tools == []
        sp, _ = agent._separate_system_prompt([])
        assert "[TOOL_INTENT:" not in sp

    def test_per_request_tools_inject_marker(self):
        agent = _make_base_agent()  # self.tools == []
        fake_tool = MagicMock()
        fake_tool.name = "read_file"
        sp, _ = agent._separate_system_prompt([], [fake_tool])
        assert "[TOOL_INTENT:" in sp

    def test_constructor_tools_still_inject_marker(self):
        model = MagicMock()
        fake_tool = MagicMock()
        fake_tool.name = "read_file"
        agent = BaseAgent(
            model=model, system_prompt="You are helpful.", tools=[fake_tool]
        )
        sp, _ = agent._separate_system_prompt([])
        assert "[TOOL_INTENT:" in sp


# ── Client-disconnect abort (retry-after-disconnect fix) ──────────────────

class TestAgentRetryDisconnectAbort:
    """The retry loop must abort promptly when the client has disconnected,
    instead of retrying a dead turn for minutes (session acd66c8a…).

    A ``disconnected`` predicate is threaded into run()/run_structured();
    when it returns True the loop raises asyncio.CancelledError (which the
    streaming handlers already treat as 'client disconnected').  Default
    None preserves the original behaviour.
    """

    @pytest.mark.asyncio
    async def test_disconnect_aborts_before_retry(self):
        """Transient error + client gone -> CancelledError, no further retry."""
        import asyncio

        agent = _make_base_agent()
        call_count = [0]

        async def always_503(*a, **kw):
            call_count[0] += 1
            raise _make_api_status_error(503, "All inference slots are busy")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = always_503
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        # Client is gone from the start.
        async def gone() -> bool:
            return True

        with patch("asyncio.sleep", new_callable=AsyncMock) as msleep:
            with pytest.raises(asyncio.CancelledError):
                await agent.run("Hi there", disconnected=gone)

        # Aborted before the very first dispatch (predicate checked at top).
        assert call_count[0] == 0
        # Never slept out a backoff.
        msleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_mid_flight_aborts_in_backoff(self):
        """Connected for the first attempt, disconnects before the backoff:
        the loop must abort rather than sleep + retry."""
        import asyncio

        agent = _make_base_agent()
        call_count = [0]
        connected = [True]

        async def flaky(*a, **kw):
            call_count[0] += 1
            connected[0] = False  # client leaves after the first failure
            raise _make_api_status_error(503, "busy")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        async def gone() -> bool:
            return not connected[0]

        with patch("asyncio.sleep", new_callable=AsyncMock) as msleep:
            with pytest.raises(asyncio.CancelledError):
                await agent.run("Hi there", disconnected=gone)

        # Exactly one dispatch happened, then the disconnect check aborted
        # before any backoff sleep.
        assert call_count[0] == 1
        msleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_predicate_preserves_retry(self):
        """disconnected=None (default) -> original retry behaviour."""
        agent = _make_base_agent()
        call_count = [0]

        async def flaky(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise _make_api_status_error(503, "busy")
            return MagicMock(content="OK", role="ai")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await agent.run("Hi there")  # no predicate

        assert response.done is True
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_connected_predicate_does_not_abort(self):
        """A predicate that always reports 'still connected' must not change
        the successful-retry path."""
        agent = _make_base_agent()
        call_count = [0]

        async def flaky(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 2:
                raise _make_api_status_error(503, "busy")
            return MagicMock(content="OK", role="ai")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        async def still_here() -> bool:
            return False

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await agent.run("Hi there", disconnected=still_here)

        assert response.done is True
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_run_structured_disconnect_aborts(self):
        import asyncio
        from pydantic import BaseModel as PydanticModel

        class Greeting(PydanticModel):
            name: str

        agent = _make_base_agent()
        call_count = [0]

        async def always_503(*a, **kw):
            call_count[0] += 1
            raise _make_api_status_error(503, "busy")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = always_503
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        async def gone() -> bool:
            return True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(asyncio.CancelledError):
                await agent.run_structured(
                    "Say hi", grammar=Greeting, disconnected=gone
                )
        assert call_count[0] == 0


class TestAgentMaxAttemptsConfigurable:
    """AGENT_MAX_RETRY_ATTEMPTS tunes the retry budget."""

    @pytest.mark.asyncio
    async def test_lower_cap_reduces_attempts(self):
        import config

        agent = _make_base_agent()
        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke.side_effect = _make_api_status_error(503, "busy")
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        original = config.AGENT_MAX_RETRY_ATTEMPTS
        config.AGENT_MAX_RETRY_ATTEMPTS = 3
        try:
            with patch("asyncio.sleep", new_callable=AsyncMock):
                response = await agent.run("Hi there")
        finally:
            config.AGENT_MAX_RETRY_ATTEMPTS = original

        assert response.done is True
        # Use 5xx status (not APIConnectionError) as transient driver.
        assert mock_lc_agent.ainvoke.call_count == 3
