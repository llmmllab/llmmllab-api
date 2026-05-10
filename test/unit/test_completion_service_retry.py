"""
Unit tests for CompletionService._build_and_run_with_retry().

Tests the retry-on-connection-error behavior added in PR #49:
- Connection errors trigger model map refresh + retry with backoff
- Non-connection errors are re-raised immediately
- Retry count is respected
- Successful completion stops retrying
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from httpx import RemoteProtocolError, ConnectError
from openai import APIConnectionError

from services.completion_service import CompletionService
from models.message import Message, MessageContent, MessageContentType, MessageRole
from graph.workflows.factory import WorkFlowType


def _make_messages():
    """Build a minimal message list for tests."""
    return [
        Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="hello")],
        )
    ]


class TestBuildAndRunWithRetry:
    """_build_and_run_with_retry retries on connection errors."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """No errors → yields events, no retry."""
        messages = _make_messages()
        call_count = [0]

        async def mock_build_and_run(*a, **kw):
            call_count[0] += 1
            yield MagicMock()
            return

        with patch.object(CompletionService, "_build_and_run", mock_build_and_run):
            events = [e async for e in CompletionService._build_and_run_with_retry(
                user_id="u1",
                messages=messages,
                model_name="model-a",
                workflow_type=WorkFlowType.IDE,
            )]

        assert call_count[0] == 1
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self):
        """ConnectError → retry with model map refresh."""
        messages = _make_messages()
        call_count = [0]

        async def mock_build_and_run(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectError("connection refused")
            yield MagicMock()
            return

        with patch.object(CompletionService, "_build_and_run", mock_build_and_run):
            with patch("services.completion_service.asyncio.sleep", new_callable=AsyncMock):
                with patch("services.runner_client.RunnerClient.refresh_model_map", new_callable=AsyncMock) as mock_refresh:
                    events = [e async for e in CompletionService._build_and_run_with_retry(
                        user_id="u1",
                        messages=messages,
                        model_name="model-a",
                        workflow_type=WorkFlowType.IDE,
                    )]

        assert call_count[0] == 2  # initial + 1 retry
        mock_refresh.assert_called_once()
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_retries_on_remote_protocol_error(self):
        """RemoteProtocolError → retry."""
        messages = _make_messages()
        call_count = [0]

        async def mock_build_and_run(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RemoteProtocolError("connection closed")
            yield MagicMock()
            return

        with patch.object(CompletionService, "_build_and_run", mock_build_and_run):
            with patch("services.completion_service.asyncio.sleep", new_callable=AsyncMock):
                with patch("services.runner_client.RunnerClient.refresh_model_map", new_callable=AsyncMock):
                    events = [e async for e in CompletionService._build_and_run_with_retry(
                        user_id="u1",
                        messages=messages,
                        model_name="model-a",
                        workflow_type=WorkFlowType.IDE,
                    )]

        assert call_count[0] == 2
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_retries_on_api_connection_error(self):
        """openai.APIConnectionError → retry."""
        messages = _make_messages()
        call_count = [0]

        async def mock_build_and_run(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise APIConnectionError(request=MagicMock())
            yield MagicMock()
            return

        with patch.object(CompletionService, "_build_and_run", mock_build_and_run):
            with patch("services.completion_service.asyncio.sleep", new_callable=AsyncMock):
                with patch("services.runner_client.RunnerClient.refresh_model_map", new_callable=AsyncMock):
                    events = [e async for e in CompletionService._build_and_run_with_retry(
                        user_id="u1",
                        messages=messages,
                        model_name="model-a",
                        workflow_type=WorkFlowType.IDE,
                    )]

        assert call_count[0] == 2
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_non_connection_error_not_retried(self):
        """ValueError → re-raised immediately, no retry."""
        messages = _make_messages()
        call_count = [0]

        async def mock_build_and_run(*a, **kw):
            call_count[0] += 1
            raise ValueError("bad model")
            yield  # make this an async generator

        with patch.object(CompletionService, "_build_and_run", mock_build_and_run):
            with pytest.raises(ValueError, match="bad model"):
                [e async for e in CompletionService._build_and_run_with_retry(
                    user_id="u1",
                    messages=messages,
                    model_name="model-a",
                    workflow_type=WorkFlowType.IDE,
                )]

        assert call_count[0] == 1  # no retry

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        """All attempts fail → original error re-raised."""
        messages = _make_messages()
        call_count = [0]

        async def mock_build_and_run(*a, **kw):
            call_count[0] += 1
            raise ConnectError("connection refused")
            yield  # make this an async generator

        with patch.object(CompletionService, "_build_and_run", mock_build_and_run):
            with patch("services.completion_service.asyncio.sleep", new_callable=AsyncMock):
                with patch("services.runner_client.RunnerClient.refresh_model_map", new_callable=AsyncMock):
                    with pytest.raises(ConnectError):
                        [e async for e in CompletionService._build_and_run_with_retry(
                            user_id="u1",
                            messages=messages,
                            model_name="model-a",
                            workflow_type=WorkFlowType.IDE,
                            max_retries=2,
                        )]

        # initial + 2 retries = 3 total
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_custom_max_retries(self):
        """max_retries=0 → no retries, fail immediately."""
        messages = _make_messages()
        call_count = [0]

        async def mock_build_and_run(*a, **kw):
            call_count[0] += 1
            raise ConnectError("connection refused")
            yield  # make this an async generator

        with patch.object(CompletionService, "_build_and_run", mock_build_and_run):
            with pytest.raises(ConnectError):
                [e async for e in CompletionService._build_and_run_with_retry(
                    user_id="u1",
                    messages=messages,
                    model_name="model-a",
                    workflow_type=WorkFlowType.IDE,
                    max_retries=0,
                )]

        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_backoff_sleep_called_with_linear_intervals(self):
        """Sleep is called with linear backoff: base*1, base*2, ..."""
        messages = _make_messages()
        call_count = [0]
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        async def mock_build_and_run(*a, **kw):
            call_count[0] += 1
            raise ConnectError("connection refused")
            yield  # make this an async generator

        with patch.object(CompletionService, "_build_and_run", mock_build_and_run):
            with patch("services.completion_service.asyncio.sleep", side_effect=mock_sleep):
                with patch("services.runner_client.RunnerClient.refresh_model_map", new_callable=AsyncMock):
                    with pytest.raises(ConnectError):
                        [e async for e in CompletionService._build_and_run_with_retry(
                            user_id="u1",
                            messages=messages,
                            model_name="model-a",
                            workflow_type=WorkFlowType.IDE,
                            max_retries=2,
                        )]

        # Linear backoff: base*(1), base*(2) — 2 sleeps for 2 retries
        assert len(sleep_calls) == 2
        assert sleep_calls[0] < sleep_calls[1]  # increasing
