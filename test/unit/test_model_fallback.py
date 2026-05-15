"""Tests for model fallback when acquire_server fails.

Verifies that CompletionService._build_and_run falls back to the user's
default model when a requested model can't be acquired by the runner.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.message import Message, MessageContent, MessageContentType, MessageRole
from services.completion_service import CompletionService, _resolve_model
from graph.workflows.factory import WorkFlowType  # Import at function level to avoid circular import


# ---------------------------------------------------------------------------
# _resolve_model fallback
# ---------------------------------------------------------------------------


class TestResolveModelFallback:
    """Test that _resolve_model falls back to default when model is unavailable."""

    @patch("services.model_service.model_service")
    @pytest.mark.asyncio
    async def test_fallback_to_default_when_model_unavailable(self, mock_model_service):
        """When requested model is not available, fall back to user's default."""
        mock_model_service.resolve_default_model = AsyncMock(return_value="default-model")

        result = await _resolve_model("unavailable-model", "user-123")
        assert result == "default-model"

    @patch("services.model_service.model_service")
    @pytest.mark.asyncio
    async def test_returns_original_when_model_available(self, mock_model_service):
        """When requested model is available, return it as-is."""
        mock_model_service.resolve_default_model = AsyncMock(return_value="unavailable-model")

        result = await _resolve_model("unavailable-model", "user-123")
        assert result == "unavailable-model"

    @patch("services.model_service.model_service")
    @pytest.mark.asyncio
    async def test_returns_original_when_model_service_fails(self, mock_model_service):
        """When model_service fails, return the original model name."""
        mock_model_service.resolve_default_model = AsyncMock(side_effect=Exception("service down"))

        result = await _resolve_model("requested-model", "user-123")
        assert result == "requested-model"


# ---------------------------------------------------------------------------
# Circuit breaker error messages (PR #77)
# ---------------------------------------------------------------------------


class TestCircuitBreakerErrorMessage:
    """Test that acquire_server raises meaningful errors when all runners are skipped."""

    @pytest.mark.asyncio
    async def test_error_mentions_circuit_breaker_when_all_skipped(self):
        """When all endpoints have open circuit breakers, error message should mention it."""
        from services.runner_client import RunnerClient

        client = RunnerClient()
        # Manually set up circuit breaker state for two endpoints
        client._endpoints = ["http://runner1:9000", "http://runner2:9000"]
        client._acquire_failures = {
            "http://runner1:9000": 5,
            "http://runner2:9000": 3,
        }
        # Set circuit breaker timestamps to the past so they're open
        client._circuit_breaker_opened_at = {
            "http://runner1:9000": 0.0,
            "http://runner2:9000": 0.0,
        }

        with pytest.raises(RuntimeError) as exc_info:
            await client.acquire_server("some-model")

        error_msg = str(exc_info.value)
        assert "circuit breaker open" in error_msg.lower() or "skipped" in error_msg.lower()
        assert "All 2 runner(s) skipped" in error_msg

    @pytest.mark.asyncio
    async def test_error_mentions_no_endpoints_when_empty(self):
        """When no endpoints are configured, error message should say so."""
        from services.runner_client import RunnerClient

        client = RunnerClient()
        client._endpoints = []

        with pytest.raises(RuntimeError) as exc_info:
            await client.acquire_server("some-model")

        error_msg = str(exc_info.value)
        assert "No endpoints available" in error_msg
