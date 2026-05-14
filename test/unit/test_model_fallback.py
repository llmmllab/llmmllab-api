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

    @patch("services.completion_service.model_service")
    def test_fallback_to_default_when_model_unavailable(self, mock_model_service):
        """When requested model is not available, fall back to user's default."""
        mock_model_service.resolve_default_model = AsyncMock(return_value="default-model")

        result = _resolve_model("unavailable-model", "user-123")
        assert result == "default-model"

    @patch("services.completion_service.model_service")
    def test_returns_original_when_model_available(self, mock_model_service):
        """When requested model is available, return it as-is."""
        mock_model_service.resolve_default_model = AsyncMock(return_value="unavailable-model")

        result = _resolve_model("unavailable-model", "user-123")
        assert result == "unavailable-model"

    @patch("services.completion_service.model_service")
    def test_returns_original_when_model_service_fails(self, mock_model_service):
        """When model_service fails, return the original model name."""
        mock_model_service.resolve_default_model = AsyncMock(side_effect=Exception("service down"))

        result = _resolve_model("requested-model", "user-123")
        assert result == "requested-model"


# ---------------------------------------------------------------------------
# _build_and_run fallback
# ---------------------------------------------------------------------------


class TestBuildAndRunFallback:
    """Test that _build_and_run falls back to default model on acquire_server failure."""

    @patch("services.completion_service.CompletionService.build_workflow")
    @patch("services.completion_service.CompletionService._run_workflow")
    @patch("services.completion_service._resolve_model")
    @patch("services.completion_service.model_service")
    @patch("services.completion_service.runner_client")
    async def test_fallback_on_acquire_server_failure(
        self,
        mock_runner_client,
        mock_model_service,
        mock_resolve_model,
        mock_run_workflow,
        mock_build_workflow,
    ):
        """When acquire_server raises RuntimeError, fall back to default model."""
        from graph.workflows.factory import WorkFlowType

        # Setup mocks
        mock_resolve_model.return_value = "requested-model"
        mock_model_service.resolve_default_model = AsyncMock(return_value="default-model")
        mock_run_workflow.return_value = AsyncMock()

        # Make build_workflow raise RuntimeError (simulating acquire_server failure)
        mock_build_workflow.side_effect = RuntimeError(
            "No healthy runner available for model requested-model. Last error: insufficient VRAM"
        )

        messages = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="Hello")],
            )
        ]

        # This should fall back to default-model instead of raising
        async for event in CompletionService._build_and_run(
            user_id="user-123",
            messages=messages,
            model_name="requested-model",
            workflow_type=WorkFlowType.IDE,
            conversation_id=1,
        ):
            pass  # Just consume events

        # Verify that resolve_default_model was called
        mock_model_service.resolve_default_model.assert_called_once_with(
            "requested-model", "user-123"
        )

    @patch("services.completion_service.CompletionService.build_workflow")
    @patch("services.completion_service.CompletionService._run_workflow")
    @patch("services.completion_service._resolve_model")
    @patch("services.completion_service.model_service")
    @patch("services.completion_service.runner_client")
    async def test_no_fallback_when_no_default_model(
        self,
        mock_runner_client,
        mock_model_service,
        mock_resolve_model,
        mock_run_workflow,
        mock_build_workflow,
    ):
        """When no default model is available, propagate the original error."""
        # Setup mocks
        mock_resolve_model.return_value = "requested-model"
        mock_model_service.resolve_default_model = AsyncMock(return_value="requested-model")
        mock_run_workflow.return_value = AsyncMock()

        # Make build_workflow raise RuntimeError
        mock_build_workflow.side_effect = RuntimeError(
            "No healthy runner available for model requested-model"
        )

        messages = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="Hello")],
            )
        ]

        # This should raise the original error (no fallback possible)
        with pytest.raises(RuntimeError, match="No healthy runner available"):
            async for event in CompletionService._build_and_run(
                user_id="user-123",
                messages=messages,
                model_name="requested-model",
                workflow_type=WorkFlowType.IDE,
                conversation_id=1,
            ):
                pass

    @patch("services.completion_service.CompletionService.build_workflow")
    @patch("services.completion_service.CompletionService._run_workflow")
    @patch("services.completion_service._resolve_model")
    @patch("services.completion_service.model_service")
    @patch("services.completion_service.runner_client")
    async def test_fallback_does_not_infinite_loop(
        self,
        mock_runner_client,
        mock_model_service,
        mock_resolve_model,
        mock_run_workflow,
        mock_build_workflow,
    ):
        """Verify that fallback doesn't cause infinite recursion."""
        # Setup mocks
        mock_resolve_model.return_value = "requested-model"
        mock_model_service.resolve_default_model = AsyncMock(return_value="default-model")
        mock_run_workflow.return_value = AsyncMock()

        # First call fails, second call (with fallback) succeeds
        call_count = {"count": 0}

        def side_effect(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise RuntimeError("No healthy runner available")
            return AsyncMock()

        mock_build_workflow.side_effect = side_effect

        messages = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="Hello")],
            )
        ]

        # Should succeed after one fallback attempt
        events = []
        async for event in CompletionService._build_and_run(
            user_id="user-123",
            messages=messages,
            model_name="requested-model",
            workflow_type=WorkFlowType.IDE,
            conversation_id=1,
        ):
            events.append(event)

        # Should have called build_workflow twice (original + fallback)
        assert call_count["count"] == 2
