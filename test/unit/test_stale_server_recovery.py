"""
Unit tests for stale-server recovery in CompletionService._build_and_run.

Tests verify that when a StaleServerError is raised, the service:
  1. Releases the stale server handle via runner_client
  2. Refreshes the model map
  3. Retries the workflow with a fresh server
  4. Propagates the error when retries are exhausted
  5. Respects the STALE_SERVER_RETRIES config value
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graph.errors import StaleServerError


def _get_workflow_type():
    """Lazily import WorkFlowType to avoid circular imports."""
    from graph.workflows.factory import WorkFlowType
    return WorkFlowType.DIALOG


def _get_completion_service():
    """Lazily import CompletionService to avoid circular imports."""
    from services.completion_service import CompletionService
    return CompletionService


class TestStaleServerError:
    """Tests for the StaleServerError class itself."""

    def test_construction_with_server_id(self):
        err = StaleServerError(server_id="srv-abc123")
        assert err.server_id == "srv-abc123"
        assert err.original_error is None
        assert "srv-abc123" in str(err)
        assert err.details["server_id"] == "srv-abc123"

    def test_construction_with_original_error(self):
        orig = ConnectionError("404 Not Found")
        err = StaleServerError(server_id="srv-abc123", original_error=orig)
        assert err.server_id == "srv-abc123"
        assert err.original_error is orig
        assert "404 Not Found" in str(err)

    def test_is_composer_error(self):
        from graph.errors import ComposerError
        err = StaleServerError(server_id="srv-abc123")
        assert isinstance(err, ComposerError)


class TestStaleServerRecovery:
    """Tests for the retry logic in CompletionService._build_and_run."""

    @pytest.fixture
    def mock_messages(self):
        from models.message import Message
        from models.message_content import MessageContent
        from models.message_content_type import MessageContentType
        return [
            Message(
                role="user",
                content=[MessageContent(type=MessageContentType.TEXT, text="Hello")],
            )
        ]

    @pytest.fixture
    def mock_chat_response(self):
        from models.chat_response import ChatResponse
        return ChatResponse(
            id="resp-1",
            model="test-model",
            content="Hello there!",
            stop_reason="end_turn",
        )

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self, mock_messages, mock_chat_response):
        """When the workflow succeeds on first try, no retry logic runs."""
        CompletionService = _get_completion_service()
        workflow_type = _get_workflow_type()

        mock_workflow = MagicMock()
        mock_builder = MagicMock()
        mock_builder.server_handle = None

        async def mock_build_workflow(*a, **kw):
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def mock_create_initial_state(*a, **kw):
            return {}

        async def mock_run_workflow(state, wf, wt, disconnected=None):
            yield mock_chat_response

        with patch.object(
            CompletionService, "build_workflow", mock_build_workflow
        ), patch(
            "services.completion_service.create_initial_state",
            mock_create_initial_state,
        ), patch.object(
            CompletionService, "_run_workflow", mock_run_workflow
        ):
            events = []
            async for ev in CompletionService._build_and_run(
                user_id="u1",
                messages=mock_messages,
                model_name="test-model",
                workflow_type=workflow_type,
                conversation_id=1,
            ):
                events.append(ev)

            assert len(events) == 1
            assert events[0] is mock_chat_response

    @pytest.mark.asyncio
    async def test_retry_on_stale_server(self, mock_messages, mock_chat_response):
        """When StaleServerError is raised, the service retries once."""
        CompletionService = _get_completion_service()
        workflow_type = _get_workflow_type()

        mock_workflow = MagicMock()
        mock_handle = MagicMock()
        mock_handle.server_id = "srv-stale"
        mock_builder = MagicMock()
        mock_builder.server_handle = mock_handle

        call_count = 0

        async def mock_build_workflow(*a, **kw):
            nonlocal call_count
            call_count += 1
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def mock_create_initial_state(*a, **kw):
            return {}

        async def mock_run_workflow(state, wf, wt, disconnected=None):
            nonlocal call_count
            if call_count == 1:
                raise StaleServerError(server_id="srv-stale")
            yield mock_chat_response

        mock_runner = AsyncMock()

        with patch.object(
            CompletionService, "build_workflow", mock_build_workflow
        ), patch(
            "services.completion_service.create_initial_state",
            mock_create_initial_state,
        ), patch.object(
            CompletionService, "_run_workflow", mock_run_workflow
        ), patch(
            "services.runner_client.runner_client", mock_runner
        ):
            events = []
            async for ev in CompletionService._build_and_run(
                user_id="u1",
                messages=mock_messages,
                model_name="test-model",
                workflow_type=workflow_type,
                conversation_id=1,
            ):
                events.append(ev)

            assert len(events) == 1
            assert events[0] is mock_chat_response
            assert call_count == 2  # original + 1 retry

            # Verify server release was called
            mock_runner.release_server.assert_called_once_with(mock_handle)
            # Verify model map refresh was called
            mock_runner.refresh_model_map.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_exhausted_propagates_error(self, mock_messages):
        """When retries are exhausted (0 retries), the error propagates immediately."""
        CompletionService = _get_completion_service()
        workflow_type = _get_workflow_type()

        mock_workflow = MagicMock()
        mock_handle = MagicMock()
        mock_handle.server_id = "srv-stale"
        mock_builder = MagicMock()
        mock_builder.server_handle = mock_handle

        async def mock_build_workflow(*a, **kw):
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def mock_create_initial_state(*a, **kw):
            return {}

        async def mock_run_workflow(state, wf, wt, disconnected=None):
            raise StaleServerError(server_id="srv-stale")
            yield  # make it an async generator

        mock_runner = AsyncMock()

        with patch.object(
            CompletionService, "build_workflow", mock_build_workflow
        ), patch(
            "services.completion_service.create_initial_state",
            mock_create_initial_state,
        ), patch.object(
            CompletionService, "_run_workflow", mock_run_workflow
        ), patch(
            "services.runner_client.runner_client", mock_runner
        ), patch(
            "services.completion_service.STALE_SERVER_RETRIES", 0
        ):
            with pytest.raises(StaleServerError) as exc_info:
                async for _ in CompletionService._build_and_run(
                    user_id="u1",
                    messages=mock_messages,
                    model_name="test-model",
                    workflow_type=workflow_type,
                    conversation_id=1,
                ):
                    pass

            assert exc_info.value.server_id == "srv-stale"

    @pytest.mark.asyncio
    async def test_retry_exhausted_after_one_retry(self, mock_messages):
        """With STALE_SERVER_RETRIES=1 (default), error propagates after 1 retry."""
        CompletionService = _get_completion_service()
        workflow_type = _get_workflow_type()

        mock_workflow = MagicMock()
        mock_handle = MagicMock()
        mock_handle.server_id = "srv-stale"
        mock_builder = MagicMock()
        mock_builder.server_handle = mock_handle

        async def mock_build_workflow(*a, **kw):
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def mock_create_initial_state(*a, **kw):
            return {}

        async def mock_run_workflow(state, wf, wt, disconnected=None):
            raise StaleServerError(server_id="srv-stale")
            yield  # make it an async generator

        mock_runner = AsyncMock()

        with patch.object(
            CompletionService, "build_workflow", mock_build_workflow
        ), patch(
            "services.completion_service.create_initial_state",
            mock_create_initial_state,
        ), patch.object(
            CompletionService, "_run_workflow", mock_run_workflow
        ), patch(
            "services.runner_client.runner_client", mock_runner
        ), patch(
            "services.completion_service.STALE_SERVER_RETRIES", 1
        ):
            with pytest.raises(StaleServerError) as exc_info:
                async for _ in CompletionService._build_and_run(
                    user_id="u1",
                    messages=mock_messages,
                    model_name="test-model",
                    workflow_type=workflow_type,
                    conversation_id=1,
                ):
                    pass

            assert exc_info.value.server_id == "srv-stale"
            # Should have retried once (2 total attempts)
            assert mock_runner.release_server.call_count == 1
            assert mock_runner.refresh_model_map.call_count == 1

    @pytest.mark.asyncio
    async def test_release_failure_does_not_stop_retry(self, mock_messages, mock_chat_response):
        """If server release fails, the retry still proceeds."""
        CompletionService = _get_completion_service()
        workflow_type = _get_workflow_type()

        mock_workflow = MagicMock()
        mock_handle = MagicMock()
        mock_handle.server_id = "srv-stale"
        mock_builder = MagicMock()
        mock_builder.server_handle = mock_handle

        call_count = 0

        async def mock_build_workflow(*a, **kw):
            nonlocal call_count
            call_count += 1
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def mock_create_initial_state(*a, **kw):
            return {}

        async def mock_run_workflow(state, wf, wt, disconnected=None):
            nonlocal call_count
            if call_count == 1:
                raise StaleServerError(server_id="srv-stale")
            yield mock_chat_response

        mock_runner = AsyncMock()
        mock_runner.release_server.side_effect = Exception("release failed")

        with patch.object(
            CompletionService, "build_workflow", mock_build_workflow
        ), patch(
            "services.completion_service.create_initial_state",
            mock_create_initial_state,
        ), patch.object(
            CompletionService, "_run_workflow", mock_run_workflow
        ), patch(
            "services.runner_client.runner_client", mock_runner
        ):
            events = []
            async for ev in CompletionService._build_and_run(
                user_id="u1",
                messages=mock_messages,
                model_name="test-model",
                workflow_type=workflow_type,
                conversation_id=1,
            ):
                events.append(ev)

            assert len(events) == 1
            assert events[0] is mock_chat_response

    @pytest.mark.asyncio
    async def test_no_server_handle_skips_release(self, mock_messages, mock_chat_response):
        """When builder has no server_handle, release is skipped."""
        CompletionService = _get_completion_service()
        workflow_type = _get_workflow_type()

        mock_workflow = MagicMock()
        mock_builder = MagicMock()
        mock_builder.server_handle = None

        call_count = 0

        async def mock_build_workflow(*a, **kw):
            nonlocal call_count
            call_count += 1
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def mock_create_initial_state(*a, **kw):
            return {}

        async def mock_run_workflow(state, wf, wt, disconnected=None):
            nonlocal call_count
            if call_count == 1:
                raise StaleServerError(server_id="srv-stale")
            yield mock_chat_response

        mock_runner = AsyncMock()

        with patch.object(
            CompletionService, "build_workflow", mock_build_workflow
        ), patch(
            "services.completion_service.create_initial_state",
            mock_create_initial_state,
        ), patch.object(
            CompletionService, "_run_workflow", mock_run_workflow
        ), patch(
            "services.runner_client.runner_client", mock_runner
        ):
            events = []
            async for ev in CompletionService._build_and_run(
                user_id="u1",
                messages=mock_messages,
                model_name="test-model",
                workflow_type=workflow_type,
                conversation_id=1,
            ):
                events.append(ev)

            assert len(events) == 1
            # release_server should NOT have been called
            mock_runner.release_server.assert_not_called()
            # but refresh_model_map should still be called
            mock_runner.refresh_model_map.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_stale_error_propagates(self, mock_messages):
        """Non-StaleServerError exceptions propagate without retry."""
        CompletionService = _get_completion_service()
        workflow_type = _get_workflow_type()

        mock_workflow = MagicMock()
        mock_builder = MagicMock()
        mock_builder.server_handle = None

        async def mock_build_workflow(*a, **kw):
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def mock_create_initial_state(*a, **kw):
            return {}

        async def mock_run_workflow(state, wf, wt, disconnected=None):
            raise ConnectionError("network down")
            yield  # make it an async generator

        with patch.object(
            CompletionService, "build_workflow", mock_build_workflow
        ), patch(
            "services.completion_service.create_initial_state",
            mock_create_initial_state,
        ), patch.object(
            CompletionService, "_run_workflow", mock_run_workflow
        ):
            with pytest.raises(ConnectionError, match="network down"):
                async for _ in CompletionService._build_and_run(
                    user_id="u1",
                    messages=mock_messages,
                    model_name="test-model",
                    workflow_type=workflow_type,
                    conversation_id=1,
                ):
                    pass


class TestStaleServerRetriesConfig:
    """Tests for the STALE_SERVER_RETRIES configuration."""

    def test_default_value(self):
        from config import STALE_SERVER_RETRIES
        assert STALE_SERVER_RETRIES == 2

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("STALE_SERVER_RETRIES", "3")
        import importlib
        import config
        importlib.reload(config)
        assert config.STALE_SERVER_RETRIES == 3

    def test_env_zero(self, monkeypatch):
        monkeypatch.setenv("STALE_SERVER_RETRIES", "0")
        import importlib
        import config
        importlib.reload(config)
        assert config.STALE_SERVER_RETRIES == 0
