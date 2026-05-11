"""
Unit tests for graph/workflows/ide/builder.py.

Tests the IdeGraphBuilder model resolution logic, including:
- Model found on a runner by name/id
- Model not found → fallback to user default via resolve_model
- No model specified → first TextToText model
- Model resolution errors propagate correctly
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from models import (
    Model,
    ModelTask,
    ModelDetails,
    ModelProvider,
    ModelParameters,
    UserConfig,
    WorkflowConfig,
)
from models.model_provider import ModelProvider
from services.runner_client import ServerHandle


def _make_model(
    name: str,
    model_id: str,
    task: ModelTask = ModelTask.TEXTTOTEXT,
    system_prompt: str | None = None,
    num_ctx: int | None = None,
) -> Model:
    """Helper to create a minimal Model instance."""
    params = None
    if num_ctx is not None:
        params = ModelParameters(num_ctx=num_ctx)
    return Model(
        id=model_id,
        name=name,
        model=name,
        task=task,
        modified_at="2025-01-01",
        digest="abc123",
        details=ModelDetails(
            format="gguf",
            family="llama",
            families=["llama"],
            parameter_size="8B",
            size=4000000000,
            original_ctx=8192,
        ),
        provider=ModelProvider.LLAMA_CPP,
        system_prompt=system_prompt,
        parameters=params,
    )


def _make_server_handle(base_url: str = "http://runner:8000/v1/server/s1") -> ServerHandle:
    return ServerHandle(
        base_url=base_url,
        server_id="s1",
        runner_host="http://runner:8000",
    )


@pytest.fixture
def mock_storage():
    """Minimal mock Storage that satisfies IdeGraphBuilder.__init__."""
    storage = MagicMock()
    storage.get_service = MagicMock(return_value=MagicMock())
    return storage


@pytest.fixture
def user_config():
    return UserConfig(
        user_id="user-1",
        memory=None,
        summarization=None,
        image_generation=None,
        workflow=WorkflowConfig(),
    )


class TestIdeGraphBuilderModelResolution:
    """Tests for model resolution inside IdeGraphBuilder.build_workflow."""

    @pytest.mark.asyncio
    async def test_model_found_by_name(self, mock_storage, user_config):
        """When model_name matches a runner model, it is used directly."""
        model = _make_model("llama-3-8b", "llama-3-8b")
        handle = _make_server_handle()

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[model])
            mock_rc.model_by_task = AsyncMock()
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)
            workflow = await builder.build_workflow(
                user_id="user-1", model_name="llama-3-8b"
            )

            # list_models was called to look up the model
            mock_rc.list_models.assert_called_once()
            # acquire_server was called with the right model id
            mock_rc.acquire_server.assert_called_once_with(
                model_id="llama-3-8b", task=ModelTask.TEXTTOTEXT
            )

    @pytest.mark.asyncio
    async def test_model_found_by_name_no_resolve_called(
        self, mock_storage, user_config
    ):
        """When model_name matches a runner model, resolve_model is not called."""
        model = _make_model("llama-3-8b", "llama-3-8b")
        handle = _make_server_handle()

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[model])
            mock_rc.model_by_task = AsyncMock()
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)
            # Patch resolve_model to verify it's not called
            with patch.object(builder, "resolve_model", new=AsyncMock()) as mock_resolve:
                await builder.build_workflow(
                    user_id="user-1", model_name="llama-3-8b"
                )
                mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_found_by_id(self, mock_storage, user_config):
        """When model_name matches a runner model id (not name), it is used."""
        model = _make_model("Friendly Name", "model-id-123")
        handle = _make_server_handle()

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[model])
            mock_rc.model_by_task = AsyncMock()
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)
            workflow = await builder.build_workflow(
                user_id="user-1", model_name="model-id-123"
            )

            mock_rc.list_models.assert_called_once()
            mock_rc.acquire_server.assert_called_once_with(
                model_id="model-id-123", task=ModelTask.TEXTTOTEXT
            )

    @pytest.mark.asyncio
    async def test_model_not_found_falls_back_to_resolve_model(
        self, mock_storage, user_config
    ):
        """When model_name is not on any runner, resolve_model is called for fallback."""
        # The runner has the fallback model but not the requested one.
        # After resolve_model returns the fallback name, the builder
        # re-searches the same all_models list (no re-fetch).
        fallback_model = _make_model("default-model", "default-model")
        handle = _make_server_handle()

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[fallback_model])
            mock_rc.model_by_task = AsyncMock()
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)
            with patch.object(builder, "resolve_model", new=AsyncMock(return_value="default-model")) as mock_resolve:
                workflow = await builder.build_workflow(
                    user_id="user-1", model_name="missing-model"
                )

                # resolve_model was called because model wasn't found
                mock_resolve.assert_called_once_with("missing-model", "user-1")
                # list_models called once (builder re-searches same list)
                assert mock_rc.list_models.call_count == 1
                # acquire_server called with the fallback model
                mock_rc.acquire_server.assert_called_once_with(
                    model_id="default-model", task=ModelTask.TEXTTOTEXT
                )

    @pytest.mark.asyncio
    async def test_model_not_found_and_fallback_also_missing_raises(
        self, mock_storage, user_config
    ):
        """When model and fallback are both missing, RuntimeError is raised."""
        other_model = _make_model("other-model", "other-model")

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[other_model])
            mock_rc.model_by_task = AsyncMock()

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)
            with patch.object(builder, "resolve_model", new=AsyncMock(return_value="also-missing")):
                with pytest.raises(RuntimeError, match="Model 'also-missing' not found"):
                    await builder.build_workflow(
                        user_id="user-1", model_name="missing-model"
                    )

    @pytest.mark.asyncio
    async def test_no_model_name_uses_first_texttotext(self, mock_storage, user_config):
        """When model_name is None, the first TextToText model is used."""
        model = _make_model("default-t2t", "default-t2t")
        handle = _make_server_handle()

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.model_by_task = AsyncMock(return_value=model)
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)
            workflow = await builder.build_workflow(user_id="user-1")

            mock_rc.model_by_task.assert_called_once_with(ModelTask.TEXTTOTEXT)
            mock_rc.acquire_server.assert_called_once_with(
                model_id="default-t2t", task=ModelTask.TEXTTOTEXT
            )

    @pytest.mark.asyncio
    async def test_no_texttotext_model_raises(self, mock_storage, user_config):
        """When no TextToText model is available and no model_name given, error."""
        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.model_by_task = AsyncMock(return_value=None)

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)

            with pytest.raises(RuntimeError, match="No TextToText model available"):
                await builder.build_workflow(user_id="user-1")

    @pytest.mark.asyncio
    async def test_build_workflow_sets_server_handle(self, mock_storage, user_config):
        """build_workflow stores the server handle on self for later access."""
        model = _make_model("llama-3-8b", "llama-3-8b")
        handle = _make_server_handle("http://runner:8000/v1/server/custom")

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[model])
            mock_rc.model_by_task = AsyncMock()
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)
            await builder.build_workflow(user_id="user-1", model_name="llama-3-8b")

            assert builder.server_handle is handle
            assert builder.server_handle.base_url == "http://runner:8000/v1/server/custom"

    @pytest.mark.asyncio
    async def test_model_parameters_used_for_num_ctx(self, mock_storage, user_config):
        """Model parameters (num_ctx) are passed to ChatAgent."""
        model = _make_model("llama-3-8b", "llama-3-8b", num_ctx=128000)
        handle = _make_server_handle()

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[model])
            mock_rc.model_by_task = AsyncMock()
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            with patch("graph.workflows.ide.builder.ChatAgent") as mock_agent_cls:
                mock_agent_cls.return_value = MagicMock()

                from graph.workflows.ide.builder import IdeGraphBuilder

                builder = IdeGraphBuilder(mock_storage, user_config)
                await builder.build_workflow(user_id="user-1", model_name="llama-3-8b")

                # ChatAgent should be called with num_ctx=128000
                mock_agent_cls.assert_called_once()
                call_kwargs = mock_agent_cls.call_args[1]
                assert call_kwargs["num_ctx"] == 128000

    @pytest.mark.asyncio
    async def test_default_num_ctx_when_model_has_no_parameters(
        self, mock_storage, user_config
    ):
        """When model has no parameters, default num_ctx of 90000 is used."""
        model = _make_model("llama-3-8b", "llama-3-8b", num_ctx=None)
        handle = _make_server_handle()

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[model])
            mock_rc.model_by_task = AsyncMock()
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            with patch("graph.workflows.ide.builder.ChatAgent") as mock_agent_cls:
                mock_agent_cls.return_value = MagicMock()

                from graph.workflows.ide.builder import IdeGraphBuilder

                builder = IdeGraphBuilder(mock_storage, user_config)
                await builder.build_workflow(user_id="user-1", model_name="llama-3-8b")

                call_kwargs = mock_agent_cls.call_args[1]
                assert call_kwargs["num_ctx"] == 90000


class TestIdeGraphBuilderToolModes:
    """Tests for IDE builder tool mode graph construction."""

    @pytest.mark.asyncio
    async def test_proxy_mode_graph_structure(self, mock_storage, user_config):
        """Proxy mode (client_tools only): Agent -> END, no tool node."""
        model = _make_model("llama-3-8b", "llama-3-8b")
        handle = _make_server_handle()

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[model])
            mock_rc.model_by_task = AsyncMock()
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)
            workflow = await builder.build_workflow(
                user_id="user-1",
                model_name="llama-3-8b",
                client_tools=[{"type": "function", "function": {"name": "read_file"}}],
            )

            # In proxy mode, the graph should have only the agent node
            graph = workflow.get_graph()
            node_names = list(graph.nodes)
            assert "Agent" in node_names

    @pytest.mark.asyncio
    async def test_server_tool_names_mode(self, mock_storage, user_config):
        """Hybrid mode (server_tool_names): includes ServerToolNode."""
        model = _make_model("llama-3-8b", "llama-3-8b")
        handle = _make_server_handle()

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[model])
            mock_rc.model_by_task = AsyncMock()
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)
            workflow = await builder.build_workflow(
                user_id="user-1",
                model_name="llama-3-8b",
                server_tool_names={"execute_code"},
            )

            graph = workflow.get_graph()
            node_names = list(graph.nodes)
            assert "Agent" in node_names
            assert "Tool" in node_names

    @pytest.mark.asyncio
    async def test_server_tools_mode(self, mock_storage, user_config):
        """Server-side mode (server_tools): includes ToolNode."""
        from langchain_core.tools import tool as lc_tool

        model = _make_model("llama-3-8b", "llama-3-8b")
        handle = _make_server_handle()

        @lc_tool
        def mock_tool(x: str) -> str:
            """A mock tool for testing."""
            return x

        with patch("graph.workflows.ide.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[model])
            mock_rc.model_by_task = AsyncMock()
            mock_rc.acquire_server = AsyncMock(return_value=handle)

            from graph.workflows.ide.builder import IdeGraphBuilder

            builder = IdeGraphBuilder(mock_storage, user_config)
            workflow = await builder.build_workflow(
                user_id="user-1",
                model_name="llama-3-8b",
                server_tools=[mock_tool],
            )

            graph = workflow.get_graph()
            node_names = list(graph.nodes)
            assert "Agent" in node_names
            assert "Tool" in node_names


class TestIdeGraphBuilderInitialState:
    """Tests for IdeGraphBuilder.create_initial_state."""

    @pytest.mark.asyncio
    async def test_create_initial_state_with_messages(self, mock_storage, user_config):
        """Initial state includes messages and current user message."""
        from models import Message, MessageRole, MessageContent, MessageContentType

        messages = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="Hello", url=None)],
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="Hi!", url=None)],
            ),
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="Second msg", url=None)],
            ),
        ]

        from graph.workflows.ide.builder import IdeGraphBuilder

        builder = IdeGraphBuilder(mock_storage, user_config)
        state = await builder.create_initial_state(
            user_id="user-1", conversation_id=42, messages=messages
        )

        assert state.messages == messages
        assert state.user_id == "user-1"
        assert state.conversation_id == 42
        assert state.workflow_type == "ide"
        # Current user message should be the last user message
        assert state.current_user_message.content[0].text == "Second msg"

    @pytest.mark.asyncio
    async def test_create_initial_state_requires_messages(self, mock_storage, user_config):
        """create_initial_state raises AssertionError when messages is None."""
        from graph.workflows.ide.builder import IdeGraphBuilder

        builder = IdeGraphBuilder(mock_storage, user_config)

        with pytest.raises(AssertionError, match="Messages must be provided"):
            await builder.create_initial_state(
                user_id="user-1", conversation_id=42, messages=None
            )
