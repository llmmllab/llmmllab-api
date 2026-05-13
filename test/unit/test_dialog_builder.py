"""
Unit tests for graph/workflows/dialog/builder.py.

Tests the DialogGraphBuilder model resolution logic, including:
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
    """Minimal mock Storage that satisfies DialogGraphBuilder.__init__."""
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


class TestDialogGraphBuilderModelResolution:
    """Tests for model resolution inside DialogGraphBuilder.build_workflow."""

    @pytest.mark.asyncio
    async def test_model_found_by_name(self, mock_storage, user_config):
        """When model_name matches a runner model, it is used directly."""
        primary = _make_model("llama-3-8b", "llama-3-8b")
        embedding = _make_model("embed-model", "embed-model", task=ModelTask.TEXTTOEMBEDDINGS)
        handle_p = _make_server_handle("http://runner:8000/v1/server/s1")
        handle_e = _make_server_handle("http://runner:8000/v1/server/s2")

        with patch("graph.workflows.dialog.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[primary, embedding])
            mock_rc.model_by_task = AsyncMock(side_effect=lambda t: embedding if t == ModelTask.TEXTTOEMBEDDINGS else None)
            mock_rc.acquire_server = AsyncMock(side_effect=[handle_p, handle_e])

            with patch("graph.workflows.dialog.builder.registry_manager") as mock_rm:
                mock_rm.get_user_registry = AsyncMock(return_value=MagicMock(
                    get_all_executable_tools=MagicMock(return_value=[])
                ))

                from graph.workflows.dialog.builder import DialogGraphBuilder

                builder = DialogGraphBuilder(mock_storage, user_config)
                workflow = await builder.build_workflow(
                    user_id="user-1", model_name="llama-3-8b"
                )

                mock_rc.list_models.assert_called_once()
                assert mock_rc.acquire_server.call_count == 2
                mock_rc.acquire_server.assert_any_call(
                    model_id="llama-3-8b", num_ctx=90000, task=ModelTask.TEXTTOTEXT
                )

    @pytest.mark.asyncio
    async def test_model_found_by_name_no_resolve_called(
        self, mock_storage, user_config
    ):
        """When model_name matches a runner model, resolve_model is not called."""
        primary = _make_model("llama-3-8b", "llama-3-8b")
        embedding = _make_model("embed-model", "embed-model", task=ModelTask.TEXTTOEMBEDDINGS)
        handle_p = _make_server_handle()
        handle_e = _make_server_handle()

        with patch("graph.workflows.dialog.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[primary, embedding])
            mock_rc.model_by_task = AsyncMock(side_effect=lambda t: embedding if t == ModelTask.TEXTTOEMBEDDINGS else None)
            mock_rc.acquire_server = AsyncMock(side_effect=[handle_p, handle_e])

            with patch("graph.workflows.dialog.builder.registry_manager") as mock_rm:
                mock_rm.get_user_registry = AsyncMock(return_value=MagicMock(
                    get_all_executable_tools=MagicMock(return_value=[])
                ))

                from graph.workflows.dialog.builder import DialogGraphBuilder

                builder = DialogGraphBuilder(mock_storage, user_config)
                with patch.object(builder, "resolve_model", new=AsyncMock()) as mock_resolve:
                    await builder.build_workflow(
                        user_id="user-1", model_name="llama-3-8b"
                    )
                    mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_not_found_falls_back_to_resolve_model(
        self, mock_storage, user_config
    ):
        """When model_name is not on any runner, resolve_model is called for fallback."""
        fallback_model = _make_model("default-model", "default-model")
        embedding = _make_model("embed-model", "embed-model", task=ModelTask.TEXTTOEMBEDDINGS)
        handle_p = _make_server_handle()
        handle_e = _make_server_handle()

        with patch("graph.workflows.dialog.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[fallback_model, embedding])
            mock_rc.model_by_task = AsyncMock(side_effect=lambda t: embedding if t == ModelTask.TEXTTOEMBEDDINGS else None)
            mock_rc.acquire_server = AsyncMock(side_effect=[handle_p, handle_e])

            with patch("graph.workflows.dialog.builder.registry_manager") as mock_rm:
                mock_rm.get_user_registry = AsyncMock(return_value=MagicMock(
                    get_all_executable_tools=MagicMock(return_value=[])
                ))

                from graph.workflows.dialog.builder import DialogGraphBuilder

                builder = DialogGraphBuilder(mock_storage, user_config)
                with patch.object(builder, "resolve_model", new=AsyncMock(return_value="default-model")) as mock_resolve:
                    await builder.build_workflow(
                        user_id="user-1", model_name="missing-model"
                    )

                    mock_resolve.assert_called_once_with("missing-model", "user-1")
                    assert mock_rc.list_models.call_count == 1
                    mock_rc.acquire_server.assert_any_call(
                        model_id="default-model", num_ctx=90000, task=ModelTask.TEXTTOTEXT
                    )

    @pytest.mark.asyncio
    async def test_model_not_found_and_fallback_falls_back_to_texttotext(
        self, mock_storage, user_config
    ):
        """When model and fallback are both missing, falls back to any available TextToText model."""
        other_model = _make_model("other-model", "other-model")
        embedding = _make_model("embed-model", "embed-model", task=ModelTask.TEXTTOEMBEDDINGS)
        handle_p = _make_server_handle()
        handle_e = _make_server_handle()

        with patch("graph.workflows.dialog.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[other_model, embedding])
            mock_rc.model_by_task = AsyncMock(
                side_effect=lambda t: other_model if t == ModelTask.TEXTTOTEXT else embedding
            )
            mock_rc.acquire_server = AsyncMock(side_effect=[handle_p, handle_e])

            with patch("graph.workflows.dialog.builder.registry_manager") as mock_rm:
                mock_rm.get_user_registry = AsyncMock(return_value=MagicMock(
                    get_all_executable_tools=MagicMock(return_value=[])
                ))

                from graph.workflows.dialog.builder import DialogGraphBuilder

                builder = DialogGraphBuilder(mock_storage, user_config)
                with patch.object(builder, "resolve_model", new=AsyncMock(return_value="also-missing")):
                    workflow = await builder.build_workflow(
                        user_id="user-1", model_name="missing-model"
                    )
                    # Should fall back to the available TextToText model
                    mock_rc.acquire_server.assert_any_call(
                        model_id="other-model", task=ModelTask.TEXTTOTEXT
                    )

    @pytest.mark.asyncio
    async def test_model_not_found_and_fallback_missing_and_no_texttotext_raises(
        self, mock_storage, user_config
    ):
        """When model, fallback, and no TextToText model exist, RuntimeError is raised."""
        other_model = _make_model("other-model", "other-model")
        embedding = _make_model("embed-model", "embed-model", task=ModelTask.TEXTTOEMBEDDINGS)

        with patch("graph.workflows.dialog.builder.runner_client") as mock_rc:
            mock_rc.list_models = AsyncMock(return_value=[other_model, embedding])
            mock_rc.model_by_task = AsyncMock(
                side_effect=lambda t: embedding if t == ModelTask.TEXTTOEMBEDDINGS else None
            )

            from graph.workflows.dialog.builder import DialogGraphBuilder

            builder = DialogGraphBuilder(mock_storage, user_config)
            with patch.object(builder, "resolve_model", new=AsyncMock(return_value="also-missing")):
                with pytest.raises(RuntimeError, match="Model 'also-missing' not found"):
                    await builder.build_workflow(
                        user_id="user-1", model_name="missing-model"
                    )

    @pytest.mark.asyncio
    async def test_no_model_name_uses_first_texttotext(self, mock_storage, user_config):
        """When model_name is None, the first TextToText model is used."""
        primary = _make_model("default-t2t", "default-t2t")
        embedding = _make_model("embed-model", "embed-model", task=ModelTask.TEXTTOEMBEDDINGS)
        handle_p = _make_server_handle()
        handle_e = _make_server_handle()

        with patch("graph.workflows.dialog.builder.runner_client") as mock_rc:
            mock_rc.model_by_task = AsyncMock(
                side_effect=lambda t: primary if t == ModelTask.TEXTTOTEXT else embedding
            )
            mock_rc.acquire_server = AsyncMock(side_effect=[handle_p, handle_e])

            with patch("graph.workflows.dialog.builder.registry_manager") as mock_rm:
                mock_rm.get_user_registry = AsyncMock(return_value=MagicMock(
                    get_all_executable_tools=MagicMock(return_value=[])
                ))

                from graph.workflows.dialog.builder import DialogGraphBuilder

                builder = DialogGraphBuilder(mock_storage, user_config)
                workflow = await builder.build_workflow(user_id="user-1")

                mock_rc.model_by_task.assert_any_call(ModelTask.TEXTTOTEXT)
                mock_rc.model_by_task.assert_any_call(ModelTask.TEXTTOEMBEDDINGS)
                mock_rc.acquire_server.assert_any_call(
                    model_id="default-t2t", num_ctx=90000, task=ModelTask.TEXTTOTEXT
                )

    @pytest.mark.asyncio
    async def test_no_texttotext_model_raises(self, mock_storage, user_config):
        """When no TextToText model is available and no model_name given, error."""
        with patch("graph.workflows.dialog.builder.runner_client") as mock_rc:
            mock_rc.model_by_task = AsyncMock(return_value=None)

            from graph.workflows.dialog.builder import DialogGraphBuilder

            builder = DialogGraphBuilder(mock_storage, user_config)

            with pytest.raises(RuntimeError, match="No TextToText model available"):
                await builder.build_workflow(user_id="user-1")

    @pytest.mark.asyncio
    async def test_no_embedding_model_raises(self, mock_storage, user_config):
        """When no TextToEmbeddings model is available, error is raised."""
        primary = _make_model("default-t2t", "default-t2t")

        with patch("graph.workflows.dialog.builder.runner_client") as mock_rc:
            mock_rc.model_by_task = AsyncMock(
                side_effect=lambda t: primary if t == ModelTask.TEXTTOTEXT else None
            )

            from graph.workflows.dialog.builder import DialogGraphBuilder

            builder = DialogGraphBuilder(mock_storage, user_config)

            with pytest.raises(RuntimeError, match="No TextToEmbeddings model available"):
                await builder.build_workflow(user_id="user-1")
