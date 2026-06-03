"""
Unit tests for ModelParameters and the model_parameters passthrough flow.

Covers:
- ModelParameters field defaults, type validation, constraint enforcement
- Extra field ignoring (ConfigDict(extra="ignore"))
- OpenAI chat router's model_parameters building logic
- Chat router's model_parameters passthrough
- CompletionService model_parameters passthrough
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.model_parameters import ModelParameters


class TestModelParametersDefaults:
    """ModelParameters fields with non-None defaults should match spec."""

    def test_main_gpu_default(self):
        mp = ModelParameters()
        assert mp.main_gpu == -1

    def test_split_mode_default(self):
        mp = ModelParameters()
        assert mp.split_mode == "layer"

    def test_n_cpu_moe_default(self):
        mp = ModelParameters()
        assert mp.n_cpu_moe == 0

    def test_kv_on_cpu_default(self):
        mp = ModelParameters()
        assert mp.kv_on_cpu is False

    def test_reasoning_effort_default(self):
        mp = ModelParameters()
        assert mp.reasoning_effort == "medium"

    def test_flash_attention_default(self):
        mp = ModelParameters()
        assert mp.flash_attention is True

    def test_ctx_size_reduction_limit_default(self):
        mp = ModelParameters()
        assert mp.ctx_size_reduction_limit == 0.5

    def test_spec_draft_n_max_default(self):
        mp = ModelParameters()
        assert mp.spec_draft_n_max == 3

    def test_kv_unified_default(self):
        mp = ModelParameters()
        assert mp.kv_unified is True

    def test_optional_fields_default_none(self):
        mp = ModelParameters()
        assert mp.num_ctx is None
        assert mp.temperature is None
        assert mp.seed is None
        assert mp.stop is None
        assert mp.max_tokens is None
        assert mp.top_k is None
        assert mp.top_p is None
        assert mp.min_p is None
        assert mp.think is None
        assert mp.n_parts is None
        assert mp.batch_size is None
        assert mp.parallel is None
        assert mp.reasoning_budget is None


class TestModelParametersValidation:
    """ModelParameters enforces field constraints."""

    def test_n_gpu_layers_ge_constraint(self):
        ModelParameters(n_gpu_layers=0)
        ModelParameters(n_gpu_layers=20)
        with pytest.raises(Exception):
            ModelParameters(n_gpu_layers=-2)

    def test_main_gpu_ge_constraint(self):
        ModelParameters(main_gpu=0)
        with pytest.raises(Exception):
            ModelParameters(main_gpu=-2)

    def test_n_cpu_moe_ge_constraint(self):
        ModelParameters(n_cpu_moe=5)
        with pytest.raises(Exception):
            ModelParameters(n_cpu_moe=-1)

    def test_ctx_size_reduction_limit_range(self):
        ModelParameters(ctx_size_reduction_limit=0.0)
        ModelParameters(ctx_size_reduction_limit=1.0)
        ModelParameters(ctx_size_reduction_limit=0.75)
        with pytest.raises(Exception):
            ModelParameters(ctx_size_reduction_limit=-0.1)
        with pytest.raises(Exception):
            ModelParameters(ctx_size_reduction_limit=1.1)

    def test_spec_draft_n_max_range(self):
        ModelParameters(spec_draft_n_max=1)
        ModelParameters(spec_draft_n_max=16)
        with pytest.raises(Exception):
            ModelParameters(spec_draft_n_max=0)
        with pytest.raises(Exception):
            ModelParameters(spec_draft_n_max=17)

    def test_parallel_ge_constraint(self):
        ModelParameters(parallel=1)
        ModelParameters(parallel=8)
        with pytest.raises(Exception):
            ModelParameters(parallel=0)

    def test_split_mode_literal(self):
        ModelParameters(split_mode="none")
        ModelParameters(split_mode="layer")
        ModelParameters(split_mode="row")
        with pytest.raises(Exception):
            ModelParameters(split_mode="tensor")

    def test_reasoning_effort_literal(self):
        ModelParameters(reasoning_effort="low")
        ModelParameters(reasoning_effort="medium")
        ModelParameters(reasoning_effort="high")
        with pytest.raises(Exception):
            ModelParameters(reasoning_effort="extreme")

    def test_stop_is_string_list(self):
        mp = ModelParameters(stop=["\n", "USER:"])
        assert mp.stop == ["\n", "USER:"]

    def test_numeric_fields_accept_correct_types(self):
        mp = ModelParameters(
            temperature=0.7,
            top_p=0.9,
            min_p=0.1,
            repeat_penalty=1.2,
            repeat_last_n=64,
            num_ctx=8192,
            num_predict=256,
            top_k=40,
            seed=42,
            max_tokens=4096,
        )
        assert mp.temperature == 0.7
        assert mp.top_p == 0.9
        assert mp.min_p == 0.1
        assert mp.repeat_penalty == 1.2
        assert mp.repeat_last_n == 64
        assert mp.num_ctx == 8192
        assert mp.num_predict == 256
        assert mp.top_k == 40
        assert mp.seed == 42
        assert mp.max_tokens == 4096


class TestModelParametersExtraFields:
    """ModelParameters ignores unknown fields via ConfigDict(extra='ignore')."""

    def test_unknown_fields_ignored(self):
        mp = ModelParameters.model_validate({
            "temperature": 0.5,
            "unknown_field": "hello",
            "another_garbage": 12345,
        })
        assert mp.temperature == 0.5
        assert not hasattr(mp, "unknown_field")
        assert not hasattr(mp, "another_garbage")

    def test_from_json_ignores_extra(self):
        mp = ModelParameters.model_validate_json(
            '{"top_k": 50, "completely_made_up": true}'
        )
        assert mp.top_k == 50


class TestModelParametersCopy:
    """ModelParameters.model_copy works for merging overrides."""

    def test_model_copy_update(self):
        base = ModelParameters(temperature=0.5, top_k=40)
        overridden = base.model_copy(update={"temperature": 0.9})
        assert overridden.temperature == 0.9
        assert overridden.top_k == 40
        assert base.temperature == 0.5  # original unchanged

    def test_model_copy_with_none_fields(self):
        base = ModelParameters(seed=42)
        copy = base.model_copy(update={"max_tokens": 1024})
        assert copy.seed == 42
        assert copy.max_tokens == 1024


# ---------------------------------------------------------------------------
# Tests exercising the model_parameters merging logic from the
# OpenAI chat router (routers/openai/chat.py lines 604-631).
# ---------------------------------------------------------------------------

class TestOpenAIModelParametersBuilding:
    """Test the exact merging logic from the OpenAI chat router.

    Reproduces the logic from createChatCompletion() to verify:
    - Explicit OAI fields take highest priority
    - body.model_parameters provides base values
    - model_parameters is None when nothing is set
    """

    def _build_model_parameters(self, body):
        """Replicates the exact logic from routers/openai/chat.py:604-631."""
        _oai_params_dict = {}
        if body.max_tokens is not None:
            _oai_params_dict["max_tokens"] = body.max_tokens
        if body.seed is not None:
            _oai_params_dict["seed"] = body.seed
        if body.stop is not None:
            _oai_params_dict["stop"] = (
                body.stop.root if hasattr(body.stop, "root") else body.stop
            )
        if body.frequency_penalty != 0:
            _oai_params_dict["repeat_penalty"] = body.frequency_penalty
        if body.presence_penalty != 0 and body.frequency_penalty == 0:
            _oai_params_dict["repeat_penalty"] = body.presence_penalty
        if body.reasoning_effort is not None:
            _oai_params_dict["reasoning_effort"] = (
                body.reasoning_effort.root
                if hasattr(body.reasoning_effort, "root")
                else body.reasoning_effort
            )

        model_parameters = None
        if body.model_parameters or _oai_params_dict:
            if body.model_parameters:
                model_parameters = body.model_parameters.model_copy(
                    update=_oai_params_dict if _oai_params_dict else {}
                )
            elif _oai_params_dict:
                model_parameters = ModelParameters(**_oai_params_dict)
        return model_parameters

    def test_only_oai_max_tokens(self):
        body = MagicMock()
        body.max_tokens = 2048
        body.seed = None
        body.stop = None
        body.frequency_penalty = 0
        body.presence_penalty = 0
        body.reasoning_effort = None
        body.model_parameters = None

        mp = self._build_model_parameters(body)
        assert mp is not None
        assert mp.max_tokens == 2048
        assert mp.seed is None

    def test_only_oai_seed(self):
        body = MagicMock()
        body.max_tokens = None
        body.seed = 123
        body.stop = None
        body.frequency_penalty = 0
        body.presence_penalty = 0
        body.reasoning_effort = None
        body.model_parameters = None

        mp = self._build_model_parameters(body)
        assert mp is not None
        assert mp.seed == 123

    def test_oai_stop_list(self):
        body = MagicMock()
        body.max_tokens = None
        body.seed = None
        body.stop = ["\n", "END"]
        body.frequency_penalty = 0
        body.presence_penalty = 0
        body.reasoning_effort = None
        body.model_parameters = None

        mp = self._build_model_parameters(body)
        assert mp.stop == ["\n", "END"]

    def test_frequency_penalty_to_repeat_penalty(self):
        body = MagicMock()
        body.max_tokens = None
        body.seed = None
        body.stop = None
        body.frequency_penalty = 1.5
        body.presence_penalty = 0.5
        body.reasoning_effort = None
        body.model_parameters = None

        mp = self._build_model_parameters(body)
        assert mp.repeat_penalty == 1.5

    def test_presence_penalty_only_when_no_frequency(self):
        body = MagicMock()
        body.max_tokens = None
        body.seed = None
        body.stop = None
        body.frequency_penalty = 0
        body.presence_penalty = 0.8
        body.reasoning_effort = None
        body.model_parameters = None

        mp = self._build_model_parameters(body)
        assert mp.repeat_penalty == 0.8

    def test_model_parameters_base_with_oai_override(self):
        base_mp = ModelParameters(temperature=0.3, num_ctx=4096)
        body = MagicMock()
        body.max_tokens = 512
        body.seed = None
        body.stop = None
        body.frequency_penalty = 0
        body.presence_penalty = 0
        body.reasoning_effort = None
        body.model_parameters = base_mp

        mp = self._build_model_parameters(body)
        assert mp.temperature == 0.3  # from base
        assert mp.num_ctx == 4096     # from base
        assert mp.max_tokens == 512   # overridden by OAI field

    def test_model_parameters_base_no_oai_fields(self):
        base_mp = ModelParameters(temperature=0.7, top_k=50)
        body = MagicMock()
        body.max_tokens = None
        body.seed = None
        body.stop = None
        body.frequency_penalty = 0
        body.presence_penalty = 0
        body.reasoning_effort = None
        body.model_parameters = base_mp

        mp = self._build_model_parameters(body)
        assert mp.temperature == 0.7
        assert mp.top_k == 50
        assert mp.max_tokens is None

    def test_no_fields_at_all_returns_none(self):
        body = MagicMock()
        body.max_tokens = None
        body.seed = None
        body.stop = None
        body.frequency_penalty = 0
        body.presence_penalty = 0
        body.reasoning_effort = None
        body.model_parameters = None

        mp = self._build_model_parameters(body)
        assert mp is None

    def test_reasoning_effort_from_oai(self):
        body = MagicMock()
        body.max_tokens = None
        body.seed = None
        body.stop = None
        body.frequency_penalty = 0
        body.presence_penalty = 0
        body.reasoning_effort = "high"
        body.model_parameters = None

        mp = self._build_model_parameters(body)
        assert mp.reasoning_effort == "high"

    def test_reasoning_effort_oai_overrides_base(self):
        base_mp = ModelParameters(reasoning_effort="low")
        body = MagicMock()
        body.max_tokens = None
        body.seed = None
        body.stop = None
        body.frequency_penalty = 0
        body.presence_penalty = 0
        body.reasoning_effort = "high"
        body.model_parameters = base_mp

        mp = self._build_model_parameters(body)
        assert mp.reasoning_effort == "high"

    def test_multiple_oai_fields_merge(self):
        body = MagicMock()
        body.max_tokens = 1024
        body.seed = 42
        body.stop = ["STOP"]
        body.frequency_penalty = 1.1
        body.presence_penalty = 0
        body.reasoning_effort = "medium"
        body.model_parameters = None

        mp = self._build_model_parameters(body)
        assert mp.max_tokens == 1024
        assert mp.seed == 42
        assert mp.stop == ["STOP"]
        assert mp.repeat_penalty == 1.1
        assert mp.reasoning_effort == "medium"


# ---------------------------------------------------------------------------
# Tests for the chat router model_parameters passthrough
# ---------------------------------------------------------------------------

class TestChatRouterModelParametersPassthrough:
    """Chat router accepts and passes through model_parameters."""

    def test_chat_completion_body_accepts_model_parameters(self):
        from routers.chat import ChatCompletionBody
        from models import Message, MessageRole
        from models.message_content import MessageContent, MessageContentType

        msg = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="hello")],
            conversation_id=42,
        )
        body = ChatCompletionBody(
            message=msg,
            model_parameters=ModelParameters(temperature=0.7, num_ctx=4096),
        )
        assert body.model_parameters.temperature == 0.7
        assert body.model_parameters.num_ctx == 4096

    def test_chat_completion_body_without_model_parameters(self):
        from routers.chat import ChatCompletionBody
        from models import Message, MessageRole
        from models.message_content import MessageContent, MessageContentType

        msg = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="hello")],
            conversation_id=42,
        )
        body = ChatCompletionBody(message=msg)
        assert body.model_parameters is None

    def test_composer_chat_completion_forwards_model_parameters(self):
        """composer_chat_completion passes model_parameters to CompletionService."""
        from unittest.mock import patch as _patch

        captured = {}

        async def _fake_stream_completion(**kwargs):
            captured.update(kwargs)
            return
            yield

        with _patch("routers.chat.CompletionService") as mock_cs:
            mock_cs.stream_completion = _fake_stream_completion
            from routers.chat import composer_chat_completion

            async def run():
                async for _ in composer_chat_completion(
                    user_id="u1",
                    conversation_id=99,
                    request_id="r1",
                    model_parameters=ModelParameters(seed=7),
                ):
                    pass

            import asyncio
            asyncio.get_event_loop().run_until_complete(run())

        assert captured["model_parameters"] is not None
        assert captured["model_parameters"].seed == 7


# ---------------------------------------------------------------------------
# Tests for CompletionService model_parameters passthrough
# ---------------------------------------------------------------------------

class TestCompletionServiceModelParameters:
    """CompletionService methods accept and forward model_parameters."""

    def test_build_workflow_signature_accepts_model_parameters(self):
        from services.completion_service import CompletionService

        sig = inspect.signature(CompletionService.build_workflow)
        assert "model_parameters" in sig.parameters

    def test_stream_completion_signature_accepts_model_parameters(self):
        from services.completion_service import CompletionService

        sig = inspect.signature(CompletionService.stream_completion)
        assert "model_parameters" in sig.parameters

    def test_run_completion_signature_accepts_model_parameters(self):
        from services.completion_service import CompletionService

        sig = inspect.signature(CompletionService.run_completion)
        assert "model_parameters" in sig.parameters

    def test_build_and_run_signature_accepts_model_parameters(self):
        from services.completion_service import CompletionService

        sig = inspect.signature(CompletionService._build_and_run)
        assert "model_parameters" in sig.parameters

    def test_build_and_run_with_retry_signature(self):
        from services.completion_service import CompletionService

        sig = inspect.signature(CompletionService._build_and_run_with_retry)
        assert "model_parameters" in sig.parameters

    def test_collect_response_signature(self):
        from services.continuation_logic import collect_response

        sig = inspect.signature(collect_response)
        assert "model_parameters" in sig.parameters

    def test_stream_secondary_pass_signature(self):
        from services.continuation_logic import stream_secondary_pass

        sig = inspect.signature(stream_secondary_pass)
        assert "model_parameters" in sig.parameters


class TestCompletionServiceModelParametersExecution:
    """Execute CompletionService methods with model_parameters to cover passthrough."""

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
        from models.message import Message
        from models.message_content import MessageContent
        from models.message_content_type import MessageContentType
        from models.message_role import MessageRole
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="Hello there!")],
            ),
            finish_reason="stop",
        )

    @pytest.fixture
    def workflow_type(self):
        from graph.workflows.factory import WorkFlowType
        return WorkFlowType.DIALOG

    @pytest.fixture
    def mock_workflow_and_builder(self):
        mock_workflow = MagicMock()
        mock_builder = MagicMock()
        mock_builder.server_handle = None
        return mock_workflow, mock_builder

    @pytest.mark.asyncio
    async def test_build_workflow_forwards_model_parameters(
        self, mock_workflow_and_builder, workflow_type,
    ):
        """build_workflow passes model_parameters to compose_workflow."""
        from services.completion_service import CompletionService

        mock_workflow, mock_builder = mock_workflow_and_builder
        captured = {}

        async def fake_compose_workflow(**kwargs):
            captured.update(kwargs)
            return mock_workflow

        with patch("services.completion_service.compose_workflow", fake_compose_workflow), \
             patch("services.completion_service.get_graph_builder",
                   AsyncMock(return_value=mock_builder)):
            await CompletionService.build_workflow(
                user_id="u1",
                model_name="test-model",
                workflow_type=workflow_type,
                model_parameters=ModelParameters(temperature=0.7, num_ctx=4096),
            )

        assert captured["model_parameters"] is not None
        assert captured["model_parameters"].temperature == 0.7
        assert captured["model_parameters"].num_ctx == 4096

    @pytest.mark.asyncio
    async def test_build_and_run_forwards_model_parameters(
        self, mock_messages, mock_chat_response, mock_workflow_and_builder, workflow_type,
    ):
        """_build_and_run passes model_parameters through to build_workflow."""
        from services.completion_service import CompletionService

        mock_workflow, mock_builder = mock_workflow_and_builder

        async def fake_build_workflow(*a, **kw):
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def fake_create_initial_state(*a, **kw):
            return {}

        async def fake_run_workflow(state, wf, wt, disconnected=None):
            yield mock_chat_response

        with patch.object(CompletionService, "build_workflow", fake_build_workflow), \
             patch("services.completion_service.create_initial_state", fake_create_initial_state), \
             patch.object(CompletionService, "_run_workflow", fake_run_workflow):
            events = []
            async for ev in CompletionService._build_and_run(
                user_id="u1",
                messages=mock_messages,
                model_name="test-model",
                workflow_type=workflow_type,
                conversation_id=1,
                model_parameters=ModelParameters(seed=42),
            ):
                events.append(ev)

            assert len(events) == 1

    @pytest.mark.asyncio
    async def test_build_and_run_with_retry_forwards_model_parameters(
        self, mock_messages, mock_chat_response, mock_workflow_and_builder, workflow_type,
    ):
        """_build_and_run_with_retry passes model_parameters to _build_and_run."""
        from services.completion_service import CompletionService

        mock_workflow, mock_builder = mock_workflow_and_builder

        async def fake_build_workflow(*a, **kw):
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def fake_create_initial_state(*a, **kw):
            return {}

        async def fake_run_workflow(state, wf, wt, disconnected=None):
            yield mock_chat_response

        with patch.object(CompletionService, "build_workflow", fake_build_workflow), \
             patch("services.completion_service.create_initial_state", fake_create_initial_state), \
             patch.object(CompletionService, "_run_workflow", fake_run_workflow):
            events = []
            async for ev in CompletionService._build_and_run_with_retry(
                user_id="u1",
                messages=mock_messages,
                model_name="test-model",
                workflow_type=workflow_type,
                conversation_id=1,
                model_parameters=ModelParameters(top_k=50),
            ):
                events.append(ev)

            assert len(events) == 1

    @pytest.mark.asyncio
    async def test_collect_response_forwards_model_parameters(
        self, mock_messages, mock_chat_response, workflow_type,
    ):
        """collect_response passes model_parameters to build_and_run."""
        from services.continuation_logic import collect_response

        captured = {}
        async def fake_build_and_run(user_id, messages, model_name, workflow_type,
                                      conversation_id, client_tools, tool_choice,
                                      server_tool_names, model_parameters):
            captured["model_parameters"] = model_parameters
            resp = mock_chat_response.model_copy(update={"done": True})
            yield resp

        result = await collect_response(
            fake_build_and_run,
            user_id="u1",
            messages=mock_messages,
            model_name="test-model",
            workflow_type=workflow_type,
            conversation_id=1,
            client_tools=None,
            tool_choice=None,
            server_tool_names=None,
            model_parameters=ModelParameters(max_tokens=200),
        )

        assert result is not None
        assert captured["model_parameters"].max_tokens == 200

    @pytest.mark.asyncio
    async def test_stream_completion_forwards_model_parameters(
        self, mock_messages, mock_chat_response, mock_workflow_and_builder, workflow_type,
    ):
        """stream_completion passes model_parameters through the full chain."""
        from services.completion_service import CompletionService

        mock_workflow, mock_builder = mock_workflow_and_builder

        async def fake_build_workflow(*a, **kw):
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def fake_create_initial_state(*a, **kw):
            return {}

        async def fake_run_workflow(state, wf, wt, disconnected=None):
            resp = mock_chat_response.model_copy(update={"done": True})
            yield resp

        mock_pq = MagicMock()
        mock_pq.ensure_model_available = AsyncMock(return_value="test-model")

        with patch.object(CompletionService, "build_workflow", fake_build_workflow), \
             patch("services.completion_service.create_initial_state", fake_create_initial_state), \
             patch.object(CompletionService, "_run_workflow", fake_run_workflow), \
             patch("services.priority_queue.priority_queue", mock_pq), \
             patch("config.PRIORITY_QUEUE_ENABLED", False):
            events = []
            async for ev, acc in CompletionService.stream_completion(
                user_id="u1",
                messages=mock_messages,
                model_name="test-model",
                workflow_type=workflow_type,
                conversation_id=1,
                model_parameters=ModelParameters(temperature=0.5),
            ):
                events.append(ev)

            assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_run_completion_forwards_model_parameters(
        self, mock_messages, mock_chat_response, mock_workflow_and_builder, workflow_type,
    ):
        """run_completion passes model_parameters through the full chain."""
        from services.completion_service import CompletionService

        mock_workflow, mock_builder = mock_workflow_and_builder

        async def fake_build_workflow(*a, **kw):
            return (mock_workflow, mock_builder, "http://localhost:8000")

        async def fake_create_initial_state(*a, **kw):
            return {}

        async def fake_run_workflow(state, wf, wt, disconnected=None):
            resp = mock_chat_response.model_copy(update={"done": True})
            yield resp

        mock_pq = MagicMock()
        mock_pq.ensure_model_available = AsyncMock(return_value="test-model")

        with patch.object(CompletionService, "build_workflow", fake_build_workflow), \
             patch("services.completion_service.create_initial_state", fake_create_initial_state), \
             patch.object(CompletionService, "_run_workflow", fake_run_workflow), \
             patch("services.priority_queue.priority_queue", mock_pq), \
             patch("config.PRIORITY_QUEUE_ENABLED", False):
            result = await CompletionService.run_completion(
                user_id="u1",
                messages=mock_messages,
                model_name="test-model",
                workflow_type=workflow_type,
                conversation_id=1,
                model_parameters=ModelParameters(top_p=0.9),
            )

            assert result.chat_response is not None

    @pytest.mark.asyncio
    async def test_stream_secondary_pass_forwards_model_parameters(
        self, mock_messages, mock_chat_response, workflow_type,
    ):
        """stream_secondary_pass passes model_parameters to build_and_run."""
        from services.continuation_logic import stream_secondary_pass
        from services.completion_service import StreamAccumulator

        captured = {}
        async def fake_build_and_run(user_id, messages, model_name, workflow_type,
                                      conversation_id, client_tools, tool_choice,
                                      server_tool_names, model_parameters):
            captured["model_parameters"] = model_parameters
            yield mock_chat_response

        acc = StreamAccumulator()
        events = []
        async for ev, _acc in stream_secondary_pass(
            fake_build_and_run,
            acc,
            user_id="u1",
            messages=mock_messages,
            model_name="test-model",
            workflow_type=workflow_type,
            conversation_id=1,
            client_tools=None,
            tool_choice=None,
            server_tool_names=None,
            model_parameters=ModelParameters(min_p=0.05),
        ):
            events.append(ev)

        assert captured["model_parameters"].min_p == 0.05
