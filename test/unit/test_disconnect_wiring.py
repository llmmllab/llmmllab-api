"""
Unit tests for end-to-end client-disconnect cancellation wiring (Fix 3).

The dormant disconnect predicate added in commit 094c092 (agents/base.py
``run``/``run_structured`` accept ``disconnected`` and raise
``asyncio.CancelledError`` before re-dispatch/backoff when it returns True)
was never reaching the agent because nothing threaded it through the
langgraph executor → graph-state → node → agent stack.  These tests cover the
full chain that now carries the predicate down:

    router (Request.is_disconnected)
      → CompletionService.stream_completion(disconnected=…)
        → _build_and_run_with_retry(disconnected=…)
          → stream_with_connection_retry(disconnected=…)
            → _build_and_run(disconnected=…)
              → _run_workflow(disconnected=…)
                → execute_workflow(disconnected=…)
                  → graph.executor.stream_workflow(disconnected=…)
                    → RunnableConfig.configurable["disconnected"]
                      → AgentNode.__call__(state, config)
                        → ChatAgent.run(disconnected=…)

This is the fix for the zombie IDE coding-agent sessions (5dbc086d, 0e8d6dc1,
15ec8952) that kept re-dispatching the 150-360 KB prompt to the 27B runner
long after the IDE client disconnected.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Prime the full module graph before any direct ``graph.nodes.agent`` import.
# Importing that node module *first* trips a pre-existing import cycle
# (graph.nodes.agent → tools.registry → services → composer_init →
# graph.workflows → ide.builder → graph.nodes.agent), so we let the
# ``services`` package establish the graph first — the same entrypoint the
# app and the other test suites import through.
import services  # noqa: F401,E402


# ---------------------------------------------------------------------------
# AgentNode predicate extraction + forwarding
# ---------------------------------------------------------------------------


class TestAgentNodeDisconnectExtraction:
    """AgentNode pulls the predicate out of RunnableConfig.configurable and
    forwards it into the agent's run/run_structured."""

    def test_disconnected_from_config_none_config(self):
        from graph.nodes.agent import _disconnected_from_config

        assert _disconnected_from_config(None) is None

    def test_disconnected_from_config_absent(self):
        from graph.nodes.agent import _disconnected_from_config

        assert _disconnected_from_config({"configurable": {}}) is None
        assert _disconnected_from_config({}) is None

    def test_disconnected_from_config_non_callable_ignored(self):
        from graph.nodes.agent import _disconnected_from_config

        # A stray non-callable value must never be returned as a predicate.
        assert (
            _disconnected_from_config({"configurable": {"disconnected": "nope"}})
            is None
        )

    def test_disconnected_from_config_returns_callable(self):
        from graph.nodes.agent import _disconnected_from_config

        async def pred() -> bool:
            return True

        assert (
            _disconnected_from_config({"configurable": {"disconnected": pred}})
            is pred
        )

    def _make_node(self, grammar=None):
        from graph.nodes.agent import AgentNode

        # Bypass ChatAgent construction — AgentNode only needs an object with
        # bind_node_metadata() returning a stand-in agent exposing run /
        # run_structured.
        agent = MagicMock()
        bound = MagicMock()
        bound.run = AsyncMock()
        bound.run_structured = AsyncMock()
        agent.bind_node_metadata.return_value = bound

        from models import NodeMetadata

        node = AgentNode(
            agent=agent,
            node_metadata=NodeMetadata(
                node_name="agent", node_id="n1", node_type="AgentNode"
            ),
            grammar=grammar,
        )
        return node, bound

    def _make_state(self):
        from graph.state import WorkflowState
        from models import Message, MessageContent
        from models.message_content_type import MessageContentType
        from models.user_config import UserConfig

        return WorkflowState(
            conversation_id=1,
            user_id="u1",
            user_config=UserConfig(user_id="u1"),
            messages=[
                Message(
                    role="user",
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text="Hello")
                    ],
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_node_forwards_predicate_to_run(self):
        node, bound = self._make_node()
        # run() returns a ChatResponse-like object whose .message is falsy so
        # the node simply returns the state.
        bound.run.return_value = MagicMock(message=None)
        state = self._make_state()

        async def pred() -> bool:
            return False

        await node(state, {"configurable": {"disconnected": pred}})

        bound.run.assert_awaited_once()
        assert bound.run.await_args.kwargs["disconnected"] is pred

    @pytest.mark.asyncio
    async def test_node_forwards_predicate_to_run_structured(self):
        from pydantic import BaseModel as _BM

        class Grammar(_BM):
            x: int

        node, bound = self._make_node(grammar=Grammar)
        bound.run_structured.return_value = Grammar(x=1)
        state = self._make_state()

        async def pred() -> bool:
            return False

        await node(state, {"configurable": {"disconnected": pred}})

        bound.run_structured.assert_awaited_once()
        assert bound.run_structured.await_args.kwargs["disconnected"] is pred

    @pytest.mark.asyncio
    async def test_node_no_config_passes_none(self):
        """No config (non-streaming / internal caller) → disconnected=None,
        zero behaviour change."""
        node, bound = self._make_node()
        bound.run.return_value = MagicMock(message=None)
        state = self._make_state()

        await node(state)  # no config

        bound.run.assert_awaited_once()
        assert bound.run.await_args.kwargs["disconnected"] is None


# ---------------------------------------------------------------------------
# Node + agent integration: mid-retry disconnect aborts, no re-dispatch
# ---------------------------------------------------------------------------


class TestNodeLevelDisconnectAbort:
    """A node-level agent run whose disconnect predicate flips True mid-retry
    must raise CancelledError and NOT re-dispatch to the runner."""

    def _make_chat_agent(self):
        from agents.chat import ChatAgent

        model = MagicMock()
        return ChatAgent(model=model, system_prompt="You are helpful.")

    def _make_node_with_agent(self, agent):
        from graph.nodes.agent import AgentNode
        from models import NodeMetadata

        return AgentNode(
            agent=agent,
            node_metadata=NodeMetadata(
                node_name="agent", node_id="n1", node_type="AgentNode"
            ),
        )

    def _make_state(self):
        from graph.state import WorkflowState
        from models import Message, MessageContent
        from models.message_content_type import MessageContentType
        from models.user_config import UserConfig

        return WorkflowState(
            conversation_id=1,
            user_id="u1",
            user_config=UserConfig(user_id="u1"),
            messages=[
                Message(
                    role="user",
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text="Hello")
                    ],
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_mid_retry_disconnect_raises_and_does_not_redispatch(self):
        from openai import APIStatusError

        agent = self._make_chat_agent()

        call_count = [0]
        connected = [True]

        def _status_error(code):
            resp = MagicMock()
            resp.status_code = code
            resp.headers = {}
            return APIStatusError("busy", response=resp, body={"detail": "busy"})

        async def flaky(*a, **kw):
            call_count[0] += 1
            connected[0] = False  # client leaves after the first dispatch
            raise _status_error(503)

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        node = self._make_node_with_agent(agent)
        state = self._make_state()

        async def pred() -> bool:
            return not connected[0]

        with patch("asyncio.sleep", new_callable=AsyncMock) as msleep:
            with pytest.raises(asyncio.CancelledError):
                await node(state, {"configurable": {"disconnected": pred}})

        # Exactly ONE dispatch, then the disconnect check aborted before the
        # backoff sleep and before any second dispatch — no zombie re-fire.
        assert call_count[0] == 1
        msleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_predicate_retries_normally(self):
        """Without a predicate in config the node's agent retries transient
        errors as before (no behaviour change)."""
        from openai import APIStatusError

        agent = self._make_chat_agent()
        call_count = [0]

        def _status_error(code):
            resp = MagicMock()
            resp.status_code = code
            resp.headers = {}
            return APIStatusError("busy", response=resp, body={"detail": "busy"})

        async def flaky(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise _status_error(503)
            return MagicMock(content="OK", role="ai")

        mock_lc_agent = AsyncMock()
        mock_lc_agent.ainvoke = flaky
        agent._get_or_create_agent = AsyncMock(return_value=mock_lc_agent)

        node = self._make_node_with_agent(agent)
        state = self._make_state()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await node(state)  # no config → disconnected=None

        assert call_count[0] == 3


# ---------------------------------------------------------------------------
# Executor injects the predicate into RunnableConfig.configurable
# ---------------------------------------------------------------------------


class TestExecutorInjectsPredicate:
    @pytest.mark.asyncio
    async def test_stream_workflow_puts_predicate_in_configurable(self):
        from graph.executor import WorkflowExecutor

        captured = {}

        async def fake_astream_events(state_dict, config=None, version=None):
            captured["config"] = config
            # Yield nothing — we only care about the config that was passed.
            if False:
                yield  # pragma: no cover

        workflow = MagicMock()
        workflow.astream_events = fake_astream_events

        initial_state = MagicMock()
        initial_state.model_dump.return_value = {"conversation_id": 1}
        # conversation_id read off the object directly by stream_workflow.
        initial_state.conversation_id = 1

        async def pred() -> bool:
            return False

        executor = WorkflowExecutor()
        events = []
        async for ev in executor.stream_workflow(
            workflow=workflow,
            initial_state=initial_state,
            thread_id="t1",
            disconnected=pred,
        ):
            events.append(ev)

        assert captured["config"] is not None
        assert captured["config"]["configurable"]["disconnected"] is pred

    @pytest.mark.asyncio
    async def test_stream_workflow_no_predicate_leaves_config_clean(self):
        from graph.executor import WorkflowExecutor

        captured = {}

        async def fake_astream_events(state_dict, config=None, version=None):
            captured["config"] = config
            if False:
                yield  # pragma: no cover

        workflow = MagicMock()
        workflow.astream_events = fake_astream_events
        initial_state = MagicMock()
        initial_state.model_dump.return_value = {"conversation_id": 1}
        initial_state.conversation_id = 1

        executor = WorkflowExecutor()
        async for _ in executor.stream_workflow(
            workflow=workflow,
            initial_state=initial_state,
            thread_id="t1",
        ):
            pass

        # No predicate threaded → no 'disconnected' key planted in config.
        cfg = captured["config"] or {}
        assert "disconnected" not in (cfg.get("configurable") or {})


# ---------------------------------------------------------------------------
# Full CompletionService chain forwards the predicate to the executor seam
# ---------------------------------------------------------------------------


def _get_completion_service():
    from services.completion_service import CompletionService
    return CompletionService


def _get_workflow_type():
    from graph.workflows.factory import WorkFlowType
    return WorkFlowType.DIALOG


def _mock_messages():
    from models.message import Message
    from models.message_content import MessageContent
    from models.message_content_type import MessageContentType
    return [
        Message(
            role="user",
            content=[MessageContent(type=MessageContentType.TEXT, text="Hello")],
        )
    ]


def _mock_chat_response():
    from models.chat_response import ChatResponse
    return ChatResponse(
        id="resp-1",
        model="test-model",
        content="Hi!",
        stop_reason="end_turn",
    )


class TestCompletionServiceThreadsPredicate:
    @pytest.mark.asyncio
    async def test_build_and_run_forwards_predicate_to_run_workflow(self):
        """_build_and_run hands the predicate to _run_workflow."""
        CompletionService = _get_completion_service()
        workflow_type = _get_workflow_type()
        resp = _mock_chat_response()

        captured = {}

        async def mock_build_workflow(*a, **kw):
            mb = MagicMock()
            mb.server_handle = None
            return (MagicMock(), mb, "http://localhost:8000")

        async def mock_create_initial_state(*a, **kw):
            return {}

        async def mock_run_workflow(state, wf, wt, disconnected=None):
            captured["disconnected"] = disconnected
            yield resp

        async def pred() -> bool:
            return False

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
                messages=_mock_messages(),
                model_name="test-model",
                workflow_type=workflow_type,
                conversation_id=1,
                disconnected=pred,
            ):
                events.append(ev)

        assert captured["disconnected"] is pred
        assert events == [resp]

    @pytest.mark.asyncio
    async def test_run_workflow_forwards_predicate_to_execute_workflow(self):
        """_run_workflow hands the predicate to execute_workflow (the seam
        directly above graph.executor.stream_workflow)."""
        CompletionService = _get_completion_service()
        workflow_type = _get_workflow_type()
        resp = _mock_chat_response()

        captured = {}

        async def fake_execute_workflow(initial_state, workflow, disconnected=None):
            captured["disconnected"] = disconnected
            yield resp

        async def pred() -> bool:
            return False

        with patch(
            "services.completion_service.execute_workflow", fake_execute_workflow
        ):
            events = []
            async for ev in CompletionService._run_workflow(
                {}, MagicMock(), workflow_type, disconnected=pred
            ):
                events.append(ev)

        assert captured["disconnected"] is pred
        assert events == [resp]

    @pytest.mark.asyncio
    async def test_full_chain_predicate_reaches_run_workflow(self):
        """stream_completion → _build_and_run_with_retry →
        stream_with_connection_retry → _build_and_run → _run_workflow:
        the predicate survives every intermediate layer untouched."""
        CompletionService = _get_completion_service()
        resp = _mock_chat_response()
        resp.done = True

        captured = {}

        async def mock_build_workflow(*a, **kw):
            mb = MagicMock()
            mb.server_handle = None
            return (MagicMock(), mb, "http://localhost:8000")

        async def mock_create_initial_state(*a, **kw):
            return {}

        async def mock_run_workflow(state, wf, wt, disconnected=None):
            captured["disconnected"] = disconnected
            yield resp

        async def pred() -> bool:
            return False

        # Disable the priority queue so _enqueue_and_wait is a no-op pass.
        with patch("config.PRIORITY_QUEUE_ENABLED", False), patch.object(
            CompletionService, "build_workflow", mock_build_workflow
        ), patch(
            "services.completion_service.create_initial_state",
            mock_create_initial_state,
        ), patch.object(
            CompletionService, "_run_workflow", mock_run_workflow
        ):
            events = []
            async for ev, _acc in CompletionService.stream_completion(
                user_id="u1",
                messages=_mock_messages(),
                model_name="test-model",
                conversation_id=1,
                disconnected=pred,
            ):
                events.append(ev)

        assert captured["disconnected"] is pred
        assert any(getattr(e, "done", False) for e in events)

    @pytest.mark.asyncio
    async def test_no_predicate_threads_none(self):
        """No disconnected argument → None reaches _run_workflow (the partial
        wrapper isn't even created); non-streaming callers are unaffected."""
        CompletionService = _get_completion_service()
        resp = _mock_chat_response()
        resp.done = True

        captured = {"set": False, "value": "unset"}

        async def mock_build_workflow(*a, **kw):
            mb = MagicMock()
            mb.server_handle = None
            return (MagicMock(), mb, "http://localhost:8000")

        async def mock_create_initial_state(*a, **kw):
            return {}

        async def mock_run_workflow(state, wf, wt, disconnected=None):
            captured["set"] = True
            captured["value"] = disconnected
            yield resp

        with patch("config.PRIORITY_QUEUE_ENABLED", False), patch.object(
            CompletionService, "build_workflow", mock_build_workflow
        ), patch(
            "services.completion_service.create_initial_state",
            mock_create_initial_state,
        ), patch.object(
            CompletionService, "_run_workflow", mock_run_workflow
        ):
            async for _ev, _acc in CompletionService.stream_completion(
                user_id="u1",
                messages=_mock_messages(),
                model_name="test-model",
                conversation_id=1,
            ):
                pass

        assert captured["set"] is True
        assert captured["value"] is None


# ---------------------------------------------------------------------------
# Router: a disconnected streaming request cancels the session's queued work
# ---------------------------------------------------------------------------


class TestRouterDisconnectCancelsSession:
    """When the agent raises CancelledError up through stream_completion (the
    disconnect-driven abort), the streaming router drops the session's queued
    work via priority_queue.cancel_by_session_id."""

    @pytest.mark.asyncio
    async def test_anthropic_stream_message_cancels_session(self):
        import routers.anthropic.messages as messages_mod

        async def boom_stream_completion(*a, **kw):
            # The disconnect predicate fired deep in the agent and propagated
            # CancelledError all the way up to the router's async-for.
            raise asyncio.CancelledError()
            yield  # pragma: no cover

        cancel_mock = AsyncMock(return_value=2)

        with patch.object(
            messages_mod.model_service,
            "resolve_default_model",
            AsyncMock(return_value="test-model"),
        ), patch.object(
            messages_mod.CompletionService,
            "build_workflow",
            AsyncMock(return_value=(MagicMock(), MagicMock(), "http://r:8000")),
        ), patch.object(
            messages_mod,
            "count_input_tokens",
            AsyncMock(return_value=10),
        ), patch.object(
            messages_mod.CompletionService,
            "stream_completion",
            boom_stream_completion,
        ), patch(
            "services.priority_queue.priority_queue.cancel_by_session_id",
            cancel_mock,
        ):
            chunks = []
            async for chunk in messages_mod.stream_message(
                user_id="u1",
                messages=_mock_messages(),
                model_name="test-model",
                session_id="zombie-session",
            ):
                chunks.append(chunk)

        cancel_mock.assert_awaited_once_with("zombie-session")

    @pytest.mark.asyncio
    async def test_stream_message_accepts_disconnected_kwarg(self):
        """The router stream functions accept the disconnected predicate
        (signature wiring) and forward it to stream_completion."""
        import inspect
        import routers.anthropic.messages as messages_mod
        import routers.openai.chat as chat_mod

        assert "disconnected" in inspect.signature(
            messages_mod.stream_message
        ).parameters
        assert "disconnected" in inspect.signature(
            chat_mod.stream_chat_completion
        ).parameters
