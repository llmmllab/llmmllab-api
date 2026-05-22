"""
Unit tests for the server-side tool execution audit fixes
(2026-05-22): iteration cap, dedup, cross-iteration cache, parallel
execution, per-tool & per-request overrides.

These tests pin the contract for ``tools/server_tool_executor.py``,
``services/tool_service.py``, and ``graph/nodes/server_tools.py``.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import config
from graph.nodes.server_tools import (
    ServerToolNode,
    make_should_continue_server_tools,
)
from graph.state import WorkflowState
from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    UserConfig,
)
from models.tool_call import ToolCall
from services.tool_service import ToolService
from tools.server_tool_executor import (
    dedupe_tool_calls,
    find_locally_executable_tools,
    tool_call_cache_key,
)


# ---------------------------------------------------------------------------
# Cache-key + dedup
# ---------------------------------------------------------------------------

def test_cache_key_stable_across_arg_ordering():
    """Same args in different key order produce identical cache keys."""
    a = tool_call_cache_key("web_search", {"query": "kubernetes", "n": 5})
    b = tool_call_cache_key("web_search", {"n": 5, "query": "kubernetes"})
    assert a == b


def test_cache_key_folds_pascal_and_snake_aliases():
    """``WebSearch`` and ``web_search`` resolve to the same canonical key."""
    a = tool_call_cache_key("WebSearch", {"query": "x"})
    b = tool_call_cache_key("web_search", {"query": "x"})
    assert a == b


def test_dedupe_keeps_first_drops_subsequent_duplicates():
    calls = [
        ToolCall(name="web_search", args={"query": "X"}),
        ToolCall(name="web_search", args={"query": "X"}),  # exact dup
        ToolCall(name="web_search", args={"query": "Y"}),
        ToolCall(name="WebSearch", args={"query": "Y"}),    # alias dup
    ]
    out = dedupe_tool_calls(calls)
    assert len(out) == 2
    assert [tc.args["query"] for tc in out] == ["X", "Y"]


# ---------------------------------------------------------------------------
# Per-tool override flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["client", "client_side", "client-side", "CLIENT"])
def test_find_locally_executable_skips_tools_with_client_execute_flag(flag):
    tools = [
        {"name": "WebSearch", "execute": flag},
        {"name": "WebFetch"},
    ]
    found = find_locally_executable_tools(tools)
    assert found == {"WebFetch"}


def test_find_locally_executable_does_not_match_unrelated_names():
    found = find_locally_executable_tools([{"name": "ReadFile"}])
    assert found == set()


# ---------------------------------------------------------------------------
# Master override (env / per-request)
# ---------------------------------------------------------------------------

def test_prepare_tools_disabled_returns_no_server_tools():
    tools = [{"name": "WebSearch"}, {"name": "ReadFile"}]
    prepared = ToolService.prepare_tools(tools, enabled=False)
    assert prepared.server_tool_names == set()
    assert prepared.client_tools == tools  # unchanged


def test_prepare_tools_enabled_detects_websearch():
    tools = [{"name": "WebSearch"}, {"name": "ReadFile"}]
    prepared = ToolService.prepare_tools(tools, enabled=True)
    assert "WebSearch" in prepared.server_tool_names


def test_prepare_tools_default_uses_config_flag():
    """When enabled is None, defaults to ``config.SERVER_SIDE_TOOLS_ENABLED``."""
    with patch.object(config, "SERVER_SIDE_TOOLS_ENABLED", False):
        prepared = ToolService.prepare_tools([{"name": "WebSearch"}], enabled=None)
    assert prepared.server_tool_names == set()


def test_prepare_tools_empty_input_short_circuits():
    prepared = ToolService.prepare_tools(None, enabled=True)
    assert prepared.server_tool_names == set()
    assert prepared.client_tools is None


# ---------------------------------------------------------------------------
# Iteration cap (routing)
# ---------------------------------------------------------------------------

def _make_state(messages=None, iterations=0, cache=None) -> WorkflowState:
    return WorkflowState(
        messages=messages or [],
        current_user_message=Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="hi")],
        ),
        conversation_id=1,
        user_id="u1",
        user_config=UserConfig(user_id="u1"),
        server_tool_iterations=iterations,
        server_tool_call_cache=cache or {},
    )


def _make_assistant_with_tool_call(name: str = "web_search") -> Message:
    return Message(
        role=MessageRole.ASSISTANT,
        content=[MessageContent(type=MessageContentType.TEXT, text="...")],
        tool_calls=[ToolCall(name=name, args={"query": "x"})],
    )


def test_should_continue_routes_to_server_tools_below_cap():
    fn = make_should_continue_server_tools({"web_search"})
    state = _make_state(messages=[_make_assistant_with_tool_call()], iterations=0)
    assert fn(state) == "server_tools"


def test_should_continue_routes_to_end_at_cap():
    fn = make_should_continue_server_tools({"web_search"})
    state = _make_state(
        messages=[_make_assistant_with_tool_call()],
        iterations=config.SERVER_TOOL_MAX_ITERATIONS,
    )
    assert fn(state) == "end"


def test_should_continue_routes_to_end_with_no_tool_calls():
    fn = make_should_continue_server_tools({"web_search"})
    bare_assistant = Message(
        role=MessageRole.ASSISTANT,
        content=[MessageContent(type=MessageContentType.TEXT, text="hello")],
    )
    state = _make_state(messages=[bare_assistant])
    assert fn(state) == "end"


# ---------------------------------------------------------------------------
# ServerToolNode behavior — dedup, cache, parallel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_server_tool_node_dedupes_identical_calls():
    """Two identical web_search calls fire execute_server_tool exactly once."""
    call_log: list[Any] = []

    async def fake_exec(tc):
        call_log.append((tc.name, dict(tc.args or {})))
        return f"result-for-{tc.args['query']}"

    state = _make_state(
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="...")],
                tool_calls=[
                    ToolCall(name="web_search", args={"query": "X"}),
                    ToolCall(name="web_search", args={"query": "X"}),
                ],
            )
        ]
    )

    with patch(
        "graph.nodes.server_tools.execute_server_tool",
        new=AsyncMock(side_effect=fake_exec),
    ):
        node = ServerToolNode({"web_search"})
        await node(state)

    assert len(call_log) == 1
    # One event added to state (the deduped one)
    assert len(state.server_tool_events) == 1


@pytest.mark.asyncio
async def test_server_tool_node_uses_cross_iteration_cache():
    """A call whose key is already in state cache does NOT re-execute."""
    state = _make_state(
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="...")],
                tool_calls=[ToolCall(name="web_search", args={"query": "X"})],
            )
        ],
        cache={tool_call_cache_key("web_search", {"query": "X"}): "cached-result"},
    )

    exec_mock = AsyncMock(return_value="should-not-be-called")
    with patch("graph.nodes.server_tools.execute_server_tool", new=exec_mock):
        node = ServerToolNode({"web_search"})
        await node(state)

    exec_mock.assert_not_called()
    assert state.server_tool_events[0]["result_text"] == "cached-result"


@pytest.mark.asyncio
async def test_server_tool_node_runs_distinct_calls_in_parallel():
    """asyncio.gather sees both calls in flight simultaneously."""
    in_flight = 0
    peak = 0
    barrier = asyncio.Event()

    async def slow_exec(tc):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        # Give both calls a chance to be scheduled before either resolves.
        if in_flight == 2:
            barrier.set()
        await barrier.wait()
        in_flight -= 1
        return f"ok:{tc.args['query']}"

    state = _make_state(
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="...")],
                tool_calls=[
                    ToolCall(name="web_search", args={"query": "A"}),
                    ToolCall(name="web_search", args={"query": "B"}),
                ],
            )
        ]
    )

    with patch(
        "graph.nodes.server_tools.execute_server_tool",
        new=AsyncMock(side_effect=slow_exec),
    ):
        node = ServerToolNode({"web_search"})
        await node(state)

    assert peak == 2, "expected both tool calls to be in-flight concurrently"


@pytest.mark.asyncio
async def test_execute_web_search_uses_mcp_when_configured():
    """When tools.mcp_client.get_default() returns a client, _execute_web_search
    delegates to MCP and reformats the result."""
    import json
    from tools.server_tool_executor import _execute_web_search

    class FakeMCP:
        async def call_tool(self, name: str, arguments: dict) -> str:
            assert name == "web_search"
            assert arguments == {"query": "kubernetes"}
            return json.dumps(
                {
                    "query": "kubernetes",
                    "contents": [
                        {
                            "title": "K8s docs",
                            "url": "https://k8s.io",
                            "content": "Container orchestration",
                            "relevance": 1.0,
                        }
                    ],
                }
            )

    with patch("tools.mcp_client.get_default", return_value=FakeMCP()):
        result = await _execute_web_search({"query": "kubernetes"})

    assert "K8s docs" in result
    assert "https://k8s.io" in result


@pytest.mark.asyncio
async def test_execute_web_fetch_uses_mcp_when_configured():
    """fetch_page on MCP returns text directly; _execute_web_fetch passes through."""
    from tools.server_tool_executor import _execute_web_fetch

    class FakeMCP:
        async def call_tool(self, name: str, arguments: dict) -> str:
            assert name == "fetch_page"
            assert arguments == {"url": "https://example.com"}
            return "Content from https://example.com:\n\nHello world"

    with patch("tools.mcp_client.get_default", return_value=FakeMCP()):
        result = await _execute_web_fetch({"url": "https://example.com"})

    assert "Hello world" in result


@pytest.mark.asyncio
async def test_execute_web_search_handles_mcp_call_error_gracefully():
    """MCPCallError from the client surfaces as an inline error string,
    not a raised exception."""
    from tools.server_tool_executor import _execute_web_search
    from tools.mcp_client import MCPCallError

    class BrokenMCP:
        async def call_tool(self, name: str, arguments: dict) -> str:
            raise MCPCallError("server gone")

    with patch("tools.mcp_client.get_default", return_value=BrokenMCP()):
        result = await _execute_web_search({"query": "x"})

    assert result.startswith("Error:")
    assert "server gone" in result


@pytest.mark.asyncio
async def test_server_tool_node_increments_iterations_and_extends_cache():
    state = _make_state(
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(type=MessageContentType.TEXT, text="...")],
                tool_calls=[ToolCall(name="web_search", args={"query": "Z"})],
            )
        ]
    )
    with patch(
        "graph.nodes.server_tools.execute_server_tool",
        new=AsyncMock(return_value="zzz"),
    ):
        node = ServerToolNode({"web_search"})
        await node(state)

    # Iteration delta of 1 set (reducer will add it to existing).
    assert state.server_tool_iterations == 1
    expected_key = tool_call_cache_key("web_search", {"query": "Z"})
    assert state.server_tool_call_cache.get(expected_key) == "zzz"
