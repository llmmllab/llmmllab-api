"""
Server-side tool execution node for the IDE workflow graph.

This node intercepts tool calls for server-side tools (web_search, web_fetch)
from the agent's response and executes them locally, appending results back
to state so the agent can continue with the tool output.

Works with the Message-based WorkflowState (not LangChain AIMessage), making
it compatible with the existing agent/state architecture.

Reliability features (added 2026-05-22 audit):
  - Hard iteration cap (``config.SERVER_TOOL_MAX_ITERATIONS``) consulted by
    ``make_should_continue_server_tools`` so the loop can't spin past it.
  - In-iteration dedup: identical tool calls within one assistant turn fire
    once.
  - Cross-iteration result cache (``state.server_tool_call_cache``): a tool
    call with the same canonical name + args as a previous iteration reuses
    the prior result without re-firing the network request.
  - Parallel execution: distinct tool calls in one iteration run via
    ``asyncio.gather``.
  - Per-tool timeouts (``_TOOL_TIMEOUTS`` in ``server_tool_executor``):
    a stalled fetch surfaces as a structured error string rather than a
    hung stream.
"""

import asyncio
from typing import Set

import config
from graph.state import WorkflowState
from tools.server_tool_executor import (
    extract_server_tool_calls,
    execute_server_tool,
    dedupe_tool_calls,
    tool_call_cache_key,
    _CLIENT_TOOL_NAME_MAP,
)
from models import MessageRole
from models.message import Message, MessageContent, MessageContentType
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ServerToolNode")


class ServerToolNode:
    """Graph node that executes server-side tool calls from the last assistant message.

    Only processes tool calls whose names match the provided server_tool_names set.
    Other tool calls (client-side) are left untouched for proxy passthrough.

    Populates ``state.server_tool_events`` with dicts of the form::

        {"tool_call": ToolCall, "result_text": str, "canonical_name": str}

    so the executor / router can emit the correct SSE content blocks.
    """

    def __init__(self, server_tool_names: Set[str]):
        self.server_tool_names = server_tool_names

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        if not state.messages:
            return state

        last_message = state.messages[-1]
        if last_message.role != MessageRole.ASSISTANT or not last_message.tool_calls:
            return state

        server_calls, _client_calls = extract_server_tool_calls(
            last_message.tool_calls, self.server_tool_names
        )

        if not server_calls:
            return state

        # In-iteration dedup: the model occasionally emits the same call
        # twice in one assistant turn (e.g. two identical web_search args).
        # Keep the first; drop subsequent duplicates.
        deduped_calls = dedupe_tool_calls(server_calls)
        if len(deduped_calls) < len(server_calls):
            logger.info(
                "Dropped in-iteration duplicate tool calls",
                extra={
                    "original": len(server_calls),
                    "kept": len(deduped_calls),
                },
            )

        # Resolve each call against the cross-iteration cache.  Items
        # that hit the cache skip execution; items that miss get
        # gathered into a parallel batch.
        cache = state.server_tool_call_cache or {}
        misses: list = []
        results_by_call: dict[int, str] = {}
        cache_hits = 0
        for idx, tc in enumerate(deduped_calls):
            key = tool_call_cache_key(tc.name, tc.args)
            cached = cache.get(key)
            if cached is not None:
                results_by_call[idx] = cached
                cache_hits += 1
            else:
                misses.append((idx, tc, key))

        if cache_hits:
            logger.info(
                "Server-tool cache hits",
                extra={"hits": cache_hits, "misses": len(misses)},
            )

        # Parallel execution of cache misses.  asyncio.gather preserves
        # ordering and lets two `web_fetch` calls overlap their latency
        # instead of being serialised.
        if misses:
            logger.info(
                "Executing server-side tool calls (parallel)",
                extra={
                    "tool_names": [tc.name for _, tc, _ in misses],
                    "count": len(misses),
                },
            )
            outputs = await asyncio.gather(
                *[execute_server_tool(tc) for _, tc, _ in misses],
                return_exceptions=False,  # individual tools handle their own errors
            )
            for (idx, _tc, _key), out in zip(misses, outputs):
                results_by_call[idx] = out

        # Build the result events + collect new cache entries to merge
        # back into state via the dict reducer.
        new_events: list[dict] = []
        result_parts: list[str] = []
        new_cache_entries: dict[str, str] = {}
        for idx, tc in enumerate(deduped_calls):
            result_text = results_by_call[idx]
            canonical = _CLIENT_TOOL_NAME_MAP.get(tc.name, tc.name)
            new_events.append(
                {
                    "tool_call": tc,
                    "result_text": result_text,
                    "canonical_name": canonical,
                }
            )
            result_parts.append(result_text)
            new_cache_entries[tool_call_cache_key(tc.name, tc.args)] = result_text

        # --- Rewrite messages so the local model understands tool results ---
        # Local models don't handle ToolMessage / tool_call_id well.
        # Instead:
        #   1. Strip server tool calls from the last assistant message
        #      (keep any text content + client-side tool calls).
        #   2. Inject a USER message containing the search/fetch results
        #      so the model sees them as normal context it can reason over.
        server_call_names = {tc.name for tc in server_calls}
        remaining_tool_calls = [
            tc
            for tc in (last_message.tool_calls or [])
            if tc.name not in server_call_names
        ]
        last_message.tool_calls = remaining_tool_calls or None

        # Ensure the assistant message has some text content (avoids empty AIMessage
        # which LangChain drops / the model sees as EOS).
        has_text = any(
            c.type == MessageContentType.TEXT and c.text for c in last_message.content
        )
        if not has_text:
            query_summaries = []
            for tc in deduped_calls:
                args = tc.args or {}
                q = args.get("query") or args.get("url") or tc.name
                query_summaries.append(q)
            last_message.content.append(
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=f"[Performed server-side tool calls: {', '.join(query_summaries)}]",
                )
            )

        # Build a single USER message with all results
        combined_results = "\n\n".join(result_parts)
        state.messages.append(
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=(
                            f"Here are the results from the tools you just invoked. "
                            f"Use these results to answer the original question:\n\n"
                            f"{combined_results}"
                        ),
                    )
                ],
            )
        )

        state.server_tool_events.extend(new_events)
        state.server_tool_iterations = 1  # reducer adds to existing count
        # Set only the *delta* on the cache field; the dict reducer
        # merges it onto the running cache.
        state.server_tool_call_cache = new_cache_entries

        return state


def make_should_continue_server_tools(server_tool_names: Set[str]):
    """Create a routing function that routes to the server tool node only when
    the last message contains server-side tool calls AND the iteration cap
    has not yet been reached.

    Returns "server_tools" if there are server tool calls AND
    ``state.server_tool_iterations < config.SERVER_TOOL_MAX_ITERATIONS``,
    "end" otherwise.

    The cap exists so a model that keeps emitting tool calls after each
    result can't spin the loop indefinitely (previously the only safety
    net was LangGraph's global ``recursion_limit=25`` which surfaced as
    a generic 500 / "Sorry, I could not complete your request").
    """

    def should_continue(state: WorkflowState) -> str:
        if not state.messages:
            return "end"

        last_message = state.messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"

        # Enforce the hard cap before checking the tool-call set, so that
        # a model wedged in the loop doesn't fire one final unanswered
        # tool call.
        if state.server_tool_iterations >= config.SERVER_TOOL_MAX_ITERATIONS:
            logger.warning(
                "Server-tool iteration cap reached; routing to END",
                extra={
                    "iterations": state.server_tool_iterations,
                    "cap": config.SERVER_TOOL_MAX_ITERATIONS,
                },
            )
            return "end"

        # Check if any tool calls are for server-side tools
        for tc in last_message.tool_calls:
            if tc.name in server_tool_names:
                return "server_tools"

        return "end"

    return should_continue
