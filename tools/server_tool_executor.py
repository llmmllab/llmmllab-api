"""
Server-side tool executor for tools that should be executed locally.

When a client sends requests with server-side tools like web_search and web_fetch,
the upstream API would normally execute them. Since we proxy to a local model,
we intercept these tool calls and execute them using our own implementations.

This module:
1. Identifies which tools in a request are server-side (vs client-side)
2. Provides execution for web_search and web_fetch using existing SearxNG + web reader
3. Returns results in a format the model can consume as tool results
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Tuple

from models import Message, MessageRole, MessageContent, MessageContentType
from models.tool_call import ToolCall
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ServerToolExecutor")

# Per-tool execution timeouts (seconds).  Any single tool call exceeding
# its budget surfaces a structured error string to the model instead of
# stalling the stream until the upstream gives up.
_TOOL_TIMEOUTS = {
    "web_search": 30.0,
    "web_fetch": 60.0,
}

# Sentinel for tools whose definition explicitly opts out of server-side
# execution.  Clients can set this on a per-tool basis to keep the tool
# in the bound-tools list (so the model knows about it) while letting the
# client own its execution.
_CLIENT_EXEC_FLAG_VALUES = {"client", "client_side", "client-side"}


def tool_call_cache_key(name: str, args: Dict[str, Any] | None) -> str:
    """Build a stable cache key from a tool call's canonical name and args.

    Canonical-name lookup (`_CLIENT_TOOL_NAME_MAP`) folds Pascal/snake-case
    aliases together so ``WebSearch(query=q)`` and ``web_search(query=q)``
    share the same cache slot.  Args are JSON-encoded with sorted keys for
    determinism.
    """
    canonical = _CLIENT_TOOL_NAME_MAP.get(name, name)
    try:
        args_str = json.dumps(args or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args_str = repr(args or {})
    return f"{canonical}|{args_str}"


def dedupe_tool_calls(calls: List[ToolCall]) -> List[ToolCall]:
    """Drop duplicate ToolCalls (same canonical name + same args) preserving order.

    The first occurrence wins; subsequent matches are dropped.  Used by
    ``ServerToolNode`` to avoid firing the same `web_search` twice when a
    model emits the call multiple times in one assistant turn.
    """
    seen: set[str] = set()
    out: List[ToolCall] = []
    for tc in calls:
        key = tool_call_cache_key(tc.name, tc.args)
        if key in seen:
            continue
        seen.add(key)
        out.append(tc)
    return out

# Server tool type patterns — versioned tool types that are normally
# executed by an upstream API. We intercept and execute locally.
_SERVER_TOOL_TYPE_PATTERNS = [
    re.compile(r"^web_search_\d+$"),
    re.compile(r"^web_fetch_\d+$"),
    re.compile(r"^text_editor_\d+$"),
    re.compile(r"^bash_\d+$"),
    re.compile(r"^computer_\d+$"),
    re.compile(r"^code_execution_\d+$"),
]

# Tool names we can actually execute server-side
_EXECUTABLE_SERVER_TOOLS = {"web_search", "web_fetch"}

# Mapping from client tool names (as sent by Claude Code) to our local
# execution functions.  Claude Code sends PascalCase client tools like
# "WebSearch" / "WebFetch" that wrap what the Anthropic API normally
# handles as server-side tools.  We detect them by name and execute
# them locally using SearxNG / web reader.
_CLIENT_TOOL_NAME_MAP: Dict[str, str] = {
    # PascalCase names from Claude Code
    "WebSearch": "web_search",
    "WebFetch": "web_fetch",
    # snake_case names from Anthropic server tools
    "web_search": "web_search",
    "web_fetch": "web_fetch",
}


def is_server_tool(tool_dict: Dict[str, Any]) -> bool:
    """Check if a tool definition is a server-side tool based on its type."""
    tool_type = tool_dict.get("type", "")
    # Server tools have versioned type strings like "web_search_20250305"
    # Client tools have type "custom" or no type
    if tool_type in ("custom", "", None):
        return False
    return any(p.match(tool_type) for p in _SERVER_TOOL_TYPE_PATTERNS)


def separate_server_tools(
    tools: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate tools into client tools and server tools.

    Returns:
        (client_tools, server_tools) tuple
    """
    client_tools = []
    server_tools = []
    for tool in tools:
        if is_server_tool(tool):
            server_tools.append(tool)
        else:
            client_tools.append(tool)
    return client_tools, server_tools


def get_server_tool_names(server_tools: List[Dict[str, Any]]) -> set[str]:
    """Get the names of server-side tools."""
    names = set()
    for tool in server_tools:
        name = tool.get("name")
        if name:
            names.add(name)
        else:
            # Derive name from type (e.g., "web_search_20250305" -> "web_search")
            tool_type = tool.get("type", "")
            base_name = re.sub(r"_\d+$", "", tool_type)
            if base_name:
                names.add(base_name)
    return names


def find_locally_executable_tools(
    tools: List[Dict[str, Any]],
) -> set[str]:
    """Find tools that can be executed locally based on their name.

    Claude Code sends client tools (type "custom") named "WebSearch" and
    "WebFetch" that wrap what the Anthropic API handles server-side.  This
    function detects them by name so we can intercept their tool calls.

    A tool may opt out of server-side execution by setting
    ``{"execute": "client"}`` (or ``client_side`` / ``client-side``) in
    the tool definition — in that case the API leaves the tool entirely
    to the client, even when the name otherwise matches.

    Returns:
        Set of tool names that should be executed locally.
    """
    names = set()
    for tool in tools:
        name = tool.get("name", "")
        execute = tool.get("execute")
        if isinstance(execute, str) and execute.lower() in _CLIENT_EXEC_FLAG_VALUES:
            continue
        if name in _CLIENT_TOOL_NAME_MAP:
            names.add(name)
    return names


def make_server_tool_definitions(
    server_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert server tool defs into client-style tool definitions with input schemas.

    Server tools don't include input_schema since they are normally executed
    by the upstream API. We create proper tool definitions so bind_tools()
    can make the local model aware of them.
    """
    definitions = []
    for tool in server_tools:
        name = tool.get("name") or re.sub(r"_\d+$", "", tool.get("type", ""))
        if name == "web_search":
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": (
                            "Search the web for current information. Returns a list of "
                            "search results with titles, URLs, and content snippets."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to execute",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            )
        elif name == "web_fetch":
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_fetch",
                        "description": (
                            "Fetch and read content from a specific web page URL. "
                            "Returns the text content of the page."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "The URL to fetch content from",
                                },
                            },
                            "required": ["url"],
                        },
                    },
                }
            )
        else:
            logger.debug(f"Skipping unsupported server tool: {name}")
    return definitions


def extract_server_tool_calls(
    tool_calls: List[ToolCall],
    server_tool_names: set[str],
) -> Tuple[List[ToolCall], List[ToolCall]]:
    """Split tool calls into server-side and client-side.

    Returns:
        (server_tool_calls, client_tool_calls) tuple
    """
    server_calls = []
    client_calls = []
    for tc in tool_calls:
        if tc.name in server_tool_names:
            server_calls.append(tc)
        else:
            client_calls.append(tc)
    return server_calls, client_calls


async def execute_server_tool(tool_call: ToolCall) -> str:
    """Execute a single server-side tool call and return the result as a string.

    Handles both snake_case names (web_search, web_fetch) from Anthropic
    server tools and PascalCase names (WebSearch, WebFetch) from Claude Code
    client tools.
    """
    name = tool_call.name
    args = tool_call.args

    # Normalize to canonical name via the mapping
    canonical = _CLIENT_TOOL_NAME_MAP.get(name, name)

    if canonical == "web_search":
        return await _execute_web_search(args)
    elif canonical == "web_fetch":
        return await _execute_web_fetch(args)
    else:
        return f"Error: Unknown server tool '{name}'"


async def _execute_web_search(args: Dict[str, Any]) -> str:
    """Execute web_search via the mcp-server-web MCP server, falling back
    to the inline SearxNG implementation when no MCP URL is configured."""
    query = args.get("query", "")
    if not query:
        return "Error: No search query provided"

    logger.info(f"🔍 Executing server-side web_search: {query}")
    timeout = _TOOL_TIMEOUTS["web_search"]

    # Prefer the deployed MCP server.  Its JSON envelope already matches
    # the shape ``_format_search_result`` expects (``contents``: [...]),
    # so we just need to call the tool and reformat the result.
    from tools.mcp_client import get_default, MCPCallError  # pylint: disable=import-outside-toplevel

    mcp = get_default()
    if mcp is not None:
        try:
            raw = await asyncio.wait_for(
                mcp.call_tool("web_search", {"query": query}), timeout=timeout
            )
            logger.info(f"✅ Web search via MCP completed for: {query}")
            return _format_search_result(raw, query)
        except asyncio.TimeoutError:
            error_msg = f"Web search (MCP) timed out after {timeout:.0f}s"
            logger.warning(error_msg, query=query)
            return f"Error: {error_msg}"
        except MCPCallError as e:
            # Surface as inline error — the model can recover or retry.
            error_msg = f"Web search (MCP) failed: {e}"
            logger.warning(error_msg, query=query)
            return f"Error: {error_msg}"
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "MCP web_search raised unexpected error; falling back to inline",
                error=str(e),
                query=query,
            )
            # fall through to inline path below

    # Inline fallback (no MCP configured, or MCP raised unexpectedly).
    from tools.static.web_search_tool import (  # pylint: disable=import-outside-toplevel
        web_search,
    )

    try:
        result = await asyncio.wait_for(
            web_search.ainvoke({"query": query}), timeout=timeout
        )
        logger.info(f"✅ Web search (inline) completed for: {query}")
        return _format_search_result(result, query)
    except asyncio.TimeoutError:
        error_msg = f"Web search timed out after {timeout:.0f}s"
        logger.warning(error_msg, query=query)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Web search failed: {str(e)}"
        logger.error(error_msg, query=query)
        return f"Error: {error_msg}"


def _format_search_result(raw: str, query: str) -> str:
    """Convert raw JSON search results into concise text for the model.

    This dramatically reduces token usage vs dumping the full JSON, and
    gives the model a cleaner signal it can act on.
    """
    import json as _json  # pylint: disable=import-outside-toplevel

    try:
        data = _json.loads(raw)
    except (ValueError, TypeError):
        # Not JSON — return as-is (already a string)
        return raw

    contents = data.get("contents", [])
    if not contents:
        error = data.get("error")
        return f"No search results found for: {query}" + (
            f" ({error})" if error else ""
        )

    lines = [f"Search results for: {query}\n"]
    for item in contents:
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("content", "")
        lines.append(f"- {title}")
        if url:
            lines.append(f"  {url}")
        if snippet:
            lines.append(f"  {snippet}")
        lines.append("")

    return "\n".join(lines).strip()


async def _execute_web_fetch(args: Dict[str, Any]) -> str:
    """Execute web_fetch via the mcp-server-web MCP server, falling back
    to the inline web-reader implementation when no MCP URL is configured.

    The MCP server's tool is named ``fetch_page`` (vs the API's
    ``web_fetch``); we translate transparently."""
    url = args.get("url", "")
    if not url:
        return "Error: No URL provided"

    logger.info(f"📖 Executing server-side web_fetch: {url}")
    timeout = _TOOL_TIMEOUTS["web_fetch"]

    from tools.mcp_client import get_default, MCPCallError  # pylint: disable=import-outside-toplevel

    mcp = get_default()
    if mcp is not None:
        try:
            result = await asyncio.wait_for(
                mcp.call_tool("fetch_page", {"url": url}), timeout=timeout
            )
            logger.info(f"✅ Web fetch via MCP completed for: {url}")
            return result
        except asyncio.TimeoutError:
            error_msg = f"Web fetch (MCP) timed out after {timeout:.0f}s"
            logger.warning(error_msg, url=url)
            return f"Error: {error_msg}"
        except MCPCallError as e:
            error_msg = f"Web fetch (MCP) failed: {e}"
            logger.warning(error_msg, url=url)
            return f"Error: {error_msg}"
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "MCP fetch_page raised unexpected error; falling back to inline",
                error=str(e),
                url=url,
            )
            # fall through to inline path below

    from tools.static.web_reader_tool import (  # pylint: disable=import-outside-toplevel
        read_web_content,
    )

    try:
        result = await asyncio.wait_for(
            read_web_content.ainvoke({"url": url}), timeout=timeout
        )
        logger.info(f"✅ Web fetch (inline) completed for: {url}")
        return str(result)
    except asyncio.TimeoutError:
        error_msg = f"Web fetch timed out after {timeout:.0f}s"
        logger.warning(error_msg, url=url)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Web fetch failed: {str(e)}"
        logger.error(error_msg, url=url)
        return f"Error: {error_msg}"


async def execute_server_tool_calls(
    tool_calls: List[ToolCall],
) -> List[Message]:
    """Execute multiple server-side tool calls and return tool result messages.

    Returns a list of TOOL role messages, one per tool call, suitable for
    appending to the conversation and re-running the model. The execution_id
    is preserved in tool_calls so the message conversion layer can match
    ToolMessages to AIMessage tool_calls.
    """
    results = []
    for tc in tool_calls:
        result_text = await execute_server_tool(tc)
        results.append(
            Message(
                role=MessageRole.TOOL,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=result_text,
                    )
                ],
                # Store execution_id in tool_calls so message_conversion can
                # extract it as the LangChain ToolMessage.tool_call_id
                tool_calls=[
                    ToolCall(
                        name=tc.name,
                        args=tc.args,
                        execution_id=tc.execution_id,
                    )
                ],
            )
        )
    return results
