# IDE Workflow

The IDE workflow (`graph/workflows/ide/builder.py`) builds LangGraph-based agent workflows for coding/IDE tasks. It supports three tool execution modes and integrates with the runner infrastructure for model selection.

## Architecture

```
CompletionService._build_and_run
  → _resolve_model()          # Service-layer model resolution (primary)
  → CompletionService.build_workflow()
    → compose_workflow()      # Caching layer
      → IdeGraphBuilder.build_workflow()
        → runner_client.list_models()   # Look up model on runners
        → self.resolve_model()          # Builder-level fallback (safety net)
        → runner_client.acquire_server() # Acquire model server
        → Build LangGraph workflow
```

## Model Resolution

Model resolution happens at two levels:

1. **Service layer** (`CompletionService._build_and_run`): Calls `_resolve_model()` which checks if the requested model is available on any runner. If not, falls back to the user's `default_model`. This is the primary resolution path.

2. **Builder layer** (`IdeGraphBuilder.build_workflow`): After receiving the (already-resolved) model name, looks it up in `runner_client.list_models()`. If the model still isn't found (edge case or direct builder usage), calls `self.resolve_model()` as a safety net.

The builder-level resolution is a **safety net** for:
- Direct builder usage (bypassing the service layer)
- Edge cases where the resolved name still doesn't match any runner model

## Tool Modes

### Proxy Mode (client_tools only)
- Tools are bound to the LLM via `bind_tools()` so it generates `tool_calls`
- Tool calls are returned to the client for execution
- Graph: `START → Agent → END`

### Server-Side Mode (server_tools)
- Tools execute locally via `ToolNode`
- Graph: `START → Agent → (tools? → ToolNode → Agent) | END`

### Hybrid Mode (server_tool_names)
- Model can call both client tools and server-side tools
- Server tool calls are intercepted and executed locally via `ServerToolNode`
- Client tool calls pass through to END for proxy back to client
- Graph: `START → Agent → (has server tool calls? → ServerToolNode → Agent) | END`

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `IdeGraphBuilder` | `graph/workflows/ide/builder.py` | Builds IDE workflow graphs |
| `GraphBuilder` (base) | `graph/workflows/base.py` | Shared DI setup, `resolve_model()` |
| `AgentNode` | `graph/nodes/agent.py` | Executes agent turns |
| `ServerToolNode` | `graph/nodes/server_tools.py` | Executes server-side tool calls |
| `WorkflowState` | `graph/state.py` | Shared state across workflow nodes |

## Configuration

- `model_name`: Optional model override. Falls back to first TextToText model if not specified.
- `client_tools`: OpenAI-format tool dicts or LangChain `BaseTool` instances for proxy mode.
- `server_tools`: LangChain `BaseTool` instances for server-side execution.
- `server_tool_names`: Set of tool names to execute server-side (hybrid mode).
- `tool_choice`: Optional tool_choice parameter (`"auto"`, `"none"`, or specific tool name).
- `response_format`: Optional Pydantic model for structured output.

## Testing

Unit tests are in `test/unit/test_ide_builder.py`:
- Model resolution (found by name/id, fallback, error cases)
- Tool mode graph construction (proxy, server-side, hybrid)
- Initial state creation

Run: `uv run pytest test/unit/test_ide_builder.py -v`
