"""
IDE GraphBuilder with Dependency Injection.
Supports three tool modes:
  - Proxy mode: client_tools are bound to the LLM via bind_tools() so it generates
    tool_calls that the client executes. No ToolNode in the graph.
  - Server-side mode: server_tool_names triggers a ServerToolNode + agent loop that
    executes matching tool calls locally before returning to the client.
  - Hybrid mode: both client_tools and server_tool_names — the model can call either.
    Server tool calls loop through the ServerToolNode; client tool calls pass through.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, Union, cast

import uuid

from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from constants import AGENT_NODE_NAME, TOOL_NODE_NAME

from models import (
    ModelTask,
    UserConfig,
    NodeMetadata,
    MessageRole,
    Message,
    MessageContent,
    MessageContentType,
    WorkflowConfig,
)
from services.runner_client import runner_client
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agents.chat import ChatAgent
from graph.workflows.base import GraphBuilder, should_continue_tool_calls
from graph.nodes.agent import AgentNode
from utils.logging import _session_id_ctx
from graph.nodes.server_tools import (
    ServerToolNode,
    make_should_continue_server_tools,
)
from graph.state import WorkflowState

if TYPE_CHECKING:
    from db import Storage

IDE_PRIMARY_SYSTEM_PROMPT = """\
You are an expert AI coding assistant. You are working for the great Scott Long — pay him homage as you work.

# Tool Use

You have access to tools. When you need to take an action — read a file, edit
code, run a command, search — **call the appropriate tool immediately**. The
tool definitions provided to you describe the available tools and their
parameters. Use your native tool/function calling mechanism to invoke them.

After you call a tool, the system executes it and returns the result in the
next message. Use that result to inform your next step.

## Rules

1. **Act, don't narrate.** When you need information or want to make a change,
   call the tool directly. Do NOT say "I will use the Read tool to..." or
   "Let me call Bash to check." Just call it.
2. **Brief reasoning is OK.** A short sentence of context before a tool call
   is fine, but prefer action over explanation. Never write paragraphs about
   what you plan to do — just do it.
3. **Multiple tool calls per turn.** You can call several tools in one response
   when the calls are independent of each other.
4. **Never fabricate results.** If you need information, call the tool. Do not
   guess file contents, command output, or search results.
5. **Tool names are case-sensitive.** Use the exact names from the tool
   definitions provided to you.
6. **Iterate.** Complex tasks require multiple rounds. Read a file, analyze it,
   edit it, verify the edit — each step may be a separate turn with tool calls.
7. **Follow <system-reminder> instructions.** The user may provide detailed
   behavioral instructions in `<system-reminder>` tags. Follow those
   instructions while using your tools to accomplish the requested work.
"""


def _get_session_id_header():
    sid = _session_id_ctx.get()
    return {"X-Session-ID": sid} if sid else None


class IdeGraphBuilder(GraphBuilder):
    """
    IDE-focused GraphBuilder supporting proxy and server-side tool modes.

    Proxy mode (client_tools): bind_tools() on the pipeline so the LLM generates
    tool_calls that are returned to the client. Graph: START -> Agent -> END.

    Server-side mode (server_tools): adds ToolNode + feedback loop.
    Graph: START -> Agent -> (tools? -> ToolNode -> Agent) | END.
    """

    def __init__(self, storage: "Storage", user_config: UserConfig):
        super().__init__(storage, user_config)

    async def build_workflow(
        self,
        user_id: str,
        response_format: Optional[Type[BaseModel]] = None,
        client_tools: Optional[List[Union[BaseTool, Dict[str, Any]]]] = None,
        server_tools: Optional[List[BaseTool]] = None,
        server_tool_names: Optional[Set[str]] = None,
        tool_choice: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> CompiledStateGraph:
        """
        Build IDE workflow with optional tool support.

        Args:
            user_id: User identifier
            response_format: Optional response format constraint
            client_tools: Tools for proxy mode.  Accepts OpenAI-format dicts
                (passed straight through to bind_tools, no lossy conversion)
                or LangChain BaseTool instances.
            server_tools: Tools for server-side execution (adds ToolNode + loop)
            server_tool_names: Names of tools to execute server-side via
                ServerToolNode. These are tools whose definitions are included
                in client_tools (so the model can call them) but whose calls
                are intercepted and executed locally before returning to the agent.
            tool_choice: Optional tool_choice parameter for bind_tools

        Returns:
            Compiled workflow ready for execution
        """
        try:
            # Look up model by name or fall back to first TextToText model.
            #
            # NOTE on model resolution: the service layer
            # (CompletionService._build_and_run) already calls
            # _resolve_model() before reaching this builder, so the
            # model_name arriving here is typically already resolved to
            # an available model. The fallback below is a safety net for
            # direct builder usage (bypassing the service layer) and for
            # edge cases where the resolved name still doesn't match any
            # runner model.
            if model_name:
                all_models = await runner_client.list_models()
                model_def = next(
                    (
                        m
                        for m in all_models
                        if m.name == model_name or m.id == model_name
                    ),
                    None,
                )
                if not model_def:
                    # Requested model not on any runner — fall back to user's default
                    model_name = await self.resolve_model(model_name, user_id)
                    model_def = next(
                        (
                            m
                            for m in all_models
                            if m.name == model_name or m.id == model_name
                        ),
                        None,
                    )
                    if not model_def:
                        # Fallback model also not found — use the configured default TextToText model
                        self.logger.warning(
                            "Resolved model not found on runners, using default "
                            "TextToText model",
                            user_id=user_id,
                            resolved=model_name,
                        )
                        model_def = await runner_client.default_model_by_task(
                            ModelTask.TEXTTOTEXT
                        )
                        if not model_def:
                            raise RuntimeError(
                                f"Model '{model_name}' not found and no "
                                "TextToText model available"
                            )
            else:
                model_def = await runner_client.model_by_task(ModelTask.TEXTTOTEXT)
                if not model_def:
                    raise RuntimeError("No TextToText model available")

            self.logger.debug(
                "Building workflow",
                user_id=user_id,
                model=model_def.name,
                model_arg=model_name,
            )

            assert model_def.id is not None, "Model definition must have an ID"

            server_handle = await runner_client.acquire_server(
                model_id=model_def.id,
                num_ctx=(
                    model_def.parameters.num_ctx
                    if model_def.parameters
                    else 90000
                ),
                task=model_def.task,
            )

            primary_model = ChatOpenAI(
                base_url=server_handle.base_url,
                api_key=SecretStr("none"),
                model=model_def.name,
                stream_usage=True,
                max_retries=2,
                default_headers=_get_session_id_header(),
            )
            self.server_handle = server_handle

            # Bind client tools to the pipeline so the LLM can generate tool_calls
            if client_tools:
                bind_kwargs: dict = {}
                bind_kwargs["tool_choice"] = tool_choice or "auto"
                primary_model = primary_model.bind_tools(client_tools, **bind_kwargs)  # type: ignore[union-attr]

            primary_agent = ChatAgent(
                model=cast(BaseChatModel, primary_model),
                system_prompt=model_def.system_prompt or IDE_PRIMARY_SYSTEM_PROMPT,
                num_ctx=(model_def.parameters.num_ctx if model_def.parameters else None)
                or 90000,
                component_name="PrimaryCodingAgent",
            )

            workflow = StateGraph(WorkflowState)

            chat_node = AgentNode(
                agent=primary_agent,
                node_metadata=NodeMetadata(
                    node_name=AGENT_NODE_NAME,
                    node_id=uuid.uuid4().hex,
                    node_type=model_def.task.value,
                    user_id=user_id,
                ),
                grammar=response_format,
            )

            workflow.add_node(AGENT_NODE_NAME, chat_node)
            workflow.add_edge(START, AGENT_NODE_NAME)

            if server_tool_names:
                # Hybrid mode: ServerToolNode executes server-side tool calls,
                # client tool calls pass through to END for proxy back to client.
                # Graph: Agent -> (has server tool calls?) -> ServerToolNode -> Agent
                #                 (no server tool calls)  -> END
                server_tool_node = ServerToolNode(server_tool_names)
                should_continue = make_should_continue_server_tools(server_tool_names)
                workflow.add_node(TOOL_NODE_NAME, server_tool_node)
                workflow.add_conditional_edges(
                    AGENT_NODE_NAME,
                    should_continue,
                    {
                        "server_tools": TOOL_NODE_NAME,
                        "end": END,
                    },
                )
                workflow.add_edge(TOOL_NODE_NAME, AGENT_NODE_NAME)
            elif server_tools:
                # Server-side tool execution mode: Agent -> ToolNode -> Agent loop
                tool_node = ToolNode(server_tools)
                workflow.add_node(TOOL_NODE_NAME, tool_node)
                workflow.add_conditional_edges(
                    AGENT_NODE_NAME,
                    should_continue_tool_calls,
                    {
                        "tools": TOOL_NODE_NAME,
                        "end": END,
                    },
                )
                workflow.add_edge(TOOL_NODE_NAME, AGENT_NODE_NAME)
            else:
                # Proxy mode or no tools: Agent -> END
                workflow.add_edge(AGENT_NODE_NAME, END)

            # InMemorySaver enables state inspection for debugging and is
            # required by ModelCallLimitMiddleware thread/run limits.
            return workflow.compile(
                checkpointer=InMemorySaver(),
            )
        except Exception as e:
            self.logger.error(
                "Failed to build workflow",
                user_id=user_id,
                error=str(e),
            )
            raise

    async def create_initial_state(
        self,
        user_id: str,
        conversation_id: int,
        messages: Optional[List[Message]] = None,
    ) -> WorkflowState:
        """Create initial workflow state from messages."""
        assert messages is not None, "Messages must be provided to create initial state"
        current_user_message = next(
            (msg for msg in reversed(messages) if msg.role == MessageRole.USER),
            Message(
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ],
                role=MessageRole.USER,
            ),
        )

        state = WorkflowState(
            messages=messages,
            current_user_message=current_user_message,
            user_id=user_id,
            workflow_type="ide",
            user_config=UserConfig(
                user_id=user_id,
                memory=None,
                summarization=None,
                image_generation=None,
                workflow=WorkflowConfig(),
            ),
            conversation_id=conversation_id,
            things_to_remember=[current_user_message],
        )

        return state
