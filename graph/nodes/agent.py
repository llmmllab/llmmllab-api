"""
Agent node for workflow execution.
Executes the chat agent with optional tool support.
"""

from typing import Awaitable, Callable, Optional, Type
from pydantic import BaseModel

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig

from tools.registry import ToolRegistry
from agents.chat import ChatAgent
from graph.state import WorkflowState
from constants import AGENT_NODE_NAME, STRUCTURED_AGENT_RUNNABLE_NAME

from models import NodeMetadata, Message, MessageRole
from utils.logging import llmmllogger


def _disconnected_from_config(
    config: Optional[RunnableConfig],
) -> Optional[Callable[[], Awaitable[bool]]]:
    """Pull the client-disconnect predicate out of the runnable config.

    The streaming routers build an ``async () -> bool`` predicate from the
    FastAPI ``Request.is_disconnected()`` and thread it down to the executor,
    which stashes it in ``RunnableConfig.configurable["disconnected"]`` (the
    only carrier that survives LangGraph state serialization — a callable
    can't live on the ``model_dump()``'d ``WorkflowState``).  LangGraph
    injects that same config into node callables that declare a ``config``
    parameter, so this is where the predicate re-enters the agent stack.

    Returns ``None`` (predicate absent / malformed) so non-streaming and
    other callers that never set it are completely unaffected — the agent's
    ``run``/``run_structured`` treat a ``None`` predicate as "still
    connected" and behave exactly as before.
    """
    if not config:
        return None
    configurable = config.get("configurable") or {}
    predicate = configurable.get("disconnected")
    return predicate if callable(predicate) else None


class AgentNode:
    """
    Executes the chat agent with optional tool support.

    When tool_registry is provided, tools are passed to the agent for tool-calling.
    When tool_registry is None, the agent runs without tools (passthrough mode).
    """

    def __init__(
        self,
        agent: ChatAgent,
        node_metadata: NodeMetadata,
        tool_registry: Optional[ToolRegistry] = None,
        grammar: Optional[Type[BaseModel]] = None,
    ):
        self.agent = agent.bind_node_metadata(node_metadata)
        self.logger = llmmllogger.bind(component=AGENT_NODE_NAME)
        self.tool_registry = tool_registry
        self.grammar = grammar

    async def __call__(
        self,
        state: WorkflowState,
        config: Optional[RunnableConfig] = None,
    ) -> WorkflowState:
        """
        Execute the agent node.

        Args:
            state: Current workflow state
            config: LangGraph-injected runnable config.  Carries the
                client-disconnect predicate under
                ``configurable["disconnected"]`` when the request came from a
                streaming endpoint; LangGraph populates this argument because
                it is named ``config`` and typed ``RunnableConfig``.  Absent
                (None) for every non-streaming / internal caller, leaving the
                agent run unchanged.

        Returns:
            Updated workflow state with agent response
        """
        assert state.conversation_id is not None
        # Client-liveness predicate (or None).  Passing it into the agent's
        # run/run_structured lets the retry/backoff loops abort promptly with
        # CancelledError once the IDE client has hung up — the fix for zombie
        # IDE coding-agent sessions that kept re-dispatching to the runner
        # long after the client disconnected.
        disconnected = _disconnected_from_config(config)
        try:
            tools = (
                self.tool_registry.get_all_executable_tools()
                if self.tool_registry
                else None
            )

            if self.grammar:
                self.logger.info("Using structured output grammar for agent response")
                structured_response = await self.agent.run_structured(
                    message_input=state.messages,
                    tools=tools,
                    grammar=self.grammar,
                    disconnected=disconnected,
                )

                runnable = RunnableLambda(
                    lambda x: x, name=STRUCTURED_AGENT_RUNNABLE_NAME
                )

                self.logger.debug(
                    f"Structured response from agent: {structured_response.model_dump_json(warnings=False)}"
                )

                runnable.invoke(structured_response)

                state.messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[],
                        structured_output=structured_response.model_dump(
                            warnings=False
                        ),
                    )
                )
            else:
                response = await self.agent.run(
                    messages=state.messages,
                    tools=tools,
                    disconnected=disconnected,
                )

                if response.message:
                    if response.message.tool_calls:
                        self.logger.info(
                            f"Generated {len(response.message.tool_calls)} tool calls"
                        )
                    state.messages.append(response.message)

            return state

        except Exception as e:
            self.logger.error(
                "Chat Agent failed",
                extra={
                    "user_id": getattr(state, "user_id", "unknown"),
                    "error": str(e),
                },
            )
            raise
