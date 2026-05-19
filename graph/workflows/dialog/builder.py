"""
Dialog GraphBuilder — conversational workflow with memory + title generation.

Uses dependency injection: all agent, storage, and node instances are
constructed up front and wired into the graph. Shared model-resolution and
primary-agent construction live on the ``GraphBuilder`` base class.
"""

from typing import Optional, Type, cast
import uuid

from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

import config
from constants import (
    AGENT_NODE_NAME,
    MEMORY_CREATE_NODE_NAME,
    MEMORY_SEARCH_NODE_NAME,
    MEMORY_STORE_NODE_NAME,
    TITLE_GENERATION_NODE_NAME,
    TOOL_NODE_NAME,
)
from models import (
    ModelTask,
    NodeMetadata,
    MessageRole,
    Message,
    MessageContent,
    MessageContentType,
    UserConfig,
)
from services.runner_client import runner_client
from langchain_openai import OpenAIEmbeddings

from agents.chat import ChatAgent
from agents.embed import EmbeddingAgent
from graph.workflows.base import (
    GraphBuilder,
    should_continue_tool_calls,
    should_generate_title,
)
from graph.nodes.agent import AgentNode
from graph.nodes.memory import (
    MemorySearchNode,
    MemoryCreationNode,
    MemoryStorageNode,
)
from tools.registry import registry_manager
from graph.state import WorkflowState, assemble_context_messages


class DialogGraphBuilder(GraphBuilder):
    """
    Conversational GraphBuilder using dependency injection and composition.

    Responsibilities:
    - Resolve the primary text model + acquire its server (shared via base)
    - Resolve the embedding model + wire the embedding agent
    - Build memory search/create/store nodes
    - Compose the title-generation tail of the graph
    """

    # The base class reads ``runner_client`` and ``ChatAgent`` from this
    # module via ``sys.modules`` at call time, which lets test-time
    # monkeypatches on ``graph.workflows.dialog.builder.runner_client``
    # flow through the shared helpers correctly.
    _chat_openai_max_retries = config.CHAT_OPENAI_MAX_RETRIES

    def __init__(self, storage: "Storage", user_config: UserConfig):
        super().__init__(storage, user_config)

    async def build_workflow(
        self,
        user_id: str,
        response_format: Optional[Type[BaseModel]] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> CompiledStateGraph:
        """
        Build a dialog workflow.

        When *model_name* is provided (typically resolved at the router / service
        layer), the builder looks up that specific model on the runners.  When
        it is ``None``, the builder falls back to the first available
        ``TEXTTOTEXT`` model — this is the same default path the router uses
        when the client sends ``"model": "default"``.
        """
        try:
            # Two-phase primary resolution: resolve the model definition first,
            # then check the embedding model is available, only THEN acquire
            # the primary server. This preserves the original "raise before
            # acquire" ordering when no embedding model is available.
            primary_model_def = await self._resolve_primary_model(user_id, model_name)

            embedding_model_def = await runner_client.model_by_task(
                ModelTask.TEXTTOEMBEDDINGS
            )
            if not embedding_model_def:
                raise RuntimeError("No TextToEmbeddings model available")

            primary_agent, _ = await self._acquire_primary_server_and_agent(
                user_id=user_id,
                model_def=primary_model_def,
                system_prompt_default="",
                component_name="PrimaryChatAgent",
            )

            embedding_handle = await runner_client.acquire_server(
                model_id=embedding_model_def.id,
                task=embedding_model_def.task,
            )

            embedding_model = OpenAIEmbeddings(
                base_url=embedding_handle.base_url,
                api_key="none",
            )

            embedding_agent = EmbeddingAgent(
                model=cast(Embeddings, embedding_model),
                component_name="EmbeddingAgent",
            )

            # Create nodes with injected agents and storage
            memory_creation_node = MemoryCreationNode(
                embedding_agent,
                NodeMetadata(
                    node_name="MemoryCreationNode",
                    node_id=uuid.uuid4().hex,
                    node_type=embedding_model_def.task.value,
                    user_id=user_id,
                ),
            )
            memory_search_node = MemorySearchNode(
                embedding_agent,
                self.memory_storage,
            )
            memory_storage_node = MemoryStorageNode(self.memory_storage)

            tool_registry = await registry_manager.get_user_registry(user_id)
            tools = tool_registry.get_all_executable_tools()

            tool_node = ToolNode(tools)

            # Create master workflow graph
            workflow = StateGraph(WorkflowState)

            # create nodes with injected dependencies
            chat_node = AgentNode(
                agent=primary_agent,
                tool_registry=tool_registry,
                node_metadata=NodeMetadata(
                    node_name=AGENT_NODE_NAME,
                    node_id=uuid.uuid4().hex,
                    node_type=primary_model_def.task.value,
                    user_id=user_id,
                ),
                grammar=response_format,
            )

            async def context_node(state: WorkflowState) -> WorkflowState:
                """Execute the context assembly subgraph and return updated state."""
                state.messages = assemble_context_messages(state)
                return state

            async def title_generation_node(state: WorkflowState) -> WorkflowState:
                """Generate and update conversation title if needed."""
                try:
                    # Check if we need to generate a title
                    if state.title and not state.title.startswith("New conversation"):
                        self.logger.debug(
                            f"Skipping title generation - conversation already has title: {state.title}"
                        )
                        return state

                    # Need at least 2 messages (user + assistant) for meaningful title
                    if not state.messages or len(state.messages) < 2:
                        self.logger.debug("Not enough messages for title generation")
                        return state

                    # Generate title using primary agent
                    self.logger.info(
                        f"Generating title for conversation {state.conversation_id}"
                    )
                    title = await primary_agent.generate_title(state.messages)

                    if title and title != "New Conversation":
                        # Update the state
                        state.title = title

                        # Persist to database
                        await self.conversation_storage.update_conversation_title(
                            title=title,
                            conversation_id=state.conversation_id,
                            user_id=state.user_id,
                        )
                        self.logger.info(
                            f"✓ Generated and saved title for conversation {state.conversation_id}: {title}"
                        )
                    else:
                        self.logger.warning(
                            f"Failed to generate valid title for conversation {state.conversation_id}"
                        )

                except Exception as e:
                    # Don't fail the workflow if title generation fails
                    self.logger.error(
                        f"Error generating title for conversation {state.conversation_id}: {e}",
                        exc_info=True,
                    )

                return state

            workflow.add_node("context_assembly", context_node)
            workflow.add_node(TITLE_GENERATION_NODE_NAME, title_generation_node)

            # Memory nodes with injected agents and storage
            workflow.add_node(MEMORY_SEARCH_NODE_NAME, memory_search_node)
            workflow.add_node(MEMORY_CREATE_NODE_NAME, memory_creation_node)
            workflow.add_node(MEMORY_STORE_NODE_NAME, memory_storage_node)
            workflow.add_node(AGENT_NODE_NAME, chat_node)
            workflow.add_node(TOOL_NODE_NAME, tool_node)
            # Build a simplified workflow graph structure:
            workflow.add_edge(START, MEMORY_SEARCH_NODE_NAME)
            workflow.add_edge(START, "context_assembly")

            workflow.add_edge("context_assembly", AGENT_NODE_NAME)
            workflow.add_edge(MEMORY_SEARCH_NODE_NAME, AGENT_NODE_NAME)
            # create conditional tool call loop
            workflow.add_conditional_edges(
                AGENT_NODE_NAME,
                should_continue_tool_calls,
                {
                    "tools": TOOL_NODE_NAME,
                    "end": MEMORY_CREATE_NODE_NAME,
                },
            )
            # Tool results flow back to agent for further processing
            workflow.add_edge(TOOL_NODE_NAME, AGENT_NODE_NAME)
            workflow.add_edge(MEMORY_CREATE_NODE_NAME, MEMORY_STORE_NODE_NAME)

            # Add conditional title generation after memory storage
            workflow.add_conditional_edges(
                MEMORY_STORE_NODE_NAME,
                should_generate_title,
                {
                    "generate_title": TITLE_GENERATION_NODE_NAME,
                    "skip_title": END,
                },
            )
            workflow.add_edge(TITLE_GENERATION_NODE_NAME, END)

            return workflow.compile()
        except Exception as e:
            self.logger.error(
                "Failed to build workflow",
                user_id=user_id,
                error=str(e),
            )
            # Try to create fallback chat workflow
            raise

    async def create_initial_state(
        self,
        user_id: str,
        conversation_id: int,
    ) -> WorkflowState:
        """Create initial workflow state from messages."""

        from services import (  # pylint: disable=import-outside-toplevel
            user_config_service,
            message_service,
            conversation_service,
            summary_service,
        )

        user_config = await user_config_service.get_user_config(user_id)

        messages = await message_service.get_conversation_history(conversation_id)

        conversation = await conversation_service.get_conversation(conversation_id)

        summaries = await summary_service.get_summaries_for_conversation(
            conversation_id
        )

        # WorkflowState expects Message objects, not BaseMessage objects
        # So we use the messages directly without LangChain conversion

        current_user_message = next(
            (msg for msg in reversed(messages) if msg.role == MessageRole.USER),
            Message(
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ],
                role=MessageRole.USER,
            ),
        )

        # Create the state with centralized user configuration and todo context
        state = WorkflowState(
            title=(
                conversation.title
                if (
                    conversation
                    and not conversation.title.startswith("New conversation")
                )
                else None
            ),
            messages=messages,  # Use Message objects directly
            summaries=summaries,
            current_user_message=current_user_message,  # Use Message object directly
            user_id=user_id,
            workflow_type="dialog",
            user_config=user_config,
            conversation_id=conversation_id,
            things_to_remember=[current_user_message],
        )

        return state
