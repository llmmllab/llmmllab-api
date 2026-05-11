"""
Base GraphBuilder — shared DI setup for workflow subclasses.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Type

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from models import Message, UserConfig
from utils.logging import llmmllogger

from ..state import WorkflowState

if TYPE_CHECKING:
    from db import Storage
    from db.userconfig_storage import UserConfigStorage
    from db.conversation_storage import ConversationStorage
    from db.message_storage import MessageStorage
    from db.memory_storage import MemoryStorage
    from db.summary_storage import SummaryStorage
    from db.search_storage import SearchStorage
    from db.checkpoint_storage import CheckpointStorage
    from services.runner_client import ServerHandle


def should_continue_tool_calls(state: WorkflowState) -> str:
    """Route to tools if the last message has tool_calls, otherwise end."""
    if not state.messages:
        return "end"
    last = state.messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


def should_generate_title(state: WorkflowState) -> str:
    """Trigger title generation only on brand-new conversations."""
    if state.title is None or state.title.startswith("New conversation"):
        return "generate_title"
    return "skip_title"


class GraphBuilder(ABC):
    """
    Base class for workflow builders.

    Holds per-user storage handles and a logger. Subclasses implement
    `build_workflow` and `create_initial_state`.
    """

    server_handle: Optional["ServerHandle"] = None

    def __init__(self, storage: "Storage", user_config: UserConfig):
        self.user_config = user_config
        self.logger = llmmllogger.logger.bind(component=type(self).__name__)

        self.user_config_storage: "UserConfigStorage" = storage.get_service(
            storage.user_config
        )
        self.conversation_storage: "ConversationStorage" = storage.get_service(
            storage.conversation
        )
        self.message_storage: "MessageStorage" = storage.get_service(storage.message)
        self.memory_storage: "MemoryStorage" = storage.get_service(storage.memory)
        self.summary_storage: "SummaryStorage" = storage.get_service(storage.summary)
        self.search_storage: "SearchStorage" = storage.get_service(storage.search)
        self.checkpoint_storage: "CheckpointStorage" = storage.get_service(
            storage.checkpoint
        )

    async def resolve_model(
        self, requested_model: str, user_id: str
    ) -> str:
        """Resolve a model name, falling back to the user's default if unavailable.

        This centralises the fallback logic so every workflow gets consistent
        behaviour: if the requested model isn't on any runner, try the user's
        ``default_model`` before giving up.

        Returns
        -------
        str
            The resolved model ID (may be the original if no fallback exists).
        """
        from services.model_service import model_service

        return await model_service.resolve_default_model(requested_model, user_id)

    @abstractmethod
    async def build_workflow(
        self,
        user_id: str,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> CompiledStateGraph:
        """Build and compile a workflow graph."""

    @abstractmethod
    async def create_initial_state(
        self,
        user_id: str,
        conversation_id: int,
        messages: Optional[List[Message]] = None,
    ) -> WorkflowState:
        """Build the initial WorkflowState for execution."""
