"""
Composer Service Interface Layer.

Provides the public API boundary for the composer component, enabling other
services to interact with composer functionality while maintaining strict
architectural decoupling. This interface abstracts LangGraph workflow
construction, execution, and state management.

Interface Functions:
- get_composer_service(): Lazy accessor / initializer for the singleton service
- shutdown_composer(): Service lifecycle management
- compose_workflow(): Create executable LangGraph workflows using user_id and messages
- create_initial_state(): Generate workflow state from user_id and messages
- execute_workflow(): Stream-enabled workflow execution
- invalidate_workflow(): Purge a single cached workflow on stale-handle retries

Architectural Role:
- Defines clean API boundaries between components
- Abstracts internal composer implementation details
- Enables dependency injection for external services
- Maintains Protocol-based decoupling requirements
"""

from typing import AsyncIterator, Awaitable, Callable, List, Optional, Type, Union
import uuid
from pydantic import BaseModel
from models import ChatResponse, Message
from utils.logging import llmmllogger
from graph.service import CompiledStateGraph, ComposerService
from graph.errors import ComposerError
from graph.executor import stream_workflow
from graph.state import ServerToolEvent
from graph.workflows.base import GraphBuilder
from graph.workflows.factory import WorkFlowType, get_builder


_composer_service: Optional[ComposerService] = None


async def get_composer_service(
    builder: Optional[GraphBuilder] = None,
) -> ComposerService:
    """Get or lazily initialize the module-level ``ComposerService`` instance.

    A ``builder`` must be supplied on the first call so the service can be
    constructed.  Subsequent calls return the cached instance and ignore any
    ``builder`` argument.
    """
    global _composer_service
    if _composer_service is None:
        if builder is None:
            raise ComposerError(
                "WorkflowBuilder is required for composer service initialization."
            )
        llmmllogger.logger.info("Initializing composer service")
        _composer_service = ComposerService(builder)
        llmmllogger.logger.info("Composer service initialized")
    return _composer_service


async def shutdown_composer() -> None:
    """Shutdown the composer service. Should be called at server shutdown."""
    global _composer_service
    if _composer_service is not None:
        llmmllogger.logger.info("Shutting down composer service")
        await _composer_service.shutdown()
        _composer_service = None


async def compose_workflow(
    user_id: str,
    builder: GraphBuilder,
    model_name: Optional[str] = None,
    response_format: Optional[Type[BaseModel]] = None,
    **build_kwargs,
) -> CompiledStateGraph:
    """
    Compose a workflow for the given user and conversation messages.

    Args:
        user_id: User ID for configuration retrieval from shared data layer
        builder: GraphBuilder instance
        response_format: Optional response format constraint
        **build_kwargs: Additional keyword arguments passed to build_workflow
            (e.g., client_tools, tool_choice for IDE workflows)

    Returns:
        CompiledStateGraph: Ready to execute LangGraph workflow
    """
    svc = await get_composer_service(builder)
    return await svc.compose_workflow(
        user_id=user_id,
        model_name=model_name,
        response_format=response_format,
        **build_kwargs,
    )


async def invalidate_workflow(user_id: str, model_name: Optional[str] = None) -> bool:
    """Purge a single cached workflow for ``(user_id, model_name)``.

    Use this from the stale-server retry path so the next ``compose_workflow``
    call rebuilds the workflow and re-acquires a fresh ``ServerHandle`` instead
    of reusing the cached workflow whose ``ChatOpenAI(base_url=...)`` points at
    a dead runner.

    Returns ``True`` if an entry was evicted, ``False`` if no entry was present.
    Safe to call even if the composer service hasn't been initialized yet.
    """
    if _composer_service is None:
        return False
    return await _composer_service.invalidate_workflow(user_id, model_name)


async def clear_workflow_cache(user_id: str) -> None:
    """
    Clear the workflow cache for a specific user.

    Args:
        user_id: User ID whose workflow cache should be cleared
    """
    try:
        if _composer_service is None:
            return
        cache = _composer_service.workflow_caches.get(user_id, None)
        if cache:
            await cache.close()
    except ComposerError as e:
        llmmllogger.logger.error(
            f"Error clearing workflow cache for user {user_id}: {e}"
        )


async def create_initial_state(
    user_id: str,
    conversation_id: int,
    builder: GraphBuilder,
    messages: Optional[List[Message]] = None,
):
    """Create initial workflow state from user messages and configuration.

    Args:
        user_id: User ID for configuration retrieval from shared data layer
        messages: List of conversation messages
        workflow_type: Type of workflow
        additional_context: Optional additional context for state initialization

    Returns:
        WorkflowState: Initial state for workflow execution

    Note:
        User configuration is retrieved from shared data layer using user_id.
        No configuration objects should be passed as arguments (architectural rule).
    """
    return await builder.create_initial_state(user_id, conversation_id, messages)


async def execute_workflow(
    initial_state: BaseModel,
    workflow: CompiledStateGraph,
    disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
) -> AsyncIterator[Union[ChatResponse, ServerToolEvent]]:
    """
    Execute a compiled workflow with the given initial state.

    Args:
        workflow: CompiledStateGraph from compose_workflow()
        initial_state: WorkflowState from create_initial_state()
        disconnected: Optional ``async () -> bool`` client-liveness predicate
            threaded down to the agent node so the in-flight run aborts when
            the streaming HTTP client disconnects.  Defaults to ``None``
            (non-streaming callers), leaving execution unchanged.

    Yields:
        Dict containing workflow events (tokens, state updates, etc.)
    """
    async for event in stream_workflow(
        initial_state,
        workflow,
        thread_id=str(uuid.uuid4()),
        disconnected=disconnected,
    ):
        yield event


async def get_graph_builder(workflow_type: WorkFlowType, user_id: str) -> GraphBuilder:
    """Get the workflow builder instance. Should be implemented by external service."""
    return await get_builder(workflow_type, user_id)


# Convenience exports for direct usage
__all__ = [
    "get_composer_service",
    "shutdown_composer",
    "compose_workflow",
    "clear_workflow_cache",
    "create_initial_state",
    "execute_workflow",
    "invalidate_workflow",
    "get_graph_builder",
]
