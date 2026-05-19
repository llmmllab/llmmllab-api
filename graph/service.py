"""
Main ComposerService orchestrator.
Central to the redesign - serves as the primary, authoritative execution runtime.

Configuration Management:
- Configuration overrides and default merging happens at the data layer
- Configuration is NOT passed as arguments in composer components
- Allowed arguments: user_id, messages/query, tools, workflow_type
- Components retrieve configuration from shared data layer using user_id
- No configuration merging logic should exist in service layer components
"""

import hashlib
from typing import Any, Dict, List, Optional, Type

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
)

from graph.cache_utils import _tools_cache_key
from graph.workflows.base import GraphBuilder
from graph.state import WorkflowState
from graph.cache import WorkflowCache
from graph.executor import WorkflowExecutor
from utils.logging import llmmllogger


class ComposerService:
    """
    Main composer service coordinating graph construction and execution.

    The Composer is responsible for:
    - Graph construction & execution
    - Streaming orchestration
    - State management
    - Tool management
    - Intent analysis
    - Error resiliency
    - Multi-agent orchestration
    """

    def __init__(self, builder: GraphBuilder):
        self.logger = llmmllogger.bind(component="ComposerService")
        self.graph_builder = builder
        # Workflow cache is now created per-user during workflow composition
        self.workflow_caches: Dict[str, WorkflowCache] = {}
        # Generic workflow executor for streaming
        self.executor = WorkflowExecutor()

    async def compose_workflow(
        self,
        user_id: str,
        model_name: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **build_kwargs,
    ) -> CompiledStateGraph:
        """
        Construct or retrieve a master workflow with intelligent routing.

        The workflow will handle intent analysis, tool selection, and routing
        internally using LangGraph's native capabilities.

        args:
            user_id: User ID for configuration retrieval

        returns:
            CompiledStateGraph: Master workflow with intelligent routing
        """
        try:
            # 1. Get user configuration from shared data layer
            from services import (
                user_config_service,
            )  # pylint: disable=import-outside-toplevel

            user_config = await user_config_service.get_user_config(user_id)

            # 2. Use per-user cache if enabled
            user_cache = None

            # Build cache key that incorporates model + session.
            #
            # Session is part of the key because GraphBuilder constructs
            # ``ChatOpenAI(..., default_headers={"X-Session-ID": sid})`` at
            # workflow build time — the session_id is captured statically
            # from the contextvar.  If we cached by (user, model) alone,
            # two distinct claude-code sessions for the same user on the
            # same model would share one cached CompiledStateGraph and
            # the second session's traffic would carry the first session's
            # X-Session-ID header all the way to the runner, contaminating
            # the slot LRU and trashing each other's KV cache.
            #
            # Including the session_id keeps each session's ChatOpenAI
            # binding isolated.  Memory cost: one CompiledStateGraph per
            # active (user, model, session) tuple — bounded by
            # WorkflowCache.max_size (1000) and TTL'd at 5 min.
            from utils.logging import _session_id_ctx
            sid = _session_id_ctx.get() or "nosession"

            cache_key = f"workflow_{user_id}"
            if model_name:
                cache_key += f"_{model_name}"
            cache_key += f"_{sid}"

            if user_config.workflow.enable_workflow_caching:
                if user_id not in self.workflow_caches:
                    self.workflow_caches[user_id] = WorkflowCache()
                user_cache = self.workflow_caches[user_id]

                # client_tools = build_kwargs.get("client_tools")
                # if client_tools:
                #     cache_key += f"_{_tools_cache_key(client_tools)}"

                # server_tool_names = build_kwargs.get("server_tool_names")
                # if server_tool_names:
                #     cache_key += f"_st{hashlib.md5(','.join(sorted(server_tool_names)).encode()).hexdigest()[:8]}"

                cached_workflow = await user_cache.get(cache_key)
                if cached_workflow:
                    self.logger.debug(
                        "Retrieved workflow from cache",
                        extra={"cache_key": cache_key},
                    )
                    return cached_workflow

            # 3. Build master workflow
            assert self.graph_builder is not None, "GraphBuilder should be initialized"

            workflow = await self.graph_builder.build_workflow(
                user_id, response_format, model_name=model_name, **build_kwargs
            )

            # Store in cache if caching is enabled
            if user_cache:
                await user_cache.set(cache_key, workflow)

            self.logger.info(
                "Master workflow composed successfully", extra={"user_id": user_id}
            )

            return workflow

        except Exception as e:
            self.logger.error(
                "Failed to compose master workflow",
                extra={"error": str(e), "user_id": user_id},
                exc_info=True,
            )
            raise

    def _build_cache_key(self, user_id: str, model_name: Optional[str]) -> str:
        """Build the same cache key used in ``compose_workflow``.

        Kept in one place so :meth:`invalidate_workflow` is guaranteed to
        match the key used when the entry was inserted.  Reads
        ``_session_id_ctx`` so the result matches the build-time key
        when invalidate is called from within the same request context
        (e.g. the stale-handle retry path in completion_service).
        """
        from utils.logging import _session_id_ctx
        sid = _session_id_ctx.get() or "nosession"

        cache_key = f"workflow_{user_id}"
        if model_name:
            cache_key += f"_{model_name}"
        cache_key += f"_{sid}"
        return cache_key

    async def invalidate_workflow(
        self, user_id: str, model_name: Optional[str] = None
    ) -> bool:
        """Purge the cached workflow for ``(user_id, model_name)``.

        Used when a ``StaleServerError`` confirms that the ``ServerHandle``
        baked into the cached workflow's ``ChatOpenAI(base_url=...)`` is
        pointing at an evicted/dead runner server.  The next ``compose_workflow``
        call will rebuild the workflow from scratch and acquire a fresh handle.

        Idempotent: returns ``False`` if no entry was present, ``True`` if an
        entry was evicted.  Thread-safe via the per-user ``WorkflowCache._lock``.
        """
        cache = self.workflow_caches.get(user_id)
        if cache is None:
            return False
        cache_key = self._build_cache_key(user_id, model_name)
        evicted = await cache.invalidate(cache_key)
        if evicted:
            self.logger.info(
                "Invalidated cached workflow due to stale server handle",
                extra={"user_id": user_id, "cache_key": cache_key},
            )
        return evicted

    async def shutdown(self):
        """Clean up resources on service shutdown."""
        self.logger.info("Shutting down ComposerService")

        # Close all per-user workflow caches
        for user_id, cache in self.workflow_caches.items():
            try:
                await cache.close()
            except Exception as e:
                self.logger.warning(f"Error closing cache for user {user_id}: {e}")
        self.workflow_caches.clear()
        # Close other resources as needed
