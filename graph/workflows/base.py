"""
Base GraphBuilder — shared DI setup for workflow subclasses.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, cast

from langgraph.graph.state import CompiledStateGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from models import Message, ModelTask, UserConfig
from models.model_parameters import ModelParameters
from utils.logging import _session_id_ctx, llmmllogger

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
    from agents.chat import ChatAgent
    from models import Model


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


async def _inject_session_id_header(request) -> None:  # type: ignore[no-untyped-def]
    """httpx ``event_hooks["request"]`` callback.

    Reads ``_session_id_ctx`` at request time (NOT at client construction
    time) and stamps ``X-Session-ID`` on each outbound runner request.
    Reading the contextvar dynamically is what lets one cached workflow /
    one ChatOpenAI instance serve many sessions correctly — each request
    runs in its own asyncio task with its own contextvar value, so the
    right session id reaches the runner's SlotLRU.

    Set per-request without touching ``default_headers`` so we don't
    statically bake a session id into the ChatOpenAI instance (which
    would be wrong the moment the workflow gets cached and re-used).
    """
    sid = _session_id_ctx.get()
    if sid:
        request.headers["X-Session-ID"] = sid


async def _log_prompt_fingerprint(request) -> None:  # type: ignore[no-untyped-def]
    """httpx ``event_hooks["request"]`` callback — diagnostic.

    Logs SHA256 + prefix hashes of every outbound runner POST.  Lets us
    correlate "cache invalidated on slot N" warnings on the runner side
    with "did the prompt body change at byte X between turns" on the API
    side, without needing to dump multi-hundred-KB request bodies.

    Three rolling-window hashes are emitted (first 256 bytes, first 8 KiB,
    full body) so the comparison can pinpoint *where* two consecutive
    requests diverge: if the 256-byte hash matches but the 8 KiB hash
    doesn't, the divergence is in bytes 256-8192 (typically the
    system prompt / tool definitions region).  If the 256-byte hash
    differs too, the divergence is at the very front (chat template /
    first system tokens).
    """
    # Skip non-POST or zero-body requests so we don't log /v1/status etc.
    if request.method != "POST":
        return
    body = request.content
    if not body:
        return
    import hashlib

    full = hashlib.sha256(body).hexdigest()[:16]
    prefix_256 = hashlib.sha256(body[:256]).hexdigest()[:16]
    prefix_8k = hashlib.sha256(body[:8192]).hexdigest()[:16]

    sid = _session_id_ctx.get() or "none"
    llmmllogger.logger.bind(component="RunnerRequestFingerprint").info(
        "Runner request fingerprint",
        session_id=sid,
        url_path=str(request.url.path),
        body_bytes=len(body),
        hash_full=full,
        hash_8k=prefix_8k,
        hash_256=prefix_256,
    )


def _make_runner_http_client(timeout: float | None = None):
    """Build an ``httpx.AsyncClient`` with the session-id event hook +
    the prompt-fingerprint diagnostic hook.

    The read/write timeout defaults to ``RUNNER_CHAT_TIMEOUT_SEC`` (30 min) —
    for a streaming response that's the inactivity gap allowed before the first
    token and between chunks, which must cover cold model load + a large prefill.
    The old hard-coded 120s was the smallest cap on the generation path and cut
    long cron turns well under their job budget. Connect stays short (the server
    should accept promptly); only the generation wait is long.

    Lazy-import so test mocks that patch ``langchain_openai`` don't have
    to also patch httpx.
    """
    import httpx  # local import — httpx is already a runtime dep

    from config import RUNNER_CHAT_TIMEOUT_SEC

    read_timeout = RUNNER_CHAT_TIMEOUT_SEC if timeout is None else timeout

    return httpx.AsyncClient(
        timeout=httpx.Timeout(read_timeout, connect=30.0),
        event_hooks={
            "request": [
                _inject_session_id_header,
                _log_prompt_fingerprint,
            ]
        },
    )


# Default fallback context window when a model definition has no explicit
# `num_ctx` parameter. Kept as a module-level constant so subclasses and
# tests can reference the same value.
DEFAULT_NUM_CTX = 90000


class GraphBuilder(ABC):
    """
    Base class for workflow builders.

    Holds per-user storage handles and a logger. Subclasses implement
    `build_workflow` and `create_initial_state`.

    Subclasses must set the ``_runner_client`` and ``_chat_agent_cls`` class
    attributes to the symbols imported in their own module. This indirection
    keeps ``unittest.mock.patch("graph.workflows.<sub>.builder.runner_client")``
    style test patches working while still letting the shared model-resolution
    logic live here.
    """

    server_handle: Optional["ServerHandle"] = None

    # ChatOpenAI retry policy — IDE uses a small fixed value; Dialog reads
    # config.CHAT_OPENAI_MAX_RETRIES. Subclasses can override.
    _chat_openai_max_retries: int = 2

    @property
    def _runner_client(self) -> Any:
        """Look up ``runner_client`` from the subclass's defining module.

        Reading through ``sys.modules`` (rather than capturing the symbol at
        class-definition time) lets tests that ``patch("graph.workflows.X.builder.runner_client")``
        affect the value the shared base-class helpers see.
        """
        import sys

        mod = sys.modules[type(self).__module__]
        return getattr(mod, "runner_client")

    @property
    def _chat_agent_cls(self) -> Any:
        """Look up ``ChatAgent`` from the subclass's defining module.

        Same rationale as ``_runner_client``: respect test-time monkeypatches.
        """
        import sys

        mod = sys.modules[type(self).__module__]
        return getattr(mod, "ChatAgent")

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

    async def _resolve_primary_model(
        self, user_id: str, model_name: Optional[str]
    ) -> "Model":
        """Resolve the primary (TextToText) model for this workflow.

        Mirrors the previous per-builder resolution logic so behaviour stays
        bit-for-bit identical:

        1. If ``model_name`` is given, look it up in ``runner_client.list_models()``.
        2. If not found, ask ``resolve_model`` for the user's default and re-search
           the same list (no re-fetch).
        3. If still not found, fall back to ``runner_client.default_model_by_task``.
        4. If no ``model_name``, take the first ``TEXTTOTEXT`` model.
        """
        rc = self._runner_client

        if model_name:
            all_models = await rc.list_models()
            model_def = next(
                (m for m in all_models if m.name == model_name or m.id == model_name),
                None,
            )
            if not model_def:
                # Requested model not on any runner — fall back to user's default
                model_name = await self.resolve_model(model_name, user_id)
                model_def = next(
                    (m for m in all_models if m.name == model_name or m.id == model_name),
                    None,
                )
                if not model_def:
                    # Fallback model also not found — use the configured default
                    # TextToText model
                    self.logger.warning(
                        "Resolved model not found on runners, using default "
                        "TextToText model",
                        user_id=user_id,
                        resolved=model_name,
                    )
                    model_def = await rc.default_model_by_task(ModelTask.TEXTTOTEXT)
                    if not model_def:
                        raise RuntimeError(
                            f"Model '{model_name}' not found and no "
                            "TextToText model available"
                        )
        else:
            model_def = await rc.model_by_task(ModelTask.TEXTTOTEXT)
            if not model_def:
                raise RuntimeError("No TextToText model available")

        return model_def

    async def _acquire_primary_server_and_agent(
        self,
        user_id: str,
        model_def: "Model",
        system_prompt_default: str,
        component_name: str,
        model_parameters: "ModelParameters | None" = None,
    ) -> Tuple["ChatAgent", Any]:
        """Acquire a server for the resolved primary model and build a ChatAgent.

        Returns the constructed agent and the underlying ``ChatOpenAI``
        instance (so callers that need to ``bind_tools`` can rewrap it — see
        IDE proxy mode).

        Side-effect: sets ``self.server_handle``.
        """
        rc = self._runner_client
        chat_agent_cls = self._chat_agent_cls

        self.logger.debug(
            "Building workflow",
            user_id=user_id,
            model=model_def.name,
        )

        assert model_def.id is not None, "Model definition must have an ID"

        # Merge request-level model parameters override onto model defaults
        if model_parameters and model_def.parameters:
            effective_params = model_def.parameters.model_copy(
                update=model_parameters.model_dump(exclude_none=True)
            )
        elif model_parameters:
            effective_params = model_parameters
        else:
            effective_params = model_def.parameters

        num_ctx = (
            effective_params.num_ctx if effective_params else DEFAULT_NUM_CTX
        )
        server_handle = await rc.acquire_server(
            model_id=model_def.id,
            num_ctx=num_ctx,
            task=model_def.task,
        )

        # When thinking is enabled, send reasoning_format=deepseek so
        # llama.cpp activates its reasoning parser (works for both Qwen
        # and Gemma-4).  Budget tags are model-specific: Qwen3.6 resolves
        # to Content-only chat format (no auto-population), so we force
        # <think>/</think> explicitly.  Gemma-4 has a dedicated PEG parser
        # (common_chat_params_init_gemma4) that auto-populates tags from
        # the template, so we omit them and let the server handle it.
        # See llama.cpp tools/server/server-task.cpp:422–426 (per-request
        # reasoning_format) and sampling.cpp:296 (budget enforcement).
        chat_openai_extras: dict = {}
        if effective_params and getattr(effective_params, "think", False):
            extra_body: dict = {
                "reasoning_format": "deepseek",
            }
            if getattr(effective_params, "reasoning_budget", None):
                extra_body["reasoning_budget_tokens"] = (
                    effective_params.reasoning_budget
                )
                _is_gemma = "gemma" in (model_def.name or "").lower()
                if not _is_gemma:
                    extra_body["reasoning_budget_start_tag"] = "<think>"
                    extra_body["reasoning_budget_end_tag"] = "</think>"
            chat_openai_extras["model_kwargs"] = {"extra_body": extra_body}

        primary_model = ChatOpenAI(
            base_url=server_handle.base_url,
            api_key=SecretStr("none"),
            model=model_def.name,
            stream_usage=True,
            max_retries=self._chat_openai_max_retries,
            # X-Session-ID is set per-request by the httpx event hook so a
            # cached workflow can be shared across sessions without
            # baking one session's id into every request.  See
            # ``_inject_session_id_header`` above.
            http_async_client=_make_runner_http_client(),
            **chat_openai_extras,
        )
        self.server_handle = server_handle

        primary_agent = chat_agent_cls(
            model=cast(BaseChatModel, primary_model),
            system_prompt=model_def.system_prompt or system_prompt_default,
            num_ctx=(effective_params.num_ctx if effective_params else None)
            or DEFAULT_NUM_CTX,
            component_name=component_name,
        )

        return primary_agent, primary_model

    async def _build_primary_agent(
        self,
        user_id: str,
        model_name: Optional[str],
        system_prompt_default: str,
        component_name: str,
        model_parameters: "ModelParameters | None" = None,
    ) -> Tuple["Model", "ChatAgent", Any]:
        """One-shot: resolve primary model, acquire server, build ChatAgent.

        Convenience wrapper for builders that don't need to interleave any
        other resolution steps (e.g. IDE). Dialog uses the two-phase pair
        ``_resolve_primary_model`` + ``_acquire_primary_server_and_agent``
        directly so it can raise on a missing embedding model before any
        server is acquired.
        """
        model_def = await self._resolve_primary_model(user_id, model_name)
        primary_agent, primary_model = await self._acquire_primary_server_and_agent(
            user_id=user_id,
            model_def=model_def,
            system_prompt_default=system_prompt_default,
            component_name=component_name,
            model_parameters=model_parameters,
        )
        return model_def, primary_agent, primary_model

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
