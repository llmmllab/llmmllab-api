"""
Base Agent class providing common functionality for all workflow agents.
Provides node metadata injection, logging setup, and common error handling patterns.
"""

import datetime
import logging
import re
from typing import (
    Optional,
    Self,
    List,
)
from pydantic import BaseModel
from langchain.agents.structured_output import ProviderStrategy
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage

import config
from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    NodeMetadata,
    ChatResponse,
    Message,
)
from utils.logging import llmmllogger, serialize_event_data
from utils.message_conversion import (
    normalize_message_input,
    messages_to_lc_messages,
    lc_message_to_message,
    MessageInput,
    extract_text_from_message,
)
from utils.grammar_generator import parse_structured_output
# services.token_counter is imported lazily inside ``_trim_messages_to_context``
# to avoid a module-load cycle: services.token_counter → services.runner_client
# → graph.* → agents.chat → agents.base.  The trim path is the only consumer.

asyncio_logger = logging.getLogger("asyncio")
# Set the logging level to WARNING or higher (e.g., ERROR, CRITICAL)
# This will prevent INFO and DEBUG messages from being displayed when run_sync is used.
asyncio_logger.setLevel(logging.WARNING)


def get_message_count(messages: MessageInput) -> int:
    """Helper function to safely get message count from MessageInput."""
    if isinstance(messages, str):
        return 1
    elif isinstance(messages, Message):
        return 1
    elif isinstance(messages, list):
        return len(messages)
    else:
        # Fallback for unknown types
        return 1


class BaseAgent:
    """
    Base class for all workflow agents providing common functionality.

    This base class provides:
    - Node metadata injection for workflow tracking
    - Consistent logging setup with component binding
    - Common error handling patterns
    - Shared initialization patterns
    - Generic typing for pipeline execution results

    All agent classes should inherit from this base class to ensure consistent
    behavior across the workflow system.
    """

    def __init__(
        self,
        model: BaseChatModel,
        system_prompt: str = "",
        num_ctx: int = 90000,
        component_name: Optional[str] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize base agent with required dependencies.

        Args:
            model: Base chat model for agent operations
            system_prompt: System prompt for the agent
            num_ctx: Context window size for summarization threshold
            component_name: Optional component name for logging. If not provided,
                          uses the class name.
        """
        # Set up component-specific logging
        component = component_name or self.__class__.__name__
        self.logger = llmmllogger.bind(component=component)

        # Store required dependencies
        self.model = model
        self.system_prompt = system_prompt
        self.num_ctx = num_ctx

        self.agent_id = f"{id(self):x}"
        # Middleware list passed to create_agent for behaviors like TodoListMiddleware
        self.middleware: List[AgentMiddleware] = middleware or []
        self.tools: List[BaseTool] = tools or []

        self.logger.debug(f"Initialized {component}")

        self._node_metadata = NodeMetadata(
            node_name="UNSET",
            node_id="UNSET",
            node_type=self.__class__.__name__,
        )

    def bind_node_metadata(self, metadata: NodeMetadata) -> Self:
        """
        Bind new node metadata to the agent for workflow tracking.

        Args:
            metadata: New node metadata to bind
        """
        self._node_metadata = metadata
        self.logger = self.logger.bind(
            node_name=metadata.node_name,
            node_id=metadata.node_id,
            node_type=metadata.node_type,
            user_id=metadata.user_id,
        )
        self.logger.debug(
            "Bound new node metadata to agent",
            node_name=metadata.node_name,
            node_type=metadata.node_type,
        )
        return self

    async def _get_or_create_agent(
        self,
        system_prompt,
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[type[BaseModel]] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
        metadata: Optional[NodeMetadata] = None,
    ):
        """
        Get the persistent agent or create it if it doesn't exist.

        For performance and server reuse, we cache the pipeline but not the agent,
        since agent configuration (system prompt, tools, grammar) varies by call.
        The pipeline (LLM server) should be reused across different agent configurations.

        Args:
            system_prompt: System prompt for the agent
            tools: List of tools to bind to the agent
            priority: Pipeline priority
            grammar: Grammar constraints for structured output

        Returns:
            The LangChain agent or ChatOpenAI model (depending on pipeline type)
        """
        # Always create new agent for different configurations, but reuse pipeline
        # This allows different system prompts, tools, and grammars while maintaining server reuse

        self.logger.debug("Creating LangChain agent (pipeline will be reused)")
        agent = create_agent(
            model=self.model,
            tools=tools or [],
            system_prompt=system_prompt,
            response_format=ProviderStrategy(grammar) if grammar else None,
            name=(
                metadata.node_name
                if metadata is not None
                else self._node_metadata.node_name
            ),
            middleware=middleware or [],
            # LangGraph's debug=True prints every workflow-state transition
            # (including full message content) to stdout, bypassing our
            # structured logger.  In production with LOG_LEVEL=debug this
            # firehose dumps every OpenClaw cron job's full prompt template
            # into the logs, polluting them with markdown content that
            # looks like errors.  Gate on the rarer "trace" level only.
            debug=config.LOG_LEVEL.lower() == "trace",
        )

        return agent

    def _log_operation_start(self, operation: str, **kwargs) -> None:
        """
        Log the start of an operation with context.

        Args:
            operation: Name of the operation being started
            **kwargs: Additional context to log
        """
        context = {
            "operation": operation,
            **kwargs,
        }

        # Add node metadata context if available
        if self._node_metadata:
            context.update(
                {
                    "node_name": self._node_metadata.node_name,
                    "user_id": self._node_metadata.user_id,
                }
            )

        self.logger.info(f"Starting {operation}", **context)

    def _log_operation_success(self, operation: str, **kwargs) -> None:
        """
        Log successful completion of an operation.

        Args:
            operation: Name of the operation that completed
            **kwargs: Additional context to log
        """
        context = {
            "operation": operation,
            **kwargs,
        }

        self.logger.info(f"Completed {operation}", **context)

    def _log_operation_error(self, operation: str, error: Exception, **kwargs) -> None:
        """
        Log operation failure with error details.

        Args:
            operation: Name of the operation that failed
            error: Exception that occurred
            **kwargs: Additional context to log
        """
        context = {
            "operation": operation,
            "error": str(error),
            "error_type": type(error).__name__,
            **kwargs,
        }

        # Add node metadata context if available
        if self._node_metadata:
            context.update(
                {
                    "node_name": self._node_metadata.node_name,
                    "user_id": self._node_metadata.user_id,
                }
            )

        self.logger.error(f"Failed {operation}", **context)

    def _handle_node_error(self, operation: str, error: Exception, **context) -> None:
        """
        Handle and wrap errors in NodeExecutionError with consistent logging.

        Args:
            operation: Name of the operation that failed
            error: Original exception
            **context: Additional context for logging
        """
        self._log_operation_error(operation, error, **context)

    _SYSTEM_PROMPT_STRIP_RE = re.compile(r"^[^*\n]*Co-Authored-By:.*$\n?", re.MULTILINE)

    def _separate_system_prompt(
        self, messages: MessageInput
    ) -> tuple[str, List[Message]]:
        """
        Extract system prompt from messages if present.

        Args:
            messages: Input messages for the agent

        returns:
            str: Extracted system prompt
        """
        msgs = normalize_message_input(messages)
        convo = []

        system_prompt = self.system_prompt

        for msg in msgs:
            if msg.role == MessageRole.SYSTEM:
                system_prompt += f"\n\n{extract_text_from_message(msg)}"
            else:
                convo.append(msg)

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        system_prompt += f"""
The current date is {current_date}."""

        # Strip injected commit trailers (Co-Authored-By) from system prompt
        system_prompt = self._SYSTEM_PROMPT_STRIP_RE.sub("", system_prompt).rstrip()

        return system_prompt, convo

    def _resolve_runner_base_url(self) -> Optional[str]:
        """Pull the runner base URL off the LangChain ChatOpenAI we wrap.

        ChatOpenAI exposes the configured endpoint as ``openai_api_base``
        (set from the ``base_url=`` constructor arg in
        ``graph/workflows/base.py``).  Reaching through ``self.model``
        keeps us decoupled from the workflow plumbing — every agent
        already has its chat model.
        """
        model = getattr(self, "model", None)
        if model is None:
            return None
        for attr in ("openai_api_base", "base_url", "_base_url"):
            value = getattr(model, attr, None)
            if value:
                return str(value)
        return None

    async def _trim_messages_to_context(
        self, messages: List[Message], system_prompt: str
    ) -> List[Message]:
        """
        Trim conversation messages so total tokens fit within the model context window.

        Keeps the most recent messages and drops the oldest ones first, ensuring
        the combined system prompt + conversation stays under the context limit.
        Leaves a 10% headroom for the model's response tokens.

        Counts use llama.cpp's ``/tokenize`` endpoint via the runner proxy —
        the previous ``len // 3`` heuristic over-counted dense JSON / tool
        payloads by ~2× and triggered false context-overflow trims under
        claude-cli traffic.  If the tokenizer call fails (network blip,
        runner unreachable) we skip the trim and let llama-server itself
        decide whether the prompt fits — its own context guard will refuse
        a request that genuinely doesn't fit.
        """
        if not self.num_ctx or self.num_ctx <= 0:
            return messages

        # Lazy import — see the note next to the token_counter import line.
        from services.token_counter import count_tokens, count_message_tokens

        base_url = self._resolve_runner_base_url()
        if not base_url:
            # No runner handle threaded through — defer to llama-server's
            # own context guard rather than fall back to estimates.
            self.logger.debug(
                "No base_url on chat model; skipping pre-trim",
                message_count=len(messages),
            )
            return messages

        # Reserve 10% of context for the model's response
        headroom_ratio = 0.10
        max_input_tokens = int(self.num_ctx * (1 - headroom_ratio))

        # Real token count from llama.cpp's tokenizer
        system_tokens = await count_tokens(system_prompt, base_url=base_url)
        if system_tokens is None:
            self.logger.warning(
                "Tokenizer unavailable; skipping pre-trim (llama-server will guard)",
                num_ctx=self.num_ctx,
                message_count=len(messages),
            )
            return messages

        per_message_tokens: List[int] = []
        total_tokens = system_tokens
        for msg in messages:
            msg_tokens = await count_message_tokens(msg, base_url=base_url)
            if msg_tokens is None:
                # Per-message tokenize failed mid-stream — abandon proactive
                # trim and let llama-server's own guard handle it.
                self.logger.warning(
                    "Per-message tokenize failed; skipping pre-trim",
                    num_ctx=self.num_ctx,
                    message_count=len(messages),
                )
                return messages
            per_message_tokens.append(msg_tokens)
            total_tokens += msg_tokens

        if total_tokens <= max_input_tokens:
            return messages

        # Trim from the front (oldest messages first)
        self.logger.warning(
            "Conversation exceeds context window, trimming messages",
            total_tokens=total_tokens,
            max_input_tokens=max_input_tokens,
            num_ctx=self.num_ctx,
            message_count=len(messages),
        )

        trimmed = list(messages)  # copy
        remaining_tokens = total_tokens

        while remaining_tokens > max_input_tokens and len(trimmed) > 1:
            removed_tokens = per_message_tokens.pop(0)
            trimmed.pop(0)
            remaining_tokens -= removed_tokens

        self.logger.info(
            "Trimmed conversation to fit context window",
            original_count=len(messages),
            trimmed_count=len(trimmed),
            remaining_tokens=remaining_tokens,
            max_input_tokens=max_input_tokens,
        )
        return trimmed

    async def run(
        self,
        messages: MessageInput,
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[type[BaseModel]] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
        metadata: Optional[NodeMetadata] = None,
    ) -> ChatResponse:
        """
        Run agent execution with node metadata injection.

        Creates a LangChain agent using create_agent() with BaseChatModel from factory,
        then executes the agent and returns the result with node metadata.

        Args:
            messages: Input messages for the agent
            tools: Optional tools for the agent
            grammar: Optional grammar constraints for structured output
            middleware: Optional middleware for the agent
            metadata: Optional node metadata for workflow tracking

        Returns:
            ChatResponse: Response with injected node metadata
        """

        try:
            self._log_operation_start(
                "create_agent_run",
                message_count=get_message_count(messages),
                has_tools=bool(tools),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )
            system_prompt, convo = self._separate_system_prompt(messages)

            # Use persistent agent - creates once and reuses for state continuity
            # Deduplicate tools by name since StructuredTool objects are not hashable
            combined_tools = (self.tools or []) + (tools or [])
            seen_tool_names = set()
            unique_tools = []
            for tool in combined_tools:
                if tool.name not in seen_tool_names:
                    unique_tools.append(tool)
                    seen_tool_names.add(tool.name)

            # Deduplicate middleware by class since middleware objects might not be hashable
            combined_middleware = (self.middleware or []) + (middleware or [])
            seen_middleware_types = set()
            unique_middleware = []
            for mw in combined_middleware:
                mw_type = type(mw).__name__
                if mw_type not in seen_middleware_types:
                    unique_middleware.append(mw)
                    seen_middleware_types.add(mw_type)

            agent = await self._get_or_create_agent(
                system_prompt,
                unique_tools,
                grammar,
                unique_middleware,
                metadata,
            )

            if agent is None:
                self.logger.error("🚨 Agent is None after _get_or_create_agent call!")
                raise ValueError("Agent creation failed - agent is None")

            # Trim conversation to fit within context window before sending.
            # Uses llama.cpp's /tokenize endpoint for exact counts; skips the
            # trim entirely if the tokenizer call fails (llama-server's own
            # guard catches genuine overflows).
            convo = await self._trim_messages_to_context(convo, system_prompt)

            # Convert messages to LangChain format
            normalized_messages = messages_to_lc_messages(convo)
            self.logger.debug(f"Running agent with {len(normalized_messages)} messages")

            # Retry transient errors (connection errors + 5xx status errors like 503)
            # up to 10 times with exponential backoff.
            # Early retries use short delays (2s, 4s, 8s); later retries
            # use longer delays (16s, 32s, 60s, 60s, 60s, 60s) to allow
            # time for the runner/API to recover.
            # Non-transient errors propagate immediately.
            from openai import APIConnectionError as _APIConnectionError
            from openai import APIStatusError as _APIStatusError

            _TRANSIENT_STATUS_CODES = frozenset({502, 503, 504})

            def _is_transient_error(e: Exception) -> bool:
                if isinstance(e, _APIConnectionError):
                    return True
                if (
                    isinstance(e, _APIStatusError)
                    and e.status_code in _TRANSIENT_STATUS_CODES
                ):
                    return True
                return False

            result = None
            max_attempts = 11
            for attempt in range(max_attempts):
                try:
                    result = await agent.ainvoke({"messages": normalized_messages})  # type: ignore
                    break
                except Exception as e:
                    if not _is_transient_error(e):
                        raise
                    last_error = e
                    if attempt < max_attempts - 1:
                        # Exponential backoff capped at 60s
                        backoff = min(2 ** (attempt + 1), 60)
                        self.logger.warning(
                            f"Transient error ({type(e).__name__}), retrying in {backoff}s "
                            f"(attempt {attempt + 1}/{max_attempts})",
                            extra={"transient_error_detail": str(e)},
                        )
                        import asyncio as _asyncio

                        await _asyncio.sleep(backoff)
                    else:
                        raise

            if isinstance(result, dict):
                if "structured_response" in result and grammar:
                    result = result["structured_response"]
                    if not isinstance(result, BaseMessage):
                        result = AIMessage(content=grammar.model_dump_json(result))
                elif "messages" in result:
                    msgs = result["messages"]
                    if isinstance(msgs, list) and len(msgs) > 0:
                        # get the last message and assert that it's ai
                        result = msgs[-1]
                        if hasattr(result, "role") and result.role != "ai":
                            self.logger.warning(
                                "🚨 Last message role is not 'ai' - unexpected result format"
                            )

            assert isinstance(result, BaseMessage), "Agent result is not a BaseMessage"
            msg = lc_message_to_message(result)
            response = ChatResponse(
                done=True,
                message=msg,
                metadata=self._node_metadata,
            )

            return response

        except Exception as e:
            self._handle_node_error(
                "create_agent_run",
                e,
                message_count=get_message_count(messages),
            )
            # Re-raise timeout errors so the workflow executor catches them
            # and yields an error response with content.  Without this, the
            # timeout is swallowed, the executor sees no LLM events, and
            # produces an empty response — which triggers the completion
            # service to retry (creating a cascade of wasted requests).
            from openai import APITimeoutError

            if isinstance(e, (APITimeoutError, TimeoutError)):
                raise

            # Detect stale server handles: 404 "Server X not found" means
            # the llama.cpp server was evicted from the runner.  Re-raise
            # as StaleServerError so the CompletionService can re-acquire
            # a fresh server and retry.
            error_body = str(e).lower()
            if (
                "404" in error_body or "not found" in error_body
            ) and "server" in error_body:
                import re

                m = re.search(r"server\s+([a-f0-9]+)", str(e), re.IGNORECASE)
                server_id = m.group(1) if m else "unknown"
                from graph.errors import StaleServerError

                raise StaleServerError(server_id, e) from e

            return ChatResponse(
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Error during agent execution: {str(e)}",
                        )
                    ],
                ),
                metadata=self._node_metadata,
            )

    async def run_structured[T: BaseModel](
        self,
        message_input: MessageInput,
        grammar: type[T],
        tools: Optional[List[BaseTool]] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
        metadata: Optional[NodeMetadata] = None,
    ) -> T:
        """
        Run agent execution with node metadata injection.

        Creates a LangChain agent using create_agent() with BaseChatModel from factory,
        then executes the agent and returns the result with node metadata.

        Args:
            messages: Input messages for the agent
            user_id: User identifier
            tools: Optional tools for the agent
            priority: Pipeline execution priority (affects model selection)

        Returns:
            ChatResponse: Response with injected node metadata
        """

        try:
            self._log_operation_start(
                "structured_agent_run",
                message_count=get_message_count(message_input),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )
            system_prompt, convo = self._separate_system_prompt(message_input)

            # Use persistent agent - creates once and reuses for state continuity
            agent = await self._get_or_create_agent(
                system_prompt,
                list((self.tools or []) + (tools or [])),
                grammar,
                list((self.middleware or []) + (middleware or [])),
                metadata,
            )

            if agent is None:
                self.logger.error("🚨 Agent is None after _get_or_create_agent call!")
                raise ValueError("Agent creation failed - agent is None")

            # Trim conversation to fit within context window before sending.
            # Uses llama.cpp's /tokenize endpoint for exact counts; skips the
            # trim entirely if the tokenizer call fails (llama-server's own
            # guard catches genuine overflows).
            convo = await self._trim_messages_to_context(convo, system_prompt)

            # Convert messages to LangChain format
            normalized_messages = messages_to_lc_messages(convo)
            self.logger.debug(f"Running agent with {len(normalized_messages)} messages")

            # Retry transient errors (connection errors + 5xx status errors like 503)
            # up to 10 times with exponential backoff.
            from openai import APIConnectionError as _APIConnectionError
            from openai import APIStatusError as _APIStatusError

            _TRANSIENT_STATUS_CODES = frozenset({502, 503, 504})

            def _is_transient_error(e: Exception) -> bool:
                if isinstance(e, _APIConnectionError):
                    return True
                if (
                    isinstance(e, _APIStatusError)
                    and e.status_code in _TRANSIENT_STATUS_CODES
                ):
                    return True
                return False

            last_error = None
            max_attempts = 11
            result = None
            for attempt in range(max_attempts):
                try:
                    result = await agent.ainvoke({"messages": normalized_messages})  # type: ignore
                    break
                except Exception as e:
                    if not _is_transient_error(e):
                        raise
                    last_error = e
                    if attempt < max_attempts - 1:
                        backoff = min(2 ** (attempt + 1), 60)
                        self.logger.warning(
                            f"Transient error ({type(e).__name__}), retrying in {backoff}s "
                            f"(attempt {attempt + 1}/{max_attempts})",
                            extra={"transient_error_detail": str(e)},
                        )
                        import asyncio as _asyncio

                        await _asyncio.sleep(backoff)
                    else:
                        raise
            self.logger.debug(
                f"Agent run result ({type(result)}): {serialize_event_data(result)}"
            )

            if isinstance(result, dict):
                if "structured_response" in result and grammar:
                    result = result["structured_response"]
                    if isinstance(result, grammar):
                        self.logger.debug("Structured response matches grammar type")
                        return result
                elif "messages" in result:
                    msgs = result["messages"]
                    if isinstance(msgs, list) and len(msgs) > 0:
                        # get the last message and assert that it's ai
                        result = msgs[-1]
                        if hasattr(result, "role") and result.role != "ai":
                            self.logger.warning(
                                "🚨 Last message role is not 'ai' - unexpected result format"
                            )

            assert isinstance(result, BaseMessage), "Agent result is not a BaseMessage"

            msg = lc_message_to_message(result)
            return parse_structured_output(extract_text_from_message(msg), grammar)

        except Exception as e:
            self._handle_node_error(
                "create_agent_run",
                e,
                message_count=get_message_count(message_input),
            )
            raise RuntimeError(f"Structured agent execution failed: {e}") from e
