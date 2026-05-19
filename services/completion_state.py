"""
Data containers for the completion service.

These dataclasses hold accumulated state for a single completion request —
``CompletionResult`` for non-streaming, ``StreamAccumulator`` for streaming.
They live in a dedicated module so that helpers and tests can import them
without pulling in the full ``CompletionService`` orchestration code.
"""

from dataclasses import dataclass, field
from typing import Optional

from models.chat_response import ChatResponse
from models.message import MessageContentType
from models.tool_call import ToolCall


@dataclass
class CompletionResult:
    """Accumulated result from a non-streaming completion."""

    chat_response: Optional[ChatResponse] = None
    output_tokens: int = 0
    context_overflow: bool = False

    @property
    def has_content(self) -> bool:
        return bool(
            self.chat_response
            and self.chat_response.message
            and self.chat_response.message.content
            and any(
                c.text
                for c in self.chat_response.message.content
                if c.type == MessageContentType.TEXT and c.text
            )
        )

    @property
    def has_tool_calls(self) -> bool:
        return bool(
            self.chat_response
            and self.chat_response.message
            and self.chat_response.message.tool_calls
        )

    @property
    def is_error(self) -> bool:
        return bool(self.chat_response and self.chat_response.finish_reason == "error")


@dataclass
class StreamAccumulator:
    """Mutable state accumulated while streaming events to the router."""

    has_content: bool = False
    has_tool_calls: bool = False
    is_error: bool = False
    finish_reason: str = ""
    final_tool_calls: list[ToolCall] = field(default_factory=list)
    final_content: str = ""
    output_tokens: int = 0
    input_tokens: int = 0
    context_overflow: bool = False
