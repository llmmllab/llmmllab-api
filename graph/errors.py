"""
Error definitions and handling for composer service.
"""

from typing import Optional


class ComposerError(Exception):
    """Base exception for composer errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class WorkflowConstructionError(ComposerError):
    """Failed to construct workflow."""

    pass


class NodeExecutionError(ComposerError):
    """Node execution failed."""

    def __init__(self, node_name: str, original_error: Optional[Exception] = None):
        self.node_name = node_name
        self.original_error = original_error

        if original_error:
            message = f"Node '{node_name}' failed: {original_error}"
            details = {
                "node_name": node_name,
                "original_error": str(original_error),
            }
        else:
            message = f"Node '{node_name}' failed"
            details = {"node_name": node_name}

        super().__init__(message, details)


class ToolGenerationError(ComposerError):
    """Failed to generate dynamic tool."""

    pass


class CircuitOpenError(ComposerError):
    """Circuit breaker is open."""

    pass


class StateManagementError(ComposerError):
    """State management operation failed."""

    pass


class StreamingError(ComposerError):
    """Streaming operation failed."""

    pass


class ContextExceededError(ComposerError):
    """Raised when the estimated input context exceeds the model's context window.

    This error is raised **before** a server is acquired, so the runner never
    starts a llama.cpp instance for a request it can't handle.
    """

    def __init__(
        self,
        estimated_tokens: int,
        model_context_window: int,
        model_name: str = "",
    ):
        self.estimated_tokens = estimated_tokens
        self.model_context_window = model_context_window
        self.model_name = model_name

        message = (
            f"The conversation context is too large for the selected model "
            f"({model_name}). Estimated input tokens ({estimated_tokens:,}) "
            f"exceed the model's context window ({model_context_window:,}). "
            f"Please start a new conversation or use a model with a larger "
            f"context window."
        )
        details = {
            "estimated_tokens": estimated_tokens,
            "model_context_window": model_context_window,
            "model_name": model_name,
        }
        super().__init__(message, details)
