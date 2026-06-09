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


class ColdStartError(ComposerError):
    """A model server is still loading (cold start) — retry after a wait.

    Raised when ``acquire_server`` gets HTTP 503 "Runner busy starting the
    model …" from a runner whose ``/v1/server/create`` is mid-load.  A fresh
    llama.cpp server (big quantised GGUF + mmproj) takes ~45-90 s to become
    ready; while it loads the runner answers 503.  This is *transient* — a
    short wait then retry succeeds — so callers (see
    ``services.retry_policies.stream_with_connection_retry``) should sleep a
    cold-start interval and re-acquire rather than surface the 503 to the
    client.

    Distinct from :class:`StaleServerError` (a 404 for an already-acquired,
    now-evicted handle): a ColdStartError means we never got a handle because
    the server is still coming up.
    """

    def __init__(
        self, model_id: str, original_error: Optional[Exception] = None
    ):
        self.model_id = model_id
        self.original_error = original_error
        message = f"Model {model_id} server is still loading (cold start)"
        if original_error:
            message += f": {original_error}"
        super().__init__(message, {"model_id": model_id})


class StaleServerError(ComposerError):
    """Server handle is stale — the llama.cpp server was evicted.

    Raised when a 404 \"Server not found\" error is detected, indicating
    the runner has removed the server that the current ``ServerHandle``
    points to.  Callers should re-acquire a fresh server and retry.
    """

    def __init__(
        self, server_id: str, original_error: Optional[Exception] = None
    ):
        self.server_id = server_id
        self.original_error = original_error
        message = f"Server {server_id} not found (stale handle)"
        if original_error:
            message += f": {original_error}"
        super().__init__(message, {"server_id": server_id})
