"""Request classification middleware - assigns priority based on request source.

Reads ``X-Request-Source`` and ``X-Request-Priority`` headers, falls back
to sensible defaults.  Attaches ``RequestMetadata`` to ``request.state``.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from models.request_metadata import Priority, RequestMetadata, RequestSource
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="priority_middleware")


def _classify_request(request: Request) -> RequestMetadata:
    """Classify an incoming request by source and assign priority."""
    source_header = request.headers.get("X-Request-Source", "").lower()
    priority_header = request.headers.get("X-Request-Priority", "").lower()

    # Determine source
    if source_header == "scheduled":
        source = RequestSource.SCHEDULED
    elif source_header == "system":
        source = RequestSource.SYSTEM
    else:
        source = RequestSource.USER

    # Determine priority (header overrides source-based default)
    if priority_header == "low":
        priority = Priority.LOW
    elif priority_header == "medium":
        priority = Priority.MEDIUM
    elif priority_header == "high":
        priority = Priority.HIGH
    else:
        # Default priority based on source
        if source == RequestSource.USER:
            priority = Priority.HIGH
        elif source == RequestSource.SCHEDULED:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW

    # Extract user_id from auth state if available
    user_id = getattr(request.state, "user_id", None)

    return RequestMetadata(
        source=source,
        priority=priority,
        user_id=user_id,
        session_id=request.headers.get("X-Session-ID"),
    )


class PriorityMiddleware(BaseHTTPMiddleware):
    """Attach RequestMetadata to every request for downstream priority scheduling."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip non-API paths
        if not request.url.path.startswith("/v1/") and not request.url.path.startswith(
            "/api/"
        ):
            response = await call_next(request)
            return response

        metadata = _classify_request(request)
        request.state.request_metadata = metadata

        response = await call_next(request)
        response.headers["X-Queue-Priority"] = metadata.priority.name.lower()
        return response
