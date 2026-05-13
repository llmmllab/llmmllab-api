"""Request classification middleware - assigns priority based on request source.

Reads ``X-Request-Source``, ``X-Request-Priority``, and ``X-Max-Queue-Wait``
headers, falls back to sensible defaults.  Attaches ``RequestMetadata`` to
``request.state``.
"""

from __future__ import annotations

from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from config import (
    PRIORITY_QUEUE_MAX_WAIT_MAX_SEC,
    PRIORITY_QUEUE_MAX_WAIT_MIN_SEC,
    PRIORITY_QUEUE_TIMEOUT_SEC,
)
from models.request_priority_metadata import (
    Priority,
    RequestPriorityMetadata,
    RequestSource,
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="priority_middleware")


def _parse_max_queue_wait(header_value: str) -> Optional[float]:
    """Parse X-Max-Queue-Wait header. Returns None if invalid/missing."""
    if not header_value:
        return None
    try:
        val = int(header_value)
    except ValueError:
        logger.warning(f"Invalid X-Max-Queue-Wait header: {header_value!r}, ignoring")
        return None
    if val < PRIORITY_QUEUE_MAX_WAIT_MIN_SEC:
        logger.warning(
            f"X-Max-Queue-Wait={val} below minimum ({PRIORITY_QUEUE_MAX_WAIT_MIN_SEC}s), "
            f"clamping to {PRIORITY_QUEUE_MAX_WAIT_MIN_SEC}"
        )
        return float(PRIORITY_QUEUE_MAX_WAIT_MIN_SEC)
    if val > PRIORITY_QUEUE_MAX_WAIT_MAX_SEC:
        logger.warning(
            f"X-Max-Queue-Wait={val} exceeds maximum ({PRIORITY_QUEUE_MAX_WAIT_MAX_SEC}s), "
            f"clamping to {PRIORITY_QUEUE_MAX_WAIT_MAX_SEC}"
        )
        return float(PRIORITY_QUEUE_MAX_WAIT_MAX_SEC)
    return float(val)


def _classify_request(request: Request) -> RequestPriorityMetadata:
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

    # Set default priority, will be overridden by header if provided
    priority = Priority.MEDIUM

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
        elif source == RequestSource.SCHEDULED or source == RequestSource.SYSTEM:
            priority = Priority.LOW

    # Extract user_id from auth state if available
    user_id = getattr(request.state, "user_id", None)

    # Parse optional max queue wait time
    max_queue_wait = _parse_max_queue_wait(
        request.headers.get("X-Max-Queue-Wait", "")
    )

    # Session ID: check multiple header sources
    session_id = (
        request.headers.get("X-Session-ID")
        or request.headers.get("X-Claude-Code-Session-ID")
    )

    return RequestPriorityMetadata(
        source=source,
        priority=priority,
        user_id=user_id,
        session_id=session_id,
        max_queue_wait=max_queue_wait,
    )


COMPLETION_ENDPOINTS = {"/chat/completions", "/messages"}


def is_completion_endpoint(path: str) -> bool:
    """Check if the request path is a completion endpoint."""
    return any(path.find(endpoint) != -1 for endpoint in COMPLETION_ENDPOINTS)


class PriorityMiddleware(BaseHTTPMiddleware):
    """Attach RequestMetadata to every request for downstream priority scheduling."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip non-API paths
        if not is_completion_endpoint(request.url.path):
            response = await call_next(request)
            return response

        metadata = _classify_request(request)
        request.state.request_priority_metadata = metadata

        response = await call_next(request)
        response.headers["X-Queue-Priority"] = metadata.priority.name.lower()
        effective_wait = metadata.max_queue_wait or PRIORITY_QUEUE_TIMEOUT_SEC
        response.headers["X-Queue-Max-Wait"] = str(int(effective_wait))
        return response
