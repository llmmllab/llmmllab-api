"""Request classification middleware - assigns priority based on request source.

Reads ``X-Request-Source``, ``X-Request-Priority``, and ``X-Max-Queue-Wait``
headers, falls back to sensible defaults.  Attaches ``RequestMetadata`` to
``request.state``.
"""

from __future__ import annotations

import json
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
from utils.logging import llmmllogger, set_session_id_ctx, reset_session_id_ctx

logger = llmmllogger.bind(component="priority_middleware")


class _BodyFields:
    __slots__ = ("session_id", "model_id")
    session_id: Optional[str]
    model_id: Optional[str]


async def _extract_body_fields(request: Request) -> Optional[_BodyFields]:
    """Extract session_id and model_id from the request body.

    OpenClaw sets compat.supportsPromptCacheKey=true which causes it to pass
    the session ID as ``prompt_cache_key`` in the JSON body.  This is the
    only way to receive session IDs from OpenClaw for ``openai-completions``
    providers, since ``resolveTransportTurnState`` doesn't fire for that
    transport path.
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        return None
    try:
        body = await request.body()
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, Exception):
        return None

    fields = _BodyFields()
    fields.session_id = data.get("prompt_cache_key") or None
    fields.model_id = data.get("model") or None
    return fields


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

    # Session ID: match any header ending with -session-id (case-insensitive)
    session_id = None
    for name, value in request.headers.items():
        if name.lower().endswith("-session-id"):
            session_id = value
            break

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

        # Fallback: extract session_id and model_id from request body.
        # OpenClaw sets compat.supportsPromptCacheKey=true which passes
        # session ID as ``prompt_cache_key`` in the JSON body.  Namespace
        # body-derived session IDs with a "pck:" prefix so they can never
        # collide with header-derived session IDs (e.g. another client
        # picking the same uuid by accident, or an OpenClaw cron job
        # using a prompt_cache_key that happens to match an active
        # claude-code anthropic-session-id).  This guarantees disjoint
        # slot-LRU namespaces between the two identification mechanisms.
        body_fields = await _extract_body_fields(request)
        if body_fields:
            if not metadata.session_id and body_fields.session_id:
                metadata.session_id = f"pck:{body_fields.session_id}"
                logger.debug(
                    "session_id derived from prompt_cache_key body field",
                    extra={"session_id": metadata.session_id},
                )
            if not getattr(metadata, "model_id", None):
                metadata.model_id = body_fields.model_id

        # Pattern-based source demotion: OpenClaw's cron-driven requests
        # arrive with prompt_cache_key prefixed `openclaw-cron-<hex>`,
        # while interactive sessions are plain UUIDs. If the caller
        # didn't set X-Request-Source or X-Request-Priority explicitly,
        # treat any `openclaw-cron-`-prefixed session as SCHEDULED → LOW
        # so interactive user turns preempt background cron work at the
        # queue layer. No-op if the explicit header path already
        # classified the request.
        if (
            not request.headers.get("X-Request-Source")
            and not request.headers.get("X-Request-Priority")
            and metadata.source == RequestSource.USER
            and body_fields is not None
            and body_fields.session_id
            and body_fields.session_id.startswith("openclaw-cron-")
        ):
            metadata.source = RequestSource.SCHEDULED
            metadata.priority = Priority.LOW
            logger.debug(
                "demoted to SCHEDULED/LOW via openclaw-cron session_id prefix",
                extra={"session_id": metadata.session_id},
            )

        request.state.request_priority_metadata = metadata

        token = set_session_id_ctx(metadata.session_id)
        try:
            response = await call_next(request)
            response.headers["X-Queue-Priority"] = metadata.priority.name.lower()
            effective_wait = metadata.max_queue_wait or PRIORITY_QUEUE_TIMEOUT_SEC
            response.headers["X-Queue-Max-Wait"] = str(int(effective_wait))
            return response
        finally:
            reset_session_id_ctx(token)
