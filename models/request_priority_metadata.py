"""Request priority classification and metadata."""

from __future__ import annotations

import time
from enum import Enum, IntEnum
from typing import Annotated, Optional

from pydantic import BaseModel, Field


class RequestSource(str, Enum):
    """Where a request originated from."""

    USER = "user"
    SCHEDULED = "scheduled"
    SYSTEM = "system"


class Priority(IntEnum):
    """Request priority levels (lower value = higher priority)."""

    HIGH = 1
    MEDIUM = 2
    LOW = 3


class RequestPriorityMetadata(BaseModel):
    """Metadata attached to every request for priority scheduling."""

    source: Annotated[
        RequestSource,
        Field(RequestSource.USER, description="The source of the request."),
    ] = RequestSource.USER
    priority: Annotated[
        Priority, Field(Priority.HIGH, description="The priority of the request.")
    ] = Priority.HIGH
    user_id: Annotated[
        Optional[str],
        Field(None, description="The ID of the user who made the request."),
    ] = None
    session_id: Annotated[
        Optional[str],
        Field(None, description="The ID of the session to which the request belongs."),
    ] = None
    scheduled_at: Annotated[
        Optional[float],
        Field(
            None,
            description="The time at which the request is scheduled to be processed.",
        ),
    ] = None
    max_queue_wait: Annotated[
        Optional[float],
        Field(
            None,
            description="Maximum seconds this request can wait in the queue before being rejected.",
        ),
    ] = None
    enqueued_at: Annotated[
        float,
        Field(
            default_factory=time.monotonic,
            description="The time at which the request was enqueued.",
        ),
    ] = time.monotonic()

    @property
    def wait_time(self) -> float:
        """Seconds this request has been waiting."""
        return time.monotonic() - self.enqueued_at
