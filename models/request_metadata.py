"""Request priority classification and metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class RequestSource(str, IntEnum):
    """Where a request originated from."""

    USER = "user"
    SCHEDULED = "scheduled"
    SYSTEM = "system"


class Priority(IntEnum):
    """Request priority levels (lower value = higher priority)."""

    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class RequestMetadata:
    """Metadata attached to every request for priority scheduling."""

    source: RequestSource = RequestSource.USER
    priority: Priority = Priority.HIGH
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    scheduled_at: Optional[float] = None
    enqueued_at: float = field(default_factory=time.monotonic)

    @property
    def wait_time(self) -> float:
        """Seconds this request has been waiting."""
        return time.monotonic() - self.enqueued_at
