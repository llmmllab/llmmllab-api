"""Exceptions for priority queue rejection."""


class QueueTimeoutError(Exception):
    """Raised when a request's max queue wait time expires."""

    def __init__(self, max_wait_sec: float, actual_wait_sec: float) -> None:
        self.max_wait_sec = max_wait_sec
        self.actual_wait_sec = actual_wait_sec
        super().__init__(
            f"Request timed out in queue after {actual_wait_sec:.1f}s "
            f"(max wait: {max_wait_sec:.0f}s)"
        )


class QueueFullError(Exception):
    """Raised when the priority queue is at capacity."""

    pass
