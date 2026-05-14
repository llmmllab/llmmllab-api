"""Async priority queue for request scheduling with aging and starvation prevention."""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union

from models.request_priority_metadata import Priority, RequestPriorityMetadata
from services.queue_exceptions import QueueFullError, QueueTimeoutError
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="priority_queue")

# Prometheus metrics (imported from centralized registry)
try:
    from middleware.api_metrics import (
        queue_enqueued_total,
        queue_dequeued_total,
        queue_wait_time_seconds,
        queue_size as _queue_size_gauge,
        queue_aged_total,
        queue_size_by_model,
        queue_size_by_source,
        queue_size_by_model_source,
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


def _inc_enqueued(priority: Priority, source: str) -> None:
    if _HAS_PROMETHEUS:
        queue_enqueued_total.labels(priority=priority.name, source=source).inc()


def _inc_dequeued(priority: Priority, source: str) -> None:
    if _HAS_PROMETHEUS:
        queue_dequeued_total.labels(priority=priority.name, source=source).inc()


def _observe_wait(seconds: float, priority: Priority, source: str) -> None:
    if _HAS_PROMETHEUS:
        queue_wait_time_seconds.labels(priority=priority.name, source=source).observe(
            seconds
        )


def _set_size(priority: Priority, size: int) -> None:
    if _HAS_PROMETHEUS:
        _queue_size_gauge.labels(priority=priority.name).set(size)


def _inc_aged(from_p: Priority, to_p: Priority) -> None:
    if _HAS_PROMETHEUS:
        queue_aged_total.labels(from_priority=from_p.name, to_priority=to_p.name).inc()


@dataclass(order=True)
class _QueueItem:
    """Internal queue item with priority ordering."""

    sort_key: tuple = field(compare=True)
    metadata: RequestPriorityMetadata = field(compare=False)
    event: asyncio.Event = field(compare=False, default_factory=asyncio.Event)

    @classmethod
    def create(cls, metadata: RequestPriorityMetadata) -> _QueueItem:
        return cls(
            sort_key=(metadata.priority.value, metadata.enqueued_at),
            metadata=metadata,
        )


class AsyncPriorityQueue:
    """Thread-safe async priority queue with aging and starvation prevention.

    Requests are dequeued in priority order (HIGH > MEDIUM > LOW).
    After ``age_threshold`` seconds, LOW requests are promoted to MEDIUM,
    and MEDIUM requests are promoted to HIGH, preventing starvation.
    """

    def __init__(
        self,
        max_size: int = 100,
        timeout_sec: float = 300,
        age_threshold_sec: float = 60,
        on_release: Optional[
            Callable[[RequestPriorityMetadata], Any]
        ] = None,
        on_complete: Optional[
            Callable[[RequestPriorityMetadata], Any]
        ] = None,
    ) -> None:
        self._max_size = max_size
        self._timeout_sec = timeout_sec
        self._age_threshold_sec = age_threshold_sec
        self._queue: list[_QueueItem] = []
        self._lock = asyncio.Lock()
        self._counter = 0
        self._sizes = {p: 0 for p in Priority}
        self._can_proceed: Optional[
            Callable[[RequestPriorityMetadata], Awaitable[bool]]
        ] = None
        self._recheck_task: Optional[asyncio.Task] = None
        self._recheck_interval = 5.0
        self._on_release = on_release
        self._on_complete = on_complete

    def set_session_callbacks(
        self,
        on_release: Optional[Callable[[RequestPriorityMetadata], Any]],
        on_complete: Optional[Callable[[RequestPriorityMetadata], Any]],
    ) -> None:
        """Set or clear session lifecycle callbacks."""
        self._on_release = on_release
        self._on_complete = on_complete

    async def enqueue(
        self,
        metadata: RequestPriorityMetadata,
        timeout_sec: Optional[float] = None,
    ) -> asyncio.Event:
        """Add a request to the queue and return an event to wait on.

        The event is set when it is this request's turn to proceed.

        Raises:
            QueueFullError: If the queue is at capacity.
            QueueTimeoutError: If the request exceeds its max wait time.
        """
        effective_timeout = timeout_sec or metadata.max_queue_wait or self._timeout_sec

        async with self._lock:
            if len(self._queue) >= self._max_size:
                raise QueueFullError(
                    f"Priority queue full ({self._max_size} items). "
                    "Consider increasing PRIORITY_QUEUE_MAX_SIZE."
                )

            item = _QueueItem.create(metadata)
            self._counter += 1
            item.sort_key = (metadata.priority.value, self._counter)
            self._queue.append(item)
            self._queue.sort(key=lambda x: x.sort_key)
            self._sizes[metadata.priority] = sum(
                1 for i in self._queue if i.metadata.priority == metadata.priority
            )
            _inc_enqueued(metadata.priority, metadata.source.value)
            self._update_gauges()

            # If this is the only item in the queue, it's at the front and
            # can proceed immediately — there's no prior request whose
            # dequeue() would set its event.
            if len(self._queue) == 1:
                item.event.set()

        # Start aging task
        asyncio.create_task(self._age_item(item, metadata))

        # Wait for turn
        try:
            await asyncio.wait_for(item.event.wait(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Request timed out in priority queue",
                extra={
                    "priority": metadata.priority.name,
                    "source": metadata.source.value,
                    "wait_time": metadata.wait_time,
                    "max_wait_sec": effective_timeout,
                },
            )
            await self._remove_item(item)
            raise QueueTimeoutError(
                max_wait_sec=effective_timeout,
                actual_wait_sec=metadata.wait_time,
            ) from None

        return item.event

    def set_can_proceed_callback(
        self,
        can_proceed: Optional[Callable[[RequestPriorityMetadata], Awaitable[bool]]],
    ) -> None:
        """Set or clear the resource availability callback used by dequeue()."""
        self._can_proceed = can_proceed
        if can_proceed and self._recheck_task is None:
            self._recheck_task = asyncio.create_task(self._recheck_blocked())
        elif not can_proceed and self._recheck_task is not None:
            self._recheck_task.cancel()
            self._recheck_task = None

    async def close(self) -> None:
        """Cancel the background recheck task."""
        if self._recheck_task is not None:
            self._recheck_task.cancel()
            try:
                await self._recheck_task
            except asyncio.CancelledError:
                pass
            self._recheck_task = None

    async def dequeue(self) -> Optional[RequestPriorityMetadata]:
        """Remove the current request and let the next eligible one through.

        Pops the highest-priority (front) item — which is the request
        that just finished. Then iterates through remaining items and
        releases the first one whose resources are available (as determined
        by the ``_can_proceed`` callback). Items that can't proceed stay
        blocked in the queue.
        """
        released_meta: Optional[RequestPriorityMetadata] = None
        async with self._lock:
            if not self._queue:
                return None
            item = self._queue.pop(0)
            self._sizes[item.metadata.priority] = sum(
                1 for i in self._queue if i.metadata.priority == item.metadata.priority
            )
            _inc_dequeued(item.metadata.priority, item.metadata.source.value)
            _observe_wait(
                item.metadata.wait_time,
                item.metadata.priority,
                item.metadata.source.value,
            )
            self._update_gauges()

            if not self._queue:
                item.event.set()
                return item.metadata

            # Always set the popped item's event so its enqueue() completes
            item.event.set()

            # No callback — release next item unconditionally (original behavior)
            if self._can_proceed is None:
                self._queue[0].event.set()
                released_meta = self._queue[0].metadata
                return item.metadata

            # Collect metadata for all pending items to check outside lock
            pending: list[tuple[int, RequestPriorityMetadata]] = [
                (i, m.metadata) for i, m in enumerate(self._queue)
            ]

        # Call callback outside the lock to avoid blocking other operations
        for idx, metadata in pending:
            can_go = await self._can_proceed(metadata)
            if can_go:
                async with self._lock:
                    if (
                        idx < len(self._queue)
                        and self._queue[idx].metadata is metadata
                        and not self._queue[idx].event.is_set()
                    ):
                        # Priority preemption: don't release LOW if HIGH waits
                        if self._has_higher_priority_waiting(idx):
                            break
                        self._queue[idx].event.set()
                        released_meta = metadata
                break

        if self._on_release and released_meta is not None:
            self._on_release(released_meta)
        if self._on_complete:
            self._on_complete(item.metadata)
        return item.metadata

    def _has_higher_priority_waiting(self, released_idx: int) -> bool:
        """Check if any higher-priority item waits behind the released item.

        Must be called while holding self._lock.
        """
        released_priority = self._queue[released_idx].metadata.priority
        if released_priority == Priority.LOW:
            for i in range(released_idx + 1, len(self._queue)):
                if self._queue[i].metadata.priority == Priority.HIGH:
                    return True
        return False

    async def _remove_item(self, item: _QueueItem) -> None:
        """Remove a specific item (e.g., on timeout)."""
        async with self._lock:
            if item not in self._queue:
                return
            was_first = item is self._queue[0]
            self._queue.remove(item)
            self._sizes[item.metadata.priority] = sum(
                1
                for i in self._queue
                if i.metadata.priority == item.metadata.priority
            )
            self._update_gauges()

            if not was_first or not self._queue:
                return

            # No callback — release unconditionally
            if self._can_proceed is None:
                self._queue[0].event.set()
                if self._on_release:
                    self._on_release(self._queue[0].metadata)
                return

            # Collect pending items to check outside lock
            pending: list[tuple[int, RequestPriorityMetadata]] = [
                (i, m.metadata) for i, m in enumerate(self._queue)
            ]

        for idx, metadata in pending:
            can_go = await self._can_proceed(metadata)
            if can_go:
                async with self._lock:
                    if (
                        idx < len(self._queue)
                        and self._queue[idx].metadata is metadata
                        and not self._queue[idx].event.is_set()
                    ):
                        if not self._has_higher_priority_waiting(idx):
                            self._queue[idx].event.set()
                            if self._on_release:
                                self._on_release(metadata)
                break

    async def _age_item(
        self, item: _QueueItem, metadata: RequestPriorityMetadata
    ) -> None:
        """Promote a request's priority after age_threshold seconds."""
        await asyncio.sleep(self._age_threshold_sec)
        async with self._lock:
            if item not in self._queue:
                return  # Already dequeued
            old_priority = metadata.priority
            if old_priority == Priority.LOW:
                metadata.priority = Priority.MEDIUM
            elif old_priority == Priority.MEDIUM:
                metadata.priority = Priority.HIGH
            else:
                return  # Already HIGH
            item.sort_key = (metadata.priority.value, metadata.enqueued_at)
            self._queue.sort(key=lambda x: x.sort_key)
            self._sizes[old_priority] = sum(
                1 for i in self._queue if i.metadata.priority == old_priority
            )
            self._sizes[metadata.priority] = sum(
                1 for i in self._queue if i.metadata.priority == metadata.priority
            )
            _inc_aged(old_priority, metadata.priority)
            self._update_gauges()
            logger.info(
                f"Request aged from {old_priority.name} to {metadata.priority.name}",
                extra={
                    "wait_time": metadata.wait_time,
                },
            )

    async def _recheck_blocked(self) -> None:
        """Periodically re-check if blocked items can now proceed."""
        while True:
            await asyncio.sleep(self._recheck_interval)
            async with self._lock:
                if not self._queue or self._can_proceed is None:
                    continue
                pending: list[tuple[int, RequestPriorityMetadata]] = [
                    (i, m.metadata)
                    for i, m in enumerate(self._queue)
                    if not m.event.is_set()
                ]
                if not pending:
                    continue

            for idx, metadata in pending:
                can_go = await self._can_proceed(metadata)
                if can_go:
                    async with self._lock:
                        if (
                            idx < len(self._queue)
                            and self._queue[idx].metadata is metadata
                            and not self._queue[idx].event.is_set()
                        ):
                            self._queue[idx].event.set()
                            if self._on_release:
                                self._on_release(metadata)
                    break

    def _update_gauges(self) -> None:
        """Update all queue size gauges (by priority, model, source, cross-tab)."""
        for p in Priority:
            _set_size(p, self._sizes.get(p, 0))

        if not _HAS_PROMETHEUS:
            return

        model_counts = {}
        source_counts = {}
        model_source_counts = {}

        for item in self._queue:
            mid = item.metadata.model_id or "unknown"
            src = item.metadata.source.value
            model_counts[mid] = model_counts.get(mid, 0) + 1
            source_counts[src] = source_counts.get(src, 0) + 1
            model_source_counts[(mid, src)] = model_source_counts.get((mid, src), 0) + 1

        for mid, count in model_counts.items():
            queue_size_by_model.labels(model_id=mid).set(count)
        for src, count in source_counts.items():
            queue_size_by_source.labels(source=src).set(count)
        for (mid, src), count in model_source_counts.items():
            queue_size_by_model_source.labels(model_id=mid, source=src).set(count)

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def sizes_by_priority(self) -> dict[str, int]:
        return {p.name: self._sizes.get(p, 0) for p in Priority}


# Module-level singleton (configured from environment)
from config import (
    PRIORITY_QUEUE_MAX_SIZE,
    PRIORITY_QUEUE_TIMEOUT_SEC,
    PRIORITY_QUEUE_AGE_THRESHOLD_SEC,
)

priority_queue = AsyncPriorityQueue(
    max_size=PRIORITY_QUEUE_MAX_SIZE,
    timeout_sec=PRIORITY_QUEUE_TIMEOUT_SEC,
    age_threshold_sec=PRIORITY_QUEUE_AGE_THRESHOLD_SEC,
)
