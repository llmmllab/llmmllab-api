"""Async priority queue for request scheduling with aging and starvation prevention."""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from models.request_priority_metadata import Priority, RequestPriorityMetadata
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="priority_queue")

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge

    _queue_enqueued_total = Counter(
        "llmmllab_api_queue_enqueued_total",
        "Total requests enqueued by priority",
        ["priority", "source"],
    )
    _queue_dequeued_total = Counter(
        "llmmllab_api_queue_dequeued_total",
        "Total requests dequeued by priority",
        ["priority", "source"],
    )
    _queue_wait_time_seconds = Histogram(
        "llmmllab_api_queue_wait_time_seconds",
        "Time spent waiting in queue by priority",
        ["priority", "source"],
    )
    _queue_size = Gauge(
        "llmmllab_api_queue_size",
        "Current queue size by priority",
        ["priority"],
    )
    _queue_aged_total = Counter(
        "llmmllab_api_queue_aged_total",
        "Total requests promoted due to aging",
        ["from_priority", "to_priority"],
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


def _inc_enqueued(priority: Priority, source: str) -> None:
    if _HAS_PROMETHEUS:
        _queue_enqueued_total.labels(priority=priority.name, source=source).inc()


def _inc_dequeued(priority: Priority, source: str) -> None:
    if _HAS_PROMETHEUS:
        _queue_dequeued_total.labels(priority=priority.name, source=source).inc()


def _observe_wait(seconds: float, priority: Priority, source: str) -> None:
    if _HAS_PROMETHEUS:
        _queue_wait_time_seconds.labels(priority=priority.name, source=source).observe(
            seconds
        )


def _set_size(priority: Priority, size: int) -> None:
    if _HAS_PROMETHEUS:
        _queue_size.labels(priority=priority.name).set(size)


def _inc_aged(from_p: Priority, to_p: Priority) -> None:
    if _HAS_PROMETHEUS:
        _queue_aged_total.labels(from_priority=from_p.name, to_priority=to_p.name).inc()


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
    ) -> None:
        self._max_size = max_size
        self._timeout_sec = timeout_sec
        self._age_threshold_sec = age_threshold_sec
        self._queue: list[_QueueItem] = []
        self._lock = asyncio.Lock()
        self._counter = 0
        self._sizes = {p: 0 for p in Priority}

    async def enqueue(self, metadata: RequestPriorityMetadata) -> asyncio.Event:
        """Add a request to the queue and return an event to wait on.

        The event is set when it is this request's turn to proceed.
        """
        async with self._lock:
            if len(self._queue) >= self._max_size:
                raise RuntimeError(
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

        # Start aging task
        asyncio.create_task(self._age_item(item, metadata))

        # Wait for turn
        try:
            await asyncio.wait_for(item.event.wait(), timeout=self._timeout_sec)
        except asyncio.TimeoutError:
            logger.warning(
                "Request timed out in priority queue",
                extra={
                    "priority": metadata.priority.name,
                    "source": metadata.source.value,
                    "wait_time": metadata.wait_time,
                },
            )
            # Still try to dequeue it
            await self._remove_item(item)
            return item.event

        return item.event

    async def dequeue(self) -> Optional[RequestPriorityMetadata]:
        """Remove and return the highest-priority request's metadata.

        Call this after processing a request to let the next one through.
        """
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
            item.event.set()
            return item.metadata

    async def _remove_item(self, item: _QueueItem) -> None:
        """Remove a specific item (e.g., on timeout)."""
        async with self._lock:
            if item in self._queue:
                self._queue.remove(item)
                self._sizes[item.metadata.priority] = sum(
                    1
                    for i in self._queue
                    if i.metadata.priority == item.metadata.priority
                )
                self._update_gauges()

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

    def _update_gauges(self) -> None:
        for p in Priority:
            _set_size(p, self._sizes.get(p, 0))

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def sizes_by_priority(self) -> dict[str, int]:
        return {p.name: self._sizes.get(p, 0) for p in Priority}


# Module-level singleton
priority_queue = AsyncPriorityQueue()
