"""Async priority queue for request scheduling with aging and starvation prevention."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from models.request_priority_metadata import Priority, RequestPriorityMetadata
from services.queue_exceptions import QueueFullError, QueueTimeoutError
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="priority_queue")

# Prometheus metrics (imported from centralized registry)
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


def _inc_enqueued(priority: Priority, source: str) -> None:
    queue_enqueued_total.labels(priority=priority.name, source=source).inc()


def _inc_dequeued(priority: Priority, source: str) -> None:
    queue_dequeued_total.labels(priority=priority.name, source=source).inc()


def _observe_wait(seconds: float, priority: Priority, source: str) -> None:
    queue_wait_time_seconds.labels(priority=priority.name, source=source).observe(
        seconds
    )


def _set_size(priority: Priority, size: int) -> None:
    _queue_size_gauge.labels(priority=priority.name).set(size)


def _inc_aged(from_p: Priority, to_p: Priority) -> None:
    queue_aged_total.labels(from_priority=from_p.name, to_priority=to_p.name).inc()


@dataclass(order=True)
class _QueueItem:
    """Internal queue item with priority ordering."""

    sort_key: tuple = field(compare=True)
    metadata: RequestPriorityMetadata = field(compare=False)
    event: asyncio.Event = field(compare=False, default_factory=asyncio.Event)
    cancelled: bool = field(compare=False, default=False)
    completed: bool = field(compare=False, default=False)

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
        on_release: Optional[Callable[[RequestPriorityMetadata], Any]] = None,
        on_complete: Optional[Callable[[RequestPriorityMetadata], Any]] = None,
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
        self._recheck_interval = 2.0
        self._session_last_served: dict[str, float] = {}
        self._seen_models: set[str] = set()
        self._seen_sources: set[str] = set()
        self._seen_model_source_pairs: set[tuple[str, str]] = set()
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
    ) -> tuple[_QueueItem, asyncio.Event]:
        """Add a request to the queue and return (item, event).

        The event is set when it is this request's turn to proceed.
        The item reference should be passed to ``dequeue()`` when done.

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
                if self._on_release:
                    self._on_release(metadata)

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

        if item.cancelled:
            raise QueueTimeoutError(
                max_wait_sec=effective_timeout,
                actual_wait_sec=metadata.wait_time,
            ) from None

        return item, item.event

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

    async def dequeue(
        self,
        item: Optional[_QueueItem] = None,
    ) -> Optional[RequestPriorityMetadata]:
        """Remove the finishing request and let the next eligible one through.

        If *item* is provided, it is marked completed and lazily removed.
        If *item* is None (legacy path, e.g. Redis fallback), the front
        non-completed item is popped as before.

        Then scans remaining items and releases the first one whose
        resources are available (as determined by the ``_can_proceed``
        callback). Items that can't proceed stay blocked in the queue.
        """
        released_meta: Optional[RequestPriorityMetadata] = None
        async with self._lock:
            if not self._queue:
                if item is not None:
                    item.completed = True
                    if self._on_complete:
                        self._on_complete(item.metadata)
                return None

            if item is not None:
                # Event-based: mark the finishing item as completed
                item.completed = True
                # Compact leading completed items
                while self._queue and self._queue[0].completed:
                    compacted = self._queue.pop(0)
                    self._sizes[compacted.metadata.priority] = max(
                        0, self._sizes[compacted.metadata.priority] - 1
                    )
                    _inc_dequeued(
                        compacted.metadata.priority, compacted.metadata.source.value
                    )
                    _observe_wait(
                        compacted.metadata.wait_time,
                        compacted.metadata.priority,
                        compacted.metadata.source.value,
                    )
                self._update_gauges()
                finished_meta = item.metadata
            else:
                # Legacy: pop front non-completed item
                finished_item = self._queue.pop(0)
                finished_meta = finished_item.metadata
                self._sizes[finished_item.metadata.priority] = sum(
                    1
                    for i in self._queue
                    if i.metadata.priority == finished_item.metadata.priority
                )
                _inc_dequeued(
                    finished_item.metadata.priority, finished_item.metadata.source.value
                )
                _observe_wait(
                    finished_item.metadata.wait_time,
                    finished_item.metadata.priority,
                    finished_item.metadata.source.value,
                )
                self._update_gauges()
                finished_item.event.set()

            if not self._queue:
                if self._on_complete:
                    self._on_complete(finished_meta)
                return finished_meta

            # Collect pending items to check outside lock
            pending: list[tuple[int, RequestPriorityMetadata]] = [
                (i, m.metadata)
                for i, m in enumerate(self._queue)
                if not m.event.is_set() and not m.completed
            ]

        if not pending:
            if self._on_complete:
                self._on_complete(finished_meta)
            return finished_meta

        # Release ALL eligible waiting items (not just one).
        # This enables concurrent request processing up to the model's
        # parallel slot count — the _can_proceed callback enforces limits.
        released_all = await self._release_eligible(pending)

        for meta in released_all:
            if self._on_release:
                self._on_release(meta)
            if meta.session_id:
                self._session_last_served[meta.session_id] = time.monotonic()
        if self._on_complete:
            self._on_complete(finished_meta)
        return finished_meta

    async def _release_eligible(
        self,
        pending: list[tuple[int, RequestPriorityMetadata]],
    ) -> list[RequestPriorityMetadata]:
        """Release all eligible waiting items from pending list.

        Respects priority ordering and session round-robin, but releases
        multiple items per call (up to what _can_proceed allows). This
        enables concurrent request processing up to the model's parallel
        slot count.
        """
        from collections import defaultdict as _dd

        released: list[RequestPriorityMetadata] = []

        by_priority: dict[int, list[tuple[int, RequestPriorityMetadata]]] = _dd(list)
        for idx, metadata in pending:
            by_priority[metadata.priority.value].append((idx, metadata))

        # Process priorities in order (HIGH first)
        for _pri in sorted(by_priority):
            candidates = list(by_priority[_pri])
            while candidates:
                selected_idx = self._select_by_session_rr(candidates)
                selected_meta: Optional[RequestPriorityMetadata] = None
                for c_idx, c_meta in candidates:
                    if c_idx == selected_idx:
                        selected_meta = c_meta
                        break

                can_go = (
                    await self._can_proceed(selected_meta)
                    if self._can_proceed
                    else True
                )
                if can_go:
                    async with self._lock:
                        if (
                            selected_idx < len(self._queue)
                            and self._queue[selected_idx].metadata is selected_meta
                            and not self._queue[selected_idx].event.is_set()
                            and not self._queue[selected_idx].completed
                        ):
                            if self._has_higher_priority_waiting(selected_idx):
                                break
                            self._queue[selected_idx].event.set()
                            released.append(selected_meta)
                # Remove tried candidate regardless of outcome (skip blocked items)
                candidates = [c for c in candidates if c[0] != selected_idx]

        return released

    def _has_higher_priority_waiting(self, released_idx: int) -> bool:
        """Check if any higher-priority item waits behind the released item
        for the SAME model.

        Backpressure exists to prevent low-priority work from starving
        higher-priority work that's stuck waiting on a shared resource —
        but two requests for different models target completely
        independent llama.cpp servers (potentially on different runner
        endpoints), so a HIGH request for model A should not block a
        LOW request for model B from being released.  Without the
        model-id filter, a single transiently-blocked HIGH request
        gates ALL subsequent work in the queue.

        Items without a model_id (None) are treated as competing with
        everything — conservative fallback for legacy paths that don't
        populate the field.

        Must be called while holding self._lock.
        """
        released_meta = self._queue[released_idx].metadata
        if released_meta.priority != Priority.LOW:
            return False
        released_model = released_meta.model_id
        for i in range(released_idx + 1, len(self._queue)):
            cand = self._queue[i].metadata
            if cand.priority != Priority.HIGH:
                continue
            # Different model => independent resources => not blocking.
            # Either side unset => assume competition (conservative).
            if (
                released_model is not None
                and cand.model_id is not None
                and cand.model_id != released_model
            ):
                continue
            return True
        return False

    def _select_by_session_rr(
        self,
        candidates: list[tuple[int, RequestPriorityMetadata]],
    ) -> int:
        """Select the candidate from the least-recently-served session."""
        if len(candidates) <= 1:
            return candidates[0][0]

        def session_score(meta: RequestPriorityMetadata) -> float:
            sid = meta.session_id
            if sid is None:
                return meta.enqueued_at
            return self._session_last_served.get(sid, meta.enqueued_at)

        best_idx = 0
        best_score = session_score(candidates[0][1])
        for i in range(1, len(candidates)):
            score = session_score(candidates[i][1])
            if score < best_score:
                best_score = score
                best_idx = i
        return candidates[best_idx][0]

    async def _remove_item(self, item: _QueueItem) -> None:
        """Remove a specific item (e.g., on timeout)."""
        async with self._lock:
            if item not in self._queue:
                return
            was_first = item is self._queue[0]
            self._queue.remove(item)
            self._sizes[item.metadata.priority] = sum(
                1 for i in self._queue if i.metadata.priority == item.metadata.priority
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
                (i, m.metadata)
                for i, m in enumerate(self._queue)
                if not m.event.is_set() and not m.completed
            ]

        if not pending:
            return

        released_all = await self._release_eligible(pending)
        for meta in released_all:
            if self._on_release:
                self._on_release(meta)
            if meta.session_id:
                self._session_last_served[meta.session_id] = time.monotonic()

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
                    if not m.event.is_set() and not m.completed
                ]
                if not pending:
                    continue

            # Release all eligible items (mirrors dequeue behavior)
            released_all = await self._release_eligible(pending)

            for meta in released_all:
                if self._on_release:
                    self._on_release(meta)
                if meta.session_id:
                    self._session_last_served[meta.session_id] = (
                        time.monotonic()
                    )

    def _update_gauges(self) -> None:
        """Update all queue size gauges (by priority, model, source, cross-tab)."""
        for p in Priority:
            _set_size(p, self._sizes.get(p, 0))

        model_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        model_source_counts: dict[tuple[str, str], int] = {}

        for item in self._queue:
            mid = item.metadata.model_id or "unknown"
            src = item.metadata.source.value
            self._seen_models.add(mid)
            self._seen_sources.add(src)
            self._seen_model_source_pairs.add((mid, src))
            model_counts[mid] = model_counts.get(mid, 0) + 1
            source_counts[src] = source_counts.get(src, 0) + 1
            model_source_counts[(mid, src)] = model_source_counts.get((mid, src), 0) + 1

        # Reset ALL seen labels (including stale ones) to 0, then set current
        for mid in self._seen_models:
            queue_size_by_model.labels(model_id=mid).set(model_counts.get(mid, 0))
        for src in self._seen_sources:
            queue_size_by_source.labels(source=src).set(source_counts.get(src, 0))
        for mid, src in self._seen_model_source_pairs:
            queue_size_by_model_source.labels(model_id=mid, source=src).set(
                model_source_counts.get((mid, src), 0)
            )

    async def cancel_by_session_id(self, session_id: str) -> int:
        """Cancel all queued items matching a session_id.

        Returns the number of items cancelled.
        """
        async with self._lock:
            cancelled = 0
            for item in self._queue:
                if item.metadata.session_id == session_id:
                    item.cancelled = True
                    item.event.set()
                    cancelled += 1
            self._queue = [
                i for i in self._queue if i.metadata.session_id != session_id
            ]
            self._update_gauges()
            for p in Priority:
                self._sizes[p] = sum(1 for i in self._queue if i.metadata.priority == p)
        return cancelled

    @staticmethod
    async def ensure_model_available(model_id: str, user_id: str | None) -> str:
        """Check if model is available on any runner. If not, resolve default.

        Returns the (possibly resolved) model_id.
        """
        try:
            from services.model_service import model_service

            return await model_service.resolve_default_model(
                model_id, user_id or "anonymous"
            )
        except Exception:
            return model_id

    @property
    def size(self) -> int:
        return sum(1 for i in self._queue if not i.completed and not i.cancelled)

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
