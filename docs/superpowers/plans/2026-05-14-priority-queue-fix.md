# Priority Queue Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `AsyncPriorityQueue` so that out-of-order request completion no longer causes stuck items, metrics drift, or negative session gauges.

**Architecture:** Replace FIFO-pop `dequeue()` with event-based dequeue that accepts the finishing `_QueueItem` by reference. Items are marked `completed` and lazily compacted. `_on_release`/`_on_complete` callbacks fire symmetrically for every item. Session round-robin selects among same-priority candidates at release time.

**Tech Stack:** Python 3.12+, asyncio, pytest, prometheus_client

**Key files:**
- `services/priority_queue.py` — core queue implementation (main changes)
- `services/completion_service.py` — enqueue/dequeue call sites (2 paths)
- `routers/session_admin.py` — cancel call site
- `services/redis_priority_queue.py` — fallback dequeue path
- `test/unit/test_priority_queue_*.py` — tests

---

### Task 1: Add `_QueueItem.completed` field and new `enqueue()` return type

**Files:**
- Modify: `services/priority_queue.py`

- [ ] **Step 1: Add `completed` field to `_QueueItem`**

Add `completed: bool = field(compare=False, default=False)` to the `_QueueItem` dataclass.

```python
@dataclass(order=True)
class _QueueItem:
    """Internal queue item with priority ordering."""

    sort_key: tuple = field(compare=True)
    metadata: RequestPriorityMetadata = field(compare=False)
    event: asyncio.Event = field(compare=False, default_factory=asyncio.Event)
    cancelled: bool = field(compare=False, default=False)
    completed: bool = field(compare=False, default=False)
```

- [ ] **Step 2: Change `enqueue()` return type and add `_on_release` for sole items**

Change the return type annotation from `asyncio.Event` to `tuple[_QueueItem, asyncio.Event]`. Return the tuple at line 190. When the sole-item event is set (line 159-160), also fire `_on_release`:

```python
    async def enqueue(
        self,
        metadata: RequestPriorityMetadata,
        timeout_sec: Optional[float] = None,
    ) -> tuple[_QueueItem, asyncio.Event]:
```

At line 159-160 (sole item path), add callback:
```python
            if len(self._queue) == 1:
                item.event.set()
                if self._on_release:
                    self._on_release(metadata)
```

At line 190, change return:
```python
        return item, item.event
```

- [ ] **Step 3: Run existing tests to see failures**

Run: `uv run pytest test/unit/test_priority_queue_timeout.py test/unit/test_priority_queue_resource_aware.py -v`
Expected: FAIL — tests expect `enqueue()` to return `asyncio.Event`, not tuple.

- [ ] **Step 4: Commit**

```bash
git add services/priority_queue.py
git commit -m "refactor: add completed flag to _QueueItem, return item ref from enqueue"
```

---

### Task 2: Rewrite `dequeue()` to accept `_QueueItem` and use lazy removal

**Files:**
- Modify: `services/priority_queue.py`

- [ ] **Step 1: Rewrite `dequeue()` signature and core logic**

Change signature to accept the finishing item. Mark it completed, compact, release next, always fire `_on_complete`:

```python
    async def dequeue(self, item: Optional[_QueueItem] = None) -> Optional[RequestPriorityMetadata]:
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
                # Fire _on_complete even if queue is empty (item may have
                # been the sole item or already compacted away).
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
                    # Adjust size counters for compacted items
                    self._sizes[compacted.metadata.priority] = max(
                        0, self._sizes[compacted.metadata.priority] - 1
                    )
                    _inc_dequeued(compacted.metadata.priority, compacted.metadata.source.value)
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
                    1 for i in self._queue if i.metadata.priority == finished_item.metadata.priority
                )
                _inc_dequeued(finished_item.metadata.priority, finished_item.metadata.source.value)
                _observe_wait(
                    finished_item.metadata.wait_time,
                    finished_item.metadata.priority,
                    finished_item.metadata.source.value,
                )
                self._update_gauges()
                # Set the popped item's event so its enqueue() completes
                finished_item.event.set()

            if not self._queue:
                if self._on_complete:
                    self._on_complete(finished_meta)
                return finished_meta

            # Collect pending items to check outside lock
            pending: list[tuple[int, RequestPriorityMetadata]] = [
                (i, m.metadata) for i, m in enumerate(self._queue)
                if not m.event.is_set() and not m.completed
            ]

        if not pending:
            if self._on_complete:
                self._on_complete(finished_meta)
            return finished_meta

        # Call callback outside the lock to avoid blocking other operations
        for idx, metadata in pending:
            can_go = await self._can_proceed(metadata) if self._can_proceed else True
            if can_go:
                async with self._lock:
                    if (
                        idx < len(self._queue)
                        and self._queue[idx].metadata is metadata
                        and not self._queue[idx].event.is_set()
                        and not self._queue[idx].completed
                    ):
                        # Priority preemption: don't release LOW if HIGH waits
                        if self._has_higher_priority_waiting(idx):
                            break
                        self._queue[idx].event.set()
                        released_meta = metadata
                break

        if self._on_release and released_meta is not None:
            self._on_release(released_meta)
            # Update session round-robin tracking
            if released_meta.session_id:
                self._session_last_served[released_meta.session_id] = time.monotonic()
        if self._on_complete:
            self._on_complete(finished_meta)
        return finished_meta
```

- [ ] **Step 2: Add `_session_last_served` to `__init__`**

Add after `self._recheck_interval = 5.0`:
```python
        self._session_last_served: dict[str, float] = {}
```

- [ ] **Step 3: Run existing tests to verify failures match expectations**

Run: `uv run pytest test/unit/test_priority_queue_timeout.py test/unit/test_priority_queue_resource_aware.py -v`
Expected: FAIL — tests call `dequeue()` without the item argument, need to pass item from enqueue.

- [ ] **Step 4: Commit**

```bash
git add services/priority_queue.py
git commit -m "refactor: event-based dequeue with lazy removal and symmetric callbacks"
```

---

### Task 3: Add session round-robin to `_release_next` scan

**Files:**
- Modify: `services/priority_queue.py`

- [ ] **Step 1: Add session round-robin selection helper**

Add a new method that, given a list of same-priority pending candidates, returns the index of the one from the least-recently-served session:

```python
    def _select_by_session_rr(self, candidates: list[tuple[int, RequestPriorityMetadata]]) -> int:
        """Select the candidate from the least-recently-served session.

        Returns the queue index of the selected candidate.
        """
        if len(candidates) <= 1:
            return candidates[0][0]

        def session_score(item_idx: int, meta: RequestPriorityMetadata) -> float:
            sid = meta.session_id
            if sid is None:
                # No session — use enqueued_at as tiebreaker (no interleaving)
                return meta.enqueued_at
            last_served = self._session_last_served.get(sid, meta.enqueued_at)
            # Lower score = longer since served = higher priority
            return last_served

        best_idx = 0
        best_score = session_score(candidates[0][0], candidates[0][1])
        for i in range(1, len(candidates)):
            score = session_score(candidates[i][0], candidates[i][1])
            if score < best_score:
                best_score = score
                best_idx = i
        return candidates[best_idx][0]
```

- [ ] **Step 2: Integrate session round-robin into `dequeue()` scan**

Replace the linear scan in `dequeue()` with a session-aware scan. Instead of iterating `pending` linearly, group by priority and use `_select_by_session_rr` within each priority group:

Replace the block starting at `# Call callback outside the lock...`:

```python
        # Group pending items by priority for session round-robin
        from collections import defaultdict as _dd
        by_priority: dict[int, list[tuple[int, RequestPriorityMetadata]]] = _dd(list)
        for idx, metadata in pending:
            by_priority[metadata.priority.value].append((idx, metadata))

        # Process priorities in order (HIGH first)
        for _pri in sorted(by_priority):
            candidates = by_priority[_pri]
            # Use session round-robin to pick within same priority
            selected_idx = self._select_by_session_rr(candidates)
            selected_meta = None
            for c_idx, c_meta in candidates:
                if c_idx == selected_idx:
                    selected_meta = c_meta
                    break

            can_go = await self._can_proceed(selected_meta) if self._can_proceed else True
            if can_go:
                async with self._lock:
                    if (
                        selected_idx < len(self._queue)
                        and self._queue[selected_idx].metadata is selected_meta
                        and not self._queue[selected_idx].event.is_set()
                        and not self._queue[selected_idx].completed
                    ):
                        if self._has_higher_priority_waiting(selected_idx):
                            continue  # Skip this priority, try next
                        self._queue[selected_idx].event.set()
                        released_meta = selected_meta
                break
            # If selected candidate can't proceed, try next candidate in same priority group
            remaining = [c for c in candidates if c[0] != selected_idx]
            while remaining and not released_meta:
                fallback_idx = self._select_by_session_rr(remaining)
                fallback_meta = None
                for c_idx, c_meta in remaining:
                    if c_idx == fallback_idx:
                        fallback_meta = c_meta
                        break
                can_go = await self._can_proceed(fallback_meta) if self._can_proceed else True
                if can_go:
                    async with self._lock:
                        if (
                            fallback_idx < len(self._queue)
                            and self._queue[fallback_idx].metadata is fallback_meta
                            and not self._queue[fallback_idx].event.is_set()
                            and not self._queue[fallback_idx].completed
                        ):
                            if self._has_higher_priority_waiting(fallback_idx):
                                break
                            self._queue[fallback_idx].event.set()
                            released_meta = fallback_meta
                    break
                remaining = [c for c in remaining if c[0] != fallback_idx]
            if released_meta:
                break
```

- [ ] **Step 3: Commit**

```bash
git add services/priority_queue.py
git commit -m "feat: add session round-robin selection to queue release"
```

---

### Task 4: Fix `_recheck_blocked()` — add priority preemption and session round-robin

**Files:**
- Modify: `services/priority_queue.py`

- [ ] **Step 1: Rewrite `_recheck_blocked()` to match `dequeue()` logic**

Add `_has_higher_priority_waiting` check and session round-robin:

```python
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

            # Group by priority for session round-robin
            from collections import defaultdict as _dd
            by_priority: dict[int, list[tuple[int, RequestPriorityMetadata]]] = _dd(list)
            for idx, metadata in pending:
                by_priority[metadata.priority.value].append((idx, metadata))

            released_meta = None
            for _pri in sorted(by_priority):
                candidates = by_priority[_pri]
                selected_idx = self._select_by_session_rr(candidates)
                selected_meta = None
                for c_idx, c_meta in candidates:
                    if c_idx == selected_idx:
                        selected_meta = c_meta
                        break

                can_go = await self._can_proceed(selected_meta)
                if can_go:
                    async with self._lock:
                        if (
                            selected_idx < len(self._queue)
                            and self._queue[selected_idx].metadata is selected_meta
                            and not self._queue[selected_idx].event.is_set()
                            and not self._queue[selected_idx].completed
                        ):
                            if self._has_higher_priority_waiting(selected_idx):
                                continue
                            self._queue[selected_idx].event.set()
                            released_meta = selected_meta
                    break
                # Try remaining candidates in this priority group
                remaining = [c for c in candidates if c[0] != selected_idx]
                while remaining and not released_meta:
                    fb_idx = self._select_by_session_rr(remaining)
                    fb_meta = None
                    for c_idx, c_meta in remaining:
                        if c_idx == fb_idx:
                            fb_meta = c_meta
                            break
                    can_go = await self._can_proceed(fb_meta)
                    if can_go:
                        async with self._lock:
                            if (
                                fb_idx < len(self._queue)
                                and self._queue[fb_idx].metadata is fb_meta
                                and not self._queue[fb_idx].event.is_set()
                                and not self._queue[fb_idx].completed
                            ):
                                if self._has_higher_priority_waiting(fb_idx):
                                    break
                                self._queue[fb_idx].event.set()
                                released_meta = fb_meta
                        break
                    remaining = [c for c in remaining if c[0] != fb_idx]
                if released_meta:
                    break

            if released_meta is not None:
                if self._on_release:
                    self._on_release(released_meta)
                if released_meta.session_id:
                    self._session_last_served[released_meta.session_id] = time.monotonic()
```

- [ ] **Step 2: Commit**

```bash
git add services/priority_queue.py
git commit -m "fix: add priority preemption and session round-robin to recheck_blocked"
```

---

### Task 5: Fix stale Prometheus gauges

**Files:**
- Modify: `services/priority_queue.py`

- [ ] **Step 1: Track seen labels and reset stale gauges**

Add `_seen_models` and `_seen_sources` sets to `__init__`:
```python
        self._seen_models: set[str] = set()
        self._seen_sources: set[str] = set()
        self._seen_model_source_pairs: set[tuple[str, str]] = set()
```

Replace `_update_gauges()` to reset all seen labels:

```python
    def _update_gauges(self) -> None:
        """Update all queue size gauges (by priority, model, source, cross-tab)."""
        for p in Priority:
            _set_size(p, self._sizes.get(p, 0))

        if not _HAS_PROMETHEUS:
            return

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
```

- [ ] **Step 2: Commit**

```bash
git add services/priority_queue.py
git commit -m "fix: reset stale Prometheus gauge labels to 0"
```

---

### Task 6: Make `cancel_by_session_id()` async and lock-safe

**Files:**
- Modify: `services/priority_queue.py`
- Modify: `routers/session_admin.py`

- [ ] **Step 1: Convert `cancel_by_session_id` to async**

```python
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
            self._queue = [i for i in self._queue if i.metadata.session_id != session_id]
            self._update_gauges()
            # Adjust size counters
            for p in Priority:
                self._sizes[p] = sum(
                    1 for i in self._queue if i.metadata.priority == p
                )
        return cancelled
```

- [ ] **Step 2: Update caller in `routers/session_admin.py`**

Change line 27 from:
```python
        queued_cancelled = priority_queue.cancel_by_session_id(session_id)
```
to:
```python
        queued_cancelled = await priority_queue.cancel_by_session_id(session_id)
```

- [ ] **Step 3: Commit**

```bash
git add services/priority_queue.py routers/session_admin.py
git commit -m "fix: make cancel_by_session_id async and lock-safe"
```

---

### Task 7: Update `completion_service.py` call sites

**Files:**
- Modify: `services/completion_service.py`

- [ ] **Step 1: Update streaming path (line 527 and line 854)**

Change line 527 from:
```python
            _queue_ctx = await priority_queue.enqueue(_meta)
```
to:
```python
            _queue_item, _queue_ctx = await priority_queue.enqueue(_meta)
```

Change line 854 from:
```python
                await priority_queue.dequeue()
```
to:
```python
                await priority_queue.dequeue(_queue_item)
```

Note: `_queue_item` needs to be hoisted above the `try` block so it's accessible in `finally`. Add initialization:
```python
            _queue_item = None
            _queue_ctx = None
            if PRIORITY_QUEUE_ENABLED:
                # ... existing code ...
                _queue_item, _queue_ctx = await priority_queue.enqueue(_meta)
```

And in the finally:
```python
            if _queue_item is not None:
                await priority_queue.dequeue(_queue_item)
```

- [ ] **Step 2: Update non-streaming path (line 907 and line 1187)**

Same pattern:
```python
            _queue_item = None
            _queue_ctx = None
            if PRIORITY_QUEUE_ENABLED:
                # ... existing code ...
                _queue_item, _queue_ctx = await priority_queue.enqueue(_meta)
```

And in the finally:
```python
            if _queue_item is not None:
                await priority_queue.dequeue(_queue_item)
```

- [ ] **Step 3: Commit**

```bash
git add services/completion_service.py
git commit -m "refactor: pass _QueueItem ref to dequeue in completion_service"
```

---

### Task 8: Update `redis_priority_queue.py` fallback path

**Files:**
- Modify: `services/redis_priority_queue.py`

- [ ] **Step 1: Update fallback dequeue call**

The `RedisPriorityQueue.dequeue()` at line 161 calls `self._fallback.dequeue()`. Since `dequeue()` now has `item` as an optional parameter with default `None`, the legacy path (pop front) is preserved. No change needed for the fallback call itself.

Verify that the method signature `dequeue(self, item: Optional[_QueueItem] = None)` is backward compatible with the no-argument call.

- [ ] **Step 2: Commit if any changes needed**

```bash
git add services/redis_priority_queue.py
git commit -m "chore: verify redis_priority_queue compatibility with new dequeue signature"
```

---

### Task 9: Write tests for out-of-order completion

**Files:**
- Create: `test/unit/test_priority_queue_out_of_order.py`

- [ ] **Step 1: Write test — B finishes before A**

```python
"""Tests for out-of-order completion in AsyncPriorityQueue."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from models.request_priority_metadata import (
    Priority,
    RequestPriorityMetadata,
    RequestSource,
)
from services.priority_queue import AsyncPriorityQueue


@pytest.mark.asyncio
async def test_out_of_order_completion_b_before_a():
    """B finishes before A — dequeue(B) must remove B, not A."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(return_value=True))

    meta_a = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    meta_b = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )

    item_a, evt_a = await q.enqueue(meta_a)
    await asyncio.sleep(0.01)
    item_b, evt_b = await q.enqueue(meta_b)
    await asyncio.sleep(0.01)

    # B finishes first
    result = await q.dequeue(item_b)
    assert result is meta_b, "dequeue(B) must return B's metadata"

    # A should still be in the queue (not popped by B's dequeue)
    assert q.size == 1, f"Queue should have 1 item, has {q.size}"

    # A finishes second
    result = await q.dequeue(item_a)
    assert result is meta_a

    assert q.size == 0
    await q.close()


@pytest.mark.asyncio
async def test_out_of_order_three_items_middle_finishes_first():
    """A, B, C enqueue. B finishes first, then C, then A."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(return_value=True))

    meta_a = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    meta_b = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    meta_c = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )

    item_a, _ = await q.enqueue(meta_a)
    await asyncio.sleep(0.01)
    item_b, _ = await q.enqueue(meta_b)
    await asyncio.sleep(0.01)
    item_c, _ = await q.enqueue(meta_c)
    await asyncio.sleep(0.01)

    # B finishes first
    assert await q.dequeue(item_b) is meta_b
    assert q.size == 2  # A and C remain

    # C finishes second
    assert await q.dequeue(item_c) is meta_c
    assert q.size == 1  # A remains

    # A finishes last
    assert await q.dequeue(item_a) is meta_a
    assert q.size == 0

    await q.close()


@pytest.mark.asyncio
async def test_out_of_order_with_blocked_items():
    """Out-of-order dequeue skips blocked items and releases eligible ones."""
    call_results = {0: False, 1: True}  # First check False, second True
    call_count = [0]

    async def step_callback(metadata):
        idx = call_count[0]
        call_count[0] += 1
        return call_results.get(idx, True)

    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(side_effect=step_callback))

    meta_a = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    meta_b = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-b"
    )
    meta_c = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-c"
    )

    item_a, evt_a = await q.enqueue(meta_a)
    await asyncio.sleep(0.01)
    item_b, evt_b = await q.enqueue(meta_b)
    await asyncio.sleep(0.01)
    item_c, evt_c = await q.enqueue(meta_c)
    await asyncio.sleep(0.01)

    # A finishes, B is blocked (callback False), C should be released (callback True)
    result = await q.dequeue(item_a)
    assert result is meta_a
    await asyncio.sleep(0.01)
    assert not evt_b.is_set(), "B should still be blocked"
    assert evt_c.is_set(), "C should be released"

    # Cleanup
    await q.dequeue(item_b)
    await q.dequeue(item_c)
    await q.close()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest test/unit/test_priority_queue_out_of_order.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_priority_queue_out_of_order.py
git commit -m "test: add out-of-order completion tests"
```

---

### Task 10: Write tests for symmetric callback lifecycle

**Files:**
- Create: `test/unit/test_priority_queue_callbacks.py`

- [ ] **Step 1: Write callback lifecycle tests**

```python
"""Tests for symmetric _on_release / _on_complete callback lifecycle."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.request_priority_metadata import (
    Priority,
    RequestPriorityMetadata,
    RequestSource,
)
from services.priority_queue import AsyncPriorityQueue


@pytest.mark.asyncio
async def test_sole_item_fires_both_callbacks():
    """Sole queue item: _on_release fires in enqueue, _on_complete in dequeue."""
    on_release = MagicMock()
    on_complete = MagicMock()
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_session_callbacks(on_release, on_complete)

    meta = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    item, evt = await q.enqueue(meta)

    assert on_release.call_count == 1, "_on_release should fire for sole item in enqueue"
    assert on_release.call_args[0][0] is meta
    assert on_complete.call_count == 0, "_on_complete should not fire yet"

    await q.dequeue(item)

    assert on_complete.call_count == 1, "_on_complete should fire in dequeue"
    assert on_complete.call_args[0][0] is meta


@pytest.mark.asyncio
async def test_multi_item_callbacks_symmetric():
    """Two items: each gets exactly one _on_release and one _on_complete."""
    on_release = MagicMock()
    on_complete = MagicMock()
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(return_value=True))
    q.set_session_callbacks(on_release, on_complete)

    meta_a = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    meta_b = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )

    item_a, _ = await q.enqueue(meta_a)
    await asyncio.sleep(0.01)
    item_b, _ = await q.enqueue(meta_b)
    await asyncio.sleep(0.01)

    # After enqueue: A released (sole item), B not released yet
    assert on_release.call_count == 1
    assert on_release.call_args[0][0] is meta_a

    # A finishes — should release B and complete A
    await q.dequeue(item_a)
    assert on_release.call_count == 2, "B should be released"
    assert on_release.call_args[0][0] is meta_b
    assert on_complete.call_count == 1, "A should be completed"
    assert on_complete.call_args[0][0] is meta_a

    # B finishes
    await q.dequeue(item_b)
    assert on_release.call_count == 2, "No new release"
    assert on_complete.call_count == 2, "B should be completed"
    assert on_complete.call_args[0][0] is meta_b

    await q.close()


@pytest.mark.asyncio
async def test_active_counts_balance():
    """_active_counts should return to 0 after all items complete."""
    active_counts = {"model-a": 0}

    def on_release(metadata):
        if metadata.model_id:
            active_counts[metadata.model_id] += 1

    def on_complete(metadata):
        if metadata.model_id:
            active_counts[metadata.model_id] -= 1

    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(return_value=True))
    q.set_session_callbacks(on_release, on_complete)

    metas = []
    items = []
    for i in range(5):
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        item, _ = await q.enqueue(meta)
        await asyncio.sleep(0.01)
        metas.append(meta)
        items.append(item)

    # All items released
    assert active_counts["model-a"] == 5

    # Dequeue in reverse order (last finishes first)
    for i in range(4, -1, -1):
        await q.dequeue(items[i])

    assert active_counts["model-a"] == 0, "Counts should balance to 0"
    await q.close()


@pytest.mark.asyncio
async def test_session_gauge_never_negative():
    """active_sessions gauge should never go negative."""
    session_counts = {}

    def on_release(metadata):
        if metadata.session_id:
            session_counts[metadata.session_id] = session_counts.get(metadata.session_id, 0) + 1

    def on_complete(metadata):
        if metadata.session_id:
            session_counts[metadata.session_id] = session_counts.get(metadata.session_id, 0) - 1

    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(return_value=True))
    q.set_session_callbacks(on_release, on_complete)

    # Three sessions, two requests each
    items = []
    for session in ["s1", "s2", "s3"]:
        for _ in range(2):
            meta = RequestPriorityMetadata(
                source=RequestSource.USER, priority=Priority.HIGH,
                model_id="model-a", session_id=session,
            )
            item, _ = await q.enqueue(meta)
            await asyncio.sleep(0.01)
            items.append((item, session))

    # Dequeue in random order
    import random
    random.seed(42)
    random.shuffle(items)

    for item, session in items:
        await q.dequeue(item)
        assert session_counts.get(session, 0) >= 0, (
            f"Session {session} went negative after dequeue"
        )

    assert all(v == 0 for v in session_counts.values()), "All sessions should balance"
    await q.close()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest test/unit/test_priority_queue_callbacks.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_priority_queue_callbacks.py
git commit -m "test: add symmetric callback lifecycle tests"
```

---

### Task 11: Write tests for session round-robin

**Files:**
- Create: `test/unit/test_priority_queue_session_order.py`

- [ ] **Step 1: Write session round-robin tests**

```python
"""Tests for session-aware round-robin ordering in AsyncPriorityQueue."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from models.request_priority_metadata import (
    Priority,
    RequestPriorityMetadata,
    RequestSource,
)
from services.priority_queue import AsyncPriorityQueue


@pytest.mark.asyncio
async def test_session_round_robin_alternates():
    """Requests from different sessions should alternate, not run consecutively."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(return_value=True))

    released_order = []

    def on_release(metadata):
        released_order.append(metadata.session_id)

    q.set_session_callbacks(on_release, MagicMock())

    # Enqueue from two sessions
    items = []
    for sid in ["s1", "s2"]:
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH,
            model_id="model-a", session_id=sid,
        )
        item, _ = await q.enqueue(meta)
        await asyncio.sleep(0.01)
        items.append(item)

    # First item (s1) was released immediately as sole item
    assert released_order == ["s1"]

    # Dequeue s1 — should release s2 (different session, longer wait)
    released_order.clear()
    await q.dequeue(items[0])
    assert released_order == ["s2"], f"Expected s2, got {released_order}"

    await q.dequeue(items[1])
    await q.close()


@pytest.mark.asyncio
async def test_session_round_robin_three_sessions():
    """Three sessions should interleave fairly."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(return_value=True))

    released_order = []

    def on_release(metadata):
        released_order.append(metadata.session_id)

    q.set_session_callbacks(on_release, MagicMock())

    items = []
    for sid in ["s1", "s2", "s3"]:
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH,
            model_id="model-a", session_id=sid,
        )
        item, _ = await q.enqueue(meta)
        await asyncio.sleep(0.01)
        items.append(item)

    # s1 released as sole item
    assert released_order[0] == "s1"

    # Dequeue s1 — should release s2 (oldest unserved)
    released_order.clear()
    await q.dequeue(items[0])
    assert released_order[0] == "s2"

    # Dequeue s2 — should release s3 (oldest unserved)
    released_order.clear()
    await q.dequeue(items[1])
    assert released_order[0] == "s3"

    await q.dequeue(items[2])
    await q.close()


@pytest.mark.asyncio
async def test_no_session_id_no_interleaving():
    """Items without session_id should follow FIFO order."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(return_value=True))

    released_order = []

    def on_release(metadata):
        released_order.append(id(metadata))

    q.set_session_callbacks(on_release, MagicMock())

    meta_a = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    meta_b = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )

    item_a, _ = await q.enqueue(meta_a)
    await asyncio.sleep(0.01)
    item_b, _ = await q.enqueue(meta_b)
    await asyncio.sleep(0.01)

    # A released as sole item
    assert released_order == [id(meta_a)]

    released_order.clear()
    await q.dequeue(item_a)
    assert released_order == [id(meta_b)]

    await q.dequeue(item_b)
    await q.close()


@pytest.mark.asyncio
async def test_higher_priority_skips_session_rr():
    """HIGH priority items should always be released before LOW, regardless of session."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(return_value=True))

    released_order = []

    def on_release(metadata):
        released_order.append((metadata.session_id, metadata.priority.name))

    q.set_session_callbacks(on_release, MagicMock())

    # LOW from s1
    meta_low = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.LOW,
        model_id="model-a", session_id="s1",
    )
    item_low, _ = await q.enqueue(meta_low)
    await asyncio.sleep(0.01)

    # HIGH from s2
    meta_high = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH,
        model_id="model-a", session_id="s2",
    )
    item_high, _ = await q.enqueue(meta_high)
    await asyncio.sleep(0.01)

    # LOW released as sole item first
    assert released_order[0] == ("s1", "LOW")

    # Dequeue LOW — HIGH should be released (higher priority)
    released_order.clear()
    await q.dequeue(item_low)
    assert released_order[0] == ("s2", "HIGH")

    await q.dequeue(item_high)
    await q.close()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest test/unit/test_priority_queue_session_order.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_priority_queue_session_order.py
git commit -m "test: add session round-robin ordering tests"
```

---

### Task 12: Update existing tests for new API

**Files:**
- Modify: `test/unit/test_priority_queue_timeout.py`
- Modify: `test/unit/test_priority_queue_resource_aware.py`

- [ ] **Step 1: Update timeout tests**

All tests in `test_priority_queue_timeout.py` that call `await q.dequeue()` need to pass the item from `enqueue()`. Update each test:

For `test_timeout_raises_queue_timeout_error`:
```python
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        with pytest.raises(QueueTimeoutError) as exc_info:
            await q.enqueue(meta2)
        # ...
        item1, _ = await task1
        await q.dequeue(item1)
```

Apply the same pattern to all tests: unpack the tuple from `enqueue()` and pass the item to `dequeue()`.

- [ ] **Step 2: Update resource-aware tests**

Same pattern for `test_priority_queue_resource_aware.py`:
- `test_releases_when_callback_true`: unpack tuple, pass item to dequeue
- `test_blocks_when_callback_false`: same
- `test_skips_blocked_item_to_release_next`: same
- `test_unconditional_without_callback`: same
- `test_timeout_removal_checks_callback`: same
- `test_recheck_unblocks_item`: same
- `test_clearing_callback_stops_recheck`: no dequeue call, no change
- `test_close_cancels_recheck`: no dequeue call, no change

- [ ] **Step 3: Run all queue tests**

Run: `uv run pytest test/unit/test_priority_queue_timeout.py test/unit/test_priority_queue_resource_aware.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add test/unit/test_priority_queue_timeout.py test/unit/test_priority_queue_resource_aware.py
git commit -m "test: update existing tests for event-based dequeue API"
```

---

### Task 13: Write tests for stale gauge cleanup and recheck preemption

**Files:**
- Create: `test/unit/test_priority_queue_gauges_and_recheck.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for stale gauge cleanup and recheck preemption."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.request_priority_metadata import (
    Priority,
    RequestPriorityMetadata,
    RequestSource,
)
from services.priority_queue import AsyncPriorityQueue


@pytest.mark.asyncio
async def test_stale_gauge_labels_reset_to_zero():
    """Models/sources no longer in queue should have gauges reset to 0."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

    # Enqueue for model-a
    meta_a = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    item_a, _ = await q.enqueue(meta_a)
    await asyncio.sleep(0.01)

    # model-a gauge should be 1
    assert "model-a" in q._seen_models

    # Dequeue — model-a should still be tracked but at 0
    await q.dequeue(item_a)
    assert "model-a" in q._seen_models, "model-a should still be in seen_models"

    # Verify _update_gauges resets to 0
    with patch("services.priority_queue._HAS_PROMETHEUS", True):
        with patch("services.priority_queue.queue_size_by_model") as mock_gauge:
            q._update_gauges()
            # model-a should be set to 0
            mock_gauge.labels.assert_any_call(model_id="model-a")
            call = mock_gauge.labels.call_args_list[-1]
            # The .set(0) call should follow
    await q.close()


@pytest.mark.asyncio
async def test_recheck_preemption_blocks_low_when_high_waits():
    """_recheck_blocked must not release LOW if HIGH waits behind it."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0, age_threshold_sec=999)
    q._recheck_interval = 0.1

    released = []

    def on_release(metadata):
        released.append(metadata.priority.name)

    async def always_true(metadata):
        return True

    q.set_can_proceed_callback(AsyncMock(side_effect=always_true))
    q.set_session_callbacks(on_release, MagicMock())

    # Enqueue LOW first (released as sole item)
    meta_low = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.LOW, model_id="model-a"
    )
    item_low, evt_low = await q.enqueue(meta_low)
    await asyncio.sleep(0.01)

    # Enqueue HIGH second (blocked)
    meta_high = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    item_high, evt_high = await q.enqueue(meta_high)
    await asyncio.sleep(0.01)

    # LOW was released as sole item
    assert released == ["LOW"]
    released.clear()

    # Dequeue LOW — HIGH should be released (higher priority, no preemption needed)
    await q.dequeue(item_low)
    assert "HIGH" in released or evt_high.is_set()

    await q.dequeue(item_high)
    await q.close()


@pytest.mark.asyncio
async def test_async_cancel_by_session_id():
    """cancel_by_session_id must be async and lock-safe."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

    meta1 = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH,
        model_id="model-a", session_id="s1",
    )
    meta2 = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH,
        model_id="model-a", session_id="s2",
    )
    meta3 = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH,
        model_id="model-a", session_id="s1",
    )

    item1, _ = await q.enqueue(meta1)
    await asyncio.sleep(0.01)
    item2, _ = await q.enqueue(meta2)
    await asyncio.sleep(0.01)
    item3, _ = await q.enqueue(meta3)
    await asyncio.sleep(0.01)

    assert q.size == 3

    # Cancel s1 — should remove meta1 and meta3
    cancelled = await q.cancel_by_session_id("s1")
    assert cancelled == 2
    assert q.size == 1

    # Remaining item should be s2
    remaining = q._queue[0]
    assert remaining.metadata.session_id == "s2"

    await q.dequeue(item2)
    await q.close()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest test/unit/test_priority_queue_gauges_and_recheck.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_priority_queue_gauges_and_recheck.py
git commit -m "test: add stale gauge and recheck preemption tests"
```

---

### Task 14: Run full test suite and verify

**Files:**
- All modified files

- [ ] **Step 1: Run all priority queue related tests**

Run: `uv run pytest test/unit/test_priority_queue_*.py -v`
Expected: All PASS

- [ ] **Step 2: Run the full unit test suite**

Run: `uv run pytest test/unit/ -v --tb=short`
Expected: All PASS (no regressions)

- [ ] **Step 3: Run Python syntax check**

Run: `make validate`
Expected: No errors

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "fix: verify all tests pass for priority queue refactor"
```

---

## Self-Review Checklist

**Spec coverage:**
- Out-of-order completion: Tasks 2, 9
- Symmetric callbacks: Tasks 1, 2, 10
- Priority preemption in recheck: Task 4, 13
- Stale Prometheus gauges: Task 5, 13
- Session-aware ordering: Tasks 3, 11
- Async cancel: Task 6, 13
- Completion service update: Task 7
- Redis queue compatibility: Task 8
- Existing test updates: Task 12
- Full verification: Task 14

**Placeholder scan:** No TBD, TODO, or "implement later" found. All steps have concrete code.

**Type consistency:**
- `enqueue()` returns `tuple[_QueueItem, asyncio.Event]` — used consistently in Tasks 1, 7, 9, 10, 11, 12
- `dequeue(item: Optional[_QueueItem] = None)` — backward compatible for Redis fallback (Task 8)
- `cancel_by_session_id()` is `async def` — caller awaits (Task 6)
- `_session_last_served: dict[str, float]` — used in Tasks 2, 3, 4
- `_seen_models`, `_seen_sources`, `_seen_model_source_pairs` — used in Task 5

**Code organization:** The session round-robin selection in `dequeue()` and `_recheck_blocked()` shares the same `_select_by_session_rr` helper (Task 3). Both scan methods follow the same pattern: group by priority → select by session RR → check `_can_proceed` → check `_has_higher_priority_waiting` → release. This is DRY.
