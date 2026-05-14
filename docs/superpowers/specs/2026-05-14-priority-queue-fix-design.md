# Priority Queue Fix Design

**Date:** 2026-05-14
**Status:** Approved

## Problem Statement

The `AsyncPriorityQueue` in `services/priority_queue.py` has several bugs that cause queued requests to become stuck even when servers are idle:

1. **Out-of-order completion:** `dequeue()` always pops index 0 (FIFO), but requests finish out of order. When a later-enqueued request finishes before an earlier one, `dequeue()` removes the wrong item from the queue, causing permanent misalignment.
2. **Asymmetric callbacks:** `_on_release` and `_on_complete` fire asymmetrically for sole-queue items and last items, causing `_active_counts` drift and `active_sessions` gauge to go negative.
3. **Missing priority preemption in recheck:** `_recheck_blocked()` can release LOW items ahead of waiting HIGH items.
4. **Stale Prometheus gauges:** `queue_size_by_model`, `queue_size_by_source`, and `queue_size_by_model_source` retain stale values for models/sources that are no longer in the queue.
5. **No session-aware ordering:** Same-priority requests are strict FIFO, allowing long-running sessions to starve new sessions.
6. **Race condition in `cancel_by_session_id()`:** Modifies the queue without holding the async lock.

## Design

### 1. Event-Based Dequeue with Lazy Removal

**Current behavior:** `dequeue()` pops `self._queue[0]`, assuming the front item finished first.

**New behavior:** `dequeue(item: _QueueItem)` accepts the finishing item by reference. The item is marked `completed=True`. Leading completed items are lazily compacted away. The next eligible waiting item is released by scanning the queue in priority order.

#### Changes to `_QueueItem`

```python
@dataclass(order=True)
class _QueueItem:
    sort_key: tuple = field(compare=True)
    metadata: RequestPriorityMetadata = field(compare=False)
    event: asyncio.Event = field(compare=False, default_factory=asyncio.Event)
    cancelled: bool = field(compare=False, default=False)
    completed: bool = field(compare=False, default=False)  # NEW
```

#### Changes to `enqueue()`

- Return `(_QueueItem, asyncio.Event)` tuple instead of just `asyncio.Event`.
- When setting event immediately for sole queue item (line 159-160), also call `self._on_release(metadata)` if set — ensures symmetric callback lifecycle.

#### Changes to `dequeue(item: _QueueItem)`

- Mark `item.completed = True`.
- Compact leading completed items: `while self._queue and self._queue[0].completed: self._queue.pop(0)`.
- Update size counters after compact.
- If queue is empty after compact: call `_on_complete(item.metadata)` and return.
- Otherwise: scan remaining items for the next to release (see `_release_next()` below).
- Always call `_on_complete(item.metadata)` before returning.

#### `_release_next()` — Internal Helper

Scans the queue from index 0 forward looking for the first item whose:
1. `event.is_set() == False` (not yet released)
2. `_can_proceed(metadata)` returns `True` (resources available)
3. No higher-priority item waits behind it (`_has_higher_priority_waiting`)
4. **Session round-robin:** Among same-priority candidates at the front, prefer the item from the least-recently-served session.

When found: set its event, call `_on_release(metadata)`, update `_session_last_served[session_id]`.

### 2. Symmetric Callback Lifecycle

Every item gets exactly one `_on_release` and one `_on_complete`:

| Path | `_on_release` fires | `_on_complete` fires |
|------|---------------------|---------------------|
| Sole item (proceeds immediately) | In `enqueue()` at line 160 | In `dequeue()` always |
| Waiting item (released by dequeue/recheck) | In `_release_next()` | In `dequeue()` always |
| Timeout/cancel | Never (never released) | Never (never dequeued normally) |
| Last item in queue | In `_release_next()` (from prior dequeue) | In `dequeue()` always |

### 3. Session-Aware Round-Robin at Release Time

The queue maintains `_session_last_served: dict[str, float]` — a mapping of session_id to the monotonic time when a request from that session was last released.

When `_release_next()` scans for the next item to release:
1. Collect all same-priority candidates at the front of the queue (consecutive items at the same priority level).
2. Among candidates, prefer the one from the session with the oldest `session_last_served` (or `enqueued_at` if no prior release).
3. Items without a `session_id` are treated as their own session (no interleaving).

This avoids re-sorting the entire queue; it only affects which item gets released next.

### 4. Priority Preemption in `_recheck_blocked()`

Add the `_has_higher_priority_waiting(idx)` check that already exists in `dequeue()` but is missing from `_recheck_blocked()`. Prevents LOW items from being released ahead of HIGH items during background rechecks.

### 5. Stale Gauge Cleanup

Track all seen model/source label combinations in `_seen_models: set[str]` and `_seen_sources: set[str]`. In `_update_gauges()`, reset all seen labels to 0 before setting current values. Remove labels that have been 0 for more than a configurable threshold (or just reset to 0 on every update).

### 6. Async `cancel_by_session_id()`

Convert from sync to async method. Acquire `self._lock` before mutating `self._queue`. Update the caller in `routers/session_admin.py` to `await` the call.

## Files Changed

| File | Changes |
|------|---------|
| `services/priority_queue.py` | Core: event-based dequeue, symmetric callbacks, session round-robin, recheck preemption, stale gauges, async cancel |
| `services/completion_service.py` | Pass `_QueueItem` ref to `dequeue(item)` |
| `app.py` | No changes (callbacks already correctly structured; symmetry is fixed in the queue) |
| `routers/session_admin.py` | `await priority_queue.cancel_by_session_id()` |
| `test/unit/test_priority_queue_resource_aware.py` | Update existing tests for new API |
| `test/unit/test_priority_queue_out_of_order.py` | NEW: tests for out-of-order completion |
| `test/unit/test_priority_queue_callbacks.py` | NEW: tests for symmetric callback lifecycle |
| `test/unit/test_priority_queue_session_order.py` | NEW: tests for session round-robin |

## Testing Strategy

- **Out-of-order completion:** Enqueue A, B, C. Dequeue B first, then A, then C. Verify correct items are removed and callbacks fire correctly.
- **Sole item:** Enqueue single item, verify `_on_release` fires in `enqueue()`, `_on_complete` fires in `dequeue()`.
- **Last item:** Enqueue A, B. Dequeue A (releases B). Dequeue B. Verify `_on_complete` fires for B.
- **Session round-robin:** Enqueue from sessions A, B, A, B. Verify alternating release order.
- **Priority preemption in recheck:** Enqueue LOW then HIGH. Verify recheck doesn't release LOW ahead of HIGH.
- **Stale gauges:** Enqueue for model-a, dequeue, verify gauge resets to 0.
- **Async cancel:** Verify no race conditions with concurrent enqueue/dequeue.
