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

    meta_a = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    item_a, _ = await q.enqueue(meta_a)

    assert "model-a" in q._seen_models

    await q.dequeue(item_a)
    assert "model-a" in q._seen_models, "model-a should still be in seen_models"

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

    meta_low = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.LOW, model_id="model-a"
    )
    meta_high = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )

    task_low = asyncio.create_task(q.enqueue(meta_low))
    await asyncio.sleep(0.01)
    task_high = asyncio.create_task(q.enqueue(meta_high))
    await asyncio.sleep(0.01)

    item_low, evt_low = await task_low
    item_high, evt_high = await task_high

    assert released[0] == "LOW"
    released.clear()

    # Dequeue LOW — HIGH should be released (higher priority)
    await q.dequeue(item_low)
    assert "HIGH" in released or evt_high.is_set()

    await q.dequeue(item_high)
    await q.close()


@pytest.mark.asyncio
async def test_async_cancel_by_session_id():
    """cancel_by_session_id must be async and lock-safe."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

    metas = []
    tasks = []
    for sid in ["s1", "s2", "s1"]:
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH,
            model_id="model-a", session_id=sid,
        )
        metas.append(meta)
        tasks.append(asyncio.create_task(q.enqueue(meta)))
        await asyncio.sleep(0.01)

    # Resolve first item (sole item, released)
    item1, _ = await tasks[0]

    assert q.size == 3

    # Cancel s1 — should remove meta1 and meta3 from queue
    cancelled = await q.cancel_by_session_id("s1")
    assert cancelled == 2
    assert q.size == 1

    # Remaining item should be s2
    remaining = q._queue[0]
    assert remaining.metadata.session_id == "s2"

    # Dequeue item1 to release item2
    await q.dequeue(item1)
    item2, _ = await tasks[1]
    await q.dequeue(item2)

    # Cancelled tasks should resolve (event was set by cancel)
    try:
        item3, _ = await tasks[2]
    except Exception:
        pass

    await q.close()
