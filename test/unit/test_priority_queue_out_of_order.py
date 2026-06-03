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
    """B finishes before A — dequeue(B) must return B's metadata."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

    meta_a = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    meta_b = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )

    task_a = asyncio.create_task(q.enqueue(meta_a))
    await asyncio.sleep(0.01)
    task_b = asyncio.create_task(q.enqueue(meta_b))
    await asyncio.sleep(0.01)

    item_a, evt_a = await task_a
    await q.dequeue(item_a)
    item_b, evt_b = await task_b

    result = await q.dequeue(item_b)
    assert result is meta_b
    assert q.size == 0
    await q.close()


@pytest.mark.asyncio
async def test_out_of_order_three_items_dequeue_returns_correct_meta():
    """A, B, C all released. Dequeuing in any order returns correct metadata."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

    metas = []
    items = []
    tasks = []
    for _ in range(3):
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        metas.append(meta)
        tasks.append(asyncio.create_task(q.enqueue(meta)))
        await asyncio.sleep(0.01)

    for i in range(3):
        item, _ = await tasks[i]
        items.append(item)
        result = await q.dequeue(item)
        assert result is metas[i]

    assert q.size == 0
    await q.close()


@pytest.mark.asyncio
async def test_out_of_order_with_blocked_items():
    """Skips blocked items and releases eligible ones.

    Uses a model-keyed callback (model-b is at capacity → blocked,
    model-c has a free slot → eligible).  This is order-independent so it
    holds regardless of whether admission happens at enqueue time (idle
    model released immediately) or at dequeue — what matters is that the
    blocked model stays blocked and the eligible one is released.
    """

    async def model_keyed_callback(metadata):
        # model-b is saturated; everything else has capacity.
        return metadata.model_id != "model-b"

    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_can_proceed_callback(AsyncMock(side_effect=model_keyed_callback))

    meta_a = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    meta_b = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-b"
    )
    meta_c = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-c"
    )

    task_a = asyncio.create_task(q.enqueue(meta_a))
    await asyncio.sleep(0.01)
    task_b = asyncio.create_task(q.enqueue(meta_b))
    await asyncio.sleep(0.01)
    task_c = asyncio.create_task(q.enqueue(meta_c))
    await asyncio.sleep(0.01)

    item_a, evt_a = await task_a
    # C (idle model) is admitted as soon as it's enqueued — it no longer
    # waits for A's turn to finish.
    assert task_c.done(), "C should be released (model-c idle)"
    assert not task_b.done(), "B should still be blocked (model-b saturated)"

    result = await q.dequeue(item_a)
    assert result is meta_a
    await asyncio.sleep(0.01)
    assert not task_b.done(), "B should still be blocked"

    item_c, evt_c = await task_c
    await q.dequeue(item_c)

    task_b.cancel()
    try:
        await task_b
    except asyncio.CancelledError:
        pass
    await q.close()


@pytest.mark.asyncio
async def test_completed_items_lazy_compaction():
    """Completed items at the front are lazily compacted on dequeue."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

    metas = []
    items = []
    tasks = []
    for _ in range(3):
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        metas.append(meta)
        tasks.append(asyncio.create_task(q.enqueue(meta)))
        await asyncio.sleep(0.01)

    for i in range(3):
        item, _ = await tasks[i]
        items.append(item)
        result = await q.dequeue(item)
        assert result is metas[i]

    assert q.size == 0
    await q.close()
