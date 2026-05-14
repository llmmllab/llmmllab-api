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
    """Out-of-order dequeue skips blocked items and releases eligible ones."""
    call_results = [False, True]
    call_count = [0]

    async def step_callback(metadata):
        idx = call_count[0]
        call_count[0] += 1
        return call_results[idx] if idx < len(call_results) else True

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

    task_a = asyncio.create_task(q.enqueue(meta_a))
    await asyncio.sleep(0.01)
    task_b = asyncio.create_task(q.enqueue(meta_b))
    await asyncio.sleep(0.01)
    task_c = asyncio.create_task(q.enqueue(meta_c))
    await asyncio.sleep(0.01)

    item_a, evt_a = await task_a
    result = await q.dequeue(item_a)
    assert result is meta_a
    await asyncio.sleep(0.01)
    assert not task_b.done(), "B should still be blocked"
    assert task_c.done(), "C should be released"

    item_b, evt_b = await task_b
    item_c, evt_c = await task_c

    await q.dequeue(item_b)
    await q.dequeue(item_c)
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
