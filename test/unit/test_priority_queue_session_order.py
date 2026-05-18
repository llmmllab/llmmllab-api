"""Tests for session-aware round-robin ordering in AsyncPriorityQueue."""

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
async def test_session_round_robin_alternates():
    """Requests from different sessions should alternate."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

    released_order = []

    def on_release(metadata):
        released_order.append(metadata.session_id)

    q.set_session_callbacks(on_release, MagicMock())

    items = []
    for sid in ["s1", "s2"]:
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH,
            model_id="model-a", session_id=sid,
        )
        task = asyncio.create_task(q.enqueue(meta))
        await asyncio.sleep(0.01)
        items.append(task)

    # First item (s1) released immediately as sole item
    assert released_order == ["s1"]

    # Dequeue s1 — should release s2 (different session)
    item_a, _ = await items[0]
    released_order.clear()
    await q.dequeue(item_a)
    assert released_order == ["s2"], f"Expected s2, got {released_order}"

    item_b, _ = await items[1]
    await q.dequeue(item_b)
    await q.close()


@pytest.mark.asyncio
async def test_session_round_robin_three_sessions():
    """Three sessions should interleave fairly."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

    released_order = []

    def on_release(metadata):
        released_order.append(metadata.session_id)

    q.set_session_callbacks(on_release, MagicMock())

    tasks = []
    for sid in ["s1", "s2", "s3"]:
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH,
            model_id="model-a", session_id=sid,
        )
        task = asyncio.create_task(q.enqueue(meta))
        await asyncio.sleep(0.01)
        tasks.append(task)

    # s1 released as sole item
    assert released_order[0] == "s1"

    # Dequeue s1 — should release both s2 and s3 (no resource constraint)
    # s2 should be released first (oldest unserved via round-robin)
    item_a, _ = await tasks[0]
    released_order.clear()
    await q.dequeue(item_a)
    assert "s2" in released_order
    assert "s3" in released_order
    assert released_order[0] == "s2"  # s2 enqueued first = oldest unserved

    item_b, _ = await tasks[1]
    item_c, _ = await tasks[2]
    await q.dequeue(item_b)
    await q.dequeue(item_c)
    await q.close()


@pytest.mark.asyncio
async def test_no_session_id_no_interleaving():
    """Items without session_id should follow FIFO order."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

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

    task_a = asyncio.create_task(q.enqueue(meta_a))
    await asyncio.sleep(0.01)
    task_b = asyncio.create_task(q.enqueue(meta_b))
    await asyncio.sleep(0.01)

    # A released as sole item
    assert released_order == [id(meta_a)]

    item_a, _ = await task_a
    released_order.clear()
    await q.dequeue(item_a)
    assert released_order == [id(meta_b)]

    item_b, _ = await task_b
    await q.dequeue(item_b)
    await q.close()


@pytest.mark.asyncio
async def test_higher_priority_skips_session_rr():
    """HIGH priority items released before LOW regardless of session."""
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

    released_order = []

    def on_release(metadata):
        released_order.append((metadata.session_id, metadata.priority.name))

    q.set_session_callbacks(on_release, MagicMock())

    # LOW from s1
    meta_low = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.LOW,
        model_id="model-a", session_id="s1",
    )
    task_low = asyncio.create_task(q.enqueue(meta_low))
    await asyncio.sleep(0.01)

    # HIGH from s2
    meta_high = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH,
        model_id="model-a", session_id="s2",
    )
    task_high = asyncio.create_task(q.enqueue(meta_high))
    await asyncio.sleep(0.01)

    # LOW released as sole item first
    assert released_order[0] == ("s1", "LOW")

    # Dequeue LOW — HIGH should be released (higher priority)
    item_low, _ = await task_low
    released_order.clear()
    await q.dequeue(item_low)
    assert released_order[0] == ("s2", "HIGH")

    item_high, _ = await task_high
    await q.dequeue(item_high)
    await q.close()
