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
    on_release = MagicMock()
    on_complete = MagicMock()
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_session_callbacks(on_release, on_complete)

    meta = RequestPriorityMetadata(
        source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
    )
    item, evt = await q.enqueue(meta)

    assert on_release.call_count == 1
    assert on_release.call_args[0][0] is meta
    assert on_complete.call_count == 0

    await q.dequeue(item)

    assert on_complete.call_count == 1
    assert on_complete.call_args[0][0] is meta


@pytest.mark.asyncio
async def test_multi_item_callbacks_symmetric():
    on_release = MagicMock()
    on_complete = MagicMock()
    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_session_callbacks(on_release, on_complete)

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

    assert on_release.call_count == 1
    assert on_release.call_args[0][0] is meta_a

    item_a, _ = await task_a
    await q.dequeue(item_a)
    assert on_release.call_count == 2
    assert on_release.call_args_list[1][0][0] is meta_b
    assert on_complete.call_count == 1
    assert on_complete.call_args[0][0] is meta_a

    item_b, _ = await task_b
    await q.dequeue(item_b)
    assert on_release.call_count == 2
    assert on_complete.call_count == 2
    assert on_complete.call_args_list[1][0][0] is meta_b

    await q.close()


@pytest.mark.asyncio
async def test_active_counts_balance():
    active_counts = {"model-a": 0}

    def on_release(metadata):
        if metadata.model_id:
            active_counts[metadata.model_id] += 1

    def on_complete(metadata):
        if metadata.model_id:
            active_counts[metadata.model_id] -= 1

    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_session_callbacks(on_release, on_complete)

    tasks = []
    for _ in range(5):
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        tasks.append(asyncio.create_task(q.enqueue(meta)))
        await asyncio.sleep(0.01)

    for i in range(5):
        item, _ = await tasks[i]
        await q.dequeue(item)

    assert active_counts["model-a"] == 0
    await q.close()


@pytest.mark.asyncio
async def test_session_gauge_never_negative():
    session_counts = {}

    def on_release(metadata):
        if metadata.session_id:
            session_counts[metadata.session_id] = session_counts.get(metadata.session_id, 0) + 1

    def on_complete(metadata):
        if metadata.session_id:
            session_counts[metadata.session_id] = session_counts.get(metadata.session_id, 0) - 1

    q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
    q.set_session_callbacks(on_release, on_complete)

    tasks = []
    sessions = []
    for session in ["s1", "s2", "s3"]:
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH,
            model_id="model-a", session_id=session,
        )
        tasks.append(asyncio.create_task(q.enqueue(meta)))
        sessions.append(session)
        await asyncio.sleep(0.01)

    for i in range(len(tasks)):
        item, _ = await tasks[i]
        await q.dequeue(item)
        session = sessions[i]
        assert session_counts.get(session, 0) >= 0

    assert all(v == 0 for v in session_counts.values())
    await q.close()
