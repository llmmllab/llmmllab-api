"""Unit tests for priority queue timeout enforcement."""

import asyncio

import pytest

from models.request_priority_metadata import (
    Priority,
    RequestPriorityMetadata,
    RequestSource,
)
from services.priority_queue import AsyncPriorityQueue
from services.queue_exceptions import QueueFullError, QueueTimeoutError


class TestQueueTimeout:
    """enqueue() raises QueueTimeoutError when timeout expires."""

    @pytest.mark.asyncio
    async def test_timeout_raises_queue_timeout_error(self):
        q = AsyncPriorityQueue(timeout_sec=300)
        # Enqueue two items — first blocks (long timeout), second times out
        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH,
            max_queue_wait=0.1,
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)  # Let task1 start and acquire lock
        with pytest.raises(QueueTimeoutError) as exc_info:
            await q.enqueue(meta2)
        assert exc_info.value.max_wait_sec == 0.1
        assert exc_info.value.actual_wait_sec >= 0.1
        await q.dequeue()  # Release first item
        await task1

    @pytest.mark.asyncio
    async def test_per_item_timeout_via_metadata(self):
        """metadata.max_queue_wait overrides the global timeout."""
        q = AsyncPriorityQueue(timeout_sec=300)
        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER,
            priority=Priority.HIGH,
            max_queue_wait=0.1,
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        with pytest.raises(QueueTimeoutError) as exc_info:
            await q.enqueue(meta2)
        assert exc_info.value.max_wait_sec == 0.1
        await q.dequeue()
        await task1

    @pytest.mark.asyncio
    async def test_timeout_param_override(self):
        """timeout_sec parameter overrides metadata and global defaults."""
        q = AsyncPriorityQueue(timeout_sec=300)
        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        with pytest.raises(QueueTimeoutError) as exc_info:
            await q.enqueue(meta2, timeout_sec=0.1)
        assert exc_info.value.max_wait_sec == 0.1
        await q.dequeue()
        await task1

    @pytest.mark.asyncio
    async def test_timeout_param_overrides_metadata(self):
        """timeout_sec parameter takes precedence over metadata.max_queue_wait."""
        q = AsyncPriorityQueue(timeout_sec=300)
        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER,
            priority=Priority.HIGH,
            max_queue_wait=10.0,
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        with pytest.raises(QueueTimeoutError) as exc_info:
            await q.enqueue(meta2, timeout_sec=0.1)
        assert exc_info.value.max_wait_sec == 0.1
        await q.dequeue()
        await task1


class TestQueueFull:
    """enqueue() raises QueueFullError when queue is at capacity."""

    @pytest.mark.asyncio
    async def test_full_raises_queue_full_error(self):
        q = AsyncPriorityQueue(max_size=2, timeout_sec=300)
        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        meta3 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        # Enqueue two items (both wait since no dequeue)
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        task2 = asyncio.create_task(q.enqueue(meta2))
        await asyncio.sleep(0.01)
        # Third should fail — queue is full
        with pytest.raises(QueueFullError):
            await q.enqueue(meta3)
        # Cleanup
        await q.dequeue()
        await task1
        await q.dequeue()
        await task2


class TestNormalFlow:
    """Normal enqueue/dequeue flow still works with timeout enforcement."""

    @pytest.mark.asyncio
    async def test_dequeue_releases_first_item(self):
        """dequeue() pops the first item and sets its event."""
        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        event_task = asyncio.create_task(q.enqueue(meta))
        await asyncio.sleep(0.01)  # Let the task start
        result = await q.dequeue()
        _, event = await event_task
        assert result is not None
        assert result.priority == Priority.HIGH
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """dequeue returns highest priority item first."""
        q = AsyncPriorityQueue(max_size=10, timeout_sec=300)
        low_meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.LOW
        )
        high_meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        task_low = asyncio.create_task(q.enqueue(low_meta))
        await asyncio.sleep(0.01)
        task_high = asyncio.create_task(q.enqueue(high_meta))
        await asyncio.sleep(0.01)
        # Dequeue should return HIGH first (sorts ahead of LOW)
        result = await q.dequeue()
        assert result.priority == Priority.HIGH
        await task_high
        result = await q.dequeue()
        assert result.priority == Priority.LOW
        await task_low

    @pytest.mark.asyncio
    async def test_timed_out_item_removed_from_queue(self):
        """Timed-out items are removed and don't block future items."""
        q = AsyncPriorityQueue(max_size=10, timeout_sec=300)
        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER,
            priority=Priority.HIGH,
            max_queue_wait=0.1,
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        with pytest.raises(QueueTimeoutError):
            await q.enqueue(meta2)
        assert q.size == 1  # Only meta1 remains
        # After cleanup, dequeue works for remaining items
        result = await q.dequeue()
        assert result is not None
        await task1


class TestConfigWiring:
    """Singleton uses config values instead of hardcoded defaults."""

    def test_singleton_uses_config(self):
        from config import (
            PRIORITY_QUEUE_MAX_SIZE,
            PRIORITY_QUEUE_TIMEOUT_SEC,
            PRIORITY_QUEUE_AGE_THRESHOLD_SEC,
        )
        from services.priority_queue import priority_queue

        assert priority_queue._max_size == PRIORITY_QUEUE_MAX_SIZE
        assert priority_queue._timeout_sec == PRIORITY_QUEUE_TIMEOUT_SEC
        assert priority_queue._age_threshold_sec == PRIORITY_QUEUE_AGE_THRESHOLD_SEC
