"""Unit tests for resource-aware priority queue dequeue."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from models.request_priority_metadata import (
    Priority,
    RequestPriorityMetadata,
    RequestSource,
)
from services.priority_queue import AsyncPriorityQueue


async def _callback(results: dict) -> bool:
    """Callback that returns pre-configured results keyed by model_id."""
    model_id = getattr(asyncio.current_task(), "_test_model_id", None)
    if model_id and model_id in results:
        return results[model_id]
    return True


class TestDequeueWithCallback:
    """dequeue() respects can_proceed callback."""

    @pytest.mark.asyncio
    async def test_releases_when_callback_true(self):
        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
        q.set_can_proceed_callback(AsyncMock(return_value=True))

        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        task2 = asyncio.create_task(q.enqueue(meta2))
        await asyncio.sleep(0.01)

        result = await q.dequeue()
        assert result is meta1
        # meta2 should be released
        await asyncio.sleep(0.01)
        assert task2.done()
        await task1
        await task2
        await q.close()

    @pytest.mark.asyncio
    async def test_blocks_when_callback_false(self):
        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
        q.set_can_proceed_callback(AsyncMock(return_value=False))

        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        task2 = asyncio.create_task(q.enqueue(meta2))
        await asyncio.sleep(0.01)

        result = await q.dequeue()
        assert result is meta1
        # meta2 should NOT be released
        await asyncio.sleep(0.05)
        assert not task2.done()
        await task1
        # Cancel task2 to avoid hanging
        task2.cancel()
        try:
            await task2
        except asyncio.CancelledError:
            pass
        await q.close()

    @pytest.mark.asyncio
    async def test_skips_blocked_item_to_release_next(self):
        """If front item can't proceed, dequeue checks the next item."""
        call_results = [False, True]  # first call False, second True
        call_idx = [0]

        async def step_callback(metadata):
            idx = call_idx[0]
            call_idx[0] += 1
            return call_results[idx] if idx < len(call_results) else True

        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
        q.set_can_proceed_callback(AsyncMock(side_effect=step_callback))

        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-b"
        )
        meta3 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-c"
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        task2 = asyncio.create_task(q.enqueue(meta2))
        await asyncio.sleep(0.01)
        task3 = asyncio.create_task(q.enqueue(meta3))
        await asyncio.sleep(0.01)

        result = await q.dequeue()
        assert result is meta1
        # meta2 blocked, meta3 released
        await asyncio.sleep(0.01)
        assert not task2.done()
        assert task3.done()
        await task1
        await task3
        task2.cancel()
        try:
            await task2
        except asyncio.CancelledError:
            pass
        await q.close()

    @pytest.mark.asyncio
    async def test_unconditional_without_callback(self):
        """No callback set: original unconditional release behavior."""
        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)

        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        task2 = asyncio.create_task(q.enqueue(meta2))
        await asyncio.sleep(0.01)

        result = await q.dequeue()
        assert result is meta1
        await asyncio.sleep(0.01)
        assert task2.done()
        await task1
        await task2


class TestRemoveItemWithCallback:
    """_remove_item() respects can_proceed on timeout removal."""

    @pytest.mark.asyncio
    async def test_timeout_removal_checks_callback(self):
        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
        q.set_can_proceed_callback(AsyncMock(return_value=False))

        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER,
            priority=Priority.HIGH,
            model_id="model-a",
            max_queue_wait=0.1,
        )
        meta3 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        task2 = asyncio.create_task(q.enqueue(meta2))
        await asyncio.sleep(0.01)
        task3 = asyncio.create_task(q.enqueue(meta3))
        await asyncio.sleep(0.01)

        # meta2 times out and is removed
        from services.queue_exceptions import QueueTimeoutError

        with pytest.raises(QueueTimeoutError):
            await task2

        # meta3 should NOT be released (callback returns False)
        await asyncio.sleep(0.05)
        assert not task3.done()
        await q.dequeue()  # release meta1
        task3.cancel()
        try:
            await task3
        except asyncio.CancelledError:
            pass
        await task1
        await q.close()


class TestRecheckBlocked:
    """Background recheck task unblocks items when resources free up."""

    @pytest.mark.asyncio
    async def test_recheck_unblocks_item(self):
        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0, age_threshold_sec=999)
        q._recheck_interval = 0.2  # Fast recheck for testing

        # Start as False, flip to True after first call
        should_proceed = [False]

        async def flip_callback(metadata):
            if should_proceed[0]:
                return True
            should_proceed[0] = True
            return False

        q.set_can_proceed_callback(AsyncMock(side_effect=flip_callback))

        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-a"
        )
        task1 = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        task2 = asyncio.create_task(q.enqueue(meta2))
        await asyncio.sleep(0.01)

        # Dequeue meta1; callback returns False for meta2
        result = await q.dequeue()
        assert result is meta1
        assert not task2.done()

        # Recheck should eventually release meta2
        await asyncio.wait_for(task2, timeout=3.0)
        await task1
        await task2
        await q.close()


class TestCallbackLifecycle:
    """set_can_proceed_callback and close manage tasks properly."""

    @pytest.mark.asyncio
    async def test_clearing_callback_stops_recheck(self):
        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
        q.set_can_proceed_callback(AsyncMock(return_value=True))
        assert q._recheck_task is not None
        q.set_can_proceed_callback(None)
        assert q._recheck_task is None

    @pytest.mark.asyncio
    async def test_close_cancels_recheck(self):
        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
        q.set_can_proceed_callback(AsyncMock(return_value=True))
        await q.close()
        assert q._recheck_task is None
