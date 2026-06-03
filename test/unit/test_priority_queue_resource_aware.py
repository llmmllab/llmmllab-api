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
        """Blocked model stays blocked; an eligible different model proceeds.

        Model-keyed callback (model-b saturated, model-c free) instead of a
        positional one, so the assertion is robust to admission happening at
        enqueue time (idle model released immediately) vs at dequeue.
        """

        async def model_keyed_callback(metadata):
            return metadata.model_id != "model-b"

        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
        q.set_can_proceed_callback(AsyncMock(side_effect=model_keyed_callback))

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

        # meta3 (model-c idle) is admitted immediately on enqueue — it no
        # longer has to wait for meta1's turn to finish.
        assert task3.done()
        assert not task2.done()

        result = await q.dequeue()
        assert result is meta1
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


class TestEnqueueImmediateAdmission:
    """enqueue() admits an eligible idle-model item without waiting for a
    prior busy turn to finish (the 4B-starves-behind-27B fix)."""

    @pytest.mark.asyncio
    async def test_idle_model_released_while_other_model_busy(self):
        """A long-running 27B turn must NOT block a 4B request whose own
        server is idle.

        Reproduces the production symptom: model-27b is admitted and its
        turn is still in flight (never dequeued).  A model-4b request then
        enqueues.  Before the fix it sat unadmitted until the 27B turn's
        dequeue() fired (minutes) or the 2s recheck poll.  After the fix it
        is released the instant it's enqueued, because its model's server is
        idle and _can_proceed says go — with NO dequeue of the busy item.
        """

        async def per_model_capacity(metadata):
            # Both models independently have free capacity.
            return True

        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0, age_threshold_sec=999)
        # Make the recheck poll effectively never fire, so this test proves
        # enqueue itself does the release (not the background poll).
        q._recheck_interval = 1000.0
        q.set_can_proceed_callback(AsyncMock(side_effect=per_model_capacity))

        big = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-27b"
        )
        small = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-4b"
        )

        # 27B admitted (sole item, self-releases) and "runs" — we never
        # dequeue it, simulating a multi-minute turn.
        task_big = asyncio.create_task(q.enqueue(big))
        await asyncio.sleep(0.01)
        item_big, _ = await task_big  # admitted

        # 4B enqueues behind the busy 27B.
        task_small = asyncio.create_task(q.enqueue(small))
        # A tiny tick — NOT a recheck interval, NOT a dequeue.
        await asyncio.sleep(0.02)

        assert task_small.done(), (
            "4B request for an idle model must be admitted immediately on "
            "enqueue, without waiting for the busy 27B turn to finish"
        )
        item_small, _ = await task_small

        await q.dequeue(item_small)
        await q.dequeue(item_big)
        await q.close()

    @pytest.mark.asyncio
    async def test_idle_admission_respects_can_proceed_block(self):
        """If the new item's own model is saturated, enqueue must NOT admit
        it — the immediate-admission pass still honors _can_proceed."""

        async def block_4b(metadata):
            return metadata.model_id != "model-4b"

        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0, age_threshold_sec=999)
        q._recheck_interval = 1000.0
        q.set_can_proceed_callback(AsyncMock(side_effect=block_4b))

        big = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-27b"
        )
        small = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, model_id="model-4b"
        )

        task_big = asyncio.create_task(q.enqueue(big))
        await asyncio.sleep(0.01)
        item_big, _ = await task_big

        task_small = asyncio.create_task(q.enqueue(small))
        await asyncio.sleep(0.02)

        assert not task_small.done(), (
            "a saturated model must stay blocked even with the immediate "
            "enqueue admission pass"
        )

        task_small.cancel()
        try:
            await task_small
        except asyncio.CancelledError:
            pass
        await q.dequeue(item_big)
        await q.close()


class TestCallbackLifecycle:
    """set_can_proceed_callback and close manage tasks properly."""

    @pytest.mark.asyncio
    async def test_recheck_interval_env_override(self, monkeypatch):
        """PRIORITY_QUEUE_RECHECK_SEC overrides the default 2.0s poll."""
        monkeypatch.setenv("PRIORITY_QUEUE_RECHECK_SEC", "0.5")
        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
        assert q._recheck_interval == 0.5

    @pytest.mark.asyncio
    async def test_recheck_interval_default(self, monkeypatch):
        monkeypatch.delenv("PRIORITY_QUEUE_RECHECK_SEC", raising=False)
        q = AsyncPriorityQueue(max_size=10, timeout_sec=5.0)
        assert q._recheck_interval == 2.0

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
