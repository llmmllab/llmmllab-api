"""Unit tests for RedisPriorityQueue with mocked Redis client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.request_priority_metadata import Priority, RequestPriorityMetadata, RequestSource
from services.redis_priority_queue import RedisPriorityQueue


def _make_mock_redis():
    """Create a mock Redis client with sorted set methods."""
    mock = AsyncMock()
    mock.zcard = AsyncMock(return_value=0)
    mock.zadd = AsyncMock(return_value=1)
    mock.hset = AsyncMock(return_value=1)
    mock.expire = AsyncMock(return_value=True)
    pipe = AsyncMock()
    pipe.zadd = AsyncMock(return_value=None)
    pipe.hset = AsyncMock(return_value=None)
    pipe.expire = AsyncMock(return_value=None)
    pipe.execute = AsyncMock(return_value=[1, 1, True])
    mock.pipeline = MagicMock(return_value=pipe)
    mock.zpopmin = AsyncMock(return_value=None)
    mock.hgetall = AsyncMock(return_value={})
    mock.delete = AsyncMock(return_value=1)
    mock.zrange = AsyncMock(return_value=[])
    mock.zrem = AsyncMock(return_value=1)
    return mock


class TestFallbackToInMemory:
    """When Redis is None, delegates to AsyncPriorityQueue."""

    @pytest.mark.asyncio
    async def test_enqueue_dequeue_fallback(self):
        q = RedisPriorityQueue(redis_client=None, max_size=10, timeout_sec=1.0)
        assert q.is_redis_available is False
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        event_task = asyncio.create_task(q.enqueue(meta))
        await asyncio.sleep(0.01)
        result = await q.dequeue()
        _, event = await event_task
        assert result is not None
        assert result.priority == Priority.HIGH
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_fallback_size(self):
        q = RedisPriorityQueue(redis_client=None, max_size=10, timeout_sec=300)
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        task = asyncio.create_task(q.enqueue(meta))
        await asyncio.sleep(0.01)
        assert q.size == 1
        await q.dequeue()
        await task


class TestRedisEnqueueDequeue:
    """Enqueue/dequeue with mocked Redis client."""

    @pytest.mark.asyncio
    async def test_enqueue_calls_redis(self):
        mock_redis = _make_mock_redis()
        q = RedisPriorityQueue(redis_client=mock_redis, max_size=10, timeout_sec=1.0)
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        # Enqueue will wait for turn (timeout since no dequeue), but we can
        # verify the Redis calls were made
        with pytest.raises(Exception):
            await asyncio.wait_for(q.enqueue(meta), timeout=0.1)
        mock_redis.zcard.assert_called_once()
        mock_redis.pipeline.assert_called()

    @pytest.mark.asyncio
    async def test_full_queue_redis(self):
        mock_redis = _make_mock_redis()
        mock_redis.zcard = AsyncMock(return_value=100)
        q = RedisPriorityQueue(redis_client=mock_redis, max_size=100, timeout_sec=300)
        meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        from services.queue_exceptions import QueueFullError

        with pytest.raises(QueueFullError):
            await q.enqueue(meta)


class TestSetRedisClient:
    """set_redis_client updates the client at runtime."""

    def test_set_client(self):
        q = RedisPriorityQueue(redis_client=None)
        assert q.is_redis_available is False
        mock_redis = _make_mock_redis()
        q.set_redis_client(mock_redis)
        assert q.is_redis_available is True

    def test_clear_client(self):
        q = RedisPriorityQueue(redis_client=_make_mock_redis())
        assert q.is_redis_available is True
        q.set_redis_client(None)
        assert q.is_redis_available is False


class TestReplayPendingItems:
    """Startup replay cleans up expired items."""

    @pytest.mark.asyncio
    async def test_replay_no_redis(self):
        q = RedisPriorityQueue(redis_client=None)
        await q.replay_pending_items()  # Should not raise

    @pytest.mark.asyncio
    async def test_replay_expired_items(self):
        mock_redis = _make_mock_redis()
        mock_redis.zrange = AsyncMock(
            return_value=[("req-1", 1.0), ("req-2", 2.0)]
        )
        mock_redis.hgetall = AsyncMock(
            side_effect=[
                {"enqueued_at": "0.0", "max_queue_wait": "1.0"},
                {"enqueued_at": "0.0", "max_queue_wait": "2.0"},
            ]
        )
        q = RedisPriorityQueue(redis_client=mock_redis, timeout_sec=300)
        await q.replay_pending_items()
        mock_redis.zrem.assert_called_once()
        mock_redis.delete.assert_called()
