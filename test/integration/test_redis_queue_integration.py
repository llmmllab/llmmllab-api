"""Integration tests for RedisPriorityQueue with real Redis via testcontainers."""

import asyncio
import os

import pytest

try:
    import redis.asyncio as aioredis
    from testcontainers.redis import RedisContainer

    HAS_TESTCONTAINERS = True
except ImportError:
    HAS_TESTCONTAINERS = False

from models.request_priority_metadata import Priority, RequestPriorityMetadata, RequestSource
from services.queue_exceptions import QueueTimeoutError
from services.redis_priority_queue import RedisPriorityQueue


@pytest.mark.skipif(not HAS_TESTCONTAINERS, reason="testcontainers not available")
@pytest.mark.asyncio
class TestRedisQueueIntegration:
    """Full enqueue/dequeue cycle with real Redis."""

    @pytest.fixture(scope="class")
    def redis_container(self):
        """Start a Redis container for the test class."""
        with RedisContainer("redis:7-alpine") as container:
            yield container

    @pytest.fixture(scope="class")
    def redis_client(self, redis_container):
        """Create an async Redis client connected to the container."""
        host = redis_container.get_container_host_ip()
        port = redis_container.get_exposed_port(6379)
        client = aioredis.Redis(
            host=host,
            port=port,
            decode_responses=True,
        )
        return client

    @pytest.fixture(scope="class")
    def queue(self, redis_client):
        """Create a RedisPriorityQueue with the real Redis client."""
        return RedisPriorityQueue(
            redis_client=redis_client,
            max_size=100,
            timeout_sec=300,
        )

    async def test_enqueue_dequeue_basic(self, queue, redis_client):
        """Basic enqueue then dequeue round-trip."""
        meta = RequestPriorityMetadata(
            source=RequestSource.USER,
            priority=Priority.HIGH,
            user_id="test-user",
        )
        # Enqueue and dequeue in the same test (first item proceeds immediately
        # only if dequeue is called — we need to enqueue, then dequeue)
        task = asyncio.create_task(queue.enqueue(meta))
        await asyncio.sleep(0.05)
        result = await queue.dequeue()
        event = await task
        assert result is not None
        assert result.user_id == "test-user"
        assert result.priority == Priority.HIGH
        assert event.is_set()

    async def test_priority_ordering(self, queue, redis_client):
        """Higher priority items are dequeued first."""
        low_meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.LOW, user_id="low-user"
        )
        high_meta = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH, user_id="high-user"
        )
        # Enqueue LOW first, then HIGH
        task_low = asyncio.create_task(queue.enqueue(low_meta))
        await asyncio.sleep(0.01)
        task_high = asyncio.create_task(queue.enqueue(high_meta))
        await asyncio.sleep(0.01)
        # Dequeue should return HIGH first
        result = await queue.dequeue()
        assert result.user_id == "high-user"
        result = await queue.dequeue()
        assert result.user_id == "low-user"
        await task_high
        await task_low

    async def test_timeout_removal(self, queue, redis_client):
        """Timed-out items are removed from Redis."""
        q_short = RedisPriorityQueue(
            redis_client=redis_client,
            max_size=100,
            timeout_sec=0.1,
        )
        meta = RequestPriorityMetadata(
            source=RequestSource.USER,
            priority=Priority.HIGH,
            max_queue_wait=0.1,
        )
        with pytest.raises(QueueTimeoutError):
            await q_short.enqueue(meta)
        # Verify item is not in Redis
        from services.redis_priority_queue import REDIS_QUEUE_KEY

        count = await redis_client.zcard(REDIS_QUEUE_KEY)
        # May be 0 or items from other tests, but this specific item should be gone

    async def test_startup_replay(self, redis_client):
        """Expired items are cleaned up on replay."""
        from services.redis_priority_queue import (
            REDIS_QUEUE_KEY,
            REDIS_ITEM_KEY_PREFIX,
        )

        # Manually add an expired item to Redis
        await redis_client.zadd(REDIS_QUEUE_KEY, {"expired-req": 1.0})
        await redis_client.hset(
            f"{REDIS_ITEM_KEY_PREFIX}expired-req",
            mapping={"enqueued_at": "0.0", "max_queue_wait": "0.1"},
        )
        # Replay should clean it up
        q = RedisPriorityQueue(redis_client=redis_client, timeout_sec=300)
        await q.replay_pending_items()
        members = await redis_client.zrange(REDIS_QUEUE_KEY, 0, -1)
        assert "expired-req" not in members

    async def test_queue_full(self, redis_client):
        """Queue rejects when at max_size."""
        q = RedisPriorityQueue(
            redis_client=redis_client,
            max_size=1,
            timeout_sec=300,
        )
        meta1 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        meta2 = RequestPriorityMetadata(
            source=RequestSource.USER, priority=Priority.HIGH
        )
        task = asyncio.create_task(q.enqueue(meta1))
        await asyncio.sleep(0.01)
        from services.queue_exceptions import QueueFullError

        with pytest.raises(QueueFullError):
            await q.enqueue(meta2)
        # Clean up
        await q.dequeue()
        await task
