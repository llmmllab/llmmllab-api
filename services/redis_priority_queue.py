"""Redis-backed priority queue using sorted sets for durability."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Optional

import redis.asyncio as aioredis

from models.request_priority_metadata import Priority, RequestPriorityMetadata, RequestSource
from services.priority_queue import AsyncPriorityQueue
from services.queue_exceptions import QueueFullError, QueueTimeoutError
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="redis_priority_queue")

REDIS_QUEUE_KEY = "llmmllab:priority_queue"
REDIS_ITEM_KEY_PREFIX = "llmmllab:priority_queue:item:"

# Multiplier for priority in Redis score to ensure priority sort dominates timestamp
_PRIORITY_MULTIPLIER = 10**15


def _make_score(priority: Priority) -> float:
    """Create a Redis sorted set score: priority first, then timestamp for FIFO."""
    return priority.value * _PRIORITY_MULTIPLIER + time.time()


def _item_key(request_id: str) -> str:
    return f"{REDIS_ITEM_KEY_PREFIX}{request_id}"


def _metadata_to_hash(metadata: RequestPriorityMetadata) -> dict[str, str]:
    """Serialize metadata to a Redis hash."""
    return {
        "priority": metadata.priority.name,
        "source": metadata.source.value,
        "user_id": metadata.user_id or "",
        "session_id": metadata.session_id or "",
        "enqueued_at": str(metadata.enqueued_at),
        "max_queue_wait": str(metadata.max_queue_wait) if metadata.max_queue_wait else "",
        "scheduled_at": str(metadata.scheduled_at) if metadata.scheduled_at else "",
    }


def _hash_to_metadata(h: dict[str, str]) -> RequestPriorityMetadata:
    """Deserialize a Redis hash back to RequestPriorityMetadata."""
    source_str = h.get("source", "user")
    try:
        source = RequestSource(source_str)
    except ValueError:
        source = RequestSource.USER

    priority_str = h.get("priority", "HIGH")
    try:
        priority = getattr(Priority, priority_str)
    except AttributeError:
        priority = Priority.HIGH

    return RequestPriorityMetadata(
        source=source,
        priority=priority,
        user_id=h.get("user_id") or None,
        session_id=h.get("session_id") or None,
        scheduled_at=float(h["scheduled_at"]) if h.get("scheduled_at") else None,
        enqueued_at=float(h.get("enqueued_at", str(time.monotonic()))),
        max_queue_wait=float(h["max_queue_wait"]) if h.get("max_queue_wait") else None,
    )


class RedisPriorityQueue:
    """Priority queue with optional Redis durability.

    When Redis is available, queue state is persisted so that pending
    requests survive server restarts. Falls back to in-memory
    AsyncPriorityQueue when Redis is unavailable.
    """

    def __init__(
        self,
        redis_client: Optional[aioredis.Redis] = None,
        max_size: int = 100,
        timeout_sec: float = 300,
        age_threshold_sec: float = 60,
    ) -> None:
        self._redis = redis_client
        self._max_size = max_size
        self._timeout_sec = timeout_sec
        self._age_threshold_sec = age_threshold_sec
        self._fallback = AsyncPriorityQueue(
            max_size=max_size,
            timeout_sec=timeout_sec,
            age_threshold_sec=age_threshold_sec,
        )
        # In-memory event store for Redis-backed items
        self._events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    @property
    def is_redis_available(self) -> bool:
        return self._redis is not None

    @property
    def size(self) -> int:
        return self._fallback.size

    @property
    def sizes_by_priority(self) -> dict[str, int]:
        return self._fallback.sizes_by_priority

    def set_redis_client(self, client: Optional[aioredis.Redis]) -> None:
        """Update the Redis client at runtime (e.g., during app startup)."""
        self._redis = client

    async def replay_pending_items(self) -> None:
        """On startup, clean up expired items from Redis.

        Original HTTP connections are gone, so we don't recreate events
        for them — they'll timeout when the client retries.
        """
        if not self._redis:
            return

        try:
            items = await self._redis.zrange(REDIS_QUEUE_KEY, 0, -1, withscores=True)
            expired = []
            for member, score in items:
                key = _item_key(member)
                h = await self._redis.hgetall(key)
                if not h:
                    expired.append(member)
                    continue
                max_wait = float(h.get("max_queue_wait", self._timeout_sec)) or self._timeout_sec
                enqueued = float(h.get("enqueued_at", 0))
                if time.time() - enqueued > max_wait:
                    expired.append(member)

            if expired:
                await self._redis.zrem(REDIS_QUEUE_KEY, *expired)
                for member in expired:
                    await self._redis.delete(_item_key(member))
                logger.info(f"Cleaned up {len(expired)} expired queue items on startup")
        except Exception as e:
            logger.warning(f"Failed to replay pending queue items: {e}")

    async def enqueue(
        self,
        metadata: RequestPriorityMetadata,
        timeout_sec: Optional[float] = None,
    ) -> asyncio.Event:
        if self.is_redis_available:
            return await self._enqueue_redis(metadata, timeout_sec)
        return await self._fallback.enqueue(metadata, timeout_sec)

    async def dequeue(self) -> Optional[RequestPriorityMetadata]:
        if self.is_redis_available:
            return await self._dequeue_redis()
        return await self._fallback.dequeue()

    # ── Redis-backed implementation ──

    async def _enqueue_redis(
        self,
        metadata: RequestPriorityMetadata,
        timeout_sec: Optional[float] = None,
    ) -> asyncio.Event:
        effective_timeout = timeout_sec or metadata.max_queue_wait or self._timeout_sec
        request_id = str(uuid.uuid4())

        async with self._lock:
            current_size = await self._redis.zcard(REDIS_QUEUE_KEY)
            if current_size >= self._max_size:
                raise QueueFullError(
                    f"Priority queue full ({self._max_size} items)."
                )

            score = _make_score(metadata.priority)
            pipe = self._redis.pipeline()
            pipe.zadd(REDIS_QUEUE_KEY, {request_id: score})
            pipe.hset(_item_key(request_id), mapping=_metadata_to_hash(metadata))
            pipe.expire(_item_key(request_id), int(effective_timeout) + 60)
            results = await pipe.execute()

        if results[0] == 0:
            await self._redis.zrem(REDIS_QUEUE_KEY, request_id)
            raise QueueFullError(
                f"Priority queue full ({self._max_size} items)."
            )

        event = asyncio.Event()
        self._events[request_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Request timed out in Redis priority queue",
                extra={
                    "priority": metadata.priority.name,
                    "source": metadata.source.value,
                    "wait_time": metadata.wait_time,
                    "max_wait_sec": effective_timeout,
                },
            )
            async with self._lock:
                await self._redis.zrem(REDIS_QUEUE_KEY, request_id)
                await self._redis.delete(_item_key(request_id))
            self._events.pop(request_id, None)
            raise QueueTimeoutError(
                max_wait_sec=effective_timeout,
                actual_wait_sec=metadata.wait_time,
            ) from None

        self._events.pop(request_id, None)
        return event

    async def _dequeue_redis(self) -> Optional[RequestPriorityMetadata]:
        async with self._lock:
            result = await self._redis.zpopmin(REDIS_QUEUE_KEY, count=1)
            if not result:
                return None

            request_id, _score = result[0]
            h = await self._redis.hgetall(_item_key(request_id))
            await self._redis.delete(_item_key(request_id))

            metadata = _hash_to_metadata(h) if h else None

            event = self._events.pop(request_id, None)
            if event:
                event.set()

            return metadata

    # ── Background aging task (same as in-memory, runs when Redis is available) ──

    async def _start_aging(self) -> None:
        """Background task that promotes aging items in Redis queue."""
        import asyncio as _asyncio

        while True:
            await _asyncio.sleep(self._age_threshold_sec)
            try:
                await self._age_redis_items()
            except Exception as e:
                logger.warning(f"Aging task failed: {e}")

    async def _age_redis_items(self) -> None:
        """Promote LOW→MEDIUM and MEDIUM→HIGH for items waiting too long."""
        if not self._redis:
            return

        items = await self._redis.zrange(REDIS_QUEUE_KEY, 0, -1, withscores=True)
        promoted = []

        for member, score in items:
            key = _item_key(member)
            h = await self._redis.hgetall(key)
            if not h:
                continue

            priority_name = h.get("priority", "HIGH")
            if priority_name == "HIGH":
                continue

            old_priority = getattr(Priority, priority_name, Priority.HIGH)
            if old_priority == Priority.LOW:
                new_priority = Priority.MEDIUM
            elif old_priority == Priority.MEDIUM:
                new_priority = Priority.HIGH
            else:
                continue

            h["priority"] = new_priority.name
            await self._redis.hset(key, mapping=h)
            new_score = _make_score(new_priority)
            await self._redis.zadd(REDIS_QUEUE_KEY, {member: new_score})
            promoted.append((old_priority.name, new_priority.name))

        if promoted:
            logger.info(f"Promoted {len(promoted)} items in Redis queue")
