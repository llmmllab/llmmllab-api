"""Async Redis client for priority queue and other async operations."""

from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as aioredis

from config import (
    REDIS_CONNECT_TIMEOUT,
    REDIS_DB,
    REDIS_ENABLED,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_POOL_SIZE,
)

logger = logging.getLogger(__name__)


class AsyncRedisClient:
    """Manages an async Redis connection for durable queue operations."""

    def __init__(self) -> None:
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Connect to Redis if enabled. Silently skips on failure."""
        if not REDIS_ENABLED:
            logger.info("Redis disabled by configuration")
            return

        try:
            self._client = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD or None,
                db=REDIS_DB,
                max_connections=REDIS_POOL_SIZE,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                decode_responses=True,
            )
            await self._client.ping()
            logger.info(
                f"Async Redis connected to {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
            )
        except Exception as e:
            logger.warning(f"Async Redis connection failed: {e}")
            self._client = None

    async def close(self) -> None:
        """Close the async Redis connection."""
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing async Redis: {e}")
            finally:
                self._client = None

    @property
    def client(self) -> Optional[aioredis.Redis]:
        return self._client


async_redis = AsyncRedisClient()
