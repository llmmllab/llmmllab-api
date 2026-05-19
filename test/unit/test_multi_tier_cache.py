"""Unit tests for the multi-tier user-config cache.

Covers both REDIS_ENABLED branches (redis_cache provided vs. None) and
verifies the memory → Redis → database fallback chain.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from db.multi_tier_cache import (
    InMemoryUserConfigCache,
    MultiTierUserConfigCache,
)
from models.user_config import UserConfig


def _make_config(user_id: str = "user-123") -> UserConfig:
    """Build a minimal UserConfig for cache assertions."""
    return UserConfig(user_id=user_id)


def _make_redis_mock() -> MagicMock:
    """Create a synchronous mock matching the redis_cache surface used by the multi-tier cache."""
    redis = MagicMock(name="redis_cache")
    redis.get_user_config_from_cache = MagicMock(return_value=None)
    redis.cache_user_config = MagicMock(return_value=None)
    redis.invalidate_user_config_cache = MagicMock(return_value=None)
    redis.is_storage_cache_enabled = MagicMock(return_value=True)
    return redis


def _make_db_mock() -> MagicMock:
    """Create an async mock matching the database_storage surface used by the multi-tier cache."""
    db = MagicMock(name="database_storage")
    db.get_user_config = AsyncMock(return_value=None)
    db.update_user_config = AsyncMock(return_value=None)
    return db


@pytest.fixture
def fresh_memory_cache(monkeypatch):
    """Replace the InMemoryUserConfigCache with one whose background cleanup thread is suppressed.

    The real implementation spawns a daemon cleanup thread on __init__ which
    is unnecessary noise during unit tests.
    """

    def _no_op_cleanup(self):  # pragma: no cover - never actually invoked
        return

    monkeypatch.setattr(
        InMemoryUserConfigCache, "_periodic_cleanup", _no_op_cleanup
    )


class TestMultiTierCacheGetBothModes:
    """Reads honour REDIS_ENABLED by toggling whether a redis_cache instance is wired in."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("redis_enabled", [True, False])
    async def test_miss_all_tiers_returns_none(
        self, fresh_memory_cache, redis_enabled
    ):
        """Cache miss across every available tier returns None and does not raise."""
        redis = _make_redis_mock() if redis_enabled else None
        db = _make_db_mock()
        cache = MultiTierUserConfigCache(redis_cache=redis, database_storage=db)

        result = await cache.get_user_config("absent-user")

        assert result is None
        db.get_user_config.assert_awaited_once_with("absent-user")
        if redis_enabled:
            assert redis is not None
            redis.get_user_config_from_cache.assert_called_once_with("absent-user")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("redis_enabled", [True, False])
    async def test_memory_hit_skips_lower_tiers(
        self, fresh_memory_cache, redis_enabled
    ):
        """A memory hit must not consult Redis or the database."""
        redis = _make_redis_mock() if redis_enabled else None
        db = _make_db_mock()
        cache = MultiTierUserConfigCache(redis_cache=redis, database_storage=db)

        cfg = _make_config()
        cache.memory_cache.set(cfg.user_id, cfg)

        result = await cache.get_user_config(cfg.user_id)

        assert result is cfg
        db.get_user_config.assert_not_called()
        if redis_enabled:
            assert redis is not None
            redis.get_user_config_from_cache.assert_not_called()


class TestMultiTierCacheRedisEnabled:
    """Behaviours that only apply when a redis_cache instance is supplied."""

    @pytest.mark.asyncio
    async def test_redis_hit_populates_memory_and_skips_db(self, fresh_memory_cache):
        """A Redis hit must hydrate the memory tier and avoid the database."""
        cfg = _make_config()
        redis = _make_redis_mock()
        redis.get_user_config_from_cache.return_value = cfg
        db = _make_db_mock()
        cache = MultiTierUserConfigCache(redis_cache=redis, database_storage=db)

        result = await cache.get_user_config(cfg.user_id)

        assert result is cfg
        db.get_user_config.assert_not_called()
        # Memory tier should now be warmed.
        assert cache.memory_cache.get(cfg.user_id) is cfg

    @pytest.mark.asyncio
    async def test_db_hit_populates_memory_and_redis(self, fresh_memory_cache):
        """A DB hit must warm both upper tiers."""
        cfg = _make_config()
        redis = _make_redis_mock()
        redis.get_user_config_from_cache.return_value = None
        db = _make_db_mock()
        db.get_user_config.return_value = cfg
        cache = MultiTierUserConfigCache(redis_cache=redis, database_storage=db)

        result = await cache.get_user_config(cfg.user_id)

        assert result is cfg
        redis.cache_user_config.assert_called_once_with(cfg.user_id, cfg)
        assert cache.memory_cache.get(cfg.user_id) is cfg

    @pytest.mark.asyncio
    async def test_set_propagates_through_all_tiers(self, fresh_memory_cache):
        """set_user_config writes through DB, memory, and Redis when all tiers are enabled."""
        cfg = _make_config()
        redis = _make_redis_mock()
        db = _make_db_mock()
        cache = MultiTierUserConfigCache(redis_cache=redis, database_storage=db)

        await cache.set_user_config(cfg.user_id, cfg)

        db.update_user_config.assert_awaited_once_with(cfg.user_id, cfg)
        redis.cache_user_config.assert_called_once_with(cfg.user_id, cfg)
        assert cache.memory_cache.get(cfg.user_id) is cfg


class TestMultiTierCacheRedisDisabled:
    """Behaviours when REDIS_ENABLED=false (modeled as redis_cache=None)."""

    @pytest.mark.asyncio
    async def test_get_skips_redis_uses_memory_then_db(self, fresh_memory_cache):
        """Reads check memory then go straight to the database when Redis is disabled."""
        cfg = _make_config()
        db = _make_db_mock()
        db.get_user_config.return_value = cfg
        cache = MultiTierUserConfigCache(redis_cache=None, database_storage=db)

        result = await cache.get_user_config(cfg.user_id)

        assert result is cfg
        db.get_user_config.assert_awaited_once_with(cfg.user_id)
        # Memory was hydrated as a side-effect; second read should not hit DB again.
        db.get_user_config.reset_mock()
        result2 = await cache.get_user_config(cfg.user_id)
        assert result2 is cfg
        db.get_user_config.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_writes_memory_and_db_but_skips_redis(self, fresh_memory_cache):
        """Writes still propagate through memory + DB but must skip Redis entirely."""
        cfg = _make_config()
        db = _make_db_mock()
        cache = MultiTierUserConfigCache(redis_cache=None, database_storage=db)

        await cache.set_user_config(cfg.user_id, cfg)

        db.update_user_config.assert_awaited_once_with(cfg.user_id, cfg)
        assert cache.memory_cache.get(cfg.user_id) is cfg


class TestMultiTierCacheRedisUnreachable:
    """Redis enabled but the client raises on access — reads should still fall through to DB."""

    @pytest.mark.asyncio
    async def test_redis_connection_error_falls_through_to_db(
        self, fresh_memory_cache
    ):
        """If the Redis client raises ConnectionError, the database tier still answers the read."""
        cfg = _make_config()
        redis = _make_redis_mock()
        redis.get_user_config_from_cache.side_effect = ConnectionError(
            "redis down"
        )
        db = _make_db_mock()
        db.get_user_config.return_value = cfg
        cache = MultiTierUserConfigCache(redis_cache=redis, database_storage=db)

        # The fallback chain must be resilient to Redis outages — any
        # exception from the Redis client is logged and treated as a miss.
        result = await cache.get_user_config(cfg.user_id)

        assert result is cfg
        db.get_user_config.assert_awaited_once_with(cfg.user_id)
