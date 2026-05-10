"""Unit tests for AsyncRedisClient."""

import pytest

from db.redis_client import AsyncRedisClient


class TestAsyncRedisClient:
    """AsyncRedisClient manages async Redis connections."""

    def test_initial_state(self):
        client = AsyncRedisClient()
        assert client.client is None

    @pytest.mark.asyncio
    async def test_connect_disabled(self, monkeypatch):
        """When REDIS_ENABLED=false, connect does nothing."""
        monkeypatch.setenv("REDIS_ENABLED", "false")
        # Need to reimport to pick up env var
        import importlib

        import db.redis_client

        importlib.reload(db.redis_client)
        rc = db.redis_client.AsyncRedisClient()
        await rc.connect()
        assert rc.client is None
        await rc.close()

    @pytest.mark.asyncio
    async def test_connect_unreachable(self, monkeypatch):
        """When Redis is unreachable, client is None (no crash)."""
        monkeypatch.setenv("REDIS_ENABLED", "true")
        monkeypatch.setenv("REDIS_HOST", "nonexistent-host-xyz")
        monkeypatch.setenv("REDIS_PORT", "1")
        monkeypatch.setenv("REDIS_PASSWORD", "")

        import importlib

        import db.redis_client

        importlib.reload(db.redis_client)
        rc = db.redis_client.AsyncRedisClient()
        await rc.connect()
        assert rc.client is None
        await rc.close()

    @pytest.mark.asyncio
    async def test_close_noop(self):
        """Close when not connected doesn't raise."""
        client = AsyncRedisClient()
        await client.close()  # Should not raise
