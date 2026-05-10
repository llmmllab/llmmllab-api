"""
Unit tests for runner_client recovery features added in PR #49.

Covers:
- validate_server_handle()
- _invalidate_model_map_for_endpoint()
- Active handle tracking (_active_handles)
- Graceful shutdown with handle release
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from services.runner_client import RunnerClient, ServerHandle


def _mock_client(**overrides):
    """Build an AsyncMock that behaves like an httpx.AsyncClient."""
    client = AsyncMock()
    client.is_closed = False
    for key, value in overrides.items():
        setattr(client, key, value)
    return client


HANDLE = ServerHandle(
    base_url="http://runner:8000/v1/server/abc123",
    server_id="abc123",
    runner_host="http://runner:8000",
)


class TestValidateServerHandle:
    """validate_server_handle checks the llama.cpp server's /health."""

    @pytest.mark.asyncio
    async def test_valid_handle_returns_true(self):
        """200 from /health → True."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock = _mock_client(get=AsyncMock(return_value=mock_resp))
        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock

        assert await client.validate_server_handle(HANDLE) is True

    @pytest.mark.asyncio
    async def test_invalid_handle_returns_false(self):
        """Non-200 from /health → False."""
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        mock = _mock_client(get=AsyncMock(return_value=mock_resp))
        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock

        assert await client.validate_server_handle(HANDLE) is False

    @pytest.mark.asyncio
    async def test_connection_error_returns_false(self):
        """Connection error → False (not raised)."""
        mock = _mock_client(
            get=AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        )
        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock

        assert await client.validate_server_handle(HANDLE) is False


class TestInvalidateModelMap:
    """_invalidate_model_map_for_endpoint removes dead runners from the map."""

    def test_removes_endpoint_from_all_models(self):
        """Endpoint present for multiple models → removed from all."""
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._model_map = {
            "model-a": ["http://r1:8000", "http://r2:8001"],
            "model-b": ["http://r2:8001"],
            "model-c": ["http://r1:8000"],
        }
        client._invalidate_model_map_for_endpoint("http://r2:8001")

        assert client._model_map["model-a"] == ["http://r1:8000"]
        assert "model-b" not in client._model_map  # was only on r2
        assert client._model_map["model-c"] == ["http://r1:8000"]

    def test_noop_for_unknown_endpoint(self):
        """Removing an endpoint not in the map is a no-op."""
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._model_map = {"model-a": ["http://r1:8000"]}
        client._invalidate_model_map_for_endpoint("http://unknown:9999")

        assert client._model_map == {"model-a": ["http://r1:8000"]}

    def test_clears_entire_map(self):
        """Removing the last endpoint for all models clears the map."""
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._model_map = {
            "model-a": ["http://r1:8000"],
            "model-b": ["http://r1:8000"],
        }
        client._invalidate_model_map_for_endpoint("http://r1:8000")

        assert client._model_map == {}


class TestActiveHandleTracking:
    """acquire_server tracks handles; release_server removes them."""

    @pytest.mark.asyncio
    async def test_acquire_adds_to_active_handles(self):
        """Successful acquire adds handle to _active_handles."""
        mock_health = MagicMock()
        mock_health.status_code = 200
        mock_health.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12e9},
            "active_servers": 0,
            "models": ["model-a"],
        }
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {
            "server_id": "abc",
            "base_url": "http://r1:8000/v1/server/abc",
            "model": "model-a",
        }
        mock_create.raise_for_status = MagicMock()

        mock = _mock_client(
            get=AsyncMock(return_value=mock_health),
            post=AsyncMock(return_value=mock_create),
        )
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}

        handle = await client.acquire_server("model-a")
        assert handle in client._active_handles

    @pytest.mark.asyncio
    async def test_release_removes_from_active_handles(self):
        """release_server removes handle from _active_handles."""
        mock_release = MagicMock()
        mock_release.status_code = 200
        mock_release.raise_for_status = MagicMock()

        mock = _mock_client(post=AsyncMock(return_value=mock_release))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._active_handles.add(HANDLE)

        await client.release_server(HANDLE)
        assert HANDLE not in client._active_handles


class TestGracefulShutdown:
    """aclose releases active handles before closing the client."""

    @pytest.mark.asyncio
    async def test_aclose_releases_active_handles(self):
        """aclose calls release_server for each active handle."""
        mock_release = MagicMock()
        mock_release.status_code = 200
        mock_release.raise_for_status = MagicMock()

        mock = _mock_client(post=AsyncMock(return_value=mock_release))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._active_handles.add(HANDLE)

        await client.aclose()

        mock.post.assert_called_once()
        assert "/v1/server/abc123/release" in mock.post.call_args[0][0]
        assert len(client._active_handles) == 0

    @pytest.mark.asyncio
    async def test_aclose_handles_release_failure(self):
        """Failed release during shutdown is logged, not raised."""
        mock = _mock_client(post=AsyncMock(side_effect=Exception("runner down")))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._active_handles.add(HANDLE)

        # Should not raise
        await client.aclose()
        assert len(client._active_handles) == 0

    @pytest.mark.asyncio
    async def test_aclose_with_multiple_handles(self):
        """All active handles are released on shutdown."""
        handle1 = ServerHandle(
            base_url="http://r1:8000/v1/server/h1",
            server_id="h1",
            runner_host="http://r1:8000",
        )
        handle2 = ServerHandle(
            base_url="http://r1:8000/v1/server/h2",
            server_id="h2",
            runner_host="http://r1:8000",
        )

        mock_release = MagicMock()
        mock_release.status_code = 200
        mock_release.raise_for_status = MagicMock()

        mock = _mock_client(post=AsyncMock(return_value=mock_release))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._active_handles.add(handle1)
        client._active_handles.add(handle2)

        await client.aclose()

        assert mock.post.call_count == 2
        assert len(client._active_handles) == 0
