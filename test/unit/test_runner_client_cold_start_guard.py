"""Unit tests for the cold-start duplicate-server guard.

Covers the duplicate-27B-server / 507-no-VRAM failure mode: a transient
``/v1/servers`` listing miss (network error, non-200) or a server that is
still ``starting`` must NOT be mistaken for "no server loaded" and so must
NOT green-light a duplicate cold-start.

The fix lives in ``RunnerClient._find_loaded_server_status``, which returns
``(server_or_None, confirmed)`` where ``confirmed`` is True only when the
listing succeeded and the absence/presence is trustworthy.  ``_select_runner``
gates ``can_cold_start`` on ``confirmed``.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from services.runner_client import RunnerClient
from test.unit.test_runner_client import _mock_client


class TestFindLoadedServerStatus:
    @pytest.mark.asyncio
    async def test_confirmed_absent(self):
        """200 listing with no matching model → (None, confirmed=True)."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"servers": [{"model_id": "other", "healthy": True}]}
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = _mock_client(get=AsyncMock(return_value=resp))

        srv, confirmed = await client._find_loaded_server_status(
            "http://r1:8000", "model-a"
        )
        assert srv is None
        assert confirmed is True  # safe to cold-start

    @pytest.mark.asyncio
    async def test_confirmed_present_healthy(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "servers": [{"model_id": "model-a", "healthy": True, "use_count": 1}]
        }
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = _mock_client(get=AsyncMock(return_value=resp))

        srv, confirmed = await client._find_loaded_server_status(
            "http://r1:8000", "model-a"
        )
        assert srv is not None
        assert confirmed is True

    @pytest.mark.asyncio
    async def test_starting_server_is_present_not_absent(self):
        """A still-starting server must read as present (don't duplicate)."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "servers": [{"model_id": "model-a", "healthy": False, "starting": True}]
        }
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = _mock_client(get=AsyncMock(return_value=resp))

        srv, confirmed = await client._find_loaded_server_status(
            "http://r1:8000", "model-a"
        )
        assert srv is not None  # present — a cold-start here would duplicate
        assert confirmed is True

    @pytest.mark.asyncio
    async def test_transient_error_is_unknown_not_absent(self):
        """Network error → (None, confirmed=False): unknown, NOT absent."""
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = _mock_client(
            get=AsyncMock(side_effect=Exception("connection refused"))
        )

        srv, confirmed = await client._find_loaded_server_status(
            "http://r1:8000", "model-a"
        )
        assert srv is None
        assert confirmed is False  # MUST NOT cold-start on this

    @pytest.mark.asyncio
    async def test_non_200_is_unknown_not_absent(self):
        resp = MagicMock()
        resp.status_code = 503
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = _mock_client(get=AsyncMock(return_value=resp))

        srv, confirmed = await client._find_loaded_server_status(
            "http://r1:8000", "model-a"
        )
        assert srv is None
        assert confirmed is False

    @pytest.mark.asyncio
    async def test_find_loaded_server_wrapper_preserves_contract(self):
        """The thin wrapper still returns just Optional[dict]."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "servers": [{"model_id": "model-a", "healthy": True}]
        }
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = _mock_client(get=AsyncMock(return_value=resp))

        srv = await client._find_loaded_server("http://r1:8000", "model-a")
        assert isinstance(srv, dict)
        assert srv.get("model_id") == "model-a"


class TestColdStartGatedOnConfirmation:
    """_select_runner must not cold-start on an unknown /v1/servers state."""

    def _client_with_health(self, servers_responder):
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._model_map = {"model-a": ["http://r1:8000"]}
        client._model_tensor_split = {("http://r1:8000", "model-a"): None}
        client._model_size_bytes = {
            ("http://r1:8000", "model-a"): 4 * 1024 * 1024 * 1024
        }
        client._model_parallel = {("http://r1:8000", "model-a"): 4}

        async def mock_get(url, **kw):
            if "/v1/servers" in url:
                return await servers_responder()
            # /health: plenty of free VRAM (would normally allow a cold start)
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {
                "status": "ok",
                "gpu": {
                    "0": {
                        "name": "GPU",
                        "total_mb": 49152,
                        "used_mb": 1024,
                        "free_mb": 48128,
                        "util_percent": 5,
                    }
                },
                "active_servers": 1,
                "models": ["model-a"],
            }
            return r

        client._client = _mock_client(get=AsyncMock(side_effect=mock_get))
        return client

    @pytest.mark.asyncio
    async def test_transient_servers_miss_does_not_cold_start(self):
        """/v1/servers throws → capacity unknown → no cold-start pick.

        A duplicate cold-start with a tight/unknown VRAM picture is the
        path that 507'd.  With the listing unknown, ``can_cold_start`` must
        be False even though /health reports ample VRAM.
        """

        async def boom():
            raise Exception("transient /v1/servers failure")

        client = self._client_with_health(boom)

        # Inspect the candidate the selector builds: cold-start must be
        # gated off because the server lookup was not confirmed.
        my_server, confirmed = await client._find_loaded_server_status(
            "http://r1:8000", "model-a"
        )
        assert confirmed is False
        # The selector's gate: can_cold_start = confirmed and _can_cold_start(...)
        can_cold_start = confirmed and client._can_cold_start(
            "http://r1:8000", "model-a", 48128 * 1024 * 1024, my_server
        )
        assert can_cold_start is False

    @pytest.mark.asyncio
    async def test_confirmed_absent_allows_cold_start(self):
        """Sanity: when absence is confirmed and VRAM fits, cold-start is OK."""

        async def empty_list():
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {"servers": []}
            return r

        client = self._client_with_health(empty_list)
        my_server, confirmed = await client._find_loaded_server_status(
            "http://r1:8000", "model-a"
        )
        assert confirmed is True
        assert my_server is None
        can_cold_start = confirmed and client._can_cold_start(
            "http://r1:8000", "model-a", 48128 * 1024 * 1024, my_server
        )
        assert can_cold_start is True
