"""Unit tests for the single-endpoint duplicate-server guard (Fix 2).

The #285 selection policy prefers a *fresh comfortable dedicated server* over
a warm peer — correct when MULTIPLE endpoints host the model (a fresh server
lands on a different runner and fans the load out).  But for a model hosted on
only ONE endpoint (e.g. Qwen3_6_27B on the lone big runner), a "fresh" server
can't fan out anywhere: it lands beside the existing warm server on the same
box, doubles the VRAM footprint, and 507s on the second load — while the IDE
zombie sessions thrash the original via LRU eviction + re-prefill.

``_select_runner`` now gates the cold-start *preference* on having more than
one candidate endpoint: a single-endpoint model with a warm free-slot server
always REUSES it (rule 2), while the #285 fresh-preferred behaviour is left
fully intact for multi-endpoint models.
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from services.runner_client import RunnerClient
from test.unit.test_runner_client import _mock_client


def _health_resp(free_mb=12000):
    h = MagicMock(status_code=200)
    h.json.return_value = {
        "status": "ok",
        "gpu": {"0": {"free_mb": free_mb}},
        "models": [{"id": "m1", "name": "m1", "task": "TextToText"}],
        "active_servers": 1,
    }
    return h


def _servers_resp(servers):
    r = MagicMock(status_code=200)
    r.json.return_value = {"active_servers": len(servers), "servers": servers}
    return r


class TestSingleEndpointReusesWarmFreeSlot:
    """A single-endpoint model with a warm free-slot server never cold-starts
    a duplicate — it reuses the warm server."""

    def _single_endpoint_client(self):
        c = RunnerClient(endpoints=["http://big:8000"])
        c._model_map = {"m1": ["http://big:8000"]}
        c._model_tensor_split = {("http://big:8000", "m1"): None}
        c._model_parallel = {("http://big:8000", "m1"): 4}
        # Comfortable VRAM headroom: a duplicate WOULD pass the can-fit check,
        # so only the single-endpoint guard prevents the duplicate cold-start.
        c._model_size_bytes = {("http://big:8000", "m1"): 4 * 1024 * 1024 * 1024}
        return c

    @pytest.mark.asyncio
    async def test_warm_free_slot_reused_not_duplicated(self):
        """Warm server with a free slot + ample VRAM to spawn a second one →
        REUSE the warm server (no duplicate cold-start)."""
        client = self._single_endpoint_client()

        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return _health_resp(free_mb=48000)  # plenty for a 2nd server
            if "/v1/servers" in url:
                # Warm server, 1 of 4 slots used → 3 free.
                return _servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True, "idle_since": None, "use_count": 1}
                ])
            return MagicMock(status_code=404)

        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen == "http://big:8000"  # reused, not a fresh duplicate

    @pytest.mark.asyncio
    async def test_idle_warm_server_reused_not_duplicated(self):
        """Even a fully-idle warm server (use_count 0) is reused on a
        single-endpoint model — the #285 'prefer fresh' path that fires for
        multi-endpoint idle servers must NOT fire here (no peer to fan to)."""
        client = self._single_endpoint_client()
        long_idle = time.time() - 9999

        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return _health_resp(free_mb=48000)
            if "/v1/servers" in url:
                return _servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True, "idle_since": long_idle, "use_count": 0}
                ])
            return MagicMock(status_code=404)

        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen == "http://big:8000"

    @pytest.mark.asyncio
    async def test_single_endpoint_no_server_still_cold_starts(self):
        """Sanity: a single-endpoint model with NO server yet still
        cold-starts (first load) — the guard only blocks DUPLICATES, not the
        initial load."""
        client = self._single_endpoint_client()

        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return _health_resp(free_mb=48000)
            if "/v1/servers" in url:
                return _servers_resp([])  # confirmed absent
            return MagicMock(status_code=404)

        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen == "http://big:8000"  # cold-start the one and only server

    @pytest.mark.asyncio
    async def test_single_endpoint_full_server_falls_through(self):
        """Single endpoint whose only server is full (no free slot) → no warm
        free-slot cohort, no cold-start (server present), ranked fallback to
        the same endpoint so the runner's own queue serializes."""
        client = self._single_endpoint_client()
        client._model_parallel = {("http://big:8000", "m1"): 1}

        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return _health_resp(free_mb=48000)
            if "/v1/servers" in url:
                return _servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True, "idle_since": None, "use_count": 1}
                ])
            return MagicMock(status_code=404)

        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen == "http://big:8000"


class TestMultiEndpointStillPrefersFreshPer285:
    """The single-endpoint guard must NOT regress the #285 multi-endpoint
    policy: a comfortable fresh dedicated server is still preferred over a
    warm peer when more than one endpoint can host the model."""

    def _multi_endpoint_client(self):
        c = RunnerClient(endpoints=["http://r1:8000", "http://r2:8000"])
        c._model_map = {"m1": ["http://r1:8000", "http://r2:8000"]}
        c._model_tensor_split = {
            ("http://r1:8000", "m1"): None,
            ("http://r2:8000", "m1"): None,
        }
        c._model_parallel = {
            ("http://r1:8000", "m1"): 2,
            ("http://r2:8000", "m1"): 2,
        }
        return c

    @pytest.mark.asyncio
    async def test_comfortable_cold_start_still_preferred_over_warm(self):
        """r1 warm idle (free slot), r2 empty with COMFORTABLE VRAM → #285
        still cold-starts the fresh dedicated server on r2 (fans out)."""
        client = self._multi_endpoint_client()
        client._model_size_bytes = {
            ("http://r1:8000", "m1"): 8 * 1024 * 1024 * 1024,
            ("http://r2:8000", "m1"): 8 * 1024 * 1024 * 1024,
        }

        def health(free_mb):
            h = MagicMock(status_code=200)
            h.json.return_value = {
                "status": "ok",
                "gpu": {"0": {"free_mb": free_mb}},
                "models": [{"id": "m1", "name": "m1", "task": "TextToText"}],
                "active_servers": 1,
            }
            return h

        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return health(16 * 1024)  # 16 GB free >= 1.25 * 8 = 10 GB
            if "r1" in url and "/v1/servers" in url:
                return _servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True, "idle_since": None, "use_count": 0}
                ])
            if "r2" in url and "/v1/servers" in url:
                return _servers_resp([])
            return MagicMock(status_code=404)

        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen == "http://r2:8000"  # fresh dedicated server, #285 intact

    @pytest.mark.asyncio
    async def test_marginal_cold_start_still_defers_to_warm_peer(self):
        """Multi-endpoint marginal cold-start (fits but not comfortably) still
        defers to the warm free-slot peer (the VRAM-pressure-500 guard)."""
        client = self._multi_endpoint_client()
        client._model_size_bytes = {
            ("http://r1:8000", "m1"): 10 * 1024 * 1024 * 1024,
            ("http://r2:8000", "m1"): 10 * 1024 * 1024 * 1024,
        }

        def health(free_mb):
            h = MagicMock(status_code=200)
            h.json.return_value = {
                "status": "ok",
                "gpu": {"0": {"free_mb": free_mb}},
                "models": [{"id": "m1", "name": "m1", "task": "TextToText"}],
                "active_servers": 1,
            }
            return h

        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return health(12 * 1024)  # 12 GB fits 10 but < 1.25*10 = 12.5
            if "r1" in url and "/v1/servers" in url:
                return _servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True, "idle_since": None, "use_count": 0}
                ])
            if "r2" in url and "/v1/servers" in url:
                return _servers_resp([])
            return MagicMock(status_code=404)

        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen == "http://r1:8000"  # warm free slot, marginal deferral
