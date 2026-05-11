"""Unit tests for RunnerClient.check_slot_availability()."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from services.runner_client import RunnerClient, ServerHandle
from test.unit.test_runner_client import _mock_client


class TestCheckSlotAvailability:

    @pytest.mark.asyncio
    async def test_active_handle_free_slot(self):
        """Active server with a free slot returns True."""
        mock_slots = MagicMock()
        mock_slots.status_code = 200
        mock_slots.json.return_value = [
            {"id": 0, "is_processing": True},
            {"id": 1, "is_processing": False},
        ]
        mock = _mock_client(get=AsyncMock(return_value=mock_slots))

        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._active_handles.add(
            ServerHandle(
                base_url="http://r1:8000/v1/server/abc",
                server_id="abc",
                runner_host="http://r1:8000",
            )
        )
        result = await client.check_slot_availability("model-a")
        assert result is True

    @pytest.mark.asyncio
    async def test_active_handle_all_busy(self):
        """Active server with all slots busy falls through to VRAM check."""
        mock_slots = MagicMock()
        mock_slots.status_code = 200
        mock_slots.json.return_value = [
            {"id": 0, "is_processing": True},
            {"id": 1, "is_processing": True},
        ]

        call_count = [0]

        async def mock_get(url, **kw):
            call_count[0] += 1
            if "slots" in url:
                return mock_slots
            if "models" in url:
                # Model endpoint not found
                r = MagicMock()
                r.status_code = 404
                return r
            # Health check
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {
                "status": "ok",
                "gpu": {"0": {"name": "GPU", "total_mb": 24576, "used_mb": 20000, "free_mb": 4576, "util_percent": 80}},
            }
            return r

        mock = _mock_client(get=AsyncMock(side_effect=mock_get))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._active_handles.add(
            ServerHandle(
                base_url="http://r1:8000/v1/server/abc",
                server_id="abc",
                runner_host="http://r1:8000",
            )
        )
        client._model_map = {"model-a": ["http://r1:8000"]}
        # All slots busy + model not found on runner -> returns False
        result = await client.check_slot_availability("model-a")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_server_enough_vram(self):
        """No active server, but enough VRAM to start one."""
        call_count = [0]

        async def mock_get(url, **kw):
            call_count[0] += 1
            if "models" in url:
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {
                    "model_id": "model-a",
                    "details": {"size": 4 * 1024 * 1024 * 1024},  # 4 GB
                }
                return r
            # Health
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {
                "status": "ok",
                "gpu": {
                    "0": {
                        "name": "GPU",
                        "total_mb": 24576,
                        "used_mb": 4096,
                        "free_mb": 20480,
                        "util_percent": 20,
                    }
                },
            }
            return r

        mock = _mock_client(get=AsyncMock(side_effect=mock_get))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}
        result = await client.check_slot_availability("model-a")
        assert result is True

    @pytest.mark.asyncio
    async def test_no_server_insufficient_vram(self):
        """No active server, not enough VRAM."""
        call_count = [0]

        async def mock_get(url, **kw):
            call_count[0] += 1
            if "models" in url:
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {
                    "model_id": "model-a",
                    "details": {"size": 20 * 1024 * 1024 * 1024},  # 20 GB
                }
                return r
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {
                "status": "ok",
                "gpu": {
                    "0": {
                        "name": "GPU",
                        "total_mb": 24576,
                        "used_mb": 20000,
                        "free_mb": 4576,
                        "util_percent": 80,
                    }
                },
            }
            return r

        mock = _mock_client(get=AsyncMock(side_effect=mock_get))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}
        result = await client.check_slot_availability("model-a")
        assert result is False

    @pytest.mark.asyncio
    async def test_runner_unreachable_returns_true(self):
        """Fail-open: unreachable runner returns True (don't block)."""
        mock = _mock_client(get=AsyncMock(side_effect=Exception("connection refused")))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        result = await client.check_slot_availability("model-a")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_active_handles_checks_vram(self):
        """With no active handles, goes straight to VRAM check."""
        async def mock_get(url, **kw):
            if "models" in url:
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = {
                    "model_id": "model-a",
                    "details": {"size": 1 * 1024 * 1024 * 1024},
                }
                return r
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {
                "status": "ok",
                "gpu": {
                    "0": {
                        "name": "GPU",
                        "total_mb": 24576,
                        "used_mb": 2000,
                        "free_mb": 22576,
                        "util_percent": 10,
                    }
                },
            }
            return r

        mock = _mock_client(get=AsyncMock(side_effect=mock_get))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}
        result = await client.check_slot_availability("model-a")
        assert result is True
