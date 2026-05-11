"""
Unit tests for services/runner_client.py.

Tests the RunnerClient HTTP client that routes requests among multiple
llmmllab-runner service instances.  The client now uses a persistent
``httpx.AsyncClient``, so tests mock ``_get_client()`` instead of patching
the ``httpx.AsyncClient`` constructor.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from services.runner_client import RunnerClient, ServerHandle
from models import ModelTask


class TestServerHandle:

    def test_construction(self):
        """Test ServerHandle dataclass fields."""
        handle = ServerHandle(
            base_url="http://runner:8000/v1/server/abc123",
            server_id="abc123",
            runner_host="http://runner:8000",
        )
        assert handle.base_url == "http://runner:8000/v1/server/abc123"
        assert handle.server_id == "abc123"
        assert handle.runner_host == "http://runner:8000"


def _mock_client(**overrides) -> AsyncMock:
    """Build an AsyncMock that behaves like an httpx.AsyncClient."""
    client = AsyncMock()
    client.is_closed = False
    for key, value in overrides.items():
        setattr(client, key, value)
    return client


class TestRunnerClientHealth:

    @pytest.mark.asyncio
    async def test_healthy_returns_data(self):
        """Mock 200 /health and verify _health returns dict."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12000000000},
            "active_servers": 0,
            "models": ["llama-3-8b"],
        }

        mock = _mock_client(get=AsyncMock(return_value=mock_response))

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        result = await client._health("http://runner1:8000")

        assert result is not None
        assert result["status"] == "ok"
        assert result["gpu"]["available_vram_bytes"] == 12000000000

    @pytest.mark.asyncio
    async def test_unhealthy_returns_none(self):
        """Mock httpx.RequestError and verify _health returns None."""
        mock = _mock_client(get=AsyncMock(side_effect=Exception("connection refused")))

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        result = await client._health("http://runner1:8000")

        assert result is None


class TestRunnerClientAcquire:

    @pytest.mark.asyncio
    async def test_acquire_returns_handle(self):
        """Mock health + create, verify acquire_server returns ServerHandle."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_health_response.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12000000000},
            "active_servers": 0,
            "models": ["llama-3-8b"],
        }

        mock_create_response = MagicMock()
        mock_create_response.status_code = 201
        mock_create_response.json.return_value = {
            "server_id": "abc123",
            "base_url": "http://runner1:8000/v1/server/abc123",
            "model": "llama-3-8b",
        }
        mock_create_response.raise_for_status = MagicMock()

        mock = _mock_client(
            get=AsyncMock(return_value=mock_health_response),
            post=AsyncMock(return_value=mock_create_response),
        )

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        handle = await client.acquire_server("llama-3-8b", task=ModelTask.TEXTTOTEXT, config_override={})
        assert handle.server_id == "abc123"
        assert handle.base_url == "http://runner1:8000/v1/server/abc123"
        assert handle.runner_host == "http://runner1:8000"

    @pytest.mark.asyncio
    async def test_acquire_raises_if_none(self):
        """All runners unhealthy raises RuntimeError."""
        mock = _mock_client(
            get=AsyncMock(side_effect=Exception("connection refused")),
            post=AsyncMock(side_effect=Exception("connection refused")),
        )

        client = RunnerClient(
            endpoints=["http://runner1:8000", "http://runner2:8001"]
        )
        client._client = mock
        with pytest.raises(RuntimeError, match="No healthy runner"):
            await client.acquire_server("llama-3-8b", task="TextGeneration", config_override={})

    @pytest.mark.asyncio
    async def test_acquire_retries_on_507(self):
        """First runner returns 507, client tries next runner."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_health_response.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12000000000},
            "active_servers": 0,
            "models": ["llama-3-8b"],
        }

        call_count = [0]

        async def mock_post(url, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                # First runner: 507
                resp = MagicMock()
                resp.status_code = 507
                resp.json.return_value = {"detail": "Insufficient capacity"}
                return resp
            else:
                # Second runner: success
                resp = MagicMock()
                resp.status_code = 201
                resp.json.return_value = {
                    "server_id": "ghi789",
                    "base_url": "http://runner2:8001/v1/server/ghi789",
                    "model": "llama-3-8b",
                }
                resp.raise_for_status = MagicMock()
                return resp

        mock = _mock_client(
            get=AsyncMock(return_value=mock_health_response),
            post=AsyncMock(side_effect=mock_post),
        )

        client = RunnerClient(
            endpoints=["http://runner1:8000", "http://runner2:8001"]
        )
        client._client = mock
        handle = await client.acquire_server("llama-3-8b", task=ModelTask.TEXTTOTEXT, config_override={})
        assert handle.server_id == "ghi789"


class TestRunnerClientRelease:

    @pytest.mark.asyncio
    async def test_release_calls_endpoint(self):
        """Verify release_server POSTs to /release endpoint."""
        mock_release_response = MagicMock()
        mock_release_response.status_code = 200
        mock_release_response.raise_for_status = MagicMock()

        mock = _mock_client(post=AsyncMock(return_value=mock_release_response))

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        handle = ServerHandle(
            base_url="http://runner1:8000/v1/server/abc123",
            server_id="abc123",
            runner_host="http://runner1:8000",
        )
        await client.release_server(handle)

        mock.post.assert_called_once()
        call_args = mock.post.call_args
        assert "/v1/server/abc123/release" in call_args[0][0]


class TestRunnerClientShutdown:

    @pytest.mark.asyncio
    async def test_shutdown_calls_endpoint(self):
        """Verify shutdown_server sends DELETE request."""
        mock_shutdown_response = MagicMock()
        mock_shutdown_response.status_code = 200
        mock_shutdown_response.raise_for_status = MagicMock()

        mock = _mock_client(delete=AsyncMock(return_value=mock_shutdown_response))

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        handle = ServerHandle(
            base_url="http://runner1:8000/v1/server/abc123",
            server_id="abc123",
            runner_host="http://runner1:8000",
        )
        await client.shutdown_server(handle)

        mock.delete.assert_called_once()
        call_args = mock.delete.call_args
        assert "/v1/server/abc123" in call_args[0][0]


class TestRunnerClientModels:

    @pytest.mark.asyncio
    async def test_list_models_aggregates(self):
        """Two runners return models, results are deduplicated by model id."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "llama-3-8b",
                "name": "Llama 3 8B",
                "model": "meta-llama/Llama-3-8B",
                "task": "TextToText",
                "modified_at": "2025-01-01",
                "digest": "abc123",
                "provider": "llama_cpp",
                "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "8B", "size": 4000000000, "original_ctx": 8192},
            },
        ]

        mock = _mock_client(get=AsyncMock(return_value=mock_response))

        client = RunnerClient(
            endpoints=["http://runner1:8000", "http://runner2:8001"]
        )
        client._client = mock
        models = await client.list_models()

        # Deduplicated: should only have one entry
        model_ids = [m.id for m in models]
        assert model_ids.count("llama-3-8b") == 1

    @pytest.mark.asyncio
    async def test_model_by_task_filters(self):
        """GET /v1/models?task=TextToEmbeddings returns first match."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "nomic-embed",
                "name": "Nomic Embed",
                "model": "nomic-ai/nomic-embed",
                "task": "TextToEmbeddings",
                "modified_at": "2025-01-01",
                "digest": "def456",
                "provider": "llama_cpp",
                "details": {"format": "gguf", "family": "nomic", "families": ["nomic"], "parameter_size": "0.3B", "size": 200000000, "original_ctx": 8192},
            },
            {
                "id": "llama-3-8b",
                "name": "Llama 3 8B",
                "model": "meta-llama/Llama-3-8B",
                "task": "TextToText",
                "modified_at": "2025-01-01",
                "digest": "abc123",
                "provider": "llama_cpp",
                "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "8B", "size": 4000000000, "original_ctx": 8192},
            },
        ]

        mock = _mock_client(get=AsyncMock(return_value=mock_response))

        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        result = await client.model_by_task(ModelTask.TEXTTOEMBEDDINGS)

        assert result is not None
        assert result.id == "nomic-embed"
        assert result.task == ModelTask.TEXTTOEMBEDDINGS

        # Verify the task query param was included in the call
        call_args = mock.get.call_args
        assert call_args[1].get("params", {}).get("task") == "TextToEmbeddings"


class TestRunnerClientHandleLifecycle:
    """Handle registry: acquire registers, release/shutdown unregister, aclose shuts down all."""

    @pytest.mark.asyncio
    async def test_acquire_registers_handle(self):
        """acquire_server() auto-registers the returned handle."""
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {
            "server_id": "abc", "base_url": "http://r1:8000/v1/server/abc", "model": "m"
        }
        mock_create.raise_for_status = MagicMock()
        mock = _mock_client(post=AsyncMock(return_value=mock_create))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"m": ["http://r1:8000"]}
        handle = await client.acquire_server("m")
        assert handle in client._active_handles

    @pytest.mark.asyncio
    async def test_release_unregisters_handle(self):
        """release_server() removes the handle from the registry."""
        mock_release = MagicMock()
        mock_release.status_code = 200
        mock_release.raise_for_status = MagicMock()
        mock = _mock_client(post=AsyncMock(return_value=mock_release))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        handle = ServerHandle(
            base_url="http://r1:8000/v1/server/abc",
            server_id="abc",
            runner_host="http://r1:8000",
        )
        client.register_handle(handle)
        assert handle in client._active_handles
        await client.release_server(handle)
        assert handle not in client._active_handles

    @pytest.mark.asyncio
    async def test_shutdown_unregisters_handle(self):
        """shutdown_server() removes the handle from the registry."""
        mock_shutdown = MagicMock()
        mock_shutdown.status_code = 200
        mock_shutdown.raise_for_status = MagicMock()
        mock = _mock_client(delete=AsyncMock(return_value=mock_shutdown))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        handle = ServerHandle(
            base_url="http://r1:8000/v1/server/abc",
            server_id="abc",
            runner_host="http://r1:8000",
        )
        client.register_handle(handle)
        await client.shutdown_server(handle)
        assert handle not in client._active_handles

    @pytest.mark.asyncio
    async def test_shutdown_all_handles_on_aclose(self):
        """aclose() calls shutdown_server for each registered handle."""
        shutdown_calls = []

        async def mock_delete(url, **kw):
            shutdown_calls.append(url)
            r = MagicMock()
            r.status_code = 200
            r.raise_for_status = MagicMock()
            return r

        mock = _mock_client(delete=AsyncMock(side_effect=mock_delete))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock

        # Register two handles
        for sid in ["h1", "h2"]:
            client.register_handle(ServerHandle(
                base_url=f"http://r1:8000/v1/server/{sid}",
                server_id=sid,
                runner_host="http://r1:8000",
            ))

        await client.aclose()

        assert len(shutdown_calls) == 2
        assert any("h1" in u for u in shutdown_calls)
        assert any("h2" in u for u in shutdown_calls)
        assert client._active_handles == set()

    @pytest.mark.asyncio
    async def test_shutdown_all_handles_skips_on_error(self):
        """If one handle fails to shutdown, others are still cleaned up."""
        call_count = [0]

        async def mock_delete(url, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("connection refused")
            r = MagicMock()
            r.status_code = 200
            r.raise_for_status = MagicMock()
            return r

        mock = _mock_client(delete=AsyncMock(side_effect=mock_delete))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock

        for sid in ["h1", "h2"]:
            client.register_handle(ServerHandle(
                base_url=f"http://r1:8000/v1/server/{sid}",
                server_id=sid,
                runner_host="http://r1:8000",
            ))

        await client.aclose()

        # Both handles attempted
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_acquire_passes_num_ctx(self):
        """acquire_server() forwards num_ctx in the POST payload."""
        captured_payload = {}

        async def mock_post(url, json=None, **kw):
            captured_payload.update(json or {})
            r = MagicMock()
            r.status_code = 201
            r.json.return_value = {
                "server_id": "abc", "base_url": "http://r1:8000/v1/server/abc", "model": "m"
            }
            r.raise_for_status = MagicMock()
            return r

        mock = _mock_client(post=AsyncMock(side_effect=mock_post))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"m": ["http://r1:8000"]}
        await client.acquire_server("m", num_ctx=128000)
        assert captured_payload["num_ctx"] == 128000

    @pytest.mark.asyncio
    async def test_acquire_without_num_ctx(self):
        """acquire_server() omits num_ctx when not provided."""
        captured_payload = {}

        async def mock_post(url, json=None, **kw):
            captured_payload.update(json or {})
            r = MagicMock()
            r.status_code = 201
            r.json.return_value = {
                "server_id": "abc", "base_url": "http://r1:8000/v1/server/abc", "model": "m"
            }
            r.raise_for_status = MagicMock()
            return r

        mock = _mock_client(post=AsyncMock(side_effect=mock_post))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"m": ["http://r1:8000"]}
        await client.acquire_server("m")
        assert "num_ctx" not in captured_payload


class TestRunnerClientConfig:

    def test_default_refresh_interval(self):
        from config import MODEL_CACHE_REFRESH_SEC
        assert MODEL_CACHE_REFRESH_SEC == 60

    def test_refresh_interval_from_env(self, monkeypatch):
        """MODEL_CACHE_REFRESH_SEC reads from env var."""
        import importlib
        monkeypatch.setenv("MODEL_CACHE_REFRESH_SEC", "120")
        import config
        importlib.reload(config)
        assert config.MODEL_CACHE_REFRESH_SEC == 120


class TestRunnerClientConnectionPooling:
    """Tests for the persistent client / connection pooling behavior."""

    def test_get_client_creates_once(self):
        """Calling _get_client() twice returns the same instance."""
        client = RunnerClient(endpoints=["http://runner1:8000"])
        c1 = client._get_client()
        c2 = client._get_client()
        assert c1 is c2

    def test_get_client_recreates_after_close(self):
        """If the client is closed, _get_client() creates a new one."""
        client = RunnerClient(endpoints=["http://runner1:8000"])
        # Simulate a closed client
        old_mock = MagicMock()
        old_mock.is_closed = True
        client._client = old_mock

        c2 = client._get_client()
        # Should have created a fresh httpx.AsyncClient, not returned the mock
        assert c2 is not old_mock
        assert isinstance(c2, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_aclose_closes_client(self):
        """aclose() closes the internal client."""
        client = RunnerClient(endpoints=["http://runner1:8000"])
        mock = AsyncMock()
        mock.is_closed = False
        client._client = mock

        await client.aclose()

        mock.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_aclose_noop_when_no_client(self):
        """aclose() is safe when no client has been created."""
        client = RunnerClient(endpoints=["http://runner1:8000"])
        await client.aclose()  # should not raise


class TestRunnerClientModelMap:
    @pytest.mark.asyncio
    async def test_refresh_builds_map(self):
        r1 = MagicMock()
        r1.status_code = 200
        r1.json.return_value = [
            {"id": "model-a", "name": "A", "model": "a", "task": "TextToText", "modified_at": "2025-01-01", "digest": "a", "provider": "llama_cpp", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "8B", "size": 4e9, "original_ctx": 8192}},
            {"id": "model-b", "name": "B", "model": "b", "task": "TextToText", "modified_at": "2025-01-01", "digest": "b", "provider": "llama_cpp", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "7B", "size": 3e9, "original_ctx": 4096}},
        ]
        r2 = MagicMock()
        r2.status_code = 200
        r2.json.return_value = [
            {"id": "model-b", "name": "B", "model": "b", "task": "TextToText", "modified_at": "2025-01-01", "digest": "b", "provider": "llama_cpp", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "7B", "size": 3e9, "original_ctx": 4096}},
            {"id": "model-c", "name": "C", "model": "c", "task": "TextToEmbeddings", "modified_at": "2025-01-01", "digest": "c", "provider": "llama_cpp", "details": {"format": "gguf", "family": "nomic", "families": ["nomic"], "parameter_size": "0.3B", "size": 2e8, "original_ctx": 8192}},
        ]
        idx = [0]
        async def mock_get(url, **kw):
            r = [r1, r2][idx[0]]; idx[0] += 1; return r
        mock = _mock_client(get=AsyncMock(side_effect=mock_get))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        await client.refresh_model_map()
        assert client._model_map["model-a"] == ["http://r1:8000"]
        assert client._model_map["model-b"] == ["http://r1:8000", "http://r2:8001"]
        assert client._model_map["model-c"] == ["http://r2:8001"]

    @pytest.mark.asyncio
    async def test_refresh_skips_unhealthy_runner(self):
        r1 = MagicMock()
        r1.status_code = 200
        r1.json.return_value = [
            {"id": "model-a", "name": "A", "model": "a", "task": "TextToText", "modified_at": "2025-01-01", "digest": "a", "provider": "llama_cpp", "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "8B", "size": 4e9, "original_ctx": 8192}},
        ]
        idx = [0]
        async def mock_get(url, **kw):
            v = [r1, Exception("conn refused")][idx[0]]; idx[0] += 1
            if isinstance(v, Exception):
                raise v
            return v
        mock = _mock_client(get=AsyncMock(side_effect=mock_get))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        await client.refresh_model_map()
        assert client._model_map["model-a"] == ["http://r1:8000"]
        assert all("http://r2:8001" not in v for v in client._model_map.values())


class TestRunnerClientSlidingRefresh:
    @pytest.mark.asyncio
    async def test_schedule_refresh_on_acquire(self):
        """Successful acquire_server schedules a refresh task."""
        mock_health = MagicMock()
        mock_health.status_code = 200
        mock_health.json.return_value = {"status": "ok", "gpu": {"available_vram_bytes": 12e9}, "active_servers": 0, "models": ["model-a"]}
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {"server_id": "abc", "base_url": "http://r1:8000/v1/server/abc", "model": "model-a"}
        mock_create.raise_for_status = MagicMock()
        mock = _mock_client(get=AsyncMock(return_value=mock_health), post=AsyncMock(return_value=mock_create))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}
        handle = await client.acquire_server("model-a")
        assert handle is not None
        assert client._refresh_task is not None
        assert isinstance(client._refresh_task, asyncio.Task)

    @pytest.mark.asyncio
    async def test_new_schedule_cancels_pending(self):
        """A second acquire cancels the pending refresh from the first."""
        mock_health = MagicMock()
        mock_health.status_code = 200
        mock_health.json.return_value = {"status": "ok", "gpu": {"available_vram_bytes": 12e9}, "active_servers": 0, "models": ["model-a"]}
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {"server_id": "abc", "base_url": "http://r1:8000/v1/server/abc", "model": "model-a"}
        mock_create.raise_for_status = MagicMock()
        mock = _mock_client(get=AsyncMock(return_value=mock_health), post=AsyncMock(return_value=mock_create))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}
        await client.acquire_server("model-a")
        first_task = client._refresh_task
        await client.acquire_server("model-a")
        second_task = client._refresh_task
        assert first_task is not second_task
        with pytest.raises(asyncio.CancelledError):
            await first_task
        assert first_task.cancelled()


class TestRunnerClientAcquireWithMap:
    @pytest.mark.asyncio
    async def test_acquire_uses_cached_map(self):
        """acquire_server uses cached map, skips health checks."""
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {"server_id": "abc", "base_url": "http://r2:8001/v1/server/abc", "model": "model-c"}
        mock_create.raise_for_status = MagicMock()
        mock = _mock_client(post=AsyncMock(return_value=mock_create))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"], "model-c": ["http://r2:8001"]}
        handle = await client.acquire_server("model-c")
        assert handle.server_id == "abc"
        assert handle.runner_host == "http://r2:8001"
        # Should NOT have called get() for health check
        mock.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_acquire_fallback_on_missing_model(self):
        """Model not in map falls back to health-check scan."""
        mock_health = MagicMock()
        mock_health.status_code = 200
        mock_health.json.return_value = {"status": "ok", "gpu": {"available_vram_bytes": 12e9}, "active_servers": 0, "models": ["model-x"]}
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {"server_id": "def", "base_url": "http://r1:8000/v1/server/def", "model": "model-x"}
        mock_create.raise_for_status = MagicMock()
        mock = _mock_client(get=AsyncMock(return_value=mock_health), post=AsyncMock(return_value=mock_create))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {}  # empty map
        handle = await client.acquire_server("model-x")
        assert handle.server_id == "def"
        # Should have called get() for health check fallback
        mock.get.assert_called()

    @pytest.mark.asyncio
    async def test_acquire_fallback_on_507(self):
        """507 on primary runner falls through to next in map."""
        calls = [0]
        async def mock_post(url, **kw):
            calls[0] += 1
            if calls[0] == 1:
                r = MagicMock(); r.status_code = 507; return r
            r = MagicMock()
            r.status_code = 201
            r.json.return_value = {"server_id": "ghi", "base_url": "http://r2:8001/v1/server/ghi", "model": "model-b"}
            r.raise_for_status = MagicMock()
            return r
        mock = _mock_client(post=AsyncMock(side_effect=mock_post))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        client._model_map = {"model-b": ["http://r1:8000", "http://r2:8001"]}
        handle = await client.acquire_server("model-b")
        assert handle.server_id == "ghi"
        assert handle.runner_host == "http://r2:8001"


class TestRunnerClientContextReduction:
    """Tests for context-reduction retry logic in acquire_server."""

    def test_reduce_context_halves(self):
        """_reduce_context halves the value."""
        client = RunnerClient(endpoints=["http://r1:8000"])
        assert client._reduce_context(100000) == 50000
        assert client._reduce_context(50000) == 25000
        assert client._reduce_context(25000) == 12500

    def test_reduce_context_floors_at_minimum(self):
        """_reduce_context respects the 2048 floor."""
        client = RunnerClient(endpoints=["http://r1:8000"])
        assert client._reduce_context(2048) == 2048
        assert client._reduce_context(1024) == 2048
        assert client._reduce_context(2049) == 2048

    def test_is_context_error_detects_context_keywords(self):
        """_is_context_error returns True for context-related 500 errors."""
        resp = MagicMock()
        resp.status_code = 500
        resp.json.return_value = {
            "detail": "All retry attempts failed for model Qwen3.5-4B. "
                      "Server cannot start with any reduced context window."
        }
        assert RunnerClient._is_context_error(resp) is True

    def test_is_context_error_detects_oom(self):
        """_is_context_error returns True for OOM errors."""
        resp = MagicMock()
        resp.status_code = 500
        resp.json.return_value = {"detail": "OOM: insufficient memory for context"}
        assert RunnerClient._is_context_error(resp) is True

    def test_is_context_error_returns_false_for_non_500(self):
        """_is_context_error returns False for non-500 responses."""
        resp = MagicMock()
        resp.status_code = 507
        resp.json.return_value = {"detail": "context too large"}
        assert RunnerClient._is_context_error(resp) is False

    def test_is_context_error_returns_false_for_unrelated_500(self):
        """_is_context_error returns False for unrelated 500 errors."""
        resp = MagicMock()
        resp.status_code = 500
        resp.json.return_value = {"detail": "Internal server error: database timeout"}
        assert RunnerClient._is_context_error(resp) is False

    @pytest.mark.asyncio
    async def test_acquire_retries_with_reduced_context(self):
        """When runner returns 500, acquire_server retries with reduced num_ctx."""
        calls = []

        async def mock_post(url, **kw):
            body = kw.get("json", {})
            calls.append(body.get("num_ctx"))
            if body.get("num_ctx", 0) > 50000:
                # First attempt with large context fails
                r = MagicMock()
                r.status_code = 500
                r.json.return_value = {
                    "detail": "All retry attempts failed. Server cannot start with any reduced context window."
                }
                r.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "500 Server Error", request=None, response=r
                )
                raise r.raise_for_status.side_effect
            else:
                # Reduced context succeeds
                r = MagicMock()
                r.status_code = 201
                r.json.return_value = {
                    "server_id": "ctx-reduced",
                    "base_url": "http://r1:8000/v1/server/ctx-reduced",
                    "model": "small-model",
                }
                r.raise_for_status = MagicMock()
                return r

        mock = _mock_client(post=AsyncMock(side_effect=mock_post))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"small-model": ["http://r1:8000"]}

        handle = await client.acquire_server("small-model", num_ctx=100000)
        assert handle.server_id == "ctx-reduced"
        # Should have tried 100000 first, then 50000
        assert calls == [100000, 50000]

    @pytest.mark.asyncio
    async def test_acquire_exhausts_context_retries(self):
        """When all context retries fail, RuntimeError is raised."""
        async def mock_post(url, **kw):
            r = MagicMock()
            r.status_code = 500
            r.json.return_value = {
                "detail": "Server cannot start with any reduced context window."
            }
            r.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Server Error", request=None, response=r
            )
            raise r.raise_for_status.side_effect

        mock = _mock_client(post=AsyncMock(side_effect=mock_post))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"small-model": ["http://r1:8000"]}

        with pytest.raises(RuntimeError, match="all reduced context retries exhausted"):
            await client.acquire_server("small-model", num_ctx=100000)

    @pytest.mark.asyncio
    async def test_acquire_no_context_retry_without_num_ctx(self):
        """When no num_ctx is provided, no context reduction retry occurs."""
        async def mock_post(url, **kw):
            r = MagicMock()
            r.status_code = 500
            r.json.return_value = {"detail": "Server start failed"}
            r.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 Server Error", request=None, response=r
            )
            raise r.raise_for_status.side_effect

        mock = _mock_client(post=AsyncMock(side_effect=mock_post))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"small-model": ["http://r1:8000"]}

        with pytest.raises(RuntimeError, match="No healthy runner"):
            await client.acquire_server("small-model")
        # Should have only 1 call (no retries)
        assert mock.post.call_count == 1
