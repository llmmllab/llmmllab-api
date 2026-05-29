"""
Unit tests for services/runner_client.py.

Tests the RunnerClient HTTP client that routes requests among multiple
llmmllab-runner service instances.  The client now uses a persistent
``httpx.AsyncClient``, so tests mock ``_get_client()`` instead of patching
the ``httpx.AsyncClient`` constructor.
"""

import asyncio
import time
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


class TestRunnerClientDefaultModel:
    """Tests for default_model_by_task() — uses /v1/models/default endpoint."""

    @pytest.mark.asyncio
    async def test_default_model_by_task_returns_default(self):
        """GET /v1/models/default returns the configured default model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "Qwen3_6_27B",
            "name": "Qwen3.6-27B",
            "model": "Qwen3.6-27B",
            "task": "TextToText",
            "modified_at": "2026-04-22",
            "digest": "abc123",
            "provider": "llama_cpp",
            "is_default": True,
            "details": {"format": "gguf", "family": "qwen", "families": ["qwen"], "parameter_size": "26.9B", "size": 35325163744, "original_ctx": 2048},
        }

        mock = _mock_client(get=AsyncMock(return_value=mock_response))
        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        result = await client.default_model_by_task(ModelTask.TEXTTOTEXT)

        assert result is not None
        assert result.id == "Qwen3_6_27B"
        assert result.is_default is True

        # Verify the /v1/models/default endpoint was called
        call_args = mock.get.call_args
        assert "/v1/models/default" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_default_model_falls_back_to_model_by_task_on_404(self):
        """If /v1/models/default returns 404, falls back to model_by_task."""
        call_count = [0]

        async def mock_get(url, **kw):
            call_count[0] += 1
            if "/v1/models/default" in url:
                r = MagicMock()
                r.status_code = 404
                return r
            else:
                r = MagicMock()
                r.status_code = 200
                r.json.return_value = [
                    {
                        "id": "fallback-model",
                        "name": "Fallback",
                        "model": "fallback",
                        "task": "TextToText",
                        "modified_at": "2026-01-01",
                        "digest": "def456",
                        "provider": "llama_cpp",
                        "details": {"format": "gguf", "family": "llama", "families": ["llama"], "parameter_size": "8B", "size": 4e9, "original_ctx": 8192},
                    },
                ]
                return r

        mock = _mock_client(get=AsyncMock(side_effect=mock_get))
        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        result = await client.default_model_by_task(ModelTask.TEXTTOTEXT)

        assert result is not None
        assert result.id == "fallback-model"
        # Should have tried /v1/models/default first, then /v1/models
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_default_model_returns_none_when_all_runners_fail(self):
        """If all runners fail, returns None."""
        mock = _mock_client(get=AsyncMock(side_effect=Exception("connection refused")))
        client = RunnerClient(endpoints=["http://runner1:8000"])
        client._client = mock
        result = await client.default_model_by_task(ModelTask.TEXTTOTEXT)
        assert result is None


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
    async def test_refresh_builds_pipeline_map(self):
        """In-process models populate _pipeline_map indexed by pipeline name."""
        r1 = MagicMock()
        r1.status_code = 200
        r1.json.return_value = [
            {"id": "hunyuan3d-2.1", "name": "Hy3D", "model": "h", "task": "ImageTo3D",
             "modified_at": "2025-01-01", "digest": "h",
             "provider": "in_process", "pipeline": "img23d",
             "details": {"format": "safetensors", "family": "h", "families": ["h"],
                         "parameter_size": "2.7B", "size": 15e9, "original_ctx": 0}},
        ]
        r2 = MagicMock()
        r2.status_code = 200
        r2.json.return_value = [
            {"id": "rmbg-2.0", "name": "Rmbg", "model": "r", "task": "ImageToImage",
             "modified_at": "2025-01-01", "digest": "r",
             "provider": "in_process", "pipeline": "rembg",
             "details": {"format": "safetensors", "family": "r", "families": ["r"],
                         "parameter_size": "0.22B", "size": 885e6, "original_ctx": 0}},
        ]
        idx = [0]
        async def mock_get(url, **kw):
            r = [r1, r2][idx[0]]; idx[0] += 1; return r
        mock = _mock_client(get=AsyncMock(side_effect=mock_get))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        await client.refresh_model_map()
        assert client._pipeline_map["img23d"] == ["http://r1:8000"]
        assert client._pipeline_map["rembg"] == ["http://r2:8001"]
        # llama_cpp / sd models don't pollute the pipeline map
        assert "llama" not in client._pipeline_map

    @pytest.mark.asyncio
    async def test_select_pipeline_endpoint_routes_by_pipeline_map(self):
        """select_pipeline_endpoint returns the runner advertising the pipeline."""
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._pipeline_map = {"rembg": ["http://r2:8001"]}
        # Stub _health so the ranked branch can pick a candidate.
        async def fake_health(ep):
            return {"gpu": {"available_vram_bytes": 12e9}}
        client._health = fake_health  # type: ignore[assignment]
        client._is_circuit_open = MagicMock(return_value=False)
        ep = await client.select_pipeline_endpoint("rembg")
        assert ep == "http://r2:8001"

    @pytest.mark.asyncio
    async def test_select_pipeline_endpoint_falls_back_when_unmapped(self):
        """Unknown pipeline name -> falls back to endpoints[0] (caller hits 404)."""
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        # Skip the lazy refresh by pre-populating the map (empty for our key).
        client._pipeline_map = {"img23d": ["http://r1:8000"]}
        ep = await client.select_pipeline_endpoint("nonexistent")
        assert ep == "http://r1:8000"

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
        """acquire_server uses cached map, skips /health checks.

        Note: acquire_server still calls GET /v1/status on the chosen
        endpoint for restart-epoch detection (added with runner-restart
        recovery). What it must *not* do is fall back to the per-endpoint
        /health scan in _select_runner, since the model map already tells
        it which runner owns this model.
        """
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {"server_id": "abc", "base_url": "http://r2:8001/v1/server/abc", "model": "model-c"}
        mock_create.raise_for_status = MagicMock()
        # /v1/status response so _check_runner_epoch succeeds quietly.
        mock_status = MagicMock()
        mock_status.status_code = 200
        mock_status.json.return_value = {"startup_epoch": 1}
        mock = _mock_client(
            post=AsyncMock(return_value=mock_create),
            get=AsyncMock(return_value=mock_status),
        )
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"], "model-c": ["http://r2:8001"]}
        handle = await client.acquire_server("model-c")
        assert handle.server_id == "abc"
        assert handle.runner_host == "http://r2:8001"
        # Must not have hit /health on either endpoint — the cached map
        # bypasses _select_runner. /v1/status calls are allowed.
        get_urls = [c.args[0] if c.args else c.kwargs.get("url", "") for c in mock.get.call_args_list]
        assert all("/health" not in url for url in get_urls), (
            f"acquire_server with cached map must skip /health checks, got: {get_urls}"
        )
        # And the only endpoint touched should be the mapped one.
        assert all("r2:8001" in url for url in get_urls), (
            f"acquire_server should only contact the mapped endpoint, got: {get_urls}"
        )

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


class TestRunnerClientCircuitBreaker:
    """Tests for circuit breaker behavior on acquire failures."""

    @pytest.mark.asyncio
    async def test_circuit_opens_on_connection_error(self):
        """A connection error during acquire immediately trips the circuit."""
        mock = _mock_client(
            post=AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        )
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}

        with pytest.raises(RuntimeError, match="No healthy runner"):
            await client.acquire_server("model-a")

        # Circuit should be open
        assert client._is_circuit_open("http://r1:8000")
        assert client._acquire_failures["http://r1:8000"] == client._MAX_ACQUIRE_FAILURES

    @pytest.mark.asyncio
    async def test_circuit_opens_on_http_error(self):
        """An HTTP error (e.g. 500) during acquire immediately trips the circuit."""
        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=MagicMock(status_code=500)
        )
        mock = _mock_client(post=AsyncMock(return_value=error_resp))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}

        with pytest.raises(RuntimeError, match="No healthy runner"):
            await client.acquire_server("model-a")

        # Circuit should be open
        assert client._is_circuit_open("http://r1:8000")
        assert client._acquire_failures["http://r1:8000"] == client._MAX_ACQUIRE_FAILURES

    @pytest.mark.asyncio
    async def test_circuit_opens_on_timeout(self):
        """A timeout during acquire immediately trips the circuit."""
        mock = _mock_client(
            post=AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
        )
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}

        with pytest.raises(RuntimeError, match="No healthy runner"):
            await client.acquire_server("model-a")

        assert client._is_circuit_open("http://r1:8000")

    @pytest.mark.asyncio
    async def test_circuit_prevents_retry_on_open_endpoint(self):
        """When circuit is open, acquire skips the endpoint entirely."""
        import time
        mock = _mock_client(post=AsyncMock(side_effect=Exception("should not be called")))
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}
        # Manually open the circuit with a recent timestamp
        client._acquire_failures["http://r1:8000"] = client._MAX_ACQUIRE_FAILURES
        client._unhealthy_since["http://r1:8000"] = time.monotonic()

        with pytest.raises(RuntimeError, match="No healthy runner"):
            await client.acquire_server("model-a")

        # post should NOT have been called — endpoint was skipped
        mock.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_circuit_closes_after_window(self):
        """Circuit resets after UNHEALTHY_WINDOW seconds."""
        import time
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._acquire_failures["http://r1:8000"] = client._MAX_ACQUIRE_FAILURES
        client._unhealthy_since["http://r1:8000"] = time.monotonic()

        assert client._is_circuit_open("http://r1:8000")

        # Simulate time passing beyond the window
        with patch("services.runner_client.time") as mock_time:
            mock_time.monotonic.return_value = client._unhealthy_since["http://r1:8000"] + client._UNHEALTHY_WINDOW + 1.0
            assert not client._is_circuit_open("http://r1:8000")

    @pytest.mark.asyncio
    async def test_cleanup_called_on_connection_error(self):
        """Orphaned servers are cleaned up when a connection error occurs."""
        # Pre-populate active servers
        mock_delete = AsyncMock()
        mock = _mock_client(
            post=AsyncMock(side_effect=httpx.ConnectError("connection refused")),
            delete=mock_delete,
        )
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}
        client._active_servers_by_endpoint["http://r1:8000"] = {"srv-1", "srv-2"}

        with pytest.raises(RuntimeError, match="No healthy runner"):
            await client.acquire_server("model-a")

        # Allow the async cleanup task to run
        await asyncio.sleep(0.1)

        # Both servers should have been cleaned up
        assert mock_delete.call_count == 2
        urls = [c[0][0] for c in mock_delete.call_args_list]
        assert "http://r1:8000/v1/server/srv-1" in urls
        assert "http://r1:8000/v1/server/srv-2" in urls

    @pytest.mark.asyncio
    async def test_cleanup_called_on_http_error(self):
        """Orphaned servers are cleaned up when an HTTP error occurs."""
        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=MagicMock(status_code=500)
        )
        mock_delete = AsyncMock()
        mock = _mock_client(
            post=AsyncMock(return_value=error_resp),
            delete=mock_delete,
        )
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"]}
        client._active_servers_by_endpoint["http://r1:8000"] = {"srv-3"}

        with pytest.raises(RuntimeError, match="No healthy runner"):
            await client.acquire_server("model-a")

        await asyncio.sleep(0.1)

        assert mock_delete.call_count == 1
        assert "http://r1:8000/v1/server/srv-3" in mock_delete.call_args[0][0]

    @pytest.mark.asyncio
    async def test_fallback_to_second_runner_on_circuit_trip(self):
        """When first runner trips circuit, acquire falls through to second."""
        calls = [0]

        async def mock_post(url, **kw):
            calls[0] += 1
            if "r1" in url:
                raise httpx.ConnectError("connection refused")
            # r2 succeeds
            r = MagicMock()
            r.status_code = 201
            r.json.return_value = {
                "server_id": "srv-ok",
                "base_url": "http://r2:8001/v1/server/srv-ok",
                "model": "model-a",
            }
            r.raise_for_status = MagicMock()
            return r

        mock = _mock_client(post=AsyncMock(side_effect=mock_post))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000", "http://r2:8001"]}

        handle = await client.acquire_server("model-a")
        assert handle.server_id == "srv-ok"
        assert handle.runner_host == "http://r2:8001"

        # First runner circuit should be open
        assert client._is_circuit_open("http://r1:8000")

    @pytest.mark.asyncio
    async def test_health_check_trips_circuit_on_connection_error(self):
        """Health check connection errors trip the circuit and clean up."""
        mock_delete = AsyncMock()
        mock = _mock_client(
            get=AsyncMock(side_effect=httpx.ConnectError("connection refused")),
            delete=mock_delete,
        )
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        client._active_servers_by_endpoint["http://r1:8000"] = {"srv-h1"}

        result = await client._health("http://r1:8000")
        assert result is None

        await asyncio.sleep(0.1)

        # Circuit should be tripped
        assert client._is_circuit_open("http://r1:8000")
        # Cleanup should have been called
        mock_delete.assert_called_once()
        assert "srv-h1" in mock_delete.call_args[0][0]

    @pytest.mark.asyncio
    async def test_trip_circuit_and_cleanup_removes_from_healthy(self):
        """_trip_circuit_and_cleanup removes endpoint from _healthy list."""
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._healthy.append("http://r1:8000")

        client._trip_circuit_and_cleanup("http://r1:8000")

        assert "http://r1:8000" not in client._healthy
        assert client._is_circuit_open("http://r1:8000")


class TestRunnerClientStickyEndpoint:
    """Pin the sticky model→endpoint routing introduced 2026-05-22.

    Maximises KV-cache reuse on the runner side: once the first
    acquire for model X picks endpoint E, every subsequent acquire
    for X is steered back to E unless E becomes ineligible.
    """

    @pytest.mark.asyncio
    async def test_acquire_pins_model_to_endpoint_on_success(self):
        """A successful acquire records ``_last_endpoint_for_model``."""
        health = MagicMock(status_code=200)
        health.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12000000000},
            "models": ["m1"],
        }
        create = MagicMock(status_code=201)
        create.json.return_value = {
            "server_id": "s1",
            "base_url": "http://r1:8000/v1/server/s1",
            "model": "m1",
        }
        create.raise_for_status = MagicMock()

        mock = _mock_client(
            get=AsyncMock(return_value=health),
            post=AsyncMock(return_value=create),
        )
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = mock
        await client.acquire_server("m1", task=ModelTask.TEXTTOTEXT, config_override={})
        assert client._last_endpoint_for_model["m1"] == "http://r1:8000"

    @pytest.mark.asyncio
    async def test_select_runner_prefers_sticky_endpoint(self):
        """``_select_runner`` returns the pinned endpoint when healthy."""
        health = MagicMock(status_code=200)
        health.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12000000000},
            "models": ["m1"],
        }
        mock = _mock_client(get=AsyncMock(return_value=health))
        client = RunnerClient(
            endpoints=["http://r1:8000", "http://r2:8000"]
        )
        client._client = mock
        client._last_endpoint_for_model["m1"] = "http://r2:8000"

        chosen = await client._select_runner("m1")
        assert chosen == "http://r2:8000"

    @pytest.mark.asyncio
    async def test_select_runner_drops_pin_when_sticky_unhealthy(self):
        """If the sticky endpoint stops hosting the model, drop the pin
        and let the ranking pick a healthy alternative."""
        async def health_for(url):
            resp = MagicMock(status_code=200)
            if "r2" in url:
                # r2 lost the model
                resp.json.return_value = {
                    "status": "ok",
                    "gpu": {"available_vram_bytes": 100},
                    "models": [],
                }
            else:
                resp.json.return_value = {
                    "status": "ok",
                    "gpu": {"available_vram_bytes": 12000000000},
                    "models": ["m1"],
                }
            return resp

        async def get_(url, *args, **kwargs):
            return await health_for(url)

        mock = _mock_client(get=AsyncMock(side_effect=get_))
        client = RunnerClient(
            endpoints=["http://r1:8000", "http://r2:8000"]
        )
        client._client = mock
        client._last_endpoint_for_model["m1"] = "http://r2:8000"

        chosen = await client._select_runner("m1")
        # Sticky r2 unhealthy → fall back to r1; pin dropped.
        assert chosen == "http://r1:8000"
        assert "m1" not in client._last_endpoint_for_model

    @pytest.mark.asyncio
    async def test_select_runner_drops_pin_when_circuit_open(self):
        """A sticky endpoint with a tripped circuit breaker is not
        returned; the pin is cleared."""
        health = MagicMock(status_code=200)
        health.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12000000000},
            "models": ["m1"],
        }
        mock = _mock_client(get=AsyncMock(return_value=health))
        client = RunnerClient(
            endpoints=["http://r1:8000", "http://r2:8000"]
        )
        client._client = mock
        client._last_endpoint_for_model["m1"] = "http://r2:8000"
        # Force r2's circuit open.
        client._acquire_failures["http://r2:8000"] = client._MAX_ACQUIRE_FAILURES + 1
        client._unhealthy_since["http://r2:8000"] = time.monotonic()

        chosen = await client._select_runner("m1")
        assert chosen == "http://r1:8000"
        assert "m1" not in client._last_endpoint_for_model


class TestEffectiveVramByTensorSplit:
    """``_effective_free_vram_bytes`` should only count GPUs the model
    can actually live on, based on its ``tensor_split`` pinning.
    """

    def test_no_tensor_split_sums_all_gpus(self):
        from services.runner_client import RunnerClient

        health = {
            "gpu": {
                "0": {"free_mb": 1000},
                "1": {"free_mb": 2000},
                "2": {"free_mb": 3000},
            }
        }
        # 1000 + 2000 + 3000 = 6000 MB
        result = RunnerClient._effective_free_vram_bytes(health, None)
        assert result == 6000 * 1024 * 1024

    def test_tensor_split_only_counts_pinned_gpus(self):
        from services.runner_client import RunnerClient

        health = {
            "gpu": {
                "0": {"free_mb": 1000},
                "1": {"free_mb": 24000},
                "2": {"free_mb": 24000},
            }
        }
        # "1,0,0" pins to device 0 only — must not credit the 3090s.
        result = RunnerClient._effective_free_vram_bytes(health, "1,0,0")
        assert result == 1000 * 1024 * 1024

    def test_tensor_split_splits_across_two_gpus(self):
        from services.runner_client import RunnerClient

        health = {
            "gpu": {
                "0": {"free_mb": 1000},
                "1": {"free_mb": 24000},
                "2": {"free_mb": 24000},
            }
        }
        # "0,1,1" skips device 0, uses devices 1 and 2.
        result = RunnerClient._effective_free_vram_bytes(health, "0,1,1")
        assert result == 48000 * 1024 * 1024

    def test_malformed_tensor_split_falls_back_to_total(self):
        from services.runner_client import RunnerClient

        health = {
            "gpu": {
                "0": {"free_mb": 1000},
                "1": {"free_mb": 24000},
            }
        }
        result = RunnerClient._effective_free_vram_bytes(health, "not,a,number")
        assert result == 25000 * 1024 * 1024

    def test_aggregate_fallback_when_no_per_gpu(self):
        """Old runner without per-GPU surface — fall back to flat field."""
        from services.runner_client import RunnerClient

        health = {"gpu": {"available_vram_bytes": 5_000_000_000}}
        result = RunnerClient._effective_free_vram_bytes(health, "1,0,0")
        assert result == 5_000_000_000

    def test_aggregate_alongside_per_gpu_prefers_per_gpu(self):
        """If both are present (current runner shape), per-GPU wins."""
        from services.runner_client import RunnerClient

        health = {
            "gpu": {
                "0": {"free_mb": 1000},
                "1": {"free_mb": 24000},
                "available_vram_bytes": 999_999_999_999,
            }
        }
        # Per-GPU sum = 25000 MB, NOT the aggregate poison value.
        result = RunnerClient._effective_free_vram_bytes(health, None)
        assert result == 25000 * 1024 * 1024

    def test_empty_health_returns_zero(self):
        from services.runner_client import RunnerClient

        assert RunnerClient._effective_free_vram_bytes({}, None) == 0
        assert RunnerClient._effective_free_vram_bytes({"gpu": {}}, "1,0,0") == 0
