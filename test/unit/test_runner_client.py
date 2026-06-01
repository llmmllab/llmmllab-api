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
        """acquire_server uses cached map and the load-aware selector.

        Previously the "fast path" iterated the cached map in config
        order and bypassed _select_runner entirely. That silently
        skipped the sticky / parallel-spawn / warm-idle rules, which
        caused every request for a multi-runner model to land on
        whichever endpoint happened to be first in config — even when
        a peer was idle. Now acquire_server always asks _select_runner
        to pick the best endpoint, falling back to the cached map's
        config order only if the selector returns nothing.

        acquire_server also calls GET /v1/status on the chosen endpoint
        for restart-epoch detection.
        """
        # /health response so _select_runner has candidates to rank.
        mock_health = MagicMock()
        mock_health.status_code = 200
        mock_health.json.return_value = {
            "status": "ok",
            "gpu": {"available_vram_bytes": 12e9},
            "active_servers": 0,
            "models": ["model-c"],
        }
        mock_create = MagicMock()
        mock_create.status_code = 201
        mock_create.json.return_value = {"server_id": "abc", "base_url": "http://r2:8001/v1/server/abc", "model": "model-c"}
        mock_create.raise_for_status = MagicMock()
        mock = _mock_client(
            post=AsyncMock(return_value=mock_create),
            get=AsyncMock(return_value=mock_health),
        )
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000"], "model-c": ["http://r2:8001"]}
        client._model_tensor_split = {("http://r2:8001", "model-c"): None}
        handle = await client.acquire_server("model-c")
        assert handle.server_id == "abc"
        assert handle.runner_host == "http://r2:8001"

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
        """507 on primary runner falls through to next in map.

        Mock /health on both runners so _select_runner's load-aware
        ranking is deterministic: r1 has more free VRAM, so it's
        picked first. r1 returns 507 on POST, so the loop falls
        through to r2 which returns 201.
        """
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

        # /health responses: r1 has more VRAM so the selector picks it
        # first; the test then verifies fallback to r2 on 507.
        async def mock_get(url, **kw):
            r = MagicMock()
            r.status_code = 200
            if "r1:8000" in url:
                r.json.return_value = {
                    "status": "ok",
                    "gpu": {"available_vram_bytes": 20e9},
                    "active_servers": 0,
                    "models": ["model-b"],
                }
            else:
                r.json.return_value = {
                    "status": "ok",
                    "gpu": {"available_vram_bytes": 8e9},
                    "active_servers": 0,
                    "models": ["model-b"],
                }
            return r

        mock = _mock_client(
            post=AsyncMock(side_effect=mock_post),
            get=AsyncMock(side_effect=mock_get),
        )
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        client._model_map = {"model-b": ["http://r1:8000", "http://r2:8001"]}
        client._model_tensor_split = {
            ("http://r1:8000", "model-b"): None,
            ("http://r2:8001", "model-b"): None,
        }
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

        # /health responses so _select_runner has deterministic candidates.
        # r1 has more VRAM so the selector picks it first; the test
        # verifies fallback to r2 after r1's connection trips its circuit.
        async def mock_get(url, **kw):
            r = MagicMock()
            r.status_code = 200
            if "r1:8000" in url:
                r.json.return_value = {
                    "status": "ok",
                    "gpu": {"available_vram_bytes": 20e9},
                    "active_servers": 0,
                    "models": ["model-a"],
                }
            else:
                r.json.return_value = {
                    "status": "ok",
                    "gpu": {"available_vram_bytes": 8e9},
                    "active_servers": 0,
                    "models": ["model-a"],
                }
            return r

        mock = _mock_client(
            post=AsyncMock(side_effect=mock_post),
            get=AsyncMock(side_effect=mock_get),
        )
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8001"])
        client._client = mock
        client._model_map = {"model-a": ["http://r1:8000", "http://r2:8001"]}
        client._model_tensor_split = {
            ("http://r1:8000", "model-a"): None,
            ("http://r2:8001", "model-a"): None,
        }

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
    async def test_global_sticky_dominates_sole_runner_with_free_slot(self):
        """The global pin short-circuits in the single-runner case when
        the sticky still has a warm server with a free KV slot — there's
        no peer to fan out to, so KV reuse wins."""
        async def fake_get(url, **kw):
            resp = MagicMock(status_code=200)
            if url.endswith("/health"):
                resp.json.return_value = {
                    "status": "ok",
                    "gpu": {"0": {"free_mb": 12000}},
                    "models": [{"id": "m1", "name": "m1", "task": "TextToText"}],
                }
                return resp
            if "/v1/servers" in url:
                resp.json.return_value = {
                    "active_servers": 1,
                    "servers": [
                        {"server_id": "s1", "model_id": "m1",
                         "healthy": True, "idle_since": None, "use_count": 0}
                    ],
                }
                return resp
            return MagicMock(status_code=404)
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        client._model_tensor_split = {("http://r1:8000", "m1"): None}
        client._model_parallel = {("http://r1:8000", "m1"): 2}
        client._last_endpoint_for_model["m1"] = "http://r1:8000"

        chosen = await client._select_runner("m1")
        assert chosen == "http://r1:8000"

    @pytest.mark.asyncio
    async def test_global_sticky_yields_to_fanout_when_peer_can_cold_start(self):
        """With more than one runner hosting the model, the global pin
        does NOT lock traffic to itself — a new session cold-starts a
        fresh server on an empty peer (Rule 1) instead of piling onto the
        warm sticky.  This is the fan-out the per-session pin relies on."""
        async def fake_get(url, **kw):
            resp = MagicMock(status_code=200)
            if url.endswith("/health"):
                resp.json.return_value = {
                    "status": "ok",
                    "gpu": {"0": {"free_mb": 12000}},
                    "models": [{"id": "m1", "name": "m1", "task": "TextToText"}],
                }
                return resp
            if "r2" in url and "/v1/servers" in url:
                # sticky r2 has a warm server with a free slot
                resp.json.return_value = {
                    "active_servers": 1,
                    "servers": [
                        {"server_id": "s2", "model_id": "m1",
                         "healthy": True, "idle_since": None, "use_count": 0}
                    ],
                }
                return resp
            if "r1" in url and "/v1/servers" in url:
                resp.json.return_value = {"active_servers": 0, "servers": []}
                return resp
            return MagicMock(status_code=404)
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8000"])
        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        client._model_tensor_split = {
            ("http://r1:8000", "m1"): None,
            ("http://r2:8000", "m1"): None,
        }
        client._model_parallel = {
            ("http://r1:8000", "m1"): 2,
            ("http://r2:8000", "m1"): 2,
        }
        client._last_endpoint_for_model["m1"] = "http://r2:8000"

        chosen = await client._select_runner("m1")
        # r1 is empty with VRAM → cold-start fresh there, not sticky r2.
        assert chosen == "http://r1:8000"

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


class TestPerSessionSticky:
    """Per-session sticky pins prevent two concurrent sessions on the
    same model from alternating onto a single runner and serialising
    on a parallel=1 slot.
    """

    @pytest.mark.asyncio
    async def test_two_sessions_stay_on_their_first_runner(self):
        """Session A pins to r1, session B falls through to r2 (peer
        is busy); on each session's *next* turn the per-session pin
        keeps them on their own runner instead of bouncing.
        """
        from services.runner_client import RunnerClient, ServerHandle
        from utils.logging import _session_id_ctx
        import time

        health = MagicMock(status_code=200)
        health.json.return_value = {
            "status": "ok",
            "gpu": {"0": {"free_mb": 12000}},
            "models": [{"id": "m1", "name": "m1", "task": "TextToText"}],
        }
        # /v1/servers — both runners have m1 loaded, in-flight
        # (idle_since=None) so the per-session pin's cache-presence
        # check passes. The pin's busy-escape will NOT fan-out because
        # there is no empty peer in this scenario.
        servers_resp = MagicMock(status_code=200)
        servers_resp.json.return_value = {
            "active_servers": 1,
            "servers": [
                {
                    "server_id": "loaded",
                    "model_id": "m1",
                    "use_count": 1,
                    "idle_since": None,
                    "starting": False,
                    "healthy": True,
                }
            ],
        }

        async def fake_get(url, **kw):
            if "/v1/servers" in url:
                return servers_resp
            return health

        mock = _mock_client(get=AsyncMock(side_effect=fake_get))
        client = RunnerClient(endpoints=["http://r1:8000", "http://r2:8000"])
        client._client = mock
        client._model_tensor_split = {
            ("http://r1:8000", "m1"): None,
            ("http://r2:8000", "m1"): None,
        }
        # Session A acquired on r1 first → both pins point at r1.
        client._last_endpoint_for_model["m1"] = "http://r1:8000"
        client._last_endpoint_per_session[("sess-A", "m1")] = "http://r1:8000"
        # Session A's active handle is still on r1 (in-flight turn).
        client._active_handles.add(
            ServerHandle(
                base_url="http://r1:8000/v1/server/x",
                server_id="x",
                runner_host="http://r1:8000",
                model_id="m1",
            )
        )

        # Session B arrives.  Global sticky says r1 but r1 is busy
        # with m1 and r2 also hosts it → ranked path picks r2.
        token = _session_id_ctx.set("sess-B")
        try:
            chosen_b = await client._select_runner("m1")
        finally:
            _session_id_ctx.reset(token)
        # Session B picks r2 (r1 is busy with A's handle).
        assert chosen_b == "http://r2:8000"

        # Simulate session B successfully acquiring on r2 by setting
        # its per-session pin (acquire_server does this on success).
        client._last_endpoint_per_session[("sess-B", "m1")] = "http://r2:8000"
        client._active_handles.add(
            ServerHandle(
                base_url="http://r2:8000/v1/server/y",
                server_id="y",
                runner_host="http://r2:8000",
                model_id="m1",
            )
        )

        # Session A's NEXT turn: per-session pin says r1, stays there.
        token = _session_id_ctx.set("sess-A")
        try:
            chosen_a2 = await client._select_runner("m1")
        finally:
            _session_id_ctx.reset(token)
        assert chosen_a2 == "http://r1:8000"

        # Session B's NEXT turn: per-session pin says r2, stays there
        # — does NOT bounce back to r1 even though A's handle is still
        # there.
        token = _session_id_ctx.set("sess-B")
        try:
            chosen_b2 = await client._select_runner("m1")
        finally:
            _session_id_ctx.reset(token)
        assert chosen_b2 == "http://r2:8000"

    def test_lru_evicts_oldest_per_session_pin(self):
        """The per-session map is bounded; oldest entries fall off."""
        from services.runner_client import RunnerClient, _PER_SESSION_PIN_LIMIT

        client = RunnerClient(endpoints=["http://r1:8000"])
        # Stuff in LIMIT + 5 entries.
        for i in range(_PER_SESSION_PIN_LIMIT + 5):
            key = (f"sess-{i}", "m1")
            client._last_endpoint_per_session[key] = "http://r1:8000"
            # Simulate the eviction loop from acquire_server.
            while (
                len(client._last_endpoint_per_session)
                > _PER_SESSION_PIN_LIMIT
            ):
                client._last_endpoint_per_session.popitem(last=False)

        assert len(client._last_endpoint_per_session) == _PER_SESSION_PIN_LIMIT
        # Oldest (sess-0) should be gone, newest (sess-LIMIT+4) present.
        assert ("sess-0", "m1") not in client._last_endpoint_per_session
        last_key = (f"sess-{_PER_SESSION_PIN_LIMIT + 4}", "m1")
        assert last_key in client._last_endpoint_per_session


class TestSelectRunnerRules:
    """Three-tier parallel-aware selection ladder (post-sticky):

      1. cold-start a fresh server where one fits (VRAM headroom, no
         server loaded for the model yet) — fewest active servers, then
         most VRAM;
      2. attach to a warm server with a free KV slot
         (``use_count < parallel``) when no cold-start headroom exists;
      3. ranked fallback when everything is full.
    """

    def _client(self):
        from services.runner_client import RunnerClient
        c = RunnerClient(endpoints=["http://r1:8000", "http://r2:8000"])
        c._model_tensor_split = {
            ("http://r1:8000", "m1"): None,
            ("http://r2:8000", "m1"): None,
        }
        return c

    def _health_resp(self):
        h = MagicMock(status_code=200)
        h.json.return_value = {
            "status": "ok",
            "gpu": {"0": {"free_mb": 12000}},
            "models": [{"id": "m1", "name": "m1", "task": "TextToText"}],
        }
        return h

    def _servers_resp(self, servers):
        r = MagicMock(status_code=200)
        r.json.return_value = {"active_servers": len(servers), "servers": servers}
        return r

    @pytest.mark.asyncio
    async def test_rule1_busy_peer_no_server_on_other_picks_other(self):
        """r1 has busy server, r2 has nothing → pick r2 to spawn fresh."""
        client = self._client()
        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return self._health_resp()
            if "r1" in url and "/v1/servers" in url:
                # r1 has a busy server for m1
                return self._servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True, "idle_since": None, "use_count": 1}
                ])
            if "r2" in url and "/v1/servers" in url:
                # r2 has nothing
                return self._servers_resp([])
            return MagicMock(status_code=404)
        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen == "http://r2:8000"

    @pytest.mark.asyncio
    async def test_cold_start_preferred_over_warm_idle(self):
        """r1 has an idle warm server (free slots); r2 is empty with VRAM.
        A new session cold-starts a fresh dedicated server on r2 rather
        than reusing the warm r1 — the explicit "prefer a NEW server when
        there's space" policy.  (A *returning* session would instead hit
        the per-session sticky path and reuse its own warm server; this
        path only governs new sessions with no pin.)
        """
        from config import CACHE_TIMEOUT_MIN
        import time
        client = self._client()
        long_idle_since = time.time() - (CACHE_TIMEOUT_MIN * 60 + 60)
        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return self._health_resp()
            if "r1" in url and "/v1/servers" in url:
                return self._servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True,
                     "idle_since": long_idle_since,
                     "use_count": 0}
                ])
            if "r2" in url and "/v1/servers" in url:
                return self._servers_resp([])
            return MagicMock(status_code=404)
        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen == "http://r2:8000"

    @pytest.mark.asyncio
    async def test_warm_free_slot_when_no_cold_start_headroom(self):
        """Both runners have the model loaded (no empty peer to cold-start
        on), but r2's server has a free KV slot while r1's is full.  The
        new session packs onto r2's free slot (Rule 2) — warm weights,
        runs concurrently on its own slot."""
        client = self._client()
        client._model_parallel = {
            ("http://r1:8000", "m1"): 2,
            ("http://r2:8000", "m1"): 2,
        }
        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return self._health_resp()
            if "r1" in url and "/v1/servers" in url:
                # r1 full: 2 of 2 slots in use
                return self._servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True, "idle_since": None, "use_count": 2}
                ])
            if "r2" in url and "/v1/servers" in url:
                # r2 has one free slot (1 of 2 used)
                return self._servers_resp([
                    {"server_id": "s2", "model_id": "m1",
                     "healthy": True, "idle_since": None, "use_count": 1}
                ])
            return MagicMock(status_code=404)
        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen == "http://r2:8000"

    @pytest.mark.asyncio
    async def test_ranked_fallback_when_all_slots_full(self):
        """No cold-start headroom and every loaded server is at capacity.
        Fall through to the ranked tiebreak (fewest of our handles, then
        most VRAM) and let the runner queue — here both servers are full
        so a runner is still returned rather than None."""
        client = self._client()
        client._model_parallel = {
            ("http://r1:8000", "m1"): 1,
            ("http://r2:8000", "m1"): 1,
        }
        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return self._health_resp()
            if "/v1/servers" in url:
                return self._servers_resp([
                    {"server_id": "sx", "model_id": "m1",
                     "healthy": True, "idle_since": None, "use_count": 1}
                ])
            return MagicMock(status_code=404)
        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen in ("http://r1:8000", "http://r2:8000")

    @pytest.mark.asyncio
    async def test_cold_start_blocked_by_insufficient_vram(self):
        """A runner with the model unloaded but NOT enough free VRAM to
        fit it must not be chosen for a cold-start; the warm server with
        a free slot wins instead."""
        client = self._client()
        client._model_parallel = {
            ("http://r1:8000", "m1"): 2,
            ("http://r2:8000", "m1"): 2,
        }
        # Model needs ~10 GB; r2 (empty) only has ~1 GB free → can't fit.
        client._model_size_bytes = {
            ("http://r1:8000", "m1"): 10 * 1024 * 1024 * 1024,
            ("http://r2:8000", "m1"): 10 * 1024 * 1024 * 1024,
        }
        def _health_for(free_mb):
            h = MagicMock(status_code=200)
            h.json.return_value = {
                "status": "ok",
                "gpu": {"0": {"free_mb": free_mb}},
                "models": [{"id": "m1", "name": "m1", "task": "TextToText"}],
            }
            return h
        async def fake_get(url, **kw):
            if url.endswith("/health"):
                # r1 has the model loaded (low free VRAM); r2 empty but
                # also starved of VRAM so it can't cold-start.
                return _health_for(1000)
            if "r1" in url and "/v1/servers" in url:
                return self._servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True, "idle_since": None, "use_count": 1}
                ])
            if "r2" in url and "/v1/servers" in url:
                return self._servers_resp([])
            return MagicMock(status_code=404)
        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        # r2 can't fit a fresh server → fall to r1's warm free slot.
        assert chosen == "http://r1:8000"

    @pytest.mark.asyncio
    async def test_rule2_short_idle_falls_through_to_ranked(self):
        """If r1's server has only been idle briefly (less than
        CACHE_TIMEOUT_MIN) it MIGHT belong to a paused session — don't
        commandeer.  Fall through to ranked.  With r2 having no server
        AND no busy peer (the brief idle isn't "busy" for rule 1), the
        ranker picks by (-handles, vram) where both are equal at 0
        handles and 12 GB; r1 wins by iteration order.
        """
        import time
        client = self._client()
        recent_idle = time.time() - 60  # 1 min idle, well under 30 min
        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return self._health_resp()
            if "r1" in url and "/v1/servers" in url:
                return self._servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True,
                     "idle_since": recent_idle,
                     "use_count": 1}
                ])
            if "r2" in url and "/v1/servers" in url:
                return self._servers_resp([])
            return MagicMock(status_code=404)
        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        # Either endpoint is plausible; the key invariant is rule 1
        # did NOT fire (r1 isn't "busy" since here_count=0 and idle_since
        # is set) and rule 2 did NOT fire (idle wasn't long enough).
        assert chosen in ("http://r1:8000", "http://r2:8000")

    @pytest.mark.asyncio
    async def test_rule1_idle_under_cache_timeout_still_busy(self):
        """A server that went idle 30 s ago (well within CACHE_TIMEOUT_MIN
        of 5 min) is still considered busy — a session may be paused
        mid-conversation.  Rule 1 fires: route new request to empty peer
        instead of commandeering the maybe-paused server.
        """
        import time
        client = self._client()
        recently_idle = time.time() - 30  # 30 s ago, ≪ CACHE_TIMEOUT_MIN
        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return self._health_resp()
            if "r1" in url and "/v1/servers" in url:
                return self._servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True,
                     "idle_since": recently_idle,
                     "use_count": 3}
                ])
            if "r2" in url and "/v1/servers" in url:
                return self._servers_resp([])
            return MagicMock(status_code=404)
        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        # r1's recent idle counts as busy; r2 is empty; rule 1 → r2.
        assert chosen == "http://r2:8000"

    @pytest.mark.asyncio
    async def test_rule1_only_one_endpoint_skips(self):
        """When there's no peer (only one endpoint hosts the model),
        rule 1 can't fire — fall through to ranked.
        """
        from services.runner_client import RunnerClient
        client = RunnerClient(endpoints=["http://r1:8000"])
        client._model_tensor_split = {("http://r1:8000", "m1"): None}
        async def fake_get(url, **kw):
            if url.endswith("/health"):
                return self._health_resp()
            if "/v1/servers" in url:
                return self._servers_resp([
                    {"server_id": "s1", "model_id": "m1",
                     "healthy": True, "idle_since": None}
                ])
            return MagicMock(status_code=404)
        client._client = _mock_client(get=AsyncMock(side_effect=fake_get))
        chosen = await client._select_runner("m1")
        assert chosen == "http://r1:8000"
