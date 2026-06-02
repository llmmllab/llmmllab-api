"""HTTP-layer tests for /v1/embeddings (OpenAI-compatible).

Patches runner_client so no real runner is required; verifies request
parsing, response mapping from the llama.cpp /v1/embeddings shape, and
input validation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.openai.embeddings import router


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _runner_patches(embedding_data):
    """Patch acquire_server / proxy_request / release_server on the
    runner_client singleton imported by the router."""
    handle = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(
        return_value={
            "object": "list",
            "data": embedding_data,
            "model": "nomic-embed-text-v2-moe",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
    )
    return (
        patch("routers.openai.embeddings.runner_client.acquire_server",
              new=AsyncMock(return_value=handle)),
        patch("routers.openai.embeddings.runner_client.proxy_request",
              new=AsyncMock(return_value=resp)),
        patch("routers.openai.embeddings.runner_client.release_server",
              new=AsyncMock(return_value=None)),
    )


def test_single_string_input_maps_response(client: TestClient):
    a, p, r = _runner_patches([
        {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
    ])
    with a, p, r:
        resp = client.post(
            "/embeddings",
            json={"model": "nomic-embed-text-v2-moe", "input": "hello world"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "list"
    assert body["model"] == "nomic-embed-text-v2-moe"
    assert len(body["data"]) == 1
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert body["data"][0]["index"] == 0
    assert body["data"][0]["object"] == "embedding"
    assert body["usage"]["prompt_tokens"] == 5


def test_list_input_preserves_order(client: TestClient):
    a, p, r = _runner_patches([
        {"object": "embedding", "embedding": [1.0], "index": 0},
        {"object": "embedding", "embedding": [2.0], "index": 1},
    ])
    with a, p, r:
        resp = client.post(
            "/embeddings",
            json={"model": "nomic-embed-text-v2-moe", "input": ["a", "b"]},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert [d["index"] for d in data] == [0, 1]
    assert [d["embedding"] for d in data] == [[1.0], [2.0]]


def test_empty_string_rejected(client: TestClient):
    resp = client.post(
        "/embeddings",
        json={"model": "nomic-embed-text-v2-moe", "input": ""},
    )
    assert resp.status_code == 400


def test_tokenized_int_input_rejected(client: TestClient):
    resp = client.post(
        "/embeddings",
        json={"model": "nomic-embed-text-v2-moe", "input": [1, 2, 3]},
    )
    assert resp.status_code == 400


def test_runner_error_maps_to_502(client: TestClient):
    handle = MagicMock()
    bad = MagicMock()
    bad.status_code = 500
    bad.text = "boom"
    with patch("routers.openai.embeddings.runner_client.acquire_server",
               new=AsyncMock(return_value=handle)), \
         patch("routers.openai.embeddings.runner_client.proxy_request",
               new=AsyncMock(return_value=bad)), \
         patch("routers.openai.embeddings.runner_client.release_server",
               new=AsyncMock(return_value=None)):
        resp = client.post(
            "/embeddings",
            json={"model": "nomic-embed-text-v2-moe", "input": "x"},
        )
    assert resp.status_code == 502
