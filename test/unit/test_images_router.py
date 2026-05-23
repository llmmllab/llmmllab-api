"""HTTP-layer tests for /v1/images/generations and /v1/images/3d.

Exercises the OpenAI-compatible image router via FastAPI's TestClient.
The image_service is patched so no runner is required; we're only
verifying request parsing, response shape, and error mapping.
"""

import base64
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.openai.images import router
from services.image_service import GeneratedImage, ImageServiceError, ImageTo3DResult, TxtToImageResult


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# /images/generations  (OpenAI-compatible)
# ---------------------------------------------------------------------------


def test_generations_returns_b64_json(client: TestClient):
    fake_b64 = base64.b64encode(b"fake-png").decode("ascii")
    fake_result = TxtToImageResult(
        images=[GeneratedImage(b64_png=fake_b64)],
        created=1700000000,
        parameters={},
    )
    with patch(
        "routers.openai.images.generate_image", new=AsyncMock(return_value=fake_result)
    ):
        response = client.post(
            "/images/generations",
            json={"prompt": "a teacup", "model": "qwen-image", "size": "1024x1024"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["created"] == 1700000000
    assert len(body["data"]) == 1
    assert body["data"][0]["b64_json"] == fake_b64
    assert body["output_format"] == "png"


def test_generations_rejects_missing_model(client: TestClient):
    response = client.post("/images/generations", json={"prompt": "x"})
    assert response.status_code == 400
    assert "model" in response.json()["detail"].lower()


def test_generations_parses_size(client: TestClient):
    """Verify size 'WxH' is split into width/height passed to the service.

    OpenAI's CreateImageRequest pins ``size`` to a Literal of allowed
    values, so we use one of those for the wire test.
    """
    fake_result = TxtToImageResult(
        images=[GeneratedImage(b64_png="b64")],
        created=1, parameters={},
    )
    mock = AsyncMock(return_value=fake_result)
    with patch("routers.openai.images.generate_image", new=mock):
        response = client.post(
            "/images/generations",
            json={"prompt": "x", "model": "m", "size": "1024x1536"},
        )
    assert response.status_code == 200

    _, kwargs = mock.call_args
    assert kwargs["width"] == 1024
    assert kwargs["height"] == 1536


def test_generations_rejects_bad_size(client: TestClient):
    """Pydantic validation rejects non-Literal sizes with 422 before
    our handler is invoked."""
    response = client.post(
        "/images/generations",
        json={"prompt": "x", "model": "m", "size": "not-a-size"},
    )
    assert response.status_code == 422


def test_generations_upstream_failure_returns_502(client: TestClient):
    with patch(
        "routers.openai.images.generate_image",
        new=AsyncMock(side_effect=ImageServiceError("kaboom", status_code=500)),
    ):
        response = client.post(
            "/images/generations", json={"prompt": "x", "model": "m"}
        )

    assert response.status_code == 502
    assert "kaboom" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /images/3d
# ---------------------------------------------------------------------------


def test_image_to_3d_returns_paths(client: TestClient):
    fake = ImageTo3DResult(
        id="abc123",
        elapsed_sec=15.4,
        mesh_path="/data/sd-out/3d/abc123.glb",
        gaussian_path=None,
        preview_b64="cHJldmlldw==",
    )
    with patch("routers.openai.images.generate_3d", new=AsyncMock(return_value=fake)):
        response = client.post(
            "/images/3d",
            json={"image_b64": "aGVsbG8=", "formats": ["mesh"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "abc123"
    assert body["mesh_path"].endswith(".glb")
    assert body["gaussian_path"] is None
    assert body["preview_b64"] == "cHJldmlldw=="
    assert body["elapsed_sec"] == 15.4


def test_image_to_3d_503_when_pipeline_missing(client: TestClient):
    with patch(
        "routers.openai.images.generate_3d",
        new=AsyncMock(side_effect=ImageServiceError("TRELLIS missing", status_code=503)),
    ):
        response = client.post(
            "/images/3d", json={"image_b64": "aGVsbG8="}
        )
    assert response.status_code == 503
    assert "TRELLIS" in response.json()["detail"]


def test_image_to_3d_other_failures_are_502(client: TestClient):
    with patch(
        "routers.openai.images.generate_3d",
        new=AsyncMock(side_effect=ImageServiceError("upstream", status_code=500)),
    ):
        response = client.post(
            "/images/3d", json={"image_b64": "aGVsbG8="}
        )
    assert response.status_code == 502
