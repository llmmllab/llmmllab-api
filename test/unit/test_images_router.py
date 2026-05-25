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
from services.image_service import (
    GeneratedImage,
    ImageServiceError,
    ImageTo3DPartsResult,
    ImageTo3DResult,
    ImageToImageResult,
    RembgResult,
    TxtToImageResult,
)


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
# /images/edits  (img2img / instruction edit)
# ---------------------------------------------------------------------------


def test_edits_returns_b64_json(client: TestClient):
    fake_b64 = base64.b64encode(b"edited-png").decode("ascii")
    fake = ImageToImageResult(
        images=[GeneratedImage(b64_png=fake_b64)],
        created=1700000123, parameters={},
    )
    with patch(
        "routers.openai.images.edit_image", new=AsyncMock(return_value=fake)
    ):
        response = client.post(
            "/images/edits",
            json={
                "prompt": "make it autumn",
                "image": "aGVsbG8=",
                "model": "qwen-image-edit-2511",
                "denoising_strength": 0.75,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["data"][0]["b64_json"] == fake_b64
    assert body["created"] == 1700000123


def test_edits_passes_denoising_strength(client: TestClient):
    fake = ImageToImageResult(
        images=[GeneratedImage(b64_png="x")], created=1, parameters={},
    )
    mock = AsyncMock(return_value=fake)
    with patch("routers.openai.images.edit_image", new=mock):
        client.post(
            "/images/edits",
            json={
                "prompt": "p",
                "image": "aGVsbG8=",
                "model": "m",
                "denoising_strength": 0.5,
            },
        )

    _, kwargs = mock.call_args
    assert kwargs["denoising_strength"] == 0.5
    assert kwargs["image_b64"] == "aGVsbG8="
    assert kwargs["model_id"] == "m"


def test_edits_rejects_out_of_range_denoising(client: TestClient):
    response = client.post(
        "/images/edits",
        json={
            "prompt": "p",
            "image": "aGVsbG8=",
            "model": "m",
            "denoising_strength": 2.0,  # > 1.0
        },
    )
    assert response.status_code == 422


def test_edits_upstream_failure_returns_502(client: TestClient):
    with patch(
        "routers.openai.images.edit_image",
        new=AsyncMock(side_effect=ImageServiceError("boom", status_code=500)),
    ):
        response = client.post(
            "/images/edits",
            json={"prompt": "p", "image": "aGVsbG8=", "model": "m"},
        )
    assert response.status_code == 502


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
    # Download URL is derived from the basename of mesh_path so clients
    # don't need pod access to fetch the artefact.
    assert body["mesh_url"] == "/v1/images/3d/abc123.glb"
    assert body["gaussian_url"] is None
    assert body["preview_b64"] == "cHJldmlldw=="
    assert body["elapsed_sec"] == 15.4


def test_image_to_3d_returns_both_urls_when_both_formats_requested(client: TestClient):
    fake = ImageTo3DResult(
        id="abc123",
        elapsed_sec=20.0,
        mesh_path="/data/sd-out/3d/abc123.glb",
        gaussian_path="/data/sd-out/3d/abc123.ply",
        preview_b64=None,
    )
    with patch("routers.openai.images.generate_3d", new=AsyncMock(return_value=fake)):
        response = client.post(
            "/images/3d",
            json={"image_b64": "aGVsbG8=", "formats": ["mesh", "gaussian"]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["mesh_url"] == "/v1/images/3d/abc123.glb"
    assert body["gaussian_url"] == "/v1/images/3d/abc123.ply"


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


# ---------------------------------------------------------------------------
# GET /images/3d/{filename} — proxy download
# ---------------------------------------------------------------------------


def _make_stream_artifact_mock(content: bytes, media_type: str = "model/gltf-binary"):
    """Build an AsyncMock matching ``stream_3d_artifact``'s return shape."""

    async def _iter():
        yield content

    async def _stream_3d_artifact(filename, **kwargs):
        return media_type, _iter()

    return AsyncMock(side_effect=_stream_3d_artifact)


def test_download_3d_streams_glb_through(client: TestClient):
    with patch(
        "routers.openai.images.stream_3d_artifact",
        new=_make_stream_artifact_mock(b"glb-bytes"),
    ):
        response = client.get("/images/3d/abc123.glb")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("model/gltf-binary")
    assert "abc123.glb" in response.headers["content-disposition"]
    assert response.content == b"glb-bytes"


def test_download_3d_404_when_artifact_missing(client: TestClient):
    with patch(
        "routers.openai.images.stream_3d_artifact",
        new=AsyncMock(side_effect=ImageServiceError("not here", status_code=404)),
    ):
        response = client.get("/images/3d/missing.glb")
    assert response.status_code == 404


def test_download_3d_400_when_filename_rejected(client: TestClient):
    # Use a simple filename (no path separators after URL decode) so the
    # FastAPI router matches the route — the rejection comes from the
    # service layer's regex check, not from path matching.
    with patch(
        "routers.openai.images.stream_3d_artifact",
        new=AsyncMock(side_effect=ImageServiceError("bad name", status_code=400)),
    ):
        response = client.get("/images/3d/bad.exe")
    assert response.status_code == 400


def test_download_3d_503_when_no_runner(client: TestClient):
    with patch(
        "routers.openai.images.stream_3d_artifact",
        new=AsyncMock(side_effect=ImageServiceError("no runner", status_code=503)),
    ):
        response = client.get("/images/3d/abc.glb")
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# /images/3d/parts (Hunyuan3D-Part / XPart)
# ---------------------------------------------------------------------------


def _fake_parts_result(gen_id: str = "abc123") -> ImageTo3DPartsResult:
    return ImageTo3DPartsResult(
        id=gen_id,
        elapsed_sec=85.7,
        mesh_path=f"/data/sd-out/3d_parts/{gen_id}_decomposed.glb",
        exploded_path=f"/data/sd-out/3d_parts/{gen_id}_exploded.glb",
        bbox_path=f"/data/sd-out/3d_parts/{gen_id}_bbox.glb",
        gt_bbox_path=f"/data/sd-out/3d_parts/{gen_id}_gt_bbox.glb",
    )


def test_3d_parts_returns_all_four_urls(client: TestClient):
    with patch(
        "routers.openai.images.generate_3d_parts",
        new=AsyncMock(return_value=_fake_parts_result()),
    ):
        response = client.post(
            "/images/3d/parts",
            json={"mesh_b64": "Z2xi", "octree_resolution": 512},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "abc123"
    # All four artefact URLs are derived from the runner-side paths so
    # clients can fetch each via the /v1/images/3d/parts/{filename}
    # proxy without runner pod access.
    assert body["mesh_url"] == "/v1/images/3d/parts/abc123_decomposed.glb"
    assert body["exploded_url"] == "/v1/images/3d/parts/abc123_exploded.glb"
    assert body["bbox_url"] == "/v1/images/3d/parts/abc123_bbox.glb"
    assert body["gt_bbox_url"] == "/v1/images/3d/parts/abc123_gt_bbox.glb"
    assert body["elapsed_sec"] == 85.7


def test_3d_parts_forwards_optional_params(client: TestClient):
    captured = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        return _fake_parts_result()

    with patch("routers.openai.images.generate_3d_parts", new=AsyncMock(side_effect=_fake)):
        response = client.post(
            "/images/3d/parts",
            json={"mesh_b64": "Z2xi", "octree_resolution": 256, "seed": 99},
        )

    assert response.status_code == 200
    assert captured["mesh_b64"] == "Z2xi"
    assert captured["octree_resolution"] == 256
    assert captured["seed"] == 99


def test_3d_parts_503_when_pipeline_missing(client: TestClient):
    with patch(
        "routers.openai.images.generate_3d_parts",
        new=AsyncMock(side_effect=ImageServiceError("XPart missing", status_code=503)),
    ):
        response = client.post(
            "/images/3d/parts", json={"mesh_b64": "Z2xi"}
        )
    assert response.status_code == 503
    assert "XPart" in response.json()["detail"]


def test_3d_parts_other_failures_are_502(client: TestClient):
    with patch(
        "routers.openai.images.generate_3d_parts",
        new=AsyncMock(side_effect=ImageServiceError("upstream", status_code=500)),
    ):
        response = client.post(
            "/images/3d/parts", json={"mesh_b64": "Z2xi"}
        )
    assert response.status_code == 502


def test_download_3d_parts_streams_glb_through(client: TestClient):
    async def _iter():
        yield b"decomposed-bytes"

    async def _fake(filename, **kwargs):
        return "model/gltf-binary", _iter()

    with patch(
        "routers.openai.images.stream_3d_parts_artifact",
        new=AsyncMock(side_effect=_fake),
    ):
        response = client.get("/images/3d/parts/abc123_decomposed.glb")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("model/gltf-binary")
    assert response.content == b"decomposed-bytes"


def test_download_3d_parts_404(client: TestClient):
    with patch(
        "routers.openai.images.stream_3d_parts_artifact",
        new=AsyncMock(side_effect=ImageServiceError("nope", status_code=404)),
    ):
        response = client.get("/images/3d/parts/missing_decomposed.glb")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /images/remove-bg (briaai/RMBG-2.0)
# ---------------------------------------------------------------------------


def test_remove_bg_returns_mask_and_cutout(client: TestClient):
    fake = RembgResult(
        id="abc123",
        mask_b64="bWFzaw==",
        transparent_b64="Y3V0b3V0",
        cutout_url="/v1/images/remove-bg/abc123.png",
        width=1024,
        height=1024,
        elapsed_sec=1.7,
    )
    with patch(
        "routers.openai.images.remove_image_background",
        new=AsyncMock(return_value=fake),
    ):
        response = client.post(
            "/images/remove-bg",
            json={"image": "aGVsbG8="},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "abc123"
    assert body["mask_b64"] == "bWFzaw=="
    assert body["transparent_b64"] == "Y3V0b3V0"
    assert body["cutout_url"] == "/v1/images/remove-bg/abc123.png"
    assert body["width"] == 1024 and body["height"] == 1024


def test_remove_bg_passes_mask_only_flag(client: TestClient):
    fake = RembgResult(
        id="x", mask_b64="m", transparent_b64=None, cutout_url=None,
        width=512, height=512, elapsed_sec=0.5,
    )
    mock = AsyncMock(return_value=fake)
    with patch("routers.openai.images.remove_image_background", new=mock):
        client.post("/images/remove-bg", json={"image": "aGVsbG8=", "mask_only": True})

    _, kwargs = mock.call_args
    assert kwargs["mask_only"] is True


def test_remove_bg_upstream_failure_returns_502(client: TestClient):
    with patch(
        "routers.openai.images.remove_image_background",
        new=AsyncMock(side_effect=ImageServiceError("boom", status_code=500)),
    ):
        response = client.post("/images/remove-bg", json={"image": "aGVsbG8="})
    assert response.status_code == 502


def test_remove_bg_503_when_no_runner(client: TestClient):
    with patch(
        "routers.openai.images.remove_image_background",
        new=AsyncMock(side_effect=ImageServiceError("no runner", status_code=503)),
    ):
        response = client.post("/images/remove-bg", json={"image": "aGVsbG8="})
    assert response.status_code == 503


def test_download_rembg_streams_png(client: TestClient):
    async def _iter():
        yield b"png-bytes"

    async def _stream(filename, **_kwargs):
        return "image/png", _iter()

    with patch("routers.openai.images.stream_rembg_artifact", new=AsyncMock(side_effect=_stream)):
        response = client.get("/images/remove-bg/abc123.png")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == b"png-bytes"


def test_download_rembg_404_when_missing(client: TestClient):
    with patch(
        "routers.openai.images.stream_rembg_artifact",
        new=AsyncMock(side_effect=ImageServiceError("not here", status_code=404)),
    ):
        response = client.get("/images/remove-bg/missing.png")
    assert response.status_code == 404
