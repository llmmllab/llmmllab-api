"""Tests for the image generation service layer.

These verify the API → runner glue: we mock out the ``RunnerClient`` so
no network or runner process is required, and confirm the service
correctly translates between the OpenAI-shaped wire protocol and the
runner's native ``/sdapi/v1/txt2img`` + ``/v1/pipelines/img23d/run``
endpoints.
"""

import asyncio
import base64
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.image_service import (
    ImageServiceError,
    edit_image,
    generate_3d,
    generate_image,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_response(status_code: int = 200, json_body: Dict[str, Any] | None = None,
                   text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text or ""
    response.json = MagicMock(return_value=json_body or {})
    return response


def _make_runner_client_for_txt2img(body: Dict[str, Any]) -> MagicMock:
    """Build a RunnerClient stub that returns *body* from proxy_request."""
    client = MagicMock()
    client.acquire_server = AsyncMock(
        return_value=MagicMock(server_id="srv-1", base_url="http://x", runner_host="http://x")
    )
    client.release_server = AsyncMock(return_value=None)
    client.proxy_request = AsyncMock(return_value=_mock_response(200, body))
    return client


def test_generate_image_returns_b64_and_releases_server():
    """Happy path: runner responds with one base64 image; we return it
    unchanged AND release the server even on success."""
    fake_b64 = base64.b64encode(b"fake-png-bytes").decode("ascii")
    client = _make_runner_client_for_txt2img(
        {"images": [fake_b64], "parameters": {"prompt": "a cat"}}
    )

    result = _run(generate_image(
        prompt="a cat", model_id="qwen-image", client=client,
    ))

    assert len(result.images) == 1
    assert result.images[0].b64_png == fake_b64
    assert result.parameters == {"prompt": "a cat"}

    # Server must be acquired with the right model and released regardless.
    client.acquire_server.assert_awaited_once_with(model_id="qwen-image")
    client.release_server.assert_awaited_once()


def test_generate_image_translates_params_into_sdapi_payload():
    """The kwargs we accept must show up in the body sent to the runner."""
    client = _make_runner_client_for_txt2img(
        {"images": ["b64"], "parameters": {}}
    )

    _run(generate_image(
        prompt="P", model_id="m", negative_prompt="bad",
        width=512, height=768, steps=20, cfg_scale=4.0,
        sampler_name="dpm++_2m", seed=123, batch_size=2,
        client=client,
    ))

    _, kwargs = client.proxy_request.call_args
    payload = kwargs["json"]
    assert payload["prompt"] == "P"
    assert payload["negative_prompt"] == "bad"
    assert payload["width"] == 512
    assert payload["height"] == 768
    assert payload["steps"] == 20
    assert payload["cfg_scale"] == 4.0
    assert payload["sampler_name"] == "dpm++_2m"
    assert payload["seed"] == 123
    assert payload["batch_size"] == 2
    assert kwargs["path"] == "sdapi/v1/txt2img"
    assert kwargs["method"] == "POST"


def test_generate_image_raises_image_service_error_on_non_200():
    client = _make_runner_client_for_txt2img({})
    client.proxy_request = AsyncMock(
        return_value=_mock_response(500, text="upstream went boom")
    )

    with pytest.raises(ImageServiceError) as exc:
        _run(generate_image(prompt="p", model_id="m", client=client))

    assert exc.value.status_code == 500
    assert "upstream went boom" in str(exc.value)
    # Server still released after failure.
    client.release_server.assert_awaited_once()


def test_generate_image_rejects_empty_response_payload():
    client = _make_runner_client_for_txt2img({"images": []})

    with pytest.raises(ImageServiceError) as exc:
        _run(generate_image(prompt="p", model_id="m", client=client))

    assert "no images" in str(exc.value).lower()


def test_generate_image_resolves_defaults_from_model_parameters():
    """When the caller omits steps/cfg/etc., values from the model's
    YAML ``parameters`` block must flow into the request body."""
    client = _make_runner_client_for_txt2img({"images": ["b64"], "parameters": {}})

    # Stub list_models so _resolve_sd_defaults reads model-level defaults.
    fake_model = MagicMock()
    fake_model.id = "qwen-image-2512"
    fake_model.parameters = MagicMock(
        steps=40,
        cfg_scale=2.5,
        sampler_name="euler",
        width=1024,
        height=1024,
        denoising_strength=None,
    )
    client.list_models = AsyncMock(return_value=[fake_model])

    _run(generate_image(
        prompt="hi", model_id="qwen-image-2512", client=client,
    ))

    _, kwargs = client.proxy_request.call_args
    payload = kwargs["json"]
    assert payload["steps"] == 40
    assert payload["cfg_scale"] == 2.5
    assert payload["sampler_name"] == "euler"
    assert payload["width"] == 1024
    assert payload["height"] == 1024


def test_generate_image_caller_kwargs_win_over_model_defaults():
    """Explicit kwargs from the router must override model-level defaults."""
    client = _make_runner_client_for_txt2img({"images": ["b64"], "parameters": {}})

    fake_model = MagicMock()
    fake_model.id = "qwen-image-2512"
    fake_model.parameters = MagicMock(steps=40, cfg_scale=2.5, sampler_name="euler", width=1024, height=1024, denoising_strength=None)
    client.list_models = AsyncMock(return_value=[fake_model])

    _run(generate_image(
        prompt="hi", model_id="qwen-image-2512",
        steps=20, cfg_scale=8.0,  # caller overrides
        client=client,
    ))

    _, kwargs = client.proxy_request.call_args
    payload = kwargs["json"]
    assert payload["steps"] == 20  # caller wins
    assert payload["cfg_scale"] == 8.0  # caller wins
    assert payload["sampler_name"] == "euler"  # model default flows through


# ---------------------------------------------------------------------------
# img2img
# ---------------------------------------------------------------------------


def test_edit_image_targets_sdapi_img2img_with_ref_images():
    fake_b64 = base64.b64encode(b"edited-png").decode("ascii")
    client = _make_runner_client_for_txt2img(
        {"images": [fake_b64], "parameters": {}}
    )

    result = _run(edit_image(
        prompt="make it autumn",
        image_b64="aGVsbG8=",
        model_id="qwen-image-edit-2511",
        denoising_strength=0.8,
        client=client,
    ))

    assert result.images[0].b64_png == fake_b64

    _, kwargs = client.proxy_request.call_args
    assert kwargs["path"] == "sdapi/v1/img2img"
    payload = kwargs["json"]
    assert payload["prompt"] == "make it autumn"
    # Source image goes in BOTH fields: ``init_images`` for legacy
    # img2img models (noise+denoise path) and ``extra_images`` so it
    # populates sd-server's ``ref_images`` and triggers the
    # QwenImageEditPlusPipeline on Qwen-Image-Edit-2511.
    assert payload["init_images"] == ["aGVsbG8="]
    assert payload["extra_images"] == ["aGVsbG8="]
    assert payload["denoising_strength"] == 0.8


def test_edit_image_releases_server_on_failure():
    client = _make_runner_client_for_txt2img({})
    client.proxy_request = AsyncMock(
        return_value=_mock_response(500, text="boom")
    )

    with pytest.raises(ImageServiceError) as exc:
        _run(edit_image(
            prompt="x", image_b64="aGVsbG8=",
            model_id="qwen-image-edit-2511", client=client,
        ))
    assert exc.value.status_code == 500
    client.release_server.assert_awaited_once()


# ---------------------------------------------------------------------------
# img23d
# ---------------------------------------------------------------------------


def _make_runner_client_for_img23d(body: Dict[str, Any], status_code: int = 200,
                                   text: str = "") -> MagicMock:
    client = MagicMock()
    client._endpoints = ["http://runner-1:8000"]
    # New: pipeline routing is delegated to ``select_pipeline_endpoint``.
    # Tests mock it as an async returning the same endpoint the http
    # mock will respond on, so the call-graph reaches the existing
    # http_client.post mock unchanged.
    client.select_pipeline_endpoint = AsyncMock(return_value="http://runner-1:8000")
    http_client = MagicMock()
    http_client.post = AsyncMock(return_value=_mock_response(status_code, body, text=text))
    client._get_client = MagicMock(return_value=http_client)
    return client


def test_generate_3d_posts_to_pipeline_endpoint():
    client = _make_runner_client_for_img23d({
        "id": "abc123",
        "elapsed_sec": 12.5,
        "mesh_path": "/data/sd-out/3d/abc123.glb",
    })

    result = _run(generate_3d(image_b64="aGVsbG8=", client=client))

    assert result.id == "abc123"
    assert result.elapsed_sec == 12.5
    assert result.mesh_path.endswith(".glb")
    assert result.gaussian_path is None

    http_client = client._get_client.return_value
    args, kwargs = http_client.post.call_args
    assert args[0] == "http://runner-1:8000/v1/pipelines/img23d/run"
    payload = kwargs["json"]
    assert payload["image_b64"] == "aGVsbG8="
    assert payload["formats"] == ["mesh"]


def test_generate_3d_forwards_custom_formats_and_params():
    client = _make_runner_client_for_img23d({"id": "x", "elapsed_sec": 0.1})

    _run(generate_3d(
        image_b64="aGVsbG8=",
        formats=["mesh", "gaussian"],
        seed=99, ss_steps=20, slat_steps=18,
        ss_cfg_strength=8.0, slat_cfg_strength=4.0,
        client=client,
    ))

    http_client = client._get_client.return_value
    _, kwargs = http_client.post.call_args
    payload = kwargs["json"]
    assert payload["formats"] == ["mesh", "gaussian"]
    assert payload["seed"] == 99
    assert payload["ss_steps"] == 20
    assert payload["slat_steps"] == 18
    assert payload["ss_cfg_strength"] == 8.0
    assert payload["slat_cfg_strength"] == 4.0


def test_generate_3d_propagates_runner_failure():
    client = _make_runner_client_for_img23d(
        {}, status_code=503, text="TRELLIS is not installed"
    )

    with pytest.raises(ImageServiceError) as exc:
        _run(generate_3d(image_b64="aGVsbG8=", client=client))

    assert exc.value.status_code == 503
    assert "TRELLIS" in str(exc.value)


def test_generate_3d_raises_when_no_runner_configured():
    client = MagicMock()
    client._endpoints = []

    with pytest.raises(ImageServiceError) as exc:
        _run(generate_3d(image_b64="aGVsbG8=", client=client))

    assert exc.value.status_code == 503
    assert "No runner" in str(exc.value)


def test_generate_3d_result_records_runner_endpoint():
    """The runner that ran the generation must be remembered so the
    later download streams from the same pod (the file only exists
    there)."""
    client = _make_runner_client_for_img23d({"id": "abc", "elapsed_sec": 0.1})

    result = _run(generate_3d(image_b64="aGVsbG8=", client=client))

    assert result.runner_endpoint == "http://runner-1:8000"


# ---------------------------------------------------------------------------
# stream_3d_artifact
# ---------------------------------------------------------------------------


class _AsyncCM:
    """Tiny async-context-manager wrapper around an existing mock response."""
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *exc):
        return False


def _make_runner_client_for_stream(
    get_status: int = 200,
    chunks=(b"chunk-1", b"chunk-2"),
):
    """Build a RunnerClient stub for ``stream_3d_artifact``.

    The runner doesn't register a HEAD route on
    ``/v1/pipelines/img23d/files/{filename}``, so the service makes
    a streaming GET only.  ``get_status`` controls the response code
    seen on that GET.
    """
    from services.image_service import stream_3d_artifact  # noqa: F401

    client = MagicMock()
    client._endpoints = ["http://runner-1:8000"]
    client.select_pipeline_endpoint = AsyncMock(return_value="http://runner-1:8000")
    http = MagicMock()

    stream_resp = MagicMock()
    stream_resp.status_code = get_status

    async def _aiter_bytes():
        for c in chunks:
            yield c

    stream_resp.aiter_bytes = MagicMock(side_effect=_aiter_bytes)
    http.stream = MagicMock(return_value=_AsyncCM(stream_resp))
    client._get_client = MagicMock(return_value=http)
    return client


def test_stream_3d_artifact_streams_glb_with_correct_media_type():
    from services.image_service import stream_3d_artifact

    client = _make_runner_client_for_stream(chunks=(b"glb-bytes",))

    async def _drain():
        mt, body = await stream_3d_artifact("abc123.glb", client=client)
        out = b""
        async for chunk in body:
            out += chunk
        return mt, out

    media_type, body = _run(_drain())
    assert media_type == "model/gltf-binary"
    assert body == b"glb-bytes"


def test_stream_3d_artifact_rejects_invalid_filename():
    from services.image_service import stream_3d_artifact

    client = _make_runner_client_for_stream()

    for bad in ["../etc/passwd", "abc.exe", "abc/def.glb", "no-extension"]:
        with pytest.raises(ImageServiceError) as exc:
            _run(stream_3d_artifact(bad, client=client))
        assert exc.value.status_code == 400, f"expected 400 for {bad!r}"


def test_stream_3d_artifact_404_when_runner_says_not_found():
    from services.image_service import stream_3d_artifact

    client = _make_runner_client_for_stream(get_status=404)

    with pytest.raises(ImageServiceError) as exc:
        _run(stream_3d_artifact("abc123.glb", client=client))

    assert exc.value.status_code == 404


def test_stream_3d_artifact_raises_when_no_runner_configured():
    from services.image_service import stream_3d_artifact

    client = MagicMock()
    client._endpoints = []

    with pytest.raises(ImageServiceError) as exc:
        _run(stream_3d_artifact("abc123.glb", client=client))

    assert exc.value.status_code == 503


def test_stream_3d_artifact_targets_pipeline_files_endpoint():
    from services.image_service import stream_3d_artifact

    client = _make_runner_client_for_stream()

    async def _drain():
        _, body = await stream_3d_artifact("abc.glb", client=client)
        async for _chunk in body:
            pass

    _run(_drain())

    http = client._get_client.return_value
    stream_args, _ = http.stream.call_args
    assert stream_args == ("GET", "http://runner-1:8000/v1/pipelines/img23d/files/abc.glb")


# ---------------------------------------------------------------------------
# generate_3d_parts (Hunyuan3D-Part / XPart)
# ---------------------------------------------------------------------------


def _make_runner_client_for_img23d_part(
    body: Dict[str, Any], status_code: int = 200, text: str = ""
) -> MagicMock:
    client = MagicMock()
    client._endpoints = ["http://runner-1:8000"]
    client.select_pipeline_endpoint = AsyncMock(return_value="http://runner-1:8000")
    http_client = MagicMock()
    http_client.post = AsyncMock(
        return_value=_mock_response(status_code, body, text=text)
    )
    client._get_client = MagicMock(return_value=http_client)
    return client


def test_generate_3d_parts_posts_to_pipeline_endpoint():
    from services.image_service import generate_3d_parts

    client = _make_runner_client_for_img23d_part({
        "id": "abc123",
        "elapsed_sec": 42.5,
        "mesh_path": "/data/sd-out/3d_parts/abc123_decomposed.glb",
        "exploded_path": "/data/sd-out/3d_parts/abc123_exploded.glb",
        "bbox_path": "/data/sd-out/3d_parts/abc123_bbox.glb",
        "gt_bbox_path": "/data/sd-out/3d_parts/abc123_gt_bbox.glb",
    })

    result = _run(generate_3d_parts(mesh_b64="Z2xi", client=client))

    assert result.id == "abc123"
    assert result.elapsed_sec == 42.5
    assert result.mesh_path.endswith("_decomposed.glb")
    assert result.exploded_path.endswith("_exploded.glb")
    assert result.bbox_path.endswith("_bbox.glb")
    assert result.gt_bbox_path.endswith("_gt_bbox.glb")
    assert result.runner_endpoint == "http://runner-1:8000"

    http = client._get_client.return_value
    args, kwargs = http.post.call_args
    assert args[0] == "http://runner-1:8000/v1/pipelines/img23d_part/run"
    assert kwargs["json"]["mesh_b64"] == "Z2xi"
    # Optional params not sent when not provided.
    assert "octree_resolution" not in kwargs["json"]
    assert "seed" not in kwargs["json"]


def test_generate_3d_parts_forwards_optional_params():
    from services.image_service import generate_3d_parts

    client = _make_runner_client_for_img23d_part({"id": "x", "elapsed_sec": 0.1})
    _run(generate_3d_parts(
        mesh_b64="Z2xi", octree_resolution=256, seed=99, client=client,
    ))

    http = client._get_client.return_value
    _, kwargs = http.post.call_args
    assert kwargs["json"]["octree_resolution"] == 256
    assert kwargs["json"]["seed"] == 99


def test_generate_3d_parts_propagates_runner_failure():
    from services.image_service import generate_3d_parts

    client = _make_runner_client_for_img23d_part(
        {}, status_code=503, text="XPart not installed"
    )

    with pytest.raises(ImageServiceError) as exc:
        _run(generate_3d_parts(mesh_b64="Z2xi", client=client))

    assert exc.value.status_code == 503
    assert "XPart" in str(exc.value)


def test_generate_3d_parts_raises_when_no_runner_configured():
    from services.image_service import generate_3d_parts

    client = MagicMock()
    client._endpoints = []

    with pytest.raises(ImageServiceError) as exc:
        _run(generate_3d_parts(mesh_b64="Z2xi", client=client))

    assert exc.value.status_code == 503


def test_stream_3d_parts_artifact_streams_glb():
    from services.image_service import stream_3d_parts_artifact

    client = _make_runner_client_for_stream(chunks=(b"glb-bytes",))

    async def _drain():
        mt, body = await stream_3d_parts_artifact(
            "abc123_decomposed.glb", client=client
        )
        out = b""
        async for chunk in body:
            out += chunk
        return mt, out

    media_type, body = _run(_drain())
    assert media_type == "model/gltf-binary"
    assert body == b"glb-bytes"


def test_stream_3d_parts_artifact_rejects_invalid_filename():
    from services.image_service import stream_3d_parts_artifact

    client = _make_runner_client_for_stream()

    for bad in [
        "../etc/passwd",
        # Missing role suffix
        "abc123.glb",
        # Unknown role
        "abc123_other.glb",
        # Wrong extension
        "abc_decomposed.png",
        # Path separator
        "abc/def_decomposed.glb",
    ]:
        with pytest.raises(ImageServiceError) as exc:
            _run(stream_3d_parts_artifact(bad, client=client))
        assert exc.value.status_code == 400, f"expected 400 for {bad!r}"


def test_stream_3d_parts_artifact_targets_pipeline_files_endpoint():
    from services.image_service import stream_3d_parts_artifact

    client = _make_runner_client_for_stream()

    async def _drain():
        _, body = await stream_3d_parts_artifact(
            "abc_decomposed.glb", client=client
        )
        async for _ in body:
            pass

    _run(_drain())

    http = client._get_client.return_value
    stream_args, _ = http.stream.call_args
    assert stream_args == (
        "GET",
        "http://runner-1:8000/v1/pipelines/img23d_part/files/abc_decomposed.glb",
    )


# ---------------------------------------------------------------------------
# remove_image_background
# ---------------------------------------------------------------------------


def _make_runner_client_for_rembg(body):
    client = MagicMock()
    client._endpoints = ["http://runner-1:8000"]
    # Pipeline routing now delegates to ``select_pipeline_endpoint`` —
    # mock it as an async returning the same endpoint the http_client
    # mock will respond on.
    client.select_pipeline_endpoint = AsyncMock(return_value="http://runner-1:8000")
    http = MagicMock()
    http.post = AsyncMock(return_value=_mock_response(200, body))
    client._get_client = MagicMock(return_value=http)
    return client


def test_remove_bg_posts_to_pipeline_endpoint_and_unwraps_response():
    from services.image_service import remove_image_background

    fake_mask = "bWFzaw=="
    fake_cutout = "Y3V0b3V0"
    client = _make_runner_client_for_rembg({
        "id": "abc123",
        "mask_b64": fake_mask,
        "transparent_b64": fake_cutout,
        "cutout_path": "/data/sd-out/rembg/abc123.png",
        "width": 1024,
        "height": 1024,
        "elapsed_sec": 1.7,
    })

    result = _run(remove_image_background(image_b64="aGVsbG8=", client=client))

    assert result.id == "abc123"
    assert result.mask_b64 == fake_mask
    assert result.transparent_b64 == fake_cutout
    assert result.cutout_url == "/v1/images/remove-bg/abc123.png"
    assert result.width == 1024 and result.height == 1024

    http = client._get_client.return_value
    args, kwargs = http.post.call_args
    assert args[0] == "http://runner-1:8000/v1/pipelines/rembg/run"
    assert kwargs["json"]["image_b64"] == "aGVsbG8="
    assert kwargs["json"]["mask_only"] is False


def test_remove_bg_propagates_mask_only_and_size():
    from services.image_service import remove_image_background

    client = _make_runner_client_for_rembg({
        "id": "x", "mask_b64": "m", "transparent_b64": None,
        "width": 512, "height": 512, "elapsed_sec": 0.5,
    })

    _run(remove_image_background(
        image_b64="aGVsbG8=", mask_only=True, size=768, client=client,
    ))

    _, kwargs = client._get_client.return_value.post.call_args
    assert kwargs["json"]["mask_only"] is True
    assert kwargs["json"]["size"] == 768


def test_remove_bg_503_when_no_runner():
    from services.image_service import remove_image_background

    client = MagicMock()
    client._endpoints = []
    with pytest.raises(ImageServiceError) as exc:
        _run(remove_image_background(image_b64="aGVsbG8=", client=client))
    assert exc.value.status_code == 503


def test_remove_bg_surfaces_runner_failure():
    from services.image_service import remove_image_background

    client = MagicMock()
    client._endpoints = ["http://r1:8000"]
    client.select_pipeline_endpoint = AsyncMock(return_value="http://r1:8000")
    http = MagicMock()
    http.post = AsyncMock(return_value=_mock_response(500, text="boom"))
    client._get_client = MagicMock(return_value=http)

    with pytest.raises(ImageServiceError) as exc:
        _run(remove_image_background(image_b64="aGVsbG8=", client=client))
    assert exc.value.status_code == 500


def test_stream_rembg_artifact_streams_png():
    from services.image_service import stream_rembg_artifact

    client = _make_runner_client_for_stream(chunks=(b"png-bytes",))

    async def _drain():
        mt, body = await stream_rembg_artifact("abc123.png", client=client)
        out = b""
        async for chunk in body:
            out += chunk
        return mt, out

    media_type, content = _run(_drain())
    assert media_type == "image/png"
    assert content == b"png-bytes"


def test_stream_rembg_artifact_rejects_path_separators():
    from services.image_service import stream_rembg_artifact

    client = _make_runner_client_for_stream()
    for bad in ["../etc/passwd", "abc/def.png", "abc\\def.png", ""]:
        with pytest.raises(ImageServiceError) as exc:
            _run(stream_rembg_artifact(bad, client=client))
        assert exc.value.status_code == 400, f"expected 400 for {bad!r}"
