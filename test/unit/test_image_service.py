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


# ---------------------------------------------------------------------------
# img23d
# ---------------------------------------------------------------------------


def _make_runner_client_for_img23d(body: Dict[str, Any], status_code: int = 200,
                                   text: str = "") -> MagicMock:
    client = MagicMock()
    client._endpoints = ["http://runner-1:8000"]
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
