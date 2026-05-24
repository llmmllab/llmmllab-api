"""Tests for the runner-admin fan-out service.

Stubs ``RunnerClient`` so no real runner traffic happens — we only
verify the fan-out wiring (URL construction, per-endpoint aggregation,
search-by-id semantics, model filtering, error reporting).
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.runner_admin_service import (
    evict_all_runner_servers,
    evict_runner_server,
    list_all_runner_servers,
    list_runner_pipelines,
    unload_runner_pipeline,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_resp(status: int = 200, json_body=None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=json_body if json_body is not None else {})
    r.text = ""
    return r


def _make_client(endpoints: List[str], responses: Dict[str, Any]) -> MagicMock:
    """Stub client that returns ``responses[url]`` for each request URL."""
    client = MagicMock()
    client._endpoints = endpoints
    http = MagicMock()

    async def _request(method, url, **_kwargs):
        if url in responses:
            entry = responses[url]
            if isinstance(entry, Exception):
                raise entry
            return entry
        return _mock_resp(404, {})

    http.request = AsyncMock(side_effect=_request)
    client._get_client = MagicMock(return_value=http)
    return client


# ---------------------------------------------------------------------------
# list_all_runner_servers
# ---------------------------------------------------------------------------


def test_list_all_runner_servers_aggregates_across_endpoints():
    client = _make_client(
        endpoints=["http://r1:8000", "http://r2:8000"],
        responses={
            "http://r1:8000/v1/servers": _mock_resp(
                200, {"active_servers": 1, "servers": [{"server_id": "abc"}]}
            ),
            "http://r2:8000/v1/servers": _mock_resp(
                200, {"active_servers": 2, "servers": [
                    {"server_id": "def"}, {"server_id": "ghi"}
                ]}
            ),
        },
    )

    invs = _run(list_all_runner_servers(client=client))

    assert len(invs) == 2
    assert invs[0].endpoint == "http://r1:8000"
    assert invs[0].active_servers == 1
    assert invs[1].endpoint == "http://r2:8000"
    assert invs[1].active_servers == 2
    assert all(inv.error is None for inv in invs)


def test_list_all_runner_servers_marks_unreachable_endpoint():
    """One runner failing must not hide the others — surface the error
    on the failed entry and keep the rest."""
    client = _make_client(
        endpoints=["http://r1:8000", "http://r2:8000"],
        responses={
            "http://r1:8000/v1/servers": ConnectionError("refused"),
            "http://r2:8000/v1/servers": _mock_resp(
                200, {"active_servers": 0, "servers": []}
            ),
        },
    )

    invs = _run(list_all_runner_servers(client=client))

    assert invs[0].error == "unreachable"
    assert invs[1].error is None


# ---------------------------------------------------------------------------
# evict_runner_server
# ---------------------------------------------------------------------------


def test_evict_runner_server_finds_and_posts_to_the_right_endpoint():
    client = _make_client(
        endpoints=["http://r1:8000", "http://r2:8000"],
        responses={
            "http://r1:8000/v1/servers": _mock_resp(200, {"servers": []}),
            "http://r2:8000/v1/servers": _mock_resp(
                200, {"servers": [{"server_id": "target123"}]}
            ),
            "http://r2:8000/v1/server/target123/evict": _mock_resp(
                200, {"status": "evicted"}
            ),
        },
    )

    result = _run(evict_runner_server("target123", client=client))

    assert result.succeeded is True
    assert result.endpoint == "http://r2:8000"
    assert result.server_id == "target123"


def test_evict_runner_server_returns_not_found_when_id_unknown():
    client = _make_client(
        endpoints=["http://r1:8000"],
        responses={"http://r1:8000/v1/servers": _mock_resp(200, {"servers": []})},
    )

    result = _run(evict_runner_server("nonexistent", client=client))

    assert result.succeeded is False
    assert result.endpoint is None
    assert "not found" in result.detail.lower()


def test_evict_runner_server_surfaces_runner_failure():
    client = _make_client(
        endpoints=["http://r1:8000"],
        responses={
            "http://r1:8000/v1/servers": _mock_resp(
                200, {"servers": [{"server_id": "abc"}]}
            ),
            "http://r1:8000/v1/server/abc/evict": _mock_resp(500, {}),
        },
    )

    result = _run(evict_runner_server("abc", client=client))

    assert result.succeeded is False
    assert result.endpoint == "http://r1:8000"


# ---------------------------------------------------------------------------
# evict_all_runner_servers
# ---------------------------------------------------------------------------


def test_evict_all_fans_out_every_runner_every_server():
    client = _make_client(
        endpoints=["http://r1:8000", "http://r2:8000"],
        responses={
            "http://r1:8000/v1/servers": _mock_resp(
                200,
                {"servers": [
                    {"server_id": "a1", "model_id": "qwen-llm"},
                    {"server_id": "a2", "model_id": "qwen-image"},
                ]},
            ),
            "http://r2:8000/v1/servers": _mock_resp(
                200, {"servers": [{"server_id": "b1", "model_id": "qwen-llm"}]}
            ),
            "http://r1:8000/v1/server/a1/evict": _mock_resp(200, {"status": "evicted"}),
            "http://r1:8000/v1/server/a2/evict": _mock_resp(200, {"status": "evicted"}),
            "http://r2:8000/v1/server/b1/evict": _mock_resp(200, {"status": "evicted"}),
        },
    )

    results = _run(evict_all_runner_servers(client=client))

    ids = {r.server_id for r in results}
    assert ids == {"a1", "a2", "b1"}
    assert all(r.succeeded for r in results)


def test_evict_all_filters_by_model():
    client = _make_client(
        endpoints=["http://r1:8000"],
        responses={
            "http://r1:8000/v1/servers": _mock_resp(
                200,
                {"servers": [
                    {"server_id": "a1", "model_id": "qwen-llm"},
                    {"server_id": "a2", "model_id": "qwen-image"},
                ]},
            ),
            "http://r1:8000/v1/server/a2/evict": _mock_resp(200, {"status": "evicted"}),
        },
    )

    results = _run(evict_all_runner_servers(model_id="qwen-image", client=client))

    assert len(results) == 1
    assert results[0].server_id == "a2"


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def test_list_runner_pipelines_aggregates_per_runner():
    client = _make_client(
        endpoints=["http://r1:8000", "http://r2:8000"],
        responses={
            "http://r1:8000/v1/pipelines": _mock_resp(
                200,
                {"pipelines": [
                    {"name": "img23d", "task": "ImageTo3D", "loaded": True}
                ]},
            ),
            "http://r2:8000/v1/pipelines": _mock_resp(
                200,
                {"pipelines": [
                    {"name": "img23d", "task": "ImageTo3D", "loaded": False}
                ]},
            ),
        },
    )

    entries = _run(list_runner_pipelines(client=client))

    assert len(entries) == 2
    by_endpoint = {e.endpoint: e for e in entries}
    assert by_endpoint["http://r1:8000"].loaded is True
    assert by_endpoint["http://r2:8000"].loaded is False


def test_unload_runner_pipeline_only_targets_loaded_runners_by_default():
    """only_loaded=True (the default) should skip runners where the
    pipeline is already unloaded — they don't need the round-trip."""
    client = _make_client(
        endpoints=["http://r1:8000", "http://r2:8000"],
        responses={
            "http://r1:8000/v1/pipelines": _mock_resp(
                200, {"pipelines": [{"name": "img23d", "loaded": True}]}
            ),
            "http://r2:8000/v1/pipelines": _mock_resp(
                200, {"pipelines": [{"name": "img23d", "loaded": False}]}
            ),
            "http://r1:8000/v1/pipelines/img23d/unload": _mock_resp(
                200, {"name": "img23d", "loaded": False}
            ),
        },
    )

    results = _run(unload_runner_pipeline("img23d", client=client))

    actions = {r["endpoint"]: r for r in results}
    assert actions["http://r1:8000"]["succeeded"] is True
    assert actions["http://r2:8000"]["skipped"] is True
