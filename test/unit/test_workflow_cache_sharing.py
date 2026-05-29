"""
Workflow cache key contract.

2026-05-22 audit dropped session_id from the cache key on the
assumption that the per-request httpx event_hook for X-Session-ID
made cross-session workflow reuse safe. It IS safe for slot routing
but it breaks fan-out: the cached workflow's ChatOpenAI has the
runner base_url baked in at build time, so every session sharing
the cache lands on whichever runner the first session acquired.

Reinstated session_id in the cache key so each session gets its own
workflow + its own acquired server handle. The X-Session-ID
event_hook still applies per request — slot routing on a single
runner remains correct for sessions sharing a workflow only when
their handle happens to land on the same runner.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from utils.logging import _session_id_ctx


# The graph package has a pre-existing circular import path
# (graph.service ↔ services.completion_service ↔ composer_init) that
# only collapses cleanly once the whole graph is loaded.  We defer
# imports to runtime inside fixtures so this test file is collectible
# on its own.
def _composer_service():
    from graph.service import ComposerService  # noqa: WPS433
    return ComposerService


def _inject_hook():
    from graph.workflows.base import _inject_session_id_header  # noqa: WPS433
    return _inject_session_id_header


def test_cache_key_distinguishes_sessions():
    """``_build_cache_key`` includes session_id so each session
    triggers its own acquire_server (enables cross-session fan-out)."""
    ComposerService = _composer_service()
    svc = ComposerService(builder=MagicMock())
    a_token = _session_id_ctx.set("session-A")
    try:
        k1 = svc._build_cache_key(user_id="alice", model_name="m1")
    finally:
        _session_id_ctx.reset(a_token)
    b_token = _session_id_ctx.set("session-B")
    try:
        k2 = svc._build_cache_key(user_id="alice", model_name="m1")
    finally:
        _session_id_ctx.reset(b_token)
    assert k1 == "workflow_alice_session-A_m1"
    assert k2 == "workflow_alice_session-B_m1"
    assert k1 != k2


def test_cache_key_includes_model_name():
    ComposerService = _composer_service()
    svc = ComposerService(builder=MagicMock())
    assert svc._build_cache_key("alice", "m1") != svc._build_cache_key("alice", "m2")


def test_cache_key_omits_model_when_none():
    ComposerService = _composer_service()
    svc = ComposerService(builder=MagicMock())
    assert svc._build_cache_key("alice", None) == "workflow_alice"


@pytest.mark.asyncio
async def test_session_header_injected_from_contextvar():
    """The httpx event hook stamps X-Session-ID from the current task's contextvar."""
    request = httpx.Request("POST", "http://runner/v1/chat/completions")
    tok = _session_id_ctx.set("test-session-xyz")
    try:
        await _inject_hook()(request)
    finally:
        _session_id_ctx.reset(tok)
    assert request.headers["X-Session-ID"] == "test-session-xyz"


@pytest.mark.asyncio
async def test_session_header_omitted_when_no_session_context():
    request = httpx.Request("POST", "http://runner/v1/chat/completions")
    tok = _session_id_ctx.set(None)
    try:
        await _inject_hook()(request)
    finally:
        _session_id_ctx.reset(tok)
    assert "X-Session-ID" not in request.headers


@pytest.mark.asyncio
async def test_session_header_is_dynamic_per_request():
    """One client / one event hook serves many sessions correctly: each
    request picks up the session_id in its OWN asyncio task scope."""
    captured = []

    async def fake_handler(request):
        captured.append(request.headers.get("X-Session-ID"))
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(fake_handler)
    async with httpx.AsyncClient(
        transport=transport,
        event_hooks={"request": [_inject_hook()]},
    ) as client:

        async def fire(sid: str):
            tok = _session_id_ctx.set(sid)
            try:
                await client.post("http://runner/v1/chat/completions", json={})
            finally:
                _session_id_ctx.reset(tok)

        import asyncio

        await asyncio.gather(fire("S-alpha"), fire("S-beta"), fire("S-gamma"))

    # Each request carried its own session id, never leaked.
    assert sorted(captured) == ["S-alpha", "S-beta", "S-gamma"]


# Note: a full compose_workflow integration test would also pin
# "build called once across two sessions" — but mocking the user-config
# fetch + builder is non-trivial and would duplicate ComposerService
# internals.  The unit tests above are sufficient: cache key is
# session-independent, and the dynamic header hook is verified to
# stamp the correct id per asyncio task.  Together those two pin the
# user-visible behavior we care about.
