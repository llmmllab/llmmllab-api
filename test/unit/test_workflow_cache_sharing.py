"""
Pins the cache-sharing fix from 2026-05-22 audit.

Before:
  Workflow cache key = (user_id, model_name, session_id).
  Each session triggered a fresh acquire_server + ChatOpenAI build —
  N concurrent sessions for one model => N workflow builds.

After:
  Workflow cache key = (user_id, model_name).
  X-Session-ID is set per-request by an httpx event_hook reading
  ``_session_id_ctx`` at request time, so a single cached workflow
  safely serves multiple sessions without contaminating slot LRU on
  the runner.
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


def test_cache_key_is_session_independent():
    """``_build_cache_key`` ignores the current session context."""
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
    assert k1 == k2 == "workflow_alice_m1"


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
