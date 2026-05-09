"""
Unit tests for the root chat/conversation router endpoints.

Tests the thin router layer in routers/chat.py and routers/conversation.py
with mocked service dependencies, verifying request validation, auth checks,
and proper delegation to services.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_app():
    """Build a minimal FastAPI app with the chat router mounted.

    Returns (app, chat_module) so that tests can patch functions on the
    actual module instance whose globals the endpoint closures reference.
    """
    app = FastAPI()

    # Patch the heavy imports before the router is loaded.
    with (
        patch.dict("sys.modules", {
            "graph": MagicMock(),
            "graph.state": MagicMock(),
            "graph.workflows": MagicMock(),
            "graph.workflows.factory": MagicMock(),
            "services": MagicMock(),
            "services.message_service": MagicMock(),
            "services.conversation_service": MagicMock(),
            "services.CompletionService": MagicMock(),
            "composer_init": MagicMock(),
            "utils.message_transformation": MagicMock(),
        }),
    ):
        import routers.chat  # noqa: F811
        app.include_router(routers.chat.router)
        chat_module = routers.chat
    return app, chat_module


def _make_client(**auth_overrides):
    """Build a TestClient with auth patched on the actual module instance.

    Because _build_app() imports routers.chat inside a sys.modules patch
    context, the module is orphaned from sys.modules on exit.  Patching
    "routers.chat.get_user_id" would re-import a *different* module
    instance.  Instead we patch the live module reference returned by
    _build_app() so the endpoint closures see the patched functions.
    """
    app, chat_mod = _build_app()

    # Patch auth helpers on the actual module instance.
    # These are functions called as get_user_id(request), so we need
    # to replace them with callables that return the desired value.
    if "get_user_id" in auth_overrides:
        val = auth_overrides["get_user_id"]
        chat_mod.get_user_id = lambda request, _v=val: _v
    if "get_request_id" in auth_overrides:
        val = auth_overrides["get_request_id"]
        chat_mod.get_request_id = lambda request, _v=val: _v
    if "is_admin" in auth_overrides:
        val = auth_overrides["is_admin"]
        chat_mod.is_admin = lambda request, _v=val: _v

    return TestClient(app), chat_mod


class TestChatCompletionAuth:
    """Chat completion requires valid auth headers."""

    def test_missing_user_id_returns_401(self):
        client, chat_mod = _make_client(get_user_id=None)
        resp = client.post(
            "/chat/completions",
            json={"message": {"role": "user", "content": [], "conversation_id": 1}},
        )
        assert resp.status_code == 401

    def test_missing_conversation_id_returns_400(self):
        client, chat_mod = _make_client(get_user_id="u-1", get_request_id="req-1")
        resp = client.post(
            "/chat/completions",
            json={"message": {"role": "user", "content": []}},
        )
        assert resp.status_code == 400
        assert "Conversation ID" in resp.json()["detail"]

    def test_missing_request_id_returns_400(self):
        client, chat_mod = _make_client(get_user_id="u-1", get_request_id=None)
        resp = client.post(
            "/chat/completions",
            json={"message": {"role": "user", "content": [], "conversation_id": 1}},
        )
        assert resp.status_code == 400
        assert "Request ID" in resp.json()["detail"]


class TestChatCompletionInvalidRole:
    """Only USER role messages are accepted."""

    def test_assistant_role_rejected(self):
        client, chat_mod = _make_client(get_user_id="u-1", get_request_id="req-1")
        resp = client.post(
            "/chat/completions",
            json={"message": {"role": "assistant", "content": [], "conversation_id": 1}},
        )
        assert resp.status_code == 400


class TestChatCompletionSuccess:
    """Happy path: valid request returns streaming response."""

    def test_valid_request_returns_200(self):
        mock_msg = MagicMock()
        mock_msg.conversation_id = 1

        client, chat_mod = _make_client(get_user_id="u-1", get_request_id="req-1")

        async def _empty_async_iter(**kwargs):
            return
            yield  # make this an async generator

        with patch.object(chat_mod, "transform_file_content_to_documents", new_callable=AsyncMock) as mock_transform:
            mock_transform.return_value = mock_msg
            with patch.object(chat_mod, "message_service") as mock_svc:
                mock_svc.add_message = AsyncMock()
                with patch.object(chat_mod, "CompletionService") as mock_cs:
                    mock_cs.stream_completion = _empty_async_iter
                    resp = client.post(
                        "/chat/completions",
                        json={"message": {"role": "user", "content": [], "conversation_id": 1}},
                    )
                assert resp.status_code == 200
                assert resp.headers["cache-control"] == "no-cache"


class TestAdminEndpoint:
    """Admin endpoint enforces admin role."""

    def test_non_admin_forbidden(self):
        client, chat_mod = _make_client(
            is_admin=False, get_user_id="u-1", get_request_id="req-1"
        )
        resp = client.get("/chat/admin")
        assert resp.status_code == 403

    def test_admin_allowed(self):
        client, chat_mod = _make_client(
            is_admin=True, get_user_id="admin-1", get_request_id="req-1"
        )
        resp = client.get("/chat/admin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["user_id"] == "admin-1"
