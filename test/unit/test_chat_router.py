"""
Unit tests for the root chat/conversation router endpoints.

Tests the thin router layer in routers/chat.py and routers/conversation.py
with mocked service dependencies, verifying request validation, auth checks,
and proper delegation to services.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


def _build_app():
    """Build a minimal FastAPI app with the chat router mounted."""
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
        from routers.chat import router as chat_router

        app.include_router(chat_router)
    return app


@pytest.fixture
def client():
    app = _build_app()
    return TestClient(app)


class TestChatCompletionAuth:
    """Chat completion requires valid auth headers."""

    def test_missing_user_id_returns_401(self, client):
        with patch("routers.chat.get_user_id", return_value=None):
            with patch("routers.chat.get_request_id", return_value="req-1"):
                resp = client.post(
                    "/chat/completions",
                    json={"message": {"role": "user", "content": [], "conversation_id": 1}},
                )
            assert resp.status_code == 401

    def test_missing_conversation_id_returns_400(self, client):
        with patch("routers.chat.get_user_id", return_value="u-1"):
            with patch("routers.chat.get_request_id", return_value="req-1"):
                resp = client.post(
                    "/chat/completions",
                    json={"message": {"role": "user", "content": []}},
                )
            assert resp.status_code == 400
            assert "Conversation ID" in resp.json()["detail"]

    def test_missing_request_id_returns_400(self, client):
        with patch("routers.chat.get_user_id", return_value="u-1"):
            with patch("routers.chat.get_request_id", return_value=None):
                resp = client.post(
                    "/chat/completions",
                    json={"message": {"role": "user", "content": [], "conversation_id": 1}},
                )
            assert resp.status_code == 400
            assert "Request ID" in resp.json()["detail"]


class TestChatCompletionInvalidRole:
    """Only USER role messages are accepted."""

    def test_assistant_role_rejected(self, client):
        with patch("routers.chat.get_user_id", return_value="u-1"):
            with patch("routers.chat.get_request_id", return_value="req-1"):
                resp = client.post(
                    "/chat/completions",
                    json={"message": {"role": "assistant", "content": [], "conversation_id": 1}},
                )
            assert resp.status_code == 400


class TestChatCompletionSuccess:
    """Happy path: valid request returns streaming response."""

    def test_valid_request_returns_200(self, client):
        mock_msg = MagicMock()
        mock_msg.conversation_id = 1

        with patch("routers.chat.get_user_id", return_value="u-1"):
            with patch("routers.chat.get_request_id", return_value="req-1"):
                with patch("routers.chat.transform_file_content_to_documents", new_callable=AsyncMock) as mock_transform:
                    mock_transform.return_value = mock_msg
                    with patch("routers.chat.message_service") as mock_svc:
                        mock_svc.add_message = AsyncMock()
                        with patch("routers.chat.CompletionService") as mock_cs:
                            mock_cs.stream_completion = AsyncMock(return_value=iter([]))
                            resp = client.post(
                                "/chat/completions",
                                json={"message": {"role": "user", "content": [], "conversation_id": 1}},
                            )
                        assert resp.status_code == 200
                        assert resp.headers["cache-control"] == "no-cache"


class TestAdminEndpoint:
    """Admin endpoint enforces admin role."""

    def test_non_admin_forbidden(self, client):
        with patch("routers.chat.is_admin", return_value=False):
            with patch("routers.chat.get_user_id", return_value="u-1"):
                with patch("routers.chat.get_request_id", return_value="req-1"):
                    resp = client.get("/chat/admin")
                assert resp.status_code == 403

    def test_admin_allowed(self, client):
        with patch("routers.chat.is_admin", return_value=True):
            with patch("routers.chat.get_user_id", return_value="admin-1"):
                with patch("routers.chat.get_request_id", return_value="req-1"):
                    resp = client.get("/chat/admin")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "success"
                assert data["user_id"] == "admin-1"
