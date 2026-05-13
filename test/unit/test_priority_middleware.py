"""
Unit tests for the priority middleware.

Tests _classify_request logic from middleware/priority.py using
mocked Starlette requests.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from models.request_priority_metadata import Priority, RequestSource


def _make_request(headers: dict | None = None, state_attrs: dict | None = None):
    """Build a minimal mock Request with configurable headers and state."""
    req = MagicMock()
    req.headers = headers or {}
    req.state = MagicMock()
    if state_attrs:
        for k, v in state_attrs.items():
            setattr(req.state, k, v)
    else:
        req.state.user_id = None
    req.url.path = "/v1/chat/completions"
    return req


class TestClassifyRequestSource:
    """Source is determined by X-Request-Source header."""

    def test_default_is_user(self):
        from middleware.priority import _classify_request

        req = _make_request()
        meta = _classify_request(req)
        assert meta.source == RequestSource.USER

    def test_scheduled_header(self):
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Request-Source": "scheduled"})
        meta = _classify_request(req)
        assert meta.source == RequestSource.SCHEDULED

    def test_system_header(self):
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Request-Source": "system"})
        meta = _classify_request(req)
        assert meta.source == RequestSource.SYSTEM

    def test_unknown_header_defaults_to_user(self):
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Request-Source": "unknown"})
        meta = _classify_request(req)
        assert meta.source == RequestSource.USER


class TestClassifyRequestPriority:
    """Priority defaults by source, but can be overridden by header."""

    def test_user_default_is_high(self):
        from middleware.priority import _classify_request

        req = _make_request()
        meta = _classify_request(req)
        assert meta.priority == Priority.HIGH

    def test_scheduled_default_is_low(self):
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Request-Source": "scheduled"})
        meta = _classify_request(req)
        assert meta.priority == Priority.LOW

    def test_system_default_is_low(self):
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Request-Source": "system"})
        meta = _classify_request(req)
        assert meta.priority == Priority.LOW

    def test_header_override_to_low(self):
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Request-Priority": "low"})
        meta = _classify_request(req)
        assert meta.priority == Priority.LOW

    def test_header_override_to_high(self):
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Request-Priority": "high"})
        meta = _classify_request(req)
        assert meta.priority == Priority.HIGH


class TestClassifyRequestMetadata:
    """User ID and session ID are extracted correctly."""

    def test_user_id_from_state(self):
        from middleware.priority import _classify_request

        req = _make_request(state_attrs={"user_id": "u-42"})
        meta = _classify_request(req)
        assert meta.user_id == "u-42"

    def test_session_id_from_header(self):
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Session-ID": "sess-abc"})
        meta = _classify_request(req)
        assert meta.session_id == "sess-abc"


class TestMaxQueueWaitHeader:
    """X-Max-Queue-Wait header is parsed and attached to metadata."""

    def test_no_header_is_none(self):
        from middleware.priority import _classify_request

        req = _make_request()
        meta = _classify_request(req)
        assert meta.max_queue_wait is None

    def test_valid_header_parsed(self):
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Max-Queue-Wait": "120"})
        meta = _classify_request(req)
        assert meta.max_queue_wait == 120.0

    def test_invalid_header_is_none(self):
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Max-Queue-Wait": "abc"})
        meta = _classify_request(req)
        assert meta.max_queue_wait is None

    def test_zero_clamped_to_minimum(self):
        from config import PRIORITY_QUEUE_MAX_WAIT_MIN_SEC
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Max-Queue-Wait": "0"})
        meta = _classify_request(req)
        assert meta.max_queue_wait == float(PRIORITY_QUEUE_MAX_WAIT_MIN_SEC)

    def test_large_value_clamped_to_maximum(self):
        from config import PRIORITY_QUEUE_MAX_WAIT_MAX_SEC
        from middleware.priority import _classify_request

        req = _make_request(headers={"X-Max-Queue-Wait": "99999"})
        meta = _classify_request(req)
        assert meta.max_queue_wait == float(PRIORITY_QUEUE_MAX_WAIT_MAX_SEC)


class TestExtractBodyFields:
    """Body field extraction for session_id and model_id from request body."""

    @pytest.mark.asyncio
    async def test_prompt_cache_key_extracted_as_session_id(self):
        from middleware.priority import _extract_body_fields

        body = b'{"model": "Qwen3_6_27B", "prompt_cache_key": "sess-from-body"}'
        req = MagicMock()
        req.headers = {"content-type": "application/json"}
        req.body = AsyncMock(return_value=body)
        fields = await _extract_body_fields(req)
        assert fields.session_id == "sess-from-body"
        assert fields.model_id == "Qwen3_6_27B"

    @pytest.mark.asyncio
    async def test_no_prompt_cache_key_returns_none_session_id(self):
        from middleware.priority import _extract_body_fields

        body = b'{"model": "Qwen3_6_27B"}'
        req = MagicMock()
        req.headers = {"content-type": "application/json"}
        req.body = AsyncMock(return_value=body)
        fields = await _extract_body_fields(req)
        assert fields.session_id is None
        assert fields.model_id == "Qwen3_6_27B"

    @pytest.mark.asyncio
    async def test_non_json_content_type_returns_none(self):
        from middleware.priority import _extract_body_fields

        req = MagicMock()
        req.headers = {"content-type": "text/plain"}
        fields = await _extract_body_fields(req)
        assert fields is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        from middleware.priority import _extract_body_fields

        req = MagicMock()
        req.headers = {"content-type": "application/json"}
        req.body = AsyncMock(return_value=b"not json")
        fields = await _extract_body_fields(req)
        assert fields is None

    @pytest.mark.asyncio
    async def test_empty_prompt_cache_key_treated_as_none(self):
        from middleware.priority import _extract_body_fields

        body = b'{"model": "Qwen3_6_27B", "prompt_cache_key": ""}'
        req = MagicMock()
        req.headers = {"content-type": "application/json"}
        req.body = AsyncMock(return_value=body)
        fields = await _extract_body_fields(req)
        assert fields.session_id is None


class TestDispatchBodyFallback:
    """PriorityMiddleware dispatch falls back to body fields when headers are missing."""

    @pytest.mark.asyncio
    async def test_session_id_from_body_when_no_header(self):
        from middleware.priority import PriorityMiddleware

        body = b'{"model": "Qwen3_6_27B", "prompt_cache_key": "sess-body-123"}'
        req = MagicMock()
        req.url.path = "/v1/chat/completions"
        req.headers = {
            "content-type": "application/json",
            "X-Request-Priority": "low",
        }
        req.body = AsyncMock(return_value=body)
        req.state = MagicMock()
        req.state.user_id = None  # prevent MagicMock from leaking into pydantic

        response = MagicMock()
        response.headers = {}
        call_next = AsyncMock(return_value=response)

        middleware = PriorityMiddleware(app=MagicMock())
        result = await middleware.dispatch(req, call_next)

        assert req.state.request_priority_metadata.session_id == "sess-body-123"
        assert req.state.request_priority_metadata.model_id == "Qwen3_6_27B"

    @pytest.mark.asyncio
    async def test_header_session_id_takes_precedence_over_body(self):
        from middleware.priority import PriorityMiddleware

        body = b'{"model": "Qwen3_6_27B", "prompt_cache_key": "sess-body"}'
        req = MagicMock()
        req.url.path = "/v1/chat/completions"
        req.headers = {
            "content-type": "application/json",
            "X-Request-Priority": "low",
            "X-OpenClaw-Session-ID": "sess-header",
        }
        req.body = AsyncMock(return_value=body)
        req.state = MagicMock()
        req.state.user_id = None  # prevent MagicMock from leaking into pydantic

        response = MagicMock()
        response.headers = {}
        call_next = AsyncMock(return_value=response)

        middleware = PriorityMiddleware(app=MagicMock())
        await middleware.dispatch(req, call_next)

        assert req.state.request_priority_metadata.session_id == "sess-header"

    @pytest.mark.asyncio
    async def test_non_completion_endpoint_skips_body_extraction(self):
        from middleware.priority import PriorityMiddleware
        from types import SimpleNamespace

        req = MagicMock()
        req.url.path = "/v1/models"
        req.headers = {}
        req.state = SimpleNamespace()  # use SimpleNamespace so hasattr works correctly

        response = MagicMock()
        response.headers = {}
        call_next = AsyncMock(return_value=response)

        middleware = PriorityMiddleware(app=MagicMock())
        await middleware.dispatch(req, call_next)

        assert not hasattr(req.state, "request_priority_metadata")
