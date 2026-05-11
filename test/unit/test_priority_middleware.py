"""
Unit tests for the priority middleware.

Tests _classify_request logic from middleware/priority.py using
mocked Starlette requests.
"""

from unittest.mock import MagicMock

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
