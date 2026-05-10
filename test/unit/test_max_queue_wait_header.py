"""Unit tests for X-Max-Queue-Wait header parsing in priority middleware."""

import pytest

from config import (
    PRIORITY_QUEUE_MAX_WAIT_MAX_SEC,
    PRIORITY_QUEUE_MAX_WAIT_MIN_SEC,
)


@pytest.fixture(autouse=True)
def _reset_module():
    """Ensure clean import for each test."""
    import importlib

    try:
        import middleware.priority

        importlib.reload(middleware.priority)
    except ImportError:
        pass


class TestParseMaxQueueWait:
    """_parse_max_queue_wait parses and validates the X-Max-Queue-Wait header."""

    def _get_parser(self):
        from middleware.priority import _parse_max_queue_wait

        return _parse_max_queue_wait

    def test_valid_values(self):
        parse = self._get_parser()
        assert parse("1") == 1.0
        assert parse("60") == 60.0
        assert parse("300") == 300.0
        assert parse("3600") == 3600.0

    def test_empty_string_returns_none(self):
        parse = self._get_parser()
        assert parse("") is None

    def test_invalid_string_returns_none(self):
        parse = self._get_parser()
        assert parse("abc") is None
        assert parse("1.5") is None
        assert parse("  ") is None

    def test_below_minimum_clamped(self):
        parse = self._get_parser()
        assert parse("0") == float(PRIORITY_QUEUE_MAX_WAIT_MIN_SEC)
        assert parse("-1") == float(PRIORITY_QUEUE_MAX_WAIT_MIN_SEC)
        assert parse("-100") == float(PRIORITY_QUEUE_MAX_WAIT_MIN_SEC)

    def test_above_maximum_clamped(self):
        parse = self._get_parser()
        assert parse("99999") == float(PRIORITY_QUEUE_MAX_WAIT_MAX_SEC)
        assert parse("100000") == float(PRIORITY_QUEUE_MAX_WAIT_MAX_SEC)


class TestClassifyRequestWithMaxQueueWait:
    """_classify_request extracts max_queue_wait from the header."""

    def _make_request(self, headers=None, state_attrs=None):
        from unittest.mock import MagicMock

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

    def test_no_header_defaults_none(self):
        from middleware.priority import _classify_request

        req = self._make_request()
        meta = _classify_request(req)
        assert meta.max_queue_wait is None

    def test_valid_header_parsed(self):
        from middleware.priority import _classify_request

        req = self._make_request(headers={"X-Max-Queue-Wait": "120"})
        meta = _classify_request(req)
        assert meta.max_queue_wait == 120.0

    def test_invalid_header_ignored(self):
        from middleware.priority import _classify_request

        req = self._make_request(headers={"X-Max-Queue-Wait": "abc"})
        meta = _classify_request(req)
        assert meta.max_queue_wait is None

    def test_clamped_low_value(self):
        from middleware.priority import _classify_request

        req = self._make_request(headers={"X-Max-Queue-Wait": "0"})
        meta = _classify_request(req)
        assert meta.max_queue_wait == float(PRIORITY_QUEUE_MAX_WAIT_MIN_SEC)

    def test_clamped_high_value(self):
        from middleware.priority import _classify_request

        req = self._make_request(headers={"X-Max-Queue-Wait": "99999"})
        meta = _classify_request(req)
        assert meta.max_queue_wait == float(PRIORITY_QUEUE_MAX_WAIT_MAX_SEC)
