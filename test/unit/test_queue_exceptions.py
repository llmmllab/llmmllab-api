"""Unit tests for queue exceptions."""

import pytest

from services.queue_exceptions import QueueFullError, QueueTimeoutError


class TestQueueTimeoutError:
    def test_message_format(self):
        exc = QueueTimeoutError(max_wait_sec=60.0, actual_wait_sec=60.5)
        assert "60.5s" in str(exc)
        assert "max wait: 60" in str(exc)

    def test_attributes(self):
        exc = QueueTimeoutError(max_wait_sec=120.0, actual_wait_sec=121.3)
        assert exc.max_wait_sec == 120.0
        assert exc.actual_wait_sec == 121.3

    def test_is_exception(self):
        exc = QueueTimeoutError(max_wait_sec=10.0, actual_wait_sec=10.0)
        assert isinstance(exc, Exception)


class TestQueueFullError:
    def test_message(self):
        exc = QueueFullError("Queue is at capacity")
        assert "Queue is at capacity" in str(exc)

    def test_empty_message(self):
        exc = QueueFullError()
        assert str(exc) == ""

    def test_is_exception(self):
        exc = QueueFullError()
        assert isinstance(exc, Exception)
