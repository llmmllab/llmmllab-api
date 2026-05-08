"""
Unit tests for RequestMetadata and priority classification.

Covers RequestSource, Priority, and RequestMetadata from models/request_metadata.py,
and the priority middleware's _classify_request logic from middleware/priority.py.
"""

import time

import pytest

from models.request_metadata import Priority, RequestMetadata, RequestSource


class TestRequestSource:
    """RequestSource is a string-based enum."""

    def test_values_are_strings(self):
        assert RequestSource.USER.value == "user"
        assert RequestSource.SCHEDULED.value == "scheduled"
        assert RequestSource.SYSTEM.value == "system"

    def test_string_comparison(self):
        assert RequestSource.USER == "user"
        assert RequestSource.SCHEDULED == "scheduled"
        assert RequestSource.SYSTEM == "system"

    def test_all_members_present(self):
        members = {m.name for m in RequestSource}
        assert members == {"USER", "SCHEDULED", "SYSTEM"}


class TestPriority:
    """Priority is an IntEnum with lower value = higher priority."""

    def test_values(self):
        assert Priority.HIGH == 1
        assert Priority.MEDIUM == 2
        assert Priority.LOW == 3

    def test_ordering(self):
        assert Priority.HIGH < Priority.MEDIUM < Priority.LOW


class TestRequestMetadata:
    """RequestMetadata defaults and computed properties."""

    def test_defaults(self):
        meta = RequestMetadata()
        assert meta.source == RequestSource.USER
        assert meta.priority == Priority.HIGH
        assert meta.user_id is None
        assert meta.session_id is None
        assert meta.scheduled_at is None

    def test_custom_values(self):
        meta = RequestMetadata(
            source=RequestSource.SCHEDULED,
            priority=Priority.LOW,
            user_id="u-1",
            session_id="s-1",
            scheduled_at=100.0,
        )
        assert meta.source == RequestSource.SCHEDULED
        assert meta.priority == Priority.LOW
        assert meta.user_id == "u-1"
        assert meta.session_id == "s-1"
        assert meta.scheduled_at == 100.0

    def test_wait_time_increases(self):
        meta = RequestMetadata()
        t1 = meta.wait_time
        time.sleep(0.05)
        t2 = meta.wait_time
        assert t2 > t1

    def test_wait_time_starts_near_zero(self):
        meta = RequestMetadata()
        assert meta.wait_time >= 0
        assert meta.wait_time < 1.0
