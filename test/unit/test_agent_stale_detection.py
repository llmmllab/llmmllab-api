"""
Unit tests for stale server detection in agents/base.py.

When the agent encounters a 404 "Server X not found" error from the
runner, it should re-raise as StaleServerError so the CompletionService
can recover by acquiring a fresh server handle.

These tests verify the detection logic in isolation without needing
a full agent graph.
"""

import pytest
import re

from graph.errors import StaleServerError


class TestStaleServerDetectionInAgent:
    """Test the stale server detection logic extracted from agents/base.py.

    The actual detection code lives in agents/base.py around the
    ``except Exception as e`` handler in ``create_agent_run``.
    We test the detection pattern directly here.
    """

    @staticmethod
    def _is_stale_server_error_message(error_text: str) -> bool:
        """Replicate the detection logic from agents/base.py."""
        lower = error_text.lower()
        return ("404" in lower or "not found" in lower) and "server" in lower

    @staticmethod
    def _extract_server_id(error_text: str) -> str:
        """Replicate the server ID extraction from agents/base.py."""
        m = re.search(r"server\s+([a-f0-9]+)", error_text, re.IGNORECASE)
        return m.group(1) if m else "unknown"

    def test_detects_404_server_not_found(self):
        """Standard 404 error with server ID is detected."""
        msg = "Error code: 404 - Server abc123def456 not found"
        assert self._is_stale_server_error_message(msg) is True

    def test_detects_not_found_without_404(self):
        """'not found' + 'server' without explicit 404 is detected."""
        msg = "Server abc123 not found on runner"
        assert self._is_stale_server_error_message(msg) is True

    def test_detects_case_insensitive(self):
        """Detection works with mixed case."""
        msg = "SERVER ABC123 NOT FOUND"
        assert self._is_stale_server_error_message(msg) is True

    def test_does_not_detect_regular_404(self):
        """A 404 without 'server' keyword is not stale."""
        msg = "Error code: 404 - Resource not found"
        assert self._is_stale_server_error_message(msg) is False

    def test_does_not_detect_server_busy(self):
        """'server' without 'not found' or '404' is not stale."""
        msg = "Server abc123 is busy"
        assert self._is_stale_server_error_message(msg) is False

    def test_does_not_detect_timeout(self):
        """Timeout errors are not stale server errors."""
        msg = "Request timed out after 30s"
        assert self._is_stale_server_error_message(msg) is False

    def test_does_not_detect_connection_error(self):
        """Connection errors are not stale server errors."""
        msg = "Connection refused by runner"
        assert self._is_stale_server_error_message(msg) is False

    def test_extract_server_id_hex(self):
        """Extract a hex server ID from the error message."""
        msg = "Error code: 404 - Server abc123def456 not found"
        assert self._extract_server_id(msg) == "abc123def456"

    def test_extract_server_id_short(self):
        """Extract a short hex server ID."""
        msg = "Server a1b2c3 not found"
        assert self._extract_server_id(msg) == "a1b2c3"

    def test_extract_server_id_fallback_unknown(self):
        """When no hex ID is found, return 'unknown'."""
        msg = "Server not found (no ID in message)"
        assert self._extract_server_id(msg) == "unknown"

    def test_stale_server_error_construction_from_agent(self):
        """Verify StaleServerError is properly constructed from agent detection."""
        orig = Exception("Error code: 404 - Server abc123 not found")
        server_id = self._extract_server_id(str(orig))
        err = StaleServerError(server_id, orig)
        assert err.server_id == "abc123"
        assert err.original_error is orig
        assert "abc123" in str(err)

    def test_stale_server_error_with_unknown_id(self):
        """StaleServerError works with 'unknown' server ID."""
        orig = Exception("Server not found")
        server_id = self._extract_server_id(str(orig))
        err = StaleServerError(server_id, orig)
        assert err.server_id == "unknown"
        assert err.original_error is orig
