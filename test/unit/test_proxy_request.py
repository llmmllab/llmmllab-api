"""
Unit tests for RunnerClient.proxy_request() — Retry-After aware proxying.

Tests the exponential backoff with timeout logic added in PR #3.
"""

import asyncio
import time

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, call

from services.runner_client import RunnerClient, ServerHandle


def _make_response(status_code, headers=None, **kwargs):
    """Build a mock httpx.Response with the given status and headers."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = httpx.Headers(headers or {})
    resp.is_stream_consumed = False
    resp.is_closed = True
    for k, v in kwargs.items():
        setattr(resp, k, v)
    return resp


def _mock_client(**overrides):
    client = AsyncMock()
    client.is_closed = False
    for key, value in overrides.items():
        setattr(client, key, value)
    return client


def _status_response(startup_epoch=1):
    """Build a mock /v1/status response (used by _check_runner_epoch).

    proxy_request calls _check_runner_epoch on every 503 retry to detect
    runner restarts. A 200 with a stable startup_epoch keeps the existing
    handle valid so the backoff/retry loop proceeds as expected.
    """
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"startup_epoch": startup_epoch}
    return resp


HANDLE = ServerHandle(
    base_url="http://runner:8000/v1/server/abc123",
    server_id="abc123",
    runner_host="http://runner:8000",
)


class TestProxyRequestSuccess:
    """Non-503 responses are returned immediately."""

    @pytest.mark.asyncio
    async def test_200_returned_immediately(self):
        resp = _make_response(200)
        mock = _mock_client(request=AsyncMock(return_value=resp))

        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock

        result = await client.proxy_request(HANDLE, "POST", "/v1/chat/completions", json={"model": "test"})

        assert result.status_code == 200
        assert mock.request.call_count == 1

    @pytest.mark.asyncio
    async def test_404_returned_immediately(self):
        resp = _make_response(404)
        mock = _mock_client(request=AsyncMock(return_value=resp))

        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock

        result = await client.proxy_request(HANDLE, "GET", "/health")

        assert result.status_code == 404
        assert mock.request.call_count == 1


class TestProxyRequestRetryAfter:
    """503 with Retry-After header triggers backoff and retry."""

    @pytest.mark.asyncio
    async def test_503_retries_with_retry_after_header(self):
        """First 503 with Retry-After: 1, then 200."""
        resp_503 = _make_response(503, {"Retry-After": "1"})
        resp_200 = _make_response(200)

        call_count = [0]
        async def mock_request(*a, **kw):
            call_count[0] += 1
            return resp_503 if call_count[0] == 1 else resp_200

        mock = _mock_client(
            request=AsyncMock(side_effect=mock_request),
            get=AsyncMock(return_value=_status_response()),
        )

        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock
        client._active_handles.add(HANDLE)

        result = await client.proxy_request(
            HANDLE, "POST", "/v1/chat/completions", json={}, timeout=10.0
        )

        assert result.status_code == 200
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_503_uses_default_retry_after_when_header_missing(self):
        """503 without Retry-After header falls back to 30s default, capped by timeout."""
        resp_503 = _make_response(503)  # no Retry-After header

        call_count = [0]
        async def mock_request(*a, **kw):
            call_count[0] += 1
            return resp_503

        mock = _mock_client(
            request=AsyncMock(side_effect=mock_request),
            get=AsyncMock(return_value=_status_response()),
        )

        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock
        client._active_handles.add(HANDLE)

        # timeout=5: first backoff is min(2^1, 30)=2s (fits), second is min(2^2,30)=4s
        # (2+4=6 > 5) so it stops after 2 attempts
        result = await client.proxy_request(
            HANDLE, "POST", "/v1/chat/completions", json={}, timeout=5.0
        )

        assert result.status_code == 503
        assert call_count[0] == 2


class TestProxyRequestExponentialBackoff:
    """Exponential backoff: 2, 4, 8, ... capped by Retry-After."""

    @pytest.mark.asyncio
    async def test_backoff_is_exponential(self):
        """Verify backoff intervals follow 2^attempt pattern."""
        resp_503 = _make_response(503, {"Retry-After": "60"})
        resp_200 = _make_response(200)

        sleep_times = []
        original_sleep = asyncio.sleep

        async def tracked_sleep(delay):
            sleep_times.append(delay)
            # Don't actually sleep in tests
            await original_sleep(0.01)

        call_count = [0]
        async def mock_request(*a, **kw):
            call_count[0] += 1
            return resp_503 if call_count[0] < 4 else resp_200

        mock = _mock_client(
            request=AsyncMock(side_effect=mock_request),
            get=AsyncMock(return_value=_status_response()),
        )

        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock
        client._active_handles.add(HANDLE)

        with patch("asyncio.sleep", tracked_sleep):
            result = await client.proxy_request(
                HANDLE, "POST", "/v1/chat/completions", json={}, timeout=120.0
            )

        assert result.status_code == 200
        # Backoff should be 2, 4, 8 (capped by Retry-After: 60)
        assert sleep_times == [2, 4, 8]

    @pytest.mark.asyncio
    async def test_backoff_capped_by_retry_after(self):
        """Backoff never exceeds Retry-After value."""
        resp_503 = _make_response(503, {"Retry-After": "3"})
        resp_200 = _make_response(200)

        sleep_times = []
        original_sleep = asyncio.sleep

        async def tracked_sleep(delay):
            sleep_times.append(delay)
            await original_sleep(0.01)

        call_count = [0]
        async def mock_request(*a, **kw):
            call_count[0] += 1
            return resp_503 if call_count[0] < 4 else resp_200

        mock = _mock_client(
            request=AsyncMock(side_effect=mock_request),
            get=AsyncMock(return_value=_status_response()),
        )

        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock
        client._active_handles.add(HANDLE)

        with patch("asyncio.sleep", tracked_sleep):
            result = await client.proxy_request(
                HANDLE, "POST", "/v1/chat/completions", json={}, timeout=120.0
            )

        assert result.status_code == 200
        # Exponential: 2, 4, 8 — but capped at Retry-After: 3
        assert sleep_times == [2, 3, 3]


class TestProxyRequestTimeout:
    """Cumulative backoff budget limits retries."""

    @pytest.mark.asyncio
    async def test_timeout_stops_retries(self):
        """When cumulative backoff exceeds timeout, return last 503."""
        resp_503 = _make_response(503, {"Retry-After": "60"})

        call_count = [0]
        async def mock_request(*a, **kw):
            call_count[0] += 1
            return resp_503

        mock = _mock_client(
            request=AsyncMock(side_effect=mock_request),
            get=AsyncMock(return_value=_status_response()),
        )

        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock
        client._active_handles.add(HANDLE)

        # timeout=5: first backoff is 2s (ok), second would be 4s (2+4=6 > 5)
        result = await client.proxy_request(
            HANDLE, "POST", "/v1/chat/completions", json={}, timeout=5.0
        )

        assert result.status_code == 503
        # First attempt + one retry (2s backoff fits in 5s budget)
        # Second retry would need 4s (2+4=6 > 5), so it stops
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_zero_timeout_returns_immediately(self):
        """timeout=0 means no backoff budget, return 503 immediately."""
        resp_503 = _make_response(503, {"Retry-After": "1"})

        call_count = [0]
        async def mock_request(*a, **kw):
            call_count[0] += 1
            return resp_503

        mock = _mock_client(request=AsyncMock(side_effect=mock_request))

        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock

        result = await client.proxy_request(
            HANDLE, "POST", "/v1/chat/completions", json={}, timeout=0.0
        )

        assert result.status_code == 503
        assert call_count[0] == 1


class TestProxyRequestPathHandling:
    """URL construction and request forwarding."""

    @pytest.mark.asyncio
    async def test_path_with_leading_slash(self):
        """Path with leading slash is handled correctly."""
        resp = _make_response(200)
        mock = _mock_client(request=AsyncMock(return_value=resp))

        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock

        await client.proxy_request(HANDLE, "POST", "/v1/chat/completions", json={})

        call_args = mock.request.call_args
        url = call_args[1]["url"]
        assert url == "http://runner:8000/v1/server/abc123/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_path_without_leading_slash(self):
        """Path without leading slash is handled correctly."""
        resp = _make_response(200)
        mock = _mock_client(request=AsyncMock(return_value=resp))

        client = RunnerClient(endpoints=["http://runner:8000"])
        client._client = mock

        await client.proxy_request(HANDLE, "POST", "v1/chat/completions", json={})

        call_args = mock.request.call_args
        url = call_args[1]["url"]
        assert url == "http://runner:8000/v1/server/abc123/v1/chat/completions"


from unittest.mock import patch
