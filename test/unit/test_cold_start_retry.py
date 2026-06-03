"""
Unit tests for the cold-start (model-still-loading) internal retry.

A FRESH model server takes ~45-90 s to load.  While it loads, the runner's
``/v1/server/create`` answers HTTP 503 ("Runner busy starting the model …"),
which ``acquire_server`` raises as ``ColdStartError``.  This used to bubble all
the way to the client (the agent turn failed with a 503).

``services.retry_policies.stream_with_connection_retry`` must now catch
``ColdStartError`` and retry INTERNALLY with a bounded number of fixed,
longer waits (``COLD_START_RETRIES`` × ``COLD_START_BACKOFF_SEC``), refreshing
the model map between attempts, before giving up.  The cold-start budget is
counted separately from the connection-error budget.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock

from openai import APIConnectionError

from graph.errors import ColdStartError
from services.retry_policies import stream_with_connection_retry, _looks_like_cold_start


def _kwargs(**overrides):
    """Default keyword args for stream_with_connection_retry."""
    base = dict(
        user_id="u1",
        messages=[],
        model_name="Qwen3_5_4B",
        workflow_type=None,
        conversation_id=0,
        client_tools=None,
        tool_choice=None,
        server_tool_names=None,
        model_parameters=None,
        max_retries=2,
        backoff_base=1,
        cold_start_retries=4,
        cold_start_backoff=20.0,
    )
    base.update(overrides)
    return base


async def _drain(agen):
    return [e async for e in agen]


# ── _looks_like_cold_start ──────────────────────────────────────────────────

class TestLooksLikeColdStart:
    def test_matches_runner_busy(self):
        assert _looks_like_cold_start(
            RuntimeError("No healthy runner … Last error: 503 Runner busy starting the model")
        )

    def test_matches_still_loading(self):
        assert _looks_like_cold_start(RuntimeError("model server is still loading"))

    def test_does_not_match_generic(self):
        assert not _looks_like_cold_start(RuntimeError("All inference slots are busy"))


# ── ColdStartError retry ────────────────────────────────────────────────────

class TestColdStartRetry:

    @pytest.mark.asyncio
    async def test_cold_start_then_success(self):
        """First call raises ColdStartError, retry succeeds → no error surfaced."""
        call_count = [0]
        refresh = AsyncMock()
        slept = []

        async def inner(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ColdStartError("Qwen3_5_4B")
            yield "ok"

        async def fake_sleep(d):
            slept.append(d)

        orig_sleep = asyncio.sleep
        asyncio.sleep = fake_sleep
        try:
            events = await _drain(
                stream_with_connection_retry(
                    inner, refresh_model_map=refresh, **_kwargs()
                )
            )
        finally:
            asyncio.sleep = orig_sleep

        assert events == ["ok"]
        assert call_count[0] == 2
        # Waited one cold-start backoff (the long one), refreshed the map.
        assert slept == [20.0]
        refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cold_start_exhaustion_raises(self):
        """ColdStartError every attempt → raises after the bounded budget."""
        call_count = [0]
        refresh = AsyncMock()
        slept = []

        async def inner(*a, **kw):
            call_count[0] += 1
            raise ColdStartError("Qwen3_5_4B")
            yield  # pragma: no cover

        async def fake_sleep(d):
            slept.append(d)

        orig_sleep = asyncio.sleep
        asyncio.sleep = fake_sleep
        try:
            with pytest.raises(ColdStartError):
                await _drain(
                    stream_with_connection_retry(
                        inner,
                        refresh_model_map=refresh,
                        **_kwargs(cold_start_retries=3, cold_start_backoff=15.0),
                    )
                )
        finally:
            asyncio.sleep = orig_sleep

        # initial attempt + 3 retries = 4 calls; 3 waits of 15 s each.
        assert call_count[0] == 4
        assert slept == [15.0, 15.0, 15.0]

    @pytest.mark.asyncio
    async def test_runtimeerror_cold_start_shape_retried(self):
        """A bare RuntimeError that reads 'still loading' uses cold-start budget."""
        call_count = [0]
        refresh = AsyncMock()
        slept = []

        async def inner(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError(
                    "No healthy runner available for model Qwen3_5_4B. "
                    "Last error: 503 model server is still loading"
                )
            yield "recovered"

        async def fake_sleep(d):
            slept.append(d)

        orig_sleep = asyncio.sleep
        asyncio.sleep = fake_sleep
        try:
            events = await _drain(
                stream_with_connection_retry(
                    inner, refresh_model_map=refresh, **_kwargs()
                )
            )
        finally:
            asyncio.sleep = orig_sleep

        assert events == ["recovered"]
        assert slept == [20.0]

    @pytest.mark.asyncio
    async def test_generic_runtimeerror_not_retried(self):
        """A non-cold-start RuntimeError propagates immediately (no retry)."""
        call_count = [0]
        refresh = AsyncMock()

        async def inner(*a, **kw):
            call_count[0] += 1
            raise RuntimeError("No healthy runner available. Last error: boom")
            yield  # pragma: no cover

        with pytest.raises(RuntimeError, match="boom"):
            await _drain(
                stream_with_connection_retry(
                    inner, refresh_model_map=refresh, **_kwargs()
                )
            )
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_cold_start_and_connection_budgets_independent(self):
        """A cold start does not consume the connection-error retry budget."""
        events_seq = []
        refresh = AsyncMock()
        slept = []

        async def inner(*a, **kw):
            events_seq.append("call")
            n = len(events_seq)
            if n == 1:
                raise ColdStartError("Qwen3_5_4B")
            if n == 2:
                raise APIConnectionError(message="conn drop", request=AsyncMock())
            yield "done"

        async def fake_sleep(d):
            slept.append(d)

        orig_sleep = asyncio.sleep
        asyncio.sleep = fake_sleep
        try:
            events = await _drain(
                stream_with_connection_retry(
                    inner,
                    refresh_model_map=refresh,
                    **_kwargs(max_retries=1, cold_start_retries=1),
                )
            )
        finally:
            asyncio.sleep = orig_sleep

        # One cold-start retry (20 s) + one connection retry (1 s) then success.
        assert events == ["done"]
        assert len(events_seq) == 3
        assert slept == [20.0, 1.0]
