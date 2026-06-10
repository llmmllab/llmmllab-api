"""Tests for _ainvoke_or_abort — cancel an in-flight generation when the client
leaves mid-generation (Starlette doesn't cancel the SSE generator during silent
prefill/generation phases, so the runner would otherwise orphan the GPU turn)."""

import asyncio
from unittest.mock import MagicMock

import pytest

from agents.base import _ainvoke_or_abort


class _FakeAgent:
    def __init__(self, delay, result="ok"):
        self.delay = delay
        self.result = result
        self.cancelled = False

    async def ainvoke(self, payload):
        try:
            await asyncio.sleep(self.delay)
            return self.result
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_returns_result_when_connected():
    agent = _FakeAgent(0.01)

    async def never_gone():
        return False

    out = await _ainvoke_or_abort(agent, {"messages": []}, never_gone,
                                  MagicMock(), poll=0.01)
    assert out == "ok"
    assert agent.cancelled is False


@pytest.mark.asyncio
async def test_no_predicate_is_plain_ainvoke():
    # disconnected=None => byte-for-byte old behaviour (no watcher task).
    agent = _FakeAgent(0.01)
    out = await _ainvoke_or_abort(agent, {"messages": []}, None, MagicMock())
    assert out == "ok"


@pytest.mark.asyncio
async def test_cancels_inflight_when_disconnected():
    agent = _FakeAgent(10.0)  # long "generation" that must be aborted
    polls = {"n": 0}

    async def gone():
        polls["n"] += 1
        return polls["n"] >= 2  # report disconnect on the 2nd poll

    with pytest.raises(asyncio.CancelledError):
        await _ainvoke_or_abort(agent, {"messages": []}, gone,
                                MagicMock(), poll=0.01)
    # the in-flight generation was actually cancelled, not left running
    assert agent.cancelled is True
