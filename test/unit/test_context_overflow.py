"""Regression tests for the context-overflow guard.

These lock in the fix for the "session dead-ends mid-conversation" bug:

  * The runner reported ``details.original_ctx`` as a bogus **2048** for
    Qwen3.6-27B. ``_get_model_num_ctx`` used to return that, so
    ``is_context_overflow`` flagged *every* conversation over 2048 tokens as
    overflow and skipped the empty-response rescue-retry — the session just
    stopped. (Masked while prompt tokens were under-reported as ~0; surfaced
    once token counting became accurate.)

  * An empty/short response that still FITS within the real window is a
    transient stop and must remain retryable, not be treated as overflow.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services import model_service as _ms
from services.completion_service import CompletionService
from services.truncation import is_context_overflow


# --- is_context_overflow: known window -----------------------------------

def test_empty_response_within_window_is_not_overflow():
    # 170k prompt, empty output, real 200k window → transient stop, retryable.
    assert is_context_overflow(170_000, "stop", 0, model_num_ctx=200_000) is False


def test_short_output_within_window_is_not_overflow():
    assert is_context_overflow(170_000, "stop", 1, model_num_ctx=200_000) is False


def test_exceeding_window_is_overflow():
    assert is_context_overflow(200_500, "stop", 0, model_num_ctx=200_000) is True


def test_bogus_small_window_would_flag_everything():
    # Demonstrates the original failure mode: a 2048 "window" flags any real
    # conversation. _get_model_num_ctx no longer returns this, but the contract
    # is explicit here so a regression in the lookup is caught downstream.
    assert is_context_overflow(170_000, "stop", 1, model_num_ctx=2048) is True


# --- is_context_overflow: unknown window (heuristic fallback) -------------

def test_unknown_window_small_prompt_not_overflow():
    assert is_context_overflow(5_000, "stop", 0, model_num_ctx=None) is False


def test_unknown_window_large_empty_is_overflow():
    assert is_context_overflow(150_000, "stop", 0, model_num_ctx=None) is True


def test_unknown_window_large_with_output_not_overflow():
    assert is_context_overflow(150_000, "stop", 5, model_num_ctx=None) is False


# --- _get_model_num_ctx: prefer runtime num_ctx, distrust 2048 default ----

@pytest.mark.asyncio
async def test_get_model_num_ctx_prefers_runtime_num_ctx():
    model = MagicMock()
    model.parameters.num_ctx = 200_000
    model.details.original_ctx = 2048  # garbage default — must be ignored
    with patch.object(_ms, "get_model_by_id", AsyncMock(return_value=model)):
        assert await CompletionService._get_model_num_ctx("Qwen3_6_27B") == 200_000


@pytest.mark.asyncio
async def test_get_model_num_ctx_ignores_bogus_original_ctx():
    model = MagicMock()
    model.parameters.num_ctx = None
    model.details.original_ctx = 2048  # too small to trust → None (heuristic)
    with patch.object(_ms, "get_model_by_id", AsyncMock(return_value=model)):
        assert await CompletionService._get_model_num_ctx("m") is None


@pytest.mark.asyncio
async def test_get_model_num_ctx_falls_back_to_plausible_original_ctx():
    model = MagicMock()
    model.parameters.num_ctx = None
    model.details.original_ctx = 131_072
    with patch.object(_ms, "get_model_by_id", AsyncMock(return_value=model)):
        assert await CompletionService._get_model_num_ctx("m") == 131_072


@pytest.mark.asyncio
async def test_get_model_num_ctx_none_when_model_missing():
    with patch.object(_ms, "get_model_by_id", AsyncMock(return_value=None)):
        assert await CompletionService._get_model_num_ctx("nope") is None
