"""Tests for incomplete tool turn recovery.

Covers:
  * last_message_has_tool_results() — detects when the model just received
    tool results and stopped (the incomplete-turn failure mode).
  * INCOMPLETE_TOOL_TURN_PROMPT — exported from prompt_templates.
  * StreamAccumulator / CompletionResult — incomplete_turn flag defaults.
"""

import pytest

from models.message import Message, MessageContent, MessageContentType, MessageRole
from services.completion_state import CompletionResult, StreamAccumulator
from services.prompt_templates import INCOMPLETE_TOOL_TURN_PROMPT
from services.response_handlers import last_message_has_tool_results


def _text(text: str) -> list[MessageContent]:
    return [MessageContent(type=MessageContentType.TEXT, text=text)]


def _tool_result(text: str) -> list[MessageContent]:
    return [MessageContent(type=MessageContentType.TOOL_RESULT, text=text)]


# ---------------------------------------------------------------------------
# last_message_has_tool_results
# ---------------------------------------------------------------------------


class TestLastMessageHasToolResults:
    def test_none_messages(self):
        assert last_message_has_tool_results(None) is False

    def test_empty_messages(self):
        assert last_message_has_tool_results([]) is False

    def test_only_assistant_messages(self):
        messages = [
            Message(role=MessageRole.ASSISTANT, content=_text("hello")),
        ]
        assert last_message_has_tool_results(messages) is False

    def test_user_text_message(self):
        messages = [
            Message(role=MessageRole.USER, content=_text("what's the weather?")),
        ]
        assert last_message_has_tool_results(messages) is False

    def test_tool_role_message(self):
        messages = [
            Message(role=MessageRole.USER, content=_text("what's the weather?")),
            Message(role=MessageRole.ASSISTANT, content=_text("let me check")),
            Message(role=MessageRole.TOOL, content=_text("72°F and sunny")),
        ]
        assert last_message_has_tool_results(messages) is True

    def test_tool_result_content_block(self):
        messages = [
            Message(role=MessageRole.USER, content=_text("what's the weather?")),
            Message(role=MessageRole.ASSISTANT, content=_text("let me check")),
            Message(
                role=MessageRole.USER,
                content=_tool_result("72°F and sunny"),
            ),
        ]
        assert last_message_has_tool_results(messages) is True

    def test_tool_result_string_type(self):
        """Handle dict-based content with 'tool_result' string type."""
        messages = [
            Message(role=MessageRole.USER, content=_text("run it")),
            Message(role=MessageRole.ASSISTANT, content=_text("")),
            Message(
                role=MessageRole.USER,
                content=[{"type": "tool_result", "text": "output here"}],
            ),
        ]
        assert last_message_has_tool_results(messages) is True

    def test_assistant_after_tool_result(self):
        """Assistant turn after tool result → NOT a tool result."""
        messages = [
            Message(role=MessageRole.TOOL, content=_text("72°F")),
            Message(role=MessageRole.ASSISTANT, content=_text("The weather is nice.")),
        ]
        assert last_message_has_tool_results(messages) is False

    def test_tool_result_deep_in_history(self):
        """Tool result buried under later user text → NOT the last message."""
        messages = [
            Message(role=MessageRole.TOOL, content=_text("result")),
            Message(role=MessageRole.USER, content=_text("now summarize")),
        ]
        assert last_message_has_tool_results(messages) is False

    def test_multiple_tool_results(self):
        messages = [
            Message(role=MessageRole.TOOL, content=_text("result 1")),
            Message(role=MessageRole.TOOL, content=_text("result 2")),
        ]
        assert last_message_has_tool_results(messages) is True

    def test_agent_role_skipped(self):
        """Agent message as last message → model already processed tool results."""
        messages = [
            Message(role=MessageRole.TOOL, content=_text("result")),
            Message(role=MessageRole.AGENT, content=_text("thinking...")),
        ]
        # Agent is the last message — model already saw the tool result
        assert last_message_has_tool_results(messages) is False


# ---------------------------------------------------------------------------
# INCOMPLETE_TOOL_TURN_PROMPT
# ---------------------------------------------------------------------------


class TestIncompleteToolTurnPrompt:
    def test_exists(self):
        assert INCOMPLETE_TOOL_TURN_PROMPT
        assert "tool results" in INCOMPLETE_TOOL_TURN_PROMPT.lower()
        assert "continue" in INCOMPLETE_TOOL_TURN_PROMPT.lower()


# ---------------------------------------------------------------------------
# incomplete_turn flags
# ---------------------------------------------------------------------------


class TestIncompleteTurnFlag:
    def test_stream_accumulator_default(self):
        acc = StreamAccumulator()
        assert acc.incomplete_turn is False

    def test_stream_accumulator_settable(self):
        acc = StreamAccumulator()
        acc.incomplete_turn = True
        assert acc.incomplete_turn is True

    def test_completion_result_default(self):
        result = CompletionResult()
        assert result.incomplete_turn is False

    def test_completion_result_settable(self):
        result = CompletionResult()
        result.incomplete_turn = True
        assert result.incomplete_turn is True
