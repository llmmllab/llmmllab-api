"""Tests for hallucinated tool name detection + nudge heuristic.

These cover the three observability/correctness layers added when the
model emits tool_use blocks with names that aren't in the bound list:

    1. Tool name extraction handles all client_tools shapes
       (Anthropic flat / Anthropic server-tool / OpenAI function /
       LangChain BaseTool).
    2. Partition correctly buckets valid vs hallucinated.
    3. The synthetic feedback prompt names the bad tool and lists
       valid alternatives (truncated when there are many).
    4. The "anticipated tool call" heuristic catches the common
       "Now let me X:" precursor pattern.
"""

from models.message import Message, MessageContent, MessageContentType, MessageRole
from models.tool_call import ToolCall
from services.prompt_templates import hallucinated_tool_feedback
from services.response_handlers import (
    extract_client_tool_names,
    looks_like_anticipated_tool_call,
    looks_like_premature_stop,
    partition_tool_calls_by_validity,
)


class TestExtractClientToolNames:
    def test_anthropic_flat_shape(self):
        tools = [
            {"name": "WebSearch", "description": "...", "input_schema": {}},
            {"name": "Read", "description": "..."},
        ]
        assert extract_client_tool_names(tools) == {"WebSearch", "Read"}

    def test_openai_function_shape(self):
        tools = [
            {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
            {"type": "function", "function": {"name": "send_email"}},
        ]
        assert extract_client_tool_names(tools) == {"get_weather", "send_email"}

    def test_anthropic_server_tool_strips_version_suffix(self):
        tools = [
            {"type": "web_search_20250305"},
            {"type": "web_fetch_20250122"},
        ]
        assert extract_client_tool_names(tools) == {"web_search", "web_fetch"}

    def test_langchain_basetool_uses_name_attr(self):
        class FakeBaseTool:
            def __init__(self, name):
                self.name = name

        tools = [FakeBaseTool("Bash"), FakeBaseTool("Edit")]
        assert extract_client_tool_names(tools) == {"Bash", "Edit"}

    def test_mixed_shapes(self):
        class FakeBaseTool:
            def __init__(self, name):
                self.name = name

        tools = [
            {"name": "mcp__freecad__create_object"},
            {"type": "function", "function": {"name": "Bash"}},
            {"type": "web_search_20250305"},
            FakeBaseTool("WebSearch"),
        ]
        assert extract_client_tool_names(tools) == {
            "mcp__freecad__create_object",
            "Bash",
            "web_search",
            "WebSearch",
        }

    def test_empty_or_none_returns_empty_set(self):
        assert extract_client_tool_names(None) == set()
        assert extract_client_tool_names([]) == set()

    def test_ignores_generic_type_values(self):
        """``type: "function"`` alone (no name, no function dict) shouldn't
        introduce a tool literally named "function".
        """
        tools = [{"type": "function"}, {"type": "custom"}, {"type": "tool"}]
        assert extract_client_tool_names(tools) == set()


class TestPartitionToolCallsByValidity:
    def _tc(self, name):
        return ToolCall(name=name, args={}, execution_id=f"id_{name}")

    def test_all_valid(self):
        valid_names = {"Bash", "Read"}
        calls = [self._tc("Bash"), self._tc("Read")]
        v, i = partition_tool_calls_by_validity(calls, valid_names)
        assert [c.name for c in v] == ["Bash", "Read"]
        assert i == []

    def test_mixed_valid_and_hallucinated(self):
        valid_names = {"Bash", "Read", "WebSearch"}
        calls = [
            self._tc("Bash"),
            self._tc("browser"),       # hallucinated
            self._tc("Read"),
            self._tc("memory_get"),    # hallucinated
        ]
        v, i = partition_tool_calls_by_validity(calls, valid_names)
        assert [c.name for c in v] == ["Bash", "Read"]
        assert [c.name for c in i] == ["browser", "memory_get"]

    def test_all_hallucinated(self):
        valid_names = {"Bash", "Read"}
        calls = [self._tc("browser"), self._tc("memory_get")]
        v, i = partition_tool_calls_by_validity(calls, valid_names)
        assert v == []
        assert [c.name for c in i] == ["browser", "memory_get"]

    def test_no_valid_names_returns_everything_as_valid(self):
        """When the caller can't supply bound names, don't pretend to
        validate — let everything through (unchanged from pre-fix
        behaviour).
        """
        calls = [self._tc("browser"), self._tc("anything")]
        v, i = partition_tool_calls_by_validity(calls, set())
        assert [c.name for c in v] == ["browser", "anything"]
        assert i == []

    def test_empty_calls(self):
        v, i = partition_tool_calls_by_validity([], {"Bash"})
        assert v == []
        assert i == []
        v, i = partition_tool_calls_by_validity(None, {"Bash"})
        assert v == []
        assert i == []


class TestHallucinatedToolFeedback:
    def test_lists_valid_names_and_names_the_bad_tool(self):
        msg = hallucinated_tool_feedback(
            ["browser"],
            {"Bash", "Read", "WebSearch", "Write"},
        )
        assert "browser" in msg
        assert "Bash" in msg
        assert "Read" in msg
        assert "WebSearch" in msg
        assert "Write" in msg
        # No truncation indicator since 4 names < default max_listed.
        assert "more" not in msg

    def test_truncates_long_valid_list(self):
        many = {f"tool_{i:03d}" for i in range(40)}
        msg = hallucinated_tool_feedback(["browser"], many, max_listed=10)
        # Should include the alphabetically-first 10 + indicate remainder.
        assert "tool_000" in msg
        assert "tool_009" in msg
        assert "tool_039" not in msg
        assert "30 more" in msg

    def test_dedupes_bad_names(self):
        msg = hallucinated_tool_feedback(
            ["browser", "browser", "browser"],
            {"Bash"},
        )
        assert msg.count("browser") == 1


class TestLooksLikeAnticipatedToolCall:
    def test_classic_let_me_x(self):
        assert looks_like_anticipated_tool_call("Now let me properly create that:")
        assert looks_like_anticipated_tool_call("Let me do that:")
        assert looks_like_anticipated_tool_call("I'll fix this:")
        assert looks_like_anticipated_tool_call("I will read the file:")

    def test_now_x(self):
        assert looks_like_anticipated_tool_call("Now reading the document:")
        assert looks_like_anticipated_tool_call("Next, I'll write the test:")

    def test_ellipsis_ending(self):
        assert looks_like_anticipated_tool_call("Let me check this...")

    def test_final_answer_not_flagged(self):
        # Complete sentence ending with period — should not flag.
        assert not looks_like_anticipated_tool_call(
            "The answer is 42, which I computed by summing the inputs."
        )

    def test_empty_or_none(self):
        assert not looks_like_anticipated_tool_call(None)
        assert not looks_like_anticipated_tool_call("")
        assert not looks_like_anticipated_tool_call("   \n\n  ")

    def test_real_world_user_example(self):
        """The exact phrasing the user reported: 'Now let me properly
        create internal threads inside that hole using ONE merged mesh
        (not 110 separate objects)' — should trigger nudge.
        """
        text = (
            "Now let me properly create internal threads inside that hole "
            "using ONE merged mesh (not 110 separate objects)"
        )
        # Original user message ended without a colon — but with
        # "let me X" pattern present anywhere in the tail it should
        # still match.  Add the colon the model usually adds:
        assert looks_like_anticipated_tool_call(text + ":")


def _text_msg(role, text):
    return Message(
        role=role,
        content=[MessageContent(type=MessageContentType.TEXT, text=text)],
    )


def _tool_result_msg_openai():
    """OpenAI shape: a dedicated TOOL-role message."""
    return Message(
        role=MessageRole.TOOL,
        content=[MessageContent(type=MessageContentType.TOOL_RESULT, text="ok")],
    )


def _tool_result_msg_anthropic():
    """Anthropic shape: tool result folded into a USER message."""
    return Message(
        role=MessageRole.USER,
        content=[MessageContent(type=MessageContentType.TOOL_RESULT, text="ok")],
    )


class TestLooksLikePrematureStop:
    """Tight gate: a tiny finish=stop answer mid-tool-loop is a bail."""

    def test_degenerate_single_token_after_tool_result(self):
        msgs = [
            _text_msg(MessageRole.USER, "fix the fstab"),
            _tool_result_msg_openai(),
        ]
        assert looks_like_premature_stop("I", msgs)
        assert looks_like_premature_stop("Done", msgs)
        assert looks_like_premature_stop("Ok", msgs)

    def test_anthropic_tool_result_shape(self):
        msgs = [_tool_result_msg_anthropic()]
        assert looks_like_premature_stop("I", msgs)

    def test_long_answer_not_premature(self):
        msgs = [_tool_result_msg_openai()]
        long = "I verified the fstab entry and both swapfiles are active now."
        assert not looks_like_premature_stop(long, msgs)

    def test_short_answer_but_not_mid_tool_loop(self):
        # Last message is a plain user question, not a tool result —
        # a short "Yes" here is a legitimate final answer, not a bail.
        msgs = [_text_msg(MessageRole.USER, "is it done?")]
        assert not looks_like_premature_stop("Yes", msgs)

    def test_empty_content_not_flagged(self):
        # Empty is handled by the empty-retry path, not this one.
        msgs = [_tool_result_msg_openai()]
        assert not looks_like_premature_stop("", msgs)
        assert not looks_like_premature_stop("   ", msgs)

    def test_no_messages(self):
        assert not looks_like_premature_stop("I", [])
        assert not looks_like_premature_stop("I", None)

    def test_assistant_turns_skipped_to_find_tool_result(self):
        # The model's own just-emitted assistant turn shouldn't mask the
        # preceding tool result.
        msgs = [
            _tool_result_msg_openai(),
            _text_msg(MessageRole.ASSISTANT, "I"),
        ]
        assert looks_like_premature_stop("I", msgs)
