"""Unit tests for helpers in routers/anthropic/messages.py.

These cover request-body massaging that happens *before* pydantic
validation, so they don't require the full router stack to import.
"""

from routers.anthropic.messages import (
    _coerce_system_messages,
    _strip_server_tool_blocks,
)


class TestCoerceSystemMessages:
    def test_hoists_string_system_message_into_top_level(self):
        body = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "be brief"},
                {"role": "assistant", "content": "ok"},
            ]
        }
        out = _coerce_system_messages(body)
        assert out["system"] == "be brief"
        assert [m["role"] for m in out["messages"]] == ["user", "assistant"]

    def test_hoists_block_form_system_message(self):
        body = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "stay terse"}],
                },
            ]
        }
        out = _coerce_system_messages(body)
        assert out["system"] == "stay terse"
        assert len(out["messages"]) == 1

    def test_appends_to_existing_string_system(self):
        body = {
            "system": "you are X",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "also Y"},
            ],
        }
        out = _coerce_system_messages(body)
        assert out["system"] == "you are X\nalso Y"

    def test_appends_to_existing_blocks_system(self):
        body = {
            "system": [{"type": "text", "text": "original"}],
            "messages": [
                {"role": "system", "content": "reminder"},
                {"role": "user", "content": "q"},
            ],
        }
        out = _coerce_system_messages(body)
        assert isinstance(out["system"], list)
        assert out["system"][-1] == {"type": "text", "text": "reminder"}

    def test_concatenates_multiple_system_messages(self):
        body = {
            "messages": [
                {"role": "user", "content": "1"},
                {"role": "system", "content": "a"},
                {"role": "user", "content": "2"},
                {"role": "system", "content": "b"},
            ]
        }
        out = _coerce_system_messages(body)
        assert out["system"] == "a\nb"
        assert [m["role"] for m in out["messages"]] == ["user", "user"]

    def test_noop_when_no_system_messages(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        out = _coerce_system_messages(body)
        assert "system" not in out
        assert out["messages"] == [{"role": "user", "content": "hi"}]

    def test_handles_empty_messages(self):
        assert _coerce_system_messages({}) == {}
        assert _coerce_system_messages({"messages": []}) == {"messages": []}

    def test_drops_non_text_content_blocks_on_system(self):
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "keep this"},
                        {"type": "image", "source": {}},
                    ],
                },
                {"role": "user", "content": "hi"},
            ]
        }
        out = _coerce_system_messages(body)
        assert out["system"] == "keep this"


class TestStripServerToolBlocks:
    """Sanity coverage so the new helper doesn't conflict with the existing one."""

    def test_strips_server_tool_use_blocks(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "server_tool_use", "id": "x"},
                    ],
                }
            ]
        }
        out = _strip_server_tool_blocks(body)
        assert out["messages"][0]["content"] == [{"type": "text", "text": "hi"}]

    def test_replaces_empty_with_placeholder(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "..."}],
                }
            ]
        }
        out = _strip_server_tool_blocks(body)
        assert out["messages"][0]["content"] == [
            {"type": "text", "text": "(content omitted)"}
        ]
