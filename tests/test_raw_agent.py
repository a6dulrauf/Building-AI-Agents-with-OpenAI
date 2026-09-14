"""Tests for the hand-written loop, using a fake client.

There is no network here. A FakeClient replays a scripted sequence of
model responses, which lets us assert on loop behaviour — tool dispatch,
approval, the turn cap — deterministically and for free.
"""
import json
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from helpdesk.database import connect, init_schema, seed
from helpdesk import raw_agent, tools


@pytest.fixture(autouse=True)
def _wire_temp_db(tmp_path):
    conn = connect(str(tmp_path / "raw.db"))
    init_schema(conn)
    seed(conn)
    tools.set_connection(conn)
    yield
    tools.set_connection(None)


def _tool_call(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response(content=None, tool_calls=None):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        model_dump=lambda exclude_none=False: {
            "role": "assistant",
            "content": content,
        },
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Replays scripted responses in order and records what it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_a_reply_with_no_tool_calls_ends_the_loop():
    client = FakeClient([_response(content="Hello.")])
    messages = raw_agent.new_conversation()
    messages.append({"role": "user", "content": "hi"})

    out = raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)

    assert out == "Hello."
    assert len(client.calls) == 1


def test_a_tool_call_is_executed_and_fed_back():
    client = FakeClient([
        _response(tool_calls=[_tool_call("c1", "get_ticket", {"ticket_id": "T-1006"})]),
        _response(content="It is a PDF export crash."),
    ])
    messages = raw_agent.new_conversation()
    messages.append({"role": "user", "content": "what is T-1006?"})

    out = raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)

    assert out == "It is a PDF export crash."
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "App crashes on export to PDF" in tool_messages[0]["content"]


def test_reads_do_not_ask_for_approval():
    asked = []
    client = FakeClient([
        _response(tool_calls=[_tool_call("c1", "get_ticket", {"ticket_id": "T-1006"})]),
        _response(content="done"),
    ])
    messages = raw_agent.new_conversation()

    def approve(name, args):
        asked.append(name)
        return True

    raw_agent.run_conversation(client, "m", messages, approve=approve)
    assert asked == []  # get_ticket is a read; no human needed


def test_writes_ask_for_approval_and_a_refusal_is_reported_to_the_model():
    client = FakeClient([
        _response(tool_calls=[
            _tool_call("c1", "assign_department",
                       {"ticket_id": "T-1006", "department": "billing"})
        ]),
        _response(content="Understood, I will reconsider."),
    ])
    messages = raw_agent.new_conversation()

    raw_agent.run_conversation(client, "m", messages, approve=lambda *_: False)

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert "rejected" in tool_messages[0]["content"].lower()
    # And critically: the database was not touched.
    assert json.loads(tools.get_ticket("T-1006"))["department"] is None


def test_an_approved_write_actually_writes():
    client = FakeClient([
        _response(tool_calls=[
            _tool_call("c1", "assign_department",
                       {"ticket_id": "T-1006", "department": "technical"})
        ]),
        _response(content="Assigned."),
    ])
    messages = raw_agent.new_conversation()

    raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)
    assert json.loads(tools.get_ticket("T-1006"))["department"] == "technical"


def test_the_turn_cap_stops_a_runaway_loop():
    """A model that always calls a tool must not loop forever."""
    client = FakeClient([
        _response(tool_calls=[_tool_call(f"c{i}", "get_ticket", {"ticket_id": "T-1006"})])
        for i in range(10)
    ])
    messages = raw_agent.new_conversation()

    out = raw_agent.run_conversation(
        client, "m", messages, approve=lambda *_: True, max_turns=3
    )

    assert "turn limit" in out.lower()
    assert len(client.calls) == 3


def test_an_unknown_tool_name_does_not_crash_the_loop():
    client = FakeClient([
        _response(tool_calls=[_tool_call("c1", "delete_everything", {})]),
        _response(content="Sorry about that."),
    ])
    messages = raw_agent.new_conversation()

    out = raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)

    assert out == "Sorry about that."
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert tool_messages[0]["content"].startswith("error:")


def test_as_dict_preserves_tool_calls_from_a_real_message():
    """Pin _as_dict's fidelity against a real ChatCompletionMessage.

    The fixtures above use a SimpleNamespace whose fake model_dump always
    returns {"role": "assistant", "content": content} — it never exercises
    what happens to tool_calls when a real pydantic message is converted.
    That matters a great deal: the "tool" messages appended right after an
    assistant turn each carry a tool_call_id, and the OpenAI API requires
    that id to have been declared in the *preceding* assistant message's
    tool_calls. If _as_dict ever dropped tool_calls (or leaked a stray
    non-None field) for a real message, multi-turn tool use would still
    pass every test above yet be rejected by a real API on the next turn.
    This test would catch that; none of the others can.
    """
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_abc123",
                type="function",
                function=Function(
                    name="get_ticket",
                    arguments='{"ticket_id": "T-1006"}',
                ),
            )
        ],
    )

    result = raw_agent._as_dict(message)

    assert "tool_calls" in result
    assert result["tool_calls"][0]["id"] == "call_abc123"
    # No stray non-None fields: everything sent back to the API next turn
    # must be exactly role/content/tool_calls, nothing extra.
    assert set(result.keys()) <= {"role", "content", "tool_calls"}
