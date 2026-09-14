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


def _raw_tool_call(call_id, name, raw_arguments):
    """Like _tool_call, but sends the arguments string through verbatim.

    Needed to script a model that emits something which is not JSON at all.
    """
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=raw_arguments),
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


@pytest.mark.parametrize("bad_arguments", ["{not json", "[1, 2]"])
def test_malformed_tool_arguments_are_blamed_on_the_json_not_on_the_tool(
    bad_arguments,
):
    """The error the model reads must name the mistake the model made.

    A model sometimes emits arguments that are not JSON. Parsing those to
    {} and calling the tool anyway produced "error: get_ticket failed
    unexpectedly: get_ticket() missing 1 required positional argument:
    'ticket_id'" — which points at the tool, not at the broken JSON, and
    leaves the model no idea what to send instead. Every error string here
    is meant to be actionable by whoever reads it; this one was not.

    "[1, 2]" is valid JSON and still unusable: func(**arguments) needs a
    mapping, and a list would raise a TypeError at the call site — outside
    safe_tool's net, so it would take the whole loop down.
    """
    client = FakeClient([
        _response(tool_calls=[_raw_tool_call("c1", "get_ticket", bad_arguments)]),
        _response(content="Sorry, let me retry that."),
    ])
    messages = raw_agent.new_conversation()

    out = raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)

    assert out == "Sorry, let me retry that."
    content = [m for m in messages if m.get("role") == "tool"][0]["content"]
    assert content.startswith("error:")
    assert "not valid JSON" in content
    assert "get_ticket" in content
    assert "missing 1 required positional argument" not in content


def test_genuinely_empty_arguments_still_reach_the_tool():
    """"{}" is a valid call, not a parse failure — the two must stay apart.

    The fix above must not turn every argument-less call into a JSON
    error. search_tickets with no filters is a real call that the tool
    itself rejects, with its own message about needing a filter.
    """
    client = FakeClient([
        _response(tool_calls=[_raw_tool_call("c1", "search_tickets", "{}")]),
        _response(content="I need a filter."),
    ])
    messages = raw_agent.new_conversation()

    raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)

    content = [m for m in messages if m.get("role") == "tool"][0]["content"]
    assert "at least one filter" in content
    assert "not valid JSON" not in content


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


def test_a_tool_call_written_as_text_is_reported_not_executed():
    """A model that TYPES a tool call must not trigger one.

    Small models sometimes emit {"name": "assign_department", ...} into the
    message body instead of using the structured tool_calls channel. The
    loop must notice and tell the operator — otherwise the turn ends with
    prose, no approval prompt, and no explanation.

    Critically it must NOT execute what it found. Text the model happened
    to type is not a request; honouring it would let a model trigger a
    write just by talking about one, bypassing approval entirely.
    """
    text = ('I propose assigning it.\n\n{"name": "assign_department", '
            '"parameters": {"ticket_id": "T-1006", "department": "billing"}}')
    # Two responses: the loop nudges the model after the first and asks again.
    client = FakeClient([
        _response(content=text),
        _response(content="Understood."),
    ])
    messages = raw_agent.new_conversation()
    events = []

    raw_agent.run_conversation(
        client, "m", messages,
        approve=lambda *_: True,
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    reported = [p for kind, p in events if kind == "text_tool_call"]
    assert reported and reported[0]["name"] == "assign_department"
    # And nothing was written, despite approve returning True.
    assert json.loads(tools.get_ticket("T-1006"))["department"] is None


def test_ordinary_prose_is_not_mistaken_for_a_tool_call():
    """The detector must not cry wolf on a normal answer."""
    client = FakeClient([_response(content="This looks like a technical issue.")])
    messages = raw_agent.new_conversation()
    events = []

    raw_agent.run_conversation(
        client, "m", messages,
        approve=lambda *_: True,
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    assert not any(kind == "text_tool_call" for kind, _ in events)


def test_a_typed_tool_call_is_stripped_from_the_transcript():
    """The malformed artifact must not survive into the history.

    Measured against llama3.1, one assistant message containing a typed-out
    tool call drops the next turn's real tool-call rate from 4/6 to 1/6 —
    the model reads its own output as an example and imitates it. Storing
    the prose without the fake call is what stops one slip from killing the
    whole conversation.
    """
    text = ('This looks technical.\n\n{"name": "assign_department", '
            '"parameters": {"ticket_id": "T-1006", "department": "technical"}}')
    client = FakeClient([
        _response(content=text),
        _response(content="Done."),
    ])
    messages = raw_agent.new_conversation()

    raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)

    stored = [m for m in messages if m.get("role") == "assistant"]
    assert "This looks technical." in stored[0]["content"]
    assert "assign_department" not in stored[0]["content"]
    assert "{" not in stored[0]["content"]


def test_the_model_is_told_to_retry_through_the_tool_interface():
    text = '{"name": "add_note", "parameters": {"ticket_id": "T-1006"}}'
    client = FakeClient([
        _response(content=text),
        _response(content="Done."),
    ])
    messages = raw_agent.new_conversation()

    raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)

    nudges = [m for m in messages
              if m.get("role") == "user" and "not a tool call" in m.get("content", "")]
    assert len(nudges) == 1
    assert "add_note" in nudges[0]["content"]


def test_the_retry_happens_only_once_per_run():
    """A model that ignores the correction twice will not be talked round."""
    text = '{"name": "add_note", "parameters": {"ticket_id": "T-1006"}}'
    client = FakeClient([_response(content=text) for _ in range(4)])
    messages = raw_agent.new_conversation()

    out = raw_agent.run_conversation(
        client, "m", messages, approve=lambda *_: True, max_turns=4)

    nudges = [m for m in messages
              if m.get("role") == "user" and "not a tool call" in m.get("content", "")]
    assert len(nudges) == 1          # nudged once, then gave up
    assert len(client.calls) == 2    # and stopped rather than burning the cap
    assert out is not None
