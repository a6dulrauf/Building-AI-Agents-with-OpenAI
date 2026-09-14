"""Tests for the SDK wiring.

No model is called here. We assert on how the tools are DESCRIBED —
because that description is the entire contract between your code and the
model, and it is generated rather than written, which makes it worth
checking.
"""
import asyncio

import pytest

from helpdesk import sdk_agent, tools


@pytest.fixture()
def sdk_tools():
    return {t.name: t for t in sdk_agent.build_tools()}


def test_the_sdk_exposes_the_same_four_tools(sdk_tools):
    assert set(sdk_tools) == set(tools.TOOL_FUNCTIONS)


def test_only_the_write_tools_need_approval(sdk_tools):
    needing = {name for name, t in sdk_tools.items() if t.needs_approval}
    assert needing == tools.WRITE_TOOLS


@pytest.mark.parametrize("name", ["get_ticket", "search_tickets",
                                  "assign_department", "add_note"])
def test_generated_schema_matches_the_hand_written_one(sdk_tools, name):
    """The decorator derives the same parameters we wrote out by hand.

    This is the point of the whole dual implementation. TOOL_SCHEMAS in
    tools.py was typed out manually; the SDK built its version from type
    hints and the docstring. The parameter names agree.
    """
    handwritten = next(
        s["function"] for s in tools.TOOL_SCHEMAS if s["function"]["name"] == name
    )
    generated = sdk_tools[name].params_json_schema

    assert set(generated["properties"]) == set(handwritten["parameters"]["properties"])


def test_generated_descriptions_come_from_the_docstring(sdk_tools):
    """Docstrings are not decoration here — they are shipped to the model."""
    schema = sdk_tools["assign_department"].params_json_schema
    assert "billing" in schema["properties"]["department"]["description"]
    assert sdk_tools["assign_department"].description.startswith("Propose assigning")


def test_strict_mode_puts_optional_params_in_required(sdk_tools):
    """A real difference between the two schemas, worth understanding.

    Our hand-written schema lists required: []. The SDK lists BOTH
    parameters as required, because OpenAI strict function calling
    demands every property appear there. Optionality is expressed with a
    "default" key instead. Same meaning, different encoding.
    """
    generated = sdk_tools["search_tickets"].params_json_schema
    assert set(generated["required"]) == {"customer_email", "status"}
    assert generated["properties"]["customer_email"]["default"] == ""
    assert generated["additionalProperties"] is False


def test_session_is_file_backed_so_memory_can_be_inspected(tmp_path):
    from helpdesk.config import load_settings

    settings = load_settings({"CHAT_DB_PATH": str(tmp_path / "chat.db")})
    session = sdk_agent.build_session(settings, "test-session")
    assert str(tmp_path / "chat.db") in str(session.db_path)


def test_clear_memory_actually_clears_the_session(tmp_path):
    """Guards against the un-awaited-coroutine bug.

    SQLiteSession.clear_session() is async. Calling it as a bare statement
    (no await, no asyncio.run) builds a coroutine object and discards it
    without ever running it — the session silently keeps every item. This
    test would have caught that: it fails if clear_memory() stops routing
    through _sync() and goes back to calling clear_session() bare.
    """
    from helpdesk.config import load_settings

    settings = load_settings({"CHAT_DB_PATH": str(tmp_path / "chat.db")})
    session = sdk_agent.build_session(settings, "test-session")

    asyncio.run(session.add_items([{"role": "user", "content": "hello"}]))
    assert asyncio.run(session.get_items()) != []

    sdk_agent.clear_memory(session)

    assert asyncio.run(session.get_items()) == []
