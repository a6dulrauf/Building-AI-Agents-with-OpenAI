"""The same agent, built with the OpenAI Agents SDK.

READ raw_agent.py FIRST. Then read this and notice what disappeared.

The while-loop is gone. Runner.run() contains it. The tool dispatch is
gone — the SDK matches the model's requested name to a decorated function
and calls it. The messages list is gone; SQLiteSession keeps it.

What did NOT disappear is the part that matters: the tools themselves are
the exact same functions from tools.py. function_tool() only wraps them
with a generated schema. Nothing about them became "AI".

The one genuine capability gain over the hand-written loop is
needs_approval=True. Doing that by hand meant a set of tool names and an
if-statement (see raw_agent._execute). Here it is one argument, and the
SDK additionally gives us a resumable RunState so the pause can survive a
round trip through a web UI.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import Agent, FunctionTool, RunResult, Runner, RunState, SQLiteSession, function_tool

from helpdesk import tools
from helpdesk.config import Settings
from helpdesk.prompts import SYSTEM_PROMPT
from helpdesk.providers import build_model

AGENT_NAME = "Helpdesk Support Agent"


def build_tools() -> list[FunctionTool]:
    """Wrap the plain functions from tools.py as SDK tools.

    The schema for each is generated from its type hints and docstring.
    Compare the result against tools.TOOL_SCHEMAS — tests/test_sdk_agent.py
    asserts they agree.
    """
    return [
        function_tool(tools.get_ticket),
        function_tool(tools.search_tickets),
        # Writes pause the run until a human decides. This one argument
        # replaces the WRITE_TOOLS check in raw_agent._execute.
        function_tool(needs_approval=True)(tools.assign_department),
        function_tool(needs_approval=True)(tools.add_note),
    ]


def build_agent(settings: Settings) -> Agent:
    """Assemble the agent: instructions, model, and tools."""
    return Agent(
        name=AGENT_NAME,
        instructions=SYSTEM_PROMPT,
        model=build_model(settings),
        tools=build_tools(),
    )


def build_session(settings: Settings, session_id: str) -> SQLiteSession:
    """Create file-backed conversation memory.

    The db_path argument is what makes this inspectable. Without it the
    SDK defaults to ":memory:" and the history vanishes on restart. With
    it you can run:

        sqlite3 data/chat.db "SELECT * FROM agent_messages LIMIT 5;"

    and see your short-term memory as rows. Being able to look at it is
    worth the one line.
    """
    return SQLiteSession(session_id, settings.chat_db_path)


def send(agent: Agent, session: SQLiteSession, message: str, max_turns: int) -> RunResult:
    """Send a user message and run until the agent finishes or pauses.

    Returns a RunResult. If result.interruptions is non-empty the run is
    PAUSED waiting on human approval — call resume() once decisions are
    recorded.
    """
    return _sync(Runner.run(agent, message, session=session, max_turns=max_turns))


def resume(agent: Agent, session: SQLiteSession, state: RunState) -> RunResult:
    """Continue a paused run after approve()/reject() have been called."""
    return _sync(Runner.run(agent, state, session=session))


def describe_interruption(item: Any) -> tuple[str, dict]:
    """Return (tool_name, arguments) for a pending approval, for display.

    ToolApprovalItem exposes .name and .arguments as properties;
    .arguments is a JSON string, which we parse so the UI can show it
    readably.
    """
    try:
        arguments = json.loads(item.arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        arguments = {}
    return item.name or "unknown_tool", arguments


def _sync(coro):
    """Run a coroutine from synchronous code.

    The SDK is async; Streamlit and our tests are not. asyncio.run() is
    the bridge. It creates a fresh event loop each call, which is fine at
    this scale and keeps the calling code readable.
    """
    return asyncio.run(coro)
