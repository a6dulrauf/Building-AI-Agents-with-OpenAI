"""The agent loop, written by hand.

THIS IS THE FILE TO READ FIRST.

An agent is an LLM in a loop with tools. Everything else is engineering
around those three parts, and all three are visible below in about sixty
lines:

  1. Send the conversation plus the tool schemas to the model.
  2. If the model asked for no tools, it has finished — return its answer.
  3. Otherwise run each requested tool, append the results, and go again.

The single most important thing to notice: THE MODEL NEVER EXECUTES
ANYTHING. It returns JSON describing a function it would like called.
Line "result = func(**arguments)" below is our code choosing to run it.
The agency lives in this loop, not in the model.

Once this is clear, read sdk_agent.py. It does exactly this, and you will
be able to point at what the framework replaced.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from helpdesk.prompts import SYSTEM_PROMPT
from helpdesk.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, WRITE_TOOLS

# Signature: approve(tool_name, arguments_dict) -> bool
ApproveFn = Callable[[str, dict[str, Any]], bool]
# Signature: on_event(kind, payload) -> None, where kind is one of
# "thinking", "tool_call", "tool_result", "rejected".
EventFn = Callable[[str, dict[str, Any]], None]


def new_conversation() -> list[dict]:
    """Start a conversation: a list containing just the system prompt.

    This list IS the agent's short-term memory. Not a database, not a
    framework abstraction — a Python list that grows by one or two entries
    per turn and gets re-sent in full on every API call.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def run_conversation(
    client,
    model: str,
    messages: list[dict],
    *,
    approve: ApproveFn,
    on_event: EventFn | None = None,
    max_turns: int = 8,
) -> str:
    """Run the agent loop until the model stops asking for tools.

    Args:
        client: A synchronous OpenAI-compatible client.
        model: Model name to call.
        messages: The conversation. MUTATED IN PLACE — after this returns,
            it holds the full transcript including tool calls and results.
        approve: Called before any write tool runs. Return False to refuse.
        on_event: Optional observer, used by the UI to display progress.
        max_turns: Hard cap. Without it a confused model loops forever.

    Returns:
        The model's final text answer, or a turn-limit message.
    """
    emit = on_event or (lambda kind, payload: None)

    for _turn in range(max_turns):
        # --- 1. Ask the model what to do next ---------------------------
        # Note that `messages` is sent in full every single time. The model
        # is stateless; the conversation only exists because we keep
        # re-sending it. That is all "short-term memory" means.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )
        message = response.choices[0].message
        messages.append(_as_dict(message))

        # --- 2. No tool calls means the model is done -------------------
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return message.content or ""

        # --- 3. Run each requested tool and feed the result back --------
        for call in tool_calls:
            name = call.function.name
            arguments = _parse_arguments(call.function.arguments)
            emit("tool_call", {"name": name, "arguments": arguments})

            result = _execute(name, arguments, approve, emit)

            emit("tool_result", {"name": name, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })
        # ...and loop, so the model can see the results and decide again.

    return (
        f"Stopped after reaching the turn limit of {max_turns}. "
        "The task may be incomplete."
    )


def _execute(
    name: str, arguments: dict, approve: ApproveFn, emit: EventFn
) -> str:
    """Run one tool, asking for approval first if it writes."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        # The model hallucinated a tool. Tell it so rather than crashing —
        # it will pick a real one on the next turn.
        return f"error: there is no tool named {name!r}."

    if name in WRITE_TOOLS and not approve(name, arguments):
        emit("rejected", {"name": name, "arguments": arguments})
        # The rejection goes back as a tool result, so the model learns the
        # action did not happen and can propose something else.
        return (
            f"error: the human operator rejected {name}. "
            "Do not retry it; suggest an alternative or ask what they want."
        )

    return func(**arguments)


def _parse_arguments(raw: str) -> dict:
    """Parse the model's JSON arguments, tolerating malformed output."""
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def _as_dict(message) -> dict:
    """Convert the SDK's message object into a plain dict for the transcript."""
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return {"role": "assistant", "content": getattr(message, "content", "")}
