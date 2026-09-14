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
# "tool_call", "tool_result", "rejected", "text_tool_call".
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
    nudged = False          # we correct a typed-out tool call once, not forever

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
        entry = _as_dict(message)

        # --- 2. No tool calls means the model is done -------------------
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            ghost = _text_tool_call_name(message.content)
            if ghost and not nudged:
                # The model meant to act but used the wrong channel. Store
                # its prose WITHOUT the fake call so the history stops
                # teaching it the wrong shape, then say so and let it try
                # again. Bounded by max_turns, and only once per run: a
                # model that ignores the correction twice will not be
                # talked round on the third attempt.
                emit("text_tool_call", {"name": ghost, "retrying": True})
                entry["content"] = _strip_text_tool_call(message.content)
                messages.append(entry)
                messages.append({
                    "role": "user",
                    "content": _RETRY_NUDGE.format(name=ghost),
                })
                nudged = True
                continue
            if ghost:
                emit("text_tool_call", {"name": ghost, "retrying": False})
                entry["content"] = _strip_text_tool_call(message.content)
            messages.append(entry)
            return message.content or ""

        messages.append(entry)

        # --- 3. Run each requested tool and feed the result back --------
        for call in tool_calls:
            name = call.function.name
            # None here means "the model did not send valid JSON", which is
            # a different failure from "the model sent {}". _execute needs
            # to tell them apart; the UI only needs something printable.
            arguments = _parse_arguments(call.function.arguments)
            emit("tool_call", {"name": name, "arguments": arguments or {}})

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
    name: str, arguments: dict | None, approve: ApproveFn, emit: EventFn
) -> str:
    """Run one tool, asking for approval first if it writes."""
    if arguments is None:
        # The model's arguments were not valid JSON. Say exactly that.
        # Falling through with {} instead would call the tool with no
        # arguments, and the model would be told "get_ticket failed
        # unexpectedly: get_ticket() missing 1 required positional
        # argument" — a message that blames the tool for the model's
        # mistake and gives it nothing it can act on. Every error string
        # in this project is written to be fixable by its reader.
        return (
            f"error: your arguments for {name} were not valid JSON. "
            "Send a JSON object matching the tool's parameters."
        )

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


def _text_tool_call_name(content: str | None) -> str | None:
    """Spot a tool call the model WROTE OUT instead of actually calling.

    Small models sometimes describe the call in the message body —

        {"name": "assign_department", "parameters": {...}}

    — rather than using the structured tool_calls channel. The loop then
    sees no tool call, ends the turn, and the operator gets prose with no
    approval prompt and no explanation of why nothing happened.

    We deliberately do NOT execute what we find. Text the model happened
    to type is not a request: honouring it would bypass the structured
    contract the whole approval design rests on, and would mean a model
    could trigger a write just by talking about one. We only name it, so
    the human knows why the turn went quiet.
    """
    if not content or "{" not in content:
        return None
    looks_structured = any(
        marker in content for marker in ('"name"', '"parameters"', '"arguments"')
    )
    if not looks_structured:
        return None
    return next((name for name in TOOL_FUNCTIONS if name in content), None)


def _strip_text_tool_call(content: str | None) -> str:
    """Return the model's prose with the typed-out tool call removed.

    Storing the raw message is actively harmful: measured against
    llama3.1, one assistant message containing a text tool call drops the
    next turn's real tool-call rate from 4/6 to 1/6. The model reads its
    own prior output as an example and imitates the wrong format, and the
    conversation never recovers.

    So we keep what the model said and drop the malformed artifact. The
    model still sees its reasoning; it just stops being taught the wrong
    shape by its own history.
    """
    if not content:
        return ""

    index = content.find("{")
    while index != -1 and '"name"' not in content[index:]:
        index = content.find("{", index + 1)
    if index == -1:
        return content.strip()

    cleaned = content[:index].rstrip()
    for fence in ("```json", "```"):          # models often wrap it in one
        if cleaned.endswith(fence):
            cleaned = cleaned[: -len(fence)].rstrip()
    return cleaned


# Sent as a "user" message rather than "system" on purpose: small models
# attend to user turns far more reliably than to mid-conversation system
# turns, and this nudge is worthless if it gets ignored.
_RETRY_NUDGE = (
    "You wrote a {name} call as text in your last reply. Text is not a "
    "tool call — nothing ran. Use the tool-calling interface to call "
    "{name}, and put no JSON in your message."
)


def _parse_arguments(raw: str) -> dict | None:
    """Parse the model's arguments. Return None if they are not a JSON object.

    Models do occasionally emit broken JSON, so this must not raise. But
    it must not quietly return {} either: an empty dict is a legitimate
    call ("no arguments"), and collapsing the two cases together makes the
    loop report a parse failure as a missing-argument bug inside the tool.
    None means "unusable"; the caller turns it into an error the model can
    actually act on.

    Valid JSON that is not an object (a list, a bare number) is unusable
    too — func(**arguments) needs a mapping — so it takes the same path.
    """
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_dict(message) -> dict:
    """Convert the client's message object into a plain dict for the transcript.

    The client hands back a pydantic model; `messages` is a plain list of
    dicts that gets re-sent verbatim on the next call, so the object has to
    be flattened. exclude_none keeps the fields the model never set —
    function_call, audio, refusal — out of the dict entirely instead of
    echoing explicit nulls back to the API.

    There is no fallback for an object without model_dump. A fallback that
    returned just role and content would drop tool_calls, and the
    role:"tool" message appended straight after would then carry a
    tool_call_id the API never saw — a 400 in place of a clean crash.
    """
    return message.model_dump(exclude_none=True)
