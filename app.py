"""Streamlit UI for the helpdesk agent.

Two things this UI does that a plain chat box would not, both of which
exist because an agent is not a chatbot:

1. It shows every tool call and its result. Without that the agent is a
   black box that silently reads and edits a database. With it, the
   support team can audit what happened.

2. It blocks writes behind an Approve/Reject decision. The agent proposes;
   a person decides. Rejecting sends a message back to the model, which
   then reconsiders rather than repeating itself.

Streamlit note: this whole file re-runs top to bottom on every click.
State that must survive lives in st.session_state; expensive objects are
built once via @st.cache_resource.

Run with:  .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from helpdesk.config import ConfigError, load_settings
from helpdesk.sdk_agent import (
    build_agent,
    build_session,
    clear_memory,
    describe_interruption,
    resume,
    send,
)

st.set_page_config(page_title="Helpdesk Support Agent", page_icon="🎧", layout="centered")

SESSION_ID = "streamlit-support-desk"


@st.cache_resource
def get_runtime():
    """Build settings, agent, and session once and reuse them.

    Without @st.cache_resource this would rebuild an HTTP client and
    reopen the database on every keystroke.
    """
    settings = load_settings()
    agent = build_agent(settings)
    session = build_session(settings, SESSION_ID)
    return settings, agent, session


def init_state() -> None:
    st.session_state.setdefault("transcript", [])   # what to render
    st.session_state.setdefault("pending", None)    # paused RunResult
    st.session_state.setdefault("decisions", {})    # call_id -> bool


def render_tool_activity(result) -> list[dict]:
    """Extract tool calls and results from a run, for display.

    result.new_items holds everything that happened during the run. We
    pick out the tool calls and their outputs so the UI can show the
    agent's work rather than only its conclusion.
    """
    activity: list[dict] = []
    for item in getattr(result, "new_items", []):
        kind = getattr(item, "type", "")
        if kind == "tool_call_item":
            name = getattr(item, "name", None) or getattr(
                getattr(item, "raw_item", None), "name", "tool"
            )
            args = getattr(getattr(item, "raw_item", None), "arguments", "")
            activity.append({"kind": "call", "name": name, "arguments": args})
        elif kind == "tool_call_output_item":
            activity.append({"kind": "output", "output": str(getattr(item, "output", ""))})
    return activity


def show_activity(activity: list[dict]) -> None:
    if not activity:
        return
    with st.expander(f"🔧 {len(activity)} tool events", expanded=True):
        for entry in activity:
            if entry["kind"] == "call":
                st.code(f"{entry['name']}({entry['arguments']})", language="python")
            else:
                output = entry["output"]
                if output.startswith("error:"):
                    st.error(output)
                else:
                    st.text(output[:600])


def record(role: str, content: str, activity: list[dict] | None = None) -> None:
    st.session_state.transcript.append(
        {"role": role, "content": content, "activity": activity or []}
    )


def handle_result(result) -> None:
    """Store a run's output, or park it if it is waiting on approval."""
    activity = render_tool_activity(result)

    if result.interruptions:
        st.session_state.pending = result
        st.session_state.decisions = {}
        record("assistant", "_Waiting for your approval below._", activity)
    else:
        st.session_state.pending = None
        record("assistant", result.final_output or "", activity)


def apply_decisions(agent, session) -> None:
    """Approve or reject each pending call, then resume the run."""
    result = st.session_state.pending
    state = result.to_state()

    for item in result.interruptions:
        approved = st.session_state.decisions.get(item.call_id, False)
        if approved:
            state.approve(item)
        else:
            # The rejection message is sent to the model, so it knows the
            # action did not happen and can propose something else.
            state.reject(
                item,
                rejection_message=(
                    "The support agent rejected this action. Do not retry it. "
                    "Suggest an alternative or ask what they would prefer."
                ),
            )

    st.session_state.pending = None
    st.session_state.decisions = {}
    handle_result(resume(agent, session, state))


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
init_state()

try:
    settings, agent, session = get_runtime()
except ConfigError as exc:
    st.error(f"Configuration problem: {exc}")
    st.stop()

st.title("🎧 Helpdesk Support Agent")
st.caption(f"Running on **{settings.provider}** / `{settings.model}`")

with st.sidebar:
    st.header("Session")
    st.write(f"Provider: `{settings.provider}`")
    st.write(f"Model: `{settings.model}`")
    st.write(f"Memory: `{settings.chat_db_path}`")

    if st.button("Clear conversation", use_container_width=True):
        # Clears BOTH the rendered transcript and the agent's actual
        # memory. Forgetting the second is a classic bug: the screen looks
        # empty but the agent still remembers everything. clear_memory()
        # (not session.clear_session() directly) is what actually awaits
        # the coroutine instead of silently discarding it.
        clear_memory(session)
        st.session_state.transcript = []
        st.session_state.pending = None
        st.session_state.decisions = {}
        st.rerun()

    st.divider()
    st.caption(
        "Reads run automatically. Writes (assign_department, add_note) "
        "pause for your approval."
    )
    st.caption("Try: *Look at T-1006 and suggest a department.*")

for entry in st.session_state.transcript:
    with st.chat_message(entry["role"]):
        show_activity(entry["activity"])
        st.markdown(entry["content"])

# --- Pending approvals ----------------------------------------------------
if st.session_state.pending is not None:
    st.warning("The agent is proposing an action. Approve or reject it.")
    interruptions = st.session_state.pending.interruptions
    for item in interruptions:
        name, arguments = describe_interruption(item)
        st.markdown(f"**Proposed:** `{name}`")
        st.json(arguments)

        if item.call_id in st.session_state.decisions:
            # Already decided this rerun, but the batch isn't resolved yet
            # (see below) - show the choice as settled instead of
            # re-rendering live buttons the user could click again.
            if st.session_state.decisions[item.call_id]:
                st.caption("✅ approved — waiting for the remaining decisions")
            else:
                st.caption("❌ rejected — waiting for the remaining decisions")
            continue

        left, right = st.columns(2)
        approve_clicked = left.button(
            "✅ Approve", key=f"a-{item.call_id}", use_container_width=True
        )
        reject_clicked = right.button(
            "❌ Reject", key=f"r-{item.call_id}", use_container_width=True
        )
        if approve_clicked or reject_clicked:
            st.session_state.decisions[item.call_id] = approve_clicked
            # The SDK resumes the run once, with every interruption
            # decided. Resolving as soon as ANY single item is clicked
            # would silently reject whatever the user had not yet
            # clicked, so wait until every pending item has a decision.
            if len(st.session_state.decisions) == len(interruptions):
                apply_decisions(agent, session)
            st.rerun()

# --- Chat input -----------------------------------------------------------
elif prompt := st.chat_input("Ask about a ticket, e.g. 'triage T-1006'"):
    record("user", prompt)
    with st.spinner("Thinking..."):
        try:
            handle_result(send(agent, session, prompt, settings.max_turns))
        except Exception as exc:  # noqa: BLE001 - surface errors instead of a blank page
            record("assistant", f"Something went wrong: `{exc}`")
    st.rerun()
