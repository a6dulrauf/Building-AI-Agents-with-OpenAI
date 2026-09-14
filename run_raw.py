"""Terminal entry point for the hand-written loop.

Run this first, before the Streamlit app. It prints every turn — what the
model asked for, what came back — so the loop is something you watch
rather than something you take on faith.

    .venv/bin/python run_raw.py
"""
from __future__ import annotations

import json

from helpdesk.config import ConfigError, load_settings
from helpdesk.providers import build_sync_client
from helpdesk.raw_agent import new_conversation, run_conversation

GREY, GREEN, YELLOW, RED, RESET = "\033[90m", "\033[92m", "\033[93m", "\033[91m", "\033[0m"


def show(kind: str, payload: dict) -> None:
    """Print loop events as they happen."""
    if kind == "tool_call":
        args = json.dumps(payload["arguments"])
        print(f"{YELLOW}  -> calling {payload['name']}({args}){RESET}")
    elif kind == "tool_result":
        # Tool results are pretty-printed (multi-line) JSON, so the raw
        # first line is just an opening "{" — that told the operator
        # nothing while 300+ characters of actual ticket data sat unseen
        # on the following lines. Collapse whitespace across the whole
        # result into one line first, then take a useful prefix of that.
        # str.split()/" ".join() also handles an empty result safely
        # (collapsed == ""), so no separate empty-result guard is needed.
        result = payload["result"]
        collapsed = " ".join(result.split())
        limit = 150
        preview = collapsed[:limit] + ("..." if len(collapsed) > limit else "")
        colour = RED if result.startswith("error:") else GREY
        print(f"{colour}  <- {preview}{RESET}")
    elif kind == "text_tool_call" and payload.get("retrying"):
        print(f"{YELLOW}  !! the model wrote a {payload['name']} call as text "
              f"instead of calling it.{RESET}")
        print(f"{YELLOW}     Removing it from the transcript and asking it to "
              f"retry properly...{RESET}")
    elif kind == "text_tool_call":
        # The model described a call instead of making one. Say so plainly —
        # otherwise the turn just ends with prose and no approval prompt,
        # and there is nothing on screen explaining why nothing happened.
        print(f"{RED}  !! the model WROTE a {payload['name']} call as text "
              f"instead of calling it.{RESET}")
        print(f"{RED}     No tool ran, so there was nothing to approve. "
              f"This is a known{RESET}")
        print(f"{RED}     limitation of small models. Rephrase and retry, "
              f"or use a larger one.{RESET}")
    elif kind == "rejected":
        print(f"{RED}  <- rejected by you{RESET}")


def ask_approval(name: str, arguments: dict) -> bool:
    """Prompt the operator before a write happens.

    This is the hand-rolled version of what needs_approval=True does for
    you in sdk_agent.py. Seeing it as an ordinary if-statement first is
    what stops the SDK version from looking like magic.
    """
    print(f"{YELLOW}  ?? {name}({json.dumps(arguments)}){RESET}")
    return input("     approve? [y/N]: ").strip().lower() in {"y", "yes"}


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"{RED}Configuration problem: {exc}{RESET}")
        return

    client = build_sync_client(settings)
    messages = new_conversation()

    print(f"Helpdesk agent — {settings.provider}/{settings.model}")
    print("Try: 'Look at T-1006 and tell me which department it belongs to.'")
    print("Ctrl-C to quit.\n")

    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        answer = run_conversation(
            client,
            settings.model,
            messages,
            approve=ask_approval,
            on_event=show,
            max_turns=settings.max_turns,
        )
        print(f"{GREEN}agent: {answer}{RESET}\n")
        print(f"{GREY}  [memory: {len(messages)} messages]{RESET}\n")


if __name__ == "__main__":
    main()
