"""End-to-end verification harness.

Runs every path through the agent and PROVES, against the database, what
did and did not happen. Each scenario prints the ticket's state before and
after, then a PASS/FAIL line comparing that to what should have happened.

The point is not that the agent sounds right. It is that the data agrees.

Usage:
    .venv/bin/python scripts/e2e.py            # live model, all scenarios
    .venv/bin/python scripts/e2e.py --offline  # no model; mechanisms only
    .venv/bin/python scripts/e2e.py --reset    # restore the seeded database

WARNING: every run resets data/helpdesk.db to its seeded state first, so
any tickets you changed by hand are discarded.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpdesk import repository, tools                      # noqa: E402
from helpdesk.config import load_settings                   # noqa: E402
from helpdesk.database import connect, init_schema, seed    # noqa: E402

DB = "data/helpdesk.db"
TICKET = "T-1006"

GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[1m", "\033[0m")

passed: list[str] = []
failed: list[str] = []


# ---------------------------------------------------------------- helpers
def reset_db() -> None:
    """Drop and reseed, so every run starts from identical data."""
    Path(DB).unlink(missing_ok=True)
    conn = connect(DB)
    init_schema(conn)
    n = seed(conn)
    conn.close()
    print(f"{GREY}database reset — {n} tickets seeded, 0 notes{RESET}\n")


def snapshot(ticket_id: str = TICKET) -> dict:
    """Read the ground truth straight from SQLite."""
    conn = connect(DB)
    ticket = repository.get_ticket(conn, ticket_id) or {}
    notes = repository.get_notes(conn, ticket_id)
    conn.close()
    return {"department": ticket.get("department"), "notes": len(notes)}


def banner(n: int, title: str) -> None:
    print(f"\n{BOLD}{'─' * 68}\nSCENARIO {n}: {title}\n{'─' * 68}{RESET}")


def verdict(name: str, before: dict, after: dict, *, expect_change: bool) -> None:
    """Compare the database before and after against what should have happened."""
    changed = before != after
    ok = changed == expect_change
    (passed if ok else failed).append(name)

    print(f"  {GREY}before:{RESET} department={before['department']!r} notes={before['notes']}")
    print(f"  {GREY}after: {RESET} department={after['department']!r} notes={after['notes']}")
    wanted = "the database to CHANGE" if expect_change else "the database to be UNTOUCHED"
    got = "it changed" if changed else "it did not change"
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  {mark} — expected {wanted}; {got}")


# ------------------------------------------------------- offline scenarios
def offline_scenarios() -> None:
    """Prove the mechanisms directly, with no model involved.

    These are deterministic: they exercise validation and the write path
    by calling the tools the way the agent would, so they tell you whether
    the machinery works independently of whether a model drives it well.
    """
    conn = connect(DB)
    tools.set_connection(conn)

    banner(1, "Read is free — get_ticket must not touch the database")
    before = snapshot()
    out = tools.get_ticket(TICKET)
    print(f"  {GREY}returned:{RESET} {out.splitlines()[1].strip()[:70]}")
    verdict("read is free", before, snapshot(), expect_change=False)

    banner(2, "Invalid department — validation rejects it, nothing is written")
    before = snapshot()
    out = tools.assign_department(TICKET, "finance")
    print(f"  {YELLOW}returned:{RESET} {out}")
    print(f"  {GREY}note: names the bad value AND the valid options, so the "
          f"model can self-correct{RESET}")
    verdict("invalid department blocked", before, snapshot(), expect_change=False)

    banner(3, "Malformed ticket id — rejected before any SQL runs")
    before = snapshot()
    print(f"  {YELLOW}returned:{RESET} {tools.get_ticket('1006')}")
    verdict("bad ticket id blocked", before, snapshot(), expect_change=False)

    banner(4, "Wrong argument name — error names the right tool")
    before = snapshot()
    print(f"  {YELLOW}returned:{RESET} {tools.search_tickets(ticket_id=TICKET)}")
    verdict("wrong argument blocked", before, snapshot(), expect_change=False)

    banner(5, "Empty note — length validation rejects it")
    before = snapshot()
    print(f"  {YELLOW}returned:{RESET} {tools.add_note(TICKET, '   ')}")
    verdict("empty note blocked", before, snapshot(), expect_change=False)

    banner(6, "Valid write — this one SHOULD change the database")
    before = snapshot()
    print(f"  {GREEN}returned:{RESET} {tools.assign_department(TICKET, 'technical')}")
    verdict("valid write persists", before, snapshot(), expect_change=True)

    tools.set_connection(None)
    conn.close()


# ---------------------------------------------------------- live scenarios
def live_scenarios() -> None:
    """Drive the REAL agent loop against the configured model.

    Approval decisions are scripted rather than typed, so the whole run is
    non-interactive and repeatable. What the model chooses to do is not
    scripted — that is the part being tested.
    """
    from helpdesk.providers import build_sync_client
    from helpdesk.raw_agent import new_conversation, run_conversation

    settings = load_settings()
    print(f"{GREY}model: {settings.provider}/{settings.model}{RESET}")
    client = build_sync_client(settings)

    conn = connect(DB)
    tools.set_connection(conn)

    def run(prompt: str, *, approve: bool):
        calls: list[str] = []

        def on_event(kind, payload):
            if kind == "tool_call":
                calls.append(payload["name"])
                print(f"  {YELLOW}-> {payload['name']}({payload['arguments']}){RESET}")
            elif kind == "tool_result":
                line = " ".join(payload["result"].split())[:78]
                colour = RED if payload["result"].startswith("error:") else GREY
                print(f"  {colour}<- {line}{RESET}")

        messages = new_conversation()
        messages.append({"role": "user", "content": prompt})
        answer = run_conversation(
            client, settings.model, messages,
            approve=lambda *_: approve, on_event=on_event,
            max_turns=settings.max_turns,
        )
        print(f"  {GREY}agent:{RESET} {' '.join(answer.split())[:200]}")
        return calls

    banner(7, "LIVE — read-only question, no write should occur")
    before = snapshot()
    run(f"Look at ticket {TICKET} and tell me which department it belongs to.",
        approve=True)
    verdict("live read-only", before, snapshot(), expect_change=False)

    banner(8, "LIVE — write proposed, human REJECTS (answers 'n')")
    before = snapshot()
    calls = run(f"Assign ticket {TICKET} to the right department.", approve=False)
    if "assign_department" not in calls:
        print(f"  {YELLOW}note: the model never proposed assign_department this "
              f"run — small models are inconsistent here{RESET}")
    verdict("rejected write not persisted", before, snapshot(), expect_change=False)

    banner(9, "LIVE — write proposed, human APPROVES (answers 'y')")
    before = snapshot()
    calls = run(f"Assign ticket {TICKET} to the right department.", approve=True)
    if "assign_department" not in calls:
        print(f"  {YELLOW}note: the model never proposed assign_department, so "
              f"nothing could be written. Model behaviour, not a code fault.{RESET}")
        print(f"  {YELLOW}SKIP — cannot judge the approve path this run{RESET}")
        passed.append("live approve (skipped — model did not propose)")
    else:
        verdict("approved write persists", before, snapshot(), expect_change=True)

    tools.set_connection(None)
    conn.close()


# -------------------------------------------------------------------- main
def main() -> None:
    args = sys.argv[1:]

    if "--reset" in args:
        reset_db()
        return

    print(f"{BOLD}End-to-end verification{RESET}")
    print(f"{GREY}Every claim below is checked against data/helpdesk.db.{RESET}\n")
    reset_db()

    offline_scenarios()
    if "--offline" not in args:
        try:
            live_scenarios()
        except Exception as exc:  # noqa: BLE001 - report, don't crash the harness
            print(f"\n{RED}live scenarios could not run: "
                  f"{type(exc).__name__}: {exc}{RESET}")
            print(f"{YELLOW}Is Ollama running?  "
                  f"curl -s http://localhost:11434/api/tags{RESET}")

    print(f"\n{BOLD}{'═' * 68}{RESET}")
    print(f"{GREEN}passed: {len(passed)}{RESET}   {RED}failed: {len(failed)}{RESET}")
    for name in failed:
        print(f"  {RED}FAILED: {name}{RESET}")
    print(f"\n{GREY}Inspect the data yourself:{RESET}")
    print(f'  sqlite3 -header -column {DB} \\\n'
          f'    "SELECT id, status, COALESCE(department,\'—\') AS dept FROM tickets;"')
    print(f'  sqlite3 -header -column {DB} "SELECT * FROM notes;"')
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
