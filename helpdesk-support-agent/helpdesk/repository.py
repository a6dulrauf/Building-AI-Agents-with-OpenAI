"""Plain SQL over the ticket database.

The rule for this file: it never imports an LLM library and never returns
a string meant for a model to read. It deals in dicts and booleans. All
the "speak to the model" concerns live one layer up in tools.py.

Keeping that boundary is what makes test_repository.py runnable with no
API key — and it is the single clearest argument for the layering.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_ticket(conn: sqlite3.Connection, ticket_id: str) -> dict | None:
    """Return one ticket as a dict, or None if no such ticket exists."""
    row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    return dict(row) if row else None


def get_notes(conn: sqlite3.Connection, ticket_id: str) -> list[dict]:
    """Return every note on a ticket, oldest first."""
    rows = conn.execute(
        "SELECT * FROM notes WHERE ticket_id = ? ORDER BY id", (ticket_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def search_tickets(
    conn: sqlite3.Connection,
    customer_email: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Return tickets matching the given filters, newest first.

    Filters are optional and combine with AND. The WHERE clause is built
    from a list of fragments so that parameters stay bound — never
    interpolate user input into SQL, even when it came from an LLM.
    """
    clauses: list[str] = []
    params: list[str] = []

    if customer_email:
        clauses.append("customer_email = ?")
        params.append(customer_email)
    if status:
        clauses.append("status = ?")
        params.append(status)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM tickets {where} ORDER BY created_at DESC", params
    ).fetchall()
    return [dict(r) for r in rows]


def assign_department(conn: sqlite3.Connection, ticket_id: str, department: str) -> bool:
    """Set a ticket's department. Returns False if the ticket does not exist.

    Returning a bool rather than raising keeps this layer boring: the
    caller in tools.py decides what a missing ticket should say to the
    model.
    """
    cur = conn.execute(
        "UPDATE tickets SET department = ? WHERE id = ?", (department, ticket_id)
    )
    conn.commit()
    return cur.rowcount > 0


def add_note(
    conn: sqlite3.Connection, ticket_id: str, note: str, author: str = "agent"
) -> bool:
    """Append a note to a ticket. Returns False if the ticket does not exist.

    We check the ticket exists first rather than relying on the foreign key
    to raise, so the caller gets the same False-means-missing contract as
    assign_department.
    """
    if get_ticket(conn, ticket_id) is None:
        return False
    conn.execute(
        "INSERT INTO notes (ticket_id, author, note, created_at) VALUES (?, ?, ?, ?)",
        (ticket_id, author, note, _now()),
    )
    conn.commit()
    return True
