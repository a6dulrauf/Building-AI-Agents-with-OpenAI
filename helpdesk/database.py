"""SQLite connection, schema creation, and seed data.

The bottom layer of the project. It knows about tables and nothing else —
no LLM imports appear here or in repository.py, which is what lets the
test suite run offline in about a second.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEPARTMENTS: tuple[str, ...] = ("billing", "technical", "account", "shipping")
STATUSES: tuple[str, ...] = ("open", "pending", "resolved")

SCHEMA_PATH = Path(__file__).parent.parent / "data" / "schema.sql"


def connect(db_path: str) -> sqlite3.Connection:
    """Open a connection with dict-style row access and FK enforcement."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # Without this, rows are plain tuples and you index them by position —
    # which breaks silently the day someone adds a column.
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not already exist."""
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


def seed(conn: sqlite3.Connection) -> int:
    """Insert sample tickets. Returns how many rows were inserted.

    Idempotent: returns 0 if the table already has rows, so running it
    twice does not duplicate data.
    """
    existing = conn.execute("SELECT COUNT(*) AS n FROM tickets").fetchone()["n"]
    if existing:
        return 0

    base = datetime.now(timezone.utc) - timedelta(days=30)

    def ts(days: int) -> str:
        return (base + timedelta(days=days)).isoformat()

    # aisha@example.com appears four times on purpose — ticket-history
    # lookups need a customer with a history to be worth demonstrating.
    rows = [
        ("T-1001", "aisha@example.com", "Double charged for March",
         "I was billed twice on 3 March for the same subscription.",
         "resolved", "high", "billing", ts(0)),
        ("T-1002", "aisha@example.com", "Refund still not received",
         "The refund for the duplicate charge has not arrived after 10 days.",
         "resolved", "high", "billing", ts(4)),
        ("T-1003", "omar@example.com", "Cannot log in after password reset",
         "I reset my password but the new one is rejected at login.",
         "resolved", "medium", "account", ts(6)),
        ("T-1004", "priya@example.com", "Package arrived damaged",
         "The box was crushed and the contents are broken.",
         "resolved", "high", "shipping", ts(8)),
        ("T-1005", "aisha@example.com", "Invoice shows wrong VAT rate",
         "My invoice applies 20% VAT but our account is registered as exempt.",
         "pending", "medium", "billing", ts(11)),
        ("T-1006", "james@example.com", "App crashes on export to PDF",
         "Clicking Export to PDF closes the app immediately. Version 4.2.1.",
         "open", "urgent", None, ts(14)),
        ("T-1007", "omar@example.com", "Two-factor codes never arrive",
         "SMS codes for 2FA are not being delivered to my phone.",
         "open", "high", None, ts(17)),
        ("T-1008", "priya@example.com", "Where is my order 88231?",
         "Order 88231 has shown 'in transit' for nine days with no updates.",
         "open", "medium", None, ts(19)),
        ("T-1009", "lucia@example.com", "Please cancel my subscription",
         "I would like to cancel and confirm I will not be charged again.",
         "open", "medium", None, ts(21)),
        ("T-1010", "james@example.com", "Export produces an empty file",
         "CSV export downloads a 0-byte file. Started after the last update.",
         "open", "high", None, ts(24)),
        ("T-1011", "aisha@example.com", "Cannot update billing address",
         "The billing address form returns 'invalid postcode' for a valid one.",
         "open", "low", None, ts(27)),
        ("T-1012", "lucia@example.com", "Charged after cancelling",
         "I cancelled last month but was charged again this month.",
         "open", "urgent", None, ts(29)),
    ]

    conn.executemany(
        "INSERT INTO tickets (id, customer_email, subject, body, status, "
        "priority, department, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)
