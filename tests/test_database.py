"""Tests for schema creation and seeding, against a temp database."""
import sqlite3

from helpdesk.database import DEPARTMENTS, connect, init_schema, seed


def test_init_schema_creates_both_tables(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    init_schema(conn)
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"tickets", "notes"} <= names


def test_rows_are_accessible_by_column_name(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    init_schema(conn)
    seed(conn)
    row = conn.execute("SELECT * FROM tickets LIMIT 1").fetchone()
    # sqlite3.Row lets us use row["id"] instead of row[0]. Worth the one
    # line in connect(), because positional indexing breaks silently when
    # a column is added.
    assert isinstance(row, sqlite3.Row)
    assert row["id"].startswith("T-")


def test_seed_is_idempotent(tmp_path):
    conn = connect(str(tmp_path / "t.db"))
    init_schema(conn)
    first = seed(conn)
    second = seed(conn)
    count = conn.execute("SELECT COUNT(*) AS n FROM tickets").fetchone()["n"]
    assert first == count
    assert second == 0  # nothing inserted the second time


def test_seed_includes_a_repeat_customer(tmp_path):
    """Ticket history is only interesting if someone has a history."""
    conn = connect(str(tmp_path / "t.db"))
    init_schema(conn)
    seed(conn)
    row = conn.execute(
        "SELECT customer_email, COUNT(*) AS n FROM tickets "
        "GROUP BY customer_email ORDER BY n DESC LIMIT 1"
    ).fetchone()
    assert row["n"] >= 3


def test_departments_are_the_agreed_four():
    assert DEPARTMENTS == ("billing", "technical", "account", "shipping")
