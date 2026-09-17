"""Tests for the SQL layer.

Note what is absent: no API key, no network, no mocking of an LLM. That is
the payoff of keeping SQL below the agent — this file runs in milliseconds
and never costs a token.
"""
import pytest

from helpdesk.database import connect, init_schema, seed
from helpdesk import repository as repo


@pytest.fixture()
def conn(tmp_path):
    c = connect(str(tmp_path / "test.db"))
    init_schema(c)
    seed(c)
    return c


def test_get_ticket_returns_a_dict(conn):
    ticket = repo.get_ticket(conn, "T-1006")
    assert ticket["subject"] == "App crashes on export to PDF"
    assert ticket["priority"] == "urgent"


def test_get_ticket_returns_none_when_missing(conn):
    assert repo.get_ticket(conn, "T-9999") is None


def test_search_by_customer_returns_their_history(conn):
    results = repo.search_tickets(conn, customer_email="aisha@example.com")
    assert len(results) == 4
    assert all(t["customer_email"] == "aisha@example.com" for t in results)


def test_search_by_status(conn):
    results = repo.search_tickets(conn, status="open")
    assert len(results) >= 5
    assert all(t["status"] == "open" for t in results)


def test_search_combines_filters(conn):
    results = repo.search_tickets(conn, customer_email="aisha@example.com", status="open")
    assert all(
        t["customer_email"] == "aisha@example.com" and t["status"] == "open"
        for t in results
    )


def test_search_with_no_filters_returns_everything(conn):
    assert len(repo.search_tickets(conn)) == 12


def test_assign_department_persists(conn):
    assert repo.assign_department(conn, "T-1006", "technical") is True
    assert repo.get_ticket(conn, "T-1006")["department"] == "technical"


def test_assign_department_returns_false_for_missing_ticket(conn):
    assert repo.assign_department(conn, "T-9999", "technical") is False


def test_add_note_then_read_it_back(conn):
    assert repo.add_note(conn, "T-1006", "Reproduced on 4.2.1.") is True
    notes = repo.get_notes(conn, "T-1006")
    assert len(notes) == 1
    assert notes[0]["note"] == "Reproduced on 4.2.1."
    assert notes[0]["author"] == "agent"


def test_add_note_returns_false_for_missing_ticket(conn):
    assert repo.add_note(conn, "T-9999", "orphan") is False
