"""Tests for the tool layer: validation, error phrasing, and schemas.

Still no API key and no network. Tools are ordinary Python functions —
the decorator in sdk_agent.py is the only thing that makes them "AI".
"""
import json

import pytest

from helpdesk.database import connect, init_schema, seed
from helpdesk import tools


@pytest.fixture(autouse=True)
def _wire_temp_db(tmp_path):
    """Point the tool layer at a throwaway database for every test."""
    conn = connect(str(tmp_path / "tools.db"))
    init_schema(conn)
    seed(conn)
    tools.set_connection(conn)
    yield
    tools.set_connection(None)


def test_get_ticket_returns_json_with_notes():
    payload = json.loads(tools.get_ticket("T-1006"))
    assert payload["subject"] == "App crashes on export to PDF"
    assert payload["notes"] == []


def test_get_ticket_rejects_a_malformed_id_without_raising():
    result = tools.get_ticket("1006")
    assert result.startswith("error:")
    assert "T-1006" in result  # the message shows the expected shape


def test_get_ticket_reports_a_missing_ticket_clearly():
    result = tools.get_ticket("T-9999")
    assert result.startswith("error:")
    assert "T-9999" in result


def test_search_requires_at_least_one_filter():
    result = tools.search_tickets()
    assert result.startswith("error:")


def test_search_by_customer_returns_history():
    payload = json.loads(tools.search_tickets(customer_email="aisha@example.com"))
    assert payload["count"] == 4


def test_assign_department_rejects_an_invalid_department():
    result = tools.assign_department("T-1006", "finance")
    assert result.startswith("error:")
    # The model can only self-correct if the message lists the valid options.
    for dept in ("billing", "technical", "account", "shipping"):
        assert dept in result


def test_assign_department_is_case_insensitive():
    assert tools.assign_department("T-1006", "TECHNICAL").startswith("ok:")


def test_assign_department_succeeds_and_persists():
    assert tools.assign_department("T-1006", "technical").startswith("ok:")
    assert json.loads(tools.get_ticket("T-1006"))["department"] == "technical"


def test_add_note_rejects_empty_text():
    assert tools.add_note("T-1006", "   ").startswith("error:")


def test_add_note_rejects_overlong_text():
    assert tools.add_note("T-1006", "x" * 501).startswith("error:")


def test_add_note_succeeds_and_is_readable_afterwards():
    assert tools.add_note("T-1006", "Reproduced on 4.2.1.").startswith("ok:")
    assert json.loads(tools.get_ticket("T-1006"))["notes"][0]["note"] == "Reproduced on 4.2.1."


def test_every_schema_has_a_matching_function():
    names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert names == set(tools.TOOL_FUNCTIONS)


def test_only_writes_require_approval():
    assert tools.WRITE_TOOLS == frozenset({"assign_department", "add_note"})


def test_unknown_keyword_argument_names_the_bad_argument_and_valid_ones():
    """An argument-binding mistake is the model's fault, not the tool's.

    The message must name what was wrong AND list the real parameters, so
    the model can self-correct on the next turn instead of giving up on
    tools the way it did against llama3.1 in the T-1006 transcript.
    """
    result = tools.search_tickets(bogus="x")
    assert result.startswith("error:")
    assert "bogus" in result
    assert "customer_email" in result
    assert "status" in result
    assert "failed unexpectedly" not in result


def test_ticket_id_passed_to_search_tickets_points_at_get_ticket():
    """The specific mistake from the live transcript: ticket_id sent to
    search_tickets, which has no such parameter. The error should point
    at the tool that actually accepts a ticket_id.
    """
    result = tools.search_tickets(ticket_id="T-1006")
    assert result.startswith("error:")
    assert "ticket_id" in result
    assert "get_ticket" in result
    assert "failed unexpectedly" not in result


def test_missing_required_argument_is_a_binding_error_not_a_crash():
    """A missing required argument is also an argument-binding mistake,
    not an internal fault — it must be caught and explained the same way
    as an unknown keyword, before the function body ever runs.
    """
    result = tools.get_ticket()
    assert result.startswith("error:")
    assert "ticket_id" in result
    assert "failed unexpectedly" not in result


def test_unexpected_errors_become_error_strings_not_crashes(tmp_path):
    """A bug in a tool must degrade one turn, not kill the session.

    We inject a CLOSED connection so the tool hits sqlite3.ProgrammingError
    deep inside repository code — an exception nobody wrote a check for,
    which is exactly what the safe_tool boundary exists to absorb.

    Note we do NOT test this by passing None: set_connection(None) resets
    the lazy-init flag, so _get_conn() would happily open the real
    database and the call would succeed.
    """
    broken = connect(str(tmp_path / "broken.db"))
    init_schema(broken)
    broken.close()
    tools.set_connection(broken)

    result = tools.get_ticket("T-1006")
    assert result.startswith("error:")
    assert "get_ticket" in result  # the message names the failing tool
