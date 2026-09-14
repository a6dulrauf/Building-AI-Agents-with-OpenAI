"""The tool layer: what the model is allowed to do, and what it may not.

Three jobs, and it is worth naming them separately:

1. VALIDATE input before it reaches the database.
2. TRANSLATE between dicts (repository) and strings (what a model reads).
3. DESCRIBE itself as JSON schema so the model knows what exists.

Every function returns a string starting with "ok:" or "error:". Errors
are returned rather than raised on purpose: the model reads the error,
fixes its input, and retries. Raising would crash the loop instead.

TOOL_SCHEMAS below is written by hand. The Agents SDK generates the same
thing automatically from type hints and docstrings — comparing the two is
the fastest way to understand what @function_tool actually does. There is
a test for exactly that in tests/test_sdk_agent.py.
"""
from __future__ import annotations

import functools
import json
import re
import sqlite3
from collections.abc import Callable

from helpdesk import repository
from helpdesk.config import load_settings
from helpdesk.database import DEPARTMENTS, STATUSES, connect, init_schema, seed

TICKET_ID_PATTERN = re.compile(r"^T-\d+$")
MAX_NOTE_LENGTH = 500

WRITE_TOOLS: frozenset[str] = frozenset({"assign_department", "add_note"})

_conn: sqlite3.Connection | None = None
_conn_initialised = False


def set_connection(conn: sqlite3.Connection | None) -> None:
    """Inject a connection. Tests use this to point at a temp database."""
    global _conn, _conn_initialised
    _conn = conn
    _conn_initialised = conn is not None


def _get_conn() -> sqlite3.Connection:
    """Return the active connection, opening the real one on first use."""
    global _conn, _conn_initialised
    if not _conn_initialised:
        settings = load_settings()
        _conn = connect(settings.db_path)
        init_schema(_conn)
        seed(_conn)
        _conn_initialised = True
    if _conn is None:
        raise RuntimeError("No database connection is configured.")
    return _conn


def _err(message: str) -> str:
    return f"error: {message}"


def _validate_ticket_id(ticket_id: str) -> str | None:
    """Return an error message, or None if the ID is well formed."""
    if not TICKET_ID_PATTERN.match(ticket_id or ""):
        return _err(
            f"{ticket_id!r} is not a valid ticket ID. "
            'IDs look like "T-1006" — the letter T, a hyphen, then digits.'
        )
    return None


def safe_tool(func: Callable[..., str]) -> Callable[..., str]:
    """Convert any unexpected exception into an error string.

    A bug inside a tool should degrade a single turn, not end the
    conversation. Validation errors are returned deliberately by each
    function; this catches only the ones nobody predicted.

    functools.wraps is NOT optional here. It sets __wrapped__, which is
    what inspect.signature() follows to find the real parameters. Copying
    __name__ and __doc__ by hand instead leaves the signature reading
    (*args, **kwargs), and the Agents SDK then fails to build a schema
    with the thoroughly misleading error "additionalProperties should not
    be set for object types". Verified: hand-copied attributes fail,
    functools.wraps works.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> str:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all boundary
            return _err(f"{func.__name__} failed unexpectedly: {exc}")

    return wrapper


@safe_tool
def get_ticket(ticket_id: str) -> str:
    """Look up one support ticket by ID, including any notes on it.

    Args:
        ticket_id: The ticket identifier, for example "T-1006".
    """
    if error := _validate_ticket_id(ticket_id):
        return error

    conn = _get_conn()
    ticket = repository.get_ticket(conn, ticket_id)
    if ticket is None:
        return _err(f"No ticket found with ID {ticket_id}.")

    ticket["notes"] = repository.get_notes(conn, ticket_id)
    return json.dumps(ticket, indent=2)


@safe_tool
def search_tickets(customer_email: str = "", status: str = "") -> str:
    """Search tickets by customer email, by status, or both.

    Use this to check a customer's history before recommending a
    department.

    Args:
        customer_email: Email address to filter by. Optional.
        status: One of open, pending, resolved. Optional.
    """
    if not customer_email and not status:
        return _err(
            "search_tickets needs at least one filter: customer_email, "
            "status, or both."
        )
    if status and status not in STATUSES:
        return _err(
            f"{status!r} is not a valid status. Choose one of: "
            f"{', '.join(STATUSES)}."
        )

    results = repository.search_tickets(
        _get_conn(), customer_email=customer_email or None, status=status or None
    )
    return json.dumps({"count": len(results), "tickets": results}, indent=2)


@safe_tool
def assign_department(ticket_id: str, department: str) -> str:
    """Propose assigning a ticket to a department. Requires human approval.

    Args:
        ticket_id: The ticket identifier, for example "T-1006".
        department: One of billing, technical, account, shipping.
    """
    if error := _validate_ticket_id(ticket_id):
        return error

    normalised = (department or "").strip().lower()
    if normalised not in DEPARTMENTS:
        return _err(
            f"{department!r} is not a valid department. Choose one of: "
            f"{', '.join(DEPARTMENTS)}."
        )

    if not repository.assign_department(_get_conn(), ticket_id, normalised):
        return _err(f"No ticket found with ID {ticket_id}.")
    return f"ok: {ticket_id} assigned to {normalised}."


@safe_tool
def add_note(ticket_id: str, note: str) -> str:
    """Propose adding an internal note to a ticket. Requires human approval.

    Args:
        ticket_id: The ticket identifier, for example "T-1006".
        note: The note text, up to 500 characters.
    """
    if error := _validate_ticket_id(ticket_id):
        return error

    text = (note or "").strip()
    if not text:
        return _err("The note is empty. Provide some text.")
    if len(text) > MAX_NOTE_LENGTH:
        return _err(
            f"The note is {len(text)} characters; the limit is "
            f"{MAX_NOTE_LENGTH}. Shorten it."
        )

    if not repository.add_note(_get_conn(), ticket_id, text):
        return _err(f"No ticket found with ID {ticket_id}.")
    return f"ok: note added to {ticket_id}."


TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "get_ticket": get_ticket,
    "search_tickets": search_tickets,
    "assign_department": assign_department,
    "add_note": add_note,
}

# Hand-written schemas for the raw loop. This is the contract the model
# sees: names, types, which arguments are required, and prose describing
# each one. The model picks tools from this and nothing else, which is why
# the descriptions matter as much as the code.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket",
            "description": "Look up one support ticket by ID, including any notes on it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": 'The ticket identifier, for example "T-1006".',
                    }
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tickets",
            "description": (
                "Search tickets by customer email, by status, or both. Use this "
                "to check a customer's history before recommending a department."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_email": {
                        "type": "string",
                        "description": "Email address to filter by. Optional.",
                    },
                    "status": {
                        "type": "string",
                        "enum": list(STATUSES),
                        "description": "Ticket status to filter by. Optional.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_department",
            "description": (
                "Propose assigning a ticket to a department. Requires human approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": 'The ticket identifier, for example "T-1006".',
                    },
                    "department": {
                        "type": "string",
                        "enum": list(DEPARTMENTS),
                        "description": "The department to assign the ticket to.",
                    },
                },
                "required": ["ticket_id", "department"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_note",
            "description": (
                "Propose adding an internal note to a ticket. Requires human approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": 'The ticket identifier, for example "T-1006".',
                    },
                    "note": {
                        "type": "string",
                        "description": "The note text, up to 500 characters.",
                    },
                },
                "required": ["ticket_id", "note"],
            },
        },
    },
]
