# Helpdesk Support Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working Helpdesk Support Agent that reads tickets, reasons about them, and proposes actions through validated function calls — with the agent loop implemented twice (by hand, then with the Agents SDK) so the mechanism is visible before it is abstracted.

**Architecture:** Four layers, each calling only the one below it: UI (`app.py`, `run_raw.py`) → agent loop (`raw_agent.py`, `sdk_agent.py`) → tools with validation (`tools.py`) → plain SQL (`repository.py`) → SQLite. Reads execute freely; writes pause for human approval. One provider module lets the same agent run on OpenAI or local Ollama.

**Tech Stack:** Python 3.12.10, `openai-agents` 0.22.2, `openai` 3.13.0, `streamlit` 1.63.0, `python-dotenv` 1.2.3, `pytest` 9.1.1, SQLite (stdlib `sqlite3`).

**Spec:** `docs/superpowers/specs/2026-09-14-helpdesk-agent-design.md`

## Global Constraints

These apply to every task. All API facts below were verified by introspecting the installed package — do not substitute remembered alternatives.

- **Python 3.12.10**, virtualenv at `.venv/` in the project root. Run everything as `.venv/bin/python` / `.venv/bin/pytest`.
- **Imports come from the `agents` top level:** `from agents import Agent, Runner, SQLiteSession, RunState, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled`. (`agents.decorators.tool` is an alias that also exists; prefer `function_tool` — it is what the bulk of the documentation uses.)
- **Approval API, verified signatures:**
  - `@function_tool(needs_approval=True)` marks a tool as requiring a human.
  - `result.interruptions` — list of `ToolApprovalItem` (a dataclass field on `RunResult`).
  - `ToolApprovalItem` exposes **properties** `.name`, `.arguments` (a JSON string), `.call_id`.
  - `result.to_state()` → `RunState`; `state.to_string()` → str; `RunState.from_json(agent, dict)` → `RunState`.
  - `state.approve(item, always_approve=False)`; `state.reject(item, always_reject=False, *, rejection_message=None)`.
  - Resume with `await Runner.run(agent, state, session=session)`.
- **`SQLiteSession(session_id, db_path=":memory:")`** — the db_path default is in-memory. **We always pass a file path.** Methods: `add_items`, `get_items`, `pop_item`, `clear_session`, `close`. It creates tables `agent_sessions` and `agent_messages`.
- **`Runner.run(starting_agent, input, *, context=None, max_turns=10, session=None)`** — `input` accepts a `str` **or** a `RunState` (that is how a paused run resumes).
- **Never commit secrets.** `.env` is already gitignored. Every key is read only in `helpdesk/config.py`, only from the environment.
- **Business logic must be testable with no API key and no network.** `repository.py` and `tools.py` never import an LLM client.
- **Departments are exactly:** `billing`, `technical`, `account`, `shipping`. Lowercase, validated everywhere.
- **Ticket ID format is exactly:** `T-` followed by one or more digits (regex `^T-\d+$`).
- **Every file gets a module docstring** explaining what it does and why it exists. This project is read as much as it is run.

---

## File Structure

| File | Responsibility |
|---|---|
| `helpdesk/config.py` | Load `.env`, expose a frozen `Settings`, fail loudly on misconfiguration |
| `helpdesk/providers.py` | Build an OpenAI **or** Ollama client; the only provider-aware module |
| `helpdesk/database.py` | Connection factory, schema creation, seeding |
| `helpdesk/repository.py` | Plain SQL functions. No LLM imports, ever |
| `helpdesk/tools.py` | Validation + the four tool functions + hand-written JSON schemas |
| `helpdesk/prompts.py` | The system prompt, isolated so it can be tuned without touching code |
| `helpdesk/raw_agent.py` | The hand-written agent loop — the teaching artifact |
| `helpdesk/sdk_agent.py` | The same agent via the Agents SDK, with HITL approval |
| `run_raw.py` | Terminal entry point for the raw loop |
| `app.py` | Streamlit UI: chat, tool-call display, approve/reject, clear |
| `data/schema.sql` | Table definitions |
| `tests/` | Offline test suite |

---

## Task 1: Configuration and provider switch

**Files:**
- Create: `helpdesk/__init__.py`, `helpdesk/config.py`, `helpdesk/providers.py`
- Create: `.env.example`, `requirements.txt`
- Test: `tests/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `Settings` frozen dataclass with fields: `provider: str`, `model: str`, `api_key: str`, `base_url: str | None`, `max_turns: int`, `db_path: str`, `chat_db_path: str`
  - `load_settings(env: Mapping[str, str] | None = None) -> Settings` — raises `ConfigError` on bad input
  - `ConfigError(Exception)`
  - `build_async_client(settings: Settings) -> AsyncOpenAI`
  - `build_sync_client(settings: Settings) -> OpenAI`
  - `build_model(settings: Settings) -> OpenAIChatCompletionsModel`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty file) and `tests/test_config.py`:

```python
"""Tests for configuration loading and validation.

These run with no network and no real API key — load_settings() takes an
explicit env mapping so we never touch os.environ in tests.
"""
import pytest

from helpdesk.config import ConfigError, load_settings


def test_ollama_needs_no_api_key():
    settings = load_settings({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen2.5"})
    assert settings.provider == "ollama"
    assert settings.model == "qwen2.5"
    assert settings.base_url == "http://localhost:11434/v1"
    # Ollama ignores the key, but the OpenAI client library requires a
    # non-empty string, so we supply a placeholder.
    assert settings.api_key == "ollama-local"


def test_openai_requires_api_key():
    with pytest.raises(ConfigError) as exc:
        load_settings({"LLM_PROVIDER": "openai"})
    assert "OPENAI_API_KEY" in str(exc.value)


def test_openai_reads_key_and_defaults_model():
    settings = load_settings({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"})
    assert settings.api_key == "sk-test"
    assert settings.model == "gpt-4o-mini"
    assert settings.base_url is None


def test_unknown_provider_is_rejected_by_name():
    with pytest.raises(ConfigError) as exc:
        load_settings({"LLM_PROVIDER": "anthropic"})
    assert "anthropic" in str(exc.value)


def test_provider_defaults_to_ollama_so_the_project_runs_offline():
    assert load_settings({}).provider == "ollama"


def test_max_turns_is_an_int():
    assert load_settings({"MAX_TURNS": "3"}).max_turns == 3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'helpdesk'`

- [ ] **Step 3: Write `helpdesk/__init__.py`**

```python
"""Helpdesk Support Agent.

A teaching project: the agent loop is implemented twice, once by hand in
raw_agent.py and once with the OpenAI Agents SDK in sdk_agent.py.
"""
```

- [ ] **Step 4: Write `helpdesk/config.py`**

```python
"""Configuration: read the environment once, validate it, fail loudly.

Every secret in this project enters through this module and nowhere else.
That is deliberate — it means there is exactly one file to audit when you
ask "where do the keys come from?"

load_settings() takes an explicit mapping rather than reading os.environ
directly, which is what makes it testable without touching the real
environment.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

# Ollama exposes an OpenAI-compatible API here. That compatibility is the
# whole reason one code path can serve both providers.
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# The OpenAI client library refuses an empty api_key. Ollama never checks
# it, so any non-empty placeholder works.
OLLAMA_PLACEHOLDER_KEY = "ollama-local"

VALID_PROVIDERS = ("ollama", "openai")


class ConfigError(Exception):
    """Raised when configuration is missing or invalid.

    Deliberately raised early with a message naming the exact variable to
    set, so a misconfigured project fails at startup with a readable error
    rather than deep inside an HTTP call.
    """


@dataclass(frozen=True)
class Settings:
    """Everything the app needs to know, resolved and validated."""

    provider: str
    model: str
    api_key: str
    base_url: str | None
    max_turns: int
    db_path: str
    chat_db_path: str


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from an env mapping (defaults to the real environment).

    Args:
        env: Mapping to read from. Pass a dict in tests. When None, .env is
            loaded and os.environ is used.

    Raises:
        ConfigError: if the provider is unknown or a required key is absent.
    """
    if env is None:
        load_dotenv()  # reads .env into os.environ; no-op if the file is absent
        env = os.environ

    provider = env.get("LLM_PROVIDER", "ollama").strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER must be one of {', '.join(VALID_PROVIDERS)}, "
            f"got {provider!r}."
        )

    if provider == "ollama":
        model = env.get("OLLAMA_MODEL", "qwen2.5")
        api_key = OLLAMA_PLACEHOLDER_KEY
        base_url: str | None = OLLAMA_BASE_URL
    else:
        api_key = env.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Add it to .env, or set "
                "LLM_PROVIDER=ollama to run locally with no key."
            )
        model = env.get("OPENAI_MODEL", "gpt-4o-mini")
        base_url = None

    return Settings(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_turns=int(env.get("MAX_TURNS", "8")),
        db_path=env.get("DB_PATH", "data/helpdesk.db"),
        chat_db_path=env.get("CHAT_DB_PATH", "data/chat.db"),
    )
```

- [ ] **Step 5: Run the config tests**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Write `helpdesk/providers.py`**

```python
"""Build an LLM client for whichever provider is configured.

This is the ONLY module that knows a provider exists. raw_agent.py and
sdk_agent.py ask for a client and stay ignorant of what is behind it —
which is why switching providers is a one-line change in .env.

The trick that makes this possible: Ollama speaks the OpenAI wire protocol
at localhost:11434/v1, so the same client class works for both. Only the
base_url differs.
"""
from __future__ import annotations

from agents import AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import OpenAI

from helpdesk.config import Settings


def build_sync_client(settings: Settings) -> OpenAI:
    """Return a synchronous client. Used by the raw loop.

    The raw loop is synchronous on purpose: a while-loop you can read top
    to bottom teaches more than one interrupted by await.
    """
    _configure_tracing(settings)
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def build_async_client(settings: Settings) -> AsyncOpenAI:
    """Return an async client. Used by the Agents SDK, which is async."""
    _configure_tracing(settings)
    return AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)


def build_model(settings: Settings) -> OpenAIChatCompletionsModel:
    """Wrap an async client as a model the SDK's Agent can accept.

    OpenAIChatCompletionsModel is the adapter that lets any
    OpenAI-compatible endpoint act as an Agent's model.
    """
    return OpenAIChatCompletionsModel(
        model=settings.model,
        openai_client=build_async_client(settings),
    )


def _configure_tracing(settings: Settings) -> None:
    """Disable SDK tracing when there is no OpenAI key to trace with.

    Tracing uploads run data to OpenAI's dashboard. On Ollama we have no
    real key, so leaving it on produces a stream of auth warnings that
    make real errors hard to spot.
    """
    if settings.provider != "openai":
        set_tracing_disabled(True)
```

- [ ] **Step 7: Write `.env.example`**

```bash
# Copy to .env and fill in. .env is gitignored and must never be committed.

# Which backend to use: ollama (local, no key needed) or openai
LLM_PROVIDER=ollama

# --- Ollama (local) ---
# Must be a model that supports tool calling, e.g. qwen2.5, llama3.1
OLLAMA_MODEL=qwen2.5

# --- OpenAI ---
# Only needed when LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# --- Limits ---
# Maximum agent loop turns before giving up. Prevents runaway loops.
MAX_TURNS=8
```

- [ ] **Step 8: Write `requirements.txt`**

```
openai-agents==0.22.2
openai==3.13.0
streamlit==1.63.0
python-dotenv==1.2.3
pytest==9.1.1
```

- [ ] **Step 9: Verify the whole suite passes and nothing secret is staged**

Run: `.venv/bin/pytest tests/ -v && git status --short`
Expected: PASS. `git status` must NOT list `.env` or `.venv/`.

- [ ] **Step 10: Commit**

```bash
git add helpdesk/ tests/ .env.example requirements.txt
git commit -m "feat: configuration and provider switch for OpenAI or Ollama"
```

---

## Task 2: Database schema, connection, and seed data

**Files:**
- Create: `data/schema.sql`, `helpdesk/database.py`
- Test: `tests/test_database.py`

**Interfaces:**
- Consumes: nothing from Task 1 (deliberately independent — the DB layer knows nothing about config)
- Produces:
  - `connect(db_path: str) -> sqlite3.Connection` — rows come back as `sqlite3.Row`
  - `init_schema(conn: sqlite3.Connection) -> None`
  - `seed(conn: sqlite3.Connection) -> int` — inserts sample tickets, returns the count
  - `DEPARTMENTS: tuple[str, ...]` and `STATUSES: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_database.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_database.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'helpdesk.database'`

- [ ] **Step 3: Write `data/schema.sql`**

```sql
-- Helpdesk schema. Two tables, deliberately small.
--
-- CHECK constraints mirror the validation in tools.py. That duplication is
-- intentional: validation in tools.py gives the model a correctable error
-- message, while the CHECK here is the last line of defence if anything
-- ever writes to this database without going through the tools.

CREATE TABLE IF NOT EXISTS tickets (
    id             TEXT PRIMARY KEY,
    customer_email TEXT NOT NULL,
    subject        TEXT NOT NULL,
    body           TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('open', 'pending', 'resolved')),
    priority       TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    department     TEXT     NULL CHECK (department IS NULL OR department IN
                                        ('billing', 'technical', 'account', 'shipping')),
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id  TEXT NOT NULL REFERENCES tickets(id),
    author     TEXT NOT NULL,
    note       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Ticket history lookups filter by customer, so index that column.
CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_email);
CREATE INDEX IF NOT EXISTS idx_notes_ticket ON notes(ticket_id);
```

- [ ] **Step 4: Write `helpdesk/database.py`**

```python
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
```

- [ ] **Step 5: Run the database tests**

Run: `.venv/bin/pytest tests/test_database.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Build the real database and look at it**

Run:
```bash
.venv/bin/python -c "
from helpdesk.database import connect, init_schema, seed
c = connect('data/helpdesk.db'); init_schema(c); print('seeded:', seed(c))
"
sqlite3 data/helpdesk.db "SELECT id, customer_email, status, department FROM tickets LIMIT 5;"
```
Expected: `seeded: 12`, then five rows. Confirm `data/helpdesk.db` does **not** appear in `git status --short`.

- [ ] **Step 7: Commit**

```bash
git add helpdesk/database.py data/schema.sql tests/test_database.py
git commit -m "feat: SQLite schema, connection, and seeded ticket data"
```

---

## Task 3: Repository — plain SQL, no AI

**Files:**
- Create: `helpdesk/repository.py`
- Test: `tests/test_repository.py`

**Interfaces:**
- Consumes: `connect`, `init_schema`, `seed`, `DEPARTMENTS` from `helpdesk.database`
- Produces (all take `conn: sqlite3.Connection` as the first argument):
  - `get_ticket(conn, ticket_id: str) -> dict | None`
  - `get_notes(conn, ticket_id: str) -> list[dict]`
  - `search_tickets(conn, customer_email: str | None = None, status: str | None = None) -> list[dict]`
  - `assign_department(conn, ticket_id: str, department: str) -> bool`
  - `add_note(conn, ticket_id: str, note: str, author: str = "agent") -> bool`

- [ ] **Step 1: Write the failing test**

Create `tests/test_repository.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_repository.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'helpdesk.repository'`

- [ ] **Step 3: Write `helpdesk/repository.py`**

```python
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
```

- [ ] **Step 4: Run the repository tests**

Run: `.venv/bin/pytest tests/test_repository.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Run the whole suite and time it**

Run: `.venv/bin/pytest tests/ -q --durations=3`
Expected: all PASS, total under two seconds. That speed is the point — note it, you will cite it in the Task 4 write-up.

- [ ] **Step 6: Commit**

```bash
git add helpdesk/repository.py tests/test_repository.py
git commit -m "feat: ticket repository with offline test coverage"
```

---

## Task 4: Tools — validation, model-facing errors, and hand-written schemas

**Files:**
- Create: `helpdesk/prompts.py`, `helpdesk/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `helpdesk.repository` functions, `helpdesk.database.connect/init_schema/seed`, `helpdesk.config.load_settings`
- Produces:
  - `get_ticket(ticket_id: str) -> str`
  - `search_tickets(customer_email: str = "", status: str = "") -> str`
  - `assign_department(ticket_id: str, department: str) -> str`
  - `add_note(ticket_id: str, note: str) -> str`
  - `TOOL_SCHEMAS: list[dict]` — hand-written JSON schemas for the raw loop
  - `TOOL_FUNCTIONS: dict[str, Callable[..., str]]` — name → function, for raw dispatch
  - `WRITE_TOOLS: frozenset[str]` — which tools need human approval
  - `set_connection(conn) -> None` — dependency injection for tests
  - `SYSTEM_PROMPT: str` in `helpdesk/prompts.py`

**Design note for the implementer:** every tool returns a **string**, because that is what goes back to the model. Validation failures return an error string rather than raising — the model reads it and corrects itself on the next turn. An exception would instead crash the loop and teach the model nothing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tools.py`:

```python
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


def test_unexpected_errors_become_error_strings_not_crashes():
    """A bug in a tool must degrade one turn, not kill the session."""
    tools.set_connection(None)  # force an internal failure
    result = tools.get_ticket("T-1006")
    assert result.startswith("error:")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'helpdesk.tools'`

- [ ] **Step 3: Write `helpdesk/prompts.py`**

```python
"""The agent's system prompt, kept in its own file.

Prompts are tuned far more often than code. Isolating this means you can
iterate on the agent's behaviour with a diff that touches one file and
reviews cleanly.
"""

SYSTEM_PROMPT = """You are a helpdesk support assistant for a software company.

Your job is to help the support team triage tickets. You can:
- look up a ticket by ID
- search a customer's ticket history
- propose assigning a ticket to a department
- propose adding a note to a ticket

How to work:
1. When given a ticket ID, look it up first. Never guess its contents.
2. Before recommending a department, check the customer's history with
   search_tickets. A customer with three prior billing tickets is telling
   you something.
3. Explain your reasoning in one or two sentences before proposing an action.
4. Departments are exactly: billing, technical, account, shipping.

Important: assign_department and add_note require human approval. You are
proposing an action, not performing it. If a proposal is rejected, do not
retry the same thing — reconsider and suggest an alternative, or ask the
support agent what they would prefer.

If a tool returns a string starting with "error:", read it, correct your
input, and try again. Do not report the raw error to the user.

Be concise. The support team is busy."""
```

- [ ] **Step 4: Write `helpdesk/tools.py`**

```python
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
```

- [ ] **Step 5: Run the tool tests**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: PASS (14 tests)

- [ ] **Step 6: Read one error message out loud**

Run:
```bash
.venv/bin/python -c "
from helpdesk import tools
print(tools.assign_department('T-1006', 'finance'))
"
```
Expected: `error: 'finance' is not a valid department. Choose one of: billing, technical, account, shipping.`

That sentence is the whole reliability argument for Task 4. It names what was wrong **and** what the valid options are, so the model's next attempt can succeed. An error that said only "invalid department" would leave it guessing.

- [ ] **Step 7: Commit**

```bash
git add helpdesk/tools.py helpdesk/prompts.py tests/test_tools.py
git commit -m "feat: validated tools with model-readable errors and JSON schemas"
```

---

## Task 5: The raw agent loop — the teaching artifact

**Files:**
- Create: `helpdesk/raw_agent.py`, `run_raw.py`
- Test: `tests/test_raw_agent.py`

**Interfaces:**
- Consumes: `tools.TOOL_SCHEMAS`, `tools.TOOL_FUNCTIONS`, `tools.WRITE_TOOLS`, `providers.build_sync_client`, `prompts.SYSTEM_PROMPT`, `config.load_settings`
- Produces:
  - `run_conversation(client, model, messages, *, approve, on_event=None, max_turns=8) -> str`
  - `new_conversation() -> list[dict]` — returns `[{"role": "system", ...}]`

**Design note:** `approve` is a callable injected by the caller — the terminal passes one that prompts with `input()`, Streamlit would pass one driven by buttons, and tests pass `lambda *_: True`. That injection is what makes this loop testable without a human.

- [ ] **Step 1: Write the failing test**

Create `tests/test_raw_agent.py`:

```python
"""Tests for the hand-written loop, using a fake client.

There is no network here. A FakeClient replays a scripted sequence of
model responses, which lets us assert on loop behaviour — tool dispatch,
approval, the turn cap — deterministically and for free.
"""
import json
from types import SimpleNamespace

import pytest

from helpdesk.database import connect, init_schema, seed
from helpdesk import raw_agent, tools


@pytest.fixture(autouse=True)
def _wire_temp_db(tmp_path):
    conn = connect(str(tmp_path / "raw.db"))
    init_schema(conn)
    seed(conn)
    tools.set_connection(conn)
    yield
    tools.set_connection(None)


def _tool_call(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response(content=None, tool_calls=None):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        model_dump=lambda exclude_none=False: {
            "role": "assistant",
            "content": content,
        },
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Replays scripted responses in order and records what it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_a_reply_with_no_tool_calls_ends_the_loop():
    client = FakeClient([_response(content="Hello.")])
    messages = raw_agent.new_conversation()
    messages.append({"role": "user", "content": "hi"})

    out = raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)

    assert out == "Hello."
    assert len(client.calls) == 1


def test_a_tool_call_is_executed_and_fed_back():
    client = FakeClient([
        _response(tool_calls=[_tool_call("c1", "get_ticket", {"ticket_id": "T-1006"})]),
        _response(content="It is a PDF export crash."),
    ])
    messages = raw_agent.new_conversation()
    messages.append({"role": "user", "content": "what is T-1006?"})

    out = raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)

    assert out == "It is a PDF export crash."
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "App crashes on export to PDF" in tool_messages[0]["content"]


def test_reads_do_not_ask_for_approval():
    asked = []
    client = FakeClient([
        _response(tool_calls=[_tool_call("c1", "get_ticket", {"ticket_id": "T-1006"})]),
        _response(content="done"),
    ])
    messages = raw_agent.new_conversation()

    def approve(name, args):
        asked.append(name)
        return True

    raw_agent.run_conversation(client, "m", messages, approve=approve)
    assert asked == []  # get_ticket is a read; no human needed


def test_writes_ask_for_approval_and_a_refusal_is_reported_to_the_model():
    client = FakeClient([
        _response(tool_calls=[
            _tool_call("c1", "assign_department",
                       {"ticket_id": "T-1006", "department": "billing"})
        ]),
        _response(content="Understood, I will reconsider."),
    ])
    messages = raw_agent.new_conversation()

    raw_agent.run_conversation(client, "m", messages, approve=lambda *_: False)

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert "rejected" in tool_messages[0]["content"].lower()
    # And critically: the database was not touched.
    assert json.loads(tools.get_ticket("T-1006"))["department"] is None


def test_an_approved_write_actually_writes():
    client = FakeClient([
        _response(tool_calls=[
            _tool_call("c1", "assign_department",
                       {"ticket_id": "T-1006", "department": "technical"})
        ]),
        _response(content="Assigned."),
    ])
    messages = raw_agent.new_conversation()

    raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)
    assert json.loads(tools.get_ticket("T-1006"))["department"] == "technical"


def test_the_turn_cap_stops_a_runaway_loop():
    """A model that always calls a tool must not loop forever."""
    client = FakeClient([
        _response(tool_calls=[_tool_call(f"c{i}", "get_ticket", {"ticket_id": "T-1006"})])
        for i in range(10)
    ])
    messages = raw_agent.new_conversation()

    out = raw_agent.run_conversation(
        client, "m", messages, approve=lambda *_: True, max_turns=3
    )

    assert "turn limit" in out.lower()
    assert len(client.calls) == 3


def test_an_unknown_tool_name_does_not_crash_the_loop():
    client = FakeClient([
        _response(tool_calls=[_tool_call("c1", "delete_everything", {})]),
        _response(content="Sorry about that."),
    ])
    messages = raw_agent.new_conversation()

    out = raw_agent.run_conversation(client, "m", messages, approve=lambda *_: True)

    assert out == "Sorry about that."
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert tool_messages[0]["content"].startswith("error:")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_raw_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'helpdesk.raw_agent'`

- [ ] **Step 3: Write `helpdesk/raw_agent.py`**

```python
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
# "thinking", "tool_call", "tool_result", "rejected".
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
        messages.append(_as_dict(message))

        # --- 2. No tool calls means the model is done -------------------
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return message.content or ""

        # --- 3. Run each requested tool and feed the result back --------
        for call in tool_calls:
            name = call.function.name
            arguments = _parse_arguments(call.function.arguments)
            emit("tool_call", {"name": name, "arguments": arguments})

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
    name: str, arguments: dict, approve: ApproveFn, emit: EventFn
) -> str:
    """Run one tool, asking for approval first if it writes."""
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


def _parse_arguments(raw: str) -> dict:
    """Parse the model's JSON arguments, tolerating malformed output."""
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def _as_dict(message) -> dict:
    """Convert the SDK's message object into a plain dict for the transcript."""
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return {"role": "assistant", "content": getattr(message, "content", "")}
```

- [ ] **Step 4: Run the raw-agent tests**

Run: `.venv/bin/pytest tests/test_raw_agent.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Write `run_raw.py`**

```python
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
        first_line = payload["result"].splitlines()[0][:100]
        colour = RED if payload["result"].startswith("error:") else GREY
        print(f"{colour}  <- {first_line}{RESET}")
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
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest tests/ -q`
Expected: all PASS (42 tests so far)

- [ ] **Step 7: Commit**

```bash
git add helpdesk/raw_agent.py run_raw.py tests/test_raw_agent.py
git commit -m "feat: hand-written agent loop with approval and turn cap"
```

---

## Task 6: The SDK agent — same behaviour, framework-managed loop

**Files:**
- Create: `helpdesk/sdk_agent.py`
- Test: `tests/test_sdk_agent.py`

**Interfaces:**
- Consumes: `helpdesk.tools` functions and `TOOL_SCHEMAS`, `helpdesk.providers.build_model`, `helpdesk.prompts.SYSTEM_PROMPT`, `helpdesk.config.Settings`
- Produces:
  - `build_tools() -> list[FunctionTool]`
  - `build_agent(settings: Settings) -> Agent`
  - `build_session(settings: Settings, session_id: str) -> SQLiteSession`
  - `send(agent, session, message: str, max_turns: int) -> RunResult`  (sync wrapper)
  - `resume(agent, session, state: RunState) -> RunResult`  (sync wrapper)
  - `describe_interruption(item) -> tuple[str, dict]` — `(tool_name, arguments_dict)`

**Design note on state:** the SDK can serialise a paused run with `state.to_string()` and restore it with `await RunState.from_json(agent, data)`. We do **not** need that here. Streamlit's `session_state` holds live Python objects across reruns because reruns happen in the same process, so the `RunResult` is simply kept there. Serialisation is for crossing a process boundary — persisting a paused run across a server restart, or handing it to a worker. Worth knowing it exists; wrong tool for this job.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sdk_agent.py`:

```python
"""Tests for the SDK wiring.

No model is called here. We assert on how the tools are DESCRIBED —
because that description is the entire contract between your code and the
model, and it is generated rather than written, which makes it worth
checking.
"""
import pytest

from helpdesk import sdk_agent, tools


@pytest.fixture()
def sdk_tools():
    return {t.name: t for t in sdk_agent.build_tools()}


def test_the_sdk_exposes_the_same_four_tools(sdk_tools):
    assert set(sdk_tools) == set(tools.TOOL_FUNCTIONS)


def test_only_the_write_tools_need_approval(sdk_tools):
    needing = {name for name, t in sdk_tools.items() if t.needs_approval}
    assert needing == tools.WRITE_TOOLS


@pytest.mark.parametrize("name", ["get_ticket", "search_tickets",
                                  "assign_department", "add_note"])
def test_generated_schema_matches_the_hand_written_one(sdk_tools, name):
    """The decorator derives the same parameters we wrote out by hand.

    This is the point of the whole dual implementation. TOOL_SCHEMAS in
    tools.py was typed out manually; the SDK built its version from type
    hints and the docstring. The parameter names agree.
    """
    handwritten = next(
        s["function"] for s in tools.TOOL_SCHEMAS if s["function"]["name"] == name
    )
    generated = sdk_tools[name].params_json_schema

    assert set(generated["properties"]) == set(handwritten["parameters"]["properties"])


def test_generated_descriptions_come_from_the_docstring(sdk_tools):
    """Docstrings are not decoration here — they are shipped to the model."""
    schema = sdk_tools["assign_department"].params_json_schema
    assert "billing" in schema["properties"]["department"]["description"]
    assert sdk_tools["assign_department"].description.startswith("Propose assigning")


def test_strict_mode_puts_optional_params_in_required(sdk_tools):
    """A real difference between the two schemas, worth understanding.

    Our hand-written schema lists required: []. The SDK lists BOTH
    parameters as required, because OpenAI strict function calling
    demands every property appear there. Optionality is expressed with a
    "default" key instead. Same meaning, different encoding.
    """
    generated = sdk_tools["search_tickets"].params_json_schema
    assert set(generated["required"]) == {"customer_email", "status"}
    assert generated["properties"]["customer_email"]["default"] == ""
    assert generated["additionalProperties"] is False


def test_session_is_file_backed_so_memory_can_be_inspected(tmp_path):
    from helpdesk.config import load_settings

    settings = load_settings({"CHAT_DB_PATH": str(tmp_path / "chat.db")})
    session = sdk_agent.build_session(settings, "test-session")
    assert str(tmp_path / "chat.db") in str(session.db_path)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_sdk_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'helpdesk.sdk_agent'`

- [ ] **Step 3: Write `helpdesk/sdk_agent.py`**

```python
"""The same agent, built with the OpenAI Agents SDK.

READ raw_agent.py FIRST. Then read this and notice what disappeared.

The while-loop is gone. Runner.run() contains it. The tool dispatch is
gone — the SDK matches the model's requested name to a decorated function
and calls it. The messages list is gone; SQLiteSession keeps it.

What did NOT disappear is the part that matters: the tools themselves are
the exact same functions from tools.py. function_tool() only wraps them
with a generated schema. Nothing about them became "AI".

The one genuine capability gain over the hand-written loop is
needs_approval=True. Doing that by hand meant a set of tool names and an
if-statement (see raw_agent._execute). Here it is one argument, and the
SDK additionally gives us a resumable RunState so the pause can survive a
round trip through a web UI.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import Agent, FunctionTool, RunResult, Runner, RunState, SQLiteSession, function_tool

from helpdesk import tools
from helpdesk.config import Settings
from helpdesk.prompts import SYSTEM_PROMPT
from helpdesk.providers import build_model

AGENT_NAME = "Helpdesk Support Agent"


def build_tools() -> list[FunctionTool]:
    """Wrap the plain functions from tools.py as SDK tools.

    The schema for each is generated from its type hints and docstring.
    Compare the result against tools.TOOL_SCHEMAS — tests/test_sdk_agent.py
    asserts they agree.
    """
    return [
        function_tool(tools.get_ticket),
        function_tool(tools.search_tickets),
        # Writes pause the run until a human decides. This one argument
        # replaces the WRITE_TOOLS check in raw_agent._execute.
        function_tool(needs_approval=True)(tools.assign_department),
        function_tool(needs_approval=True)(tools.add_note),
    ]


def build_agent(settings: Settings) -> Agent:
    """Assemble the agent: instructions, model, and tools."""
    return Agent(
        name=AGENT_NAME,
        instructions=SYSTEM_PROMPT,
        model=build_model(settings),
        tools=build_tools(),
    )


def build_session(settings: Settings, session_id: str) -> SQLiteSession:
    """Create file-backed conversation memory.

    The db_path argument is what makes this inspectable. Without it the
    SDK defaults to ":memory:" and the history vanishes on restart. With
    it you can run:

        sqlite3 data/chat.db "SELECT * FROM agent_messages LIMIT 5;"

    and see your short-term memory as rows. Being able to look at it is
    worth the one line.
    """
    return SQLiteSession(session_id, settings.chat_db_path)


def send(agent: Agent, session: SQLiteSession, message: str, max_turns: int) -> RunResult:
    """Send a user message and run until the agent finishes or pauses.

    Returns a RunResult. If result.interruptions is non-empty the run is
    PAUSED waiting on human approval — call resume() once decisions are
    recorded.
    """
    return _sync(Runner.run(agent, message, session=session, max_turns=max_turns))


def resume(agent: Agent, session: SQLiteSession, state: RunState) -> RunResult:
    """Continue a paused run after approve()/reject() have been called."""
    return _sync(Runner.run(agent, state, session=session))


def describe_interruption(item: Any) -> tuple[str, dict]:
    """Return (tool_name, arguments) for a pending approval, for display.

    ToolApprovalItem exposes .name and .arguments as properties;
    .arguments is a JSON string, which we parse so the UI can show it
    readably.
    """
    try:
        arguments = json.loads(item.arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        arguments = {}
    return item.name or "unknown_tool", arguments


def _sync(coro):
    """Run a coroutine from synchronous code.

    The SDK is async; Streamlit and our tests are not. asyncio.run() is
    the bridge. It creates a fresh event loop each call, which is fine at
    this scale and keeps the calling code readable.
    """
    return asyncio.run(coro)
```

- [ ] **Step 4: Run the SDK tests**

Run: `.venv/bin/pytest tests/test_sdk_agent.py -v`
Expected: PASS (9 tests, including 4 parametrised)

- [ ] **Step 5: Print both schemas side by side**

Run:
```bash
.venv/bin/python -c "
import json
from helpdesk import tools, sdk_agent
hand = next(s['function'] for s in tools.TOOL_SCHEMAS if s['function']['name']=='assign_department')
gen  = {t.name: t for t in sdk_agent.build_tools()}['assign_department']
print('=== HAND-WRITTEN ==='); print(json.dumps(hand['parameters'], indent=2))
print('=== SDK-GENERATED ==='); print(json.dumps(gen.params_json_schema, indent=2))
"
```

Look at both. Same properties, same types, same descriptions — the SDK read them out of your docstring. The differences (`title`, `additionalProperties`, everything in `required`) are strict-mode bookkeeping, not meaning.

This is the single most useful two minutes in the project. `@function_tool` is a schema generator, nothing more.

- [ ] **Step 6: Commit**

```bash
git add helpdesk/sdk_agent.py tests/test_sdk_agent.py
git commit -m "feat: Agents SDK implementation with approval-gated write tools"
```

---

## Task 7: Streamlit UI — chat, tool visibility, approval

**Files:**
- Create: `app.py`
- Modify: none

**Interfaces:**
- Consumes: `sdk_agent.build_agent/build_session/send/resume/describe_interruption`, `config.load_settings`, `config.ConfigError`
- Produces: the runnable app. No new Python API.

**Design note:** Streamlit re-executes this entire file on every interaction. Anything that must survive a click lives in `st.session_state`; anything expensive to build is wrapped in `@st.cache_resource`. Understanding those two facts is most of what makes Streamlit make sense.

- [ ] **Step 1: Write `app.py`**

```python
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

import json

import streamlit as st

from helpdesk.config import ConfigError, load_settings
from helpdesk.sdk_agent import (
    build_agent,
    build_session,
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


def handle_result(agent, session, result) -> None:
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
    handle_result(agent, session, resume(agent, session, state))


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
        # empty but the agent still remembers everything.
        session.clear_session()
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
    for item in st.session_state.pending.interruptions:
        name, arguments = describe_interruption(item)
        st.markdown(f"**Proposed:** `{name}`")
        st.json(arguments)

        left, right = st.columns(2)
        if left.button("✅ Approve", key=f"a-{item.call_id}", use_container_width=True):
            st.session_state.decisions[item.call_id] = True
            apply_decisions(agent, session)
            st.rerun()
        if right.button("❌ Reject", key=f"r-{item.call_id}", use_container_width=True):
            st.session_state.decisions[item.call_id] = False
            apply_decisions(agent, session)
            st.rerun()

# --- Chat input -----------------------------------------------------------
elif prompt := st.chat_input("Ask about a ticket, e.g. 'triage T-1006'"):
    record("user", prompt)
    with st.spinner("Thinking..."):
        try:
            handle_result(agent, session, send(agent, session, prompt, settings.max_turns))
        except Exception as exc:  # noqa: BLE001 - surface errors instead of a blank page
            record("assistant", f"Something went wrong: `{exc}`")
    st.rerun()
```

- [ ] **Step 2: Verify the app imports cleanly before launching it**

Run: `.venv/bin/python -c "import ast, pathlib; ast.parse(pathlib.Path('app.py').read_text()); print('app.py parses')"`
Expected: `app.py parses`

- [ ] **Step 3: Run the full test suite**

Run: `.venv/bin/pytest tests/ -q`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit UI with tool-call visibility and write approval"
```

---

## Task 8: README, first run, and verification

**Files:**
- Create: `README.md`
- Test: manual end-to-end run

**Interfaces:**
- Consumes: everything
- Produces: documentation

- [ ] **Step 1: Confirm Ollama is running and pick a tool-calling model**

Run:
```bash
ollama list
```

If the server is not running, start the Ollama app first. Pick a model that supports tool calling — `qwen2.5`, `llama3.1`, or `mistral-nemo`. Avoid `llama3.2:1b`; it is too small to use tools reliably and will produce confusing failures that look like bugs in your code.

If nothing suitable is installed:
```bash
ollama pull qwen2.5
```

- [ ] **Step 2: Create your `.env`**

```bash
cp .env.example .env
# edit OLLAMA_MODEL to match a model from `ollama list`
```

Verify it is ignored: `git status --short` must not list `.env`.

- [ ] **Step 3: Run the raw loop and watch a full turn**

Run: `.venv/bin/python run_raw.py`

Type: `Look at T-1006 and suggest which department it belongs to.`

Expected: the agent calls `get_ticket`, probably then `search_tickets`, then proposes `assign_department` and asks for approval. **Reject it once** and watch what it does next — it should reconsider rather than repeat itself. That moment is the clearest evidence you have that this is an agent and not a chatbot; it is worth a screenshot for Task 1.

- [ ] **Step 4: Run the Streamlit app**

Run: `.venv/bin/streamlit run app.py`

Check each of these:
- tool calls appear in the expander with their results
- a write proposal shows Approve / Reject buttons
- approving changes the database; rejecting does not
- "Clear conversation" empties the chat and the agent genuinely forgets

- [ ] **Step 5: Look at your memory as rows**

Run: `sqlite3 data/chat.db "SELECT id, substr(message_data, 1, 80) FROM agent_messages LIMIT 6;"`

That table is the short-term memory. Not a metaphor — rows, replayed into the model's context on every turn.

- [ ] **Step 6: Write `README.md`**

```markdown
# Helpdesk Support Agent

A support agent that reads tickets, reasons about them, and proposes
actions — built so the mechanism is visible.

The agent loop is implemented **twice**: by hand in `helpdesk/raw_agent.py`,
and with the OpenAI Agents SDK in `helpdesk/sdk_agent.py`. They share the
same tools and do the same job. Reading them in that order is the point of
the project.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # defaults to Ollama — no API key needed

.venv/bin/python run_raw.py           # the loop, in your terminal
.venv/bin/streamlit run app.py        # the UI, for the support team
```

## Read it in this order

1. `helpdesk/raw_agent.py` — the loop, ~60 lines, no framework.
2. `helpdesk/tools.py` — what the model may do, and the schemas describing it.
3. `helpdesk/sdk_agent.py` — the same agent via the SDK. Note what vanished.
4. `app.py` — the UI, tool visibility, and the approval gate.

## How it works

```
  app.py / run_raw.py          how a human talks to it
        |
  raw_agent.py / sdk_agent.py  THE LOOP
        |
  tools.py                     validation + schemas
        |
  repository.py                plain SQL, no AI
        |
  SQLite
```

Each layer calls only the one below it. `repository.py` never imports an
LLM library, which is why the tests run offline in about a second.

## Switching providers

One line in `.env`:

```
LLM_PROVIDER=ollama     # local, no key
LLM_PROVIDER=openai     # needs OPENAI_API_KEY
```

This works because Ollama serves an OpenAI-compatible API at
`localhost:11434/v1`, so a single client class covers both. Use a model
that supports tool calling — `qwen2.5` or `llama3.1`, not `llama3.2:1b`.

## Approval

Reads (`get_ticket`, `search_tickets`) run automatically. Writes
(`assign_department`, `add_note`) pause for a human. Rejecting one sends a
message back to the model, which then proposes something else.

## Security

- `.env` is gitignored and was never committed.
- Keys are read only in `helpdesk/config.py`, only from the environment.
- `config.py` fails at startup with a message naming the missing variable.
- **In production, `.env` is not enough** — use a secrets manager (AWS
  Secrets Manager, Streamlit Cloud secrets) and rotate keys. A `.env` file
  is a local-development convenience.

## Tests

```bash
.venv/bin/pytest tests/ -v
```

No API key, no network, about a second. That is a consequence of the
layering: SQL and validation sit below the LLM, so they can be tested
without one.

## Not built (deliberately)

Handoffs to specialist agents, RAG over a knowledge base, streaming
responses, and long-term memory. Long-term memory is the interesting one:
it would be a `customer_profile` table the agent *retrieves from
selectively*, which is what distinguishes it from the session memory here
— session memory is replayed in full every turn.
```

- [ ] **Step 7: Final verification**

Run:
```bash
.venv/bin/pytest tests/ -q
git status --short
git log --oneline
```

Expected: all tests pass. `git status` shows no `.env`, no `.venv/`, no `*.db`.

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "docs: README with reading order, setup, and security notes"
```

---

## Self-Review Notes

**Spec coverage.** Every spec section maps to a task: config/providers → 1; data model → 2; repository → 3; tools, validation, error policy → 4; raw loop, memory-as-a-list, turn cap → 5; SDK agent, HITL, session memory → 6; UI transparency, approval, clear button → 7; security and assignment mapping → 8.

**Two spec corrections found while verifying the API:**

1. The spec says `RunState` serialisation is "load-bearing for Streamlit". It is not — `st.session_state` holds live objects across reruns, so the `RunResult` is kept directly. Serialisation matters only across process boundaries. Task 6 documents this.
2. The spec's tool-wrapper design would have broken schema generation. Copying `__name__`/`__doc__` by hand leaves `inspect.signature()` reading `(*args, **kwargs)`, and the SDK then fails with "additionalProperties should not be set for object types" — an error pointing nowhere near the cause. `functools.wraps` is required. Verified empirically; Task 4 carries the explanation.

**Verified API facts** (introspected from `openai-agents` 0.22.2, not recalled): `function_tool` accepts `needs_approval`; `ToolApprovalItem.name`/`.arguments`/`.call_id` are properties; `RunResult.interruptions` is a dataclass field; `RunState.from_json` is an async staticmethod; `SQLiteSession` defaults to `:memory:` and exposes `clear_session()`; strict mode places every property in `required`.
