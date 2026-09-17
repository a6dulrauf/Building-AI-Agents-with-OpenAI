# Helpdesk Support Agent — Design

**Date:** 2026-09-14
**Status:** Approved for implementation

## Purpose

A working Helpdesk Support Agent that reads support tickets, reasons about
them, and takes action through validated function calls — built as a
teaching artifact. Every layer is readable, and the agent loop is
implemented twice: once by hand, once with the OpenAI Agents SDK.

Two audiences, one codebase:

1. **The learner.** Read `raw_agent.py` and the agent loop stops being
   magic. Read `sdk_agent.py` and see what the framework removed.
2. **The support team.** A Streamlit chat UI over a real SQLite ticket
   database.

## Goals

- Show the agent loop explicitly before hiding it behind a framework.
- Run fully offline against Ollama; switch to OpenAI with one env var.
- Keep business logic testable without an API key or network access.
- Model real secret handling — nothing hardcoded, nothing committed.
- Keep a human in front of every database write, so the agent
  *suggests* actions rather than silently performing them.

## Non-goals

- Multi-agent handoffs, RAG, and streaming. They are mentioned in the
  README as next steps, not built. Scope discipline is the point.
- Authentication, deployment, real ticketing-system integration.
- Retry/backoff policies beyond a turn cap.

## A note on terminology

**AgentKit** is OpenAI's agent platform: Agent Builder (a visual canvas),
ChatKit (embeddable chat UI), the Connector Registry, and Evals. The
**Agents SDK** is its code-first Python component, and that is what this
project uses. "AgentKit's Agents SDK" is the precise phrasing.

## Architecture

```
  app.py  /  run_raw.py            how a human talks to it
        |
        v
  raw_agent.py  <->  sdk_agent.py  THE LOOP (two implementations)
        |
        v
  tools.py                         validation + JSON schemas (LLM-facing)
        |
        v
  repository.py                    plain SQL (knows nothing about AI)
        |
        v
  SQLite
```

The load-bearing rule: **each layer may only call the one below it.**
`repository.py` never imports an LLM library. `tools.py` never writes
SQL. That separation is what makes the tests possible without a key.

### The dual-implementation trick

`tools.py` exports both the business functions and hand-written JSON
schemas describing them.

- `raw_agent.py` passes those schemas to the API directly and dispatches
  tool calls itself.
- `sdk_agent.py` wraps the *same functions* in `@function_tool` and lets
  the SDK generate schemas from type hints and docstrings.

Both schemas can be printed side by side. They match. That is the moment
the decorator becomes ordinary.

## Data model

SQLite, two tables.

**tickets**
| column | type | notes |
|---|---|---|
| id | TEXT PK | format `T-####` |
| customer_email | TEXT NOT NULL | |
| subject | TEXT NOT NULL | |
| body | TEXT NOT NULL | |
| status | TEXT NOT NULL | open / pending / resolved |
| priority | TEXT NOT NULL | low / medium / high / urgent |
| department | TEXT NULL | NULL until assigned |
| created_at | TEXT NOT NULL | ISO-8601 |

**notes**
| column | type | notes |
|---|---|---|
| id | INTEGER PK AUTOINCREMENT | |
| ticket_id | TEXT NOT NULL | FK -> tickets.id |
| author | TEXT NOT NULL | "agent" or a human name |
| note | TEXT NOT NULL | |
| created_at | TEXT NOT NULL | ISO-8601 |

Seeded with ~12 tickets across several customers, with repeat customers
so ticket-history lookups return something interesting.

## Tools

Four tools, each mapping to a stated assignment requirement.

| Tool | Requirement | Validation | Approval |
|---|---|---|---|
| `get_ticket(ticket_id)` | read support tickets | ID matches `T-\d+`; missing ID returns a clear error | auto |
| `search_tickets(customer_email=None, status=None)` | check ticket history | at least one filter required | auto |
| `assign_department(ticket_id, department)` | route to the right department | department in {billing, technical, account, shipping} | **human** |
| `add_note(ticket_id, note)` | audit trail for humans | non-empty, <= 500 chars | **human** |

Reads run freely. Writes are *proposed* and wait for a person — which is
what the brief means by "suggest the right action".

### Error handling policy

Validation failures are **returned to the model as tool output**, never
raised as exceptions. The model sees:

```
error: 'finance' is not a valid department. Choose one of:
billing, technical, account, shipping.
```

and corrects itself on the next turn. Self-correction is observable in
the raw loop's turn-by-turn trace. Exceptions would instead crash the
loop and teach nothing.

Unexpected exceptions are caught at the dispatch boundary and converted
to a generic error string, so a bug in a tool degrades the turn rather
than killing the session.

## Human-in-the-loop write approval

The agent never mutates the ticket database unsupervised. Write tools are
marked `needs_approval=True`; the SDK pauses the run and surfaces the
pending call, and a person decides.

```python
result = await Runner.run(agent, message, session=session)

while result.interruptions:          # run is paused
    state = result.to_state()        # serializable snapshot
    for item in result.interruptions:
        if human_said_yes(item):
            state.approve(item)
        else:
            state.reject(item, rejection_message="Rejected by support agent.")
    result = await Runner.run(agent, state)   # resume
```

The paused run is held in `st.session_state`, which survives Streamlit's
re-execution of the whole script on every interaction.

*Corrected during implementation planning:* an earlier draft of this spec
claimed `RunState` string serialization was load-bearing here. It is not.
Streamlit reruns happen in the same process, so `session_state` holds live
Python objects and the `RunResult` is kept directly. `to_string()` /
`RunState.from_json()` matter only across a process boundary — persisting
a paused run across a server restart, or handing it to a worker.

A rejection is not a dead end: `rejection_message` is fed back to the
model, which can then propose something else. Rejecting a wrong
department assignment and watching the agent reconsider is the clearest
demonstration in the project that this is an agent and not a chatbot.

**The raw loop implements the same pause by hand** — a set of write-tool
names, an `input("Approve? [y/N]: ")`, and an error string pushed back
into `messages` on refusal. Seeing those twenty lines first is what makes
`needs_approval=True` legible rather than magical.

## UI transparency

The Streamlit UI renders every tool call inline, not just the final
answer:

```
🔧 get_ticket("T-1021")
   → {"status": "open", "customer": "aisha@...", ...}
🔧 search_tickets(customer_email="aisha@...")
   → 3 prior tickets, 2 billing
⏸  assign_department("T-1021", "billing")   [Approve] [Reject]
```

Without this the agent is a black box that silently edits a database.
With it, the support team can audit what was read and what is about to
change. This is the concrete mechanism behind the claim that the UI keeps
a human in review — the claim would otherwise be decorative.

## Memory

Short-term conversational memory, two implementations of one idea.

- **raw_agent.py** — a literal `messages: list[dict]` that grows each
  turn. Printable. Memory is demystified by being visibly just a list.
- **sdk_agent.py** — `SQLiteSession(session_id)`, keyed per Streamlit
  session so each browser tab is an independent conversation.

Turn cap (`MAX_TURNS`, default 8) bounds both loops. Hitting it returns
a partial answer and a warning rather than looping forever.

## Provider switching

`.env` selects the backend:

```
LLM_PROVIDER=ollama     # or: openai
```

`providers.py` returns a configured client for either, because Ollama
exposes an OpenAI-compatible endpoint at `http://localhost:11434/v1`.
Agent code is provider-agnostic.

- **openai** — `OPENAI_API_KEY` required; `OPENAI_MODEL` defaults to
  `gpt-4o-mini` (tool-calling capable, cheapest sensible default).
- **ollama** — no key needed (a placeholder string satisfies the client).
  `set_tracing_disabled(True)`, since there is no OpenAI key to trace
  with. `OLLAMA_MODEL` must name a model that supports tool calling.

`config.py` validates the selected provider's requirements at import and
raises a readable error naming the missing variable.

## Security

- `.env` is gitignored in the first commit, before it ever exists.
- `.env.example` is committed with empty values as the template.
- Keys are read only in `config.py`, only from the environment. No key
  appears in a Streamlit widget, a log line, or a prompt.
- `data/helpdesk.db` is gitignored — it is generated, not source.
- README documents what changes in production: a secrets manager
  (AWS Secrets Manager, Streamlit Cloud secrets) replaces `.env`, which
  is a local-development convenience only.

## Testing

`pytest`, against a temporary database, with **no API calls and no key**:

- `test_repository.py` — CRUD, filtering, missing-row behaviour.
- `test_tools.py` — every validation rule, including that invalid input
  returns an error *string* rather than raising.

The suite runs offline in about a second. That speed is the direct
payoff of keeping SQL and validation below the LLM layer, and is the
strongest available argument for the layering.

## File layout

```
helpdesk-agent/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                    # entry point: Streamlit UI
├── run_raw.py                # entry point: the teaching loop
├── helpdesk/
│   ├── __init__.py
│   ├── config.py
│   ├── providers.py
│   ├── database.py
│   ├── repository.py
│   ├── prompts.py
│   ├── tools.py
│   ├── raw_agent.py
│   └── sdk_agent.py
├── data/
│   ├── schema.sql
│   └── helpdesk.db           # gitignored, generated
└── tests/
    ├── test_repository.py
    └── test_tools.py
```

## Environment

- Python 3.12.10, `.venv` inside the project.
- `openai-agents` (version pinned at install; HITL approval and
  `agents.decorators.tool` require a recent release), `openai`,
  `streamlit`, `python-dotenv`, `pytest`.
- Ollama installed locally; server must be running to use that provider.
  The specific model is pinned once `ollama list` can be read.

## Assignment mapping

The four written tasks are answered by artifacts in this repo, so the
learner writes the prose from evidence rather than from memory.

| Task | Where the evidence lives |
|---|---|
| 1. Why agents over a chatbot | `run_raw.py` turn trace: the model chooses tools, acts, and re-plans after a rejection, rather than only replying |
| 2. Architecture overview | The layer diagram above; `raw_agent.py` is the flow in executable form |
| 3. Setup strategy | `.env.example`, `.gitignore`, `config.py` validation, README production note |
| 4. Reliability | Memory across turns; validation returning correctable errors; write approval + inline tool-call display putting a human in review |
