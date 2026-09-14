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

`.env` must point `OLLAMA_MODEL` at a model that actually supports tool
calling and that you have pulled locally (`ollama list`). `llama3.1`,
`qwen2.5`, and `mistral-nemo` all work; small models like `llama3.2:1b`
do not call tools reliably.

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
that supports tool calling — `llama3.1` or `qwen2.5`, not `llama3.2:1b`.

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
