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

`.env` must point `OLLAMA_MODEL` at a model that supports tool calling and
that you have pulled locally (`ollama list`).

What was actually tested: `llama3.1` (8B). It does call tools, but not
dependably — see [Approval](#approval). `qwen2.5` and `mistral-nemo` are
suggestions on the basis that they advertise tool calling; neither was run
here, so treat them as untested. Small models such as `llama3.2:1b` do not
call tools reliably at all and will produce failures that look like bugs
in your code.

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

That is the DATA path, and along it each layer calls only the one below.
It is what makes `repository.py`'s discipline possible: it never imports an
LLM library, which is why the tests run offline in about a second.

Four modules sit beside this stack rather than in it — `config.py`,
`prompts.py`, `providers.py` and `database.py` are cross-cutting, and
several layers reach them directly. `app.py` imports `config` as well as
`sdk_agent`; `tools.py` imports `config`; `run_raw.py` imports
`providers`. The strict one-layer-down rule is a claim about the data
path, not about the import graph as a whole.

## Switching providers

One line in `.env`:

```
LLM_PROVIDER=ollama     # local, no key
LLM_PROVIDER=openai     # needs OPENAI_API_KEY
```

This works because Ollama serves an OpenAI-compatible API at
`localhost:11434/v1`, so a single client class covers both. Use a model
that supports tool calling. On the Ollama side only `llama3.1` was run
here, and only partially successfully; `qwen2.5` and `mistral-nemo` are
plausible untested alternatives, and `llama3.2:1b` is too small. On the
OpenAI side, `gpt-4o-mini` is the default and handles the tool sequence
reliably.

## Approval

Reads (`get_ticket`, `search_tickets`) run automatically. Writes
(`assign_department`, `add_note`) pause for a human. Rejecting one sends a
message back to the model, which then proposes something else.

**Expect a small local model to wobble here.** Running the reject-then-
reconsider flow twice on `llama3.1` (8B) gave two different outcomes: once
it replied in prose without calling the tool at all, and once it called the
tool with empty/invalid arguments while skipping the `get_ticket` lookup the
system prompt requires first. Nothing in the code misbehaved — the
validation in `tools.py` caught the bad call and returned an error the model
could act on, which is the whole reason that layer exists. It is a
model-capability limit. A larger local model, or `gpt-4o-mini` on the OpenAI
path, follows the sequence far more reliably. The approval gate itself is
enforced in code and does not depend on the model cooperating.

## Security

- `.env` is gitignored and was never committed.
- Keys are read only in `helpdesk/config.py`, only from the environment.
- `config.py` fails at startup with a message naming the missing variable.
- **Tracing is on when `LLM_PROVIDER=openai`.** The Agents SDK exports each
  run to OpenAI's traces dashboard — prompts, tool calls and their
  arguments, and tool results verbatim, which here means ticket subjects
  and bodies, customer email addresses and note text. It is switched off
  for every other provider (see `helpdesk/providers._configure_tracing`).
  For real support data, disable it unconditionally with
  `set_tracing_disabled(True)`.
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
