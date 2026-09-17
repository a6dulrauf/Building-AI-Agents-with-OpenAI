# Building AI Agents with OpenAI

Projects built while working through the Coursera course of the same name.
Each lives in its own directory with its own README, dependencies and tests.

## Projects

### [`helpdesk-support-agent/`](helpdesk-support-agent/)

A support agent that reads tickets, reasons about them, and **proposes** actions —
checking a customer's history, routing a ticket to a department, adding a note.
Writes wait for a human; reads run freely.

Its defining choice: **the agent loop is implemented twice.** Once by hand in
`raw_agent.py` (about sixty lines, no framework), and once with OpenAI's Agents SDK
in `sdk_agent.py`. Reading them in that order is the point — you can see exactly what
the framework replaced, rather than assuming.

Runs against a local Ollama model or the OpenAI API, switched by one line in `.env`.
68 tests, all offline, no API key required.

- [How the agent loop works](helpdesk-support-agent/docs/agent-loop.md) — diagram and walkthrough
- [Design spec](helpdesk-support-agent/docs/superpowers/specs/) and [implementation plan](helpdesk-support-agent/docs/superpowers/plans/)

```bash
cd helpdesk-support-agent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # defaults to Ollama, no API key needed
.venv/bin/pytest tests/ -q      # 68 tests, offline, ~1s
.venv/bin/python run_raw.py     # watch the loop, turn by turn
```

## Layout

Each project is self-contained — its own `.venv`, `requirements.txt` and tests — so
they can use different dependencies without interfering with one another.

```
Building-AI-Agents-with-OpenAI/
└── helpdesk-support-agent/
    ├── helpdesk/          the package
    ├── tests/             68 offline tests
    ├── scripts/           end-to-end verification harness
    ├── docs/              how it works, plus the spec and plan
    ├── app.py             Streamlit UI
    └── run_raw.py         terminal entry point
```
