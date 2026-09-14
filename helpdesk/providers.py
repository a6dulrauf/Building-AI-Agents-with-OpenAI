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
