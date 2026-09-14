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
