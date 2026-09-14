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


def test_a_non_numeric_max_turns_raises_ConfigError_not_ValueError():
    """A typo in .env must fail the same readable way every other bad value does.

    int() raises a bare ValueError, which app.py and run_raw.py do not
    catch — the user would get a traceback instead of the one-line message
    this module promises. The wrapper turns it into a ConfigError naming
    the variable and the offending value.
    """
    with pytest.raises(ConfigError) as exc:
        load_settings({"MAX_TURNS": "eight"})
    assert "MAX_TURNS" in str(exc.value)
    assert "eight" in str(exc.value)
