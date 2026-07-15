from __future__ import annotations

import pytest

from weave_agent_signals.judges.families import model_family


@pytest.mark.parametrize(
    ("model_id", "family"),
    [
        ("claude-opus-4", "anthropic"),
        ("claude-sonnet-4-20250514", "anthropic"),
        ("gpt-oss-20b", "openai"),
        ("gpt-4o", "openai"),
        ("gpt-5.1", "openai"),
        ("o4-mini", "openai"),
        ("deepseek-v4", "deepseek"),
        ("qwen3-30b", "qwen"),
        ("Llama-3.1-8B", "meta"),
        ("granite-4.1-8b", "ibm"),
        ("gemini-2.5-flash", "google"),
        ("mistral-large", "mistral"),
    ],
)
def test_model_family(model_id: str, family: str) -> None:
    assert model_family(model_id) == family


@pytest.mark.parametrize(
    ("model_id", "family"),
    [
        ("openai/gpt-5.1", "openai"),
        ("anthropic/claude-sonnet-5", "anthropic"),
        ("google/gemini-2.5-pro", "google"),
    ],
)
def test_model_family_normalizes_provider_prefix(model_id: str, family: str) -> None:
    assert model_family(model_id) == family


@pytest.mark.parametrize("model_id", ["", "some-custom-model"])
def test_model_family_unknown(model_id: str) -> None:
    assert model_family(model_id) == "unknown"
