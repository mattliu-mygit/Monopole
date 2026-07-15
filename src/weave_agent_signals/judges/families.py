"""Model-family classification for audit metadata and selection warnings."""

from __future__ import annotations

_PREFIX_TO_FAMILY: list[tuple[str, str]] = [
    ("claude", "anthropic"),
    ("gpt-oss", "openai"),
    ("gpt-5", "openai"),
    ("gpt-4", "openai"),
    ("gpt-3", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
    ("llama", "meta"),
    ("granite", "ibm"),
    ("gemma", "google"),
    ("gemini", "google"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("phi", "microsoft"),
    ("command", "cohere"),
]


def model_family(model_id: str) -> str:
    if not model_id:
        return "unknown"
    lower = model_id.lower().rsplit("/", 1)[-1]
    for prefix, family in _PREFIX_TO_FAMILY:
        if lower.startswith(prefix):
            return family
    return "unknown"
