"""Model family mapping for bias-free judge selection.

Agent is Claude → judges must be non-Anthropic. gpt-oss counts as OpenAI-family
(shared training distribution, arXiv:2410.21819).
"""
from __future__ import annotations

import logging

log = logging.getLogger("weave_agent_signals.judges")

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
    lower = model_id.lower()
    for prefix, family in _PREFIX_TO_FAMILY:
        if lower.startswith(prefix):
            return family
    return "unknown"


def select_judges(
    agent_model: str | None,
    candidates: list[str],
    n: int = 1,
) -> list[str]:
    """Select up to n judge models from candidates, excluding the agent's family."""
    exclude = model_family(agent_model) if agent_model else ""
    eligible = [m for m in candidates if model_family(m) != exclude]
    if not eligible:
        eligible = candidates
    seen_families: set[str] = set()
    selected: list[str] = []
    for m in eligible:
        fam = model_family(m)
        if fam not in seen_families:
            selected.append(m)
            seen_families.add(fam)
            if len(selected) >= n:
                break
    # The anti-self-judging guarantee only holds when a cross-family candidate
    # exists; a single-family roster forces the fallback above to reuse the
    # agent's family. Surface that rather than silently biasing the score.
    if exclude and any(model_family(m) == exclude for m in selected):
        log.warning(
            "No cross-family judge available for agent family %r; using "
            "same-family judge(s) %s (self-preference bias risk)",
            exclude, selected,
        )
    return selected


def select_panel(
    agent_model: str | None,
    candidates: list[str],
    *,
    max_same: int = 1,
    min_cross: int = 2,
) -> list[str]:
    """Build a PoLL panel: up to ``max_same`` same-family judge(s) plus one judge
    per distinct non-same-family (targeting ``min_cross`` cross-family minimum).

    A single strong same-family reader (e.g. Claude judging a Claude agent),
    outvoted by ≥2 cross-family judges, trades a little self-preference bias
    (arXiv:2410.21819) for capability. Warns when the candidate pool can't seat
    the target (e.g. a single-family backend). When the agent family is unknown,
    no same-family slot is filled and the panel is purely cross-family.
    """
    agent_fam = model_family(agent_model) if agent_model else ""
    same: list[str] = []
    cross: dict[str, str] = {}
    for m in candidates:
        fam = model_family(m)
        if agent_fam and fam == agent_fam:
            if len(same) < max_same:
                same.append(m)
        elif fam not in cross:
            cross[fam] = m
    panel = same + list(cross.values())
    if len(cross) < min_cross:
        log.warning(
            "PoLL panel wants >=%d non-same-family judges but only %d available "
            "on this backend (%s); scoring with %d judge(s).",
            min_cross, len(cross), list(cross), len(panel),
        )
    return panel
