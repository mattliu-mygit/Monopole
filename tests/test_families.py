from __future__ import annotations

from weave_agent_signals.judges.families import model_family, select_judges, select_panel


def test_model_family_claude():
    assert model_family("claude-opus-4") == "anthropic"
    assert model_family("claude-sonnet-4-20250514") == "anthropic"


def test_model_family_openai():
    assert model_family("gpt-oss-20b") == "openai"
    assert model_family("gpt-4o") == "openai"
    assert model_family("gpt-5.1") == "openai"
    assert model_family("o4-mini") == "openai"


def test_cli_roster_pairs_cross_family_for_claude_agent():
    # The temporary CLI backend roster must offer a non-Anthropic judge for a
    # Claude agent (→ routed to the Codex CLI).
    candidates = ["claude-sonnet-5", "gpt-5.1"]
    selected = select_judges("claude-opus-4-8", candidates, n=1)
    assert selected == ["gpt-5.1"]


def test_model_family_others():
    assert model_family("deepseek-v4") == "deepseek"
    assert model_family("qwen3-30b") == "qwen"
    assert model_family("Llama-3.1-8B") == "meta"
    assert model_family("granite-4.1-8b") == "ibm"
    assert model_family("gemini-2.5-flash") == "google"
    assert model_family("mistral-large") == "mistral"


def test_model_family_unknown():
    assert model_family("") == "unknown"
    assert model_family("some-custom-model") == "unknown"


def test_select_judges_excludes_agent_family():
    candidates = ["gpt-oss-20b", "Llama-3.1-8B", "granite-4.1-8b"]
    selected = select_judges("claude-opus-4", candidates, n=1)
    assert len(selected) == 1
    assert model_family(selected[0]) != "anthropic"


def test_select_judges_excludes_openai_for_gpt_agent():
    candidates = ["gpt-oss-20b", "Llama-3.1-8B", "granite-4.1-8b"]
    selected = select_judges("gpt-4o", candidates, n=1)
    assert "gpt-oss-20b" not in selected


def test_select_judges_diverse_families():
    candidates = ["gpt-oss-20b", "Llama-3.1-8B", "granite-4.1-8b", "deepseek-v4"]
    selected = select_judges("claude-opus-4", candidates, n=3)
    assert len(selected) == 3
    families = [model_family(m) for m in selected]
    assert len(set(families)) == 3


def test_select_judges_fallback_when_all_excluded():
    candidates = ["claude-sonnet-4"]
    selected = select_judges("claude-opus-4", candidates, n=1)
    assert len(selected) == 1


def test_select_judges_none_agent():
    candidates = ["gpt-oss-20b", "Llama-3.1-8B"]
    selected = select_judges(None, candidates, n=1)
    assert len(selected) == 1


# --- select_panel: 1 same-family + >=2 non-same-family ---

def test_select_panel_one_same_two_cross_for_claude_agent():
    candidates = ["claude-sonnet-5", "gpt-5.1", "gemini-2.5-pro"]
    panel = select_panel("claude-opus-4-8", candidates)
    fams = [model_family(m) for m in panel]
    assert len(panel) == 3
    assert fams.count("anthropic") == 1          # exactly one same-family
    assert set(fams) == {"anthropic", "openai", "google"}  # + two distinct cross


def test_select_panel_dedups_within_cross_family():
    candidates = ["claude-sonnet-5", "gpt-5.1", "gpt-4o", "gemini-2.5-pro"]
    panel = select_panel("claude-opus-4-8", candidates)
    fams = [model_family(m) for m in panel]
    assert fams.count("openai") == 1             # only one openai despite two candidates


def test_select_panel_pure_cross_family_when_no_same_available():
    # wandb-style roster: no Anthropic model → panel is all cross-family.
    candidates = ["gpt-oss-120b", "DeepSeek-V4", "Qwen3-30B"]
    panel = select_panel("claude-opus-4-8", candidates)
    assert len(panel) == 3
    assert all(model_family(m) != "anthropic" for m in panel)


def test_select_panel_degrades_when_too_few_cross():
    # only one cross-family available → panel can't meet the >=2 minimum.
    panel = select_panel("claude-opus-4-8", ["claude-sonnet-5", "gpt-5.1"])
    fams = [model_family(m) for m in panel]
    assert fams.count("anthropic") == 1 and fams.count("openai") == 1
