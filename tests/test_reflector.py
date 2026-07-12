"""Tests for the GEPA reflector: artifact extraction, feedback formatting, diff output."""
from __future__ import annotations

import textwrap
from unittest.mock import MagicMock

from weave_agent_signals.reflector import (
    Artifact,
    Proposal,
    _make_artifact_evaluator,
    extract_artifacts,
    format_evaluation_batch,
    render_proposal_diff,
)


def test_extract_artifacts_from_paths(tmp_path):
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Project\nBe concise.")
    skill_dir = tmp_path / ".claude" / "commands"
    skill_dir.mkdir(parents=True)
    (skill_dir / "fix.md").write_text("Run the fix command.")

    artifacts = extract_artifacts(str(tmp_path))
    assert len(artifacts) >= 1
    by_name = {a.name: a for a in artifacts}
    assert "CLAUDE.md" in by_name
    assert "Be concise" in by_name["CLAUDE.md"].content


def test_extract_artifacts_missing_claude_md(tmp_path):
    artifacts = extract_artifacts(str(tmp_path))
    assert len(artifacts) == 0


def test_format_evaluation_batch():
    feedback = [
        {
            "feedback_type": "weave_agent_signals.judge.verification",
            "payload": {
                "rating": 0.3,
                "tags": ["no_verification"],
                "reason": "Agent never ran tests after editing code.",
                "details": {
                    "rationale": "Agent never ran tests after editing code.",
                    "config_version": "v1",
                },
            },
        },
        {
            "feedback_type": "weave_agent_signals.outcome.test",
            "payload": {
                "rating": 0.0,
                "tags": ["fail"],
                "reason": "3 tests failed",
                "details": {"config_version": "v1"},
            },
        },
    ]
    batch = format_evaluation_batch(feedback)
    assert len(batch) >= 1
    entry = batch[0]
    assert "score" in entry
    assert "feedback" in entry


def test_render_proposal_diff():
    original = Artifact(name="CLAUDE.md", path="CLAUDE.md", content="# Old\nBe verbose.")
    proposed = Artifact(name="CLAUDE.md", path="CLAUDE.md", content="# New\nBe concise.")
    proposal = Proposal(
        artifacts=[proposed],
        rationale="Analysis shows verbosity correlates with lower scores.",
        score_delta=0.15,
    )
    diff = render_proposal_diff([original], proposal)
    assert "CLAUDE.md" in diff
    assert "concise" in diff
    assert "-" in diff or "+" in diff


def test_render_proposal_no_change():
    original = Artifact(name="CLAUDE.md", path="CLAUDE.md", content="# Same")
    proposed = Artifact(name="CLAUDE.md", path="CLAUDE.md", content="# Same")
    proposal = Proposal(artifacts=[proposed], rationale="No changes needed.", score_delta=0.0)
    diff = render_proposal_diff([original], proposal)
    assert "no changes" in diff.lower() or diff.strip() == ""


# --- evaluator (candidate ranking signal) ---

def test_artifact_evaluator_scores_candidate_via_judge():
    """The evaluator must score the PROPOSED candidate, not echo a constant."""
    client = MagicMock()
    resp = MagicMock()
    client.chat_json.return_value = ({"score": 0.82, "rationale": "addresses test gaps"}, resp)

    evaluator = _make_artifact_evaluator(
        judge_client=client, judge_model="gpt-4o",
        coaching_text="test pass rate is low", baseline=0.4,
    )
    score, side_info = evaluator({"CLAUDE.md": "Always run tests after editing."})

    assert score == 0.82
    # The proposed artifact text must reach the judge.
    _, kwargs = client.chat_json.call_args
    sent = str(kwargs["messages"])
    assert "Always run tests after editing." in sent
    assert side_info["predicted_quality"] == 0.82


def test_artifact_evaluator_varies_with_candidate():
    """Different candidates must be able to receive different scores."""
    client = MagicMock()
    resp = MagicMock()
    scores = iter([
        ({"score": 0.3, "rationale": "weak"}, resp),
        ({"score": 0.9, "rationale": "strong"}, resp),
    ])
    client.chat_json.side_effect = lambda **kw: next(scores)

    evaluator = _make_artifact_evaluator(
        judge_client=client, judge_model="gpt-4o",
        coaching_text="x", baseline=0.5,
    )
    s1, _ = evaluator({"CLAUDE.md": "weak version"})
    s2, _ = evaluator({"CLAUDE.md": "strong version"})
    assert s1 != s2


def test_artifact_evaluator_without_judge_falls_back_to_baseline():
    evaluator = _make_artifact_evaluator(
        judge_client=None, judge_model=None, coaching_text="x", baseline=0.42,
    )
    score, side_info = evaluator({"CLAUDE.md": "anything"})
    assert score == 0.42
    assert side_info["ranking_enabled"] is False
