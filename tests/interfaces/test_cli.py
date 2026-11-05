from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from weave_agent_signals import cli
from weave_agent_signals.catalogs import build_model_catalog
from weave_agent_signals.ces_commands import Insights
from weave_agent_signals.patterns import ConfigStats, Stat
from weave_agent_signals.runs.bundles import bundle_from_content_map


def test_judge_parser_accepts_ordered_explicit_panel():
    args = cli.build_parser().parse_args(
        [
            "judge",
            "--judge-model",
            "claude:claude-sonnet-5",
            "--judge-model",
            "agy:gemini-3.1-pro-high",
        ]
    )

    assert args.judge_models == [
        "claude:claude-sonnet-5",
        "agy:gemini-3.1-pro-high",
    ]


def test_reflect_parser_exposes_explicit_evaluator_and_rejects_zero_budget():
    args = cli.build_parser().parse_args(
        [
            "reflect",
            "--target-registry",
            "targets.json",
            "--proposal-evaluator-model",
            "gpt-4o",
        ]
    )
    assert args.proposal_evaluator_model == "gpt-4o"

    with pytest.raises(SystemExit, match="2"):
        cli.build_parser().parse_args(
            [
                "reflect",
                "--target-registry",
                "targets.json",
                "--candidate-budget",
                "0",
            ]
        )


def test_serve_defaults_to_the_local_trust_boundary():
    args = cli.build_parser().parse_args(
        [
            "serve",
            "--target-registry",
            "targets.json",
            "--sandbox-runtime",
            "runtime.json",
        ]
    )

    assert args.host == "127.0.0.1"


def test_reflection_model_defaults_come_from_the_unified_catalog(monkeypatch):
    catalog = build_model_catalog(which=lambda _executable: "/bin/fake")
    monkeypatch.setattr(cli, "build_model_catalog", lambda: catalog)
    args = argparse.Namespace(
        model=None,
        proposal_evaluator_model=None,
    )

    writer, evaluator = cli._resolve_reflection_models(args)

    assert writer.id == catalog.recommended_proposal_model
    assert evaluator.id == catalog.proposal_evaluator_preferences[0]


def test_standalone_reflect_stops_before_local_or_model_setup_for_audit_only_feedback(
    tmp_path,
    monkeypatch,
    capsys,
):
    feedback = [
        {
            "feedback_type": "weave_agent_signals.judge.verification",
            "payload": {
                "rating": 0.2,
                "granularity": "turn",
                "details": {"evaluation_unit": "episode"},
            },
        }
    ]

    monkeypatch.setattr(cli.ces_commands, "read_scores", lambda _args: feedback)
    monkeypatch.setattr(
        cli,
        "load_target_registry",
        lambda _path: pytest.fail("audit-only feedback must stop before target setup"),
    )
    monkeypatch.setattr(
        cli,
        "_make_model_client",
        lambda *_args: pytest.fail("audit-only feedback must not open model clients"),
    )

    rc = cli.cmd_reflect(
        argparse.Namespace(
            entity="entity",
            project="project",
            limit=10,
            target_registry=str(tmp_path / "targets.json"),
            model="codex:gpt-5.6-sol",
            proposal_evaluator_model="claude:claude-sonnet-5",
            candidate_budget=3,
        )
    )

    assert rc == 0
    assert "No eligible scored feedback found" in capsys.readouterr().out


def test_standalone_reflect_always_infers_with_resolved_roles_and_only_prints_diff(
    tmp_path,
    monkeypatch,
    capsys,
):
    catalog = build_model_catalog(which=lambda _executable: "/bin/fake")
    baseline = bundle_from_content_map({"CLAUDE.md": "before\n"})
    proposed = bundle_from_content_map({"CLAUDE.md": "after\n"})
    candidate = SimpleNamespace(
        candidate_id="candidate-1",
        bundle=proposed,
        score=0.8,
        score_delta=0.2,
        resolved_writer_model="gpt-5.6-sol",
    )
    result = SimpleNamespace(
        baseline=baseline,
        baseline_score=0.6,
        candidates=(candidate,),
        recommended_candidate_id=candidate.candidate_id,
        baseline_won=False,
    )
    feedback = [
        {
            "feedback_type": "weave_agent_signals.outcome.test",
            "payload": {"rating": 0.5},
        },
        {
            "feedback_type": "weave_agent_signals.judge.verification",
            "payload": {
                "rating": 0.2,
                "granularity": "turn",
                "details": {"evaluation_unit": "episode"},
            },
        },
    ]
    adapter = MagicMock()
    adapter.capture.return_value = baseline
    reflect = MagicMock(return_value=result)

    monkeypatch.setattr(cli.ces_commands, "read_scores", lambda _args: feedback)
    monkeypatch.setattr(cli, "load_target_registry", lambda _path: object())
    monkeypatch.setattr(cli, "TargetPromoter", lambda _registry: adapter)
    monkeypatch.setattr(cli, "build_model_catalog", lambda: catalog)
    monkeypatch.setattr(cli, "coaching_digest", lambda _feedback: "coaching")
    monkeypatch.setattr(cli, "_make_model_client", lambda _args, _model: nullcontext(object()))
    monkeypatch.setattr(cli, "run_reflection", reflect)

    rc = cli.cmd_reflect(
        argparse.Namespace(
            entity="entity",
            project="project",
            limit=10,
            target_registry=str(tmp_path / "targets.json"),
            model="codex:gpt-5.6-sol",
            proposal_evaluator_model="claude:claude-sonnet-5",
            candidate_budget=3,
        )
    )

    assert rc == 0
    call = reflect.call_args.kwargs
    assert call["feedback"] == [
        {
            "feedback_type": "weave_agent_signals.outcome.test",
            "payload": {"rating": 0.5},
        }
    ]
    assert call["baseline"] is baseline
    assert call["requested_writer"].id == "codex:gpt-5.6-sol"
    assert call["requested_evaluator"].id == "claude:claude-sonnet-5"
    assert call["resolve_locator"] == adapter.resolve_locator
    assert call["scope_policy"] == adapter.contract_manifest.return_value
    output = capsys.readouterr().out
    assert "proposal_writer=codex:gpt-5.6-sol" in output
    assert "proposal_evaluator=claude:claude-sonnet-5" in output
    assert "--- A/CLAUDE.md" in output
    assert "+++ B/CLAUDE.md" in output
    assert "preview-only" in output


def test_bundle_diff_separates_multiple_files_without_trailing_newlines():
    past = bundle_from_content_map({"CLAUDE.md": "before", ".claude/skills/review.md": "old"})
    proposed = bundle_from_content_map({"CLAUDE.md": "after", ".claude/skills/review.md": "new"})

    rendered = cli._render_bundle_diff(past, proposed)

    assert "-old\n+new\n--- A/CLAUDE.md" in rendered
    assert "-before\n+after\n" in rendered


def test_cli_help_lists_current_commands():
    parser = cli.build_parser()
    help_text = parser.format_help()

    for command in ("score", "backfill", "judge", "inspect", "monitor", "reflect"):
        assert command in help_text


@pytest.mark.parametrize("command", ["judge", "reflect"])
def test_model_help_describes_catalog_selection(command, capsys):
    with pytest.raises(SystemExit, match="0"):
        cli.build_parser().parse_args([command, "--help"])

    output = capsys.readouterr().out
    assert "model ID" in output
    assert "custom URL" not in output


def _analysis(monkeypatch, records):
    stats = ConfigStats(
        {"v1": {"outcome.test": Stat(3, 1.0, True)}, "v2": {"outcome.test": Stat(1, 0.0, True)}},
        {"v1": 3, "v2": 1},
    )
    asked = []

    def read_analysis(_args, *, feedback: bool):
        asked.append(feedback)
        return stats, records if feedback else []

    monkeypatch.setattr(cli.ces_commands, "read_analysis", read_analysis)
    return asked


def test_analyze_ab_reads_only_query_stats(monkeypatch, capsys):
    asked = _analysis(monkeypatch, [])

    assert cli.main(["analyze", "--ab"]) == 0

    out = capsys.readouterr().out
    assert asked == [False]
    assert "Config: v1  (3 evaluated targets)" in out
    assert "outcome.test" in out and "100% pass  (n=3" in out


def test_analyze_digest_summarizes_query_stats_with_raw_examples(monkeypatch, capsys):
    asked = _analysis(
        monkeypatch, [{"feedback_type": "weave_agent_signals.efficiency", "payload": {"rating": 1}}]
    )

    assert cli.main(["analyze"]) == 0

    out = capsys.readouterr().out
    assert asked == [True]
    assert "**outcome.test**: 75% pass rate (n=4" in out
    assert "efficiency" not in out  # raw Scores feed only examples and trends
    assert "## A/B Comparison" in out


def _insights(monkeypatch, **found):
    asked = []

    def read_insights(args):
        asked.append(args.subagents)
        return Insights(found.get("subagents"))

    def read_analysis(*_args, **_kwargs):
        pytest.fail("an insight flag alone reads no Score analysis")

    monkeypatch.setattr(cli.ces_commands, "read_insights", read_insights)
    monkeypatch.setattr(cli.ces_commands, "read_analysis", read_analysis)
    return asked


def test_analyze_subagents_reports_trajectory_counts_by_config(monkeypatch, capsys):
    counts = {"v1": Counter({0: 40, 1: 10, 2: 2}), "v2": Counter({0: 5})}
    asked = _insights(monkeypatch, subagents=counts)

    assert cli.main(["analyze", "--subagents"]) == 0

    out = capsys.readouterr().out
    assert asked == [True]
    assert (
        "Config: v1  sessions=40  subagents=12 (0.30 per session; depth 1: 10, depth 2: 2)" in out
    )
    assert "Config: v2  sessions=5  subagents=0 (0.00 per session)" in out


def test_analyze_with_an_insight_and_ab_reads_both(monkeypatch, capsys):
    asked = _analysis(monkeypatch, [])
    monkeypatch.setattr(
        cli.ces_commands,
        "read_insights",
        lambda args: Insights({"v1": Counter({0: 1})}),
    )

    assert cli.main(["analyze", "--ab", "--subagents"]) == 0

    out = capsys.readouterr().out
    assert asked == [False]
    assert "A/B Leaderboard" in out and "sessions=1" in out
    assert "# Coaching Digest" not in out
