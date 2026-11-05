from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from weave_agent_signals import cli
from weave_agent_signals.catalogs import build_model_catalog
from weave_agent_signals.ces_commands import Insights
from weave_agent_signals.ces_queries import Group
from weave_agent_signals.patterns import ConfigStats, SessionSuccess, Stat
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

    def read_analysis(_args, *, feedback: bool, success: bool = False, trends: bool = False):
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
        asked.append((args.subagents, args.judges, args.segments))
        return Insights(found.get("subagents"), found.get("judge_tables"), found.get("segments"))

    def read_analysis(*_args, **_kwargs):
        pytest.fail("an insight flag alone reads no Score analysis")

    monkeypatch.setattr(cli.ces_commands, "read_insights", read_insights)
    monkeypatch.setattr(cli.ces_commands, "read_analysis", read_analysis)
    return asked


TESTS_PASSED = "outcome.session_tests_passed"


def _depths(targets, rates=None):
    """{depth: Group} with each depth's targets and its tests-passed Stat."""
    return {
        depth: Group(
            (None, depth),
            n,
            {} if rates is None or depth not in rates else {TESTS_PASSED: rates[depth]},
        )
        for depth, n in targets.items()
    }


def test_analyze_subagents_reports_counts_and_compares_outcome_rates(monkeypatch, capsys):
    found = {
        # 35 of 50 sessions passed; subagents at depth 1 and 2 pool to 12 of 30.
        "v1": _depths(
            {0: 60, 1: 30, 2: 5},
            {0: Stat(50, 0.7, True), 1: Stat(25, 0.4, True), 2: Stat(5, 0.4, True)},
        ),
        "v2": _depths({0: 5}),
    }
    asked = _insights(monkeypatch, subagents=found)

    assert cli.main(["analyze", "--subagents"]) == 0

    out = capsys.readouterr().out
    assert asked == [(True, False, None)]
    assert (
        "Config: v1  sessions=60  subagents=35 (0.58 per session; depth 1: 30, depth 2: 5)" in out
    )
    assert (
        f"    {TESTS_PASSED:35s} sessions 70% (n=50) vs subagents 40% (n=30)  -0.30, "
        "CI -0.49–-0.08  significant"
    ) in out
    assert "Config: v2  sessions=5  subagents=0 (0.00 per session)" in out


def test_analyze_summary_shows_success_rate_and_spend_per_success(monkeypatch, capsys):
    found = {
        "v1": SessionSuccess(
            3, 2, 5, {"measure.tokens": 11000.0, "measure.turns": 32.0, "measure.duration_s": 900.0}
        ),
        "v2": SessionSuccess(0, 0, 4, {}),
        "v3": SessionSuccess(0, 2, 0, {"measure.tokens": 10.0}),
    }

    def read_analysis(_args, *, feedback, success=False, trends=False):
        stats = ConfigStats({"v1": {"outcome.test": Stat(3, 1.0, True)}}, {"v1": 3})
        return replace(stats, success=found if success else None), []

    monkeypatch.setattr(cli.ces_commands, "read_analysis", read_analysis)

    assert cli.main(["analyze", "--ab"]) == 0

    out = capsys.readouterr().out
    assert "Config: v1  success 60% (3 of 5, CI 23%–88%)  no test run: 5" in out
    assert "    per success: 3,667 tokens, 10.7 turns, 300 s" in out
    assert "Config: v2  no session ran tests (4 sessions)" in out
    assert "Config: v3  success 0% (0 of 2" in out and "per success: no success yet" in out


def test_analyze_judges_reports_agreement_and_the_gap(monkeypatch, capsys):
    tables = [
        (
            "judge.verification.panel-abc",
            "implicit.completion",
            {(True, 0.75): 6, (True, 0.25): 2, (False, 0.25): 3, (False, 0.75): 1},
        ),
        ("judge.verification.panel-abc", "judge.verification.panel-new", {}),
    ]
    _insights(monkeypatch, judge_tables=tables)

    assert cli.main(["analyze", "--judges"]) == 0

    out = capsys.readouterr().out
    assert "judge.verification.panel-abc vs implicit.completion" in out
    assert (
        "n=12  agree 75% (CI 47%–91%)  judge mean 0.62 when the reference passes vs 0.38 when "
        "it fails (+0.25, CI -0.04–+0.54)\n"
    ) in out
    assert "no session has both Scores" in out


def test_analyze_segments_reports_only_segments_with_enough_sessions(monkeypatch, capsys):
    completion = "implicit.completion"
    segments = {
        ("v1", "main"): {completion: Stat(70, 56 / 70, True)},
        ("v1", "dev"): {completion: Stat(80, 48 / 80, True)},
        ("v1", "tiny"): {completion: Stat(3, 1.0, True)},
    }
    asked = _insights(monkeypatch, segments=segments)

    assert cli.main(["analyze", "--segments", "git_branch"]) == 0

    out = capsys.readouterr().out
    assert asked == [(False, False, "git_branch")]
    assert "Segments by git_branch" in out
    # The 3 "tiny" sessions join the rest of v1 but get no line of their own.
    assert "main  implicit.completion" in out and "80% vs 61%  (n=70 vs 83" in out
    assert "significant" in out
    assert "tiny" not in out


def test_analyze_segments_says_when_no_session_has_the_field(monkeypatch, capsys):
    segments = {("v1", None): {"implicit.completion": Stat(70, 0.8, True)}}
    asked = _insights(monkeypatch, segments=segments)

    assert cli.main(["analyze", "--segments"]) == 0

    assert asked == [(False, False, "repo")]
    assert "No scored session has a dim.repo Score yet" in capsys.readouterr().out


def test_analyze_with_an_insight_and_ab_reads_both(monkeypatch, capsys):
    asked = _analysis(monkeypatch, [])
    monkeypatch.setattr(
        cli.ces_commands,
        "read_insights",
        lambda args: Insights({"v1": _depths({0: 1})}, None, None),
    )

    assert cli.main(["analyze", "--ab", "--subagents"]) == 0

    out = capsys.readouterr().out
    assert asked == [False]
    assert "A/B Leaderboard" in out and "sessions=1" in out
    assert "# Coaching Digest" not in out
