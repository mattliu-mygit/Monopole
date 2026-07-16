from __future__ import annotations

import argparse
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from weave_agent_signals import cli
from weave_agent_signals.catalogs import build_model_catalog
from weave_agent_signals.runs.bundles import bundle_from_content_map


def test_judge_parser_accepts_ordered_explicit_review_policy():
    args = cli.build_parser().parse_args(
        [
            "judge",
            "--judge-backend",
            "cli",
            "--review-depth",
            "selective",
            "--judge-model",
            "claude-sonnet-5",
            "--judge-model",
            "gpt-5.6-sol",
            "--second-opinion-margin",
            "0.1",
        ]
    )

    assert args.judge_models == ["claude-sonnet-5", "gpt-5.6-sol"]
    assert args.review_depth == "selective"
    assert args.second_opinion_margin == 0.1


def test_reflect_parser_exposes_explicit_evaluator_and_rejects_zero_budget():
    args = cli.build_parser().parse_args(["reflect", "--proposal-evaluator-model", "gpt-4o"])
    assert args.proposal_evaluator_model == "gpt-4o"

    with pytest.raises(SystemExit, match="2"):
        cli.build_parser().parse_args(["reflect", "--candidate-budget", "0"])


def test_serve_defaults_to_the_local_trust_boundary():
    args = cli.build_parser().parse_args(["serve"])

    assert args.host == "127.0.0.1"


def test_reflection_model_defaults_come_from_the_selected_catalog_backend(monkeypatch):
    catalog = build_model_catalog(which=lambda _executable: "/bin/fake")
    monkeypatch.setattr(cli, "build_model_catalog", lambda: catalog)
    args = argparse.Namespace(
        model=None,
        judge_backend="cli",
        proposal_evaluator_model=None,
    )

    writer, evaluator = cli._resolve_reflection_models(args)

    assert writer.id == catalog.proposal.recommended_model
    assert evaluator.id == catalog.backend("cli").proposal_evaluator_preferences[0]


def test_standalone_reflect_stops_before_local_or_model_setup_for_audit_only_feedback(
    tmp_path,
    monkeypatch,
    capsys,
):
    weave = MagicMock()
    weave.__enter__.return_value.query_project_feedback.return_value = [
        {
            "feedback_type": "weave_agent_signals.judge.verification",
            "payload": {
                "rating": 0.2,
                "granularity": "turn",
                "details": {"evaluation_unit": "episode"},
            },
        }
    ]

    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(
        cli,
        "ProjectFileAdapter",
        lambda _root: pytest.fail("audit-only feedback must stop before target setup"),
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
            project_root=str(tmp_path),
            model="gpt-5.6-sol",
            judge_backend="cli",
            proposal_evaluator_model="claude-sonnet-5",
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
    weave = MagicMock()
    weave.__enter__.return_value.query_project_feedback.return_value = [
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

    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(cli, "ProjectFileAdapter", lambda _root: adapter)
    monkeypatch.setattr(cli, "build_model_catalog", lambda: catalog)
    monkeypatch.setattr(cli, "coaching_digest", lambda _feedback: "coaching")
    monkeypatch.setattr(cli, "_make_model_client", lambda _args, _model: nullcontext(object()))
    monkeypatch.setattr(cli, "run_reflection", reflect)

    rc = cli.cmd_reflect(
        argparse.Namespace(
            entity="entity",
            project="project",
            limit=10,
            project_root=str(tmp_path),
            model="gpt-5.6-sol",
            judge_backend="cli",
            proposal_evaluator_model="claude-sonnet-5",
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
    assert call["requested_writer"].id == "gpt-5.6-sol"
    assert call["requested_evaluator"].id == "claude-sonnet-5"
    assert call["build_candidate"] == adapter.bundle_from_content_map
    assert call["scope_policy"] == adapter.contract_manifest.return_value
    output = capsys.readouterr().out
    assert "proposal_writer=gpt-5.6-sol" in output
    assert "proposal_evaluator=claude-sonnet-5" in output
    assert "--- B/CLAUDE.md" in output
    assert "+++ C/CLAUDE.md" in output
    assert "preview-only" in output


def test_bundle_diff_separates_multiple_files_without_trailing_newlines():
    past = bundle_from_content_map({"CLAUDE.md": "before", ".claude/skills/review.md": "old"})
    proposed = bundle_from_content_map({"CLAUDE.md": "after", ".claude/skills/review.md": "new"})

    rendered = cli._render_bundle_diff(past, proposed)

    assert "-old\n+new\n--- B/CLAUDE.md" in rendered
    assert "-before\n+after\n" in rendered


def test_cli_help_lists_current_commands():
    parser = cli.build_parser()
    help_text = parser.format_help()

    for command in ("score", "backfill", "judge", "inspect", "monitor", "reflect"):
        assert command in help_text


@pytest.mark.parametrize("command", ["judge", "reflect"])
def test_model_backend_help_describes_catalog_selection(command, capsys):
    with pytest.raises(SystemExit, match="0"):
        cli.build_parser().parse_args([command, "--help"])

    output = capsys.readouterr().out
    assert "catalog backend ID" in output
    assert "custom URL" not in output
