from __future__ import annotations

import argparse
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from weave_agent_signals import cli
from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog


def _catalog():
    return build_model_catalog(which=lambda _name: "/bin/fake")


def _args(**updates):
    values = {
        "entity": "entity",
        "project": "project",
        "since": None,
        "limit": 10,
        "rubric": None,
        "judge_backend": "cli",
        "judge_models": ["claude-sonnet-5", "gpt-5.6-sol"],
        "review_depth": "selective",
        "second_opinion_margin": 0.1,
        "force": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _turn():
    turn = MagicMock()
    turn.trace_id = "turn-1"
    turn.conversation_id = "session-1"
    turn.started_at = datetime(2026, 7, 15, tzinfo=timezone.utc)
    turn.config_version = "cfg"
    turn.git_branch = "main"
    return turn


def test_default_policy_resolves_catalog_recommendations(monkeypatch):
    catalog = _catalog()
    monkeypatch.setattr(cli, "build_model_catalog", lambda: catalog)
    policy = cli._resolve_judge_policy(
        _args(judge_backend=None, judge_models=None, review_depth=None, second_opinion_margin=None)
    )
    backend = catalog.backend(catalog.recommended_judge_backend)
    assert [judge.id for judge in policy.judges] == list(backend.recommended_judges)
    assert policy.depth == backend.recommended_review_depth


def test_direct_cli_uses_one_session_plan_runner_and_in_memory_artifacts(monkeypatch, capsys):
    turn = _turn()
    weave = MagicMock()
    weave.query_turns.return_value = [turn]
    weave.__enter__.return_value = weave
    inference = MagicMock()
    inference.__enter__.return_value = inference
    plan = {
        "plan_id": "plan",
        "totals": {"sessions_planned": 1, "windows_planned": 2},
        "sessions": [],
    }
    judge = MagicMock(return_value=[])
    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    monkeypatch.setattr(cli, "build_rubric_catalog", build_rubric_catalog)
    monkeypatch.setattr(cli, "build_judging_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(cli, "_make_model_client", lambda *_args: inference)
    monkeypatch.setattr(cli, "judge_session", judge)
    rc = cli.cmd_judge(_args(rubric="judge.session_outcome"))
    assert rc == 0
    assert judge.call_count == 1
    kwargs = judge.call_args.kwargs
    assert kwargs["judging_plan"] is plan
    assert callable(kwargs["artifact_loader"])
    assert callable(kwargs["artifact_recorder"])
    assert "1 sessions across 2 reviewer windows" in capsys.readouterr().out


def test_direct_cli_rejects_unknown_rubric_before_reading_weave(monkeypatch, capsys):
    weave = MagicMock()
    monkeypatch.setattr(cli, "WeaveClient", weave)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    assert cli.cmd_judge(_args(rubric="judge.unknown")) == 2
    weave.assert_not_called()
    assert "Unknown rubric" in capsys.readouterr().out


def test_hydration_failure_aborts_before_inference(monkeypatch):
    weave = MagicMock()
    weave.query_turns.return_value = [_turn()]
    weave.hydrate_turns_batch.side_effect = RuntimeError("detail hydration truncated")
    weave.__enter__.return_value = weave
    make_client = MagicMock()
    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    monkeypatch.setattr(cli, "_make_model_client", make_client)
    with pytest.raises(RuntimeError, match="detail hydration truncated"):
        cli.cmd_judge(_args())
    make_client.assert_not_called()
