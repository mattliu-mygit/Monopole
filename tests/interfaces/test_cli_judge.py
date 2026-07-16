from __future__ import annotations

import argparse
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from weave_agent_signals import cli
from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.models import Score, SessionView


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


def test_group_sessions_breaks_equal_timestamp_ties_by_trace_id():
    turn_b = _turn()
    turn_b.trace_id = "turn-b"
    turn_a = _turn()
    turn_a.trace_id = "turn-a"

    sessions = cli._group_sessions([turn_b, turn_a])

    assert [turn.trace_id for turn in sessions[0].turns] == ["turn-a", "turn-b"]


def test_direct_cli_uses_one_session_plan_runner_and_in_memory_artifacts(monkeypatch, capsys):
    turn = _turn()
    session = SessionView(
        conversation_id="session-1",
        turns=[turn],
        config_version="cfg",
        git_branch="main",
    )
    weave = MagicMock()
    weave.query_turns.return_value = [turn]
    weave.query_session.return_value = session
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
    weave.query_session.assert_called_once_with("session-1")
    weave.hydrate_turns_batch.assert_called_once_with([turn])
    assert "1 sessions across 2 reviewer windows" in capsys.readouterr().out


def test_direct_cli_discovers_conversations_then_judges_complete_hydrated_sessions(monkeypatch):
    recent_one = _turn()
    recent_one.trace_id = "session-1-recent"
    duplicate_one = _turn()
    duplicate_one.trace_id = "session-1-other-recent"
    recent_two = _turn()
    recent_two.trace_id = "session-2-recent"
    recent_two.conversation_id = "session-2"
    older_one = _turn()
    older_one.trace_id = "session-1-older"
    older_one.started_at = datetime(2026, 7, 1, tzinfo=timezone.utc)

    session_one = SessionView(
        conversation_id="session-1",
        turns=[older_one, duplicate_one, recent_one],
        config_version="cfg",
        git_branch="main",
    )
    session_two = SessionView(
        conversation_id="session-2",
        turns=[recent_two],
        config_version="cfg",
        git_branch="main",
    )
    weave = MagicMock()
    weave.query_turns.return_value = [recent_two, recent_one, duplicate_one]
    weave.query_session.side_effect = [session_one, session_two]
    weave.query_existing_feedback.return_value = []
    weave.__enter__.return_value = weave
    inference = MagicMock()
    inference.__enter__.return_value = inference
    plan = {
        "plan_id": "plan",
        "totals": {"sessions_planned": 2, "windows_planned": 2},
        "sessions": [],
    }
    captured_sessions = []

    def build_plan(sessions, **_kwargs):
        captured_sessions.extend(sessions)
        return plan

    score = Score("judge.session_outcome", 0.75, [], {}, "session")
    judge = MagicMock(side_effect=[[score], []])
    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    monkeypatch.setattr(cli, "build_judging_plan", build_plan)
    monkeypatch.setattr(cli, "_make_model_client", lambda *_args: inference)
    monkeypatch.setattr(cli, "judge_session", judge)

    since = datetime(2026, 7, 14, tzinfo=timezone.utc)
    assert cli.cmd_judge(_args(limit=3, since=since, rubric="judge.session_outcome")) == 0

    weave.query_turns.assert_called_once_with(limit=3, since=since)
    assert [call.args[0] for call in weave.query_session.call_args_list] == [
        "session-1",
        "session-2",
    ]
    hydrated = weave.hydrate_turns_batch.call_args.args[0]
    assert [turn.trace_id for turn in hydrated] == [
        "session-1-older",
        "session-1-other-recent",
        "session-1-recent",
        "session-2-recent",
    ]
    assert captured_sessions == [session_one, session_two]
    assert judge.call_args_list[0].args[0] is session_one
    assert [turn.trace_id for turn in judge.call_args_list[0].args[0].turns] == [
        "session-1-older",
        "session-1-other-recent",
        "session-1-recent",
    ]
    assert weave.write_score.call_count == 1


def test_direct_cli_fails_when_complete_session_omits_a_discovery_root(monkeypatch):
    discovered = _turn()
    reloaded = _turn()
    reloaded.trace_id = "different-root"
    session = SessionView(
        conversation_id="session-1",
        turns=[reloaded],
        config_version="cfg",
        git_branch="main",
    )
    weave = MagicMock()
    weave.query_turns.return_value = [discovered]
    weave.query_session.return_value = session
    weave.__enter__.return_value = weave
    build_plan = MagicMock(side_effect=AssertionError("must not plan partial discovery"))
    make_client = MagicMock()
    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    monkeypatch.setattr(cli, "build_judging_plan", build_plan)
    monkeypatch.setattr(cli, "_make_model_client", make_client)

    with pytest.raises(RuntimeError, match="missing discovered root"):
        cli.cmd_judge(_args(rubric="judge.session_outcome"))

    weave.hydrate_turns_batch.assert_not_called()
    build_plan.assert_not_called()
    make_client.assert_not_called()
    weave.write_score.assert_not_called()


def test_direct_cli_rejects_unknown_rubric_before_reading_weave(monkeypatch, capsys):
    weave = MagicMock()
    monkeypatch.setattr(cli, "WeaveClient", weave)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    assert cli.cmd_judge(_args(rubric="judge.unknown")) == 2
    weave.assert_not_called()
    assert "Unknown rubric" in capsys.readouterr().out


def test_hydration_failure_aborts_before_inference(monkeypatch):
    turn = _turn()
    weave = MagicMock()
    weave.query_turns.return_value = [turn]
    weave.query_session.return_value = SessionView(
        conversation_id="session-1",
        turns=[turn],
        config_version="cfg",
        git_branch="main",
    )
    weave.hydrate_turns_batch.side_effect = RuntimeError("detail hydration truncated")
    weave.__enter__.return_value = weave
    make_client = MagicMock()
    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    monkeypatch.setattr(cli, "_make_model_client", make_client)
    with pytest.raises(RuntimeError, match="detail hydration truncated"):
        cli.cmd_judge(_args())
    make_client.assert_not_called()


def test_direct_cli_buffers_all_sessions_and_writes_nothing_on_later_failure(monkeypatch):
    turns = [_turn(), _turn()]
    turns[1].trace_id = "turn-2"
    turns[1].conversation_id = "session-2"
    weave = MagicMock()
    weave.query_turns.return_value = turns
    weave.__enter__.return_value = weave
    inference = MagicMock()
    inference.__enter__.return_value = inference
    sessions = []
    for index in (1, 2):
        session = MagicMock()
        session.conversation_id = f"session-{index}"
        session.turns = [turns[index - 1]]
        session.config_version = "cfg"
        session.git_branch = "main"
        session.ref_for.return_value = f"weave:///session-{index}"
        sessions.append(session)
    weave.query_session.side_effect = sessions
    plan = {
        "plan_id": "plan",
        "totals": {"sessions_planned": 2, "windows_planned": 2},
        "sessions": [],
    }
    score = Score("judge.session_outcome", 0.75, [], {}, "session")
    judge = MagicMock(side_effect=[[score], RuntimeError("later failure")])
    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    monkeypatch.setattr(cli, "build_judging_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(cli, "_make_model_client", lambda *_args: inference)
    monkeypatch.setattr(cli, "judge_session", judge)

    assert cli.cmd_judge(_args(rubric="judge.session_outcome")) == 1
    weave.write_score.assert_not_called()
    weave.query_existing_feedback.assert_not_called()
