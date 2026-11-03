from __future__ import annotations

import argparse
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from weave_agent_signals import ces_commands, cli
from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.ces_client import CesSession
from weave_agent_signals.models import Score, SessionView, TraceRole


def _catalog():
    return build_model_catalog(which=lambda _name: "/bin/fake")


def _args(**updates):
    values = {
        "entity": "entity",
        "project": "project",
        "since": None,
        "limit": 10,
        "rubric": None,
        "judge_models": ["claude:claude-sonnet-5", "codex:gpt-5.6-sol"],
        "force": False,
        "ces_url": None,
        "ces_scope_ref": None,
        "ces_dataset_id": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _turn(trace_id="turn-1", conversation_id="session-1", role=TraceRole.AGENT_SESSION):
    turn = MagicMock()
    turn.trace_id = trace_id
    turn.conversation_id = conversation_id
    turn.started_at = datetime(2026, 7, 15, tzinfo=timezone.utc)
    turn.config_version = "cfg"
    turn.git_branch = "main"
    turn.trace_role = role
    return turn


def _session(index: int) -> SessionView:
    return SessionView(
        f"session-{index}", [_turn(f"turn-{index}", f"session-{index}")], "cfg", "main"
    )


@pytest.fixture
def ces(monkeypatch):
    """The CES session cmd_judge opens; tests set which sessions it finds."""
    client = MagicMock()
    client.close = MagicMock()
    monkeypatch.setattr(ces_commands, "connect", lambda _args: CesSession(client, None))
    return client


def _plan(sessions: int):
    return {
        "plan_id": "plan",
        "totals": {"sessions_planned": sessions, "windows_planned": 2},
        "sessions": [],
    }


def test_default_panel_resolves_catalog_recommendations(monkeypatch):
    catalog = _catalog()
    monkeypatch.setattr(cli, "build_model_catalog", lambda: catalog)
    judges = cli._resolve_judge_panel(_args(judge_models=None))
    assert [judge.id for judge in judges] == list(catalog.recommended_judges)


def test_direct_cli_uses_one_session_plan_runner_and_in_memory_calls(ces, monkeypatch, capsys):
    found = {}
    inference = MagicMock()
    inference.__enter__.return_value = inference
    plan = _plan(1)
    judge = MagicMock(return_value=[])
    monkeypatch.setattr(
        ces_commands,
        "sessions_to_judge",
        lambda _client, _catalog, **kwargs: found.update(kwargs) or [_session(1)],
    )
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    monkeypatch.setattr(cli, "build_rubric_catalog", build_rubric_catalog)
    monkeypatch.setattr(cli, "build_canonical_judging_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(cli, "judging_plan_totals", lambda value: value["totals"])
    monkeypatch.setattr(cli, "_make_model_client", lambda *_args: inference)
    monkeypatch.setattr(cli, "judge_session", judge)

    rc = cli.cmd_judge(_args(rubric="judge.session_outcome"))

    assert rc == 0
    assert found["limit"] == 10
    assert judge.call_count == 1
    kwargs = judge.call_args.kwargs
    assert kwargs["judging_plan"] is plan
    assert [item.id for item in kwargs["judges"]] == ["claude:claude-sonnet-5", "codex:gpt-5.6-sol"]
    assert callable(kwargs["call_loader"])
    assert callable(kwargs["call_recorder"])
    assert "1 sessions across 2 reviewer windows" in capsys.readouterr().out
    ces.close.assert_called_once()


def test_direct_cli_rejects_unknown_rubric_before_connecting_to_ces(monkeypatch, capsys):
    monkeypatch.setattr(ces_commands, "connect", lambda _args: pytest.fail("must not connect"))
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)

    assert cli.cmd_judge(_args(rubric="judge.unknown")) == 2
    assert "Unknown rubric" in capsys.readouterr().out


def test_a_session_read_failure_aborts_before_inference(ces, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("session read failed")

    make_client = MagicMock()
    monkeypatch.setattr(ces_commands, "sessions_to_judge", fail)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    monkeypatch.setattr(cli, "_make_model_client", make_client)

    with pytest.raises(RuntimeError, match="session read failed"):
        cli.cmd_judge(_args())
    make_client.assert_not_called()


def test_direct_cli_buffers_all_sessions_and_writes_nothing_on_later_failure(ces, monkeypatch):
    inference = MagicMock()
    inference.__enter__.return_value = inference
    score = Score("judge.session_outcome", 0.75, [], {"review_status": "complete"}, "session")
    judge = MagicMock(side_effect=[[score], RuntimeError("later failure")])
    write = MagicMock()
    monkeypatch.setattr(
        ces_commands, "sessions_to_judge", lambda *_args, **_kwargs: [_session(1), _session(2)]
    )
    monkeypatch.setattr(ces_commands, "write_judge_scores", write)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    monkeypatch.setattr(cli, "build_canonical_judging_plan", lambda *_args, **_kwargs: _plan(2))
    monkeypatch.setattr(cli, "judging_plan_totals", lambda value: value["totals"])
    monkeypatch.setattr(cli, "_make_model_client", lambda *_args: inference)
    monkeypatch.setattr(cli, "judge_session", judge)

    assert cli.cmd_judge(_args(rubric="judge.session_outcome")) == 1
    write.assert_not_called()


def test_sessions_to_judge_keeps_only_agent_sessions_with_turns():
    client = MagicMock()
    client.find_sessions.return_value = ["agent", "mixed", "empty"]
    sessions = {
        "agent": SessionView("agent", [_turn("a1", "agent")], "cfg", "main"),
        "mixed": SessionView(
            "mixed",
            [_turn("m1", "mixed"), _turn("m2", "mixed", TraceRole.SIGNAL_EVALUATION)],
            "cfg",
            "main",
        ),
        "empty": SessionView("empty", [], "cfg", "main"),
    }
    client.read_session.side_effect = sessions.__getitem__
    since = datetime(2026, 7, 14, tzinfo=timezone.utc)

    found = ces_commands.sessions_to_judge(client, None, since=since, limit=3)

    assert [session.conversation_id for session in found] == ["agent"]
    client.find_sessions.assert_called_once_with(None, since=since, limit=3)
