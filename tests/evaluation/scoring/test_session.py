"""Whole-Trajectory Scorers: session outcomes from every turn's tool calls."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from weave_agent_signals.models import SessionView, SubagentSpan, ToolSpan, TurnSpan
from weave_agent_signals.scorers import score_trajectory
from weave_agent_signals.scorers.session import (
    SESSION_BUILDS,
    SESSION_TESTS,
    score_session_outcomes,
)

START = datetime(2026, 11, 3, 23, 50, tzinfo=timezone.utc)


def _bash(command, exit_code, minute):
    return ToolSpan(
        span_id=f"sp-{minute}",
        tool_name="Bash",
        arguments=json.dumps({"command": command}),
        result="",
        status_code="OK",
        started_at=START + timedelta(minutes=minute),
        ended_at=START + timedelta(minutes=minute, seconds=30),
        exit_code=exit_code,
    )


def _turn(index, tool_calls=(), subagents=(), **fields):
    values = {
        "trace_id": f"t{index}",
        "conversation_id": "s1",
        "started_at": START + timedelta(minutes=10 * index),
        "ended_at": START + timedelta(minutes=10 * index + 4),
        "model": "gpt-6",
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_read_tokens": 50,
        "status_code": "OK",
        "config_version": "cfg1",
        "git_branch": "main",
        "effort_level": None,
        "session_id": "sess-1",
        "steering_count": 0,
        "denial_count": 0,
        "tool_error_count": 0,
        "events": [],
        "tool_calls": list(tool_calls),
        "chat_spans": [],
        "subagents": list(subagents),
    }
    return TurnSpan(**{**values, **fields})


def _session(*turns, trajectory_id="s1"):
    return SessionView(trajectory_id, list(turns), "cfg1", "main")


def _by_name(scores):
    return {s.scorer: s for s in scores}


def test_the_last_test_run_decides_tests_passed():
    fixed = _session(
        _turn(0, [_bash("pytest -q", 1, 1)]), _turn(1, [_bash("pytest tests/x.py", 0, 11)])
    )
    broke = _session(_turn(0, [_bash("pytest -q", 0, 1), _bash("pytest -q", 1, 2)]))

    fixed_score = _by_name(score_session_outcomes(fixed))[SESSION_TESTS]
    assert fixed_score.value is True
    assert fixed_score.metadata["runs"] == 2 and fixed_score.metadata["failed_runs"] == 1
    assert fixed_score.reason == "last of 2 test runs (pytest) passed"
    assert _by_name(score_session_outcomes(broke))[SESSION_TESTS].value is False


def test_a_subagents_test_run_counts_for_its_turn_in_start_time_order():
    # The turn's own failing run starts after the subagent's passing one, so it is the last.
    helper = SubagentSpan("traj-a", "worker", [_bash("go test ./...", 0, 2)])
    session = _session(_turn(0, [_bash("go test ./...", 1, 3)], subagents=[helper]))
    # And a subagent's passing run after the turn's own failing one is the last.
    later = SubagentSpan("traj-b", "worker", [_bash("go test ./...", 0, 4)])
    fixed = _session(_turn(0, [_bash("go test ./...", 1, 3)], subagents=[later]))

    assert _by_name(score_session_outcomes(session))[SESSION_TESTS].value is False
    assert _by_name(score_session_outcomes(fixed))[SESSION_TESTS].value is True


def test_any_failed_build_fails_no_failed_builds_even_when_a_later_one_passes():
    session = _session(_turn(0, [_bash("npm run build", 1, 1), _bash("npm run build", 0, 2)]))
    clean = _session(_turn(0, [_bash("cargo build", 0, 1)]))

    failed = _by_name(score_session_outcomes(session))[SESSION_BUILDS]
    assert failed.value is False and failed.reason == "1 of 2 builds failed"
    assert _by_name(score_session_outcomes(clean))[SESSION_BUILDS].value is True


def test_no_test_or_build_writes_no_outcome():
    session = _session(_turn(0, [_bash("ls", 0, 1)]))

    assert score_session_outcomes(session) == []


def test_an_empty_trajectory_gets_no_whole_trajectory_score():
    assert score_trajectory(_session()) == []


def test_a_subagent_trajectory_is_scored_from_its_own_turns():
    subagent = _session(_turn(0, [_bash("pytest", 0, 1)]), trajectory_id="traj-a")

    [score] = [s for s in score_trajectory(subagent) if s.scorer == SESSION_TESTS]
    assert (score.value, score.granularity) == (True, "session")
