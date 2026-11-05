"""Whole-Trajectory Scorers: session outcomes from every turn's tool calls."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from weave_agent_signals.models import SessionView, SubagentSpan, ToolSpan, TurnSpan
from weave_agent_signals.scorers import score_trajectory
from weave_agent_signals.scorers.session import (
    MEASURE_DURATION,
    MEASURE_TOKENS,
    MEASURE_TURNS,
    SESSION_BUILDS,
    SESSION_TESTS,
    repo_identity,
    score_dimensions,
    score_session_measures,
    score_session_outcomes,
    session_dimensions,
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
    assert score_trajectory(_session(), {"repo": ("core", "derived")}) == []


def test_a_subagent_trajectory_is_scored_from_its_own_turns():
    subagent = _session(_turn(0, [_bash("pytest", 0, 1)]), trajectory_id="traj-a")

    [score] = [s for s in score_trajectory(subagent, {}) if s.scorer == SESSION_TESTS]
    assert (score.value, score.granularity) == (True, "session")


def test_measures_count_own_turns_all_tokens_below_and_turn_time_without_gaps():
    helper = SubagentSpan("traj-a", "worker", [], tokens=500)
    session = _session(
        _turn(0, subagents=[helper]),  # 4 minutes
        _turn(1),  # 4 minutes, after a 6-minute gap
        _turn(2, ended_at=None),  # no timestamps
    )

    measures = _by_name(score_session_measures(session))

    assert measures[MEASURE_TURNS].value == 3.0
    # Each turn has 100 input and 10 output tokens; cache reads aren't counted.
    assert measures[MEASURE_TOKENS].value == 3 * 110 + 500
    assert measures[MEASURE_TOKENS].metadata == {"own_tokens": 330, "subagent_tokens": 500}
    assert measures[MEASURE_DURATION].value == 480.0
    assert measures[MEASURE_DURATION].metadata == {"timed_turns": 2}


def test_no_timed_turn_means_no_duration():
    measures = _by_name(score_session_measures(_session(_turn(0, ended_at=None))))

    assert MEASURE_DURATION not in measures and measures[MEASURE_TURNS].value == 1.0


def test_the_adapters_root_case_fields_win_over_derived_dimensions():
    root = {
        "repo": "wandb/core",
        "git_branch": "release",
        "day": "2026-11-04",
        "cwd": "/Users/x/repos/core-worktrees/evals",
        "started_at": "2026-11-03T23:50:00Z",
    }

    assert session_dimensions(_session(_turn(0)), root) == {
        "repo": ("wandb/core", "root_case"),
        "git_branch": ("release", "root_case"),
        "day": ("2026-11-04", "root_case"),
    }


def test_an_older_session_derives_its_dimensions_from_its_cases():
    # 23:50 at UTC-5 is 04:50 the next day in UTC.
    root = {"cwd": "/Users/x/repos/core-worktrees/evals/", "started_at": "2026-11-03T23:50-05:00"}
    session = _session(_turn(0, git_branch=None), _turn(1, git_branch="feat/x"))

    assert session_dimensions(session, root) == {
        "repo": ("evals", "derived"),
        "git_branch": ("feat/x", "derived"),
        "day": ("2026-11-04", "derived"),
    }


def test_without_a_root_case_the_day_comes_from_the_first_timed_turn():
    session = _session(_turn(0, ended_at=None), _turn(1))

    # The untimed turn's start is the read time, so it's skipped; turn 1 starts 00:00 on the 4th.
    assert session_dimensions(session, {}) == {
        "git_branch": ("main", "derived"),
        "day": ("2026-11-04", "derived"),
    }


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """A function making a real git checkout under tmp_path, isolated from the user's git
    config, with `origin` set to the given URL when there is one."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    def make(name, origin=None):
        path = tmp_path / name
        path.mkdir()
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        if origin:
            subprocess.run(["git", "-C", str(path), "remote", "add", "origin", origin], check=True)
        return path

    return make


@pytest.mark.parametrize(
    "origin",
    [
        "git@github.com:Wandb/Core.git",
        "ssh://git@github.com:2222/wandb/core.git",
        "https://x-access-token:secret@github.com/wandb/core.git/",
    ],
)
def test_a_checkout_with_a_hosted_origin_is_named_owner_slash_name(git_repo, origin):
    checkout = git_repo("core-worktree", origin)
    (checkout / "src").mkdir()

    assert repo_identity(str(checkout / "src")) == "wandb/core"


def test_a_checkout_without_a_hosted_origin_is_named_by_its_top_level_folder(git_repo):
    bare = git_repo("Core")
    local = git_repo("weave", "/srv/git/weave.git")
    (bare / "src").mkdir()

    assert repo_identity(str(bare / "src")) == "core"
    assert repo_identity(str(local)) == "weave"


def test_a_missing_directory_is_named_by_its_own_folder(tmp_path):
    assert repo_identity(str(tmp_path / "gone" / "Core")) == "core"
    assert repo_identity("C:\\work\\core\\") == "core"
    assert repo_identity("  ") is None and repo_identity(None) is None


def test_an_older_session_in_a_checkout_takes_the_adapters_repo(git_repo):
    checkout = git_repo("core", "git@github.com:wandb/core.git")

    found = session_dimensions(_session(_turn(0)), {"cwd": str(checkout)})

    assert found["repo"] == ("wandb/core", "derived")


def test_dimension_scores_are_strings_named_by_field():
    [score] = score_dimensions({"repo": ("core", "derived")})

    assert (score.scorer, score.value, score.metadata) == (
        "dim.repo",
        "core",
        {"source": "derived"},
    )
