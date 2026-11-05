"""`score`, `backfill`, `inspect`, `judge`, and the analysis commands over a fake CES client."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from weave_agent_signals import ces_commands, cli
from weave_agent_signals.ces_catalog import (
    DETAILS_SUFFIX,
    DETERMINISTIC,
    Catalog,
    ScorerRef,
    judge_spec,
)
from weave_agent_signals.ces_client import CesSession, session_view
from weave_agent_signals.models import Score, SessionView, TraceRole
from weave_agent_signals.patterns import ConfigStats, SessionSuccess, Stat
from weave_agent_signals.scorers import score_session, score_trajectory
from weave_agent_signals.scorers.session import scored_turns

CATALOG = Catalog(
    "ev-turns",
    "ev-sessions",
    {
        s.name: ScorerRef(s, f"sc-{s.name}", "v", f"sc-{s.name}{DETAILS_SUFFIX}", "v")
        for s in DETERMINISTIC
    },
)


def _case(case_id, kind, turn, step, day, **fields):
    output = fields.pop("output", {})
    return {
        "case_id": case_id,
        "input": {
            "kind": kind,
            "turn_id": turn,
            "turn_index": int(turn[-1]),
            "step_index": step,
            "started_at": f"2026-11-0{day}T10:00:0{step}.000Z",
            "trace_role": "agent_session",
            **fields,
        },
        "output": output,
        "child_trajectory_ids": [],
    }


SESSION = session_view(
    "s1",
    [
        _case("old", "user_message", "t1", 0, 1, text="before the window"),
        _case("c1", "user_message", "t2", 0, 3, text="run the tests"),
        _case("c2", "tool_call", "t2", 1, 3, tool_name="Bash", arguments={"command": "pytest -q"}),
    ],
    {},
)


class FakeCes:
    def __init__(self, turn_rows=None, session_rows=None):
        self.turn_rows, self.session_rows = turn_rows or {}, session_rows or {}
        self.root = {"cwd": "/Users/x/repos/core", "started_at": "2026-11-03T23:30:00Z"}
        self.writes: list[tuple] = []
        self.session_score_reads: list[list[str]] = []
        self.closed = False

    def find_sessions(self, catalog, *, since, until=None, limit=None):
        return ["s1"]

    def read_session(self, trajectory_id, *, subagents=True):
        return SESSION

    def root_case_input(self, trajectory_id):
        return self.root

    def turn_scores(self, catalog, trajectory_id):
        return self.turn_rows

    def session_scores(self, catalog, trajectory_ids, *, subagents=False):
        self.session_score_reads.append((list(trajectory_ids), subagents))
        # Every Trajectory in session_rows that isn't asked for is taken as a subagent.
        wanted = [*trajectory_ids, *(self.session_rows if subagents else ())]
        return {t: self.session_rows.get(t, []) for t in wanted}

    def write_scores(self, catalog, pass_id, kind, target, scores):
        self.writes.append((pass_id, kind, target, [s.scorer for s in scores]))
        return []

    def close(self):
        self.closed = True


@pytest.fixture
def fake(monkeypatch):
    client = FakeCes()
    monkeypatch.setattr(ces_commands, "connect", lambda args: CesSession(client, CATALOG))
    monkeypatch.setattr(
        ces_commands,
        "score_turn",
        lambda turn: [Score("outcome.test", 1.0, ["verified"], {}, "turn")],
    )
    monkeypatch.setattr(
        ces_commands,
        "score_session",
        lambda session: [Score("implicit.completion", True, [], {}, "session")],
    )
    monkeypatch.setattr(ces_commands, "score_trajectory", lambda trajectory, dimensions: [])
    return client


def _run(*argv):
    return cli.main(list(argv))


def test_score_writes_turn_scores_on_the_first_case_and_session_scores_on_the_trajectory(
    fake, capsys
):
    assert _run("score", "--since", "2026-11-02") == 0

    assert [(kind, target, names) for _, kind, target, names in fake.writes] == [
        ("case", "c1", ["outcome.test"]),
        ("trajectory", "s1", ["implicit.completion"]),
    ]
    assert len({pass_id for pass_id, *_ in fake.writes}) == 1
    assert "Scored 1 turns, 1 sessions, 0 subagents (2 scores). Wrote 2." in capsys.readouterr().out
    assert fake.closed


def test_an_existing_score_is_skipped_unless_forced(fake):
    fake.turn_rows = {"c1": [{"scorer_id": "sc-outcome.test", "value": 0.0}]}
    fake.session_rows = {"s1": [{"scorer_id": "sc-implicit.completion.details", "value": {}}]}

    _run("score", "--since", "2026-11-02")
    _run("score", "--since", "2026-11-02", "--force")

    first, second = fake.writes[:2], fake.writes[2:]
    assert [names for *_, names in first] == [[], ["implicit.completion"]]
    assert [names for *_, names in second] == [["outcome.test"], ["implicit.completion"]]
    assert first[0][0] != second[0][0]


def test_scoring_reads_existing_session_scores_once_for_all_sessions():
    client = FakeCes()

    ces_commands._score_sessions(
        client,
        Catalog("ev-t", "ev-s", {}),
        ["s1", "s2"],
        since=datetime(2000, 1, 1, tzinfo=timezone.utc),
        until=None,
        force=False,
    )

    assert client.session_score_reads == [(["s1", "s2"], True)]


SUBAGENT = session_view(
    "traj-a",
    [
        _case("a1", "user_message", "k1", 0, 3, text="test it", agent_id="worker"),
        _case(
            "a2",
            "tool_call",
            "k1",
            1,
            3,
            tool_name="Bash",
            arguments={"command": "pytest -q"},
            agent_id="worker",
            output={"status": "completed", "exit_code": 0},
        ),
    ],
    {},
)


def test_score_writes_whole_trajectory_scores_on_the_session_and_each_subagent(
    fake, monkeypatch, capsys
):
    monkeypatch.setattr(ces_commands, "score_trajectory", score_trajectory)
    monkeypatch.setattr(
        fake, "read_session", lambda sid, **_: replace(SESSION, subagent_trajectories=[SUBAGENT])
    )
    # The subagent already has its test outcome from its current turns, so only a forced pass
    # writes it again.
    fake.session_rows = {"traj-a": _recorded(SUBAGENT, "outcome.session_tests_passed")}

    _run("score", "--since", "2026-11-02")
    _run("score", "--since", "2026-11-02", "--force")

    sessions = [
        (target, [n for n in names if not n.startswith(("measure.", "dim."))])
        for _, kind, target, names in fake.writes
        if kind == "trajectory"
    ]
    assert sessions == [
        ("s1", ["implicit.completion", "outcome.session_tests_passed"]),
        ("traj-a", []),
        ("s1", ["implicit.completion", "outcome.session_tests_passed"]),
        ("traj-a", ["outcome.session_tests_passed"]),
    ]
    assert "1 sessions, 1 subagents" in capsys.readouterr().out


def test_a_subagent_takes_its_sessions_dimensions(fake, monkeypatch):
    written = []
    monkeypatch.setattr(ces_commands, "score_trajectory", score_trajectory)
    monkeypatch.setattr(
        fake, "read_session", lambda sid, **_: replace(SESSION, subagent_trajectories=[SUBAGENT])
    )
    monkeypatch.setattr(
        fake,
        "write_scores",
        lambda catalog, pass_id, kind, target, scores: written.append((target, scores)),
    )
    fake.root = {**fake.root, "repo": "wandb/core"}

    _run("score", "--since", "2026-11-02")

    dims = {
        target: {s.scorer: s.value for s in scores if s.scorer.startswith("dim.")}
        for target, scores in written
        if target in ("s1", "traj-a")
    }
    # The branch is derived: SESSION's turns name none, so there is no dim.git_branch.
    assert dims == {
        "s1": {"dim.repo": "wandb/core", "dim.day": "2026-11-03"},
        "traj-a": {"dim.repo": "wandb/core", "dim.day": "2026-11-03"},
    }


def _recorded(view: SessionView, *names: str, turns: dict | None = None) -> list[dict]:
    """Score rows on a Trajectory: each named Scorer's value and its details, which record the
    turns it was computed from, `view`'s current turns unless `turns` says otherwise."""
    details = {"details": {"scored_turns": turns or scored_turns(view)}}
    return [
        row
        for name in names
        for row in (
            {"scorer_id": f"sc-{name}", "value": True},
            # Read back from JSON, as CES returns it.
            {"scorer_id": f"sc-{name}{DETAILS_SUFFIX}", "value": json.loads(json.dumps(details))},
        )
    ]


WHOLE = ["outcome.session_tests_passed", "measure.turns", "dim.day"]


def _trajectory_writes(fake, *argv):
    _run("score", "--since", "2026-11-02", *argv)
    writes = [(target, names) for _, kind, target, names in fake.writes if kind == "trajectory"]
    fake.writes.clear()
    return writes


@pytest.fixture
def whole(fake, monkeypatch):
    """The fake with the real whole-Trajectory Scorers and SESSION with SUBAGENT."""
    monkeypatch.setattr(ces_commands, "score_trajectory", score_trajectory)
    monkeypatch.setattr(
        fake, "read_session", lambda sid, **_: replace(SESSION, subagent_trajectories=[SUBAGENT])
    )
    return fake


def test_a_session_that_gained_turns_since_its_scores_is_rescored(whole):
    # s1's Scores were computed when it had one turn; it has two now. traj-a hasn't changed.
    earlier = {"count": 1, "last_turn_id": "old", "last_turn_ended_at": None}
    whole.session_rows = {
        "s1": _recorded(SESSION, "implicit.completion", *WHOLE, turns=earlier),
        "traj-a": _recorded(SUBAGENT, *WHOLE),
    }

    writes = dict(_trajectory_writes(whole))

    # implicit.completion isn't whole-Trajectory, so its existing Score still stands.
    assert set(WHOLE) <= set(writes["s1"]) and "implicit.completion" not in writes["s1"]
    assert not set(WHOLE) & set(writes["traj-a"])


def test_an_unchanged_session_is_skipped_unless_forced(whole):
    whole.session_rows = {
        "s1": _recorded(SESSION, "implicit.completion", *WHOLE),
        "traj-a": _recorded(SUBAGENT, *WHOLE),
    }

    skipped = dict(_trajectory_writes(whole))
    forced = dict(_trajectory_writes(whole, "--force"))

    assert not set(WHOLE) & {name for names in skipped.values() for name in names}
    assert set(WHOLE) <= set(forced["s1"]) and set(WHOLE) <= set(forced["traj-a"])


def test_a_whole_trajectory_score_records_the_turns_it_came_from(whole, monkeypatch):
    written = {}
    monkeypatch.setattr(
        whole,
        "write_scores",
        lambda catalog, pass_id, kind, target, scores: written.update(
            {(target, s.scorer): s.metadata for s in scores}
        ),
    )

    _run("score", "--since", "2026-11-02")

    assert written[("s1", "measure.turns")]["scored_turns"] == {
        "count": 2,
        "last_turn_id": "c1",
        "last_turn_ended_at": "2026-11-03T10:00:01+00:00",
    }
    assert written[("traj-a", "measure.turns")]["scored_turns"] == scored_turns(SUBAGENT)
    assert "scored_turns" not in written[("s1", "implicit.completion")]


def test_a_session_whose_root_case_cant_be_read_writes_no_session_scores(fake, monkeypatch):
    def fail(trajectory_id):
        raise RuntimeError("CES is down")

    monkeypatch.setattr(fake, "root_case_input", fail)

    assert _run("score", "--since", "2026-11-02") == 1
    assert [kind for _, kind, *_ in fake.writes] == ["case"]


def test_backfill_reports_bash_coverage(fake, capsys):
    assert _run("backfill", "--start", "2026-11-02", "--end", "2026-11-04") == 0

    out = capsys.readouterr().out
    assert "Backfilled 2 scores across 1 turns, 1 sessions, 0 subagents." in out
    assert "Bash coverage: 1/1 classified (100%)" in out


def test_backfill_adds_only_the_session_scorers_a_historical_session_lacks(fake, monkeypatch):
    monkeypatch.setattr(ces_commands, "score_session", score_session)
    monkeypatch.setattr(ces_commands, "score_trajectory", score_trajectory)
    monkeypatch.setattr(
        fake, "read_session", lambda sid, **_: replace(SESSION, subagent_trajectories=[SUBAGENT])
    )
    # Scored before the whole-Trajectory Scorers existed: only the older session Scorers.
    older = ["implicit.completion", "implicit.correction_free_rate", "efficiency.session"]
    fake.session_rows = {"s1": [{"scorer_id": f"sc-{name}"} for name in older]}

    assert _run("backfill", "--start", "2026-11-01") == 0

    written = {target: names for _, kind, target, names in fake.writes if kind == "trajectory"}
    assert written == {
        "s1": [
            "outcome.session_tests_passed",
            "measure.turns",
            "measure.tokens",
            "measure.duration_s",
            "dim.repo",
            "dim.day",
        ],
        "traj-a": [
            "outcome.session_tests_passed",
            "measure.turns",
            "measure.tokens",
            "measure.duration_s",
            "dim.repo",
            "dim.day",
        ],
    }


def test_inspect_prints_a_session_with_its_scores(fake, capsys):
    fake.turn_rows = {
        "c1": [
            {"scorer_id": "sc-outcome.test", "value": 1.0},
            {
                "scorer_id": "sc-outcome.test.details",
                "value": {"tags": ["verified"], "reason": "passed"},
            },
        ]
    }

    assert _run("inspect", "--session", "s1", "--scores") == 0

    out = capsys.readouterr().out
    assert "Session s1" in out and "Turns: 2" in out
    assert "outcome.test                    rating=1.0  tags=['verified']  \"passed\"" in out


def test_connect_reads_the_adapter_state_when_flags_are_missing(tmp_path, monkeypatch):
    state = tmp_path / "evaluations.json"
    state.write_text('{"scope_ref": "scope-1", "dataset_id": "ds-1"}')
    monkeypatch.setenv("WEAVE_AGENT_ADAPTER_EVALUATIONS_STATE", str(state))
    made = {}

    class Client:
        def __init__(self, url, scope):
            made["scope"] = scope

        def ensure_catalog(self, dataset):
            made["dataset"] = dataset
            return CATALOG

    monkeypatch.setattr(ces_commands, "CesClient", Client)
    args = cli.build_parser().parse_args(["inspect", "--session", "s1"])

    ces_commands.connect(args)

    assert made == {"scope": "scope-1", "dataset": "ds-1"}


def test_connect_without_a_scope_or_dataset_explains_how_to_set_them(tmp_path, monkeypatch):
    monkeypatch.setenv("WEAVE_AGENT_ADAPTER_EVALUATIONS_STATE", str(tmp_path / "missing.json"))
    args = cli.build_parser().parse_args(["inspect", "--session", "s1"])

    with pytest.raises(SystemExit, match="--ces-scope-ref"):
        ces_commands.connect(args)


def _judge_score(rubric="judge.task_completion", status="complete", models=("claude:a", "codex:b")):
    metadata = {
        "review_status": status,
        "rubric_version": "3",
        "panel_contract_version": 2,
        "requested_judge_models": list(models),
    }
    return Score(rubric, 0.8, ["on_task"], metadata, "session", reason="did the task")


class FakeJudgeCes(FakeCes):
    def __init__(self, session_rows=None):
        super().__init__(session_rows=session_rows)
        self.added: list[str] = []

    def add_scorers(self, catalog, specs):
        self.added += [spec.name for spec in specs]
        refs = {
            s.name: ScorerRef(s, f"sc-{s.name}", "v", f"sc-{s.name}{DETAILS_SUFFIX}", "v")
            for s in specs
        }
        return Catalog(
            catalog.turns_evaluation_id, catalog.sessions_evaluation_id, {**catalog.scorers, **refs}
        )


def test_only_finished_reviews_write_judge_scores_under_their_panel_scorer():
    client = FakeJudgeCes()
    pending = [(_judge_score(), SESSION), (_judge_score("judge.focus", status="degraded"), SESSION)]

    written = ces_commands.write_judge_scores(client, CATALOG, pending, force=False)

    [(_, kind, target, names)] = client.writes
    assert (written, kind, target) == (1, "trajectory", "s1")
    assert names == client.added and names[0].startswith("judge.task_completion.panel-")


def test_a_different_panel_or_rubric_version_gets_a_different_scorer():
    base = judge_spec("judge.x", _judge_score().metadata)
    reordered = judge_spec("judge.x", _judge_score(models=("codex:b", "claude:a")).metadata)
    newer = judge_spec("judge.x", {**_judge_score().metadata, "rubric_version": "4"})

    assert base == judge_spec("judge.x", _judge_score().metadata)
    assert len({base.name, reordered.name, newer.name}) == 3
    assert (base.granularity, base.value_type, base.description) == (
        "session",
        "number",
        "judge.x v3 judged by claude:a, codex:b",
    )


def test_an_existing_panel_score_is_skipped_unless_forced():
    name = judge_spec("judge.task_completion", _judge_score().metadata).name
    client = FakeJudgeCes(session_rows={"s1": [{"scorer_id": f"sc-{name}", "value": 0.5}]})

    assert (
        ces_commands.write_judge_scores(client, CATALOG, [(_judge_score(), SESSION)], force=False)
        == 0
    )
    assert (
        ces_commands.write_judge_scores(client, CATALOG, [(_judge_score(), SESSION)], force=True)
        == 1
    )


def test_judge_judges_found_sessions_and_writes_finished_scores(monkeypatch, capsys):
    client = FakeJudgeCes()
    monkeypatch.setattr(ces_commands, "connect", lambda args: CesSession(client, CATALOG))
    seen = {}

    def judge_sessions(args, sessions, judges, rubrics):
        seen["sessions"] = [s.conversation_id for s in sessions]
        return "plan", [(_judge_score(), SESSION)], 0

    monkeypatch.setattr(cli, "_judge_sessions", judge_sessions)
    monkeypatch.setattr(
        cli, "judging_plan_totals", lambda plan: {"sessions_planned": 1, "windows_planned": 2}
    )

    assert cli.main(["judge", "--judge-model", "claude:claude-sonnet-5"]) == 0

    assert seen["sessions"] == ["s1"]
    assert (
        "Judged 1 sessions across 2 reviewer windows (1 rubric scores). Wrote 1."
        in capsys.readouterr().out
    )
    assert client.closed


class _ScoresOnly:
    def __init__(self, records):
        self.records, self.reads = records, []

    def read_all_scores(self, catalog, limit=None):
        self.reads.append(limit)
        return self.records

    def close(self):
        pass


@pytest.mark.parametrize(("feedback", "reads"), [(True, [7]), (False, [])])
def test_read_analysis_reads_the_newest_scores_only_for_feedback(monkeypatch, feedback, reads):
    records = [{"feedback_type": "weave_agent_signals.efficiency", "payload": {"rating": 1.0}}]
    client = _ScoresOnly(records)
    stats = ConfigStats({"v1": {"efficiency": Stat(1, 1.0, False)}}, {"v1": 1})
    monkeypatch.setattr(ces_commands, "connect", lambda args: CesSession(client, CATALOG))
    monkeypatch.setattr(ces_commands.ces_queries, "config_stats", lambda *_: stats)
    args = cli.build_parser().parse_args(["analyze", "--limit", "7"])

    got = ces_commands.read_analysis(args, feedback=feedback)

    assert got == (stats, records if feedback else [])
    assert client.reads == reads


def test_read_analysis_adds_session_success_on_the_same_connection(monkeypatch):
    client, connections = _ScoresOnly([]), []
    stats = ConfigStats({"v1": {"efficiency": Stat(1, 1.0, False)}}, {"v1": 1})
    success = {"v1": SessionSuccess(1, 1, 0, {})}

    def connect(args):
        connections.append(args)
        return CesSession(client, CATALOG)

    monkeypatch.setattr(ces_commands, "connect", connect)
    monkeypatch.setattr(ces_commands.ces_queries, "config_stats", lambda *_: stats)
    monkeypatch.setattr(ces_commands.ces_queries, "session_success_by_config", lambda *_: success)
    args = cli.build_parser().parse_args(["analyze"])

    got, _ = ces_commands.read_analysis(args, feedback=False, success=True)

    assert (got.by_config, got.success) == (stats.by_config, success)
    assert len(connections) == 1
    assert ces_commands.read_analysis(args, feedback=False)[0].success is None


@pytest.mark.parametrize(
    ("flags", "want"),
    [
        (["--subagents"], ["subagents"]),
        (["--judges", "--segments", "day"], ["scorers", "judges", "segments day"]),
    ],
)
def test_read_insights_runs_only_the_asked_queries_on_one_connection(monkeypatch, flags, want):
    client, calls, connections = _ScoresOnly([]), [], []
    queries = ces_commands.ces_queries

    def connect(args):
        connections.append(args)
        return CesSession(client, CATALOG)

    def session_scorers(_client, _catalog):
        calls.append("scorers")
        return ["s"]

    def judge_tables(_client, _catalog, _state, scorers):
        calls.append("judges")
        return [("j", "r", {})] if scorers == ["s"] else []

    def segment_stats(_client, _catalog, _state, scorers, field):
        calls.append(f"segments {field}")
        return {}

    monkeypatch.setattr(ces_commands, "connect", connect)
    monkeypatch.setattr(queries, "subagent_outcomes", lambda *_: calls.append("subagents") or {})
    monkeypatch.setattr(queries, "session_scorers", session_scorers)
    monkeypatch.setattr(queries, "judge_tables", judge_tables)
    monkeypatch.setattr(queries, "segment_stats", segment_stats)

    got = ces_commands.read_insights(cli.build_parser().parse_args(["analyze", *flags]))

    assert calls == want and len(connections) == 1
    assert (got.subagents is None) == ("subagents" not in want)
    assert got.judge_tables == ([("j", "r", {})] if "judges" in want else None)


def test_score_and_backfill_return_cleanly_when_no_sessions_match(fake, monkeypatch, capsys):
    monkeypatch.setattr(fake, "find_sessions", lambda *_args, **_kwargs: [])

    assert _run("score") == 0
    assert _run("backfill", "--start", "2026-11-01") == 0

    out = capsys.readouterr().out
    assert "No turns found." in out and "No turns found in date range." in out
    assert fake.writes == []


def test_a_session_with_non_agent_turns_is_not_scored(fake, monkeypatch):
    mixed = replace(SESSION.turns[0], trace_role=TraceRole.SIGNAL_EVALUATION)
    monkeypatch.setattr(
        fake,
        "read_session",
        lambda _sid: SessionView("s1", [mixed, *SESSION.turns[1:]], None, None),
    )

    assert _run("score", "--since", "2026-11-01") == 0

    assert fake.writes == []
