"""`score`, `backfill`, `inspect`, `judge`, and the analysis commands over a fake CES client."""

from __future__ import annotations

from dataclasses import replace

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
        self.writes: list[tuple] = []
        self.closed = False

    def find_sessions(self, catalog, *, since, until=None, limit=None):
        return ["s1"]

    def read_session(self, trajectory_id):
        return SESSION

    def turn_scores(self, catalog, trajectory_id):
        return self.turn_rows

    def session_scores(self, catalog, trajectory_ids):
        return {t: self.session_rows.get(t, []) for t in trajectory_ids}

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
    assert "Scored 1 turns, 1 sessions (2 scores). Wrote 2." in capsys.readouterr().out
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


def test_backfill_reports_bash_coverage(fake, capsys):
    assert _run("backfill", "--start", "2026-11-02", "--end", "2026-11-04") == 0

    out = capsys.readouterr().out
    assert "Backfilled 2 scores across 1 turns, 1 sessions." in out
    assert "Bash coverage: 1/1 classified (100%)" in out


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

    assert _run("inspect", "--session", "s1", "--feedback") == 0

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


def test_analysis_commands_read_feedback_from_ces(monkeypatch):
    records = [{"feedback_type": "weave_agent_signals.efficiency", "payload": {"rating": 1.0}}]
    monkeypatch.setattr(ces_commands, "read_feedback", lambda args: records)
    args = cli.build_parser().parse_args(["analyze"])

    assert cli._project_feedback(args) is records


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
