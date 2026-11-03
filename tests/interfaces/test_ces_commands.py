"""`score`, `backfill`, and `inspect` with `--backend ces`, over a fake CES client."""

from __future__ import annotations

import pytest

from weave_agent_signals import ces_commands, cli
from weave_agent_signals.ces_catalog import DETAILS_SUFFIX, DETERMINISTIC, Catalog, ScorerRef
from weave_agent_signals.ces_client import session_view
from weave_agent_signals.models import Score

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
    monkeypatch.setattr(ces_commands, "connect", lambda args: (client, CATALOG))
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
    return cli.main([argv[0], "--backend", "ces", *argv[1:]])


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
    args = cli.build_parser().parse_args(["inspect", "--backend", "ces", "--session", "s1"])

    ces_commands.connect(args)

    assert made == {"scope": "scope-1", "dataset": "ds-1"}


def test_connect_without_a_scope_or_dataset_explains_how_to_set_them(tmp_path, monkeypatch):
    monkeypatch.setenv("WEAVE_AGENT_ADAPTER_EVALUATIONS_STATE", str(tmp_path / "missing.json"))
    args = cli.build_parser().parse_args(["inspect", "--backend", "ces", "--session", "s1"])

    with pytest.raises(SystemExit, match="--ces-scope-ref"):
        ces_commands.connect(args)


def test_the_weave_backend_stays_the_default():
    args = cli.build_parser().parse_args(["score"])

    assert args.backend == "weave"
