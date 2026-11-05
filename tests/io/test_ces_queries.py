"""CES EvaluationQueries: create once by content, page results, retry busy, detect stddev."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from weave_agent_signals.ces_catalog import DETERMINISTIC, Catalog, ScorerRef
from weave_agent_signals.ces_client import CesClient
from weave_agent_signals.ces_queries import (
    QueryScorer,
    QueryState,
    config_stats,
    judge_tables,
    run_query,
    subagent_counts,
)
from weave_agent_signals.patterns import Stat

BASE = "http://ces.test/v1/namespaces/wandb/scopes/scope-1"
QUERIES = f"{BASE}/evaluations/ev-turns/queries"
SESSION_QUERIES = f"{BASE}/evaluations/ev-sessions/queries"
# What CES answers before wandb/core#61599 to a query on a Trajectory Evaluation.
CASE_EVALUATIONS_ONLY = httpx.Response(
    400,
    json={
        "error": {
            "code": "invalid_request",
            "message": "EvaluationQueries need an Evaluation with primary_target_kind case, and "
            "each query's target_kind picks Cases or Trajectories",
        }
    },
)
JUDGE = "judge.goal.panel-abc"
CATALOG = Catalog(
    "ev-turns",
    "ev-sessions",
    {
        spec.name: ScorerRef(
            spec, f"sc-{spec.name}", f"v-{spec.name}", f"sc-{spec.name}.d", f"v-{spec.name}.d"
        )
        for spec in DETERMINISTIC
    },
)
SESSION_SELECTION = [
    {"scorer_id": f"sc-{spec.name}{suffix}", "version_id": f"v-{spec.name}{suffix}"}
    for spec in DETERMINISTIC
    if spec.granularity == "session"
    for suffix in ("", ".d")
]
TURN_ROWS = [
    {
        "group": ["cfg1"],
        "targets": 30,
        "metrics": [
            {"name": "efficiency:mean", "value": 0.6},
            {"name": "efficiency:count", "value": 4},
            {"name": "efficiency:stddev", "value": 0.1},
            {"name": "outcome.test:mean", "value": 0.75},
            {"name": "outcome.test:count", "value": 4},
            {"name": "outcome.build:mean", "value": None},
            {"name": "outcome.build:count", "value": 0},
        ],
    },
    {
        "group": [None],
        "targets": 3,
        "metrics": [
            {"name": "outcome.test:mean", "value": 1.0},
            {"name": "outcome.test:count", "value": 1},
        ],
    },
]
SESSION_ROWS = [
    {
        "group": ["cfg1", 0],
        "targets": 2,
        "metrics": [
            {"name": "implicit.completion:mean", "value": 0.5},
            {"name": "implicit.completion:count", "value": 2},
            {"name": f"{JUDGE}:mean", "value": 0.7},
            {"name": f"{JUDGE}:count", "value": 2},
            {"name": f"{JUDGE}:stddev", "value": 0.1},
        ],
    },
    {
        "group": ["cfg1", 1],
        "targets": 5,
        "metrics": [
            {"name": f"{JUDGE}:mean", "value": 0.0},
            {"name": f"{JUDGE}:count", "value": 5},
        ],
    },
]


class Routes:
    """The CES routes one config_stats call uses, recording what it sends. `sessions_create`
    answers a create on the sessions Evaluation; by default CES accepts it."""

    def __init__(
        self, *, judges=((JUDGE, "sc-j", "v-j"),), turn_rows=TURN_ROWS, sessions_create=None
    ):
        self.creates: list[tuple[str, dict]] = []
        self.selected: list[dict] = []
        self.turn_selections = 0
        selection = list(SESSION_SELECTION)
        for name, scorer_id, version_id in judges:
            for suffix in ("", ".details"):
                selection.append({"scorer_id": scorer_id + suffix, "version_id": version_id})
                respx.get(f"{BASE}/scorers/{scorer_id}{suffix}/versions/{version_id}").mock(
                    return_value=httpx.Response(200, json={"name": name + suffix})
                )
        respx.patch(f"{BASE}/evaluations/ev-sessions/scorers").mock(
            return_value=httpx.Response(200, json={"scorers": selection})
        )
        respx.patch(f"{BASE}/evaluations/ev-turns/scorers").mock(side_effect=self._select)
        self.create = respx.post(QUERIES).mock(side_effect=self._create)
        self.sessions_create = respx.post(SESSION_QUERIES).mock(
            return_value=sessions_create, side_effect=None if sessions_create else self._create
        )
        respx.get(f"{QUERIES}/q-case/results").mock(
            return_value=httpx.Response(200, json={"rows": turn_rows})
        )
        for queries in (QUERIES, SESSION_QUERIES):
            respx.get(f"{queries}/q-trajectory/results").mock(
                return_value=httpx.Response(200, json={"rows": SESSION_ROWS})
            )

    def _select(self, request):
        self.turn_selections += 1
        self.selected = json.loads(request.content)["scorers"]
        return httpx.Response(200, json={"scorers": self.selected})

    def _create(self, request):
        body = json.loads(request.content)
        self.creates.append((request.headers["Idempotency-Key"], body))
        kind = body["definition"]["target_kind"]
        return httpx.Response(200, json={"evaluation_query_id": f"q-{kind}"})

    def keys(self) -> list[str]:
        return [key for key, _ in self.creates]


def _client(**kwargs) -> CesClient:
    return CesClient("http://ces.test", "scope-1", api_key="k", **kwargs)


def _no_feedback():
    pytest.fail("raw Scores are read only without a CES stddev")


@respx.mock
def test_config_stats_group_turns_by_config_and_keep_depth_0_sessions():
    Routes()
    state = QueryState()

    stats = config_stats(_client(), CATALOG, state, _no_feedback)

    assert stats.by_config == {
        "cfg1": {
            "efficiency": Stat(4, 0.6, False, 0.1),
            "outcome.test": Stat(4, 0.75, True),
            "implicit.completion": Stat(2, 0.5, True),
            JUDGE: Stat(2, 0.7, False, 0.1),
        },
        "unknown": {"outcome.test": Stat(1, 1.0, True)},
    }
    # The most-counted Scorer per group: 4 turns and 2 sessions.
    assert stats.targets == {"cfg1": 6, "unknown": 1}
    assert state.stddev is True


@respx.mock
def test_session_queries_run_on_the_sessions_evaluation_when_ces_accepts_them():
    routes = Routes()
    state = QueryState()

    config_stats(_client(), CATALOG, state, _no_feedback)

    assert state.trajectory_evaluations is True
    assert routes.turn_selections == 0
    (turns_key, turns), (sessions_key, sessions) = routes.creates
    assert turns_key.startswith("monopole-query-v1:ev-turns:")
    assert sessions_key.startswith("monopole-query-v1:ev-sessions:")
    assert sorted({m["scorer_version_id"] for m in sessions["definition"]["metrics"]}) == [
        "v-efficiency.session",
        "v-implicit.completion",
        "v-implicit.correction_free_rate",
        "v-j",
    ]
    assert turns["definition"]["target_kind"] == "case"
    assert turns["definition"]["group_by"] == [
        {"kind": "dataset_field", "source": "input", "name": "config_version"}
    ]
    assert sessions["definition"]["group_by"][1] == {"kind": "depth"}
    metrics = {m["name"]: m for m in turns["definition"]["metrics"]}
    assert metrics["efficiency:stddev"]["scorer_version_id"] == "v-efficiency"
    assert "outcome.test:stddev" not in metrics


@respx.mock
def test_a_ces_without_trajectory_evaluation_queries_runs_them_on_the_turns_evaluation():
    routes = Routes(sessions_create=CASE_EVALUATIONS_ONLY)
    state = QueryState()

    stats = config_stats(_client(), CATALOG, state, _no_feedback)

    assert state.trajectory_evaluations is False
    assert sorted(s["scorer_id"] for s in routes.selected) == [
        "sc-efficiency.session",
        "sc-implicit.completion",
        "sc-implicit.correction_free_rate",
        "sc-j",
    ]
    assert [key.split(":")[1] for key in routes.keys()] == ["ev-turns", "ev-turns"]
    assert routes.creates[1][1]["definition"]["target_kind"] == "trajectory"
    assert stats.by_config["cfg1"][JUDGE] == Stat(2, 0.7, False, 0.1)

    # The process remembers: later calls go straight to the turns Evaluation.
    config_stats(_client(), CATALOG, state, _no_feedback)
    assert routes.sessions_create.call_count == 1
    assert routes.turn_selections == 2


@respx.mock
def test_an_unrelated_rejection_on_the_sessions_evaluation_is_raised():
    routes = Routes(
        sessions_create=httpx.Response(
            400,
            json={
                "error": {
                    "code": "invalid_request",
                    "message": "column input.config_version is not catalogued on the "
                    "Evaluation's Dataset",
                }
            },
        )
    )
    state = QueryState()

    with pytest.raises(httpx.HTTPStatusError):
        config_stats(_client(), CATALOG, state, _no_feedback)

    assert state.trajectory_evaluations is None
    assert routes.turn_selections == 0


@respx.mock
def test_a_query_is_created_once_per_process_and_replayed_by_content():
    routes = Routes()
    state = QueryState()

    config_stats(_client(), CATALOG, state, _no_feedback)
    config_stats(_client(), CATALOG, state, _no_feedback)
    first = routes.keys()
    config_stats(_client(), CATALOG, QueryState(), _no_feedback)

    assert len(first) == 2
    # A new process replays the same keys, so CES returns the same query IDs.
    assert routes.keys()[2:] == first


@respx.mock
def test_the_same_query_on_another_evaluation_gets_its_own_create_key():
    # A query ID saved for one Evaluation must never be reused for another.
    keys = []

    def create(request):
        keys.append(request.headers["Idempotency-Key"])
        return httpx.Response(200, json={"evaluation_query_id": f"q-{len(keys)}"})

    for evaluation in ("ev-turns", "ev-sessions"):
        queries = f"{BASE}/evaluations/{evaluation}/queries"
        respx.post(queries).mock(side_effect=create)
        for query_id in ("q-1", "q-2"):
            respx.get(f"{queries}/{query_id}/results").mock(
                return_value=httpx.Response(200, json={"rows": []})
            )
    state = QueryState()
    scorers = [QueryScorer("implicit.completion", "sc-c", "v-c", True)]
    query = {"name": "q", "target_kind": "trajectory", "group_by": [], "scorers": scorers}

    for evaluation in ("ev-turns", "ev-sessions", "ev-turns"):
        run_query(_client(), state, evaluation, **query)

    turns, sessions = keys
    assert turns.startswith("monopole-query-v1:ev-turns:")
    assert sessions.startswith("monopole-query-v1:ev-sessions:")
    assert turns.split(":")[-1] == sessions.split(":")[-1]
    assert state.ids == {turns: "q-1", sessions: "q-2"}


@respx.mock
def test_a_new_judge_panel_makes_a_new_session_query():
    before = Routes()
    config_stats(_client(), CATALOG, QueryState(), _no_feedback)
    respx.reset()
    after = Routes(judges=((JUDGE, "sc-j", "v-j"), ("judge.goal.panel-new", "sc-n", "v-n")))
    config_stats(_client(), CATALOG, QueryState(), _no_feedback)

    assert after.keys()[0] == before.keys()[0]
    assert after.keys()[1] != before.keys()[1]


@respx.mock
def test_results_are_read_page_by_page():
    routes = Routes()
    pages = respx.get(f"{QUERIES}/q-case/results").mock(
        side_effect=[
            httpx.Response(200, json={"rows": TURN_ROWS[:1], "next_cursor": "p2"}),
            httpx.Response(200, json={"rows": TURN_ROWS[1:]}),
        ]
    )

    stats = config_stats(_client(), CATALOG, QueryState(), _no_feedback)

    assert routes.keys()
    assert [call.request.url.params.get("cursor") for call in pages.calls] == [None, "p2"]
    assert stats.by_config["unknown"] == {"outcome.test": Stat(1, 1.0, True)}


@respx.mock
def test_a_busy_ces_is_retried_with_growing_waits_then_raises():
    Routes()
    busy = httpx.Response(503, headers={"Retry-After": "1"})
    respx.get(f"{QUERIES}/q-case/results").mock(
        side_effect=[busy, busy, httpx.Response(200, json={"rows": []})]
    )
    slept: list[float] = []

    config_stats(_client(sleep=slept.append), CATALOG, QueryState(), _no_feedback)

    assert slept == [1.0, 2]

    respx.get(f"{QUERIES}/q-case/results").mock(return_value=busy)
    slept.clear()
    with pytest.raises(httpx.HTTPStatusError):
        config_stats(_client(sleep=slept.append), CATALOG, QueryState(), _no_feedback)
    assert slept == [1.0, 2, 4, 8, 16]


def _raw(scorer, rating, config):
    return {
        "id": f"s-{scorer}-{rating}",
        "feedback_type": f"weave_agent_signals.{scorer.split('.panel-')[0]}",
        "target_id": "t",
        "payload": {
            "rating": rating,
            "details": {"config_version": config, "ces_scorer": scorer},
        },
    }


@respx.mock
def test_without_ces_stddev_continuous_scorers_fall_back_to_raw_scores():
    routes = Routes()
    rejected = httpx.Response(
        422,
        json={
            "error": {
                "code": "unprocessable_entity",
                "details": [
                    {
                        "location": "body.definition.metrics[2].aggregation",
                        "message": "expected value to be one of mean, sum, min, max, count",
                    }
                ],
            }
        },
    )

    def create(request):
        reply = routes._create(request)
        return rejected if b'"stddev"' in request.content else reply

    routes.create.side_effect = create
    reads = []

    def feedback():
        reads.append(1)
        return [_raw("efficiency", 0.2, "cfg1"), _raw("efficiency", 0.4, "cfg1")]

    state = QueryState()
    stats = config_stats(_client(), CATALOG, state, feedback)

    assert state.stddev is False
    first, retried, sessions = [body for _, body in routes.creates]
    assert any(m["aggregation"] == "stddev" for m in first["definition"]["metrics"])
    for body in (retried, sessions):
        assert all(m["aggregation"] != "stddev" for m in body["definition"]["metrics"])
    assert len({key for key, _ in routes.creates}) == 3
    assert reads == [1]
    assert stats.by_config["cfg1"]["efficiency"] == Stat(
        2, pytest.approx(0.3), False, pytest.approx(0.1414, abs=1e-4)
    )
    # The judge has no raw Scores here, so it has no stats; binary Scorers still come from CES.
    assert JUDGE not in stats.by_config["cfg1"]
    assert stats.by_config["cfg1"]["outcome.test"] == Stat(4, 0.75, True)
    assert stats.targets["cfg1"] == 6


@respx.mock
def test_a_rejection_unrelated_to_stddev_is_raised():
    routes = Routes()
    routes.create.side_effect = [httpx.Response(400, json={"error": {"message": "no field"}})]

    with pytest.raises(httpx.HTTPStatusError):
        config_stats(_client(), CATALOG, QueryState(), _no_feedback)


@respx.mock
def test_a_dataset_without_subagents_gets_turn_stats():
    # CES doesn't catalogue a field no Case ever set, so on a Dataset that never ran a subagent
    # it rejects grouping by agent_id. Turn Scores only ever target main-agent Cases, so the
    # turn query needs no agent_id group.
    routes = Routes()
    uncatalogued = httpx.Response(
        400,
        json={"error": {"message": "column input.agent_id is not catalogued on the Dataset"}},
    )

    def create(request):
        reply = routes._create(request)
        return uncatalogued if b'"agent_id"' in request.content else reply

    routes.create.side_effect = create

    stats = config_stats(_client(), CATALOG, QueryState(), _no_feedback)

    assert stats.by_config["cfg1"]["efficiency"] == Stat(4, 0.6, False, 0.1)


CONFIG = {"kind": "dataset_field", "source": "input", "name": "config_version"}


class Insight:
    """One Trajectory query on the sessions Evaluation per create, each answered with the rows
    `results` gives for its definition."""

    def __init__(self, results, *, sessions_create=None):
        self.definitions: list[dict] = []
        self.results = results
        self.turn_selections: list[list[dict]] = []
        respx.patch(f"{BASE}/evaluations/ev-turns/scorers").mock(side_effect=self._select)
        for queries in (QUERIES, SESSION_QUERIES):
            respx.post(queries).mock(side_effect=self._create)
            respx.get(url__regex=rf"{queries}/q-\d+/results").mock(side_effect=self._results)
        if sessions_create is not None:
            respx.post(SESSION_QUERIES).mock(return_value=sessions_create)

    def _select(self, request):
        self.turn_selections.append(json.loads(request.content)["scorers"])
        return httpx.Response(200, json={"scorers": self.turn_selections[-1]})

    def _create(self, request):
        self.definitions.append(json.loads(request.content)["definition"])
        return httpx.Response(200, json={"evaluation_query_id": f"q-{len(self.definitions) - 1}"})

    def _results(self, request):
        index = int(request.url.path.split("/")[-2].removeprefix("q-"))
        return httpx.Response(200, json={"rows": self.results(self.definitions[index])})


def _row(group, targets, **metrics):
    return {
        "group": group,
        "targets": targets,
        "metrics": [{"name": name, "value": value} for name, value in metrics.items()],
    }


@respx.mock
def test_subagent_counts_count_every_trajectory_by_config_and_depth():
    rows = [
        _row(["cfg1", 0], 4, **{"implicit.completion:mean": 0.5, "implicit.completion:count": 3}),
        # Subagent Trajectories have no session Score, but each group still counts them.
        _row(["cfg1", 1], 6, **{"implicit.completion:mean": None, "implicit.completion:count": 0}),
        _row(["cfg1", 2], 1),
        _row([None, 0], 2),
    ]
    routes = Insight(lambda _definition: rows)

    counts = subagent_counts(_client(), CATALOG, QueryState())

    assert counts == {"cfg1": {0: 4, 1: 6, 2: 1}, "unknown": {0: 2}}
    (definition,) = routes.definitions
    assert definition["target_kind"] == "trajectory"
    assert definition["group_by"] == [CONFIG, {"kind": "depth"}]
    assert {m["scorer_version_id"] for m in definition["metrics"]} == {"v-implicit.completion"}


@respx.mock
def test_insight_queries_fall_back_to_the_turns_evaluation_like_session_queries():
    routes = Insight(
        lambda _definition: [_row(["cfg1", 1], 6)], sessions_create=CASE_EVALUATIONS_ONLY
    )
    state = QueryState()

    counts = subagent_counts(_client(), CATALOG, state)

    assert counts == {"cfg1": {1: 6}}
    assert state.trajectory_evaluations is False
    assert routes.turn_selections == [
        [{"scorer_id": "sc-implicit.completion", "version_id": "v-implicit.completion"}]
    ]


def test_an_unscored_session_group_adds_no_config_to_config_stats():
    with respx.mock:
        Routes()
        respx.get(f"{SESSION_QUERIES}/q-trajectory/results").mock(
            return_value=httpx.Response(200, json={"rows": SESSION_ROWS + [_row(["cfg2", 0], 3)]})
        )

        stats = config_stats(_client(), CATALOG, QueryState(), _no_feedback)

    assert stats.targets == {"cfg1": 6, "unknown": 1}
    assert "cfg2" not in stats.by_config


GOAL_NEW = "judge.goal.panel-new"
TOOL = "judge.tool.panel-x"
JUDGE_SCORERS = [
    QueryScorer("implicit.completion", "sc-c", "v-implicit.completion", True),
    QueryScorer(JUDGE, "sc-j", "v-j", False),
    QueryScorer(GOAL_NEW, "sc-n", "v-n", False),
    QueryScorer(TOOL, "sc-t", "v-t", False),
]


@respx.mock
def test_judge_tables_pair_each_panel_with_completion_and_other_panels_of_its_rubric():
    def results(definition):
        judge = definition["group_by"][1]["name"]
        return [
            _row([True, 0.75], 6, **{f"{judge}:mean": 0.75, f"{judge}:count": 6}),
            _row([False, 0.25], 3, **{f"{judge}:mean": 0.25, f"{judge}:count": 3}),
            # Sessions without the reference Score, or without the judge's, aren't paired.
            _row([None, 0.5], 4, **{f"{judge}:mean": 0.5, f"{judge}:count": 4}),
            _row([True, None], 2, **{f"{judge}:mean": None, f"{judge}:count": 0}),
        ]

    routes = Insight(results)

    tables = judge_tables(_client(), CATALOG, QueryState(), JUDGE_SCORERS)

    assert [(judge, reference) for judge, reference, _ in tables] == [
        (JUDGE, "implicit.completion"),
        (GOAL_NEW, "implicit.completion"),
        (TOOL, "implicit.completion"),
        (JUDGE, GOAL_NEW),
    ]
    assert tables[0][2] == {(True, 0.75): 6, (False, 0.25): 3}
    pair = routes.definitions[-1]
    assert pair["target_kind"] == "trajectory"
    assert pair["group_by"] == [
        {"kind": "score", "name": GOAL_NEW},
        {"kind": "score", "name": JUDGE},
    ]
    assert {m["scorer_version_id"] for m in pair["metrics"]} == {"v-j", "v-n"}


@respx.mock
def test_judge_tables_select_the_reference_too_when_falling_back():
    routes = Insight(lambda _definition: [], sessions_create=CASE_EVALUATIONS_ONLY)

    judge_tables(_client(), CATALOG, QueryState(), JUDGE_SCORERS[:2])

    assert [s["version_id"] for s in routes.turn_selections[0]] == ["v-j", "v-implicit.completion"]
