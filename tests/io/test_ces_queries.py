"""CES EvaluationQueries: create once by content, page results, retry busy, detect stddev."""

from __future__ import annotations

import json
from datetime import date

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
    segment_stats,
    session_day_stats,
    session_success_by_config,
    subagent_outcomes,
)
from weave_agent_signals.patterns import SessionSuccess, Stat

BASE = "http://ces.test/v1/namespaces/wandb/scopes/scope-1"
QUERIES = f"{BASE}/evaluations/ev-turns/queries"
SESSION_QUERIES = f"{BASE}/evaluations/ev-sessions/queries"
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
    """The CES routes one config_stats call uses, recording what it sends. The turn query is
    created on the turns Evaluation and the session query on the sessions Evaluation; any
    other request, such as selecting Scorers on the turns Evaluation, isn't mocked and fails."""

    def __init__(self, *, judges=((JUDGE, "sc-j", "v-j"),), turn_rows=TURN_ROWS):
        self.creates: list[tuple[str, dict]] = []
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
        self.create = respx.post(QUERIES).mock(side_effect=self._create)
        respx.post(SESSION_QUERIES).mock(side_effect=self._create)
        respx.get(f"{QUERIES}/q-case/results").mock(
            return_value=httpx.Response(200, json={"rows": turn_rows})
        )
        respx.get(f"{SESSION_QUERIES}/q-trajectory/results").mock(
            return_value=httpx.Response(200, json={"rows": SESSION_ROWS})
        )

    def _create(self, request):
        body = json.loads(request.content)
        self.creates.append((request.headers["Idempotency-Key"], body))
        kind = body["definition"]["target_kind"]
        return httpx.Response(200, json={"evaluation_query_id": f"q-{kind}"})

    def keys(self) -> list[str]:
        return [key for key, _ in self.creates]


def _client(**kwargs) -> CesClient:
    return CesClient("http://ces.test", "scope-1", api_key="k", **kwargs)


@respx.mock
def test_config_stats_group_turns_by_config_and_keep_depth_0_sessions():
    Routes()
    state = QueryState()

    stats = config_stats(_client(), CATALOG, state)

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
def test_session_queries_run_on_the_sessions_evaluation():
    routes = Routes()

    config_stats(_client(), CATALOG, QueryState())

    (turns_key, turns), (sessions_key, sessions) = routes.creates
    assert turns_key.startswith("monopole-query-v1:ev-turns:")
    assert sessions_key.startswith("monopole-query-v1:ev-sessions:")
    assert sorted({m["scorer_version_id"] for m in sessions["definition"]["metrics"]}) == [
        "v-efficiency.session",
        "v-implicit.completion",
        "v-implicit.correction_free_rate",
        "v-j",
        "v-outcome.session_no_failed_builds",
        "v-outcome.session_tests_passed",
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
def test_a_query_is_created_once_per_process_and_replayed_by_content():
    routes = Routes()
    state = QueryState()

    config_stats(_client(), CATALOG, state)
    config_stats(_client(), CATALOG, state)
    first = routes.keys()
    config_stats(_client(), CATALOG, QueryState())

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
    config_stats(_client(), CATALOG, QueryState())
    respx.reset()
    after = Routes(judges=((JUDGE, "sc-j", "v-j"), ("judge.goal.panel-new", "sc-n", "v-n")))
    config_stats(_client(), CATALOG, QueryState())

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

    stats = config_stats(_client(), CATALOG, QueryState())

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

    config_stats(_client(sleep=slept.append), CATALOG, QueryState())

    assert slept == [1.0, 2]

    respx.get(f"{QUERIES}/q-case/results").mock(return_value=busy)
    slept.clear()
    with pytest.raises(httpx.HTTPStatusError):
        config_stats(_client(sleep=slept.append), CATALOG, QueryState())
    assert slept == [1.0, 2, 4, 8, 16]


STDDEV_REJECTED = httpx.Response(
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


def _histograms(routes: Routes, results: dict) -> list[dict]:
    """Answer a create naming stddev with CES's rejection, and a value-histogram query, one
    grouped by a Score, with `results[scorer name]`: rows, or a Response. Returns the histogram
    definitions in create order."""
    definitions = []

    def create(request):
        body = json.loads(request.content)
        if b'"stddev"' in request.content:
            routes._create(request)
            return STDDEV_REJECTED
        column = body["definition"]["group_by"][-1]
        if column.get("kind") != "score":
            return routes._create(request)
        definitions.append(body["definition"])
        evaluation = request.url.path.split("/")[-2]
        query = f"q-hist-{column['name']}"
        reply = results[column["name"]]
        respx.get(f"{BASE}/evaluations/{evaluation}/queries/{query}/results").mock(
            return_value=reply
            if isinstance(reply, httpx.Response)
            else httpx.Response(200, json={"rows": reply})
        )
        return httpx.Response(200, json={"evaluation_query_id": query})

    routes.create.side_effect = create
    respx.post(SESSION_QUERIES).mock(side_effect=create)
    return definitions


def _counts(scorer, group, value, count):
    return _row([*group, value], count, **{f"{scorer}:mean": value, f"{scorer}:count": count})


@respx.mock
def test_without_ces_stddev_a_continuous_scorer_gets_its_exact_stddev_from_a_value_histogram():
    routes = Routes()
    definitions = _histograms(
        routes,
        {
            # Efficiency ratings 0.5, 0.5, 1.0 in cfg1; the unscored group has no count.
            "efficiency": [
                _counts("efficiency", ["cfg1"], 0.5, 2),
                _counts("efficiency", ["cfg1"], 1.0, 1),
                _row(["cfg1", None], 9),
            ],
            # Sessions rated 0.6 and 0.8; a subagent's 0.0 is left out.
            JUDGE: [
                _counts(JUDGE, ["cfg1", 0], 0.6, 1),
                _counts(JUDGE, ["cfg1", 0], 0.8, 1),
                _counts(JUDGE, ["cfg1", 1], 0.0, 5),
            ],
        },
    )
    state = QueryState()

    stats = config_stats(_client(), CATALOG, state)

    assert state.stddev is False
    (first, _), (retried, _), _sessions = routes.creates
    assert first != retried  # the retry without stddev is new content under a new key
    assert stats.by_config["cfg1"]["efficiency"] == Stat(
        3, pytest.approx(2 / 3), False, pytest.approx((1 / 12) ** 0.5)
    )
    assert stats.by_config["cfg1"][JUDGE] == Stat(
        2, pytest.approx(0.7), False, pytest.approx(0.02**0.5)
    )
    # Binary Scorers and target counts still come from the config queries.
    assert stats.by_config["cfg1"]["outcome.test"] == Stat(4, 0.75, True)
    assert stats.targets["cfg1"] == 6
    # One histogram per continuous Scorer that has Scores: none for efficiency.session.
    assert [d["group_by"] for d in definitions] == [
        [CONFIG, {"kind": "score", "name": "efficiency"}],
        [CONFIG, {"kind": "depth"}, {"kind": "score", "name": JUDGE}],
    ]
    assert [d["target_kind"] for d in definitions] == ["case", "trajectory"]
    for definition in definitions:
        assert all(m["aggregation"] != "stddev" for m in definition["metrics"])


@respx.mock
def test_a_histogram_too_large_falls_back_to_a_conservative_stddev(monkeypatch):
    monkeypatch.setattr("weave_agent_signals.ces_queries.HISTOGRAM_MAX_ROWS", 2)
    routes = Routes()
    too_large = httpx.Response(
        400,
        json={
            "error": {
                "message": "query exceeds the configured limit of 10000 result rows per "
                "request, request a smaller page"
            }
        },
    )
    _histograms(
        routes,
        {
            # Three distinct values pass the limit of 2 rows.
            "efficiency": [_counts("efficiency", ["cfg1"], v, 1) for v in (0.2, 0.6, 1.0)],
            JUDGE: too_large,
        },
    )

    stats = config_stats(_client(), CATALOG, QueryState())

    # The config query's 4 efficiency ratings with mean 0.6: sqrt(4/3 * 0.6 * 0.4).
    assert stats.by_config["cfg1"]["efficiency"] == Stat(
        4, 0.6, False, pytest.approx((4 / 3 * 0.24) ** 0.5), conservative=True
    )
    assert stats.by_config["cfg1"][JUDGE] == Stat(
        2, 0.7, False, pytest.approx((2 * 0.21) ** 0.5), conservative=True
    )


@respx.mock
def test_a_histogram_error_other_than_too_large_is_raised():
    routes = Routes()
    _histograms(
        routes,
        {
            "efficiency": httpx.Response(400, json={"error": {"message": "no column"}}),
            JUDGE: [],
        },
    )

    with pytest.raises(httpx.HTTPStatusError):
        config_stats(_client(), CATALOG, QueryState())


@respx.mock
def test_a_rejection_unrelated_to_stddev_is_raised():
    routes = Routes()
    routes.create.side_effect = [httpx.Response(400, json={"error": {"message": "no field"}})]

    with pytest.raises(httpx.HTTPStatusError):
        config_stats(_client(), CATALOG, QueryState())


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

    stats = config_stats(_client(), CATALOG, QueryState())

    assert stats.by_config["cfg1"]["efficiency"] == Stat(4, 0.6, False, 0.1)


CONFIG = {"kind": "dataset_field", "source": "input", "name": "config_version"}


class Insight:
    """One Trajectory query on the sessions Evaluation per create, each answered with the rows
    `results` gives for its definition. Nothing on the turns Evaluation is mocked."""

    def __init__(self, results):
        self.definitions: list[dict] = []
        self.results = results
        respx.post(SESSION_QUERIES).mock(side_effect=self._create)
        respx.get(url__regex=rf"{SESSION_QUERIES}/q-\d+/results").mock(side_effect=self._results)

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


TESTS_PASSED = "outcome.session_tests_passed"
NO_FAILED_BUILDS = "outcome.session_no_failed_builds"


@respx.mock
def test_subagent_outcomes_count_every_trajectory_and_rate_outcomes_by_depth():
    rows = [
        _row(["cfg1", 0], 4, **{f"{TESTS_PASSED}:mean": 0.5, f"{TESTS_PASSED}:count": 2}),
        _row(["cfg1", 1], 6, **{f"{TESTS_PASSED}:mean": 1.0, f"{TESTS_PASSED}:count": 1}),
        # A Trajectory with no outcome Score still counts.
        _row(["cfg1", 2], 1),
        _row([None, 0], 2),
    ]
    routes = Insight(lambda _definition: rows)

    found = subagent_outcomes(_client(), CATALOG, QueryState())

    assert {c: {d: g.targets for d, g in depths.items()} for c, depths in found.items()} == {
        "cfg1": {0: 4, 1: 6, 2: 1},
        "unknown": {0: 2},
    }
    assert found["cfg1"][1].stats == {TESTS_PASSED: Stat(1, 1.0, True)}
    (definition,) = routes.definitions
    assert definition["target_kind"] == "trajectory"
    assert definition["group_by"] == [CONFIG, {"kind": "depth"}]
    assert {m["scorer_version_id"] for m in definition["metrics"]} == {
        f"v-{TESTS_PASSED}",
        f"v-{NO_FAILED_BUILDS}",
    }


@respx.mock
def test_session_success_splits_each_configs_sessions_by_tests_passed_and_sums_spend():
    def spend(tokens, turns, count):
        return {
            "measure.tokens:mean": tokens,
            "measure.tokens:count": count,
            "measure.turns:mean": turns,
            "measure.turns:count": count,
        }

    rows = [
        _row(["cfg1", 0, True], 3, **spend(1000.0, 4.0, 3)),
        _row(["cfg1", 0, False], 2, **spend(4000.0, 10.0, 2)),
        _row(["cfg1", 0, None], 5, **spend(50.0, 1.0, 5)),
        # A subagent's spend is already in its session's tokens.
        _row(["cfg1", 1, True], 7, **spend(9.0, 9.0, 7)),
    ]
    routes = Insight(lambda _definition: rows)

    found = session_success_by_config(_client(), CATALOG, QueryState())

    assert found == {
        "cfg1": SessionSuccess(3, 2, 5, {"measure.tokens": 11000.0, "measure.turns": 32.0})
    }
    (definition,) = routes.definitions
    assert definition["group_by"] == [
        CONFIG,
        {"kind": "depth"},
        {"kind": "score", "name": TESTS_PASSED},
    ]
    assert {m["scorer_version_id"] for m in definition["metrics"]} == {
        "v-measure.turns",
        "v-measure.tokens",
        "v-measure.duration_s",
    }
    assert {m["aggregation"] for m in definition["metrics"]} >= {"mean", "count"}


def test_an_unscored_session_group_adds_no_config_to_config_stats():
    with respx.mock:
        Routes()
        respx.get(f"{SESSION_QUERIES}/q-trajectory/results").mock(
            return_value=httpx.Response(200, json={"rows": SESSION_ROWS + [_row(["cfg2", 0], 3)]})
        )

        stats = config_stats(_client(), CATALOG, QueryState())

    assert stats.targets == {"cfg1": 6, "unknown": 1}
    assert "cfg2" not in stats.by_config


GOAL_NEW = "judge.goal.panel-new"
TOOL = "judge.tool.panel-x"
JUDGE_SCORERS = [
    QueryScorer("implicit.completion", "sc-c", "v-implicit.completion", True),
    QueryScorer(TESTS_PASSED, "sc-p", f"v-{TESTS_PASSED}", True),
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
        (JUDGE, TESTS_PASSED),
        (JUDGE, "implicit.completion"),
        (GOAL_NEW, TESTS_PASSED),
        (GOAL_NEW, "implicit.completion"),
        (TOOL, TESTS_PASSED),
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
    # Only the judge's count is read; the reference is a Score column.
    assert {m["scorer_version_id"] for m in pair["metrics"]} == {"v-j"}


@respx.mock
def test_segment_stats_group_scored_sessions_by_config_and_dimension_score():
    rows = [
        _row(
            ["cfg1", 0, "core"],
            5,
            **{"implicit.completion:mean": 0.8, "implicit.completion:count": 5},
        ),
        _row(
            [None, 0, None], 2, **{"implicit.completion:mean": 0.5, "implicit.completion:count": 2}
        ),
        # Subagents carry their session's dimension, but only sessions are segments.
        _row(
            ["cfg1", 1, "core"],
            9,
            **{"implicit.completion:mean": 0.1, "implicit.completion:count": 9},
        ),
        _row(["cfg1", 0, "weave"], 3),
    ]
    routes = Insight(lambda _definition: rows)

    segments = segment_stats(_client(), CATALOG, QueryState(), JUDGE_SCORERS[:1], "repo")

    assert segments == {
        ("cfg1", "core"): {"implicit.completion": Stat(5, 0.8, True)},
        ("unknown", None): {"implicit.completion": Stat(2, 0.5, True)},
    }
    (definition,) = routes.definitions
    assert definition["group_by"] == [
        CONFIG,
        {"kind": "depth"},
        {"kind": "score", "name": "dim.repo"},
    ]


COMPLETION = "implicit.completion"
DAY = {"kind": "score", "name": "dim.day"}


def _day_routes(results, *, stddev: bool) -> Insight:
    """The sessions Evaluation's selection, judge name, and queries for session_day_stats.
    Without `stddev`, a create naming it gets CES's rejection."""
    selection = SESSION_SELECTION + [
        {"scorer_id": f"sc-j{suffix}", "version_id": "v-j"} for suffix in ("", ".details")
    ]
    respx.patch(f"{BASE}/evaluations/ev-sessions/scorers").mock(
        return_value=httpx.Response(200, json={"scorers": selection})
    )
    for suffix in ("", ".details"):
        respx.get(f"{BASE}/scorers/sc-j{suffix}/versions/v-j").mock(
            return_value=httpx.Response(200, json={"name": JUDGE + suffix})
        )
    insight = Insight(results)
    if not stddev:
        create = insight._create
        respx.post(SESSION_QUERIES).mock(
            side_effect=lambda r: STDDEV_REJECTED if b'"stddev"' in r.content else create(r)
        )
    return insight


def _completion(count, mean):
    return {f"{COMPLETION}:mean": mean, f"{COMPLETION}:count": count}


@respx.mock
def test_session_day_stats_pool_each_days_sessions_across_configs():
    rows = [
        _row(["cfg1", 0, "2026-07-01"], 2, **_completion(2, 1.0)),
        _row(["cfg2", 0, "2026-07-01"], 2, **_completion(2, 0.0)),
        _row([None, 0, "2026-07-09"], 3, **_completion(3, 0.0)),
        # A subagent, a session without dim.day, and a day that isn't one are left out.
        _row(["cfg3", 1, "2026-07-09"], 5, **_completion(5, 1.0)),
        _row(["cfg1", 0, None], 4, **_completion(4, 1.0)),
        _row(["cfg1", 0, "soon"], 4, **_completion(4, 1.0)),
        _row(
            ["cfg1", 0, "2026-07-09"],
            2,
            **{f"{JUDGE}:mean": 0.5, f"{JUDGE}:count": 2, f"{JUDGE}:stddev": 0.1},
        ),
    ]
    routes = _day_routes(lambda _definition: rows, stddev=True)

    days = session_day_stats(_client(), CATALOG, QueryState())

    assert days == {
        COMPLETION: {date(2026, 7, 1): Stat(4, 0.5, True), date(2026, 7, 9): Stat(3, 0.0, True)},
        JUDGE: {date(2026, 7, 9): Stat(2, 0.5, False, 0.1)},
    }
    (definition,) = routes.definitions
    assert definition["target_kind"] == "trajectory"
    assert definition["group_by"] == [CONFIG, {"kind": "depth"}, DAY]
    # Rating Scorers only: no measurement or dimension is a metric.
    named = {m["scorer_version_id"] for m in definition["metrics"]}
    assert f"v-{COMPLETION}" in named and "v-j" in named
    assert not any(v.startswith(("v-measure.", "v-dim.")) for v in named)


@respx.mock
def test_without_ces_stddev_session_day_stats_take_each_days_spread_from_a_histogram():
    def results(definition):
        if definition["group_by"][-1] == {"kind": "score", "name": JUDGE}:
            # Jul 9: cfg1 rated 0.25 and 0.75, cfg2 0.75; a subagent's 0.0 is left out.
            return [
                _counts(JUDGE, ["cfg1", 0, "2026-07-09"], 0.25, 1),
                _counts(JUDGE, ["cfg1", 0, "2026-07-09"], 0.75, 1),
                _counts(JUDGE, ["cfg2", 0, "2026-07-09"], 0.75, 1),
                _counts(JUDGE, ["cfg1", 1, "2026-07-09"], 0.0, 4),
            ]
        judge = {f"{JUDGE}:mean": 0.5, f"{JUDGE}:count": 2}
        return [
            _row(["cfg1", 0, "2026-07-09"], 2, **judge),
            _row(["cfg2", 0, "2026-07-09"], 1, **{f"{JUDGE}:mean": 0.75, f"{JUDGE}:count": 1}),
        ]

    routes = _day_routes(results, stddev=False)
    state = QueryState()

    days = session_day_stats(_client(), CATALOG, state)

    # Pooled ratings 0.25, 0.75, 0.75: mean 7/12, sample stddev sqrt(1/12).
    assert state.stddev is False
    assert days[JUDGE][date(2026, 7, 9)] == Stat(
        3, pytest.approx(7 / 12), False, pytest.approx((1 / 12) ** 0.5)
    )
    assert routes.definitions[-1]["group_by"] == [
        CONFIG,
        {"kind": "depth"},
        DAY,
        {"kind": "score", "name": JUDGE},
    ]
