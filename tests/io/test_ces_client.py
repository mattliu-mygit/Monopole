"""CES reads: session reconstruction from the adapter's Cases, paging, and retries."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from weave_agent_signals.ces_catalog import VERDICT, Catalog, ScorerSpec
from weave_agent_signals.ces_client import (
    MAX_RETRY_AFTER_S,
    PAGE_LIMIT,
    QUERY_SLOTS,
    ROW_SEARCH_PAGE_LIMIT,
    CesClient,
    _score_record,
    _tool_span,
    session_view,
    turns_from_rows,
)
from weave_agent_signals.ces_score_index import ScoreIndex
from weave_agent_signals.models import Score
from weave_agent_signals.scorers.outcome import _extract_exit_code

BASE = "http://ces.test/v1/namespaces/wandb/scopes/scope-1"


def _case(case_id, kind, turn_id, step, children=(), output=None, **fields):
    return {
        "case_id": case_id,
        "input": {
            "kind": kind,
            "turn_id": turn_id,
            "turn_index": int(turn_id[-1]),
            "step_index": step,
            "started_at": f"2026-11-02T10:0{int(turn_id[-1])}:0{step}.000Z",
            "model": "gpt-6",
            "effort": "high",
            "git_branch": "main",
            "config_version": "cfg1",
            "session_id": "sess-1",
            "trace_role": "agent_session",
            **fields,
        },
        "output": output or {},
        "child_trajectory_ids": list(children),
    }


SESSION = [
    {
        "case_id": "c-session",
        "input": {"kind": "session", "session_id": "sess-1"},
        "output": {},
        "child_trajectory_ids": [],
    },
    _case(
        "c1",
        "user_message",
        "t1",
        0,
        text="count files",
        output={
            "turn_status": "completed",
            "turn_input_tokens": 300,
            "turn_output_tokens": 30,
            "turn_cache_read_tokens": 50,
            "turn_steering_count": 1,
            "turn_denial_count": 0,
            "turn_tool_error_count": 1,
        },
    ),
    _case(
        "c2",
        "assistant_message",
        "t1",
        1,
        output={
            "text": "Asking a helper.",
            "finish_reason": "tool_calls",
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_read_tokens": 50,
            "cache_write_tokens": 5,
        },
    ),
    _case(
        "c3",
        "tool_call",
        "t1",
        2,
        children=["traj-a"],
        tool_name="Task",
        arguments={"prompt": "count"},
        output={"status": "completed", "output": "3 files", "error": None},
    ),
    _case(
        "c4",
        "assistant_message",
        "t1",
        3,
        output={"text": "There are 3 files.", "finish_reason": "stop"},
    ),
    _case("c5", "user_message", "t2", 0, text="list them", output={"turn_status": "aborted"}),
    _case("c6", "compaction", "t2", 1, trigger="auto"),
]
AGENT_A = [
    _case("a1", "user_message", "k1", 0, text="count"),
    _case(
        "a2",
        "tool_call",
        "k1",
        1,
        children=["traj-b"],
        tool_name="Task",
        arguments={"prompt": "really"},
        output={"status": "completed", "output": "3"},
    ),
]
AGENT_B = [
    _case(
        "b1",
        "tool_call",
        "n1",
        0,
        tool_name="Bash",
        arguments={"command": "ls"},
        output={"status": "failed", "output": "denied", "error": "denied"},
    )
]


def test_a_session_becomes_turns_with_tools_chats_events_and_nested_subagents():
    view = session_view("traj-s", SESSION, {"traj-a": AGENT_A, "traj-b": AGENT_B})

    first, second = view.turns
    assert (view.conversation_id, view.config_version, view.git_branch) == (
        "traj-s",
        "cfg1",
        "main",
    )
    assert (
        first.trace_id,
        first.conversation_id,
        first.status_code,
        first.model,
        first.effort_level,
    ) == ("c1", "traj-s", "OK", "gpt-6", "high")
    assert (first.input_tokens, first.output_tokens, first.cache_read_tokens) == (300, 30, 50)
    assert (first.steering_count, first.tool_error_count, first.session_id) == (1, 1, "sess-1")
    assert (first.user_input, first.assistant_output) == ("count files", "There are 3 files.")
    assert [
        (t.span_id, t.tool_name, t.arguments, t.result, t.status_code) for t in first.tool_calls
    ] == [("c3", "Task", '{"prompt": "count"}', "3 files", "OK")]
    assert [(c.span_id, c.finish_reason, c.cache_creation_tokens) for c in first.chat_spans] == [
        ("c2", "tool_calls", 5),
        ("c4", "stop", 0),
    ]
    assert [(s.span_id, [t.status_code for t in s.tool_calls]) for s in first.subagents] == [
        ("traj-a", ["OK"]),
        ("traj-b", ["ERROR"]),
    ]
    assert (second.trace_id, second.status_code, [e.name for e in second.events]) == (
        "c5",
        "ERROR",
        ["compaction"],
    )
    assert second.subagents == []
    assert (first.case_ids, second.case_ids) == (["c1", "c2", "c3", "c4"], ["c5", "c6"])


@respx.mock
def test_read_session_pages_and_follows_subagents_with_retries():
    respx.get(f"{BASE}/trajectories/traj-s/cases").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, json={"cases": SESSION[:3], "next_cursor": "p2"}),
            httpx.Response(200, json={"cases": SESSION[3:], "next_cursor": None}),
        ]
    )
    respx.get(f"{BASE}/trajectories/traj-a/cases").mock(
        return_value=httpx.Response(200, json={"cases": AGENT_A})
    )
    respx.get(f"{BASE}/trajectories/traj-b/cases").mock(
        return_value=httpx.Response(200, json={"cases": AGENT_B})
    )
    slept = []
    client = CesClient("http://ces.test", "scope-1", api_key="k", sleep=slept.append)

    view = client.read_session("traj-s")

    assert [t.trace_id for t in view.turns] == ["c1", "c5"]
    assert [s.span_id for s in view.turns[0].subagents] == ["traj-a", "traj-b"]
    assert slept == [1.0]


def _catalog_routes():
    respx.post(f"{BASE}/evaluations").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={"evaluation_id": "ev-turns" if b'"case"' in request.content else "ev-sessions"},
        )
    )
    made = []

    def scorer(request):
        made.append(request.headers["Idempotency-Key"])
        return httpx.Response(
            200, json={"scorer_id": f"sc-{len(made)}", "version_id": f"v-{len(made)}"}
        )

    respx.post(f"{BASE}/scorers").mock(side_effect=scorer)
    respx.patch(url__regex=rf"{BASE}/evaluations/ev-.*/scorers").mock(
        return_value=httpx.Response(200, json={"scorers": []})
    )
    return made


@respx.mock
def test_the_catalog_makes_a_value_and_a_details_scorer_for_each_scorer():
    made = _catalog_routes()
    catalog = CesClient("http://ces.test", "scope-1", api_key="k").ensure_catalog("ds-1")

    assert (catalog.turns_evaluation_id, catalog.sessions_evaluation_id) == (
        "ev-turns",
        "ev-sessions",
    )
    assert made[:2] == [
        "monopole-scorer-v1:efficiency:number:case",
        "monopole-scorer-v1:efficiency.details:json:case",
    ]
    assert "monopole-scorer-v1:implicit.completion:boolean:trajectory" in made
    assert catalog.by_scorer_id()["sc-2"] == ("efficiency", True)


@respx.mock
def test_a_comparison_groups_cases_by_role_and_its_scorer_joins_no_evaluation():
    made = _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    route = respx.post(f"{BASE}/comparisons").mock(
        return_value=httpx.Response(
            200, json={"comparison_ids": ["cmp-1"], "commit_id": 1, "publish_operation_id": "op"}
        )
    )

    comparison_id = client.create_comparison(
        [("case-a", "A"), ("case-b", "B")],
        key="ab-comparison-v1:x",
        attributes={"challenge_id": "x"},
    )
    catalog = client.add_scorers(Catalog("ev-turns", "ev-sessions", {}), (VERDICT,))

    assert comparison_id == "cmp-1"
    request = route.calls.last.request
    assert request.headers["Idempotency-Key"] == "ab-comparison-v1:x"
    assert json.loads(request.content) == {
        "comparisons": [
            {
                "members": [{"case_id": "case-a", "role": "A"}, {"case_id": "case-b", "role": "B"}],
                "attributes": {"challenge_id": "x"},
            }
        ]
    }
    assert made == [
        "monopole-scorer-v1:ab.verdict:number:comparison",
        "monopole-scorer-v1:ab.verdict.details:json:comparison",
    ]
    assert all(call.request.method != "PATCH" for call in respx.calls)
    assert catalog.scorers["ab.verdict"].spec is VERDICT


@respx.mock
def test_scores_and_their_details_go_in_one_request_keyed_by_pass_and_target():
    _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")
    route = respx.post(f"{BASE}/scores").mock(
        return_value=httpx.Response(200, json={"score_ids": ["s1", "s2"]})
    )
    ids = client.write_scores(
        catalog,
        "pass-1",
        "case",
        "c1",
        [Score("efficiency", 1, ["fast"], {"waste": 0}, "turn", reason="ok")],
    )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert ids == ["s1", "s2"]
    assert request.headers["Idempotency-Key"].startswith("score-v2:pass-1:case:c1:")
    assert [s["value"] for s in body["scores"]][0] == 1.0
    assert body["scores"][1]["value"]["tags"] == ["fast"]
    assert {s["target"]["id"] for s in body["scores"]} == {"c1"}


@respx.mock
def test_a_score_outside_the_catalog_or_of_the_wrong_type_is_refused():
    _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")

    with pytest.raises(ValueError, match="not in the CES catalog"):
        client.write_scores(catalog, "p", "case", "c1", [Score("unknown", 1.0, [], {}, "turn")])
    with pytest.raises(ValueError, match="is boolean"):
        client.write_scores(
            catalog, "p", "trajectory", "s1", [Score("implicit.completion", 0.5, [], {}, "session")]
        )
    with pytest.raises(ValueError, match="is string"):
        client.write_scores(catalog, "p", "trajectory", "s1", [Score("dim.day", 3.0, [], {}, "x")])
    with pytest.raises(ValueError, match="is number"):
        client.write_scores(
            catalog, "p", "trajectory", "s1", [Score("measure.turns", "3", [], {}, "session")]
        )


@respx.mock
def test_a_dimension_score_is_written_as_a_string():
    made = _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")
    route = respx.post(f"{BASE}/scores").mock(
        return_value=httpx.Response(200, json={"score_ids": ["s1", "s2"]})
    )

    client.write_scores(
        catalog, "p", "trajectory", "s1", [Score("dim.repo", "core", [], {}, "session")]
    )

    assert "monopole-scorer-v1:dim.repo:string:trajectory" in made
    assert json.loads(route.calls.last.request.content)["scores"][0]["value"] == "core"


@respx.mock
def test_find_sessions_reads_newest_first_and_stops_before_since():
    _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")
    rows = [
        {
            "input": {
                "started_at": f"2026-11-0{day}T10:00:00.000Z",
                "session_trajectory_id": f"s{day % 2}",
            }
        }
        for day in (5, 4, 3, 1)
    ]
    route = respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(
        return_value=httpx.Response(200, json={"rows": rows, "next_cursor": "more"})
    )

    found = client.find_sessions(
        catalog,
        since=datetime(2026, 11, 2, tzinfo=timezone.utc),
        until=datetime(2026, 11, 4, 12, tzinfo=timezone.utc),
    )

    assert found == ["s0", "s1"]
    assert route.call_count == 1
    assert json.loads(route.calls.last.request.content)["order_by"] == "i.started_at desc"


@respx.mock
def test_find_sessions_skips_a_turn_without_a_start_time_instead_of_stopping():
    _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")
    rows = [
        {"input": {"session_trajectory_id": "s-unstamped"}},
        {"input": {"started_at": "2026-11-03T10:00:00.000Z", "session_trajectory_id": "s-in"}},
        {"input": {"started_at": "2026-11-01T10:00:00.000Z", "session_trajectory_id": "s-old"}},
    ]
    respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(
        return_value=httpx.Response(200, json={"rows": rows})
    )

    found = client.find_sessions(catalog, since=datetime(2026, 11, 2, tzinfo=timezone.utc))

    assert found == ["s-in"]


@respx.mock
def test_session_scores_keep_only_the_requested_trajectories():
    _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")
    respx.get(f"{BASE}/evaluations/ev-sessions/trajectory-rows").mock(
        return_value=httpx.Response(
            200,
            json={
                "rows": [
                    {"trajectory": {"trajectory_id": "s1"}, "scores": [{"scorer_id": "sc-1"}]},
                    {"trajectory": {"trajectory_id": "s9"}, "scores": []},
                ]
            },
        )
    )

    assert client.session_scores(catalog, ["s1"]) == {"s1": [{"scorer_id": "sc-1"}]}


@respx.mock
def test_a_description_is_sent_only_for_scorers_that_have_one():
    _catalog_routes()
    bodies = []
    respx.post(f"{BASE}/scorers").mock(
        side_effect=lambda request: (
            bodies.append(json.loads(request.content))
            or httpx.Response(200, json={"scorer_id": f"sc-{len(bodies)}", "version_id": "v"})
        )
    )
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")
    spec = ScorerSpec("judge.x.panel-1", "number", "session", description="judge.x v3 judged by a")

    merged = client.add_scorers(catalog, (spec,))

    assert all("description" not in body for body in bodies[: 2 * len(catalog.scorers)])
    assert [b.get("description") for b in bodies[-2:]] == ["judge.x v3 judged by a"] * 2
    assert set(merged.scorers) == {*catalog.scorers, "judge.x.panel-1"}


def test_a_turn_without_a_prompt_keeps_its_span_ids_distinct_from_its_turn_id():
    reply = _case(
        "r1", "assistant_message", "t1", 0, output={"turn_status": "aborted", "text": "partial"}
    )
    call = _case("r2", "tool_call", "t1", 1, tool_name="exec", output={"status": "completed"})

    [turn] = session_view("traj-s", [reply, call], {}).turns

    assert turn.trace_id == "r1"
    assert [c.span_id for c in turn.chat_spans] == ["r1/span"]
    assert [t.span_id for t in turn.tool_calls] == ["r2"]


@respx.mock
def test_session_scores_fall_back_to_the_local_index_without_the_route(tmp_path):
    _catalog_routes()
    client = CesClient(
        "http://ces.test", "scope-1", api_key="k", score_index=ScoreIndex(str(tmp_path / "i.jsonl"))
    )
    catalog = client.ensure_catalog("ds-1")
    respx.post(f"{BASE}/scores").mock(
        side_effect=[
            httpx.Response(200, json={"score_ids": ["old-v", "old-d"]}),
            httpx.Response(200, json={"score_ids": ["new-v", "new-d"]}),
        ]
    )
    for value in (False, True):
        score = Score("implicit.completion", value, [], {}, "session")
        client.write_scores(catalog, f"pass-{value}", "trajectory", "s1", [score])
    respx.get(f"{BASE}/evaluations/ev-sessions/trajectory-rows").mock(
        return_value=httpx.Response(404, json={"error": {"code": "not_found"}})
    )
    respx.get(url__regex=rf"{BASE}/scores/.*").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                "score_id": request.url.path.rsplit("/", 1)[-1],
                "scorer_id": "x",
                "scorer_version_id": "v",
                "value": True,
                "attributes": {},
            },
        )
    )

    rows = client.session_scores(catalog, ["s1", "s2"])

    assert sorted(r["score_id"] for r in rows["s1"]) == ["new-d", "new-v"]
    assert set(rows["s1"][0]) == {"score_id", "scorer_id", "scorer_version_id", "value"}
    assert "s2" not in rows


@respx.mock
def test_a_route_error_other_than_missing_is_not_hidden_by_the_fallback(tmp_path):
    _catalog_routes()
    client = CesClient(
        "http://ces.test", "scope-1", api_key="k", score_index=ScoreIndex(str(tmp_path / "i.jsonl"))
    )
    catalog = client.ensure_catalog("ds-1")
    respx.get(f"{BASE}/evaluations/ev-sessions/trajectory-rows").mock(
        return_value=httpx.Response(400, json={"error": {"code": "invalid_request"}})
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.session_scores(catalog, ["s1"])


@respx.mock
def test_read_all_scores_turns_ces_rows_into_feedback_records(tmp_path):
    _catalog_routes()
    client = CesClient(
        "http://ces.test", "scope-1", api_key="k", score_index=ScoreIndex(str(tmp_path / "i.jsonl"))
    )
    catalog = client.ensure_catalog("ds-1")
    case_input = {
        "config_version": "cfg1",
        "git_branch": "main",
        "started_at": "2026-11-02T10:00:00.000Z",
    }
    respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "rows": [
                    {
                        "case_id": "c1",
                        "input": case_input,
                        "scores": [
                            {
                                "score_id": "1",
                                "scorer_id": "sc-v",
                                "scorer_version_id": "v1",
                                "value": 0.9,
                            },
                            {
                                "score_id": "2",
                                "scorer_id": "sc-d",
                                "scorer_version_id": "v1",
                                "value": {
                                    "tags": ["efficient"],
                                    "reason": "fine",
                                    "granularity": "turn",
                                    "details": {"waste": 0},
                                },
                            },
                        ],
                    }
                ]
            },
        )
    )
    respx.get(f"{BASE}/evaluations/ev-sessions/trajectory-rows").mock(
        return_value=httpx.Response(
            200,
            json={
                "rows": [
                    {
                        "trajectory": {"trajectory_id": "s1", "depth": 0},
                        "root_case": {
                            "input": {**case_input, "started_at": "2026-11-03T10:00:00.000Z"}
                        },
                        "scores": [
                            {
                                "score_id": "3",
                                "scorer_id": "sc-j",
                                "scorer_version_id": "v1",
                                "value": 0.75,
                            },
                            {
                                "score_id": "4",
                                "scorer_id": "sc-jd",
                                "scorer_version_id": "v1",
                                "value": {
                                    "tags": [],
                                    "reason": "ok",
                                    "details": {"review_status": "complete"},
                                },
                            },
                        ],
                    },
                    {
                        "trajectory": {"trajectory_id": "a1", "depth": 1},
                        "root_case": {},
                        "scores": [
                            {
                                "score_id": "5",
                                "scorer_id": "sc-j",
                                "scorer_version_id": "v1",
                                "value": 0.1,
                            }
                        ],
                    },
                ]
            },
        )
    )
    names = {
        "sc-v": "efficiency",
        "sc-d": "efficiency.details",
        "sc-j": "judge.tool_choice.panel-abc",
        "sc-jd": "judge.tool_choice.panel-abc.details",
    }
    respx.get(url__regex=rf"{BASE}/scorers/[^/]+/versions/v1").mock(
        side_effect=lambda request: httpx.Response(
            200, json={"name": names[request.url.path.split("/")[-3]]}
        )
    )

    judge, turn = client.read_all_scores(catalog)

    assert (turn["feedback_type"], turn["target_id"], turn["payload"]["rating"]) == (
        "weave_agent_signals.efficiency",
        "c1",
        0.9,
    )
    assert (turn["payload"]["tags"], turn["payload"]["details"]["config_version"]) == (
        ["efficient"],
        "cfg1",
    )
    assert turn["payload"]["details"]["turn_started_at"] == "2026-11-02T10:00:00.000Z"
    assert (judge["feedback_type"], judge["target_id"]) == (
        "weave_agent_signals.judge.tool_choice",
        "s1",
    )
    assert judge["payload"]["details"]["ces_scorer"] == "judge.tool_choice.panel-abc"
    assert judge["payload"]["details"]["review_status"] == "complete"


@respx.mock
def test_a_limit_reads_the_newest_turns_and_only_their_sessions_scores():
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = Catalog("ev-turns", "ev-sessions", {})
    turns = respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "rows": [
                    {"case_id": "c-new", "input": {"session_trajectory_id": "s-new"}, "scores": []},
                    {"case_id": "c-old", "input": {"session_trajectory_id": "s-old"}, "scores": []},
                ],
                "next_cursor": "more",
            },
        )
    )
    respx.get(f"{BASE}/evaluations/ev-sessions/trajectory-rows").mock(
        return_value=httpx.Response(
            200,
            json={
                "rows": [
                    {
                        "trajectory": {"trajectory_id": tid},
                        "scores": [
                            {
                                "score_id": tid,
                                "scorer_id": "sc",
                                "scorer_version_id": "v",
                                "value": 1,
                            }
                        ],
                    }
                    for tid in ("s-new", "s-old")
                ]
            },
        )
    )
    names = respx.get(f"{BASE}/scorers/sc/versions/v").mock(
        return_value=httpx.Response(200, json={"name": "efficiency.session"})
    )

    records = client.read_all_scores(catalog, limit=1)
    client.read_all_scores(catalog, limit=1)

    assert turns.call_count == 2
    assert json.loads(turns.calls.last.request.content)["order_by"] == "i.started_at desc"
    assert [r["target_id"] for r in records] == ["s-new"]
    assert names.call_count == 1


@respx.mock
def test_cohort_scores_searches_each_sessions_turns_with_an_and_only_filter():
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    turns = respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(
        return_value=httpx.Response(200, json={"rows": []})
    )
    respx.get(f"{BASE}/evaluations/ev-sessions/trajectory-rows").mock(
        return_value=httpx.Response(200, json={"rows": []})
    )

    client.cohort_scores(Catalog("ev-turns", "ev-sessions", {}), ["s1", "s2"])

    assert sorted(json.loads(call.request.content)["filter"] for call in turns.calls) == [
        'i.step_index = 0 AND i.agent_id = null AND i.session_trajectory_id = "s1"',
        'i.step_index = 0 AND i.agent_id = null AND i.session_trajectory_id = "s2"',
    ]


def test_a_score_that_is_not_a_rating_is_skipped():
    names = {("sc", "v"): "efficiency", ("sc2", "v"): "outcome.test"}
    scores = [
        {"score_id": "1", "scorer_id": "sc", "scorer_version_id": "v", "value": "high"},
        {"score_id": "2", "scorer_id": "sc2", "scorer_version_id": "v", "value": 0.5},
    ]

    records = _score_record("c1", {}, scores, names)

    assert [(r["feedback_type"], r["payload"]["rating"]) for r in records] == [
        ("weave_agent_signals.outcome.test", 0.5)
    ]


def test_session_view_skips_a_case_without_a_turn_and_reads_an_offsetless_time_as_utc():
    stray = {"case_id": "c-stray", "input": {"kind": "user_message"}, "output": {}}
    naive = _case("c-naive", "assistant_message", "t1", 9, started_at="2026-11-02T10:01:09.000")

    view = session_view("traj-s", [*SESSION, stray, naive], {})

    assert all(turn.started_at.tzinfo is not None for turn in view.turns)
    assert "c-stray" not in {case_id for turn in view.turns for case_id in turn.case_ids}
    assert "c-naive" in {case_id for turn in view.turns for case_id in turn.case_ids}


def test_a_tool_calls_exit_code_reaches_the_outcome_scorers():
    case = {
        "case_id": "c-tool",
        "input": {
            "kind": "tool_call",
            "tool_name": "Bash",
            "started_at": "2026-11-02T10:00:00.000Z",
        },
        "output": {
            "status": "failed",
            "output": "1 failed",
            "exit_code": 2,
            "permission_decision": "approved",
        },
    }

    span = _tool_span(case)

    assert span.exit_code == 2
    assert _extract_exit_code(span) == 2
    assert span.permission_decision == "approved"


@respx.mock
def test_a_score_from_an_unknown_scorer_version_is_skipped():
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    respx.get(f"{BASE}/scorers/gone/versions/v").mock(return_value=httpx.Response(404, json={}))
    respx.get(f"{BASE}/scorers/sc/versions/v").mock(
        return_value=httpx.Response(200, json={"name": "efficiency"})
    )
    scores = [
        {"score_id": "1", "scorer_id": "gone", "scorer_version_id": "v", "value": 0.1},
        {"score_id": "2", "scorer_id": "sc", "scorer_version_id": "v", "value": 0.9},
    ]

    records = client._score_records([("c1", {}, scores)])

    assert [r["feedback_type"] for r in records] == ["weave_agent_signals.efficiency"]


@respx.mock
def test_find_sessions_compares_times_not_strings_at_the_window_edge():
    _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")
    rows = [
        {"input": {"started_at": "2026-11-02T00:00:00+00:00", "session_trajectory_id": "s-edge"}},
        {"input": {"started_at": "2026-11-01T23:59:59+00:00", "session_trajectory_id": "s-old"}},
    ]
    respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(
        return_value=httpx.Response(200, json={"rows": rows})
    )

    found = client.find_sessions(catalog, since=datetime(2026, 11, 2, tzinfo=timezone.utc))

    assert found == ["s-edge"]


@respx.mock
def test_a_scores_key_repeats_for_the_same_body_and_changes_with_it():
    _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")
    route = respx.post(f"{BASE}/scores").mock(
        return_value=httpx.Response(200, json={"score_ids": ["s1", "s2"]})
    )

    def key(value):
        client.write_scores(
            catalog, "pass-1", "case", "c1", [Score("efficiency", value, [], {}, "turn")]
        )
        return route.calls.last.request.headers["Idempotency-Key"]

    first, same, changed = key(0.5), key(0.5), key(0.75)

    assert first == same != changed


@respx.mock
def test_a_5xx_is_retried_and_a_transport_error_raises_after_the_last_attempt():
    slept = []
    client = CesClient("http://ces.test", "scope-1", api_key="k", sleep=slept.append)
    respx.get(f"{BASE}/scores/s1").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"score_id": "s1"})]
    )
    respx.get(f"{BASE}/scores/s2").mock(side_effect=httpx.ConnectError("down"))

    assert client.get_score("s1") == {"score_id": "s1"}
    with pytest.raises(httpx.ConnectError):
        client.get_score("s2")
    assert slept == [1, 1, 2, 4, 8, 16]


@respx.mock
def test_a_long_retry_after_is_capped():
    slept = []
    client = CesClient("http://ces.test", "scope-1", api_key="k", sleep=slept.append)
    respx.get(f"{BASE}/scores/s1").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "120"}),
            httpx.Response(200, json={}),
        ]
    )

    client.get_score("s1")

    assert slept == [MAX_RETRY_AFTER_S]


@respx.mock
def test_find_sessions_stops_reading_at_its_limit():
    _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")
    rows = [
        {"input": {"started_at": "2026-11-03T10:00:00.000Z", "session_trajectory_id": sid}}
        for sid in ("s1", "s2")
    ]
    route = respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(
        return_value=httpx.Response(200, json={"rows": rows, "next_cursor": "more"})
    )

    found = client.find_sessions(catalog, since=datetime(2026, 11, 1, tzinfo=timezone.utc), limit=1)

    assert found == ["s1"]
    assert route.call_count == 1


@respx.mock
def test_turn_scores_are_keyed_by_each_turns_first_case():
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(
        return_value=httpx.Response(
            200,
            json={"rows": [{"case_id": "c1", "scores": [{"score_id": "x"}]}, {"case_id": "c5"}]},
        )
    )

    scores = client.turn_scores(Catalog("ev-turns", "ev-sessions", {}), "traj-s")

    assert scores == {"c1": [{"score_id": "x"}], "c5": []}


def _row(
    case_id, *, kind, turn_id, turn_index, text=None, duration=None, agent_id=None, session="traj-s"
):
    return {
        "case_id": case_id,
        "input": {
            "kind": kind,
            "turn_id": turn_id,
            "turn_index": turn_index,
            "step_index": 0,
            "started_at": f"2026-11-02T10:0{turn_index}:00.000Z",
            "text": text,
            "agent_id": agent_id,
            "model": "gpt-6",
            "session_id": "sess-1",
            "session_trajectory_id": session,
            "trace_role": "agent_session",
        },
        "output": {
            "turn_input_tokens": 7,
            "turn_output_tokens": 3,
            "turn_status": "completed",
            **({"turn_duration_ms": duration} if duration is not None else {}),
        },
        "scores": [],
    }


def test_turns_from_rows_keeps_the_sessions_own_turns_in_order_with_their_totals():
    rows = [
        _row("c-t2", kind="user_message", turn_id="t2", turn_index=2, text="next", duration=2000),
        _row("c-sub", kind="user_message", turn_id="k1", turn_index=0, agent_id="explorer"),
        _row("c-t1", kind="compaction", turn_id="t1", turn_index=1),
        _row("c-session", kind="session", turn_id=None, turn_index=0),
    ]

    first, second = turns_from_rows("traj-s", rows)

    assert (first.trace_id, first.user_input, first.ended_at) == ("c-t1", None, None)
    assert (second.trace_id, second.user_input, second.input_tokens, second.status_code) == (
        "c-t2",
        "next",
        7,
        "OK",
    )
    assert second.ended_at - second.started_at == timedelta(seconds=2)


@respx.mock
def test_cohort_scores_reads_no_turn_rows_it_is_given():
    turns = respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(
        return_value=httpx.Response(200, json={"rows": []})
    )
    respx.get(f"{BASE}/evaluations/ev-sessions/trajectory-rows").mock(
        return_value=httpx.Response(200, json={"rows": []})
    )

    CesClient("http://ces.test", "scope-1", api_key="k").cohort_scores(
        Catalog("ev-turns", "ev-sessions", {}), ["s1"], turn_rows={"s1": []}
    )

    assert turns.call_count == 0


@respx.mock
def test_read_session_reads_each_subagent_once_in_launch_order():
    root = [
        _case("c1", "user_message", "t1", 0, text="go"),
        _case(
            "c2", "tool_call", "t1", 1, children=["traj-a", "traj-b", "traj-a"], tool_name="Task"
        ),
    ]
    respx.get(f"{BASE}/trajectories/traj-s/cases").mock(
        return_value=httpx.Response(200, json={"cases": root})
    )
    reads = {
        name: respx.get(f"{BASE}/trajectories/{name}/cases").mock(
            return_value=httpx.Response(200, json={"cases": cases})
        )
        for name, cases in {
            "traj-a": [_case("a1", "tool_call", "k1", 0, children=["traj-c"], tool_name="Task")],
            "traj-b": [_case("b1", "tool_call", "k1", 0, children=["traj-c"], tool_name="Task")],
            "traj-c": [_case("n1", "tool_call", "n1", 0, tool_name="Bash")],
        }.items()
    }

    view = CesClient("http://ces.test", "scope-1", api_key="k").read_session("traj-s")

    assert [s.span_id for s in view.turns[0].subagents] == ["traj-a", "traj-b", "traj-c"]
    assert {name: route.call_count for name, route in reads.items()} == {
        "traj-a": 1,
        "traj-b": 1,
        "traj-c": 1,
    }


@respx.mock
def test_clients_sharing_a_scorer_name_cache_look_each_version_up_once():
    version = respx.get(f"{BASE}/scorers/sc-1/versions/v-1").mock(
        return_value=httpx.Response(200, json={"name": "efficiency"})
    )
    names: dict[tuple[str, str], str] = {}
    score = {"score_id": "s-1", "scorer_id": "sc-1", "scorer_version_id": "v-1", "value": 0.5}

    for _ in range(2):
        client = CesClient("http://ces.test", "scope-1", api_key="k", scorer_names=names)
        assert client._score_records([("case-1", {}, [score])])[0]["target_id"] == "case-1"

    assert version.call_count == 1


@respx.mock
def test_read_session_without_subagents_names_them_without_reading_them():
    respx.get(f"{BASE}/trajectories/traj-s/cases").mock(
        return_value=httpx.Response(200, json={"cases": SESSION})
    )
    child = respx.get(f"{BASE}/trajectories/traj-a/cases").mock(
        return_value=httpx.Response(200, json={"cases": AGENT_A})
    )

    view = CesClient("http://ces.test", "scope-1", api_key="k").read_session(
        "traj-s", subagents=False
    )

    [subagent] = view.turns[0].subagents
    assert (subagent.span_id, subagent.agent_type, subagent.tool_calls) == ("traj-a", None, [])
    assert child.call_count == 0


@respx.mock
def test_turn_subagents_reads_only_that_turns_subagents():
    root = [
        _case("c1", "user_message", "t1", 0, text="go"),
        _case("c2", "tool_call", "t1", 1, children=["traj-a"], tool_name="Task"),
        _case("c3", "user_message", "t2", 0, text="more"),
        _case("c4", "tool_call", "t2", 1, children=["traj-b"], tool_name="Task"),
    ]
    respx.get(f"{BASE}/trajectories/traj-s/cases").mock(
        return_value=httpx.Response(200, json={"cases": root})
    )
    first = respx.get(f"{BASE}/trajectories/traj-a/cases").mock(
        return_value=httpx.Response(200, json={"cases": AGENT_B})
    )
    second = respx.get(f"{BASE}/trajectories/traj-b/cases").mock(
        return_value=httpx.Response(200, json={"cases": AGENT_B})
    )
    respx.get(f"{BASE}/trajectories/traj-gone/cases").mock(return_value=httpx.Response(404))
    client = CesClient("http://ces.test", "scope-1", api_key="k")

    subagents = client.turn_subagents("traj-s", "c3")

    assert [(s.span_id, len(s.tool_calls)) for s in subagents] == [("traj-b", 1)]
    assert (first.call_count, second.call_count) == (0, 1)
    assert client.turn_subagents("traj-s", "c4") is None  # a turn is named by its first Case
    assert client.turn_subagents("traj-gone", "c1") is None


@respx.mock
def test_read_session_also_views_each_subagent_trajectory_with_its_own_subagents():
    respx.get(f"{BASE}/trajectories/traj-s/cases").mock(
        return_value=httpx.Response(200, json={"cases": SESSION})
    )
    respx.get(f"{BASE}/trajectories/traj-a/cases").mock(
        return_value=httpx.Response(200, json={"cases": AGENT_A})
    )
    respx.get(f"{BASE}/trajectories/traj-b/cases").mock(
        return_value=httpx.Response(200, json={"cases": AGENT_B})
    )

    view = CesClient("http://ces.test", "scope-1", api_key="k").read_session("traj-s")

    agent_a, agent_b = view.subagent_trajectories
    assert [t.trace_id for t in agent_a.turns] == ["a1"]
    # traj-b descends from traj-a's own Case a2, so it attaches to traj-a's turn.
    assert [s.span_id for s in agent_a.turns[0].subagents] == ["traj-b"]
    assert (agent_b.conversation_id, [t.trace_id for t in agent_b.turns]) == ("traj-b", ["b1"])
    assert agent_b.turns[0].subagents == []


@respx.mock
def test_session_scores_with_subagents_adds_every_trajectory_below_the_sessions():
    def row(trajectory_id, depth, parent=None):
        trajectory = {"trajectory_id": trajectory_id, "depth": depth}
        trajectory["parent_trajectory_id"] = parent
        return {"trajectory": trajectory, "root_case": {}, "scores": [{"score_id": trajectory_id}]}

    rows = [
        row("s1", 0),
        row("a1", 1, "s1"),
        row("b1", 2, "a1"),
        row("s2", 0),
        row("a2", 1, "s2"),
    ]
    respx.get(f"{BASE}/evaluations/ev-sessions/trajectory-rows").mock(
        return_value=httpx.Response(200, json={"rows": rows})
    )
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = Catalog("ev-turns", "ev-sessions", {})

    assert set(client.session_scores(catalog, ["s1"], subagents=True)) == {"s1", "a1", "b1"}
    assert set(client.session_scores(catalog, ["s1"])) == {"s1"}


def test_a_subagent_span_counts_the_tokens_of_its_own_turns():
    totals = {"turn_input_tokens": 40, "turn_output_tokens": 2}
    child = [
        _case("a1", "user_message", "k1", 0, output=totals),
        _case("a2", "assistant_message", "k1", 1, output={"input_tokens": 99}),
        _case("a3", "user_message", "k2", 0, output=totals),
    ]

    view = session_view("traj-s", SESSION, {"traj-a": child})

    assert view.turns[0].subagents[0].tokens == 84


def test_a_measurement_score_never_becomes_a_rating_record():
    names = {("sc-m", "v"): "measure.turns", ("sc-r", "v"): "implicit.completion"}
    scores = [
        {"score_id": "1", "scorer_id": "sc-m", "scorer_version_id": "v", "value": 1.0},
        {"score_id": "2", "scorer_id": "sc-r", "scorer_version_id": "v", "value": True},
    ]

    records = _score_record("s1", {}, scores, names)

    assert [r["feedback_type"] for r in records] == ["weave_agent_signals.implicit.completion"]


@respx.mock
def test_root_case_input_reads_the_trajectory_then_its_root_case():
    respx.get(f"{BASE}/trajectories/traj-s").mock(
        return_value=httpx.Response(200, json={"root_case_id": "c-session"})
    )
    respx.get(f"{BASE}/cases/c-session").mock(
        return_value=httpx.Response(200, json={"input": {"kind": "session", "cwd": "/r/core"}})
    )

    root = CesClient("http://ces.test", "scope-1", api_key="k").root_case_input("traj-s")

    assert root == {"kind": "session", "cwd": "/r/core"}


class _InFlight:
    """A respx side effect that holds each request briefly and records the most in flight."""

    def __init__(self, response: dict):
        self.response, self.now, self.most = response, 0, 0
        self.lock = threading.Lock()

    def __call__(self, request):
        with self.lock:
            self.now += 1
            self.most = max(self.most, self.now)
        time.sleep(0.05)
        with self.lock:
            self.now -= 1
        return httpx.Response(200, json=self.response)


@respx.mock
def test_reads_that_take_a_ces_query_slot_run_at_most_query_slots_at_once():
    rows = _InFlight({"rows": []})
    respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(side_effect=rows)
    launch = _case("c1", "tool_call", "t1", 0, children=[f"traj-{i}" for i in range(8)])
    respx.get(f"{BASE}/trajectories/traj-s/cases").mock(
        return_value=httpx.Response(200, json={"cases": [launch]})
    )
    cases = _InFlight({"cases": []})
    respx.get(url__regex=rf"{BASE}/trajectories/traj-\d/cases").mock(side_effect=cases)
    client = CesClient("http://ces.test", "scope-1", api_key="k")

    client.session_turn_rows(Catalog("ev-turns", "ev-sessions", {}), [f"s{i}" for i in range(8)])
    client.read_session("traj-s")

    assert rows.most == QUERY_SLOTS
    assert cases.most > QUERY_SLOTS  # plain entity reads take no slot


@respx.mock
def test_row_searches_page_by_the_row_search_limit_and_other_rows_by_the_page_limit():
    catalog = Catalog("ev-turns", "ev-sessions", {})
    search = respx.post(f"{BASE}/evaluations/ev-turns/rows/search").mock(
        return_value=httpx.Response(200, json={"rows": []})
    )
    listing = respx.get(f"{BASE}/evaluations/ev-sessions/trajectory-rows").mock(
        return_value=httpx.Response(200, json={"rows": []})
    )
    client = CesClient("http://ces.test", "scope-1", api_key="k")

    client.read_all_scores(catalog)
    client.find_sessions(catalog, since=datetime(2026, 11, 1, tzinfo=timezone.utc))

    assert [json.loads(c.request.content)["limit"] for c in search.calls] == [
        ROW_SEARCH_PAGE_LIMIT,
        ROW_SEARCH_PAGE_LIMIT,
    ]
    assert listing.calls[0].request.url.params["limit"] == str(PAGE_LIMIT)
