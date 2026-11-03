"""CES reads: session reconstruction from the adapter's Cases, paging, and retries."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
import respx

from weave_agent_signals.ces_client import CesClient, session_view
from weave_agent_signals.models import Score

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
        return_value=httpx.Response(200, json={})
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
    assert request.headers["Idempotency-Key"] == "score-v1:pass-1:case:c1"
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
def test_session_scores_keep_only_the_requested_trajectories():
    _catalog_routes()
    client = CesClient("http://ces.test", "scope-1", api_key="k")
    catalog = client.ensure_catalog("ds-1")
    respx.post(f"{BASE}/evaluations/ev-sessions/trajectory-rows/search").mock(
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
