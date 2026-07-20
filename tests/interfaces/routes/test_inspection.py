from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from weave_agent_signals.judges.tokens import count_tokens
from weave_agent_signals.judges.windowing import render_raw_turn
from weave_agent_signals.models import SessionView, ToolSpan, TraceRole, TurnSpan
from weave_agent_signals.routes.inspection import create_inspection_router


def _turn(
    trace_id: str,
    conversation_id: str,
    *,
    hour: int,
    user_input: str,
    trace_role: TraceRole = TraceRole.AGENT_SESSION,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> TurnSpan:
    started = datetime(2026, 7, 14, hour, tzinfo=timezone.utc)
    return TurnSpan(
        trace_id=trace_id,
        conversation_id=conversation_id,
        started_at=started,
        ended_at=started + timedelta(minutes=1),
        model="gpt-5.6-sol",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        status_code="SUCCESS",
        config_version="config-v1",
        git_branch="main",
        effort_level="medium",
        session_id=f"sid-{conversation_id}",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
        user_input=user_input,
        assistant_output="done",
        trace_role=trace_role,
    )


class FakeClient:
    entity = "weave-team"
    project = "agent-sessions"

    def __init__(self):
        self.turns = [
            _turn("trace-a", "session/one", hour=12, user_input="Build the feature"),
            _turn("trace-b", "session-two", hour=11, user_input="Fix the bug"),
            _turn(
                "trace-judge",
                "synthetic",
                hour=10,
                user_input="## Scoring criteria\nJudge this",
                trace_role=TraceRole.JUDGE_EVALUATION,
            ),
        ]
        self.query_calls = []
        self.hydrated = []
        self.hydrated_tool = None
        self.feedback_batches = []
        self.feedback_by_ref = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def query_turns_paginated(self, **kwargs):
        self.query_calls.append(kwargs)
        return list(self.turns)

    def query_session(self, conversation_id: str):
        turns = [turn for turn in self.turns if turn.conversation_id == conversation_id]
        return SessionView(
            conversation_id=conversation_id,
            turns=turns,
            config_version="config-v1",
            git_branch="main",
        )

    def hydrate_turns_batch(self, turns):
        self.hydrated.append([turn.trace_id for turn in turns])
        if self.hydrated_tool is not None:
            turns[0].tool_calls.append(self.hydrated_tool)

    def query_all_feedback_batch(self, refs: list[str]):
        self.feedback_batches.append(refs)
        return {
            ref: self.feedback_by_ref.get(
                ref,
                [
                    {
                        "id": "feedback-1",
                        "weave_ref": ref,
                        "feedback_type": "weave_agent_signals.efficiency",
                        "payload": {"rating": 1.0},
                        "created_at": "2026-07-14T12:05:00+00:00",
                    }
                ],
            )
            for ref in refs
        }

    def query_project_feedback(self, *, limit: int):
        assert limit == 1000
        return []


def _client():
    backend = FakeClient()
    app = FastAPI()
    app.include_router(create_inspection_router(lambda: backend))
    return TestClient(app), backend


def test_session_listing_filters_synthetic_sessions_and_reports_truncation():
    client, backend = _client()

    response = client.get(
        "/api/sessions",
        params={
            "since": "2026-07-14T09:00:00-03:00",
            "until": "2026-07-14T13:00:00+00:00",
            "timezone": "UTC",
            "limit": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["truncated"] is False
    assert response.json()["sessions"][0]["conversation_id"] == "session/one"
    assert "judging_token_estimates" not in response.json()["sessions"][0]
    assert backend.query_calls == [
        {
            "page_size": 500,
            "since": datetime(2026, 7, 14, 12, tzinfo=timezone.utc),
            "include_details": True,
        }
    ]
    assert response.json()["sessions"][0]["signal_evidence"] == []
    assert backend.feedback_batches == [[backend.turns[0].ref_for(backend.entity, backend.project)]]


def test_session_listing_reports_usage_total_without_unhydrated_judging_estimates():
    client, backend = _client()
    backend.turns.append(
        _turn(
            "trace-a2",
            "session/one",
            hour=13,
            user_input="Continue",
            input_tokens=80,
            output_tokens=20,
        )
    )

    response = client.get(
        "/api/sessions",
        params={"since": "2026-07-14T12:00:00Z", "until": "2026-07-14T14:00:00Z"},
    )

    summary = response.json()["sessions"][0]
    assert summary["total_tokens"] == 115
    assert "judging_token_estimates" not in summary


def test_session_listing_excludes_explicit_non_agent_and_mixed_sessions():
    client, backend = _client()
    backend.turns.extend(
        [
            _turn(
                "trace-signal",
                "signal-session",
                hour=9,
                user_input="Ordinary-looking evaluator prompt",
                trace_role=TraceRole.SIGNAL_EVALUATION,
            ),
            _turn("trace-mixed-agent", "mixed-session", hour=8, user_input="User work"),
            _turn(
                "trace-mixed-judge",
                "mixed-session",
                hour=7,
                user_input="Continuation",
                trace_role=TraceRole.JUDGE_EVALUATION,
            ),
        ]
    )

    response = client.get(
        "/api/sessions",
        params={"since": "2026-07-14T07:00:00Z", "until": "2026-07-14T13:00:00Z"},
    )

    assert response.status_code == 200
    assert [item["conversation_id"] for item in response.json()["sessions"]] == [
        "session/one",
        "session-two",
    ]


def test_session_listing_checks_roles_outside_requested_date_range():
    client, backend = _client()
    backend.turns.extend(
        [
            _turn(
                "trace-hidden-judge",
                "mixed-across-range",
                hour=11,
                user_input="Earlier evaluation",
                trace_role=TraceRole.JUDGE_EVALUATION,
            ),
            _turn(
                "trace-visible-agent",
                "mixed-across-range",
                hour=12,
                user_input="Visible user work",
            ),
        ]
    )

    response = client.get(
        "/api/sessions",
        params={"since": "2026-07-14T12:00:00Z", "until": "2026-07-14T13:00:00Z"},
    )

    assert response.status_code == 200
    assert [item["conversation_id"] for item in response.json()["sessions"]] == ["session/one"]


@pytest.mark.parametrize("conversation_id", ["signal-session", "mixed-session"])
def test_session_detail_rejects_non_agent_and_mixed_sessions(conversation_id):
    client, backend = _client()
    backend.turns.extend(
        [
            _turn(
                "trace-signal",
                "signal-session",
                hour=9,
                user_input="Evaluator",
                trace_role=TraceRole.SIGNAL_EVALUATION,
            ),
            _turn("trace-mixed-agent", "mixed-session", hour=8, user_input="User work"),
            _turn(
                "trace-mixed-system",
                "mixed-session",
                hour=7,
                user_input="System work",
                trace_role=TraceRole.OTHER_SYSTEM,
            ),
        ]
    )

    response = client.get(f"/api/sessions/{conversation_id}")

    assert response.status_code == 404
    assert backend.hydrated == []
    assert backend.feedback_batches == []


def test_session_listing_hydrates_low_signal_feedback_from_exact_turn_refs():
    client, backend = _client()
    turn = backend.turns[0]
    turn_ref = turn.ref_for(backend.entity, backend.project)
    backend.feedback_by_ref[turn_ref] = [
        {
            "id": "feedback-low",
            "weave_ref": turn_ref,
            "feedback_type": "wandb.agent_monitor",
            "runnable_ref": (
                "weave:///weave-team/agent-sessions/object/"
                "agent-signal-user-frustration-v1-scorer:digest"
            ),
            "created_at": "2026-07-14T12:02:00Z",
            "scorer_ratings": {"_rating_": 0.25},
            "payload": {
                "output": {
                    "value": 0.25,
                    "reason": "The user explicitly says they are frustrated.",
                }
            },
        }
    ]

    response = client.get(
        "/api/sessions",
        params={
            "since": "2026-07-14T12:00:00Z",
            "until": "2026-07-14T13:00:00Z",
        },
    )

    assert response.status_code == 200
    assert response.json()["sessions"][0]["signal_evidence"] == [
        {
            "signal": "user-frustration",
            "version": "v1",
            "rating": 0.25,
            "reason": "The user explicitly says they are frustrated.",
            "turn_id": "trace-a",
            "turn_started_at": "2026-07-14T12:00:00+00:00",
        }
    ]


def test_session_listing_uses_newest_low_signal_and_ignores_healthy_or_unrelated_rows():
    client, backend = _client()
    turn = backend.turns[0]
    turn_ref = turn.ref_for(backend.entity, backend.project)
    scorer_ref = (
        "weave:///weave-team/agent-sessions/object/agent-signal-user-frustration-v1-scorer:digest"
    )
    backend.feedback_by_ref[turn_ref] = [
        {
            "id": "feedback-old",
            "feedback_type": "wandb.agent_monitor",
            "runnable_ref": scorer_ref,
            "created_at": "2026-07-14T12:01:00Z",
            "scorer_ratings": {"_rating_": 0.5},
            "payload": {"output": {"value": 0.5, "reason": "Old reason"}},
        },
        {
            "id": "feedback-new",
            "feedback_type": "wandb.agent_monitor",
            "runnable_ref": scorer_ref,
            "created_at": "2026-07-14T12:02:00Z",
            "scorer_ratings": {"_rating_": 0.25},
            "payload": {"output": {"rating": 0.25, "reason": "New reason"}},
        },
        {
            "id": "feedback-healthy",
            "feedback_type": "wandb.agent_monitor",
            "runnable_ref": (
                "weave:///weave-team/agent-sessions/object/"
                "agent-signal-low-quality-response-v1-scorer:digest"
            ),
            "created_at": "2026-07-14T12:03:00Z",
            "scorer_ratings": {"_rating_": 0.75},
            "payload": {"output": {"value": 0.75, "reason": "No material issue"}},
        },
        {
            "id": "feedback-unrelated",
            "feedback_type": "wandb.agent_monitor",
            "runnable_ref": "weave:///weave-team/agent-sessions/object/other-scorer:digest",
        },
    ]

    response = client.get(
        "/api/sessions",
        params={"since": "2026-07-14T12:00:00Z", "until": "2026-07-14T13:00:00Z"},
    )

    assert response.status_code == 200
    assert response.json()["sessions"][0]["signal_evidence"] == [
        {
            "signal": "user-frustration",
            "version": "v1",
            "rating": 0.25,
            "reason": "New reason",
            "turn_id": "trace-a",
            "turn_started_at": "2026-07-14T12:00:00+00:00",
        }
    ]


def test_session_listing_newer_healthy_signal_supersedes_older_low_signal():
    client, backend = _client()
    turn = backend.turns[0]
    turn_ref = turn.ref_for(backend.entity, backend.project)
    scorer_ref = (
        "weave:///weave-team/agent-sessions/object/agent-signal-user-frustration-v1-scorer:digest"
    )
    backend.feedback_by_ref[turn_ref] = [
        {
            "id": "feedback-low",
            "feedback_type": "wandb.agent_monitor",
            "runnable_ref": scorer_ref,
            "created_at": "2026-07-14T12:01:00Z",
            "scorer_ratings": {"_rating_": 0.25},
            "payload": {"output": {"value": 0.25, "reason": "Old low result"}},
        },
        {
            "id": "feedback-healthy",
            "feedback_type": "wandb.agent_monitor",
            "runnable_ref": scorer_ref,
            "created_at": "2026-07-14T12:02:00Z",
            "scorer_ratings": {"_rating_": 0.75},
            "payload": {"output": {"value": 0.75, "reason": "New healthy result"}},
        },
    ]

    response = client.get(
        "/api/sessions",
        params={"since": "2026-07-14T12:00:00Z", "until": "2026-07-14T13:00:00Z"},
    )

    assert response.status_code == 200
    assert response.json()["sessions"][0]["signal_evidence"] == []


def test_session_listing_ignores_legacy_signal_feedback_without_typed_rating():
    client, backend = _client()
    turn_ref = backend.turns[0].ref_for(backend.entity, backend.project)
    backend.feedback_by_ref[turn_ref] = [
        {
            "id": "feedback-legacy",
            "feedback_type": "wandb.agent_monitor",
            "runnable_ref": (
                "weave:///weave-team/agent-sessions/object/"
                "agent-signal-user-frustration-v1-scorer:legacy"
            ),
            "created_at": "2026-07-14T12:02:00Z",
            "scorer_ratings": {},
            "payload": {
                "output": {
                    "value": "0.25",
                    "reason": "Legacy monitor emitted a numeric string.",
                }
            },
        }
    ]

    response = client.get(
        "/api/sessions",
        params={"since": "2026-07-14T12:00:00Z", "until": "2026-07-14T13:00:00Z"},
    )

    assert response.status_code == 200
    assert response.json()["sessions"][0]["signal_evidence"] == []


@pytest.mark.parametrize(
    ("output", "created_at", "scorer_rating"),
    [
        ({"value": True, "reason": "Invalid"}, "2026-07-14T12:02:00Z", True),
        ({"value": 0.3, "reason": "Invalid"}, "2026-07-14T12:02:00Z", 0.3),
        (
            {"rating": 0.25, "value": 0.5, "reason": "Invalid"},
            "2026-07-14T12:02:00Z",
            0.25,
        ),
        ({"value": 0.25, "reason": "  "}, "2026-07-14T12:02:00Z", 0.25),
        ({"value": 0.25, "reason": "Invalid"}, "2026-07-14T12:02:00", 0.25),
        ({"value": 0.25, "reason": "Invalid"}, "2026-07-14T12:02:00Z", 0.5),
    ],
)
def test_session_listing_rejects_malformed_eligible_signal_feedback(
    output,
    created_at,
    scorer_rating,
):
    client, backend = _client()
    turn_ref = backend.turns[0].ref_for(backend.entity, backend.project)
    backend.feedback_by_ref[turn_ref] = [
        {
            "id": "feedback-malformed",
            "feedback_type": "wandb.agent_monitor",
            "runnable_ref": (
                "weave:///weave-team/agent-sessions/object/"
                "agent-signal-user-frustration-v1-scorer:digest"
            ),
            "created_at": created_at,
            "scorer_ratings": {"_rating_": scorer_rating},
            "payload": {"output": output},
        }
    ]

    with pytest.raises(ValueError, match="eligible Signal feedback"):
        client.get(
            "/api/sessions",
            params={
                "since": "2026-07-14T12:00:00Z",
                "until": "2026-07-14T13:00:00Z",
            },
        )


def test_session_detail_hydrates_children_and_returns_feedback():
    client, backend = _client()
    started = backend.turns[0].started_at
    backend.hydrated_tool = ToolSpan(
        span_id="tool-hydrated",
        tool_name="Read",
        arguments="input " * 100,
        result="output " * 100,
        status_code="SUCCESS",
        started_at=started,
        ended_at=started + timedelta(seconds=1),
    )

    response = client.get("/api/sessions/session%2Fone")

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "session/one"
    assert body["turns"][0]["trace_id"] == "trace-a"
    assert body["session_feedback"][0]["id"] == "feedback-1"
    assert body["turn_feedback"]["trace-a"][0]["id"] == "feedback-1"
    rendered = render_raw_turn(backend.turns[0], 1)
    assert body["judging_token_estimates"]["utf8_bytes_div_3"] == {
        "total_tokens": count_tokens(rendered, "utf8_bytes_div_3"),
        "largest_turn_tokens": count_tokens(rendered, "utf8_bytes_div_3"),
        "turn_tokens": [count_tokens(rendered, "utf8_bytes_div_3")],
    }
    assert backend.hydrated == [["trace-a"]]
    assert len(backend.feedback_batches) == 1
    assert len(backend.feedback_batches[0]) == 2


def test_session_detail_segments_scores_signals_and_unknown_feedback():
    client, backend = _client()
    turn = backend.turns[0]
    turn_ref = turn.ref_for(backend.entity, backend.project)
    session_ref = SessionView(
        conversation_id=turn.conversation_id,
        turns=[turn],
        config_version=turn.config_version,
        git_branch=turn.git_branch,
    ).ref_for(backend.entity, backend.project)
    score = {
        "id": "score-1",
        "feedback_type": "weave_agent_signals.outcome.test",
        "created_at": "2026-07-14T12:01:00Z",
        "payload": {"rating": 1.0, "tags": [], "details": {}},
    }
    signal = {
        "id": "signal-1",
        "feedback_type": "wandb.agent_monitor",
        "runnable_ref": (
            "weave:///weave-team/agent-sessions/object/"
            "agent-signal-user-frustration-v1-scorer:digest"
        ),
        "created_at": "2026-07-14T12:02:00Z",
        "scorer_ratings": {"_rating_": 0.25},
        "payload": {
            "output": {
                "rating": 0.25,
                "reason": "The user explicitly says they are frustrated.",
            }
        },
    }
    unknown = {
        "id": "unknown-1",
        "feedback_type": "another.product.feedback",
        "created_at": "2026-07-14T12:03:00Z",
        "payload": {"shape": "not-a-score"},
    }
    backend.feedback_by_ref[session_ref] = [score, unknown]
    backend.feedback_by_ref[turn_ref] = [score, signal, unknown]

    response = client.get("/api/sessions/session%2Fone")

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["session_feedback"]] == ["score-1"]
    assert [item["id"] for item in body["turn_feedback"]["trace-a"]] == ["score-1"]
    assert body["signal_evidence"] == [
        {
            "signal": "user-frustration",
            "version": "v1",
            "rating": 0.25,
            "reason": "The user explicitly says they are frustrated.",
            "turn_id": "trace-a",
            "turn_started_at": "2026-07-14T12:00:00+00:00",
        }
    ]


def test_session_detail_partitions_scores_from_agent_signal_evidence():
    client, backend = _client()
    turn = backend.turns[0]
    turn_ref = turn.ref_for(backend.entity, backend.project)
    backend.feedback_by_ref[turn_ref] = [
        {
            "id": "score-feedback",
            "weave_ref": turn_ref,
            "feedback_type": "weave_agent_signals.efficiency",
            "payload": {"rating": 1.0},
            "created_at": "2026-07-14T12:01:00Z",
        },
        {
            "id": "signal-feedback",
            "weave_ref": turn_ref,
            "feedback_type": "wandb.agent_monitor",
            "runnable_ref": (
                "weave:///weave-team/agent-sessions/object/"
                "agent-signal-user-frustration-v1-scorer:digest"
            ),
            "created_at": "2026-07-14T12:02:00Z",
            "scorer_ratings": {"_rating_": 0.25},
            "payload": {
                "output": {
                    "value": 0.25,
                    "reason": "The user explicitly says they are frustrated.",
                }
            },
        },
        {
            "id": "unknown-feedback",
            "weave_ref": turn_ref,
            "feedback_type": "other.product.score",
            "payload": {"value": 0.9},
            "created_at": "2026-07-14T12:03:00Z",
        },
    ]

    response = client.get("/api/sessions/session%2Fone")

    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body["turn_feedback"]["trace-a"]] == ["score-feedback"]
    assert body["signal_evidence"] == [
        {
            "signal": "user-frustration",
            "version": "v1",
            "rating": 0.25,
            "reason": "The user explicitly says they are frustrated.",
            "turn_id": "trace-a",
            "turn_started_at": "2026-07-14T12:00:00+00:00",
        }
    ]


def test_inspection_errors_are_http_specific_and_empty_analysis_is_stable():
    client, _backend = _client()

    invalid_timezone = client.get("/api/sessions", params={"timezone": "Mars/Olympus"})
    invalid_range = client.get(
        "/api/sessions",
        params={
            "since": "2026-07-15T00:00:00+00:00",
            "until": "2026-07-14T00:00:00+00:00",
        },
    )
    analysis = client.get("/api/analyze")

    assert invalid_timezone.status_code == 400
    assert invalid_timezone.json()["detail"]["code"] == "invalid_timezone"
    assert invalid_range.status_code == 400
    assert invalid_range.json()["detail"]["code"] == "invalid_date_range"
    assert analysis.json() == {
        "summary": [],
        "ab_leaderboard": [],
        "trends": [],
        "coaching_markdown": "",
    }
