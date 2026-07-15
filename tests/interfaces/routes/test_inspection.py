from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from weave_agent_signals.models import SessionView, TurnSpan
from weave_agent_signals.routes.inspection import create_inspection_router


def _turn(
    trace_id: str,
    conversation_id: str,
    *,
    hour: int,
    user_input: str,
) -> TurnSpan:
    started = datetime(2026, 7, 14, hour, tzinfo=timezone.utc)
    return TurnSpan(
        trace_id=trace_id,
        conversation_id=conversation_id,
        started_at=started,
        ended_at=started + timedelta(minutes=1),
        model="gpt-5.6-sol",
        input_tokens=10,
        output_tokens=5,
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
            ),
        ]
        self.query_calls = []
        self.hydrated = []
        self.feedback_batches = []

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

    def query_all_feedback_batch(self, refs: list[str]):
        self.feedback_batches.append(refs)
        return {ref: [{"id": "feedback-1", "weave_ref": ref}] for ref in refs}

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
    assert backend.query_calls == [
        {
            "page_size": 500,
            "since": datetime(2026, 7, 14, 12, tzinfo=timezone.utc),
            "include_details": True,
        }
    ]


def test_session_detail_hydrates_children_and_returns_feedback():
    client, backend = _client()

    response = client.get("/api/sessions/session%2Fone")

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "session/one"
    assert body["turns"][0]["trace_id"] == "trace-a"
    assert body["session_feedback"][0]["id"] == "feedback-1"
    assert body["turn_feedback"]["trace-a"][0]["id"] == "feedback-1"
    assert backend.hydrated == [["trace-a"]]
    assert len(backend.feedback_batches) == 1
    assert len(backend.feedback_batches[0]) == 2


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
