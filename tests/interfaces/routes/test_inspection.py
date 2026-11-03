from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
        status_code="OK",
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
        case_ids=[trace_id],
    )


def _record(target: str, scorer: str, rating: float, reason: str = "") -> dict:
    return {
        "id": f"score-{target}-{scorer}",
        "feedback_type": f"weave_agent_signals.{scorer}",
        "target_id": target,
        "payload": {"rating": rating, "reason": reason, "scorer_version": "v1", "tags": []},
    }


class FakeCes:
    """A CesSession stand-in serving sessions and Score records from memory."""

    def __init__(self):
        self.turns = [
            _turn("trace-a", "session-one", hour=12, user_input="Build the feature"),
            _turn("trace-b", "session-two", hour=11, user_input="Fix the bug"),
            _turn(
                "trace-judge",
                "synthetic",
                hour=10,
                user_input="## Scoring criteria\nJudge this",
                trace_role=TraceRole.JUDGE_EVALUATION,
            ),
        ]
        self.feedback: list[dict] = []
        self.all_scores: list[dict] = []
        self.windows: list[tuple] = []
        self.feedback_calls: list[list[str]] = []
        self.client = self
        self.catalog = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def find_sessions(self, _catalog, *, since, until=None, limit=None):
        self.windows.append((since, until))
        in_window = [
            turn
            for turn in self.turns
            if turn.started_at >= since and (until is None or turn.started_at <= until)
        ]
        newest = sorted(in_window, key=lambda turn: turn.started_at, reverse=True)
        return list(dict.fromkeys(turn.conversation_id for turn in newest))

    def read_session(self, trajectory_id):
        turns = sorted(
            (turn for turn in self.turns if turn.conversation_id == trajectory_id),
            key=lambda turn: turn.started_at,
        )
        if not turns:
            request = httpx.Request("GET", "http://ces.test")
            raise httpx.HTTPStatusError("missing", request=request, response=httpx.Response(404))
        return SessionView(trajectory_id, turns, "config-v1", "main")

    def cohort_scores(self, _catalog, session_ids):
        self.feedback_calls.append(list(session_ids))
        targets = set(session_ids) | {
            turn.trace_id for turn in self.turns if turn.conversation_id in session_ids
        }
        return [record for record in self.feedback if record["target_id"] in targets]

    def read_all_scores(self, _catalog, limit=None):
        return list(self.all_scores)


def _client():
    backend = FakeCes()
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

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["total"], body["truncated"]) == (1, False)
    assert body["sessions"][0]["conversation_id"] == "session-one"
    assert "judging_token_estimates" not in body["sessions"][0]
    assert body["sessions"][0]["signal_evidence"] == []
    assert backend.windows == [
        (
            datetime(2026, 7, 14, 12, tzinfo=timezone.utc),
            datetime(2026, 7, 14, 13, tzinfo=timezone.utc),
        )
    ]
    assert backend.feedback_calls == [["session-one"]]


def test_session_listing_stops_at_the_limit_newest_first():
    client, backend = _client()

    body = client.get("/api/sessions", params={"since": "2026-07-14T00:00:00Z", "limit": 1}).json()

    assert [s["conversation_id"] for s in body["sessions"]] == ["session-one"]
    assert (body["total"], body["truncated"]) == (3, True)


def test_session_listing_totals_only_turns_inside_the_window():
    client, backend = _client()
    backend.turns.append(
        _turn(
            "trace-a2",
            "session-one",
            hour=13,
            user_input="Continue",
            input_tokens=80,
            output_tokens=20,
        )
    )
    backend.turns.append(
        _turn("trace-a0", "session-one", hour=9, user_input="Earlier", input_tokens=500)
    )

    response = client.get(
        "/api/sessions",
        params={"since": "2026-07-14T12:00:00Z", "until": "2026-07-14T14:00:00Z"},
    )

    summary = response.json()["sessions"][0]
    assert (summary["total_tokens"], summary["turn_count"]) == (115, 2)
    assert "judging_token_estimates" not in summary


def test_session_listing_excludes_mixed_sessions_even_when_the_judge_turn_is_outside_the_window():
    client, backend = _client()
    backend.turns.append(
        _turn(
            "trace-judge-2",
            "session-two",
            hour=1,
            user_input="judge",
            trace_role=TraceRole.SIGNAL_EVALUATION,
        )
    )

    body = client.get("/api/sessions", params={"since": "2026-07-14T09:00:00Z"}).json()

    assert [s["conversation_id"] for s in body["sessions"]] == ["session-one"]


@pytest.mark.parametrize("conversation_id", ["synthetic", "session-gone"])
def test_session_detail_rejects_missing_non_agent_and_mixed_sessions(conversation_id):
    client, _backend = _client()

    response = client.get(f"/api/sessions/{conversation_id}")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"


def test_low_turn_scores_with_a_reason_become_signal_evidence():
    client, backend = _client()
    backend.feedback = [
        _record("trace-a", "outcome.test", 0.0, "tests failed"),
        _record("trace-a", "efficiency", 0.25),
        _record("trace-a", "outcome.build", 1.0, "build passed"),
        _record("session-one", "implicit.completion", 0.0, "abandoned"),
    ]

    body = client.get("/api/sessions", params={"since": "2026-07-14T09:00:00Z"}).json()

    [summary] = [s for s in body["sessions"] if s["conversation_id"] == "session-one"]
    assert summary["signal_evidence"] == [
        {
            "signal": "outcome.test",
            "version": "v1",
            "rating": 0.0,
            "reason": "tests failed",
            "turn_id": "trace-a",
            "turn_started_at": "2026-07-14T12:00:00+00:00",
        }
    ]


def test_session_detail_returns_children_and_splits_session_and_turn_feedback():
    client, backend = _client()
    backend.turns[0].tool_calls.append(
        ToolSpan(
            span_id="tool-1",
            tool_name="Bash",
            arguments='{"command":"pytest"}',
            result="1 failed",
            status_code="ERROR",
            started_at=datetime(2026, 7, 14, 12, tzinfo=timezone.utc),
            ended_at=None,
        )
    )
    backend.feedback = [
        _record("trace-a", "outcome.test", 0.0, "tests failed"),
        _record("session-one", "efficiency.session", 1.0, "no waste"),
    ]

    response = client.get("/api/sessions/session-one")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["turns"][0]["tool_calls"][0]["tool_name"] == "Bash"
    assert [item["feedback_type"] for item in body["session_feedback"]] == [
        "weave_agent_signals.efficiency.session"
    ]
    assert list(body["turn_feedback"]) == ["trace-a"]
    assert [item["signal"] for item in body["signal_evidence"]] == ["outcome.test"]
    assert set(body["judging_token_estimates"])


def test_empty_analysis_is_stable():
    client, _backend = _client()

    response = client.get("/api/analyze")

    assert response.status_code == 200
    assert response.json() == {
        "summary": [],
        "ab_leaderboard": [],
        "trends": [],
        "coaching_markdown": "",
    }
