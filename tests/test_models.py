from __future__ import annotations

from datetime import datetime, timezone

from weave_agent_signals.models import (
    ChatSpan,
    Score,
    SessionView,
    ToolSpan,
    TurnSpan,
)


def _ts(h=12, m=0):
    return datetime(2026, 7, 9, h, m, tzinfo=timezone.utc)


def _tool(name="Bash", status="OK", args="", result=""):
    return ToolSpan(
        span_id=f"sp-{name}",
        tool_name=name,
        arguments=args,
        result=result,
        status_code=status,
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )


def _turn(trace_id="t1", conv_id="c1", **kw):
    defaults = dict(
        trace_id=trace_id,
        conversation_id=conv_id,
        started_at=_ts(),
        ended_at=_ts(12, 5),
        model="claude-opus-4",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        status_code="OK",
        config_version="abc123",
        git_branch="main",
        effort_level="high",
        session_id="sess-1",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
    )
    defaults.update(kw)
    return TurnSpan(**defaults)


def test_turn_ref():
    t = _turn(trace_id="abc123")
    assert t.ref == "weave:///mliu-wandb-weights-biases/agent-sessions/agent_turn/abc123"


def test_session_ref():
    s = SessionView(
        conversation_id="conv-42",
        turns=[_turn()],
        config_version="abc123",
        git_branch="main",
    )
    assert "agent_conversation/conv-42" in s.ref


def test_session_total_tokens():
    t1 = _turn(input_tokens=100, output_tokens=50)
    t2 = _turn(input_tokens=200, output_tokens=80)
    s = SessionView(conversation_id="c1", turns=[t1, t2], config_version=None, git_branch=None)
    assert s.total_tokens == 430


def test_score_to_feedback_payload():
    score = Score(
        scorer="outcome.test",
        value=1.0,
        tags=["test_pass"],
        confidence=1.0,
        metadata={"passed": 42, "failed": 0},
        granularity="turn",
    )
    payload = score.to_feedback_payload(
        ref="weave:///mliu-wandb-weights-biases/agent-sessions/agent_turn/t1",
        project_id="mliu-wandb-weights-biases/agent-sessions",
    )
    assert payload["feedback_type"] == "weave_agent_signals.outcome.test"
    assert payload["scorer_ratings"]["_rating_"] == 1.0
    assert payload["scorer_rating_confidences"]["_rating_"] == 1.0
    assert "test_pass" in payload["scorer_tags"]
    assert payload["weave_ref"].endswith("/agent_turn/t1")


def test_score_to_feedback_bool_value():
    score = Score(
        scorer="outcome.verified_before_done",
        value=True,
        tags=["verified"],
        confidence=0.9,
        metadata={},
        granularity="session",
    )
    payload = score.to_feedback_payload(
        ref="weave:///test/proj/agent_conversation/c1",
        project_id="test/proj",
    )
    assert payload["scorer_ratings"]["_rating_"] == 1.0
