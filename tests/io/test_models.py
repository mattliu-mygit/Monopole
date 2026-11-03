from __future__ import annotations

from datetime import datetime, timezone

from weave_agent_signals.models import (
    SessionView,
    ToolSpan,
    TraceRole,
    TurnSpan,
    session_is_evaluable,
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


def test_session_total_tokens():
    t1 = _turn(input_tokens=100, output_tokens=50)
    t2 = _turn(input_tokens=200, output_tokens=80)
    s = SessionView(conversation_id="c1", turns=[t1, t2], config_version=None, git_branch=None)
    assert s.total_tokens == 430


def test_session_is_evaluable_requires_every_turn_to_be_an_agent_session():
    assert session_is_evaluable([_turn(), _turn(trace_id="t2")]) is True
    assert (
        session_is_evaluable(
            [_turn(), _turn(trace_id="t2", trace_role=TraceRole.SIGNAL_EVALUATION)]
        )
        is False
    )
    assert session_is_evaluable([]) is False
