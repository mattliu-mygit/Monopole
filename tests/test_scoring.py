from __future__ import annotations

from datetime import datetime, timezone

from weave_agent_signals.models import SessionView, SubagentSpan, ToolSpan, TurnSpan
from weave_agent_signals.scorers import score_session, score_turn


def _ts(h=12, m=0):
    return datetime(2026, 7, 9, h, m, tzinfo=timezone.utc)


def _bash(cmd, result="", status="OK"):
    return ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments=f'{{"command": "{cmd}"}}',
        result=result,
        status_code=status,
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )


def _turn(tool_calls, steering=0, denials=0, status="OK"):
    return TurnSpan(
        trace_id="t1",
        conversation_id="c1",
        started_at=_ts(),
        ended_at=_ts(12, 5),
        model="claude-opus-4",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        status_code=status,
        config_version="abc",
        git_branch="main",
        effort_level="high",
        session_id="s1",
        steering_count=steering,
        denial_count=denials,
        tool_error_count=0,
        events=[],
        tool_calls=tool_calls,
        chat_spans=[],
        subagents=[],
    )


def _session(turns):
    return SessionView(
        conversation_id="c1",
        turns=turns,
        config_version="abc",
        git_branch="main",
    )


def test_score_turn_returns_all_applicable():
    tc1 = _bash("pytest tests/", result="===== 5 passed in 1.0s =====")
    tc2 = _bash("git commit -m 'fix'", result=" 1 file changed, 3 insertions(+)")
    scores = score_turn(_turn([tc1, tc2]))
    scorer_names = {s.scorer for s in scores}
    assert "outcome.test" in scorer_names
    assert "outcome.git" in scorer_names
    assert "efficiency" in scorer_names


def test_score_turn_no_tools():
    scores = score_turn(_turn([]))
    assert len(scores) == 1
    assert scores[0].scorer == "efficiency"


def test_score_session_returns_all():
    t1 = _turn([_bash("pytest", result="===== 3 passed in 0.5s =====")], steering=1)
    t2 = _turn([_bash("ruff check .", result="All checks passed!")])
    scores = score_session(_session([t1, t2]))
    scorer_names = {s.scorer for s in scores}
    assert "implicit.correction_density" in scorer_names
    assert "implicit.abandonment" in scorer_names
    assert "efficiency.session" in scorer_names
    assert all(s.granularity == "session" for s in scores)


def test_score_session_empty():
    scores = score_session(_session([]))
    assert len(scores) == 3


def test_score_turn_includes_subagent_outcomes():
    """Test runs in subagents should be scored at the turn level."""
    sub_tool = _bash("pytest tests/", result="===== 10 passed in 2.0s =====")
    subagent = SubagentSpan(span_id="sub-1", agent_type="general-purpose", tool_calls=[sub_tool])
    turn = _turn([])
    turn.subagents = [subagent]
    scores = score_turn(turn)
    scorer_names = {s.scorer for s in scores}
    assert "outcome.test" in scorer_names
