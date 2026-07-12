from __future__ import annotations

from datetime import datetime, timezone

from weave_agent_signals.judges.digest import build_turn_digest, build_judge_messages
from weave_agent_signals.models import ToolSpan, TurnSpan


def _ts(h=12, m=0):
    return datetime(2026, 7, 9, h, m, tzinfo=timezone.utc)


def _turn(tool_calls=None, steering=0, denials=0, errors=0):
    return TurnSpan(
        trace_id="t1",
        conversation_id="c1",
        started_at=_ts(),
        ended_at=_ts(12, 5),
        model="claude-opus-4",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        status_code="OK",
        config_version="abc",
        git_branch="main",
        effort_level="high",
        session_id="s1",
        steering_count=steering,
        denial_count=denials,
        tool_error_count=errors,
        events=[],
        tool_calls=tool_calls or [],
        chat_spans=[],
        subagents=[],
    )


def _bash(cmd, result=""):
    return ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments=f'{{"command": "{cmd}"}}',
        result=result,
        status_code="OK",
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )


def test_digest_basic_structure():
    turn = _turn()
    digest = build_turn_digest(turn)
    assert "Turn ID:" in digest
    assert "claude-opus-4" in digest
    assert "Tokens:" in digest


def test_digest_includes_tool_calls():
    turn = _turn(tool_calls=[_bash("pytest tests/", "5 passed")])
    digest = build_turn_digest(turn)
    assert "Tool calls (1):" in digest
    assert "Bash" in digest
    assert "pytest" in digest
    assert "5 passed" in digest


def test_digest_includes_events():
    turn = _turn(steering=2, denials=1, errors=3)
    digest = build_turn_digest(turn)
    assert "steering=2" in digest
    assert "denials=1" in digest
    assert "tool_errors=3" in digest


def test_digest_truncates_long_results():
    long_result = "x" * 1000
    turn = _turn(tool_calls=[_bash("cat file.py", long_result)])
    digest = build_turn_digest(turn)
    assert "chars total" in digest
    assert len(digest) < 2000


def test_digest_no_events_when_zero():
    turn = _turn()
    digest = build_turn_digest(turn)
    assert "Events:" not in digest


def test_build_judge_messages():
    messages = build_judge_messages(
        rubric_system="You are a judge.",
        rubric_criteria="- 1.0: Perfect",
        digest="Turn data here",
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "You are a judge." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Turn data here" in messages[1]["content"]
    assert "Scoring criteria" in messages[1]["content"]
    assert '{"score":' in messages[1]["content"]
