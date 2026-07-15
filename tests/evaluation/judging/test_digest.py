from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weave_agent_signals.judges.digest import (
    JudgeDigest,
    build_judge_messages,
    build_session_digest,
    build_turn_digest,
    build_turn_digest_with_context,
)
from weave_agent_signals.models import SessionView, ToolSpan, TurnSpan


def _ts(h=12, m=0):
    return datetime(2026, 7, 9, h, m, tzinfo=timezone.utc)


def _turn(
    trace_id="t1",
    tool_calls=None,
    steering=0,
    denials=0,
    errors=0,
    user_input=None,
    assistant_output=None,
):
    return TurnSpan(
        trace_id=trace_id,
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
        user_input=user_input,
        assistant_output=assistant_output,
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
    assert "Turn ID:" in digest.text
    assert "claude-opus-4" not in digest.text
    assert "Tokens:" in digest.text


def test_digest_includes_bounded_request_and_assistant_output():
    turn = _turn(
        user_input="Please update the evaluator. " + "u" * 2_000 + "USER_TAIL",
        assistant_output="Implemented and verified. " + "a" * 3_000 + "ASSISTANT_TAIL",
    )

    digest = build_turn_digest(turn)

    assert "User request:" in digest.text
    assert "Please update the evaluator." in digest.text
    assert "Assistant output:" in digest.text
    assert "Implemented and verified." in digest.text
    assert "chars omitted" in digest.text
    assert "USER_TAIL" not in digest.text
    assert "ASSISTANT_TAIL" not in digest.text


def test_digest_includes_tool_calls():
    turn = _turn(tool_calls=[_bash("pytest tests/", "5 passed")])
    digest = build_turn_digest(turn)
    assert "Tool calls (1):" in digest.text
    assert "Bash" in digest.text
    assert "pytest" in digest.text
    assert "5 passed" in digest.text


def test_digest_includes_events():
    turn = _turn(steering=2, denials=1, errors=3)
    digest = build_turn_digest(turn)
    assert "steering=2" in digest.text
    assert "denials=1" in digest.text
    assert "tool_errors=3" in digest.text


def test_digest_truncates_long_results():
    long_result = "x" * 1000
    turn = _turn(tool_calls=[_bash("cat file.py", long_result)])
    digest = build_turn_digest(turn)
    assert "chars total" in digest.text
    assert len(digest.text) < 2000


def test_digest_no_events_when_zero():
    turn = _turn()
    digest = build_turn_digest(turn)
    assert "Events:" not in digest.text


def test_session_digest_is_hierarchical_and_tail_aware():
    turns = [
        _turn(
            f"trace-{index}",
            user_input=f"request-{index}",
            assistant_output=f"result-{index}",
        )
        for index in range(6)
    ]
    session = SessionView(
        conversation_id="session-1",
        turns=turns,
        config_version="abc",
        git_branch="main",
    )

    digest = build_session_digest(session, max_turns=4)

    assert "## Session overview" in digest.text
    assert "## Evidence selection" in digest.text
    assert "## Selected turn evidence" in digest.text
    assert "Selected evidence IDs: trace-0, trace-1, trace-4, trace-5" in digest.text
    assert "2 turns omitted" in digest.text
    assert "positions 3-4" in digest.text
    assert "result-0" in digest.text
    assert "result-5" in digest.text
    assert "result-2" not in digest.text
    assert "result-3" not in digest.text
    assert "claude-opus-4" not in digest.text


def test_session_digest_prioritizes_requested_evidence_ids():
    turns = [
        _turn(
            f"trace-{index}",
            user_input=f"request-{index}",
            assistant_output=f"result-{index}",
        )
        for index in range(8)
    ]
    session = SessionView("session-1", turns, "abc", "main")

    with pytest.raises(ValueError, match="missing requested evidence IDs: missing-trace"):
        build_session_digest(
            session,
            max_turns=4,
            evidence_trace_ids=["trace-3", "missing-trace", "trace-6"],
        )


def test_prior_turn_summary_keeps_head_and_tail_tools_with_omission_count():
    prior = _turn(
        "prior",
        tool_calls=[
            _bash("inspect-first", "first"),
            _bash("middle-one", "middle"),
            _bash("middle-two", "middle"),
            _bash("apply_patch-last", "updated"),
        ],
    )

    digest = build_turn_digest_with_context(_turn("current"), [prior], window=1)

    assert "inspect-first" in digest.text
    assert "apply_patch-last" in digest.text
    assert "middle-one" not in digest.text
    assert "middle-two" not in digest.text
    assert "2 tool calls omitted" in digest.text


def test_digest_returns_exact_visible_evidence_ids():
    turn_digest = build_turn_digest(_turn("trace-current"))
    assert isinstance(turn_digest, JudgeDigest)
    assert turn_digest.evidence_ids == ("trace-current",)
    assert "evidence_id=trace-current" in turn_digest.text

    context_digest = build_turn_digest_with_context(
        _turn("trace-current"),
        [_turn("trace-prior-1"), _turn("trace-prior-2")],
        window=1,
    )
    assert context_digest.evidence_ids == ("trace-prior-2", "trace-current")
    for evidence_id in context_digest.evidence_ids:
        assert context_digest.text.count(f"evidence_id={evidence_id}") == 1

    session = SessionView(
        "session-1",
        [_turn(f"trace-{index}") for index in range(6)],
        "abc",
        "main",
    )
    session_digest = build_session_digest(session, max_turns=4)
    assert session_digest.evidence_ids == ("trace-0", "trace-1", "trace-4", "trace-5")
    for evidence_id in session_digest.evidence_ids:
        assert f"evidence_id={evidence_id}" in session_digest.text

    limited_session = SessionView(
        "session-2",
        [
            _turn("trace-initial", user_input="original request"),
            _turn("trace-terminal", user_input="follow-up"),
        ],
        "abc",
        "main",
    )
    limited_digest = build_session_digest(limited_session, max_turns=1)
    assert limited_digest.evidence_ids == ("trace-initial", "trace-terminal")
    assert "Initial request [evidence_id=trace-initial]: original request" in limited_digest.text


def test_judge_messages_require_citations_and_allow_abstention():
    messages = build_judge_messages(
        rubric_system="You are a judge.",
        rubric_criteria="- 1.0: Perfect",
        digest=JudgeDigest(
            text="Turn 1 [evidence_id=trace-1]: Turn data here",
            evidence_ids=("trace-1",),
        ),
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "You are a judge." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Turn data here" in messages[1]["content"]
    assert "Scoring criteria" in messages[1]["content"]
    assert '"schema_version": 3' in messages[1]["content"]
    assert '"observations"' in messages[1]["content"]
    assert '"status": "scored"' in messages[1]["content"]
    assert '"id": "trace-1"' in messages[1]["content"]
    assert (
        "Every scored verdict must cite at least one allowed evidence ID" in messages[1]["content"]
    )
    assert '"status": "insufficient_evidence"' in messages[1]["content"]
    assert '"score": null' in messages[1]["content"]
    assert '"evidence": []' in messages[1]["content"]
