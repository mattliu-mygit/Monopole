from __future__ import annotations

from dataclasses import replace
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


def _ts() -> datetime:
    return datetime(2026, 7, 9, 12, tzinfo=timezone.utc)


def _turn(trace_id: str, *, user_input: str, assistant_output: str) -> TurnSpan:
    return TurnSpan(
        trace_id=trace_id,
        conversation_id="session-1",
        started_at=_ts(),
        ended_at=_ts(),
        model="model-that-must-not-be-visible",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        status_code="OK",
        config_version="config-1",
        git_branch="main",
        effort_level="high",
        session_id="session-id-1",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=[
            ToolSpan(
                span_id=f"tool-{trace_id}",
                tool_name="Bash",
                arguments='{"command":"run everything"}',
                result="complete result",
                status_code="OK",
                started_at=_ts(),
                ended_at=_ts(),
            )
        ],
        chat_spans=[],
        subagents=[],
        user_input=user_input,
        assistant_output=assistant_output,
    )


def test_session_digest_compatibility_wrapper_renders_all_raw_evidence():
    session = SessionView(
        "session-1",
        [
            _turn("trace-1", user_input="u" * 2_000, assistant_output="a" * 3_000),
            _turn("trace-2", user_input="second request", assistant_output="second response"),
        ],
        "config-1",
        "main",
    )

    digest = build_session_digest(session, max_turns=1, evidence_trace_ids=["trace-2"])

    assert digest.evidence_ids == (
        "trace-1",
        "tool-trace-1",
        "trace-2",
        "tool-trace-2",
    )
    assert "u" * 2_000 in digest.text
    assert "a" * 3_000 in digest.text
    assert "second request" in digest.text
    assert "second response" in digest.text
    assert "run everything" in digest.text
    assert "complete result" in digest.text
    assert "chars omitted" not in digest.text
    assert "model-that-must-not-be-visible" not in digest.text


def test_turn_digest_allows_every_visible_span_citation():
    digest = build_turn_digest(_turn("trace-1", user_input="request", assistant_output="response"))

    assert digest.evidence_ids == ("trace-1", "tool-trace-1")


def test_context_digest_compatibility_wrapper_never_truncates_prior_evidence():
    prior_turns = [
        _turn(f"trace-{index}", user_input=f"request-{index}", assistant_output="response")
        for index in range(1, 4)
    ]
    current = _turn("trace-4", user_input="current", assistant_output="response")

    digest = build_turn_digest_with_context(current, prior_turns, window=1)

    for trace_id in ("trace-1", "trace-2", "trace-3", "trace-4"):
        assert trace_id in digest.evidence_ids
        assert f"evidence_id={trace_id}" in digest.text


def test_turn_digest_rejects_blank_citation_ids():
    turn = _turn("trace-1", user_input="request", assistant_output="response")
    turn = replace(turn, tool_calls=[replace(turn.tool_calls[0], span_id=" ")])

    with pytest.raises(ValueError, match="evidence IDs must be nonblank"):
        build_turn_digest(turn)


def test_turn_digest_rejects_duplicate_citation_ids():
    turn = _turn("trace-1", user_input="request", assistant_output="response")
    turn = replace(turn, tool_calls=[replace(turn.tool_calls[0], span_id="trace-1")])

    with pytest.raises(ValueError, match="evidence IDs must be globally unique: trace-1"):
        build_turn_digest(turn)


def test_context_digest_rejects_blank_citation_ids():
    prior = _turn("trace-1", user_input="prior", assistant_output="response")
    prior = replace(prior, tool_calls=[replace(prior.tool_calls[0], span_id="")])
    current = _turn("trace-2", user_input="current", assistant_output="response")

    with pytest.raises(ValueError, match="evidence IDs must be nonblank"):
        build_turn_digest_with_context(current, [prior])


def test_context_digest_rejects_duplicate_citation_ids():
    prior = _turn("trace-1", user_input="prior", assistant_output="response")
    current = _turn("trace-2", user_input="current", assistant_output="response")
    current = replace(current, tool_calls=[replace(current.tool_calls[0], span_id="trace-1")])

    with pytest.raises(ValueError, match="evidence IDs must be globally unique: trace-1"):
        build_turn_digest_with_context(current, [prior])


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
