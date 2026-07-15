"""Tests for explicit, reviewed session judging."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.inference import JudgeResponse
from weave_agent_signals.judges.runner import JudgeExecutionError, judge_session
from weave_agent_signals.models import SessionView, TurnSpan
from weave_agent_signals.run_config import PositionedJudge, RubricDescriptor


def _turn(trace_id: str = "turn-1", *, model: str | None = "claude-opus-4") -> TurnSpan:
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    return TurnSpan(
        trace_id=trace_id,
        conversation_id="session-1",
        started_at=now,
        ended_at=now,
        model=model,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        status_code="OK",
        config_version="cfg",
        git_branch="main",
        effort_level="high",
        session_id="session-1",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
    )


def _session(turns: list[TurnSpan] | None = None) -> SessionView:
    return SessionView("session-1", turns or [_turn()], "cfg", "main")


def _rubric(rubric_id: str = "judge.session_outcome") -> RubricDescriptor:
    return build_rubric_catalog().rubric(rubric_id)


def _judge(
    model_id: str,
    position: int,
    *,
    family: str = "openai",
) -> PositionedJudge:
    return PositionedJudge(
        id=model_id,
        label=model_id,
        family=family,
        backend="cli",
        supported_roles=("judge",),
        position=position,
    )


def _scored_verdict(score: object, rationale: str = "reviewed") -> dict[str, object]:
    return {
        "schema_version": 3,
        "status": "scored",
        "score": score,
        "rationale": rationale,
        "evidence": [
            {
                "id": "turn-1",
                "observations": ["The selected turn supports the verdict."],
            }
        ],
    }


def _insufficient_verdict() -> dict[str, object]:
    return {
        "schema_version": 3,
        "status": "insufficient_evidence",
        "score": None,
        "rationale": "The selected turns do not establish the required state.",
        "evidence": [],
    }


class _Client:
    backend = "cli"

    def __init__(self, outcomes: dict[str, Sequence[object]]) -> None:
        self.outcomes = {model: list(values) for model, values in outcomes.items()}
        self.calls: list[dict[str, object]] = []

    def chat_json(self, *, model: str, messages: list[dict[str, str]], **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        outcome = self.outcomes[model].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        verdict = outcome if isinstance(outcome, dict) else _scored_verdict(*outcome)
        return (
            verdict,
            JudgeResponse(
                content=json.dumps(verdict),
                model=model,
                usage={"total_tokens": 8},
                output_mode="json_schema",
                schema_name="judge_verdict",
            ),
        )


def _judge_session(
    client: object,
    *,
    session: SessionView | None = None,
    rubrics: Sequence[RubricDescriptor] = (_rubric(),),
    judges: Sequence[PositionedJudge] = (_judge("judge-1", 1),),
    review_depth: str = "primary",
    second_opinion_margin: float | None = None,
    evidence_trace_ids: Sequence[str] = ("turn-1",),
):
    return judge_session(
        session or _session(),
        client,
        rubrics=rubrics,
        judges=judges,
        review_depth=review_depth,
        second_opinion_margin=second_opinion_margin,
        evidence_trace_ids=evidence_trace_ids,
    )


def test_session_uses_same_review_policy_and_exact_pinned_judges() -> None:
    judges = (
        _judge("claude-first", 1, family="anthropic"),
        _judge("claude-second", 2, family="anthropic"),
        _judge("claude-third", 3, family="anthropic"),
    )
    client = _Client(
        {
            "claude-first": [RuntimeError("first offline")],
            "claude-second": [(0.5, "near")],
            "claude-third": [(0.75, "third opinion")],
        }
    )

    score = _judge_session(
        client,
        judges=judges,
        review_depth="selective",
        second_opinion_margin=0.1,
    )[0]

    assert [call["model"] for call in client.calls] == [
        "claude-first",
        "claude-second",
        "claude-third",
    ]
    assert score.value == pytest.approx(0.625)
    assert score.granularity == "session"
    assert score.metadata["review_status"] == "degraded"
    assert [attempt["trigger"] for attempt in score.metadata["attempts"]] == [
        "initial",
        "judge_1_failed",
        "only_success_near_boundary",
    ]


def test_session_keeps_unresolved_rating_and_audit_metadata() -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2, family="meta"))
    client = _Client(
        {
            "judge-1": [(0.5, "equal passes")],
            "judge-2": [(0.25, "fails")],
        }
    )

    score = _judge_session(
        client,
        judges=judges,
        review_depth="selective",
        second_opinion_margin=0.0,
    )[0]

    assert score.value == pytest.approx(0.375)
    assert score.tags == ["unresolved"]
    assert score.metadata["review_status"] == "unresolved"
    assert score.metadata["evaluation_unit"] == "session"
    assert score.metadata["requested_judge_models"] == ["judge-1", "judge-2"]
    assert score.metadata["attempt_count"] == 2
    assert score.metadata["successful_reviewer_count"] == 2
    assert score.metadata["evaluated_models"] == ["claude-opus-4"]
    assert score.metadata["evaluated_families"] == ["anthropic"]


def test_session_allows_missing_or_same_family_evaluated_models() -> None:
    client = _Client({"claude-judge": [(0.75, "audited")]})
    session = _session([_turn(model=None)])

    score = _judge_session(
        client,
        session=session,
        judges=(_judge("claude-judge", 1, family="anthropic"),),
    )[0]

    assert [call["model"] for call in client.calls] == ["claude-judge"]
    assert score.metadata["evaluated_models"] == []
    assert score.metadata["evaluated_families"] == ["unknown"]


def test_session_zero_success_retains_all_attempts() -> None:
    judges = (
        _judge("judge-1", 1),
        _judge("judge-2", 2, family="meta"),
        _judge("judge-3", 3, family="ibm"),
    )
    client = _Client(
        {
            "judge-1": [RuntimeError("first offline")],
            "judge-2": [RuntimeError("second offline")],
            "judge-3": [RuntimeError("third offline")],
        }
    )

    with pytest.raises(JudgeExecutionError) as caught:
        _judge_session(
            client,
            judges=judges,
            review_depth="selective",
            second_opinion_margin=0.1,
        )

    assert caught.value.scores == ()
    assert len(caught.value.failures) == 1
    assert len(caught.value.failures[0].attempts) == 3


def test_selective_review_advances_after_insufficient_evidence() -> None:
    judges = (
        _judge("judge-1", 1),
        _judge("judge-2", 2, family="meta"),
    )
    client = _Client(
        {
            "judge-1": [_insufficient_verdict()],
            "judge-2": [_scored_verdict(0.75, "recovered")],
        }
    )

    score = _judge_session(
        client,
        judges=judges,
        review_depth="selective",
        second_opinion_margin=0.1,
    )[0]

    assert [call["model"] for call in client.calls] == ["judge-1", "judge-2"]
    assert [attempt["status"] for attempt in score.metadata["attempts"]] == [
        "abstained",
        "succeeded",
    ]
    assert [attempt["trigger"] for attempt in score.metadata["attempts"]] == [
        "initial",
        "judge_1_abstained",
    ]
    assert score.value == 0.75
    assert score.metadata["review_status"] == "degraded"


def test_missing_pinned_session_evidence_fails_before_inference() -> None:
    client = _Client({"judge-1": [(0.75, "must not run")]})

    with pytest.raises(ValueError, match="missing requested evidence IDs: missing-turn"):
        _judge_session(client, evidence_trace_ids=("missing-turn",))

    assert client.calls == []


def test_session_rejects_episode_rubric_before_inference() -> None:
    client = _Client({})

    with pytest.raises(ValueError, match="requires session rubrics"):
        _judge_session(client, rubrics=(_rubric("judge.verification"),))

    assert client.calls == []


def test_session_empty_rubrics_is_an_explicit_noop() -> None:
    assert _judge_session(object(), rubrics=()) == []
