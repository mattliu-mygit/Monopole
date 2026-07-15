from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from weave_agent_signals.judges.review import (
    AttemptObservation,
    ReviewOutcome,
    ReviewPolicy,
    execute_review,
)
from weave_agent_signals.run_config import PositionedJudge


def _judge(position: int) -> PositionedJudge:
    return PositionedJudge(
        id=f"judge-{position}",
        label=f"Judge {position}",
        family=f"family-{position}",
        backend="cli",
        supported_roles=("judge",),
        position=position,
    )


def _policy(
    depth: str,
    judge_count: int,
    *,
    margin: float | None,
) -> ReviewPolicy:
    return ReviewPolicy(
        depth=depth,
        judges=tuple(_judge(position) for position in range(1, judge_count + 1)),
        second_opinion_margin=margin,
    )


def _success(score: float, *, model: str = "resolved-model") -> AttemptObservation:
    return AttemptObservation(
        status="succeeded",
        resolved_model=model,
        score=score,
        rationale=f"score={score}",
        usage={"total_tokens": 10},
        error_type=None,
        message=None,
    )


def _failure(message: str = "judge unavailable") -> AttemptObservation:
    return AttemptObservation(
        status="failed",
        resolved_model=None,
        score=None,
        rationale=None,
        usage={},
        error_type="RuntimeError",
        message=message,
    )


def _abstention(message: str = "insufficient evidence") -> AttemptObservation:
    return AttemptObservation(
        status="abstained",
        resolved_model="resolved-model",
        score=None,
        rationale=message,
        usage={"total_tokens": 8},
        error_type=None,
        message=None,
    )


@dataclass
class _ScriptedInvoke:
    observations: Sequence[AttemptObservation]

    def __post_init__(self) -> None:
        self.calls: list[PositionedJudge] = []

    def __call__(self, judge: PositionedJudge) -> AttemptObservation:
        observation = self.observations[len(self.calls)]
        self.calls.append(judge)
        return observation


@pytest.mark.parametrize(
    (
        "case",
        "policy",
        "observations",
        "expected_triggers",
        "expected_rating",
        "expected_status",
        "expected_successful_count",
    ),
    [
        (
            "primary",
            _policy("primary", 1, margin=None),
            (_success(0.8),),
            ("initial",),
            0.8,
            "complete",
            1,
        ),
        (
            "selective_far_from_boundary",
            _policy("selective", 3, margin=0.1),
            (_success(0.8),),
            ("initial",),
            0.8,
            "complete",
            1,
        ),
        (
            "inclusive_margin",
            _policy("selective", 2, margin=0.1),
            (_success(0.6), _success(0.8)),
            ("initial", "near_boundary"),
            0.7,
            "complete",
            2,
        ),
        (
            "judge_1_failed",
            _policy("selective", 2, margin=0.1),
            (_failure(), _success(0.8)),
            ("initial", "judge_1_failed"),
            0.8,
            "degraded",
            1,
        ),
        (
            "threshold_disagreement",
            _policy("selective", 3, margin=0.1),
            (_success(0.6), _success(0.4), _success(0.8)),
            ("initial", "near_boundary", "threshold_disagreement"),
            0.6,
            "complete",
            3,
        ),
        (
            "only_success_near_boundary_after_judge_1_failure",
            _policy("selective", 3, margin=0.1),
            (_failure(), _success(0.55), _success(0.75)),
            ("initial", "judge_1_failed", "only_success_near_boundary"),
            0.65,
            "degraded",
            2,
        ),
        (
            "only_success_near_boundary_after_judge_2_failure",
            _policy("selective", 3, margin=0.1),
            (_success(0.55), _failure(), _success(0.75)),
            ("initial", "near_boundary", "only_success_near_boundary"),
            0.65,
            "degraded",
            2,
        ),
        (
            "both_prior_failed",
            _policy("selective", 3, margin=0.1),
            (_failure("first failed"), _failure("second failed"), _success(0.7)),
            ("initial", "judge_1_failed", "both_prior_failed"),
            0.7,
            "degraded",
            1,
        ),
        (
            "full_panel",
            _policy("full_panel", 3, margin=None),
            (_success(0.2), _success(0.4), _success(0.6)),
            ("initial", "full_panel", "full_panel"),
            0.4,
            "complete",
            3,
        ),
        (
            "full_panel_with_resolved_failure",
            _policy("full_panel", 3, margin=None),
            (_success(0.6), _failure(), _success(0.8)),
            ("initial", "full_panel", "full_panel"),
            0.7,
            "degraded",
            2,
        ),
        (
            "zero_success",
            _policy("primary", 1, margin=None),
            (_failure(),),
            ("initial",),
            None,
            "failed",
            0,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_review_policy_decision_table(
    case: str,
    policy: ReviewPolicy,
    observations: Sequence[AttemptObservation],
    expected_triggers: tuple[str, ...],
    expected_rating: float | None,
    expected_status: str,
    expected_successful_count: int,
) -> None:
    invoke = _ScriptedInvoke(observations)

    outcome = execute_review(policy, threshold=0.5, invoke=invoke)

    assert isinstance(outcome, ReviewOutcome), case
    assert tuple(attempt.trigger for attempt in outcome.attempts) == expected_triggers
    assert [judge.position for judge in invoke.calls] == list(range(1, len(expected_triggers) + 1))
    if expected_rating is None:
        assert outcome.rating is None
    else:
        assert outcome.rating == pytest.approx(expected_rating)
    assert outcome.status == expected_status
    assert outcome.successful_count == expected_successful_count
    assert tuple(attempt.position for attempt in outcome.attempts) == tuple(
        range(1, len(expected_triggers) + 1)
    )
    assert all(attempt.role == "judge" for attempt in outcome.attempts)
    assert tuple(attempt.requested_model for attempt in outcome.attempts) == tuple(
        f"judge-{position}" for position in range(1, len(expected_triggers) + 1)
    )
    assert tuple(attempt.requested_family for attempt in outcome.attempts) == tuple(
        f"family-{position}" for position in range(1, len(expected_triggers) + 1)
    )
    assert all(attempt.requested_backend == "cli" for attempt in outcome.attempts)
    assert tuple(attempt.observation for attempt in outcome.attempts) == tuple(
        observations[: len(expected_triggers)]
    )


def test_score_equal_to_threshold_is_passing_and_zero_margin_is_inclusive() -> None:
    invoke = _ScriptedInvoke((_success(0.5), _success(0.49)))

    outcome = execute_review(
        _policy("selective", 2, margin=0.0),
        threshold=0.5,
        invoke=invoke,
    )

    assert tuple(attempt.trigger for attempt in outcome.attempts) == (
        "initial",
        "near_boundary",
    )
    assert outcome.rating == pytest.approx(0.495)
    assert outcome.status == "unresolved"


def test_failed_third_opinion_leaves_two_success_threshold_split_unresolved() -> None:
    invoke = _ScriptedInvoke((_success(0.6), _success(0.4), _failure("third failed")))

    outcome = execute_review(
        _policy("selective", 3, margin=0.1),
        threshold=0.5,
        invoke=invoke,
    )

    assert tuple(attempt.trigger for attempt in outcome.attempts) == (
        "initial",
        "near_boundary",
        "threshold_disagreement",
    )
    assert outcome.rating == pytest.approx(0.5)
    assert outcome.status == "unresolved"
    assert outcome.successful_count == 2


def test_scored_verdict_after_fallback_is_degraded() -> None:
    invoke = _ScriptedInvoke((_failure("invalid verdict"), _success(0.75)))

    outcome = execute_review(
        _policy("selective", 2, margin=0.1),
        threshold=0.5,
        invoke=invoke,
    )

    assert tuple(attempt.trigger for attempt in outcome.attempts) == (
        "initial",
        "judge_1_failed",
    )
    assert outcome.rating == 0.75
    assert outcome.status == "degraded"
    assert outcome.successful_count == 1


def test_full_panel_aggregates_only_scored_verdicts() -> None:
    invoke = _ScriptedInvoke((_success(0.25), _abstention(), _success(0.75)))

    outcome = execute_review(
        _policy("full_panel", 3, margin=None),
        threshold=0.5,
        invoke=invoke,
    )

    assert [judge.position for judge in invoke.calls] == [1, 2, 3]
    assert [attempt.observation.status for attempt in outcome.attempts] == [
        "succeeded",
        "abstained",
        "succeeded",
    ]
    assert outcome.rating == 0.5
    assert outcome.status == "unresolved"
    assert outcome.successful_count == 2


def test_selective_abstained_then_failed_advances_to_third_scored_verdict() -> None:
    invoke = _ScriptedInvoke((_abstention(), _failure("second judge failed"), _success(0.75)))

    outcome = execute_review(
        _policy("selective", 3, margin=0.1),
        threshold=0.5,
        invoke=invoke,
    )

    assert [judge.position for judge in invoke.calls] == [1, 2, 3]
    assert tuple(attempt.trigger for attempt in outcome.attempts) == (
        "initial",
        "judge_1_abstained",
        "both_prior_unsuccessful",
    )
    assert outcome.rating == 0.75
    assert outcome.status == "degraded"
    assert outcome.successful_count == 1


def test_execute_review_does_not_convert_invoke_exceptions() -> None:
    expected = RuntimeError("transport failure")

    def invoke(_judge: PositionedJudge) -> AttemptObservation:
        raise expected

    with pytest.raises(RuntimeError) as caught:
        execute_review(
            _policy("primary", 1, margin=None),
            threshold=0.5,
            invoke=invoke,
        )

    assert caught.value is expected


def test_review_policy_is_a_frozen_dataclass() -> None:
    instance = _policy("primary", 1, margin=None)

    with pytest.raises(AttributeError):
        instance.depth = "selective"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("depth", "judge_count", "margin", "message"),
    [
        ("primary", 2, None, "primary review requires exactly 1 judge"),
        ("selective", 1, 0.1, "selective review requires 2 or 3 judges"),
        ("full_panel", 2, None, "full_panel review requires exactly 3 judges"),
        ("selective", 2, None, "selective review requires second_opinion_margin"),
        ("selective", 2, -0.01, "second_opinion_margin must be between 0 and 0.5"),
        ("primary", 1, 0.1, "second_opinion_margin must be null"),
    ],
)
def test_review_policy_rejects_invalid_shape(
    depth: str,
    judge_count: int,
    margin: float | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _policy(depth, judge_count, margin=margin)


def test_review_policy_requires_judges_in_contiguous_position_order() -> None:
    with pytest.raises(ValueError, match="judge positions must be contiguous"):
        ReviewPolicy(
            depth="selective",
            judges=(_judge(2), _judge(1)),
            second_opinion_margin=0.1,
        )


@pytest.mark.parametrize(
    "threshold",
    [True, -0.01, 1.01],
)
def test_execute_review_rejects_invalid_threshold(threshold: object) -> None:
    with pytest.raises(ValueError, match="threshold must be numeric between 0 and 1"):
        execute_review(
            _policy("primary", 1, margin=None),
            threshold=threshold,  # type: ignore[arg-type]
            invoke=_ScriptedInvoke((_success(0.5),)),
        )


@pytest.mark.parametrize(
    "observation",
    [
        {
            "status": "succeeded",
            "resolved_model": "resolved-model",
            "score": None,
            "rationale": None,
            "usage": {},
            "error_type": None,
            "message": None,
        },
        {
            "status": "succeeded",
            "resolved_model": "resolved-model",
            "score": 1.01,
            "rationale": None,
            "usage": {},
            "error_type": None,
            "message": None,
        },
        {
            "status": "failed",
            "resolved_model": None,
            "score": 0.5,
            "rationale": None,
            "usage": {},
            "error_type": "RuntimeError",
            "message": "failed",
        },
        {
            "status": "failed",
            "resolved_model": None,
            "score": None,
            "rationale": None,
            "usage": {},
            "error_type": None,
            "message": "failed",
        },
        {
            "status": "succeeded",
            "resolved_model": "resolved-model",
            "score": 0.5,
            "rationale": "valid verdict",
            "usage": {},
            "error_type": None,
            "message": None,
            "raw_output_digest": "raw output is not a SHA-256 digest",
        },
    ],
)
def test_attempt_observation_rejects_invalid_shape(observation: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AttemptObservation(**observation)  # type: ignore[arg-type]


def test_attempt_observation_freezes_validated_usage() -> None:
    usage = {"input_tokens": 4}

    observation = AttemptObservation(
        status="succeeded",
        resolved_model="resolved-model",
        score=0.5,
        rationale=None,
        usage=usage,
        error_type=None,
        message=None,
    )
    usage["input_tokens"] = 8

    assert observation.usage == {"input_tokens": 4}
    with pytest.raises(TypeError):
        observation.usage["input_tokens"] = 8  # type: ignore[index]
