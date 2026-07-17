from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from weave_agent_signals.judges.review import (
    AttemptObservation,
    PanelOutcome,
    execute_panel,
)
from weave_agent_signals.run_config import PositionedJudge


def _judge(position: int) -> PositionedJudge:
    return PositionedJudge(
        id=f"judge-{position}",
        label=f"Judge {position}",
        provider="codex",
        provider_model=f"judge-{position}",
        family=f"family-{position}",
        supported_roles=("judge",),
        position=position,
    )


def _judges(count: int) -> tuple[PositionedJudge, ...]:
    return tuple(_judge(position) for position in range(1, count + 1))


def _success(score: float) -> AttemptObservation:
    return AttemptObservation(
        status="succeeded",
        resolved_model="resolved-model",
        score=score,
        rationale=f"score={score}",
        usage={"total_tokens": 10},
        error_type=None,
        message=None,
        behavioral_feedback={
            "success": "The response met the rubric.",
            "problem": None,
            "desired_behavior": None,
        },
    )


def _failure() -> AttemptObservation:
    return AttemptObservation(
        status="failed",
        resolved_model=None,
        score=None,
        rationale=None,
        usage={},
        error_type="RuntimeError",
        message="judge unavailable",
    )


def _abstention() -> AttemptObservation:
    return AttemptObservation(
        status="abstained",
        resolved_model="resolved-model",
        score=None,
        rationale="insufficient evidence",
        usage={"total_tokens": 8},
        error_type=None,
        message=None,
    )


def _skipped() -> AttemptObservation:
    return AttemptObservation(
        status="skipped",
        skip_reason="insufficient_context_capacity",
        resolved_model=None,
        score=None,
        rationale=None,
        usage={},
        error_type=None,
        message=None,
    )


@dataclass
class _ScriptedInvoke:
    observations: Sequence[AttemptObservation]

    def __post_init__(self) -> None:
        self.calls: list[PositionedJudge] = []
        self._lock = threading.Lock()

    def __call__(self, judge: PositionedJudge) -> AttemptObservation:
        with self._lock:
            self.calls.append(judge)
        return self.observations[judge.position - 1]


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((0.8,), (0.8, 0.8, 0.8, 0.0)),
        ((0.25, 0.75), (0.5, 0.25, 0.75, 0.5)),
        ((0.25, 0.5, 1.0), (pytest.approx(0.5833333333), 0.25, 1.0, 0.75)),
    ],
)
def test_panel_runs_every_judge_and_reports_disagreement(scores, expected) -> None:
    invoke = _ScriptedInvoke(tuple(_success(score) for score in scores))

    outcome = execute_panel(_judges(len(scores)), invoke, threshold=0.5)

    assert isinstance(outcome, PanelOutcome)
    assert sorted(judge.position for judge in invoke.calls) == list(range(1, len(scores) + 1))
    assert tuple(attempt.trigger for attempt in outcome.attempts) == ("panel",) * len(scores)
    assert (outcome.rating, outcome.minimum, outcome.maximum, outcome.spread) == expected
    assert outcome.status == "complete"
    assert outcome.successful_count == len(scores)


def test_failed_member_fails_panel_coverage() -> None:
    scripted = (_success(0.25), _failure(), _success(0.75))
    started = threading.Barrier(3, timeout=1)

    def invoke(judge: PositionedJudge) -> AttemptObservation:
        started.wait()
        return scripted[judge.position - 1]

    outcome = execute_panel(_judges(3), invoke, threshold=0.5)

    assert outcome.status == "failed"
    assert outcome.rating is None
    assert outcome.minimum is None
    assert outcome.maximum is None
    assert outcome.spread is None
    assert outcome.successful_count == 2
    assert [attempt.position for attempt in outcome.attempts] == [1, 2, 3]


def test_panel_judges_begin_concurrently() -> None:
    started = threading.Barrier(3, timeout=1)

    def invoke(judge: PositionedJudge) -> AttemptObservation:
        started.wait()
        return _success(judge.position / 4)

    outcome = execute_panel(_judges(3), invoke, threshold=0.5)

    assert outcome.status == "complete"
    assert [attempt.position for attempt in outcome.attempts] == [1, 2, 3]


def test_failed_member_cancels_outstanding_panel_work() -> None:
    started = threading.Barrier(2, timeout=1)
    release_sibling = threading.Event()
    cancellations = []

    def invoke(judge: PositionedJudge) -> AttemptObservation:
        started.wait()
        if judge.position == 1:
            return _failure()
        assert release_sibling.wait(timeout=1)
        return _success(0.75)

    def cancel_pending() -> None:
        cancellations.append(True)
        release_sibling.set()

    outcome = execute_panel(
        _judges(2),
        invoke,
        threshold=0.5,
        cancel_pending=cancel_pending,
    )

    assert cancellations == [True]
    assert outcome.status == "failed"


def test_multiple_scores_with_clean_abstention_are_degraded() -> None:
    invoke = _ScriptedInvoke((_success(0.25), _abstention(), _success(0.75)))

    outcome = execute_panel(_judges(3), invoke, threshold=0.5)

    assert outcome.status == "degraded"
    assert outcome.rating == 0.5
    assert outcome.minimum == 0.25
    assert outcome.maximum == 0.75
    assert outcome.spread == 0.5
    assert outcome.successful_count == 2


def test_one_score_with_clean_abstentions_is_degraded() -> None:
    invoke = _ScriptedInvoke((_success(0.75), _abstention(), _abstention()))

    outcome = execute_panel(_judges(3), invoke, threshold=0.5)

    assert outcome.status == "degraded"
    assert outcome.rating == 0.75
    assert outcome.successful_count == 1
    assert outcome.minimum == 0.75
    assert outcome.maximum == 0.75
    assert outcome.spread == 0.0


def test_unanimous_abstention_is_not_evaluable_instead_of_failed() -> None:
    invoke = _ScriptedInvoke((_abstention(), _abstention(), _abstention()))

    outcome = execute_panel(_judges(3), invoke, threshold=0.5)

    assert outcome.status == "not_evaluable"
    assert outcome.rating is None
    assert outcome.successful_count == 0
    assert len(outcome.attempts) == 3


def test_success_plus_skip_is_degraded_without_invoking_failure_semantics() -> None:
    outcome = execute_panel(
        _judges(2), _ScriptedInvoke((_skipped(), _success(0.75))), threshold=0.5
    )

    assert outcome.status == "degraded"
    assert outcome.rating == 0.75
    assert [attempt.observation.status for attempt in outcome.attempts] == [
        "skipped",
        "succeeded",
    ]


def test_panel_with_no_eligible_reviewers_is_not_evaluable() -> None:
    outcome = execute_panel(_judges(2), _ScriptedInvoke((_skipped(), _skipped())), threshold=0.5)

    assert outcome.status == "not_evaluable"
    assert outcome.rating is None
    assert outcome.successful_count == 0


def test_eligible_failure_still_fails_when_another_reviewer_is_skipped() -> None:
    outcome = execute_panel(_judges(2), _ScriptedInvoke((_skipped(), _failure())), threshold=0.5)

    assert outcome.status == "failed"
    assert outcome.rating is None


def test_skipped_observation_rejects_inference_fields() -> None:
    with pytest.raises(ValueError, match="skipped observations"):
        AttemptObservation(
            status="skipped",
            skip_reason="insufficient_context_capacity",
            resolved_model="must-not-exist",
            score=None,
            rationale=None,
            usage={},
            error_type=None,
            message=None,
        )


def test_execute_panel_does_not_convert_invoke_exceptions() -> None:
    expected = RuntimeError("transport failure")

    def invoke(_judge: PositionedJudge) -> AttemptObservation:
        raise expected

    with pytest.raises(RuntimeError) as caught:
        execute_panel(_judges(1), invoke, threshold=0.5)

    assert caught.value is expected


@pytest.mark.parametrize("judges", [(), _judges(3) + (_judge(4),)])
def test_panel_requires_one_through_three_judges(judges) -> None:
    with pytest.raises(ValueError, match="one through three"):
        execute_panel(judges, _ScriptedInvoke(()), threshold=0.5)


def test_panel_requires_contiguous_unique_order() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        execute_panel((_judge(2), _judge(1)), _ScriptedInvoke(()), threshold=0.5)
    with pytest.raises(ValueError, match="unique"):
        duplicate = _judge(2).model_copy(update={"id": "judge-1"})
        execute_panel((_judge(1), duplicate), _ScriptedInvoke(()), threshold=0.5)


@pytest.mark.parametrize("threshold", [True, -0.01, 1.01])
def test_execute_panel_rejects_invalid_threshold(threshold: object) -> None:
    with pytest.raises(ValueError, match="threshold must be numeric between 0 and 1"):
        execute_panel(
            _judges(1),
            _ScriptedInvoke((_success(0.5),)),
            threshold=threshold,  # type: ignore[arg-type]
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
    ],
)
def test_attempt_observation_rejects_invalid_shape(observation: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AttemptObservation(**observation)  # type: ignore[arg-type]


def test_attempt_observation_freezes_validated_usage() -> None:
    usage = {"input_tokens": 4}
    observation = _success(0.5)
    observation = AttemptObservation(**{**observation.__dict__, "usage": usage})
    usage["input_tokens"] = 8

    assert observation.usage == {"input_tokens": 4}
    with pytest.raises(TypeError):
        observation.usage["input_tokens"] = 8  # type: ignore[index]
