"""Pure reviewer escalation and score aggregation policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Literal

from weave_agent_signals.run_config import PositionedJudge

ReviewDepth = Literal["primary", "selective", "full_panel"]
ObservationStatus = Literal["succeeded", "abstained", "failed"]
ReviewStatus = Literal["complete", "degraded", "unresolved", "failed"]
InferencePhase = Literal["digest", "window", "merge"]


def _unit_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric between 0 and 1")
    result = float(value)
    if not isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{field_name} must be numeric between 0 and 1")
    return result


def _nonblank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")
    return value


@dataclass(frozen=True)
class ReviewPolicy:
    depth: ReviewDepth
    judges: tuple[PositionedJudge, ...]
    second_opinion_margin: float | None

    def __post_init__(self) -> None:
        if self.depth not in {"primary", "selective", "full_panel"}:
            raise ValueError("depth must be primary, selective, or full_panel")
        if not isinstance(self.judges, tuple) or any(
            not isinstance(judge, PositionedJudge) for judge in self.judges
        ):
            raise ValueError("judges must be a tuple of PositionedJudge values")

        judge_count = len(self.judges)
        expected_counts = {
            "primary": {1},
            "selective": {2, 3},
            "full_panel": {3},
        }
        if judge_count not in expected_counts[self.depth]:
            requirement = {
                "primary": "exactly 1 judge",
                "selective": "2 or 3 judges",
                "full_panel": "exactly 3 judges",
            }[self.depth]
            raise ValueError(f"{self.depth} review requires {requirement}")

        expected_positions = tuple(range(1, judge_count + 1))
        if tuple(judge.position for judge in self.judges) != expected_positions:
            raise ValueError("judge positions must be contiguous and ordered from 1")
        judge_ids = tuple(judge.id for judge in self.judges)
        if len(judge_ids) != len(set(judge_ids)):
            raise ValueError("judge model IDs must be unique")

        if self.depth == "selective":
            margin = self.second_opinion_margin
            if margin is None:
                raise ValueError("selective review requires second_opinion_margin")
            if isinstance(margin, bool) or not isinstance(margin, (int, float)):
                raise ValueError("second_opinion_margin must be between 0 and 0.5")
            normalized_margin = float(margin)
            if not isfinite(normalized_margin) or not 0 <= normalized_margin <= 0.5:
                raise ValueError("second_opinion_margin must be between 0 and 0.5")
            object.__setattr__(self, "second_opinion_margin", normalized_margin)
        elif self.second_opinion_margin is not None:
            raise ValueError("second_opinion_margin must be null outside selective review")


@dataclass(frozen=True)
class InferenceStepAudit:
    phase: InferencePhase
    artifact_id: str
    requested_model: str
    resolved_model: str | None
    usage: Mapping[str, int]
    output_mode: str | None
    schema_name: str
    schema_fallback_reason: str | None
    transport_request_count: int
    raw_output_digest: str | None
    reused: bool = False

    def __post_init__(self) -> None:
        if self.phase not in {"digest", "window", "merge"}:
            raise ValueError("phase must be digest, window, or merge")
        for name in ("artifact_id", "requested_model", "schema_name"):
            _nonblank(getattr(self, name), name)
        if self.resolved_model is not None:
            _nonblank(self.resolved_model, "resolved_model")
        usage = dict(self.usage)
        if any(not isinstance(key, str) or not key.strip() for key in usage):
            raise ValueError("usage keys must be nonblank strings")
        if any(type(value) is not int or value < 0 for value in usage.values()):
            raise ValueError("usage values must be nonnegative integers")
        object.__setattr__(self, "usage", MappingProxyType(usage))
        if self.output_mode is not None and self.output_mode not in {
            "json_object",
            "json_schema",
            "json_object_fallback",
        }:
            raise ValueError("output_mode is invalid")
        if self.schema_fallback_reason is not None:
            _nonblank(self.schema_fallback_reason, "schema_fallback_reason")
        if type(self.transport_request_count) is not int or self.transport_request_count < 0:
            raise ValueError("transport_request_count must be a nonnegative integer")
        if type(self.reused) is not bool:
            raise ValueError("reused must be a boolean")
        if self.raw_output_digest is not None and (
            not isinstance(self.raw_output_digest, str)
            or len(self.raw_output_digest) != 64
            or any(
                character not in "0123456789abcdefABCDEF" for character in self.raw_output_digest
            )
        ):
            raise ValueError("raw_output_digest must be a SHA-256 hex digest")


@dataclass(frozen=True)
class AttemptObservation:
    status: ObservationStatus
    resolved_model: str | None
    score: float | None
    rationale: str | None
    usage: Mapping[str, int]
    error_type: str | None
    message: str | None
    evidence_ids: tuple[str, ...] = ()
    output_mode: str | None = None
    schema_name: str | None = None
    schema_fallback_reason: str | None = None
    transport_request_count: int = 0
    verdict_schema_version: int | None = None
    raw_output_digest: str | None = None
    behavioral_feedback: Mapping[str, str | None] | None = None
    steps: tuple[InferenceStepAudit, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "abstained", "failed"}:
            raise ValueError("status must be succeeded, abstained, or failed")
        if not isinstance(self.usage, Mapping):
            raise ValueError("usage must be a mapping")
        usage = dict(self.usage)
        if any(not isinstance(key, str) or not key.strip() for key in usage):
            raise ValueError("usage keys must be nonblank strings")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in usage.values()
        ):
            raise ValueError("usage values must be nonnegative integers")
        object.__setattr__(self, "usage", MappingProxyType(usage))

        if not isinstance(self.evidence_ids, tuple):
            raise ValueError("evidence_ids must be a tuple")
        if any(not isinstance(value, str) or not value.strip() for value in self.evidence_ids):
            raise ValueError("evidence_ids must contain nonblank strings")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids must be unique")

        if self.output_mode is not None and self.output_mode not in {
            "json_object",
            "json_schema",
            "json_object_fallback",
        }:
            raise ValueError("output_mode is invalid")
        for name in (
            "schema_name",
            "schema_fallback_reason",
        ):
            value = getattr(self, name)
            if value is not None:
                _nonblank(value, name)
        if self.raw_output_digest is not None and (
            not isinstance(self.raw_output_digest, str)
            or len(self.raw_output_digest) != 64
            or any(
                character not in "0123456789abcdefABCDEF" for character in self.raw_output_digest
            )
        ):
            raise ValueError("raw_output_digest must be a SHA-256 hex digest")
        if (
            isinstance(self.transport_request_count, bool)
            or not isinstance(self.transport_request_count, int)
            or self.transport_request_count < 0
        ):
            raise ValueError("transport_request_count must be a nonnegative integer")
        if self.verdict_schema_version is not None and (
            isinstance(self.verdict_schema_version, bool)
            or not isinstance(self.verdict_schema_version, int)
            or self.verdict_schema_version < 1
        ):
            raise ValueError("verdict_schema_version must be a positive integer or null")

        if self.resolved_model is not None:
            _nonblank(self.resolved_model, "resolved_model")
        if self.rationale is not None and not isinstance(self.rationale, str):
            raise ValueError("rationale must be a string or null")
        if not isinstance(self.steps, tuple) or any(
            not isinstance(step, InferenceStepAudit) for step in self.steps
        ):
            raise ValueError("steps must be a tuple of InferenceStepAudit values")

        feedback = self.behavioral_feedback
        if feedback is not None:
            if not isinstance(feedback, Mapping):
                raise ValueError("behavioral_feedback must be a mapping or null")
            normalized_feedback = dict(feedback)
            if set(normalized_feedback) != {"success", "problem", "desired_behavior"}:
                raise ValueError("behavioral_feedback fields are invalid")
            if any(
                value is not None and not isinstance(value, str)
                for value in normalized_feedback.values()
            ):
                raise ValueError("behavioral_feedback values must be strings or null")
            if not any(
                value is not None and value.strip() for value in normalized_feedback.values()
            ):
                raise ValueError("behavioral_feedback requires a nonblank value")
            object.__setattr__(
                self,
                "behavioral_feedback",
                MappingProxyType(normalized_feedback),
            )

        if self.status == "succeeded":
            _nonblank(self.resolved_model, "resolved_model")
            object.__setattr__(self, "score", _unit_float(self.score, "score"))
            if self.error_type is not None or self.message is not None:
                raise ValueError("successful observations cannot contain an error")
            if self.behavioral_feedback is None:
                raise ValueError("successful observations require behavioral feedback")
            return

        if self.status == "abstained":
            _nonblank(self.resolved_model, "resolved_model")
            if self.score is not None:
                raise ValueError("abstained observations cannot contain a score")
            _nonblank(self.rationale, "rationale")
            if self.error_type is not None or self.message is not None:
                raise ValueError("abstained observations cannot contain an error")
            if self.behavioral_feedback is not None:
                raise ValueError("abstained observations cannot contain behavioral feedback")
            return

        if self.score is not None or self.rationale is not None or self.evidence_ids:
            raise ValueError("failed observations cannot contain a score, rationale, or evidence")
        if self.behavioral_feedback is not None:
            raise ValueError("failed observations cannot contain behavioral feedback")
        _nonblank(self.error_type, "error_type")
        if not isinstance(self.message, str):
            raise ValueError("failed observations require an error message")


@dataclass(frozen=True)
class ReviewAttempt:
    position: int
    role: str
    trigger: str
    requested_model: str
    requested_family: str
    requested_backend: str
    observation: AttemptObservation


@dataclass(frozen=True)
class ReviewOutcome:
    rating: float | None
    status: ReviewStatus
    attempts: tuple[ReviewAttempt, ...]
    successful_count: int


def _near_boundary(score: float, threshold: float, margin: float) -> bool:
    return abs(score - threshold) <= margin


def _split_across_threshold(scores: tuple[float, ...], threshold: float) -> bool:
    return len(scores) == 2 and (scores[0] >= threshold) != (scores[1] >= threshold)


def _outcome(attempts: list[ReviewAttempt], threshold: float) -> ReviewOutcome:
    successful_scores = tuple(
        attempt.observation.score
        for attempt in attempts
        if attempt.observation.status == "succeeded" and attempt.observation.score is not None
    )
    successful_count = len(successful_scores)
    if not successful_scores:
        return ReviewOutcome(
            rating=None,
            status="failed",
            attempts=tuple(attempts),
            successful_count=0,
        )

    rating = sum(successful_scores) / successful_count
    if _split_across_threshold(successful_scores, threshold):
        status: ReviewStatus = "unresolved"
    elif any(attempt.observation.status != "succeeded" for attempt in attempts):
        status = "degraded"
    else:
        status = "complete"
    return ReviewOutcome(
        rating=rating,
        status=status,
        attempts=tuple(attempts),
        successful_count=successful_count,
    )


def execute_review(
    policy: ReviewPolicy,
    *,
    threshold: float,
    invoke: Callable[[PositionedJudge], AttemptObservation],
) -> ReviewOutcome:
    """Execute one ordered review without performing or converting model I/O."""

    if not isinstance(policy, ReviewPolicy):
        raise TypeError("policy must be a ReviewPolicy")
    normalized_threshold = _unit_float(threshold, "threshold")
    attempts: list[ReviewAttempt] = []

    def attempt(judge: PositionedJudge, trigger: str) -> AttemptObservation:
        observation = invoke(judge)
        if not isinstance(observation, AttemptObservation):
            raise TypeError("invoke must return an AttemptObservation")
        attempts.append(
            ReviewAttempt(
                position=judge.position,
                role=judge.role,
                trigger=trigger,
                requested_model=judge.id,
                requested_family=judge.family,
                requested_backend=judge.backend,
                observation=observation,
            )
        )
        return observation

    first = attempt(policy.judges[0], "initial")
    if policy.depth == "primary":
        return _outcome(attempts, normalized_threshold)

    if policy.depth == "full_panel":
        attempt(policy.judges[1], "full_panel")
        attempt(policy.judges[2], "full_panel")
        return _outcome(attempts, normalized_threshold)

    margin = policy.second_opinion_margin
    if margin is None:  # ReviewPolicy validation makes this unreachable.
        raise AssertionError("selective review requires a margin")
    if first.status == "failed":
        second_trigger = "judge_1_failed"
    elif first.status == "abstained":
        second_trigger = "judge_1_abstained"
    elif first.score is not None and _near_boundary(
        first.score,
        normalized_threshold,
        margin,
    ):
        second_trigger = "near_boundary"
    else:
        return _outcome(attempts, normalized_threshold)

    second = attempt(policy.judges[1], second_trigger)
    if len(policy.judges) == 2:
        return _outcome(attempts, normalized_threshold)

    first_two_scores = tuple(
        observation.score
        for observation in (first, second)
        if observation.status == "succeeded" and observation.score is not None
    )
    third_trigger: str | None = None
    if _split_across_threshold(first_two_scores, normalized_threshold):
        third_trigger = "threshold_disagreement"
    elif len(first_two_scores) == 1 and _near_boundary(
        first_two_scores[0],
        normalized_threshold,
        margin,
    ):
        third_trigger = "only_success_near_boundary"
    elif not first_two_scores:
        statuses = (first.status, second.status)
        if statuses == ("failed", "failed"):
            third_trigger = "both_prior_failed"
        elif statuses == ("abstained", "abstained"):
            third_trigger = "both_prior_abstained"
        else:
            third_trigger = "both_prior_unsuccessful"

    if third_trigger is not None:
        attempt(policy.judges[2], third_trigger)
    return _outcome(attempts, normalized_threshold)
