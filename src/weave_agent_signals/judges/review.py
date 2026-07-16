"""Pure fixed-panel reviewer execution and score aggregation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, CancelledError, ThreadPoolExecutor, wait
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Literal

from weave_agent_signals.judges.inference import InferenceCancelled
from weave_agent_signals.run_config import PositionedJudge

ObservationStatus = Literal["succeeded", "abstained", "failed", "skipped"]
ReviewStatus = Literal["complete", "degraded", "not_evaluable", "failed"]
InferencePhase = Literal["digest", "window", "merge"]
_SCHEMA_FALLBACK_REASON = "schema_output_unsupported"
PANEL_CONTRACT_VERSION = "1"
log = logging.getLogger("weave_agent_signals.judges")


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


def _validate_schema_fallback_metadata(
    output_mode: str | None,
    schema_fallback_reason: str | None,
) -> None:
    expected_reason = _SCHEMA_FALLBACK_REASON if output_mode == "json_object_fallback" else None
    if schema_fallback_reason != expected_reason:
        raise ValueError("schema fallback metadata is invalid")


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
        _validate_schema_fallback_metadata(self.output_mode, self.schema_fallback_reason)
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
    skip_reason: Literal["insufficient_context_capacity"] | None = None
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
        if self.status not in {"succeeded", "abstained", "failed", "skipped"}:
            raise ValueError("status must be succeeded, abstained, failed, or skipped")
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

        if self.output_mode is not None and self.output_mode not in {
            "json_object",
            "json_schema",
            "json_object_fallback",
        }:
            raise ValueError("output_mode is invalid")
        if self.schema_name is not None:
            _nonblank(self.schema_name, "schema_name")
        _validate_schema_fallback_metadata(self.output_mode, self.schema_fallback_reason)
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
            if self.skip_reason is not None:
                raise ValueError("non-skipped observations cannot contain a skip reason")
            _nonblank(self.resolved_model, "resolved_model")
            object.__setattr__(self, "score", _unit_float(self.score, "score"))
            if self.error_type is not None or self.message is not None:
                raise ValueError("successful observations cannot contain an error")
            if self.behavioral_feedback is None:
                raise ValueError("successful observations require behavioral feedback")
            return

        if self.status == "abstained":
            if self.skip_reason is not None:
                raise ValueError("non-skipped observations cannot contain a skip reason")
            _nonblank(self.resolved_model, "resolved_model")
            if self.score is not None:
                raise ValueError("abstained observations cannot contain a score")
            _nonblank(self.rationale, "rationale")
            if self.error_type is not None or self.message is not None:
                raise ValueError("abstained observations cannot contain an error")
            if self.behavioral_feedback is not None:
                raise ValueError("abstained observations cannot contain behavioral feedback")
            return

        if self.status == "skipped":
            if self.skip_reason != "insufficient_context_capacity":
                raise ValueError("skipped observations require a valid skip reason")
            if (
                self.resolved_model is not None
                or self.score is not None
                or self.rationale is not None
                or self.evidence_ids
                or self.usage
                or self.error_type is not None
                or self.message is not None
                or self.output_mode is not None
                or self.schema_name is not None
                or self.schema_fallback_reason is not None
                or self.transport_request_count != 0
                or self.verdict_schema_version is not None
                or self.raw_output_digest is not None
                or self.behavioral_feedback is not None
                or self.steps
            ):
                raise ValueError("skipped observations cannot contain inference fields")
            return

        if self.skip_reason is not None:
            raise ValueError("non-skipped observations cannot contain a skip reason")
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
class PanelOutcome:
    rating: float | None
    status: ReviewStatus
    attempts: tuple[ReviewAttempt, ...]
    successful_count: int
    minimum: float | None
    maximum: float | None
    spread: float | None


def execute_panel(
    judges: tuple[PositionedJudge, ...],
    invoke: Callable[[PositionedJudge], AttemptObservation],
    *,
    threshold: float,
    cancel_pending: Callable[[], None] = lambda: None,
) -> PanelOutcome:
    """Invoke the selected panel concurrently and aggregate valid observations."""

    _unit_float(threshold, "threshold")
    if not isinstance(judges, tuple) or any(
        not isinstance(judge, PositionedJudge) for judge in judges
    ):
        raise ValueError("judges must be a tuple of PositionedJudge values")
    if not 1 <= len(judges) <= 3:
        raise ValueError("judges must contain one through three models")
    if tuple(judge.position for judge in judges) != tuple(range(1, len(judges) + 1)):
        raise ValueError("judge positions must be contiguous and ordered from 1")
    if len({judge.id for judge in judges}) != len(judges):
        raise ValueError("judge model IDs must be unique")
    if not callable(cancel_pending):
        raise TypeError("cancel_pending must be callable")

    observations: dict[int, AttemptObservation] = {}
    first_error: Exception | None = None
    failure_seen = False
    cancellation_requested = False
    executor = ThreadPoolExecutor(max_workers=len(judges), thread_name_prefix="judge-panel")
    future_judges = {executor.submit(invoke, judge): judge for judge in judges}
    pending = set(future_judges)

    def cancel_outstanding() -> None:
        nonlocal cancellation_requested
        if cancellation_requested:
            return
        cancellation_requested = True
        for future in pending:
            future.cancel()
        try:
            cancel_pending()
        except Exception as error:
            log.warning(
                "Judge panel cancellation callback failed: error_type=%s",
                type(error).__name__,
            )

    try:
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda value: future_judges[value].position):
                judge = future_judges[future]
                try:
                    observation = future.result()
                except (CancelledError, InferenceCancelled) as error:
                    if not failure_seen and first_error is None:
                        first_error = error
                        cancel_outstanding()
                    continue
                except Exception as error:
                    if first_error is None:
                        first_error = error
                        cancel_outstanding()
                    continue
                if not isinstance(observation, AttemptObservation):
                    if first_error is None:
                        first_error = TypeError("invoke must return an AttemptObservation")
                        cancel_outstanding()
                    continue
                observations[judge.position] = observation
                if observation.status == "failed" and not failure_seen:
                    failure_seen = True
                    cancel_outstanding()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if first_error is not None:
        raise first_error

    attempts = [
        ReviewAttempt(
            position=judge.position,
            role=judge.role,
            trigger="panel",
            requested_model=judge.id,
            requested_family=judge.family,
            requested_backend=judge.backend,
            observation=observations[judge.position],
        )
        for judge in judges
        if judge.position in observations
    ]
    if failure_seen:
        successful_count = sum(attempt.observation.status == "succeeded" for attempt in attempts)
        return PanelOutcome(
            None,
            "failed",
            tuple(attempts),
            successful_count,
            None,
            None,
            None,
        )

    if len(attempts) != len(judges):
        raise AssertionError("completed judge panel is missing an observation")

    scores = tuple(
        attempt.observation.score
        for attempt in attempts
        if attempt.observation.status == "succeeded" and attempt.observation.score is not None
    )
    eligible = tuple(attempt for attempt in attempts if attempt.observation.status != "skipped")
    if not eligible or all(attempt.observation.status == "abstained" for attempt in eligible):
        return PanelOutcome(None, "not_evaluable", tuple(attempts), 0, None, None, None)
    if not scores:
        return PanelOutcome(None, "failed", tuple(attempts), len(scores), None, None, None)
    minimum = min(scores)
    maximum = max(scores)
    return PanelOutcome(
        rating=sum(scores) / len(scores),
        status="complete" if len(scores) == len(attempts) else "degraded",
        attempts=tuple(attempts),
        successful_count=len(scores),
        minimum=minimum,
        maximum=maximum,
        spread=maximum - minimum,
    )
