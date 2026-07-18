"""Normalized persistence contracts for reflection inputs and results."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from weave_agent_signals.run_config import ModelDescriptor
from weave_agent_signals.runs.bundles import BundleSnapshot
from weave_agent_signals.runs.challenges.contracts import ChallengeResult

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SCORE_BASIS = "predicted_evaluator"
_NO_VALID_PROPOSAL = "No valid proposal generated"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


def _bundle(value: object) -> BundleSnapshot:
    if isinstance(value, BundleSnapshot):
        return value
    if isinstance(value, Mapping):
        return BundleSnapshot.from_dict(value)
    raise ValueError("bundle must be a BundleSnapshot")


def _usage(value: dict[str, int]) -> dict[str, int]:
    if any(not key.strip() for key in value):
        raise ValueError("usage keys must be nonblank")
    if any(type(count) is not int or count < 0 for count in value.values()):
        raise ValueError("usage values must be nonnegative integers")
    return value


class FeedbackIdentity(_Record):
    id: str = Field(min_length=1)
    weave_ref: str = Field(min_length=1)
    feedback_type: str = Field(min_length=1)
    digest: str

    @field_validator("digest")
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("feedback digest must be a SHA-256 identity")
        return value


class ReflectionInputRecord(_Record):
    cohort_id: str = Field(min_length=1)
    captured_at: datetime
    feedback: tuple[FeedbackIdentity, ...]
    target_registry: dict[str, Any]
    baseline: BundleSnapshot

    @field_validator("baseline", mode="before")
    @classmethod
    def _valid_baseline(cls, value: object) -> BundleSnapshot:
        return _bundle(value)

    @field_validator("target_registry")
    @classmethod
    def _json_registry(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("target_registry must contain JSON values") from error
        return value

    @field_serializer("baseline")
    def _serialize_baseline(self, value: BundleSnapshot) -> dict[str, Any]:
        return value.to_dict()

    @classmethod
    def from_epoch9(cls, value: Mapping[str, Any]) -> ReflectionInputRecord:
        expected = {
            "schema_version",
            "cohort_id",
            "captured_at",
            "turn_count",
            "session_count",
            "feedback_count",
            "feedback",
            "target_registry",
            "baseline",
        }
        if set(value) != expected or value.get("schema_version") != "3":
            raise ValueError("invalid epoch-9 reflection input")
        feedback = value.get("feedback")
        feedback_count = value.get("feedback_count")
        if not isinstance(feedback, list) or feedback_count != len(feedback):
            raise ValueError("feedback_count does not match feedback")
        for name in ("turn_count", "session_count"):
            count = value.get(name)
            if type(count) is not int or count < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        return cls.model_validate(
            {
                "cohort_id": value["cohort_id"],
                "captured_at": value["captured_at"],
                "feedback": feedback,
                "target_registry": value["target_registry"],
                "baseline": value["baseline"],
            }
        )


class GenerationAttemptRecord(_Record):
    attempt_id: str = Field(min_length=1)
    number: int = Field(ge=1)
    status: Literal["succeeded", "failed", "cancelled"]
    requested_writer: ModelDescriptor
    resolved_model: str | None = None
    resolved_family: str | None = None
    resolved_backend: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    candidate_id: str | None = None
    bundle: BundleSnapshot | None = None
    changed_paths: tuple[str, ...] = ()
    response_digest: str | None = None
    response_excerpt: str | None = None
    error_type: str | None = None
    error: str | None = None

    @field_validator("bundle", mode="before")
    @classmethod
    def _valid_bundle(cls, value: object) -> BundleSnapshot | None:
        return None if value is None else _bundle(value)

    @field_validator("usage")
    @classmethod
    def _valid_usage(cls, value: dict[str, int]) -> dict[str, int]:
        return _usage(value)

    @field_validator("response_digest")
    @classmethod
    def _valid_digest(cls, value: str | None) -> str | None:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("response_digest must be a SHA-256 identity")
        return value

    @field_serializer("bundle")
    def _serialize_bundle(self, value: BundleSnapshot | None) -> dict[str, Any] | None:
        return value.to_dict() if value is not None else None

    @model_validator(mode="after")
    def _valid_outcome(self) -> Self:
        if "proposal_writer" not in self.requested_writer.supported_roles:
            raise ValueError("requested_writer does not support proposal generation")
        if len(set(self.changed_paths)) != len(self.changed_paths) or any(
            not path for path in self.changed_paths
        ):
            raise ValueError("changed_paths must be unique and nonblank")
        if self.status == "succeeded":
            required = (
                self.resolved_model,
                self.resolved_family,
                self.resolved_backend,
                self.candidate_id,
                self.bundle,
            )
            if any(value is None for value in required) or self.error_type or self.error:
                raise ValueError("successful attempt requires candidate and writer provenance")
        elif self.candidate_id is not None or self.bundle is not None:
            raise ValueError("unsuccessful attempt cannot own a candidate")
        elif not self.error_type or not self.error:
            raise ValueError("unsuccessful attempt requires an error")
        return self


class EvaluatorRecord(_Record):
    evaluation_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    requested_family: str = Field(min_length=1)
    requested_backend: str = Field(min_length=1)
    resolved_model: str = Field(min_length=1)
    resolved_family: str = Field(min_length=1)
    resolved_backend: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
    usage: dict[str, int] = Field(default_factory=dict)

    @field_validator("usage")
    @classmethod
    def _valid_usage(cls, value: dict[str, int]) -> dict[str, int]:
        return _usage(value)

    @property
    def identity(self) -> tuple[str, ...]:
        return (
            self.requested_model,
            self.requested_family,
            self.requested_backend,
            self.resolved_model,
            self.resolved_family,
            self.resolved_backend,
        )


class DerivedCandidate(_Record):
    candidate_id: str
    bundle: BundleSnapshot
    score: float
    score_delta: float
    rationale: str
    generation_attempt_id: str
    requested_writer: ModelDescriptor
    resolved_writer_model: str
    resolved_writer_family: str
    resolved_writer_backend: str
    evaluation: EvaluatorRecord

    @field_serializer("bundle")
    def _serialize_bundle(self, value: BundleSnapshot) -> dict[str, Any]:
        return value.to_dict()


class ReflectionResultRecord(_Record):
    baseline: BundleSnapshot
    attempts: tuple[GenerationAttemptRecord, ...]
    evaluations: tuple[EvaluatorRecord, ...]
    recommended_candidate_id: str | None
    baseline_won: bool
    reason: str | None = None
    score_basis: Literal["predicted_evaluator"] = _SCORE_BASIS
    provisional_candidate_id: str | None = None
    challenge: ChallengeResult | dict[str, Any] | None = None

    @field_validator("baseline", mode="before")
    @classmethod
    def _valid_baseline(cls, value: object) -> BundleSnapshot:
        return _bundle(value)

    @field_serializer("baseline")
    def _serialize_baseline(self, value: BundleSnapshot) -> dict[str, Any]:
        return value.to_dict()

    @model_validator(mode="after")
    def _valid_links(self) -> Self:
        if self.reason is not None and not self.attempts and not self.evaluations:
            if (
                not self.reason.strip()
                or self.recommended_candidate_id is not None
                or self.provisional_candidate_id is not None
                or self.baseline_won
                or self.challenge is not None
            ):
                raise ValueError("empty reflection result has inconsistent evidence")
            return self
        attempt_ids = tuple(attempt.attempt_id for attempt in self.attempts)
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("attempt IDs must be unique")
        if tuple(attempt.number for attempt in self.attempts) != tuple(
            range(1, len(self.attempts) + 1)
        ):
            raise ValueError("attempt numbers must be contiguous and ordered")
        successful = {
            attempt.candidate_id: attempt
            for attempt in self.attempts
            if attempt.status == "succeeded" and attempt.candidate_id is not None
        }
        if len(successful) != sum(attempt.status == "succeeded" for attempt in self.attempts):
            raise ValueError("successful candidate IDs must be unique")
        evaluations = {evaluation.target_id: evaluation for evaluation in self.evaluations}
        expected_targets = {"baseline", *successful}
        if len(evaluations) != len(self.evaluations) or set(evaluations) != expected_targets:
            raise ValueError("evaluation targets must match baseline and successful candidates")
        identities = {evaluation.identity for evaluation in self.evaluations}
        if len(identities) != 1:
            raise ValueError("evaluations must use one authoritative evaluator identity")
        selected = (self.recommended_candidate_id, self.provisional_candidate_id)
        if any(value is not None and value not in successful for value in selected):
            raise ValueError("selected candidate is not present")
        if self.reason is not None:
            if self.reason != _NO_VALID_PROPOSAL or successful or self.recommended_candidate_id:
                raise ValueError("no-valid-proposal result has inconsistent evidence")
        elif self.baseline_won != (self.recommended_candidate_id is None):
            raise ValueError("baseline winner and recommendation disagree")
        revisions = [self.baseline.revision]
        revisions.extend(
            attempt.bundle.revision for attempt in successful.values() if attempt.bundle is not None
        )
        if len(revisions) != len(set(revisions)):
            raise ValueError("baseline and candidate revisions must be unique")
        if self.challenge is not None:
            if isinstance(self.challenge, ChallengeResult):
                challenge_candidate_id = self.challenge.candidate_id
                recommendation = self.challenge.recommendation()
            else:
                challenge_candidate_id = self.challenge.get("candidate_id")
                recommendation = (
                    challenge_candidate_id if self.challenge.get("winner") == "candidate" else None
                )
            if self.provisional_candidate_id != challenge_candidate_id:
                raise ValueError("challenge does not target the provisional candidate")
            if self.recommended_candidate_id != recommendation:
                raise ValueError("recommendation does not match challenge evidence")
        return self

    def candidate(self, candidate_id: str) -> DerivedCandidate:
        attempt = next(
            (
                value
                for value in self.attempts
                if value.status == "succeeded" and value.candidate_id == candidate_id
            ),
            None,
        )
        if attempt is None or attempt.bundle is None:
            raise KeyError(candidate_id)
        evaluation = next(value for value in self.evaluations if value.target_id == candidate_id)
        baseline = next(value for value in self.evaluations if value.target_id == "baseline")
        return DerivedCandidate(
            candidate_id=candidate_id,
            bundle=attempt.bundle,
            score=evaluation.score,
            score_delta=evaluation.score - baseline.score,
            rationale=evaluation.rationale,
            generation_attempt_id=attempt.attempt_id,
            requested_writer=attempt.requested_writer,
            resolved_writer_model=attempt.resolved_model or "",
            resolved_writer_family=attempt.resolved_family or "",
            resolved_writer_backend=attempt.resolved_backend or "",
            evaluation=evaluation,
        )

    @classmethod
    def from_epoch9(cls, value: Mapping[str, Any]) -> ReflectionResultRecord:
        from weave_agent_signals.runs.reflection import ReflectionResult

        normalized = json.loads(json.dumps(value))
        owners = (
            *normalized.get("generation_attempts", []),
            *normalized.get("candidates", []),
        )
        for owner in owners:
            descriptor = owner.get("requested_writer") if isinstance(owner, dict) else None
            if isinstance(descriptor, dict) and "provider" not in descriptor:
                descriptor["provider"] = descriptor.pop("backend")
                descriptor["provider_model"] = descriptor["id"]
        raw_challenge = normalized.get("challenge")
        challenge = _challenge_from_epoch9(raw_challenge)
        normalized["challenge"] = None
        legacy = ReflectionResult.from_dict(normalized)
        candidates = {candidate.generation_attempt_id: candidate for candidate in legacy.candidates}
        attempts = tuple(
            GenerationAttemptRecord(
                attempt_id=attempt.attempt_id,
                number=attempt.number,
                status=attempt.status,
                requested_writer=attempt.requested_writer,
                resolved_model=attempt.resolved_model,
                resolved_family=attempt.resolved_family,
                resolved_backend=attempt.resolved_backend,
                usage=dict(attempt.usage),
                candidate_id=(
                    candidates[attempt.attempt_id].candidate_id
                    if attempt.attempt_id in candidates
                    else None
                ),
                bundle=(
                    candidates[attempt.attempt_id].bundle
                    if attempt.attempt_id in candidates
                    else None
                ),
                changed_paths=attempt.changed_paths,
                response_digest=attempt.response_digest,
                response_excerpt=attempt.response_excerpt,
                error_type=attempt.error_type,
                error=attempt.error,
            )
            for attempt in legacy.generation_attempts
        )

        def evaluation(value: Any, target_id: str) -> EvaluatorRecord:
            return EvaluatorRecord(
                evaluation_id=value.evaluation_id,
                target_id=target_id,
                requested_model=value.requested_model,
                requested_family=value.requested_family,
                requested_backend=value.requested_backend,
                resolved_model=value.resolved_model,
                resolved_family=value.resolved_family,
                resolved_backend=value.resolved_backend,
                score=value.score,
                rationale=value.rationale,
                usage=dict(value.usage),
            )

        evaluations = (evaluation(legacy.baseline_evaluation, "baseline"),) + tuple(
            evaluation(candidate.evaluation, candidate.candidate_id)
            for candidate in legacy.candidates
        )
        return cls(
            baseline=legacy.baseline,
            attempts=attempts,
            evaluations=evaluations,
            recommended_candidate_id=legacy.recommended_candidate_id,
            baseline_won=legacy.baseline_won,
            reason=legacy.reason,
            score_basis=legacy.score_basis,
            provisional_candidate_id=legacy.provisional_candidate_id,
            challenge=challenge,
        )


def _challenge_from_epoch9(
    value: object,
) -> ChallengeResult | dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("historical challenge must be an object")
    historical = json.loads(json.dumps(value))
    try:
        return ChallengeResult.from_dict(historical)
    except ValueError:
        task = historical.get("task")
        if not isinstance(task, dict) or task.get("schema_version") != "2":
            return historical
    task["schema_version"] = "3"
    task["task_id"] = "pending"
    task.setdefault("required_files", [])
    task.setdefault("required_executables", [])
    task.setdefault("requires_git_metadata", False)
    historical["challenge_id"] = "pending"
    return ChallengeResult.from_dict(historical)


__all__ = [
    "DerivedCandidate",
    "EvaluatorRecord",
    "FeedbackIdentity",
    "GenerationAttemptRecord",
    "ReflectionInputRecord",
    "ReflectionResultRecord",
]
