"""Execute session-only sliding reviews against an exact pinned plan."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.families import model_family
from weave_agent_signals.judges.inference import ChatClient, InferenceCancelled
from weave_agent_signals.judges.review import ReviewAttempt, ReviewPolicy, execute_review
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.judges.sliding import ArtifactLoader, ArtifactRecorder, SlidingReviewer
from weave_agent_signals.models import Score, SessionView
from weave_agent_signals.run_config import (
    JudgingContextPolicy,
    PositionedJudge,
    ReviewDepth,
    RubricDescriptor,
)

REVIEW_POLICY_VERSION = "3"
_REASON_LIMIT = 1200


class JudgeFailure:
    def __init__(
        self,
        rubric: str,
        message: str,
        error_type: str,
        attempts: tuple[dict[str, object], ...] = (),
    ) -> None:
        self.rubric = rubric
        self.message = message
        self.error_type = error_type
        self.attempts = attempts


class JudgeExecutionError(RuntimeError):
    def __init__(self, scores: Sequence[Score], failures: Sequence[JudgeFailure]) -> None:
        self.scores = tuple(scores)
        self.failures = tuple(failures)
        super().__init__(
            f"{len(self.failures)} rubric(s) failed: "
            + "; ".join(f"{item.rubric} ({item.message})" for item in self.failures)
        )


def _step_record(step: object) -> dict[str, object]:
    return {
        name: (dict(value) if name == "usage" else value)
        for name in (
            "phase",
            "artifact_id",
            "requested_model",
            "resolved_model",
            "usage",
            "output_mode",
            "schema_name",
            "schema_fallback_reason",
            "transport_request_count",
            "raw_output_digest",
            "reused",
        )
        if (value := getattr(step, name, None)) is not None
    }


def _attempt_record(attempt: ReviewAttempt) -> dict[str, object]:
    observation = attempt.observation
    return {
        "position": attempt.position,
        "role": attempt.role,
        "trigger": attempt.trigger,
        "requested_model": attempt.requested_model,
        "requested_family": attempt.requested_family,
        "requested_backend": attempt.requested_backend,
        "status": observation.status,
        "resolved_model": observation.resolved_model,
        "resolved_family": (
            model_family(observation.resolved_model) if observation.resolved_model else None
        ),
        "score": observation.score,
        "rationale": observation.rationale,
        "evidence_ids": list(observation.evidence_ids),
        "usage": dict(observation.usage),
        "output_mode": observation.output_mode,
        "schema_name": observation.schema_name,
        "schema_fallback_reason": observation.schema_fallback_reason,
        "transport_request_count": observation.transport_request_count,
        "verdict_schema_version": observation.verdict_schema_version,
        "raw_output_digest": observation.raw_output_digest,
        "error_type": observation.error_type,
        "message": observation.message,
        "behavioral_feedback": (
            dict(observation.behavioral_feedback)
            if observation.behavioral_feedback is not None
            else None
        ),
        "steps": [_step_record(step) for step in observation.steps],
    }


def _reason(attempts: Sequence[ReviewAttempt]) -> str:
    parts: list[str] = []
    for attempt in attempts:
        feedback = attempt.observation.behavioral_feedback
        if attempt.observation.status != "succeeded" or feedback is None:
            continue
        for label, key in (
            ("Success", "success"),
            ("Problem", "problem"),
            ("Next", "desired_behavior"),
        ):
            value = feedback.get(key)
            if value:
                parts.append(f"{label}: {' '.join(value.split())}")
    return " | ".join(dict.fromkeys(parts))[:_REASON_LIMIT]


def _resolve(descriptor: RubricDescriptor) -> object:
    if not isinstance(descriptor, RubricDescriptor):
        raise TypeError("rubrics must contain RubricDescriptor values")
    if descriptor.evaluation_unit != "session" or descriptor.id not in SESSION_RUBRICS:
        raise ValueError("judge_session requires current session rubrics")
    if descriptor != build_rubric_catalog().rubric(descriptor.id):
        raise ValueError(f"Pinned rubric {descriptor.id} does not match current prompt content")
    return SESSION_RUBRICS[descriptor.id]


def judge_session(
    session: SessionView,
    client: ChatClient,
    *,
    rubrics: Sequence[RubricDescriptor],
    judges: Sequence[PositionedJudge],
    review_depth: ReviewDepth,
    second_opinion_margin: float | None,
    judging_plan: Mapping[str, object],
    context_policy: JudgingContextPolicy,
    artifact_loader: ArtifactLoader,
    artifact_recorder: ArtifactRecorder,
    cancel_requested: Callable[[], bool] = lambda: False,
) -> list[Score]:
    """Return one merged score per requested rubric for one pinned session."""

    if not rubrics:
        return []
    policy = ReviewPolicy(
        depth=review_depth,
        judges=tuple(judges),
        second_opinion_margin=second_opinion_margin,
    )
    session_rows = [
        value
        for value in judging_plan.get("sessions", [])  # type: ignore[union-attr]
        if isinstance(value, Mapping) and value.get("conversation_id") == session.conversation_id
    ]
    if len(session_rows) != 1:
        raise ValueError("session is not uniquely pinned in the judging plan")
    expected_ids = [
        value.get("id")
        for value in session_rows[0].get("rubrics", [])
        if isinstance(value, Mapping)
    ]
    if [descriptor.id for descriptor in rubrics] != expected_ids:
        raise ValueError("requested rubrics do not match the exact pinned session plan")

    reviewer_cache: dict[str, SlidingReviewer] = {}

    def reviewer(judge: PositionedJudge) -> SlidingReviewer:
        current = reviewer_cache.get(judge.id)
        if current is None:
            current = SlidingReviewer(
                session=session,
                judge=judge,
                judging_plan=judging_plan,
                context_policy=context_policy,
                client=client,
                load_artifact=artifact_loader,
                record_artifact=artifact_recorder,
                is_cancelled=cancel_requested,
            )
            reviewer_cache[judge.id] = current
        return current

    scores: list[Score] = []
    failures: list[JudgeFailure] = []
    evaluated_models = sorted({turn.model for turn in session.turns if turn.model})
    evaluated_families = sorted({model_family(turn.model or "") for turn in session.turns})
    for descriptor in rubrics:
        rubric = _resolve(descriptor)
        try:
            outcome = execute_review(
                policy,
                threshold=descriptor.pass_threshold,
                invoke=lambda judge: reviewer(judge).review(descriptor),
            )
        except InferenceCancelled:
            raise
        attempts = tuple(_attempt_record(value) for value in outcome.attempts)
        if outcome.rating is None:
            failures.append(
                JudgeFailure(
                    descriptor.id,
                    f"No reviewer produced a valid score for {descriptor.id}",
                    "ReviewFailed",
                    attempts,
                )
            )
            continue
        if outcome.status == "unresolved":
            tags = ["unresolved"]
        elif outcome.rating < descriptor.pass_threshold:
            tags = list(rubric.tags_on_low)
        else:
            tags = list(rubric.tags_on_high)
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for attempt in outcome.attempts
                for evidence_id in attempt.observation.evidence_ids
            )
        )
        feedback = [
            dict(attempt.observation.behavioral_feedback)
            for attempt in outcome.attempts
            if attempt.observation.behavioral_feedback is not None
        ]
        scores.append(
            Score(
                scorer=descriptor.id,
                value=outcome.rating,
                tags=tags,
                granularity="session",
                reason=_reason(outcome.attempts),
                metadata={
                    "rubric_id": descriptor.id,
                    "rubric_version": descriptor.version,
                    "rubric_threshold": descriptor.pass_threshold,
                    "evaluation_unit": "session",
                    "review_depth": policy.depth,
                    "review_policy_version": REVIEW_POLICY_VERSION,
                    "second_opinion_margin": policy.second_opinion_margin,
                    "requested_judge_models": [judge.id for judge in policy.judges],
                    "aggregate": "mean",
                    "review_status": outcome.status,
                    "attempt_count": len(outcome.attempts),
                    "successful_reviewer_count": outcome.successful_count,
                    "attempts": list(attempts),
                    "behavioral_feedback": feedback,
                    "evidence_trace_ids": evidence_ids,
                    "raw_coverage_trace_ids": list(
                        session_rows[0]["raw_coverage_trace_ids"]  # type: ignore[index]
                    ),
                    "plan_id": judging_plan["plan_id"],
                    "reviewer_context": [
                        value["judge"]
                        for value in session_rows[0]["reviewers"]  # type: ignore[index]
                    ],
                    "evaluated_models": evaluated_models,
                    "evaluated_families": evaluated_families,
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
    if failures:
        raise JudgeExecutionError(scores, failures)
    return scores
