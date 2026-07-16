"""Execute session-only sliding reviews against an exact pinned plan."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.families import model_family
from weave_agent_signals.judges.inference import ChatClient, InferenceCancelled
from weave_agent_signals.judges.review import (
    PANEL_CONTRACT_VERSION,
    AttemptObservation,
    ReviewAttempt,
    execute_panel,
)
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.judges.sliding import (
    ActivityRecorder,
    ArtifactLoader,
    ArtifactRecorder,
    SlidingReviewer,
    sliding_protocol_contract_manifest,
)
from weave_agent_signals.judges.windowing import WindowPlanInapplicable, build_window_plan
from weave_agent_signals.models import Score, SessionView
from weave_agent_signals.run_config import (
    JudgingContextPolicy,
    PositionedJudge,
    RubricDescriptor,
)

_REASON_LIMIT = 1200
log = logging.getLogger("weave_agent_signals.judges")


def _emit_activity(activity: ActivityRecorder | None, event: Mapping[str, object]) -> None:
    if activity is None:
        return
    try:
        activity(dict(event))
    except Exception as error:
        log.warning(
            "Judge activity callback failed: error_type=%s",
            type(error).__name__,
        )


def _plan_digest(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _authenticate_plan_policy(
    plan: Mapping[str, object],
    *,
    session: SessionView,
    rubrics: Sequence[RubricDescriptor],
    judges: Sequence[PositionedJudge],
    context_policy: JudgingContextPolicy,
) -> Mapping[str, object]:
    body = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("schema_version") != "3" or plan.get("plan_id") != _plan_digest(body):
        raise ValueError("judging plan schema or content digest is invalid")
    if plan.get("input_policy") != context_policy.model_dump(mode="json"):
        raise ValueError("context policy does not match the pinned judging plan")
    if plan.get("protocol") != sliding_protocol_contract_manifest():
        raise ValueError("protocol does not match the pinned judging plan")
    sessions = plan.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("pinned judging sessions are invalid")
    matches = [
        value
        for value in sessions
        if isinstance(value, Mapping) and value.get("conversation_id") == session.conversation_id
    ]
    if len(matches) != 1:
        raise ValueError("session is not uniquely pinned in the judging plan")
    session_row = matches[0]
    if session_row.get("turn_count") != len(session.turns):
        raise ValueError("session turn count does not match the pinned judging plan")
    if session_row.get("raw_coverage_trace_ids") != [turn.trace_id for turn in session.turns]:
        raise ValueError("session raw coverage does not match the pinned judging plan")
    reviewers = session_row.get("reviewers")
    if not isinstance(reviewers, list) or [
        value.get("judge") if isinstance(value, Mapping) else None for value in reviewers
    ] != [judge.model_dump(mode="json") for judge in judges]:
        raise ValueError("ordered judges do not match the pinned judging plan")
    for reviewer, judge in zip(reviewers, judges, strict=True):
        if not isinstance(reviewer, Mapping):  # pragma: no cover - guarded above
            raise ValueError("reviewer dispositions are invalid")
        status = reviewer.get("status")
        skip_reason = reviewer.get("skip_reason")
        window_plan = reviewer.get("window_plan")
        work_bounds = reviewer.get("work_bounds")
        try:
            expected_window_plan = build_window_plan(
                session,
                context_policy,
                judge.max_input_tokens,
                judge.token_counter,
            )
        except WindowPlanInapplicable as error:
            if (
                status != "skipped"
                or skip_reason != error.reason
                or str(error) != error.reason
                or window_plan is not None
                or work_bounds
                != {
                    "digest_calls": 0,
                    "window_calls_per_rubric": 0,
                    "merge_calls_per_rubric": 0,
                }
            ):
                raise ValueError("reviewer disposition does not match context capacity") from error
            continue
        chunk_count = expected_window_plan["chunk_count"]
        if (
            status != "planned"
            or skip_reason is not None
            or window_plan != expected_window_plan
            or work_bounds
            != {
                "digest_calls": chunk_count,
                "window_calls_per_rubric": chunk_count,
                "merge_calls_per_rubric": 1,
            }
        ):
            raise ValueError("reviewer disposition does not match context capacity")
    applicable_count = sum(
        isinstance(value, Mapping) and value.get("status") == "planned" for value in reviewers
    )
    expected_rows = [
        {
            **descriptor.model_dump(mode="json"),
            "minimum_reviewer_attempts": applicable_count,
            "maximum_reviewer_attempts": applicable_count,
        }
        for descriptor in rubrics
    ]
    if session_row.get("rubrics") != expected_rows:
        raise ValueError("rubrics or attempt bounds do not match the pinned judging plan")
    return session_row


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


class JudgeNotEvaluable:
    def __init__(
        self,
        rubric: str,
        attempts: tuple[dict[str, object], ...],
    ) -> None:
        self.rubric = rubric
        self.attempts = attempts


class JudgeExecutionError(RuntimeError):
    def __init__(
        self,
        scores: Sequence[Score],
        failures: Sequence[JudgeFailure],
        not_evaluable: Sequence[JudgeNotEvaluable] = (),
    ) -> None:
        self.scores = tuple(scores)
        self.failures = tuple(failures)
        self.not_evaluable = tuple(not_evaluable)
        super().__init__(
            f"{len(self.failures)} rubric(s) failed and "
            f"{len(self.not_evaluable)} were not evaluable"
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
        "skip_reason": observation.skip_reason,
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


def _review_failure_message(
    rubric_id: str,
    attempts: Sequence[ReviewAttempt],
    successful_count: int,
) -> str:
    failed = [attempt for attempt in attempts if attempt.observation.status == "failed"]
    if not failed:
        return f"No reviewer produced a valid score for {rubric_id}"
    success_note = ""
    if successful_count:
        noun = "reviewer" if successful_count == 1 else "reviewers"
        success_note = f" after {successful_count} {noun} succeeded"
    details = "; ".join(
        f"{attempt.requested_model} ({attempt.observation.error_type}): "
        f"{attempt.observation.message}"
        for attempt in failed
    )
    return f"Review failed for {rubric_id}{success_note}: {details}"


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
    judging_plan: Mapping[str, object],
    context_policy: JudgingContextPolicy,
    artifact_loader: ArtifactLoader,
    artifact_recorder: ArtifactRecorder,
    cancel_requested: Callable[[], bool] = lambda: False,
    activity: ActivityRecorder | None = None,
) -> list[Score]:
    """Return one merged score per requested rubric for one pinned session."""

    if not rubrics:
        return []
    panel = tuple(judges)
    session_row = _authenticate_plan_policy(
        judging_plan,
        session=session,
        rubrics=rubrics,
        judges=judges,
        context_policy=context_policy,
    )

    reviewer_cache: dict[str, SlidingReviewer] = {}
    reviewer_rows = session_row["reviewers"]
    dispositions = {
        row["judge"]["id"]: (row["status"], row["skip_reason"])
        for row in reviewer_rows  # type: ignore[union-attr]
    }

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
                activity=activity,
            )
            reviewer_cache[judge.id] = current
        return current

    scores: list[Score] = []
    not_evaluable: list[JudgeNotEvaluable] = []
    abort = getattr(client, "abort", None)
    cancel_pending = abort if callable(abort) else lambda: None
    evaluated_models = sorted({turn.model for turn in session.turns if turn.model})
    evaluated_families = sorted({model_family(turn.model or "") for turn in session.turns})
    for descriptor in rubrics:
        rubric = _resolve(descriptor)
        _emit_activity(
            activity,
            {
                "phase": "rubric_started",
                "message": f"Reviewing {descriptor.label}",
                "conversation_id": session.conversation_id,
                "rubric": descriptor.id,
            },
        )
        try:
            outcome = execute_panel(
                panel,
                invoke=lambda judge: (
                    AttemptObservation(
                        status="skipped",
                        skip_reason="insufficient_context_capacity",
                        resolved_model=None,
                        score=None,
                        rationale=None,
                        usage={},
                        error_type=None,
                        message=None,
                    )
                    if dispositions[judge.id][0] == "skipped"
                    else reviewer(judge).review(descriptor)
                ),
                threshold=descriptor.pass_threshold,
                cancel_pending=cancel_pending,
            )
        except InferenceCancelled:
            raise
        _emit_activity(
            activity,
            {
                "phase": "rubric_completed",
                "message": f"Completed {descriptor.label}: {outcome.status.replace('_', ' ')}",
                "conversation_id": session.conversation_id,
                "rubric": descriptor.id,
                "status": outcome.status,
            },
        )
        attempts = tuple(_attempt_record(value) for value in outcome.attempts)
        if outcome.rating is None:
            if outcome.status == "not_evaluable":
                not_evaluable.append(JudgeNotEvaluable(descriptor.id, attempts))
                continue
            raise JudgeExecutionError(
                scores,
                [
                    JudgeFailure(
                        descriptor.id,
                        _review_failure_message(
                            descriptor.id,
                            outcome.attempts,
                            outcome.successful_count,
                        ),
                        "ReviewFailed",
                        attempts,
                    )
                ],
                not_evaluable,
            )
        if outcome.rating < descriptor.pass_threshold:
            tags = list(rubric.tags_on_low)
        else:
            tags = list(rubric.tags_on_high)
        evidence_ids = [
            evidence_id
            for attempt in outcome.attempts
            for evidence_id in attempt.observation.evidence_ids
        ]
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
                    "panel_contract_version": PANEL_CONTRACT_VERSION,
                    "requested_judge_models": [judge.id for judge in panel],
                    "panel_size": len(panel),
                    "aggregate": "mean",
                    "review_status": outcome.status,
                    "minimum": outcome.minimum,
                    "maximum": outcome.maximum,
                    "spread": outcome.spread,
                    "attempt_count": len(outcome.attempts),
                    "successful_reviewer_count": outcome.successful_count,
                    "attempts": list(attempts),
                    "behavioral_feedback": feedback,
                    "evidence_trace_ids": evidence_ids,
                    "raw_coverage_trace_ids": list(
                        session_row["raw_coverage_trace_ids"]  # type: ignore[index]
                    ),
                    "plan_id": judging_plan["plan_id"],
                    "reviewer_context": [
                        value["judge"]
                        for value in session_row["reviewers"]  # type: ignore[index]
                    ],
                    "evaluated_models": evaluated_models,
                    "evaluated_families": evaluated_families,
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
    if not_evaluable:
        raise JudgeExecutionError(scores, (), not_evaluable)
    return scores
