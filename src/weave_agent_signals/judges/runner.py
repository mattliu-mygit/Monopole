"""Execute session-only sliding reviews against an exact pinned plan."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.families import model_family
from weave_agent_signals.judges.inference import ChatClient
from weave_agent_signals.judges.plan import JudgingPlan, SessionPlan, build_canonical_judging_plan
from weave_agent_signals.judges.review import (
    PANEL_CONTRACT_VERSION,
    AttemptObservation,
    PanelOutcome,
    ReviewAttempt,
    execute_panel,
)
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.judges.sliding import (
    ActivityRecorder,
    CallLoader,
    CallRecorder,
    SlidingReviewer,
)
from weave_agent_signals.models import Score, SessionView
from weave_agent_signals.run_config import (
    JudgingContextPolicy,
    PositionedJudge,
    RubricDescriptor,
)

_REASON_LIMIT = 1200
_PROVIDER_CONCURRENCY = {"wandb": 4, "codex": 2}
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


def _authenticate_plan_policy(
    plan: JudgingPlan,
    *,
    session: SessionView,
    rubrics: Sequence[RubricDescriptor],
    judges: Sequence[PositionedJudge],
    context_policy: JudgingContextPolicy,
) -> SessionPlan:
    if not isinstance(plan, JudgingPlan):
        raise TypeError("judging_plan must be a JudgingPlan")
    if plan.context_policy != context_policy:
        raise ValueError("context policy does not match the pinned judging plan")
    if plan.rubrics != tuple(rubrics):
        raise ValueError("rubrics do not match the pinned judging plan")
    if plan.reviewers != tuple(judges):
        raise ValueError("ordered judges do not match the pinned judging plan")
    try:
        session_row = plan.session(session.conversation_id)
    except KeyError:
        raise ValueError("session is not uniquely pinned in the judging plan")
    expected = build_canonical_judging_plan(
        [session],
        cohort_id=plan.cohort_id,
        rubrics=rubrics,
        judge_models=judges,
        context_policy=context_policy,
    ).sessions[0]
    if session_row != expected:
        raise ValueError("session evidence or reviewer plans do not match the pinned judging plan")
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
    record = {
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
    diagnostics = getattr(step, "response_diagnostics", ())
    record["response_diagnostics"] = [item.model_dump(mode="json") for item in diagnostics]
    return record


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
    clients: Mapping[str, ChatClient],
    *,
    rubrics: Sequence[RubricDescriptor],
    judges: Sequence[PositionedJudge],
    judging_plan: JudgingPlan,
    context_policy: JudgingContextPolicy,
    call_loader: CallLoader,
    call_recorder: CallRecorder,
    cancel_requested: Callable[[], bool] = lambda: False,
    activity: ActivityRecorder | None = None,
) -> list[Score]:
    """Return one merged score per requested rubric for one pinned session."""

    if not rubrics:
        return []
    judging_plan = JudgingPlan.model_validate(judging_plan.model_dump(mode="json"))
    panel = tuple(judges)
    session_row = _authenticate_plan_policy(
        judging_plan,
        session=session,
        rubrics=rubrics,
        judges=judges,
        context_policy=context_policy,
    )

    reviewer_rows = session_row.reviewers
    dispositions = {
        judge.id: (row.status, row.skip_reason)
        for judge, row in zip(judges, reviewer_rows, strict=True)
    }

    provider_gates = {
        provider: threading.BoundedSemaphore(limit)
        for provider, limit in _PROVIDER_CONCURRENCY.items()
    }
    reviewer_cache: dict[str, SlidingReviewer] = {}
    for judge in judges:
        if dispositions[judge.id][0] == "skipped":
            continue
        try:
            client = clients[judge.id]
        except KeyError as exc:
            raise ValueError(f"missing chat client for judge {judge.id}") from exc
        reviewer_cache[judge.id] = SlidingReviewer(
            session=session,
            judge=judge,
            plan_id=judging_plan.plan_id,
            window_plan=session_row.reviewer(judge.position).window_plan.model_dump(mode="json"),
            context_policy=context_policy,
            client=client,
            load_call=call_loader,
            record_call=call_recorder,
            is_cancelled=cancel_requested,
            activity=activity,
            call_gate=provider_gates.get(judge.provider),
        )

    def reviewer(judge: PositionedJudge) -> SlidingReviewer:
        return reviewer_cache[judge.id]

    scores: list[Score] = []
    failures: list[JudgeFailure] = []
    not_evaluable: list[JudgeNotEvaluable] = []

    evaluated_models = sorted({turn.model for turn in session.turns if turn.model})
    evaluated_families = sorted({model_family(turn.model or "") for turn in session.turns})

    def evaluate(descriptor: RubricDescriptor) -> PanelOutcome:
        _emit_activity(
            activity,
            {
                "phase": "rubric_started",
                "message": f"Reviewing {descriptor.label}",
                "conversation_id": session.conversation_id,
                "rubric": descriptor.id,
            },
        )
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
        )
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
        return outcome

    with ThreadPoolExecutor(
        max_workers=len(rubrics),
        thread_name_prefix="judge-rubric",
    ) as executor:
        outcomes = tuple(executor.map(evaluate, rubrics))

    for descriptor, outcome in zip(rubrics, outcomes, strict=True):
        rubric = _resolve(descriptor)
        attempts = tuple(_attempt_record(value) for value in outcome.attempts)
        if outcome.rating is None:
            if outcome.status == "not_evaluable":
                not_evaluable.append(JudgeNotEvaluable(descriptor.id, attempts))
                continue
            failures.append(
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
            )
            continue
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
                    "raw_coverage_trace_ids": [turn.trace_id for turn in session_row.turns],
                    "plan_id": judging_plan.plan_id,
                    "reviewer_context": [
                        judge.model_dump(mode="json") for judge in judging_plan.reviewers
                    ],
                    "evaluated_models": evaluated_models,
                    "evaluated_families": evaluated_families,
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
    if failures or not_evaluable:
        raise JudgeExecutionError(scores, failures, not_evaluable)
    return scores
