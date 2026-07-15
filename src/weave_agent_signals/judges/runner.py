"""Execute pinned rubric reviews against turn and session evidence."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import ValidationError

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.digest import (
    JudgeDigest,
    build_judge_messages,
    build_session_digest,
    build_turn_digest,
    build_turn_digest_with_context,
)
from weave_agent_signals.judges.families import model_family
from weave_agent_signals.judges.inference import ChatClient, InferenceCancelled
from weave_agent_signals.judges.review import (
    AttemptObservation,
    ReviewAttempt,
    ReviewPolicy,
    execute_review,
)
from weave_agent_signals.judges.rubrics import (
    CROSS_TURN_RUBRICS,
    RUBRICS,
    SESSION_RUBRICS,
    Rubric,
)
from weave_agent_signals.judges.verdicts import (
    JUDGE_VERDICT_SCHEMA,
    JUDGE_VERDICT_SCHEMA_VERSION,
    parse_judge_verdict,
)
from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.run_config import PositionedJudge, ReviewDepth, RubricDescriptor

log = logging.getLogger("weave_agent_signals.judges")

REVIEW_POLICY_VERSION = "2"
_MAX_CONCURRENT_RUBRICS = {
    "cli": 4,
    "wandb": 8,
    "openai": 8,
}
_AUDIT_TEXT_LIMIT = 500


@dataclass(frozen=True)
class JudgeFailure:
    rubric: str
    message: str
    error_type: str
    attempts: tuple[dict[str, object], ...] = ()


class JudgeExecutionError(RuntimeError):
    """One or more rubrics had zero successful reviewer attempts."""

    def __init__(
        self,
        scores: Sequence[Score],
        failures: Sequence[JudgeFailure],
    ) -> None:
        self.scores = tuple(scores)
        self.failures = tuple(failures)
        details = "; ".join(f"{failure.rubric} ({failure.message})" for failure in self.failures)
        super().__init__(f"{len(self.failures)} rubric(s) failed: {details}")


class _ReviewFailed(RuntimeError):
    def __init__(self, rubric_id: str, attempts: tuple[dict[str, object], ...]) -> None:
        self.rubric_id = rubric_id
        self.attempts = attempts
        super().__init__(f"No reviewer produced a valid score for {rubric_id}")


def _max_workers(client: ChatClient) -> int:
    return _MAX_CONCURRENT_RUBRICS.get(client.backend, 4)


def _normalized_usage(raw_usage: object) -> dict[str, int]:
    """Keep portable top-level counters from provider-specific usage metadata."""

    if not isinstance(raw_usage, Mapping):
        return {}
    normalized: dict[str, int] = {}
    for key, value in raw_usage.items():
        if isinstance(key, str) and key.strip() and type(value) is int and value >= 0:
            normalized[key] = value
    return normalized


def _audit_text(value: object) -> str:
    """Bound persisted diagnostics and remove control/formatting whitespace."""

    return " ".join(str(value).split())[:_AUDIT_TEXT_LIMIT]


def _sanitized_error(error: Exception) -> str:
    """Describe validation failures without copying their untrusted input values."""

    if isinstance(error, ValidationError):
        details: list[str] = []
        for issue in error.errors(include_url=False, include_input=False):
            location = ".".join(str(part) for part in issue.get("loc", ()))
            message = _audit_text(issue.get("msg", "validation failed"))
            details.append(f"{location}: {message}" if location else message)
        return _audit_text("; ".join(details) or "judge verdict validation failed")
    return _audit_text(error) or type(error).__name__


def _sanitized_transport_error(error: Exception) -> str:
    """Keep transport diagnostics useful without retaining stdout or response bodies."""

    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if type(status_code) is int:
        return f"judge transport returned HTTP {status_code}"
    if type(error).__name__ == "TimeoutExpired":
        return "judge invocation timed out"
    return "judge invocation failed"


def _resolve_rubrics(
    descriptors: Sequence[RubricDescriptor],
    *,
    evaluation_unit: str,
    caller: str,
) -> list[tuple[RubricDescriptor, Rubric]]:
    current_catalog = build_rubric_catalog()
    source = {**RUBRICS, **SESSION_RUBRICS}
    resolved: list[tuple[RubricDescriptor, Rubric]] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, RubricDescriptor):
            raise TypeError("rubrics must contain RubricDescriptor values")
        if descriptor.evaluation_unit != evaluation_unit:
            raise ValueError(f"{caller} requires {evaluation_unit} rubrics")
        try:
            current_descriptor = current_catalog.rubric(descriptor.id)
            rubric = source[descriptor.id]
        except KeyError as error:
            raise ValueError(f"Unknown pinned rubric: {descriptor.id}") from error
        if descriptor != current_descriptor:
            raise ValueError(f"Pinned rubric {descriptor.id} does not match current prompt content")
        resolved.append((descriptor, rubric))
    return resolved


def _attempt_record(attempt: ReviewAttempt) -> dict[str, object]:
    observation = attempt.observation
    resolved_family = (
        model_family(observation.resolved_model) if observation.resolved_model is not None else None
    )
    return {
        "position": attempt.position,
        "role": attempt.role,
        "trigger": attempt.trigger,
        "requested_model": attempt.requested_model,
        "requested_family": attempt.requested_family,
        "requested_backend": attempt.requested_backend,
        "status": observation.status,
        "resolved_model": observation.resolved_model,
        "resolved_family": resolved_family,
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
    }


def _run_reviewed_rubric(
    client: ChatClient,
    *,
    descriptor: RubricDescriptor,
    rubric: Rubric,
    digest: JudgeDigest,
    policy: ReviewPolicy,
    evaluated_models: Sequence[str],
    evaluated_families: Sequence[str],
) -> Score:
    messages = build_judge_messages(
        rubric.system_prompt,
        rubric.criteria_text,
        digest,
        granularity="session" if descriptor.evaluation_unit == "session" else "turn",
    )

    def invoke(judge: PositionedJudge) -> AttemptObservation:
        resolved_model: str | None = None
        usage: dict[str, int] = {}
        output_mode: str | None = None
        schema_name = JUDGE_VERDICT_SCHEMA.name
        schema_fallback_reason: str | None = None
        transport_request_count = 1
        raw_output_digest: str | None = None
        transport_completed = False
        try:
            parsed, response = client.chat_json(
                model=judge.id,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                response_schema=JUDGE_VERDICT_SCHEMA,
            )
            transport_completed = True
            resolved_model = response.model
            usage = _normalized_usage(response.usage)
            output_mode = response.output_mode
            schema_name = response.schema_name or JUDGE_VERDICT_SCHEMA.name
            schema_fallback_reason = (
                _audit_text(response.schema_fallback_reason)
                if response.schema_fallback_reason is not None
                else None
            )
            transport_request_count = response.transport_request_count
            raw_output_digest = response.raw_output_digest
            verdict = parse_judge_verdict(parsed, digest.evidence_ids)
            evidence_ids = tuple(item.id for item in verdict.evidence)
            if verdict.status == "insufficient_evidence":
                return AttemptObservation(
                    status="abstained",
                    resolved_model=resolved_model,
                    score=None,
                    rationale=verdict.rationale,
                    usage=usage,
                    error_type=None,
                    message=None,
                    evidence_ids=evidence_ids,
                    output_mode=output_mode,
                    schema_name=schema_name,
                    schema_fallback_reason=schema_fallback_reason,
                    transport_request_count=transport_request_count,
                    verdict_schema_version=verdict.schema_version,
                    raw_output_digest=raw_output_digest,
                )
            return AttemptObservation(
                status="succeeded",
                resolved_model=resolved_model,
                score=verdict.score,
                rationale=verdict.rationale,
                usage=usage,
                error_type=None,
                message=None,
                evidence_ids=evidence_ids,
                output_mode=output_mode,
                schema_name=schema_name,
                schema_fallback_reason=schema_fallback_reason,
                transport_request_count=transport_request_count,
                verdict_schema_version=verdict.schema_version,
                raw_output_digest=raw_output_digest,
            )
        except InferenceCancelled:
            raise
        except Exception as error:
            if not transport_completed:
                error_request_count = getattr(error, "_transport_request_count", 1)
                if type(error_request_count) is int and error_request_count > 0:
                    transport_request_count = error_request_count
            return AttemptObservation(
                status="failed",
                resolved_model=resolved_model,
                score=None,
                rationale=None,
                usage=usage,
                error_type=type(error).__name__,
                message=(
                    _sanitized_error(error)
                    if transport_completed
                    else _sanitized_transport_error(error)
                ),
                output_mode=output_mode,
                schema_name=schema_name,
                schema_fallback_reason=schema_fallback_reason,
                transport_request_count=transport_request_count,
                verdict_schema_version=JUDGE_VERDICT_SCHEMA_VERSION,
                raw_output_digest=raw_output_digest,
            )

    outcome = execute_review(
        policy,
        threshold=descriptor.pass_threshold,
        invoke=invoke,
    )
    attempts = tuple(_attempt_record(attempt) for attempt in outcome.attempts)
    if outcome.rating is None:
        raise _ReviewFailed(descriptor.id, attempts)

    if outcome.status == "unresolved":
        tags = ["unresolved"]
    elif outcome.rating < descriptor.pass_threshold:
        tags = list(rubric.tags_on_low)
    else:
        tags = list(rubric.tags_on_high)
    rationale = next(
        (
            attempt.observation.rationale
            for attempt in outcome.attempts
            if attempt.observation.status == "succeeded"
            and attempt.observation.rationale is not None
        ),
        "",
    )
    metadata: dict[str, object] = {
        "rubric_id": descriptor.id,
        "rubric_version": descriptor.version,
        "rubric_threshold": descriptor.pass_threshold,
        "evaluation_unit": descriptor.evaluation_unit,
        "review_depth": policy.depth,
        "review_policy_version": REVIEW_POLICY_VERSION,
        "second_opinion_margin": policy.second_opinion_margin,
        "requested_judge_models": [judge.id for judge in policy.judges],
        "aggregate": "mean",
        "review_status": outcome.status,
        "attempt_count": len(outcome.attempts),
        "successful_reviewer_count": outcome.successful_count,
        "attempts": list(attempts),
        "evaluated_models": list(evaluated_models),
        "evaluated_families": list(evaluated_families),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    return Score(
        scorer=descriptor.id,
        value=outcome.rating,
        tags=tags,
        metadata=metadata,
        granularity="session" if descriptor.evaluation_unit == "session" else "turn",
        confidence=None,
        reason=rationale,
    )


def _execute_rubrics(
    client: ChatClient,
    resolved: Sequence[tuple[RubricDescriptor, Rubric]],
    run_one: Callable[[RubricDescriptor, Rubric], Score],
) -> list[Score]:
    scores: dict[int, Score] = {}
    failures: dict[int, JudgeFailure] = {}
    workers = min(_max_workers(client), len(resolved))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, descriptor, rubric): (index, descriptor)
            for index, (descriptor, rubric) in enumerate(resolved)
        }
        for future in as_completed(futures):
            index, descriptor = futures[future]
            try:
                scores[index] = future.result()
            except InferenceCancelled:
                raise
            except _ReviewFailed as error:
                failures[index] = JudgeFailure(
                    rubric=descriptor.id,
                    message=str(error),
                    error_type="ReviewFailed",
                    attempts=error.attempts,
                )
                outcomes = ", ".join(
                    f"{attempt.get('requested_model', 'unknown')}="
                    f"{attempt.get('status', 'unknown')}"
                    + (
                        f"({attempt['error_type']})"
                        if attempt.get("error_type") is not None
                        else ""
                    )
                    for attempt in error.attempts
                )
                log.warning(
                    "No reviewer produced a valid score for rubric %s; attempts=%d outcomes=%s",
                    descriptor.id,
                    len(error.attempts),
                    outcomes,
                )

    ordered_scores = [scores[index] for index in sorted(scores)]
    if failures:
        ordered_failures = [failures[index] for index in sorted(failures)]
        raise JudgeExecutionError(ordered_scores, ordered_failures)
    return ordered_scores


def judge_turn(
    turn: TurnSpan,
    client: ChatClient,
    *,
    rubrics: Sequence[RubricDescriptor],
    judges: Sequence[PositionedJudge],
    review_depth: ReviewDepth,
    second_opinion_margin: float | None,
    prior_turns: Sequence[TurnSpan] = (),
) -> list[Score]:
    """Judge pinned episode rubrics with the exact configured reviewers."""

    if not rubrics:
        return []
    resolved = _resolve_rubrics(
        rubrics,
        evaluation_unit="episode",
        caller="judge_turn",
    )
    policy = ReviewPolicy(
        depth=review_depth,
        judges=tuple(judges),
        second_opinion_margin=second_opinion_margin,
    )
    digest = build_turn_digest(turn)
    context_digest = (
        build_turn_digest_with_context(turn, list(prior_turns), window=len(prior_turns))
        if prior_turns
        else digest
    )
    evaluated_models = [turn.model] if turn.model else []
    evaluated_families = [model_family(turn.model or "")]

    def run_one(descriptor: RubricDescriptor, rubric: Rubric) -> Score:
        evidence = context_digest if descriptor.id in CROSS_TURN_RUBRICS else digest
        return _run_reviewed_rubric(
            client,
            descriptor=descriptor,
            rubric=rubric,
            digest=evidence,
            policy=policy,
            evaluated_models=evaluated_models,
            evaluated_families=evaluated_families,
        )

    return _execute_rubrics(client, resolved, run_one)


def judge_session(
    session: SessionView,
    client: ChatClient,
    *,
    rubrics: Sequence[RubricDescriptor],
    judges: Sequence[PositionedJudge],
    review_depth: ReviewDepth,
    second_opinion_margin: float | None,
    evidence_trace_ids: Sequence[str],
) -> list[Score]:
    """Judge pinned session rubrics with the exact configured reviewers."""

    if not rubrics:
        return []
    resolved = _resolve_rubrics(
        rubrics,
        evaluation_unit="session",
        caller="judge_session",
    )
    policy = ReviewPolicy(
        depth=review_depth,
        judges=tuple(judges),
        second_opinion_margin=second_opinion_margin,
    )
    digest = build_session_digest(
        session,
        evidence_trace_ids=list(evidence_trace_ids),
    )
    evaluated_models = sorted({turn.model for turn in session.turns if turn.model})
    evaluated_families = sorted({model_family(turn.model or "") for turn in session.turns})

    def run_one(descriptor: RubricDescriptor, rubric: Rubric) -> Score:
        return _run_reviewed_rubric(
            client,
            descriptor=descriptor,
            rubric=rubric,
            digest=digest,
            policy=policy,
            evaluated_models=evaluated_models,
            evaluated_families=evaluated_families,
        )

    return _execute_rubrics(client, resolved, run_one)
