"""Execute a pinned model-judging plan for one immutable run cohort."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from weave_agent_signals.client import WeaveClient
from weave_agent_signals.judges.inference import ChatClient
from weave_agent_signals.judges.plan import build_judging_plan
from weave_agent_signals.judges.runner import (
    JudgeExecutionError,
    JudgeFailure,
    judge_session,
)
from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.run_config import EffectiveRunConfig, RubricDescriptor
from weave_agent_signals.runs.stages import StageCancelled
from weave_agent_signals.runs.store import (
    Run,
    RunStatus,
    RunStore,
    RunStoreConflictError,
)

log = logging.getLogger(__name__)

_MAX_PERSISTED_OUTCOMES = 100
_MAX_USAGE_FIELDS = 20
_MAX_EVIDENCE_IDS = 100
_SUMMARY_TEXT_LIMIT = 500

_Target = SessionView


@dataclass(frozen=True)
class JudgingDependencies:
    store: RunStore
    client_factory: Callable[[], WeaveClient]
    chat_client_factory: Callable[[], AbstractContextManager[ChatClient]]
    hydrate_cohort: Callable[
        [Mapping[str, Any]],
        tuple[list[TurnSpan], dict[str, SessionView]],
    ]


@dataclass
class _State:
    planned_rubrics: int
    minimum_reviewer_attempts: int
    maximum_reviewer_attempts: int
    maximum_digest_steps: int
    maximum_window_steps: int
    maximum_merge_steps: int
    rubrics_completed: int = 0
    reviewer_attempts_completed: int = 0
    digest_steps_completed: int = 0
    window_steps_completed: int = 0
    merge_steps_completed: int = 0
    scores_written: int = 0
    write_failure_count: int = 0
    scores: list[Score] = field(default_factory=list)
    failures: list[JudgeFailure] = field(default_factory=list)
    pending: list[tuple[Score, _Target]] = field(default_factory=list)
    attempt_summaries: list[dict[str, Any]] = field(default_factory=list)
    attempt_summary_count: int = 0
    failure_details: list[dict[str, Any]] = field(default_factory=list)
    failure_detail_count: int = 0
    completed_artifact_ids: set[str] = field(default_factory=set)

    def payload(
        self,
        plan_id: str,
        message: str,
        *,
        coverage_complete: bool,
    ) -> dict[str, Any]:
        return {
            "plan_id": plan_id,
            "planned_rubrics": self.planned_rubrics,
            "rubrics_completed": self.rubrics_completed,
            "rated_rubrics": len(self.scores),
            "minimum_reviewer_attempts": self.minimum_reviewer_attempts,
            "maximum_reviewer_attempts": self.maximum_reviewer_attempts,
            "reviewer_attempts_completed": self.reviewer_attempts_completed,
            "digest_steps_completed": self.digest_steps_completed,
            "maximum_digest_steps": self.maximum_digest_steps,
            "window_steps_completed": self.window_steps_completed,
            "maximum_window_steps": self.maximum_window_steps,
            "merge_steps_completed": self.merge_steps_completed,
            "maximum_merge_steps": self.maximum_merge_steps,
            "scores_written": self.scores_written,
            "failure_count": len(self.failures),
            "write_failure_count": self.write_failure_count,
            "coverage_complete": coverage_complete,
            "status_message": message,
            "attempt_summary_count": self.attempt_summary_count,
            "attempt_summaries_truncated": (
                self.attempt_summary_count > len(self.attempt_summaries)
            ),
            "attempt_summaries": self.attempt_summaries,
            "failure_detail_count": self.failure_detail_count,
            "failure_details_truncated": (self.failure_detail_count > len(self.failure_details)),
            "failure_details": self.failure_details,
        }


def _active(store: RunStore, run_id: str, cancel: threading.Event) -> Run:
    if cancel.is_set():
        raise StageCancelled()
    current = store.get(run_id)
    if current is None:
        raise ValueError(f"Run {run_id} not found")
    if current.status is RunStatus.CANCELLED:
        raise StageCancelled()
    if current.status is not RunStatus.JUDGING:
        raise RunStoreConflictError(
            f"Run {run_id} is no longer judging; current status is {current.status.value}"
        )
    return current


def _translate_cancel(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    error: RunStoreConflictError,
) -> None:
    current = store.get(run_id)
    if cancel.is_set() or (current and current.status is RunStatus.CANCELLED):
        raise StageCancelled() from error


def _persist(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    value: Mapping[str, Any],
    *,
    result: bool = False,
) -> None:
    try:
        if result:
            store.record_stage_result(run_id, stage=RunStatus.JUDGING, result=value)
        else:
            store.record_stage_progress(run_id, stage=RunStatus.JUDGING, progress=value)
    except RunStoreConflictError as error:
        _translate_cancel(store, run_id, cancel, error)
        raise


def _text(value: object) -> str | None:
    return None if value is None else str(value)[:_SUMMARY_TEXT_LIMIT]


def _usage(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    bounded: dict[str, object] = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))[:_MAX_USAGE_FIELDS]:
        bounded[str(key)[:80]] = (
            item if isinstance(item, (int, float, bool)) or item is None else _text(item)
        )
    return bounded


def _evidence_ids(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item)[:_SUMMARY_TEXT_LIMIT] for item in value[:_MAX_EVIDENCE_IDS]]


def _attempts(values: object) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    records: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            records.append({"status": "invalid", "message": _text(value)})
            continue
        record = {
            key: value.get(key)
            for key in (
                "position",
                "role",
                "trigger",
                "requested_model",
                "requested_family",
                "requested_backend",
                "status",
                "resolved_model",
                "resolved_family",
                "score",
                "transport_request_count",
                "verdict_schema_version",
            )
        }
        record.update(
            usage=_usage(value.get("usage")),
            rationale=_text(value.get("rationale")),
            evidence_ids=_evidence_ids(value.get("evidence_ids")),
            output_mode=_text(value.get("output_mode")),
            schema_name=_text(value.get("schema_name")),
            schema_fallback_reason=_text(value.get("schema_fallback_reason")),
            raw_output_digest=_text(value.get("raw_output_digest")),
            error_type=_text(value.get("error_type")),
            message=_text(value.get("message")),
            behavioral_feedback=(
                dict(value["behavioral_feedback"])
                if isinstance(value.get("behavioral_feedback"), Mapping)
                else None
            ),
            steps=[dict(step) for step in value.get("steps", []) if isinstance(step, Mapping)],
        )
        records.append(record)
    return records


def _scope(trace_id: str | None, conversation_id: str) -> dict[str, str]:
    result = {"conversation_id": conversation_id}
    if trace_id:
        result["trace_id"] = trace_id
    return result


def _append_bounded(items: list[dict[str, Any]], value: dict[str, Any]) -> None:
    if len(items) < _MAX_PERSISTED_OUTCOMES:
        items.append(value)


def _record_score(
    state: _State,
    score: Score,
    *,
    unit: str,
    trace_id: str | None,
    conversation_id: str,
) -> None:
    attempts = _attempts(score.metadata.get("attempts"))
    _record_steps(state, score.metadata.get("attempts", []))
    _record_score_summary(
        state,
        score,
        attempts,
        unit=unit,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )


def _record_steps(state: _State, attempts: object) -> None:
    if not isinstance(attempts, (list, tuple)):
        return
    for attempt in attempts:
        if isinstance(attempt, Mapping):
            for step in attempt.get("steps", []):
                if isinstance(step, Mapping):
                    phase = step.get("phase")
                    artifact_id = step.get("artifact_id")
                    if (
                        not isinstance(artifact_id, str)
                        or artifact_id in state.completed_artifact_ids
                    ):
                        continue
                    state.completed_artifact_ids.add(artifact_id)
                    if phase == "digest":
                        state.digest_steps_completed += 1
                    elif phase == "window":
                        state.window_steps_completed += 1
                    elif phase == "merge":
                        state.merge_steps_completed += 1


def _record_score_summary(
    state: _State,
    score: Score,
    attempts: list[dict[str, Any]],
    *,
    unit: str,
    trace_id: str | None,
    conversation_id: str,
) -> None:
    state.scores.append(score)
    state.reviewer_attempts_completed += len(attempts)
    state.attempt_summary_count += 1
    _append_bounded(
        state.attempt_summaries,
        {
            "scope": unit,
            "rubric": score.scorer,
            "review_status": score.metadata.get("review_status", "complete"),
            "rating": float(score.value),
            "attempt_count": len(attempts),
            "successful_reviewer_count": score.metadata.get("successful_reviewer_count"),
            "attempts": attempts,
            **_scope(trace_id, conversation_id),
        },
    )


def _record_failure(
    state: _State,
    failure: JudgeFailure,
    *,
    unit: str,
    trace_id: str | None,
    conversation_id: str,
) -> None:
    attempts = _attempts(failure.attempts)
    _record_steps(state, failure.attempts)
    state.failures.append(failure)
    state.reviewer_attempts_completed += len(attempts)
    state.attempt_summary_count += 1
    context = _scope(trace_id, conversation_id)
    _append_bounded(
        state.attempt_summaries,
        {
            "scope": unit,
            "rubric": failure.rubric,
            "review_status": "failed",
            "rating": None,
            "attempt_count": len(attempts),
            "successful_reviewer_count": 0,
            "attempts": attempts,
            **context,
        },
    )
    state.failure_detail_count += 1
    # A failed rubric has no feedback record, so this is its authoritative
    # durable attempt audit and must not be subject to the display-summary cap.
    state.failure_details.append(
        {
            "scope": unit,
            "rubric": failure.rubric,
            "error_type": failure.error_type,
            "message": _text(failure.message),
            "attempt_count": len(attempts),
            "attempts": attempts,
            **context,
        }
    )


def _consume(
    state: _State,
    expected: Sequence[RubricDescriptor],
    scores: Sequence[Score],
    failures: Sequence[JudgeFailure],
    *,
    unit: str,
    trace_id: str | None,
    conversation_id: str,
) -> list[Score]:
    score_groups: dict[str, list[Score]] = {}
    failure_groups: dict[str, list[JudgeFailure]] = {}
    for score in scores:
        score_groups.setdefault(score.scorer, []).append(score)
    for failure in failures:
        failure_groups.setdefault(failure.rubric, []).append(failure)

    accepted: list[Score] = []
    expected_ids = {descriptor.id for descriptor in expected}
    for descriptor in expected:
        rubric_scores = score_groups.get(descriptor.id, [])
        rubric_failures = failure_groups.get(descriptor.id, [])
        if rubric_failures:
            _record_failure(
                state,
                rubric_failures[0],
                unit=unit,
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
        elif len(rubric_scores) == 1:
            score = rubric_scores[0]
            accepted.append(score)
            _record_score(
                state,
                score,
                unit=unit,
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
        else:
            duplicate = bool(rubric_scores)
            _record_failure(
                state,
                JudgeFailure(
                    rubric=descriptor.id,
                    message=f"Judge returned {'duplicate scores' if duplicate else 'no score'} "
                    f"for planned rubric {descriptor.id}",
                    error_type=("DuplicateJudgeScore" if duplicate else "MissingJudgeScore"),
                ),
                unit=unit,
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
    for rubric_id in sorted((set(score_groups) | set(failure_groups)) - expected_ids):
        _record_failure(
            state,
            JudgeFailure(
                rubric=rubric_id,
                message=f"Judge returned unplanned rubric {rubric_id}",
                error_type="UnexpectedJudgeOutcome",
            ),
            unit=unit,
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
    state.rubrics_completed += len(expected)
    return accepted


def _run_unit(
    state: _State,
    expected: Sequence[RubricDescriptor],
    invoke: Callable[[], list[Score]],
    *,
    unit: str,
    trace_id: str | None,
    conversation_id: str,
) -> list[Score]:
    try:
        scores, failures = invoke(), ()
    except JudgeExecutionError as error:
        scores, failures = list(error.scores), error.failures
    return _consume(
        state,
        expected,
        scores,
        failures,
        unit=unit,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )


def _applicable(
    rows: Sequence[Mapping[str, Any]],
    descriptors: Mapping[str, RubricDescriptor],
) -> list[RubricDescriptor]:
    try:
        return [descriptors[row["id"]] for row in rows]
    except KeyError as error:
        raise RuntimeError("Pinned judging plan does not match configured rubrics") from error


def _ref(target: _Target, client: WeaveClient) -> str:
    return target.ref_for(client.entity, client.project)


def _stamp(
    score: Score,
    target: _Target,
    *,
    plan_id: str,
    evidence_trace_ids: Sequence[str],
) -> None:
    run_time = target.turns[0].started_at if target.turns else None
    score.stamp(
        config_version=target.config_version,
        git_branch=target.git_branch,
        run_time=run_time,
    )
    score.metadata.update(
        plan_id=plan_id,
        evaluation_unit="session",
        evidence_trace_ids=list(evidence_trace_ids),
    )


def run_judging_stage(
    run: Run,
    config: EffectiveRunConfig,
    cancel: threading.Event,
    *,
    dependencies: JudgingDependencies,
) -> None:
    """Build/reuse the plan, rate every applicable rubric, then write once."""

    run_id = run.run_id
    current = _active(dependencies.store, run_id, cancel)
    if current.turn_cohort is None:
        raise ValueError(f"Run {run_id} has no pinned turn cohort")
    turns, sessions = dependencies.hydrate_cohort(current.turn_cohort)
    _active(dependencies.store, run_id, cancel)

    built_plan = build_judging_plan(
        list(sessions.values()),
        cohort_id=current.turn_cohort["cohort_id"],
        rubrics=config.rubrics,
        review_depth=config.review_depth,
        judge_models=config.models.judges,
        context_policy=config.judging_context,
    )
    try:
        current = dependencies.store.pin_judging_plan(run_id, built_plan)
    except RunStoreConflictError as error:
        _translate_cancel(dependencies.store, run_id, cancel, error)
        raise
    plan = current.judging_plan
    if plan is None:  # pragma: no cover - pin_judging_plan just returned it
        raise RuntimeError("Judging plan was not pinned")
    _active(dependencies.store, run_id, cancel)

    descriptors = {descriptor.id: descriptor for descriptor in config.rubrics}
    totals = plan["totals"]
    state = _State(
        planned_rubrics=totals["planned_rubrics"],
        minimum_reviewer_attempts=totals["minimum_reviewer_attempts"],
        maximum_reviewer_attempts=totals["maximum_reviewer_attempts"],
        maximum_digest_steps=totals["maximum_digest_calls"],
        maximum_window_steps=totals["maximum_window_calls"],
        maximum_merge_steps=totals["maximum_merge_calls"],
    )

    def progress(message: str, *, coverage_complete: bool = False) -> None:
        _persist(
            dependencies.store,
            run_id,
            cancel,
            state.payload(
                plan["plan_id"],
                message,
                coverage_complete=coverage_complete,
            ),
        )

    progress(f"Starting {state.planned_rubrics} planned rubric judgment(s)...")
    chat_context: AbstractContextManager[ChatClient | None] = (
        dependencies.chat_client_factory() if state.planned_rubrics else nullcontext(None)
    )
    _active(dependencies.store, run_id, cancel)
    with chat_context as chat_client:
        if chat_client is not None:
            set_cancel = getattr(chat_client, "set_cancel", None)
            if callable(set_cancel):
                set_cancel(cancel)
        for session_plan in plan["sessions"]:
            conversation_id = session_plan["conversation_id"]
            session = sessions.get(conversation_id)
            if session is None:
                raise RuntimeError(f"Pinned judging session is missing: {conversation_id}")
            expected = _applicable(session_plan["rubrics"], descriptors)
            if not expected:
                continue
            _active(dependencies.store, run_id, cancel)
            evidence_ids = list(session_plan["raw_coverage_trace_ids"])

            def load_artifact(artifact_id: str) -> Mapping[str, Any] | None:
                active = _active(dependencies.store, run_id, cancel)
                return (active.judging_artifacts or {}).get(artifact_id)

            def record_artifact(artifact_id: str, artifact: Mapping[str, Any]) -> object:
                return dependencies.store.record_judging_artifact(run_id, artifact_id, artifact)

            accepted = _run_unit(
                state,
                expected,
                lambda: judge_session(
                    session,
                    chat_client,
                    rubrics=expected,
                    judges=config.models.judges,
                    review_depth=config.review_depth,
                    second_opinion_margin=config.second_opinion_margin,
                    judging_plan=plan,
                    context_policy=config.judging_context,
                    artifact_loader=load_artifact,
                    artifact_recorder=record_artifact,
                    cancel_requested=lambda: cancel.is_set(),
                ),
                unit="session",
                trace_id=None,
                conversation_id=conversation_id,
            )
            for score in accepted:
                _stamp(
                    score,
                    session,
                    plan_id=plan["plan_id"],
                    evidence_trace_ids=evidence_ids,
                )
                state.pending.append((score, session))
            progress(f"Judged {state.rubrics_completed} of {state.planned_rubrics} planned rubrics")

    _active(dependencies.store, run_id, cancel)
    if state.failures:
        failed = state.payload(
            plan["plan_id"],
            "Judging coverage incomplete",
            coverage_complete=False,
        )
        _persist(dependencies.store, run_id, cancel, failed)
        _persist(dependencies.store, run_id, cancel, failed, result=True)
        raise JudgeExecutionError(state.scores, state.failures)

    if state.pending:
        with dependencies.client_factory() as client:
            refs = list(dict.fromkeys(_ref(target, client) for _, target in state.pending))
            existing = client.query_existing_feedback_batch(refs)
            _active(dependencies.store, run_id, cancel)
            writes: list[tuple[Score, str, list[dict]]] = []
            for score, target in state.pending:
                ref = _ref(target, client)
                prior = existing.get((ref, f"weave_agent_signals.{score.scorer}"), [])
                if prior and not config.force:
                    continue
                writes.append((score, ref, prior))
            if writes:
                progress(f"Writing {len(writes)} judge score(s)...", coverage_complete=True)
                _active(dependencies.store, run_id, cancel)
                try:
                    with dependencies.store.external_write_barrier(run_id, RunStatus.JUDGING):
                        for score, ref, prior in writes:
                            try:
                                client.write_score(score, ref)
                                state.scores_written += 1
                            except Exception as error:
                                state.write_failure_count += 1
                                log.warning("Run %s judge score write failed: %s", run_id, error)
                                continue
                            try:
                                if prior:
                                    client.delete_feedback_ids(prior)
                            except Exception as error:
                                state.write_failure_count += 1
                                log.warning(
                                    "Run %s prior judge feedback cleanup failed: %s",
                                    run_id,
                                    error,
                                )
                except RunStoreConflictError as error:
                    _translate_cancel(dependencies.store, run_id, cancel, error)
                    raise

    _active(dependencies.store, run_id, cancel)
    message = (
        "Judging complete" if not state.write_failure_count else "Judge feedback write incomplete"
    )
    final = state.payload(plan["plan_id"], message, coverage_complete=True)
    _persist(dependencies.store, run_id, cancel, final)
    _persist(dependencies.store, run_id, cancel, final, result=True)
    if state.write_failure_count:
        raise RuntimeError(
            f"Judge feedback write incomplete: {state.write_failure_count} failure(s)"
        )
