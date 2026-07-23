"""Reflection over one pinned cohort and one captured baseline bundle."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from weave_agent_signals.judges.inference import ChatClient, InferenceCancelled
from weave_agent_signals.patterns import is_evaluation_feedback_eligible
from weave_agent_signals.run_config import EffectiveRunConfig, ModelDescriptor
from weave_agent_signals.runs.bundles import BundleSnapshot
from weave_agent_signals.runs.challenges.contracts import ChallengeResult
from weave_agent_signals.runs.challenges.smol import ChallengeCancelled
from weave_agent_signals.runs.events import sanitize_event
from weave_agent_signals.runs.reflection import (
    NO_IMPROVEMENT_PATIENCE,
    ReflectionCancelled,
    ReflectionEvaluationError,
    ReflectionResult,
    run_reflection,
)
from weave_agent_signals.runs.reflection_records import (
    ReflectionInputRecord,
    ReflectionResultRecord,
)
from weave_agent_signals.runs.stages import StageCancelled
from weave_agent_signals.runs.store import (
    Run,
    RunStatus,
    RunStore,
    RunStoreConflictError,
)

log = logging.getLogger(__name__)


class ReflectionTargetAdapter(Protocol):
    def contract_manifest(self) -> Mapping[str, Any]: ...

    def capture(self) -> BundleSnapshot: ...

    def resolve_locator(
        self,
        locator: str,
        *,
        require_absent_for_create: bool = False,
    ) -> object: ...


ReflectionRunner = Callable[..., ReflectionResult]
ChallengeRunner = Callable[..., ChallengeResult]
FeedbackDigest = Callable[[list[dict[str, Any]]], str]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class ReflectionDependencies:
    store: RunStore
    client_factory: Callable[[], AbstractContextManager[Any]]
    adapter_factory: Callable[[], ReflectionTargetAdapter]
    writer_client_factory: Callable[[ModelDescriptor], AbstractContextManager[ChatClient]]
    evaluator_client_factory: Callable[[ModelDescriptor], AbstractContextManager[ChatClient]]
    coaching_digest: FeedbackDigest
    reflect: ReflectionRunner = run_reflection
    challenge_runner: ChallengeRunner | None = None
    clock: Clock = lambda: datetime.now(timezone.utc)


class _ReflectionProgress:
    """Persist current counters and append bounded activity rows."""

    _COUNTERS = ("attempted", "valid", "rejected", "scored")

    def __init__(
        self,
        store: RunStore,
        run_id: str,
        cancel: threading.Event,
        *,
        total_attempts: int,
        context: Mapping[str, Any],
        initial: Mapping[str, Any] | None,
        clock: Clock,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.cancel = cancel
        self.clock = clock
        self.context = dict(context)
        prior = dict(initial or {})
        self.started_at = str(prior.get("started_at") or clock().isoformat())
        self.counts = {name: int(prior.get(name, 0)) for name in self._COUNTERS}
        self.total_attempts = max(total_attempts, int(prior.get("total_attempts", 0)))

    def record(self, phase: str, message: str, **details: Any) -> None:
        counters = {
            key: details.pop(key) for key in (*self._COUNTERS, "total_attempts") if key in details
        }
        for key in self._COUNTERS:
            value = counters.get(key)
            if type(value) is int and value >= 0:
                self.counts[key] = max(self.counts[key], value)
        value = counters.get("total_attempts")
        if type(value) is int and value >= 0:
            self.total_attempts = max(self.total_attempts, value)
        try:
            draft = sanitize_event("reflecting", phase, message, details, self.clock())
            self.store.append_run_event(self.run_id, draft)
            _record_progress(
                self.store,
                self.run_id,
                self.cancel,
                {
                    **self.context,
                    "phase": draft.phase,
                    "status_message": draft.message,
                    "started_at": self.started_at,
                    **self.counts,
                    "total_attempts": self.total_attempts,
                },
            )
        except StageCancelled:
            raise
        except Exception:
            log.warning("Could not persist reflection progress", exc_info=True)

    def handle(self, event: Mapping[str, Any]) -> None:
        value = dict(event)
        self.record(str(value.pop("phase")), str(value.pop("message")), **value)


class ReflectionStageError(RuntimeError):
    """Safe terminal reflection failure exposed beyond the stage boundary."""


class ReflectionInputMismatchError(ReflectionStageError):
    """Current external feedback no longer matches the pinned reflection input."""


def _terminal_progress_message(error: BaseException) -> str:
    """Describe terminal failure without retaining arbitrary exception text."""

    if isinstance(error, ReflectionInputMismatchError):
        return "Reflection inputs no longer match the pinned run evidence"
    if isinstance(error, ReflectionEvaluationError):
        lowered = str(error).lower()
        if "credits are disabled" in lowered:
            role = "Proposal evaluator" if "evaluator" in lowered else "Reflection model"
            return f"{role} failed: W&B credits are disabled"
        if "timed out" in lowered or "invocation timeout" in lowered:
            return "Reflection model invocation timed out"
        http_status = re.search(r"\bhttp\s+(\d{3})\b", lowered)
        if http_status is not None:
            return f"Reflection model transport returned HTTP {http_status.group(1)}"
        process_exit = re.search(r"\bprocess exited\s+(-?\d+)\b", lowered)
        if process_exit is not None:
            return f"Reflection model process exited {process_exit.group(1)}"
        if "valid rated signal" in lowered:
            return "Reflection requires at least one valid rated evaluation signal"
        return "Reflection evaluation failed"
    if isinstance(error, ValueError):
        return "Reflection inputs or model output were invalid"
    return "Reflection failed unexpectedly"


def _require_active(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
) -> Run:
    if cancel.is_set():
        raise StageCancelled()
    current = store.get(run_id)
    if current is None:
        raise ValueError(f"Run {run_id} not found")
    if current.status is RunStatus.CANCELLED:
        raise StageCancelled()
    if current.status is not RunStatus.REFLECTING:
        raise RunStoreConflictError(
            f"Run {run_id} is no longer reflecting; current status is {current.status.value}"
        )
    return current


def _translate_cancellation(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    conflict: RunStoreConflictError,
) -> None:
    current = store.get(run_id)
    if cancel.is_set() or (current is not None and current.status is RunStatus.CANCELLED):
        raise StageCancelled() from conflict


def _record_progress(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    progress: Mapping[str, Any],
) -> None:
    try:
        store.record_stage_progress(
            run_id,
            stage=RunStatus.REFLECTING,
            progress=progress,
        )
    except RunStoreConflictError as exc:
        _translate_cancellation(store, run_id, cancel, exc)
        raise


def _record_result(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    result: ReflectionResultRecord,
) -> Run:
    try:
        return store.record_stage_result(
            run_id,
            stage=RunStatus.REFLECTING,
            result=result,
        )
    except RunStoreConflictError as exc:
        _translate_cancellation(store, run_id, cancel, exc)
        raise


def _feedback_identity(feedback: Mapping[str, Any]) -> dict[str, Any]:
    feedback_id = feedback.get("id")
    weave_ref = feedback.get("weave_ref")
    feedback_type = feedback.get("feedback_type")
    if not all(
        isinstance(value, str) and value for value in (feedback_id, weave_ref, feedback_type)
    ):
        raise ValueError("Reflection feedback requires id, weave_ref, and feedback_type")
    canonical = json.dumps(
        dict(feedback),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return {
        "id": feedback_id,
        "weave_ref": weave_ref,
        "feedback_type": feedback_type,
        "digest": f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}",
    }


def _cohort_refs(cohort: Mapping[str, Any]) -> list[str]:
    turns = cohort.get("turns")
    sessions = cohort.get("sessions")
    if not isinstance(turns, list) or not isinstance(sessions, list):
        raise ValueError("Pinned cohort has invalid turn or session identities")
    refs: list[str] = []
    for item in [*turns, *sessions]:
        weave_ref = item.get("weave_ref") if isinstance(item, Mapping) else None
        if not isinstance(weave_ref, str) or not weave_ref:
            raise ValueError("Pinned cohort entries require weave_ref")
        refs.append(weave_ref)
    if len(refs) != len(set(refs)):
        raise ValueError("Pinned cohort references must be unique")
    return refs


def _selected_feedback(
    records: list[dict[str, Any]],
    refs: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_refs = set(refs)
    selected = [
        item
        for item in records
        if item.get("weave_ref") in valid_refs and is_evaluation_feedback_eligible(item)
    ]
    identified = [(item, _feedback_identity(item)) for item in selected]
    identified.sort(
        key=lambda pair: (
            pair[1]["weave_ref"],
            pair[1]["feedback_type"],
            pair[1]["id"],
            pair[1]["digest"],
        )
    )
    return (
        [item for item, _identity in identified],
        [identity for _item, identity in identified],
    )


def _target_registry(adapter: ReflectionTargetAdapter) -> dict[str, Any]:
    manifest = adapter.contract_manifest()
    if not isinstance(manifest, Mapping):
        raise ValueError("Target registry manifest must be an object")
    try:
        return json.loads(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Target registry manifest must contain JSON values") from exc


def _require_pinned_registry(
    reflection_input: ReflectionInputRecord,
    target_registry: Mapping[str, Any],
) -> None:
    if reflection_input.target_registry != target_registry:
        raise ReflectionInputMismatchError(
            "Pinned reflection target registry does not match the current registry"
        )


def _new_reflection_input(
    cohort: Mapping[str, Any],
    identities: list[dict[str, Any]],
    baseline: BundleSnapshot,
    target_registry: Mapping[str, Any],
    captured_at: str,
) -> ReflectionInputRecord:
    return ReflectionInputRecord.model_validate(
        {
            "cohort_id": cohort.get("cohort_id"),
            "captured_at": captured_at,
            "feedback": identities,
            "target_registry": dict(target_registry),
            "baseline": baseline,
        }
    )


def _pinned_baseline(
    reflection_input: ReflectionInputRecord,
    cohort: Mapping[str, Any],
    identities: list[dict[str, Any]],
    target_registry: Mapping[str, Any],
) -> BundleSnapshot:
    if reflection_input.cohort_id != cohort.get("cohort_id"):
        raise ReflectionInputMismatchError("Pinned reflection input does not match the run cohort")
    if [item.model_dump(mode="json") for item in reflection_input.feedback] != identities:
        raise ReflectionInputMismatchError(
            "Evaluation feedback changed after the reflection input was pinned"
        )
    _require_pinned_registry(reflection_input, target_registry)
    return reflection_input.baseline


def _set_cancel(client: ChatClient, cancel: threading.Event) -> None:
    setter = getattr(client, "set_cancel", None)
    if callable(setter):
        setter(cancel)


def _finish_without_inference(
    dependencies: ReflectionDependencies,
    run_id: str,
    cancel: threading.Event,
    recorder: _ReflectionProgress,
    baseline: BundleSnapshot,
    *,
    phase: str,
    reason: str,
) -> None:
    _require_active(dependencies.store, run_id, cancel)
    recorder.record(phase, reason)
    _record_result(
        dependencies.store,
        run_id,
        cancel,
        ReflectionResultRecord(
            baseline=baseline,
            attempts=(),
            evaluations=(),
            recommended_candidate_id=None,
            baseline_won=False,
            reason=reason,
        ),
    )


def _finalize_persisted_result(current: Run, store: RunStore) -> bool:
    """Finish review initialization after a crash without rerunning reflection."""

    evidence = current.reflecting_result
    if evidence is None:
        return False
    if current.reflection_review is not None:
        return True
    if not evidence.attempts and evidence.reason is not None:
        return True
    selected = evidence.recommended_candidate_id or evidence.provisional_candidate_id
    if selected is not None:
        store.initialize_reflection_review(
            current.run_id,
            {
                "status": "pending",
                "selected_candidate_id": selected,
                "draft": None,
            },
            expected_revision=current.reflection_review_revision,
        )
    return True


def _execute_reflection_stage(
    run: Run,
    config: EffectiveRunConfig,
    cancel: threading.Event,
    recorder: _ReflectionProgress,
    *,
    dependencies: ReflectionDependencies,
) -> None:
    """Execute reflection after the public stage boundary has been established."""

    current = run
    adapter = dependencies.adapter_factory()
    target_registry = _target_registry(adapter)
    if current.reflection_input is not None:
        _require_pinned_registry(
            current.reflection_input,
            target_registry,
        )
    elif current.reflecting_result is not None:
        raise ReflectionInputMismatchError(
            "Persisted reflection result has no pinned target registry"
        )
    if _finalize_persisted_result(current, dependencies.store):
        return
    cohort = current.turn_cohort
    if not isinstance(cohort, Mapping):
        raise ValueError(f"Run {run.run_id} has no pinned turn cohort")

    refs = _cohort_refs(cohort)
    recorder.record(
        "loading_feedback",
        "Loading evaluation feedback for the pinned cohort",
    )
    with dependencies.client_factory() as client:
        records = client.query_feedback_for_refs(refs)
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise ValueError("Feedback query must return a list of objects")
    feedback, identities = _selected_feedback(records, refs)
    _require_active(dependencies.store, run.run_id, cancel)

    if current.reflection_input is None:
        baseline = adapter.capture()
        if not isinstance(baseline, BundleSnapshot):
            raise ValueError("Target adapter capture must return BundleSnapshot")
        _require_active(dependencies.store, run.run_id, cancel)
        reflection_input = _new_reflection_input(
            cohort,
            identities,
            baseline,
            target_registry,
            dependencies.clock().isoformat(),
        )
        try:
            current = dependencies.store.pin_reflection_input(
                run.run_id,
                reflection_input,
            )
        except RunStoreConflictError as exc:
            _translate_cancellation(dependencies.store, run.run_id, cancel, exc)
            raise
    else:
        baseline = _pinned_baseline(
            current.reflection_input,
            cohort,
            identities,
            target_registry,
        )

    recorder.record(
        "inputs_pinned",
        f"Pinned {len(feedback)} feedback signals and {len(baseline.targets)} targets",
    )
    if not feedback:
        _finish_without_inference(
            dependencies,
            run.run_id,
            cancel,
            recorder,
            baseline,
            phase="no_feedback",
            reason="No evaluation feedback was found for the pinned cohort.",
        )
        return
    if not any(target.exists for target in baseline.targets):
        _finish_without_inference(
            dependencies,
            run.run_id,
            cancel,
            recorder,
            baseline,
            phase="no_targets",
            reason="No managed instruction targets were found.",
        )
        return

    recorder.record(
        "building_digest",
        "Preparing the evaluation evidence for reflection",
    )
    coaching_text = dependencies.coaching_digest(feedback)
    if not isinstance(coaching_text, str):
        raise ValueError("coaching_digest must return a string")
    _require_active(dependencies.store, run.run_id, cancel)
    recorder.record(
        "starting_reflection",
        f"Starting up to {config.candidate_budget} proposal attempts",
    )

    with ExitStack() as stack:
        writer_client = stack.enter_context(
            dependencies.writer_client_factory(config.models.proposal_writer)
        )
        evaluator_client = stack.enter_context(
            dependencies.evaluator_client_factory(config.models.proposal_evaluator)
        )
        _set_cancel(writer_client, cancel)
        _set_cancel(evaluator_client, cancel)
        result = dependencies.reflect(
            baseline=baseline,
            feedback=feedback,
            coaching_text=coaching_text,
            scope_policy=target_registry,
            requested_writer=config.models.proposal_writer,
            requested_evaluator=config.models.proposal_evaluator,
            writer_client=writer_client,
            evaluator_client=evaluator_client,
            resolve_locator=adapter.resolve_locator,
            candidate_budget=config.candidate_budget,
            progress_callback=recorder.handle,
            cancel_requested=cancel.is_set,
        )

        if not isinstance(result, ReflectionResult):
            raise ValueError("Reflection runner must return ReflectionResult")
        if result.provisional_candidate_id is not None:
            if dependencies.challenge_runner is None:
                raise ValueError("paired challenge runner is required for provisional B")
            judge_clients = {}
            for judge in config.models.challenge_judges:
                client = stack.enter_context(dependencies.evaluator_client_factory(judge.model))
                _set_cancel(client, cancel)
                judge_clients[judge.id] = client
            candidate = next(
                item
                for item in result.candidates
                if item.candidate_id == result.provisional_candidate_id
            )
            _require_active(dependencies.store, run.run_id, cancel)
            recorder.record(
                "starting_challenge",
                "Running the provisional bundle against the baseline in paired sandboxes",
            )
            challenge = dependencies.challenge_runner(
                baseline=baseline,
                candidate=candidate,
                coaching_text=coaching_text,
                config=config,
                cohort=cohort,
                adapter=adapter,
                author_client=evaluator_client,
                judge_clients=judge_clients,
                cancel_requested=cancel.is_set,
            )
            if not isinstance(challenge, ChallengeResult):
                raise ValueError("Challenge runner must return ChallengeResult")
            result = result.with_challenge(challenge)

    _require_active(dependencies.store, run.run_id, cancel)
    attempted_count = len(result.generation_attempts)
    valid_count = sum(attempt.status == "succeeded" for attempt in result.generation_attempts)
    rejected_count = sum(attempt.status == "failed" for attempt in result.generation_attempts)
    recorder.record(
        "finalizing_reflection",
        "Persisting evaluated bundle evidence",
        attempted=attempted_count,
        valid=valid_count,
        rejected=rejected_count,
        scored=len(result.candidates),
        total_attempts=config.candidate_budget,
    )
    persisted = _record_result(
        dependencies.store,
        run.run_id,
        cancel,
        ReflectionResultRecord.from_epoch9(result.to_dict()),
    )

    selected = result.recommended_candidate_id or result.provisional_candidate_id
    if selected is not None:
        dependencies.store.initialize_reflection_review(
            run.run_id,
            {
                "status": "pending",
                "selected_candidate_id": selected,
                "draft": None,
            },
            expected_revision=persisted.reflection_review_revision,
        )
    recorder.record(
        "reflection_complete",
        (
            result.reason
            or (
                "Reflection complete; the evaluated baseline remains best"
                if result.baseline_won
                else "Reflection complete; a proposal is ready for review"
            )
        ),
        attempted=attempted_count,
        valid=valid_count,
        rejected=rejected_count,
        scored=len(result.candidates),
        total_attempts=config.candidate_budget,
    )


def _record_terminal_failure(
    recorder: _ReflectionProgress | None,
    error: BaseException,
    message: str,
) -> None:
    if recorder is None:
        return
    try:
        recorder.record(
            "reflection_failed",
            message,
            error_type=type(error).__name__,
            error=message,
        )
    except StageCancelled:
        raise
    except Exception:
        # The original stage failure remains authoritative. A secondary
        # persistence failure must not expose either exception's raw text.
        return


def run_reflection_stage(
    run: Run,
    config: EffectiveRunConfig,
    cancel: threading.Event,
    *,
    dependencies: ReflectionDependencies,
) -> None:
    """Pin exact inputs and expose only safe terminal failures."""

    recorder: _ReflectionProgress | None = None
    try:
        current = _require_active(dependencies.store, run.run_id, cancel)
        recorder = _ReflectionProgress(
            dependencies.store,
            run.run_id,
            cancel,
            total_attempts=config.candidate_budget,
            context={
                "proposal_writer": config.models.proposal_writer.id,
                "proposal_evaluator": config.models.proposal_evaluator.id,
                "no_improvement_patience": NO_IMPROVEMENT_PATIENCE,
            },
            initial=current.reflecting_progress,
            clock=dependencies.clock,
        )
        _execute_reflection_stage(
            current,
            config,
            cancel,
            recorder,
            dependencies=dependencies,
        )
    except StageCancelled:
        raise
    except (ReflectionCancelled, InferenceCancelled, ChallengeCancelled) as exc:
        raise StageCancelled() from exc
    except Exception as exc:
        if isinstance(exc, RunStoreConflictError):
            _translate_cancellation(dependencies.store, run.run_id, cancel, exc)
        message = _terminal_progress_message(exc)
        _record_terminal_failure(recorder, exc, message)
        if isinstance(exc, ReflectionInputMismatchError):
            raise
        raise ReflectionStageError(message) from exc
