"""Reflection over one pinned cohort and one captured baseline bundle."""

from __future__ import annotations

import hashlib
import json
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
from weave_agent_signals.runs.progress import ReflectionProgressRecorder
from weave_agent_signals.runs.reflection import (
    NO_IMPROVEMENT_PATIENCE,
    ReflectionCancelled,
    ReflectionEvaluationError,
    ReflectionResult,
    run_reflection,
)
from weave_agent_signals.runs.stages import StageCancelled
from weave_agent_signals.runs.store import (
    Run,
    RunStatus,
    RunStore,
    RunStoreConflictError,
)


class ReflectionTargetAdapter(Protocol):
    def contract_manifest(self) -> Mapping[str, Any]: ...

    def capture(self) -> BundleSnapshot: ...

    def bundle_from_content_map(
        self,
        contents: Mapping[str, str],
    ) -> BundleSnapshot: ...


ReflectionRunner = Callable[..., ReflectionResult]
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
    clock: Clock = lambda: datetime.now(timezone.utc)


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
    result: Mapping[str, Any],
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


def _target_adapter_contract(adapter: ReflectionTargetAdapter) -> dict[str, Any]:
    manifest = adapter.contract_manifest()
    if not isinstance(manifest, Mapping):
        raise ValueError("Target adapter contract manifest must be an object")
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
        raise ValueError("Target adapter contract manifest must contain JSON values") from exc


def _require_pinned_adapter_contract(
    reflection_input: Mapping[str, Any],
    target_adapter_contract: Mapping[str, Any],
) -> None:
    if (
        reflection_input.get("schema_version") != "2"
        or reflection_input.get("target_adapter_contract") != target_adapter_contract
    ):
        raise ReflectionInputMismatchError(
            "Pinned reflection target adapter contract does not match the current adapter contract"
        )


def _new_reflection_input(
    cohort: Mapping[str, Any],
    identities: list[dict[str, Any]],
    baseline: BundleSnapshot,
    target_adapter_contract: Mapping[str, Any],
    captured_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": "2",
        "cohort_id": cohort.get("cohort_id"),
        "captured_at": captured_at,
        "turn_count": cohort.get("turn_count"),
        "session_count": cohort.get("session_count"),
        "feedback_count": len(identities),
        "feedback": identities,
        "target_adapter_contract": dict(target_adapter_contract),
        "baseline": baseline.to_dict(),
    }


def _pinned_baseline(
    reflection_input: Mapping[str, Any],
    cohort: Mapping[str, Any],
    identities: list[dict[str, Any]],
    target_adapter_contract: Mapping[str, Any],
) -> BundleSnapshot:
    expected_keys = {
        "schema_version",
        "cohort_id",
        "captured_at",
        "turn_count",
        "session_count",
        "feedback_count",
        "feedback",
        "target_adapter_contract",
        "baseline",
    }
    if set(reflection_input) != expected_keys:
        raise ReflectionInputMismatchError("Pinned reflection input has an invalid shape")
    if reflection_input.get("schema_version") != "2" or any(
        reflection_input.get(key) != cohort.get(key)
        for key in ("cohort_id", "turn_count", "session_count")
    ):
        raise ReflectionInputMismatchError("Pinned reflection input does not match the run cohort")
    if reflection_input.get("feedback_count") != len(identities) or (
        reflection_input.get("feedback") != identities
    ):
        raise ReflectionInputMismatchError(
            "Evaluation feedback changed after the reflection input was pinned"
        )
    _require_pinned_adapter_contract(reflection_input, target_adapter_contract)
    raw_baseline = reflection_input.get("baseline")
    if not isinstance(raw_baseline, Mapping):
        raise ReflectionInputMismatchError("Pinned reflection baseline is invalid")
    try:
        return BundleSnapshot.from_dict(raw_baseline)
    except ValueError as exc:
        raise ReflectionInputMismatchError("Pinned reflection baseline is invalid") from exc


def _set_cancel(client: ChatClient, cancel: threading.Event) -> None:
    setter = getattr(client, "set_cancel", None)
    if callable(setter):
        setter(cancel)


def _finish_without_inference(
    dependencies: ReflectionDependencies,
    run_id: str,
    cancel: threading.Event,
    recorder: ReflectionProgressRecorder,
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
        {"candidates": [], "reason": reason},
    )


def _finalize_persisted_result(current: Run, store: RunStore) -> bool:
    """Finish review initialization after a crash without rerunning reflection."""

    evidence = current.reflecting_result
    if evidence is None:
        return False
    if current.reflection_review is not None:
        return True
    if set(evidence) == {"candidates", "reason"}:
        if evidence.get("candidates") != [] or not isinstance(evidence.get("reason"), str):
            raise ValueError("Persisted empty reflection result has an invalid shape")
        return True

    result = ReflectionResult.from_dict(evidence)
    selected = result.recommended_candidate_id
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
    recorder: ReflectionProgressRecorder,
    *,
    dependencies: ReflectionDependencies,
) -> None:
    """Execute reflection after the public stage boundary has been established."""

    current = run
    adapter = dependencies.adapter_factory()
    target_adapter_contract = _target_adapter_contract(adapter)
    if current.reflection_input is not None:
        _require_pinned_adapter_contract(
            current.reflection_input,
            target_adapter_contract,
        )
    elif current.reflecting_result is not None:
        raise ReflectionInputMismatchError(
            "Persisted reflection result has no pinned target adapter contract"
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
            target_adapter_contract,
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
            target_adapter_contract,
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
            scope_policy=target_adapter_contract,
            requested_writer=config.models.proposal_writer,
            requested_evaluator=config.models.proposal_evaluator,
            writer_client=writer_client,
            evaluator_client=evaluator_client,
            build_candidate=adapter.bundle_from_content_map,
            candidate_budget=config.candidate_budget,
            progress_callback=recorder.handle,
            cancel_requested=cancel.is_set,
        )

    if not isinstance(result, ReflectionResult):
        raise ValueError("Reflection runner must return ReflectionResult")
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
        result.to_dict(),
    )

    if result.recommended_candidate_id is not None:
        dependencies.store.initialize_reflection_review(
            run.run_id,
            {
                "status": "pending",
                "selected_candidate_id": result.recommended_candidate_id,
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
    recorder: ReflectionProgressRecorder | None,
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

    recorder: ReflectionProgressRecorder | None = None
    try:
        current = _require_active(dependencies.store, run.run_id, cancel)
        recorder = ReflectionProgressRecorder(
            lambda snapshot: _record_progress(
                dependencies.store,
                run.run_id,
                cancel,
                snapshot,
            ),
            total_attempts=config.candidate_budget,
            context={
                "proposal_writer": config.models.proposal_writer.id,
                "proposal_evaluator": config.models.proposal_evaluator.id,
                "no_improvement_patience": NO_IMPROVEMENT_PATIENCE,
            },
            initial_snapshot=current.reflecting_progress,
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
    except (ReflectionCancelled, InferenceCancelled) as exc:
        raise StageCancelled() from exc
    except Exception as exc:
        if isinstance(exc, RunStoreConflictError):
            _translate_cancellation(dependencies.store, run.run_id, cancel, exc)
        message = _terminal_progress_message(exc)
        _record_terminal_failure(recorder, exc, message)
        if isinstance(exc, ReflectionInputMismatchError):
            raise
        raise ReflectionStageError(message) from exc
