from __future__ import annotations

import copy
import threading
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.patterns import coaching_digest
from weave_agent_signals.run_config import RunConfig, resolve_run_config
from weave_agent_signals.runs.bundles import ScopeDescriptor, bundle_from_content_map
from weave_agent_signals.runs.reflection import (
    NO_VALID_PROPOSAL_REASON,
    EvaluatorRecord,
    GenerationAttempt,
    ReflectionCancelled,
    ReflectionCandidate,
    ReflectionEvaluationError,
    ReflectionResult,
)
from weave_agent_signals.runs.stages import StageCancelled
from weave_agent_signals.runs.stages.reflection import (
    ReflectionDependencies,
    ReflectionInputMismatchError,
    ReflectionStageError,
    run_reflection_stage,
)
from weave_agent_signals.runs.store import DataSelection, RunStatus, RunStore

SCOPE = ScopeDescriptor("file", "/project", ("CLAUDE.md", ".claude/**/*.md"))


def _manifest(digest: str = "sha256:test-registry") -> dict[str, object]:
    return {
        "schema_version": "1",
        "targets": [{"kind": "file", "id": "agents"}],
        "digest": digest,
    }


@pytest.fixture
def store(tmp_path):
    instance = RunStore(tmp_path / "runs.db")
    yield instance
    instance.close()


def _cohort() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "pinned_at": "2026-07-14T00:00:00+00:00",
        "cohort_id": "cohort-test",
        "turn_count": 1,
        "session_count": 1,
        "turns": [
            {
                "trace_id": "turn-1",
                "weave_ref": "weave:///turn-1",
                "conversation_id": "conv-1",
                "started_at": "2026-07-14T00:00:00+00:00",
                "model": "gpt-5.6-sol",
                "model_family": "openai",
            }
        ],
        "sessions": [
            {
                "conversation_id": "conv-1",
                "weave_ref": "weave:///session-1",
                "turn_count": 1,
            }
        ],
    }


def _reflecting_run(store: RunStore, *, candidate_budget: int = 2):
    models = build_model_catalog(which=lambda name: f"/bin/{name}")
    rubrics = build_rubric_catalog()
    requested = RunConfig(
        model_catalog_version=models.catalog_version,
        rubric_catalog_version=rubrics.catalog_version,
        judge_backend="cli",
        judge_models=("claude-sonnet-5", "gpt-5.6-sol"),
        proposal_model="gpt-5.6-sol",
        proposal_evaluator_model="claude-sonnet-5",
        rubrics=(rubrics.rubrics[0].id,),
        candidate_budget=candidate_budget,
        force=False,
    )
    effective = resolve_run_config(
        requested,
        model_catalog=models,
        rubric_catalog=rubrics,
    )
    selection = DataSelection(session_ids=("conv-1",))
    created = store.create()
    store.save_selection(created.run_id, selection)
    store.save_config(created.run_id, requested)
    started = store.start(
        created.run_id,
        expected_selection=selection,
        expected_config=requested,
        turn_cohort=_cohort(),
        effective_config=effective,
    )
    store.record_stage_result(
        started.run_id,
        stage=RunStatus.SCORING,
        result={"written": 1},
    )
    store.finalize_stage_success(
        started.run_id,
        stage=RunStatus.SCORING,
        advance=True,
    )
    store.record_stage_result(
        started.run_id,
        stage=RunStatus.JUDGING,
        result={"written": 1},
    )
    reflecting = store.finalize_stage_success(
        started.run_id,
        stage=RunStatus.JUDGING,
        advance=True,
    )
    return reflecting, effective


def _feedback() -> dict[str, Any]:
    return {
        "id": "feedback-1",
        "weave_ref": "weave:///turn-1",
        "feedback_type": "weave_agent_signals.outcome.test",
        "payload": {"rating": 0.3, "details": {"rationale": "No tests."}},
    }


def _session_feedback() -> dict[str, Any]:
    feedback = _feedback()
    feedback["weave_ref"] = "weave:///session-1"
    feedback["feedback_type"] = "weave_agent_signals.judge.verification"
    feedback["payload"] = {
        "rating": 0.3,
        "reason": "Problem: No tests. | Next: Verify before completion.",
        "granularity": "session",
        "details": {
            "evaluation_unit": "session",
            "review_status": "complete",
            "rubric_version": "4",
            "rubric_threshold": 0.5,
            "panel_contract_version": "1",
            "requested_judge_models": ["judge-a", "judge-b"],
            "panel_size": 2,
            "turn_started_at": "2026-07-14T00:00:00+00:00",
            "behavioral_feedback": [
                {
                    "success": None,
                    "problem": "No tests were run after the final change.",
                    "desired_behavior": "Run relevant tests before claiming completion.",
                }
            ],
            "evidence_trace_ids": ["turn-1"],
        },
    }
    return feedback


def _deterministic_feedback() -> dict[str, Any]:
    feedback = _feedback()
    feedback["id"] = "deterministic-1"
    feedback["feedback_type"] = "weave_agent_signals.outcome.test"
    return feedback


def _audit_only_judge_feedback(kind: str) -> dict[str, Any]:
    feedback = copy.deepcopy(_session_feedback())
    feedback["id"] = f"audit-{kind}"
    details = feedback["payload"]["details"]
    if kind == "legacy_episode":
        feedback["payload"]["granularity"] = "turn"
        details["evaluation_unit"] = "episode"
    elif kind == "missing_context":
        del details["rubric_version"]
    elif kind == "conflicting_unit":
        details["evaluation_unit"] = "episode"
    elif kind == "noncomplete":
        details["review_status"] = "degraded"
    elif kind == "stale_panel":
        details["panel_contract_version"] = "2"
    elif kind == "mismatched_panel":
        details["panel_size"] = 1
    else:
        raise ValueError(f"unknown audit-only feedback kind: {kind}")
    return feedback


class WeaveClient:
    def __init__(self, feedback: list[dict[str, Any]]) -> None:
        self.feedback = feedback
        self.queried_refs: list[str] | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def query_feedback_for_refs(self, refs):
        self.queried_refs = list(refs)
        return list(self.feedback)


class Adapter:
    def __init__(
        self,
        baseline,
        manifest: dict[str, object] | None = None,
    ) -> None:
        self.baseline = baseline
        self.manifest = manifest or _manifest()
        self.capture_calls = 0

    def contract_manifest(self):
        return self.manifest

    def capture(self):
        self.capture_calls += 1
        return self.baseline

    def resolve_locator(self, locator, **_kwargs):
        return locator


class ModelClient:
    def __init__(self) -> None:
        self.cancel: threading.Event | None = None

    def set_cancel(self, cancel: threading.Event) -> None:
        self.cancel = cancel


def _result(baseline, config) -> ReflectionResult:
    candidate_bundle = bundle_from_content_map(
        {
            "CLAUDE.md": "new",
            ".claude/skills/review.md": "Review changes.",
        },
        scope=SCOPE,
    )
    evaluation = EvaluatorRecord(
        evaluation_id="evaluation-1",
        target_revision=baseline.revision,
        requested_model=config.models.proposal_evaluator.id,
        requested_family=config.models.proposal_evaluator.family,
        requested_backend=config.models.proposal_evaluator.backend,
        resolved_model=config.models.proposal_evaluator.id,
        resolved_family=config.models.proposal_evaluator.family,
        resolved_backend=config.models.proposal_evaluator.backend,
        score=0.3,
        rationale="Baseline misses verification.",
        usage={},
    )
    candidate_evaluation = EvaluatorRecord(
        evaluation_id="evaluation-2",
        target_revision=candidate_bundle.revision,
        requested_model=config.models.proposal_evaluator.id,
        requested_family=config.models.proposal_evaluator.family,
        requested_backend=config.models.proposal_evaluator.backend,
        resolved_model=config.models.proposal_evaluator.id,
        resolved_family=config.models.proposal_evaluator.family,
        resolved_backend=config.models.proposal_evaluator.backend,
        score=0.8,
        rationale="Candidate adds verification.",
        usage={},
    )
    attempt = GenerationAttempt(
        attempt_id="attempt-1",
        number=1,
        status="succeeded",
        requested_writer=config.models.proposal_writer,
        resolved_model=config.models.proposal_writer.id,
        resolved_family=config.models.proposal_writer.family,
        resolved_backend=config.models.proposal_writer.backend,
        usage={},
        candidate_revision=candidate_bundle.revision,
        changed_paths=("CLAUDE.md", ".claude/skills/review.md"),
        response_digest=None,
        response_excerpt=None,
        error_type=None,
        error=None,
    )
    failed_attempt = GenerationAttempt(
        attempt_id="attempt-2",
        number=2,
        status="failed",
        requested_writer=config.models.proposal_writer,
        resolved_model=None,
        resolved_family=None,
        resolved_backend=None,
        usage={},
        candidate_revision=None,
        changed_paths=("../secret.md",),
        response_digest=f"sha256:{'0' * 64}",
        response_excerpt='{"path":"../secret.md"}',
        error_type="BundleValidationError",
        error="outside managed scope",
    )
    candidate = ReflectionCandidate(
        candidate_id="candidate-1",
        bundle=candidate_bundle,
        score=0.8,
        score_delta=0.5,
        rationale="Candidate adds verification.",
        generation_attempt_id=attempt.attempt_id,
        requested_writer=config.models.proposal_writer,
        resolved_writer_model=config.models.proposal_writer.id,
        resolved_writer_family=config.models.proposal_writer.family,
        resolved_writer_backend=config.models.proposal_writer.backend,
        evaluation=candidate_evaluation,
    )
    return ReflectionResult(
        baseline=baseline,
        baseline_score=0.3,
        baseline_evaluation=evaluation,
        candidates=(candidate,),
        generation_attempts=(attempt, failed_attempt),
        recommended_candidate_id=candidate.candidate_id,
        baseline_won=False,
    )


def _all_invalid_result(baseline, config) -> ReflectionResult:
    evaluation = EvaluatorRecord(
        evaluation_id="evaluation-1",
        target_revision=baseline.revision,
        requested_model=config.models.proposal_evaluator.id,
        requested_family=config.models.proposal_evaluator.family,
        requested_backend=config.models.proposal_evaluator.backend,
        resolved_model=config.models.proposal_evaluator.id,
        resolved_family=config.models.proposal_evaluator.family,
        resolved_backend=config.models.proposal_evaluator.backend,
        score=0.3,
        rationale="Baseline evidence remains authoritative.",
        usage={},
    )
    rejected = GenerationAttempt(
        attempt_id="attempt-1",
        number=1,
        status="failed",
        requested_writer=config.models.proposal_writer,
        resolved_model=config.models.proposal_writer.id,
        resolved_family=config.models.proposal_writer.family,
        resolved_backend=config.models.proposal_writer.backend,
        usage={},
        candidate_revision=None,
        changed_paths=("../secret.md",),
        response_digest=f"sha256:{'1' * 64}",
        response_excerpt='{"path":"../secret.md"}',
        error_type="BundleValidationError",
        error="outside managed scope",
    )
    return ReflectionResult(
        baseline=baseline,
        baseline_score=evaluation.score,
        baseline_evaluation=evaluation,
        candidates=(),
        generation_attempts=(rejected,),
        recommended_candidate_id=None,
        baseline_won=False,
        reason=NO_VALID_PROPOSAL_REASON,
    )


def test_stage_pins_exact_input_uses_effective_models_and_finalizes_review(store):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE)
    adapter = Adapter(baseline)
    weave = WeaveClient(
        [
            _session_feedback(),
            {**_session_feedback(), "id": "ignored", "weave_ref": "weave:///other"},
            {
                **_session_feedback(),
                "id": "foreign",
                "feedback_type": "other.product.score",
            },
        ]
    )
    writer_client = ModelClient()
    evaluator_client = ModelClient()
    calls: list[dict[str, Any]] = []

    def reflect(**kwargs):
        calls.append(kwargs)
        kwargs["progress_callback"](
            {
                "phase": "generating_candidate",
                "message": "Writer is proposing candidate 1",
                "candidate": 1,
                "attempt_id": "attempt-1",
                "model": config.models.proposal_writer.id,
                "acting_role": "proposal_writer",
            }
        )
        kwargs["progress_callback"](
            {
                "phase": "evaluating_candidate",
                "message": "Evaluator is scoring candidate 1",
                "candidate": 1,
                "attempt_id": "attempt-1",
                "evaluation_id": "evaluation-2",
                "model": config.models.proposal_evaluator.id,
                "acting_role": "proposal_evaluator",
            }
        )
        return _result(kwargs["baseline"], config)

    dependencies = ReflectionDependencies(
        store=store,
        client_factory=lambda: weave,
        adapter_factory=lambda: adapter,
        writer_client_factory=lambda descriptor: nullcontext(writer_client),
        evaluator_client_factory=lambda descriptor: nullcontext(evaluator_client),
        coaching_digest=coaching_digest,
        reflect=reflect,
        clock=lambda: datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
    )

    run_reflection_stage(
        run,
        config,
        threading.Event(),
        dependencies=dependencies,
    )

    assert weave.queried_refs == ["weave:///turn-1", "weave:///session-1"]
    assert adapter.capture_calls == 1
    assert len(calls) == 1
    assert calls[0]["baseline"] is baseline
    assert calls[0]["requested_writer"] is config.models.proposal_writer
    assert calls[0]["requested_evaluator"] is config.models.proposal_evaluator
    assert calls[0]["writer_client"] is writer_client
    assert calls[0]["evaluator_client"] is evaluator_client
    assert calls[0]["candidate_budget"] == config.candidate_budget
    assert calls[0]["feedback"] == [_session_feedback()]
    assert "## Behavioral feedback" in calls[0]["coaching_text"]
    assert "No tests were run after the final change." in calls[0]["coaching_text"]
    assert "Run relevant tests before claiming completion." in calls[0]["coaching_text"]
    assert calls[0]["scope_policy"] == adapter.contract_manifest()
    assert calls[0]["resolve_locator"] == adapter.resolve_locator
    assert writer_client.cancel is not None
    assert evaluator_client.cancel is writer_client.cancel

    updated = store.get(run.run_id)
    assert updated.reflection_input["cohort_id"] == "cohort-test"
    assert updated.reflection_input["feedback_count"] == 1
    assert updated.reflection_input["feedback"][0] == {
        "id": "feedback-1",
        "weave_ref": "weave:///session-1",
        "feedback_type": "weave_agent_signals.judge.verification",
        "digest": updated.reflection_input["feedback"][0]["digest"],
    }
    assert updated.reflection_input["baseline"] == baseline.to_dict()
    assert updated.reflecting_result == _result(baseline, config).to_dict()
    assert updated.reflection_review == {
        "status": "pending",
        "selected_candidate_id": "candidate-1",
        "draft": None,
    }
    assert updated.reflection_review_revision == 1
    assert updated.reflecting_progress["attempted"] == 2
    assert updated.reflecting_progress["valid"] == 1
    assert updated.reflecting_progress["rejected"] == 1
    assert updated.reflecting_progress["scored"] == 1
    assert updated.reflecting_progress["total_attempts"] == config.candidate_budget
    assert "candidate_budget" not in updated.reflecting_progress
    events = updated.reflecting_progress["events"]
    assert any(
        event.get("acting_role") == "proposal_writer"
        and event.get("model") == config.models.proposal_writer.id
        for event in events
    )
    assert any(
        event.get("acting_role") == "proposal_evaluator"
        and event.get("model") == config.models.proposal_evaluator.id
        for event in events
    )


def test_all_invalid_stage_result_persists_audit_without_review(store):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE)
    result = _all_invalid_result(baseline, config)

    def reflect(**kwargs):
        attempt = result.generation_attempts[0]
        kwargs["progress_callback"](
            {
                "phase": "candidate_rejected",
                "message": "Rejected candidate 1: outside managed scope",
                "attempt_id": attempt.attempt_id,
                "model": config.models.proposal_writer.id,
                "acting_role": "proposal_writer",
                "error_type": attempt.error_type,
                "error": attempt.error,
                "changed_paths": attempt.changed_paths,
                "response_digest": attempt.response_digest,
                "response_excerpt": attempt.response_excerpt,
                "attempted": 1,
                "valid": 0,
                "rejected": 1,
                "scored": 0,
                "total_attempts": config.candidate_budget,
            }
        )
        return result

    dependencies = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([_feedback()]),
        adapter_factory=lambda: Adapter(baseline),
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=reflect,
    )

    run_reflection_stage(run, config, threading.Event(), dependencies=dependencies)

    updated = store.get(run.run_id)
    assert updated.reflecting_result == result.to_dict()
    assert updated.reflecting_result["reason"] == NO_VALID_PROPOSAL_REASON
    assert updated.reflection_review is None
    assert updated.reflecting_progress["status_message"] == NO_VALID_PROPOSAL_REASON
    rejected_event = next(
        event
        for event in updated.reflecting_progress["events"]
        if event["phase"] == "candidate_rejected"
    )
    assert rejected_event["changed_paths"] == ["../secret.md"]
    assert rejected_event["attempt_id"] == result.generation_attempts[0].attempt_id
    assert rejected_event["error_type"] == result.generation_attempts[0].error_type
    assert rejected_event["error"] == result.generation_attempts[0].error
    assert rejected_event["response_digest"] == result.generation_attempts[0].response_digest
    assert rejected_event["response_excerpt"] == result.generation_attempts[0].response_excerpt


def test_stage_failure_progress_keeps_actionable_evaluator_message(store):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE)
    message = "Proposal evaluator failed: W&B credits are disabled"
    dependencies = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([_feedback()]),
        adapter_factory=lambda: Adapter(baseline),
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **_kwargs: (_ for _ in ()).throw(ReflectionEvaluationError(message)),
    )

    with pytest.raises(ReflectionStageError, match="credits are disabled") as captured:
        run_reflection_stage(run, config, threading.Event(), dependencies=dependencies)
    assert isinstance(captured.value.__cause__, ReflectionEvaluationError)
    assert str(captured.value.__cause__) == message

    progress = store.get(run.run_id).reflecting_progress
    assert progress["phase"] == "reflection_failed"
    assert progress["status_message"] == message
    assert progress["events"][-1]["message"] == message


def test_stage_failure_progress_does_not_persist_arbitrary_exception_text(store):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE)
    secret = "SENTINEL_PRIVATE_MODEL_OUTPUT"
    dependencies = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([_feedback()]),
        adapter_factory=lambda: Adapter(baseline),
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(secret + "x" * 5_000)),
    )

    with pytest.raises(ReflectionStageError, match="failed unexpectedly") as captured:
        run_reflection_stage(run, config, threading.Event(), dependencies=dependencies)
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert secret in str(captured.value.__cause__)

    progress = store.get(run.run_id).reflecting_progress
    terminal = progress["events"][-1]
    assert secret not in str(progress)
    assert terminal["phase"] == "reflection_failed"
    assert terminal["message"] == "Reflection failed unexpectedly"
    assert terminal["error"] == "Reflection failed unexpectedly"
    assert terminal["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    "failure_point",
    [
        "adapter_factory",
        "adapter_contract",
        "feedback_io",
        "baseline_capture",
        "input_pinning",
        "digest",
        "model_setup",
        "final_validation",
    ],
)
def test_stage_sanitizes_failures_across_the_full_reflection_boundary(
    store,
    monkeypatch,
    failure_point,
):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE)
    adapter = Adapter(baseline)
    weave = WeaveClient([_feedback()])
    secret = f"SENTINEL_{failure_point.upper()}_OUTPUT"
    raw_error = RuntimeError(secret)

    def fail(*_args, **_kwargs):
        raise raw_error

    def adapter_factory():
        return adapter

    def client_factory():
        return weave

    def writer_client_factory(_descriptor):
        return nullcontext(ModelClient())

    def coaching_digest(_feedback):
        return "digest"

    def reflect(**_kwargs):
        return _result(baseline, config)

    if failure_point == "adapter_factory":
        adapter_factory = fail
    elif failure_point == "adapter_contract":
        monkeypatch.setattr(adapter, "contract_manifest", fail)
    elif failure_point == "feedback_io":
        monkeypatch.setattr(weave, "query_feedback_for_refs", fail)
    elif failure_point == "baseline_capture":
        monkeypatch.setattr(adapter, "capture", fail)
    elif failure_point == "input_pinning":
        monkeypatch.setattr(store, "pin_reflection_input", fail)
    elif failure_point == "digest":
        coaching_digest = fail
    elif failure_point == "model_setup":
        writer_client_factory = fail
    elif failure_point == "final_validation":
        monkeypatch.setattr(ReflectionResult, "to_dict", fail)

    dependencies = ReflectionDependencies(
        store=store,
        client_factory=client_factory,
        adapter_factory=adapter_factory,
        writer_client_factory=writer_client_factory,
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=coaching_digest,
        reflect=reflect,
    )

    with pytest.raises(ReflectionStageError, match="failed unexpectedly") as captured:
        run_reflection_stage(run, config, threading.Event(), dependencies=dependencies)

    assert captured.value.__cause__ is raw_error
    progress = store.get(run.run_id).reflecting_progress
    assert progress["phase"] == "reflection_failed"
    assert progress["status_message"] == "Reflection failed unexpectedly"
    assert secret not in str(progress)


def test_reflection_input_pins_target_registry_manifest(store):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"AGENTS.md": "old"}, scope=SCOPE)
    manifest = _manifest("sha256:first")
    adapter = Adapter(baseline, manifest)

    def interrupted(**_kwargs):
        raise RuntimeError("stop after pinning")

    first = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([_feedback()]),
        adapter_factory=lambda: adapter,
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=interrupted,
    )
    with pytest.raises(ReflectionStageError, match="failed unexpectedly"):
        run_reflection_stage(run, config, threading.Event(), dependencies=first)

    pinned = store.get(run.run_id).reflection_input
    assert pinned["schema_version"] == "3"
    assert pinned["target_registry"] == manifest

    changed_manifest = _manifest("sha256:changed")
    second = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([_feedback()]),
        adapter_factory=lambda: Adapter(baseline, changed_manifest),
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **_kwargs: pytest.fail("mismatched registry must fail closed"),
    )

    with pytest.raises(ReflectionInputMismatchError, match="target registry"):
        run_reflection_stage(
            store.get(run.run_id),
            config,
            threading.Event(),
            dependencies=second,
        )


def test_stage_reuses_pinned_baseline_instead_of_recapturing_on_resume(store):
    run, config = _reflecting_run(store)
    original = bundle_from_content_map({"CLAUDE.md": "original"}, scope=SCOPE)
    changed = bundle_from_content_map({"CLAUDE.md": "changed"}, scope=SCOPE)
    first_adapter = Adapter(original)
    second_adapter = Adapter(changed)
    weave = WeaveClient([_feedback()])

    def interrupted(**_kwargs):
        raise RuntimeError("process interrupted after inputs were pinned")

    first = ReflectionDependencies(
        store=store,
        client_factory=lambda: weave,
        adapter_factory=lambda: first_adapter,
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=interrupted,
    )
    with pytest.raises(ReflectionStageError, match="failed unexpectedly"):
        run_reflection_stage(
            run,
            config,
            threading.Event(),
            dependencies=first,
        )
    assert store.get(run.run_id).reflecting_progress["phase"] == "reflection_failed"

    baselines = []

    def resumed(**kwargs):
        baselines.append(kwargs["baseline"])
        return _result(kwargs["baseline"], config)

    second = ReflectionDependencies(
        store=store,
        client_factory=lambda: weave,
        adapter_factory=lambda: second_adapter,
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=resumed,
    )
    run_reflection_stage(
        store.get(run.run_id),
        config,
        threading.Event(),
        dependencies=second,
    )

    assert first_adapter.capture_calls == 1
    assert second_adapter.capture_calls == 0
    assert baselines == [original]
    assert store.get(run.run_id).reflecting_result["baseline"] == original.to_dict()


def test_stage_cancellation_before_input_capture_writes_nothing(store):
    run, config = _reflecting_run(store)
    adapter = Adapter(bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE))
    weave = WeaveClient([_feedback()])
    cancel = threading.Event()
    cancel.set()
    dependencies = ReflectionDependencies(
        store=store,
        client_factory=lambda: weave,
        adapter_factory=lambda: adapter,
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **_kwargs: pytest.fail("reflection should not start"),
    )

    with pytest.raises(StageCancelled):
        run_reflection_stage(run, config, cancel, dependencies=dependencies)

    updated = store.get(run.run_id)
    assert weave.queried_refs is None
    assert adapter.capture_calls == 0
    assert updated.reflection_input is None
    assert updated.reflecting_result is None
    assert updated.reflection_review is None


def test_stage_translates_reflector_cancellation_without_finalizing(store):
    run, config = _reflecting_run(store)
    adapter = Adapter(bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE))
    dependencies = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([_feedback()]),
        adapter_factory=lambda: adapter,
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **_kwargs: (_ for _ in ()).throw(ReflectionCancelled("cancelled")),
    )

    with pytest.raises(StageCancelled):
        run_reflection_stage(
            run,
            config,
            threading.Event(),
            dependencies=dependencies,
        )

    updated = store.get(run.run_id)
    assert updated.reflection_input is not None
    assert updated.reflecting_result is None
    assert updated.reflection_review is None


def test_stage_records_no_feedback_without_opening_model_clients(store):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE)
    opened: list[str] = []
    dependencies = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([]),
        adapter_factory=lambda: Adapter(baseline),
        writer_client_factory=lambda _descriptor: opened.append("writer"),
        evaluator_client_factory=lambda _descriptor: opened.append("evaluator"),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **_kwargs: pytest.fail("reflection should not start"),
    )

    run_reflection_stage(
        run,
        config,
        threading.Event(),
        dependencies=dependencies,
    )

    updated = store.get(run.run_id)
    assert opened == []
    assert updated.reflecting_result == {
        "candidates": [],
        "reason": "No evaluation feedback was found for the pinned cohort.",
    }
    assert updated.reflection_review is None


@pytest.mark.parametrize(
    "kind",
    [
        "legacy_episode",
        "missing_context",
        "conflicting_unit",
        "noncomplete",
        "stale_panel",
        "mismatched_panel",
    ],
)
def test_stage_treats_audit_only_judge_feedback_as_no_reflection_evidence(store, kind):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE)
    opened: list[str] = []
    digest_calls: list[list[dict[str, Any]]] = []
    dependencies = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([_audit_only_judge_feedback(kind)]),
        adapter_factory=lambda: Adapter(baseline),
        writer_client_factory=lambda _descriptor: opened.append("writer"),
        evaluator_client_factory=lambda _descriptor: opened.append("evaluator"),
        coaching_digest=lambda feedback: digest_calls.append(feedback) or "digest",
        reflect=lambda **_kwargs: pytest.fail("reflection should not start"),
    )

    run_reflection_stage(run, config, threading.Event(), dependencies=dependencies)

    updated = store.get(run.run_id)
    assert opened == []
    assert digest_calls == []
    assert updated.reflection_input["feedback_count"] == 0
    assert updated.reflection_input["feedback"] == []
    assert updated.reflecting_result == {
        "candidates": [],
        "reason": "No evaluation feedback was found for the pinned cohort.",
    }


def test_stage_pins_and_passes_only_eligible_reflection_feedback(store):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE)
    eligible = [_deterministic_feedback(), _session_feedback()]
    records = [
        _audit_only_judge_feedback("legacy_episode"),
        eligible[1],
        _audit_only_judge_feedback("missing_context"),
        eligible[0],
        _audit_only_judge_feedback("noncomplete"),
    ]
    reflected: list[list[dict[str, Any]]] = []
    digested: list[list[dict[str, Any]]] = []

    def reflect(**kwargs):
        reflected.append(kwargs["feedback"])
        return _result(kwargs["baseline"], config)

    dependencies = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient(records),
        adapter_factory=lambda: Adapter(baseline),
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda feedback: digested.append(feedback) or "digest",
        reflect=reflect,
    )

    run_reflection_stage(run, config, threading.Event(), dependencies=dependencies)

    selected = [eligible[1], eligible[0]]
    assert reflected == [selected]
    assert digested == [selected]
    pinned = store.get(run.run_id).reflection_input
    assert pinned["feedback_count"] == 2
    assert [(item["id"], item["feedback_type"]) for item in pinned["feedback"]] == [
        ("feedback-1", "weave_agent_signals.judge.verification"),
        ("deterministic-1", "weave_agent_signals.outcome.test"),
    ]


def test_resume_ignores_changes_to_unusable_judge_audit_rows(store):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE)
    records = [_deterministic_feedback(), _audit_only_judge_feedback("legacy_episode")]

    first = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient(records),
        adapter_factory=lambda: Adapter(baseline),
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(ReflectionStageError, match="failed unexpectedly"):
        run_reflection_stage(run, config, threading.Event(), dependencies=first)

    changed_audit = _audit_only_judge_feedback("conflicting_unit")
    changed_audit["id"] = "new-audit-row"
    reflected: list[list[dict[str, Any]]] = []
    second = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([changed_audit, _deterministic_feedback()]),
        adapter_factory=lambda: Adapter(baseline),
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **kwargs: (
            reflected.append(kwargs["feedback"]) or _result(kwargs["baseline"], config)
        ),
    )

    run_reflection_stage(
        store.get(run.run_id),
        config,
        threading.Event(),
        dependencies=second,
    )

    assert reflected == [[_deterministic_feedback()]]
    assert [item["id"] for item in store.get(run.run_id).reflection_input["feedback"]] == [
        "deterministic-1"
    ]


def test_stage_finalizes_persisted_evidence_without_rerunning_inference(store):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"CLAUDE.md": "old"}, scope=SCOPE)
    adapter = Adapter(baseline)
    pinning = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([_feedback()]),
        adapter_factory=lambda: adapter,
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(ReflectionStageError, match="failed unexpectedly"):
        run_reflection_stage(run, config, threading.Event(), dependencies=pinning)

    evidence = _result(baseline, config)
    store.record_stage_result(
        run.run_id,
        stage=RunStatus.REFLECTING,
        result=evidence.to_dict(),
    )
    dependencies = ReflectionDependencies(
        store=store,
        client_factory=lambda: pytest.fail("feedback must not be queried"),
        adapter_factory=lambda: adapter,
        writer_client_factory=lambda _descriptor: pytest.fail("writer must not run"),
        evaluator_client_factory=lambda _descriptor: pytest.fail("evaluator must not run"),
        coaching_digest=lambda _feedback: pytest.fail("digest must not be rebuilt"),
        reflect=lambda **_kwargs: pytest.fail("reflection must not rerun"),
    )

    run_reflection_stage(
        store.get(run.run_id),
        config,
        threading.Event(),
        dependencies=dependencies,
    )

    updated = store.get(run.run_id)
    assert adapter.capture_calls == 1
    assert updated.reflecting_result == evidence.to_dict()
    assert updated.reflection_review["selected_candidate_id"] == "candidate-1"


def test_stage_revalidates_pinned_policy_before_finalizing_persisted_result(store):
    run, config = _reflecting_run(store)
    baseline = bundle_from_content_map({"AGENTS.md": "old"}, scope=SCOPE)
    first_adapter = Adapter(baseline)
    first = ReflectionDependencies(
        store=store,
        client_factory=lambda: WeaveClient([_feedback()]),
        adapter_factory=lambda: first_adapter,
        writer_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        evaluator_client_factory=lambda _descriptor: nullcontext(ModelClient()),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(ReflectionStageError, match="failed unexpectedly"):
        run_reflection_stage(run, config, threading.Event(), dependencies=first)

    evidence = _result(baseline, config)
    store.record_stage_result(
        run.run_id,
        stage=RunStatus.REFLECTING,
        result=evidence.to_dict(),
    )
    changed_adapter = Adapter(baseline, _manifest("sha256:changed"))
    second = ReflectionDependencies(
        store=store,
        client_factory=lambda: pytest.fail("feedback must not be queried"),
        adapter_factory=lambda: changed_adapter,
        writer_client_factory=lambda _descriptor: pytest.fail("writer must not run"),
        evaluator_client_factory=lambda _descriptor: pytest.fail("evaluator must not run"),
        coaching_digest=lambda _feedback: pytest.fail("digest must not be rebuilt"),
        reflect=lambda **_kwargs: pytest.fail("reflection must not rerun"),
    )

    with pytest.raises(ReflectionInputMismatchError, match="target registry"):
        run_reflection_stage(
            store.get(run.run_id),
            config,
            threading.Event(),
            dependencies=second,
        )

    assert changed_adapter.capture_calls == 0
    assert store.get(run.run_id).reflection_review is None
