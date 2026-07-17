from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.routes import serialize_run
from weave_agent_signals.run_config import (
    PIPELINE_VERSION,
    EffectiveRunConfig,
    RunConfig,
    resolve_run_config,
)
from weave_agent_signals.runs.bundles import ScopeDescriptor, bundle_from_content_map
from weave_agent_signals.runs.service import RunService, _configuration_error
from weave_agent_signals.runs.stages import StageCancelled
from weave_agent_signals.runs.stages.reflection import (
    ReflectionDependencies,
    run_reflection_stage,
)
from weave_agent_signals.runs.store import (
    DataSelection,
    Run,
    RunStatus,
    RunStore,
    RunStoreConflictError,
)


class SynchronousExecutor:
    def submit(self, function, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


@dataclass
class StageSpy:
    store: RunStore
    complete: bool = True
    error: BaseException | None = None
    calls: list[tuple[Run, EffectiveRunConfig, threading.Event]] = field(default_factory=list)

    def __call__(
        self,
        run: Run,
        config: EffectiveRunConfig,
        cancel: threading.Event,
    ) -> None:
        self.calls.append((run, config, cancel))
        if self.error is not None:
            raise self.error
        if self.complete:
            self.store.record_stage_result(
                run.run_id,
                stage=run.status,
                result={"stage": run.status.value},
            )


@pytest.fixture
def store(tmp_path):
    instance = RunStore(tmp_path / "runs.db")
    yield instance
    instance.close()


def _catalogs():
    return build_model_catalog(which=lambda name: f"/bin/{name}"), build_rubric_catalog()


def _request() -> RunConfig:
    models, rubrics = _catalogs()
    return RunConfig(
        model_catalog_version=models.catalog_version,
        rubric_catalog_version=rubrics.catalog_version,
        judge_models=("claude:claude-sonnet-5", "codex:gpt-5.6-sol"),
        challenge_judge_models=("codex:gpt-5.6-sol",),
        proposal_model="codex:gpt-5.6-sol",
        proposal_evaluator_model="claude:claude-sonnet-5",
        rubrics=("judge.verification", "judge.session_outcome"),
        candidate_budget=3,
        force=False,
    )


def _selection() -> DataSelection:
    return DataSelection(
        since="2026-07-01T00:00:00+00:00",
        until="2026-07-14T00:00:00+00:00",
        timezone="UTC",
        session_ids=("session-1",),
    )


def _cohort() -> dict:
    return {
        "schema_version": 1,
        "pinned_at": "2026-07-14T12:00:00+00:00",
        "cohort_id": "sha256:test-cohort",
        "turn_count": 1,
        "session_count": 1,
        "turns": [
            {
                "trace_id": "trace-1",
                "weave_ref": "weave:///weave-team/agent-sessions/agent_turn/trace-1",
                "conversation_id": "session-1",
                "started_at": "2026-07-14T11:00:00+00:00",
                "model": "claude-sonnet-5",
                "model_family": "anthropic",
            }
        ],
        "sessions": [
            {
                "conversation_id": "session-1",
                "weave_ref": ("weave:///weave-team/agent-sessions/agent_conversation/session-1"),
                "turn_count": 1,
            }
        ],
    }


def _service(
    store: RunStore,
    *,
    scoring: StageSpy | None = None,
    judging: StageSpy | None = None,
    reflection: StageSpy | None = None,
    discover=None,
) -> tuple[RunService, StageSpy, StageSpy, StageSpy]:
    models, rubrics = _catalogs()
    scoring = scoring or StageSpy(store)
    judging = judging or StageSpy(store)
    reflection = reflection or StageSpy(store)
    service = RunService(
        store=store,
        build_model_catalog=lambda: models,
        build_rubric_catalog=lambda: rubrics,
        discover_cohort=discover or (lambda _selection: _cohort()),
        scoring_stage=scoring,
        judging_stage=judging,
        reflection_stage=reflection,
        executor=SynchronousExecutor(),
    )
    return service, scoring, judging, reflection


def _configured(service: RunService, *, auto_run: bool = False) -> Run:
    run = service.create()
    service.save_selection(run.run_id, _selection())
    service.save_config(run.run_id, _request())
    service.set_auto_run(run.run_id, auto_run)
    return service.get(run.run_id)


def _start_directly(store: RunStore, service: RunService, *, auto_run: bool = False) -> Run:
    run = _configured(service, auto_run=auto_run)
    models, rubrics = _catalogs()
    return store.start(
        run.run_id,
        expected_selection=_selection(),
        expected_config=_request(),
        turn_cohort=_cohort(),
        effective_config=resolve_run_config(
            _request(),
            model_catalog=models,
            rubric_catalog=rubrics,
        ),
    )


def test_list_summaries_delegates_to_store_projection(store):
    service, _, _, _ = _service(store)
    run = service.create()
    service.save_selection(run.run_id, _selection())

    summaries = service.list_summaries()

    assert len(summaries) == 1
    assert summaries[0].run_id == run.run_id
    assert summaries[0].session_count == 1


def test_advance_discovers_then_pins_and_executes_with_effective_config(store):
    observations: list[tuple[DataSelection, RunConfig | None]] = []

    def discover(selection: DataSelection) -> dict:
        current = service.get(configured.run_id)
        observations.append((selection, current.run_config))
        return _cohort()

    service, scoring, _, _ = _service(store, discover=discover)
    configured = _configured(service)

    advanced = service.advance(configured.run_id)

    pinned = service.get(configured.run_id)
    assert advanced == pinned
    assert pinned.status is RunStatus.SCORING
    assert observations == [(_selection(), _request())]
    assert pinned.turn_cohort == _cohort()
    assert pinned.current_stage_succeeded is True
    assert pinned.scoring_succeeded is True
    assert pinned.effective_config is not None
    assert pinned.effective_config.pipeline_version == PIPELINE_VERSION
    assert scoring.calls[0][0].run_id == pinned.run_id
    assert scoring.calls[0][0].status is RunStatus.SCORING
    assert scoring.calls[0][0].scoring_result is None
    assert scoring.calls[0][1] == pinned.effective_config
    assert scoring.calls[0][2].is_set() is False
    assert any(
        warning.code == "judge_evaluated_family_overlap"
        for warning in pinned.effective_config.selection_warnings
    )


def test_pipeline_version_mismatch_fails_before_stage_external_work(store, tmp_path):
    service, scoring, _, _ = _service(store)
    run = _configured(service)
    service.advance(run.run_id)
    pinned = service.get(run.run_id)
    stale = pinned.effective_config.model_copy(update={"pipeline_version": "old"})

    with sqlite3.connect(tmp_path / "runs.db") as connection:
        connection.execute(
            "UPDATE runs SET effective_config = ? WHERE run_id = ?",
            (stale.model_dump_json(), run.run_id),
        )

    scoring.calls.clear()
    service.execute_stage(run.run_id, RunStatus.SCORING)

    failed = service.get(run.run_id)
    assert failed.status is RunStatus.FAILED
    assert "pipeline version" in failed.error.lower()
    assert "start a new run" in failed.error.lower()
    assert scoring.calls == []


def test_old_pipeline_snapshot_fails_compatibility_gate_before_stage_work():
    models, rubrics = _catalogs()
    effective = resolve_run_config(
        _request(),
        model_catalog=models,
        rubric_catalog=rubrics,
    )
    old = effective.model_copy(update={"pipeline_version": "1"})

    message = _configuration_error(SimpleNamespace(effective_config=old))

    assert message == "Pinned pipeline version does not match this server; start a new run."


def test_manual_advance_requires_result_and_uses_same_pinned_config(store):
    service, scoring, judging, reflection = _service(store)
    run = _configured(service)
    service.advance(run.run_id)

    judging_run = service.advance(run.run_id)
    assert judging_run.status is RunStatus.JUDGING
    assert judging_run.current_stage_succeeded is True
    assert judging_run.judging_succeeded is True
    assert judging.calls[0][1] == scoring.calls[0][1]

    reflecting_run = service.advance(run.run_id)
    assert reflecting_run.status is RunStatus.COMPLETE
    assert reflection.calls[0][1] == scoring.calls[0][1]


def test_auto_run_chains_successful_stages_and_completes(store):
    scoring = StageSpy(store, complete=True)
    judging = StageSpy(store, complete=True)
    reflection = StageSpy(store, complete=True)
    service, scoring, judging, reflection = _service(
        store,
        scoring=scoring,
        judging=judging,
        reflection=reflection,
    )
    run = _configured(service, auto_run=True)

    completed = service.advance(run.run_id)

    assert completed.status is RunStatus.COMPLETE
    assert [len(stage.calls) for stage in (scoring, judging, reflection)] == [1, 1, 1]
    assert completed.scoring_result == {"stage": "scoring"}
    assert completed.judging_result == {"stage": "judging"}
    assert completed.reflecting_result == {"stage": "reflecting"}
    assert completed.scoring_succeeded is True
    assert completed.judging_succeeded is True
    assert completed.reflecting_succeeded is True
    assert completed.current_stage_succeeded is True


def test_manual_advance_rejects_a_result_before_worker_success_is_finalized(store):
    service, _, _, _ = _service(store)
    run = _start_directly(store, service)
    store.record_stage_result(
        run.run_id,
        stage=RunStatus.SCORING,
        result={"stage": "scoring"},
    )

    with pytest.raises(RunStoreConflictError, match="not finalized successfully"):
        service.advance(run.run_id)

    current = service.get(run.run_id)
    assert current.status is RunStatus.SCORING
    assert current.scoring_result == {"stage": "scoring"}
    assert current.current_stage_succeeded is False


def test_restart_finalizes_a_persisted_success_without_repeating_stage_work(store):
    service, scoring, _, _ = _service(store)
    run = _start_directly(store, service)
    store.record_stage_result(
        run.run_id,
        stage=RunStatus.SCORING,
        result={
            "turns_scored": 1,
            "sessions_scored": 1,
            "scores_written": 3,
            "errors": 0,
            "turn_details": [],
        },
    )

    service.recover_interrupted_runs()

    recovered = service.get(run.run_id)
    assert recovered.status is RunStatus.SCORING
    assert recovered.current_stage_succeeded is True
    assert scoring.calls == []


def test_restart_fails_a_persisted_partial_scoring_result(store):
    service, scoring, _, _ = _service(store)
    run = _start_directly(store, service)
    partial_result = {
        "turns_scored": 1,
        "sessions_scored": 1,
        "scores_written": 2,
        "errors": 1,
        "turn_details": [],
    }
    store.record_stage_result(
        run.run_id,
        stage=RunStatus.SCORING,
        result=partial_result,
    )

    service.recover_interrupted_runs()

    recovered = service.get(run.run_id)
    assert recovered.status is RunStatus.FAILED
    assert recovered.scoring_result == partial_result
    assert recovered.scoring_succeeded is False
    assert recovered.error == "Persisted scoring result is incomplete. Start a new run."
    assert scoring.calls == []


@pytest.mark.parametrize("errors", [False, 0.0])
def test_restart_rejects_a_non_integer_zero_scoring_error_count(store, errors):
    service, _, _, _ = _service(store)
    run = _start_directly(store, service)
    store.record_stage_result(
        run.run_id,
        stage=RunStatus.SCORING,
        result={
            "turns_scored": 1,
            "sessions_scored": 1,
            "scores_written": 2,
            "errors": errors,
            "turn_details": [],
        },
    )

    service.recover_interrupted_runs()

    recovered = service.get(run.run_id)
    assert recovered.status is RunStatus.FAILED
    assert recovered.scoring_succeeded is False


def test_restart_fails_unfinished_work_instead_of_repeating_external_calls(store):
    service, scoring, _, _ = _service(store)
    run = _start_directly(store, service)

    service.recover_interrupted_runs()

    recovered = service.get(run.run_id)
    assert recovered.status is RunStatus.FAILED
    assert "server stopped before scoring completed" in recovered.error.lower()
    assert "start a new run" in recovered.error.lower()
    assert scoring.calls == []


def test_restart_preserves_failed_judging_result_as_audit_evidence(store):
    service, _, judging, _ = _service(store)
    run = _start_directly(store, service)
    store.record_stage_result(
        run.run_id,
        stage=RunStatus.SCORING,
        result={
            "turns_scored": 1,
            "sessions_scored": 1,
            "scores_written": 3,
            "errors": 0,
            "turn_details": [],
        },
    )
    store.finalize_stage_success(
        run.run_id,
        stage=RunStatus.SCORING,
        advance=True,
    )
    failed_result = {
        "coverage_complete": False,
        "failure_count": 1,
        "write_failure_count": 0,
        "status_message": "Judging coverage incomplete",
    }
    store.record_stage_result(
        run.run_id,
        stage=RunStatus.JUDGING,
        result=failed_result,
    )

    service.recover_interrupted_runs()

    recovered = service.get(run.run_id)
    assert recovered.status is RunStatus.FAILED
    assert recovered.judging_result == failed_result
    assert recovered.error == "Judging coverage incomplete. Start a new run."
    assert judging.calls == []


def test_restart_fails_a_manual_pause_pinned_to_an_old_pipeline(store, tmp_path):
    service, _, _, _ = _service(store)
    run = _start_directly(store, service)
    store.record_stage_result(
        run.run_id,
        stage=RunStatus.SCORING,
        result={
            "turns_scored": 1,
            "sessions_scored": 1,
            "scores_written": 3,
            "errors": 0,
            "turn_details": [],
        },
    )
    store.finalize_stage_success(
        run.run_id,
        stage=RunStatus.SCORING,
        advance=False,
    )
    stale = run.effective_config.model_copy(update={"pipeline_version": "old"})
    with sqlite3.connect(tmp_path / "runs.db") as connection:
        connection.execute(
            "UPDATE runs SET effective_config = ? WHERE run_id = ?",
            (stale.model_dump_json(), run.run_id),
        )

    service.recover_interrupted_runs()

    recovered = service.get(run.run_id)
    assert recovered.status is RunStatus.FAILED
    assert "pipeline version" in recovered.error.lower()


def test_stage_failure_is_recorded_but_cancellation_is_not_rewritten(store):
    failing = StageSpy(store, error=RuntimeError("judge transport failed"))
    service, _, _, _ = _service(store, scoring=failing)
    run = _configured(service)
    failed = service.advance(run.run_id)

    assert failed.status is RunStatus.FAILED
    assert failed.error == "judge transport failed"

    cancelled_stage = StageSpy(store, error=StageCancelled())
    other_service, _, _, _ = _service(store, scoring=cancelled_stage)
    other = _configured(other_service)
    cancelled = other_service.advance(other.run_id)

    assert cancelled.status is RunStatus.SCORING
    assert cancelled.error is None


def test_reflection_stage_failure_exposes_only_safe_error_through_run_service(store, caplog):
    secret = "SENTINEL_PRIVATE_REFLECTION_OUTPUT"
    raw_error = RuntimeError(secret + "x" * 5_000)
    scope = ScopeDescriptor("file", "/project", ("AGENTS.md",))
    baseline = bundle_from_content_map({"AGENTS.md": "old"}, scope=scope)

    class FeedbackClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query_feedback_for_refs(self, _refs):
            return [
                {
                    "id": "feedback-1",
                    "weave_ref": _cohort()["turns"][0]["weave_ref"],
                    "feedback_type": "weave_agent_signals.outcome.test",
                    "payload": {"rating": 0.2, "details": {}},
                }
            ]

    class Adapter:
        def contract_manifest(self):
            return {
                "schema_version": "1",
                "targets": [{"kind": "file", "id": "agents"}],
                "digest": "sha256:test-registry",
            }

        def capture(self):
            return baseline

        def resolve_locator(self, locator, **_kwargs):
            return locator

    dependencies = ReflectionDependencies(
        store=store,
        client_factory=FeedbackClient,
        adapter_factory=Adapter,
        writer_client_factory=lambda _descriptor: nullcontext(object()),
        evaluator_client_factory=lambda _descriptor: nullcontext(object()),
        coaching_digest=lambda _feedback: "digest",
        reflect=lambda **_kwargs: (_ for _ in ()).throw(raw_error),
    )
    captured: list[RuntimeError] = []

    def reflection_stage(run, config, cancel):
        try:
            run_reflection_stage(run, config, cancel, dependencies=dependencies)
        except RuntimeError as error:
            captured.append(error)
            raise

    service, _, _, _ = _service(store, reflection=reflection_stage)
    scoring = _start_directly(store, service)
    store.record_stage_result(
        scoring.run_id,
        stage=RunStatus.SCORING,
        result={"stage": "scoring"},
    )
    judging = store.finalize_stage_success(
        scoring.run_id,
        stage=RunStatus.SCORING,
        advance=True,
    )
    store.record_stage_result(
        judging.run_id,
        stage=RunStatus.JUDGING,
        result={"stage": "judging"},
    )
    reflecting = store.finalize_stage_success(
        judging.run_id,
        stage=RunStatus.JUDGING,
        advance=True,
    )

    with caplog.at_level("WARNING", logger="weave_agent_signals.runs.service"):
        service.execute_stage(reflecting.run_id, RunStatus.REFLECTING)

    failed = service.get(reflecting.run_id)
    serialized = json.dumps(serialize_run(failed), sort_keys=True)
    assert failed.status is RunStatus.FAILED
    assert failed.error == "Reflection failed unexpectedly"
    assert secret not in caplog.text
    assert secret not in failed.error
    assert secret not in serialized
    assert len(captured) == 1
    assert str(captured[0]) == "Reflection failed unexpectedly"
    assert captured[0].__cause__ is raw_error
