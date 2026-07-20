from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.judges.plan import build_canonical_judging_plan, judging_plan_totals
from weave_agent_signals.judges.records import JudgeCallAudit, JudgeCallRecord
from weave_agent_signals.models import SessionView, TurnSpan
from weave_agent_signals.run_config import RunConfig, resolve_run_config
from weave_agent_signals.runs.bundles import ScopeDescriptor, bundle_from_content_map
from weave_agent_signals.runs.events import sanitize_event
from weave_agent_signals.runs.reflection_records import (
    EvaluatorRecord,
    GenerationAttemptRecord,
    ReflectionInputRecord,
    ReflectionResultRecord,
)
from weave_agent_signals.runs.store import (
    DataSelection,
    ReflectionReviewLifecycleConflictError,
    ReflectionReviewRevisionConflictError,
    RunCancellationConflictError,
    RunLifecycleConflictError,
    RunStatus,
    RunStore,
    RunStoreConflictError,
)


@pytest.fixture
def store(tmp_path):
    instance = RunStore(tmp_path / "runs.db")
    yield instance
    instance.close()


def _all_executables(name: str) -> str:
    return f"/bin/{name}"


def _run_inputs(*, candidate_budget: int = 3):
    models = build_model_catalog(which=_all_executables)
    rubrics = build_rubric_catalog()
    requested = RunConfig(
        model_catalog_version=models.catalog_version,
        rubric_catalog_version=rubrics.catalog_version,
        judge_models=("claude:claude-sonnet-5", "codex:gpt-5.6-sol"),
        challenge_judge_models=("codex:gpt-5.6-sol",),
        proposal_model="codex:gpt-5.6-sol",
        proposal_evaluator_model="claude:claude-sonnet-5",
        rubrics=(),
        candidate_budget=candidate_budget,
        force=False,
    )
    effective = resolve_run_config(
        requested,
        model_catalog=models,
        rubric_catalog=rubrics,
    )
    return requested, effective


def _selection(*session_ids: str) -> DataSelection:
    return DataSelection(
        since="2026-07-01T00:00:00+00:00",
        until="2026-07-14T00:00:00+00:00",
        timezone="UTC",
        session_ids=session_ids or ("conv-1",),
    )


def _turn_cohort(*trace_ids: str) -> dict:
    identities = trace_ids or ("turn-1",)
    turns = [
        {
            "trace_id": trace_id,
            "weave_ref": f"weave:///weave-team/agent-sessions/agent_turn/{trace_id}",
            "conversation_id": "conv-1",
            "started_at": f"2026-07-{position:02d}T12:00:00+00:00",
            "model": None,
            "model_family": "unknown",
        }
        for position, trace_id in enumerate(identities, start=1)
    ]
    return {
        "schema_version": "1",
        "pinned_at": "2026-07-14T00:00:00+00:00",
        "cohort_id": "cohort-test",
        "turn_count": len(turns),
        "session_count": 1,
        "turns": turns,
        "sessions": [
            {
                "conversation_id": "conv-1",
                "weave_ref": "weave:///weave-team/agent-sessions/agent_conversation/conv-1",
                "turn_count": len(turns),
            }
        ],
    }


def _start(store: RunStore):
    run = store.create()
    selection = _selection()
    requested, effective = _run_inputs()
    store.save_selection(run.run_id, selection)
    store.save_config(run.run_id, requested)
    started = store.start(
        run.run_id,
        expected_selection=selection,
        expected_config=requested,
        turn_cohort=_turn_cohort(),
        effective_config=effective,
    )
    return started, requested, effective


def _advance_to_reflecting(store: RunStore):
    started, requested, effective = _start(store)
    store.record_stage_result(
        started.run_id,
        stage=RunStatus.SCORING,
        result={"written": 2},
    )
    store.finalize_stage_success(
        started.run_id,
        stage=RunStatus.SCORING,
        advance=True,
    )
    store.record_stage_result(
        started.run_id,
        stage=RunStatus.JUDGING,
        result={"written": 4},
    )
    reflecting = store.finalize_stage_success(
        started.run_id,
        stage=RunStatus.JUDGING,
        advance=True,
    )
    return reflecting, requested, effective


def _judging_plan() -> dict:
    _, effective = _run_inputs()
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    turn = TurnSpan(
        trace_id="turn-1",
        conversation_id="conv-1",
        started_at=now,
        ended_at=now,
        model="agent",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        status_code="OK",
        config_version="cfg",
        git_branch="main",
        effort_level=None,
        session_id="sid",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
        user_input="request",
        assistant_output="response",
    )
    return build_canonical_judging_plan(
        [SessionView("conv-1", [turn], "cfg", "main")],
        cohort_id="cohort-test",
        rubrics=effective.rubrics[:1],
        judge_models=effective.models.judges,
        context_policy=effective.judging_context,
    )


def _reflection_evidence() -> ReflectionResultRecord:
    scope = ScopeDescriptor("file", "/project", ("AGENTS.md",))
    baseline = bundle_from_content_map({"AGENTS.md": "before"}, scope=scope)
    candidate = bundle_from_content_map({"AGENTS.md": "after"}, scope=scope)
    writer = next(
        model
        for model in build_model_catalog().available_models
        if "proposal_writer" in model.supported_roles
    )
    attempt = GenerationAttemptRecord(
        attempt_id="attempt-1",
        number=1,
        status="succeeded",
        requested_writer=writer,
        resolved_model=writer.provider_model,
        resolved_family=writer.family,
        resolved_backend=writer.provider,
        candidate_id="candidate-1",
        bundle=candidate,
        changed_paths=("AGENTS.md",),
    )
    evaluations = tuple(
        EvaluatorRecord(
            evaluation_id=f"evaluation-{index}",
            target_id=target,
            requested_model="evaluator",
            requested_family="evaluator-family",
            requested_backend="test",
            resolved_model="evaluator",
            resolved_family="evaluator-family",
            resolved_backend="test",
            score=score,
            rationale="test",
        )
        for index, (target, score) in enumerate((("baseline", 0.2), ("candidate-1", 0.8)), start=1)
    )
    return ReflectionResultRecord(
        baseline=baseline,
        attempts=(attempt,),
        evaluations=evaluations,
        recommended_candidate_id="candidate-1",
        baseline_won=False,
    )


def test_store_rejects_unknown_schema_without_deleting_rows(tmp_path):
    path = tmp_path / "runs.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, obsolete TEXT)")
    connection.execute("INSERT INTO runs VALUES ('legacy', '{}')")
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()

    with pytest.raises(Exception, match="unsupported run database schema"):
        RunStore(path)

    connection = sqlite3.connect(path)
    assert connection.execute("SELECT run_id FROM runs").fetchone()[0] == "legacy"
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    connection.close()


def test_list_summaries_projects_only_scalar_fields_without_full_run_decoding(
    store,
    monkeypatch,
):
    run = store.create()
    sentinel = "DO-NOT-TRANSFER-FULL-RUN-EVIDENCE"
    large_content = sentinel + ("x" * 250_000)
    store._conn.execute(
        """
        UPDATE runs
        SET status = ?, data_selection = ?, run_config = ?, effective_config = ?,
            reflection_input = ?, reflecting_result = ?, reflecting_succeeded = 1,
            reflection_review = ?
        WHERE run_id = ?
        """,
        (
            RunStatus.COMPLETE.value,
            json.dumps(
                {
                    "since": "2026-07-01T00:00:00+00:00",
                    "until": None,
                    "timezone": "UTC",
                    "session_ids": ["session-1", "session-2"],
                }
            ),
            f"invalid requested config {large_content}",
            f"invalid effective config {large_content}",
            json.dumps({"baseline": {"content": large_content}}),
            json.dumps(
                {
                    "baseline": {"content": large_content},
                    "baseline_won": True,
                    "reason": None,
                }
            ),
            json.dumps({"status": "pending", "draft": {"content": large_content}}),
            run.run_id,
        ),
    )
    store._conn.commit()
    statements: list[str] = []
    store._conn.set_trace_callback(statements.append)
    monkeypatch.setattr(
        store,
        "_row_to_run",
        lambda _row: pytest.fail("summary listing must not decode a full run"),
    )

    summaries = store.list_summaries()

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.run_id == run.run_id
    assert summary.status is RunStatus.COMPLETE
    assert summary.created_at == run.created_at
    assert summary.current_stage_succeeded is True
    assert summary.session_count == 2
    assert summary.selection_since == "2026-07-01T00:00:00+00:00"
    assert summary.selection_until is None
    assert summary.selection_timezone == "UTC"
    assert summary.review_status == "pending"
    assert summary.baseline_won is True
    assert summary.reflection_reason is None
    assert sentinel not in repr(summary)

    query = next(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    )
    assert "SELECT *" not in query.upper()
    assert "run_config" not in query
    assert "effective_config" not in query
    assert "reflection_input" not in query
    assert "json_array_length(data_selection" in query
    assert "json_extract(reflection_review" in query
    assert "json_extract(reflecting_result" in query


def test_atomic_start_pins_pydantic_json_and_has_final_run_shape(store):
    run = store.create()
    selection = _selection()
    requested, effective = _run_inputs()
    store.save_selection(run.run_id, selection)
    store.save_config(run.run_id, requested)

    started = store.start(
        run.run_id,
        expected_selection=selection,
        expected_config=requested,
        turn_cohort=_turn_cohort(),
        effective_config=effective,
    )

    assert started.status is RunStatus.SCORING
    assert started.data_selection == selection
    assert started.run_config == requested
    assert started.effective_config == effective
    raw = store._conn.execute(
        "SELECT run_config, effective_config FROM runs WHERE run_id = ?",
        (run.run_id,),
    ).fetchone()
    assert json.loads(raw["run_config"]) == json.loads(requested.model_dump_json())
    assert json.loads(raw["effective_config"]) == json.loads(effective.model_dump_json())


def test_stage_result_cannot_advance_until_worker_finalizes_success(store):
    started, _requested, _effective = _start(store)
    store.record_stage_result(
        started.run_id,
        stage=RunStatus.SCORING,
        result={"stage": "scoring"},
    )

    with pytest.raises(RunStoreConflictError, match="not finalized successfully"):
        store.transition(
            started.run_id,
            expected_status=RunStatus.SCORING,
            new_status=RunStatus.JUDGING,
        )

    pending = store.get(started.run_id)
    assert pending is not None
    assert pending.scoring_succeeded is False
    finalized = store.finalize_stage_success(
        started.run_id,
        stage=RunStatus.SCORING,
        advance=False,
    )
    assert finalized.status is RunStatus.SCORING
    assert finalized.scoring_succeeded is True

    transitioned = store.transition(
        started.run_id,
        expected_status=RunStatus.SCORING,
        new_status=RunStatus.JUDGING,
    )
    assert transitioned.status is RunStatus.JUDGING


def test_stage_success_and_auto_advance_are_one_store_write(store):
    started, _requested, _effective = _start(store)
    store.record_stage_result(
        started.run_id,
        stage=RunStatus.SCORING,
        result={"stage": "scoring"},
    )

    advanced = store.finalize_stage_success(
        started.run_id,
        stage=RunStatus.SCORING,
        advance=True,
    )

    assert advanced.status is RunStatus.JUDGING
    assert advanced.scoring_succeeded is True


def test_start_compare_and_swap_rejects_changed_configuration_without_partial_pins(tmp_path):
    path = tmp_path / "runs.db"
    first = RunStore(path)
    second = RunStore(path)
    run = first.create()
    selection = _selection()
    before, effective = _run_inputs(candidate_budget=3)
    after, _ = _run_inputs(candidate_budget=4)
    first.save_selection(run.run_id, selection)
    first.save_config(run.run_id, before)
    second.save_config(run.run_id, after)

    with pytest.raises(RunStoreConflictError, match="configuration changed"):
        first.start(
            run.run_id,
            expected_selection=selection,
            expected_config=before,
            turn_cohort=_turn_cohort(),
            effective_config=effective,
        )

    current = second.get(run.run_id)
    assert current.status is RunStatus.CREATED
    assert current.run_config == after
    assert current.turn_cohort is None
    assert current.effective_config is None
    first.close()
    second.close()


def test_setup_writes_are_created_only_and_return_current_snapshot(store):
    run = store.create()
    selection = _selection()
    requested, _ = _run_inputs()

    selected = store.save_selection(run.run_id, selection)
    configured = store.save_config(run.run_id, requested)
    automatic = store.save_auto_run(run.run_id, True)
    assert selected.data_selection == selection
    assert configured.run_config == requested
    assert automatic.auto_run is True

    started = store.start(
        run.run_id,
        expected_selection=selection,
        expected_config=requested,
        turn_cohort=_turn_cohort(),
        effective_config=_run_inputs()[1],
    )
    with pytest.raises(RunLifecycleConflictError):
        store.save_auto_run(started.run_id, False)


def test_typed_stage_writes_and_transition_enforce_active_status(store):
    started, _, _ = _start(store)

    progressed = store.record_stage_progress(
        started.run_id,
        stage=RunStatus.SCORING,
        progress={"processed": 1, "total": 2},
    )
    completed = store.record_stage_result(
        started.run_id,
        stage=RunStatus.SCORING,
        result={"written": 2},
    )
    store.finalize_stage_success(
        started.run_id,
        stage=RunStatus.SCORING,
        advance=False,
    )
    judging = store.transition(
        started.run_id,
        expected_status=RunStatus.SCORING,
        new_status=RunStatus.JUDGING,
    )

    assert progressed.scoring_progress == {"processed": 1, "total": 2}
    assert completed.scoring_result == {"written": 2}
    assert judging.status is RunStatus.JUDGING
    assert not hasattr(store, "update")
    with pytest.raises(RunStoreConflictError, match="no longer scoring"):
        store.record_stage_result(
            started.run_id,
            stage=RunStatus.SCORING,
            result={"written": 3},
        )
    with pytest.raises(RunStoreConflictError, match="expected scoring"):
        store.transition(
            started.run_id,
            expected_status=RunStatus.SCORING,
            new_status=RunStatus.JUDGING,
        )


def test_fail_is_typed_atomic_and_cannot_overwrite_cancellation(store):
    started, _, _ = _start(store)
    cancelled = store.cancel_if_safe(started.run_id)
    assert cancelled.status is RunStatus.CANCELLED

    with pytest.raises(RunStoreConflictError, match="expected scoring"):
        store.fail(
            started.run_id,
            expected_status=RunStatus.SCORING,
            error="late failure",
        )
    current = store.get(started.run_id)
    assert current.status is RunStatus.CANCELLED
    assert current.error is None


def test_external_write_barrier_serializes_cancellation_across_connections(tmp_path):
    path = tmp_path / "runs.db"
    writer = RunStore(path)
    canceller = RunStore(path)
    started, _, _ = _start(writer)
    finished = threading.Event()
    result = []

    def cancel() -> None:
        result.append(canceller.cancel_if_safe(started.run_id))
        finished.set()

    with writer.external_write_barrier(started.run_id, RunStatus.SCORING):
        thread = threading.Thread(target=cancel)
        thread.start()
        assert not finished.wait(0.05)

    thread.join(timeout=2)
    assert finished.is_set()
    assert result[0].status is RunStatus.CANCELLED
    writer.close()
    canceller.close()


def test_judging_plan_and_reflection_input_are_validated_and_write_once(store):
    started, _, _ = _start(store)
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
    plan = _judging_plan()
    planned = store.pin_judging_plan(started.run_id, plan)
    assert planned.judging_plan == plan
    assert store.pin_judging_plan(started.run_id, plan).judging_plan == plan
    changed_plan = plan.model_copy(update={"cohort_id": "changed"})
    with pytest.raises(ValueError, match="plan ID"):
        store.pin_judging_plan(started.run_id, changed_plan)
    store.record_stage_result(
        started.run_id,
        stage=RunStatus.JUDGING,
        result={"written": 1},
    )
    store.finalize_stage_success(
        started.run_id,
        stage=RunStatus.JUDGING,
        advance=True,
    )
    reflection_input = ReflectionInputRecord(
        cohort_id="cohort-test",
        captured_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        feedback=(),
        target_registry={},
        baseline=_reflection_evidence().baseline,
    )
    pinned = store.pin_reflection_input(started.run_id, reflection_input)
    assert pinned.reflection_input == reflection_input
    assert store.pin_reflection_input(started.run_id, reflection_input).reflection_input == (
        reflection_input
    )
    with pytest.raises(RunStoreConflictError, match="already pinned"):
        store.pin_reflection_input(
            started.run_id,
            reflection_input.model_copy(update={"captured_at": datetime.now(timezone.utc)}),
        )


def test_store_accepts_strict_skipped_reviewer_disposition(store, monkeypatch):
    started, _, effective = _start(store)
    store.record_stage_result(started.run_id, stage=RunStatus.SCORING, result={"written": 1})
    store.finalize_stage_success(started.run_id, stage=RunStatus.SCORING, advance=True)
    from weave_agent_signals.judges import plan as plan_module
    from weave_agent_signals.judges.windowing import WindowPlanInapplicable

    monkeypatch.setattr(
        plan_module,
        "build_window_plan",
        lambda *_args: (_ for _ in ()).throw(WindowPlanInapplicable()),
    )
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    turn = TurnSpan(
        "turn-1",
        "conv-1",
        now,
        now,
        "agent",
        1,
        1,
        0,
        "OK",
        "cfg",
        "main",
        None,
        "sid",
        0,
        0,
        0,
        [],
        [],
        [],
        [],
        "request",
        "response",
    )
    plan = build_canonical_judging_plan(
        [SessionView("conv-1", [turn], "cfg", "main")],
        cohort_id="cohort-test",
        rubrics=effective.rubrics[:1],
        judge_models=effective.models.judges,
        context_policy=effective.judging_context,
    )

    pinned = store.pin_judging_plan(started.run_id, plan)

    assert pinned.judging_plan.sessions[0].reviewers[0].status == "skipped"
    assert judging_plan_totals(pinned.judging_plan)["maximum_reviewer_attempts"] == 0


def _judge_call(*, request_id: str = "sha256:" + "a" * 64, reusable: bool = True):
    return JudgeCallRecord(
        request_id=request_id,
        phase="window",
        conversation_id="conv-1",
        reviewer_position=1,
        requested_model_id="codex:gpt-5.6-sol",
        rubric_id="correctness",
        status="succeeded",
        reusable=reusable,
        result={"score": 0.8} if reusable else None,
        audit=JudgeCallAudit(
            schema_name="window_findings",
            transport_request_count=1,
        ),
        created_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
    )


def test_judge_calls_are_immutable_and_only_reusable_successes_are_loaded(store):
    started, _, _ = _start(store)
    store.record_stage_result(started.run_id, stage=RunStatus.SCORING, result={"written": 1})
    store.finalize_stage_success(started.run_id, stage=RunStatus.SCORING, advance=True)
    reusable = _judge_call()
    audit_only = _judge_call(request_id="sha256:" + "b" * 64, reusable=False)

    assert store.record_judge_call(started.run_id, reusable) == reusable
    assert store.record_judge_call(started.run_id, reusable) == reusable
    store.record_judge_call(started.run_id, audit_only)

    assert store.get_judge_call(started.run_id, reusable.request_id) == reusable
    assert store.get_judge_call(started.run_id, audit_only.request_id) is None
    assert store.list_judge_calls(started.run_id) == [reusable, audit_only]
    changed = reusable.model_copy(update={"result": {"score": 0.1}})
    with pytest.raises(RunStoreConflictError, match="different content"):
        store.record_judge_call(started.run_id, changed)


def test_delete_removes_terminal_run_and_owned_rows(store):
    started, _, _ = _start(store)
    store.record_stage_result(started.run_id, stage=RunStatus.SCORING, result={"written": 1})
    store.finalize_stage_success(started.run_id, stage=RunStatus.SCORING, advance=True)
    store.record_judge_call(started.run_id, _judge_call())
    store.append_run_event(
        started.run_id,
        sanitize_event(
            "judging",
            "working",
            "Reviewing",
            {},
            datetime(2026, 7, 17, tzinfo=timezone.utc),
        ),
    )
    store.fail(started.run_id, expected_status=RunStatus.JUDGING, error="test failure")

    store.delete(started.run_id)

    assert store.get(started.run_id) is None
    assert store.list_judge_calls(started.run_id) == []
    assert store.list_run_events(started.run_id) == []


def test_delete_rejects_active_and_missing_runs(store):
    started, _, _ = _start(store)

    with pytest.raises(RunStoreConflictError, match="cancel it before deleting"):
        store.delete(started.run_id)
    with pytest.raises(ValueError, match="Run missing not found"):
        store.delete("missing")


def test_run_events_receive_monotonic_sequences_and_prune_oldest_rows(store):
    run = store.create()
    for index in range(105):
        store.append_run_event(
            run.run_id,
            sanitize_event(
                "reflecting",
                "working",
                f"Event {index}",
                {},
                datetime(2026, 7, 17, tzinfo=timezone.utc),
            ),
        )

    events = store.list_run_events(run.run_id, stage="reflecting")

    assert len(events) == 100
    assert [event.sequence for event in events] == list(range(6, 106))


def test_review_revision_cas_round_trips_promotion_receipt(store):
    reflecting, _, _ = _advance_to_reflecting(store)
    evidence = _reflection_evidence()
    store.record_stage_result(
        reflecting.run_id,
        stage=RunStatus.REFLECTING,
        result=evidence,
    )
    pending = store.initialize_reflection_review(
        reflecting.run_id,
        {"status": "pending", "selected_candidate_id": "candidate-1"},
        expected_revision=0,
    )
    assert pending.reflection_review_revision == 1
    store.finalize_stage_success(
        reflecting.run_id,
        stage=RunStatus.REFLECTING,
        advance=True,
    )
    receipt = {
        "receipt_id": "promotion-1",
        "actions": [{"action": "update", "locator": "AGENTS.md"}],
        "promoted_revision": "candidate-rev",
    }
    promoted = store.update_reflection_review(
        reflecting.run_id,
        {
            "status": "promoted",
            "selected_candidate_id": "candidate-1",
            "promotion_receipt": receipt,
        },
        expected_revision=1,
    )

    assert promoted.reflection_review_revision == 2
    assert promoted.reflection_review["promotion_receipt"] == receipt
    with pytest.raises(ReflectionReviewRevisionConflictError) as exc_info:
        store.update_reflection_review(
            reflecting.run_id,
            {"status": "pending", "selected_candidate_id": "candidate-1"},
            expected_revision=1,
        )
    assert exc_info.value.current_revision == 2


def test_review_initialization_requires_finalized_matching_candidate(store):
    reflecting, _, _ = _advance_to_reflecting(store)
    with pytest.raises(ReflectionReviewLifecycleConflictError, match="not finalized"):
        store.initialize_reflection_review(
            reflecting.run_id,
            {"status": "pending", "selected_candidate_id": "candidate-1"},
            expected_revision=0,
        )

    store.record_stage_result(
        reflecting.run_id,
        stage=RunStatus.REFLECTING,
        result=_reflection_evidence(),
    )
    with pytest.raises(ReflectionReviewLifecycleConflictError, match="selected candidate"):
        store.initialize_reflection_review(
            reflecting.run_id,
            {"status": "pending", "selected_candidate_id": "missing"},
            expected_revision=0,
        )


def test_reflecting_result_blocks_cancellation_and_is_immutable_after_review(store):
    reflecting, _, _ = _advance_to_reflecting(store)
    evidence = _reflection_evidence()
    store.record_stage_result(
        reflecting.run_id,
        stage=RunStatus.REFLECTING,
        result=evidence,
    )
    with pytest.raises(RunCancellationConflictError) as exc_info:
        store.cancel_if_safe(reflecting.run_id)
    assert exc_info.value.reflection_finalizing is True

    store.initialize_reflection_review(
        reflecting.run_id,
        {"status": "pending", "selected_candidate_id": "candidate-1"},
        expected_revision=0,
    )
    with pytest.raises(ReflectionReviewLifecycleConflictError, match="immutable"):
        store.record_stage_result(
            reflecting.run_id,
            stage=RunStatus.REFLECTING,
            result=evidence.model_copy(
                update={
                    "evaluations": (
                        evidence.evaluations[0].model_copy(update={"rationale": "changed"}),
                        *evidence.evaluations[1:],
                    )
                }
            ),
        )
