from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.judges.plan import build_judging_plan
from weave_agent_signals.models import SessionView, TurnSpan
from weave_agent_signals.run_config import RunConfig, resolve_run_config
from weave_agent_signals.runs.store import (
    _SCHEMA,
    RUN_DB_SCHEMA_VERSION,
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
        judge_backend="cli",
        judge_models=("claude-sonnet-5", "gpt-5.6-sol"),
        proposal_model="gpt-5.6-sol",
        proposal_evaluator_model="claude-sonnet-5",
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
    return build_judging_plan(
        [SessionView("conv-1", [turn], "cfg", "main")],
        cohort_id="cohort-test",
        rubrics=effective.rubrics[:1],
        judge_models=effective.models.judges,
        context_policy=effective.judging_context,
    )


def _reflection_evidence() -> dict:
    return {
        "baseline": {"revision": "baseline-rev"},
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "revision": "candidate-rev",
            }
        ],
        "recommended_candidate_id": "candidate-1",
    }


def test_schema_v6_resets_disposable_database_on_version_mismatch(tmp_path):
    path = tmp_path / "runs.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, obsolete TEXT)")
    connection.execute("INSERT INTO runs VALUES ('legacy', '{}')")
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()

    store = RunStore(path)
    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(runs)").fetchall()}

    assert RUN_DB_SCHEMA_VERSION == 6
    assert store.get("legacy") is None
    assert {"run_id", "run_config", "effective_config"} <= columns
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 6
    store.close()


def test_schema_v6_resets_collided_v5_database_missing_judging_artifacts(tmp_path):
    path = tmp_path / "runs.db"
    connection = sqlite3.connect(path)
    legacy_schema = _SCHEMA.replace("    judging_artifacts TEXT,\n", "")
    connection.execute(legacy_schema)
    connection.execute(
        "INSERT INTO runs (run_id, status, created_at) VALUES (?, ?, ?)",
        ("legacy-active", "judging", "2026-07-15T00:00:00+00:00"),
    )
    connection.execute("PRAGMA user_version = 5")
    connection.commit()
    connection.close()

    store = RunStore(path)
    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(runs)").fetchall()}

    assert RUN_DB_SCHEMA_VERSION == 6
    assert "judging_artifacts" in columns
    assert store.list_active() == []
    store.close()


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
    changed_plan = {**plan, "schema_version": "changed"}
    with pytest.raises(ValueError, match="schema_version"):
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
    reflection_input = {
        "feedback": [{"feedback_id": "feedback-1", "weave_ref": "weave:///turn-1"}],
        "feedback_count": 1,
    }
    pinned = store.pin_reflection_input(started.run_id, reflection_input)
    assert pinned.reflection_input == reflection_input
    assert store.pin_reflection_input(started.run_id, reflection_input).reflection_input == (
        reflection_input
    )
    with pytest.raises(RunStoreConflictError, match="already pinned"):
        store.pin_reflection_input(
            started.run_id,
            {"feedback": [], "feedback_count": 0},
        )


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
            result={"candidates": []},
        )
