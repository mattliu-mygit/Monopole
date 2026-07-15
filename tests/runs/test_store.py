from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.run_config import RunConfig, resolve_run_config
from weave_agent_signals.runs.store import (
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
        review_depth="selective",
        judge_models=("claude-sonnet-5", "gpt-5.6-sol"),
        second_opinion_margin=0.1,
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


def _started_judging_run(store: RunStore):
    started, _, _ = _start(store)
    store.record_stage_result(
        started.run_id,
        stage=RunStatus.SCORING,
        result={"written": 1},
    )
    return store.finalize_stage_success(
        started.run_id,
        stage=RunStatus.SCORING,
        advance=True,
    )


def _payload_digest(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _judging_plan() -> dict:
    session_rubric = next(
        rubric for rubric in build_rubric_catalog().rubrics if rubric.evaluation_unit == "session"
    )
    body = {
        "schema_version": "1",
        "cohort_id": "cohort-test",
        "requested_rubrics": [session_rubric.model_dump(mode="json")],
        "review_depth": "selective",
        "judge_count": 2,
        "max_episodes_per_session": 8,
        "sessions": [
            {
                "conversation_id": "conv-1",
                "turn_count": 1,
                "omitted_turn_count": 0,
                "session_rubrics": [
                    {
                        **session_rubric.model_dump(mode="json"),
                        "applicability": "applicable",
                        "minimum_reviewer_attempts": 1,
                        "maximum_reviewer_attempts": 2,
                    }
                ],
                "selected_episodes": [
                    {
                        "trace_id": "turn-1",
                        "turn_index": 0,
                        "selection_kind": "deterministic_trigger",
                        "selection_reasons": ["terminal"],
                        "evidence_trace_ids": ["turn-1"],
                        "rubrics": [],
                    }
                ],
            }
        ],
        "totals": {
            "turns_considered": 1,
            "episodes_selected": 1,
            "planned_episode_rubrics": 0,
            "planned_session_rubrics": 1,
            "planned_rubrics": 1,
            "minimum_episode_reviewer_attempts": 0,
            "maximum_episode_reviewer_attempts": 0,
            "minimum_session_reviewer_attempts": 1,
            "maximum_session_reviewer_attempts": 2,
            "minimum_reviewer_attempts": 1,
            "maximum_reviewer_attempts": 2,
        },
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {"plan_id": f"sha256:{hashlib.sha256(canonical).hexdigest()}", **body}


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


def test_schema_v5_resets_disposable_database_on_version_mismatch(tmp_path):
    path = tmp_path / "runs.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, obsolete TEXT)")
    connection.execute("INSERT INTO runs VALUES ('legacy', '{}')")
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()

    store = RunStore(path)
    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(runs)").fetchall()}

    assert RUN_DB_SCHEMA_VERSION == 5
    assert store.get("legacy") is None
    assert {"run_id", "run_config", "effective_config", "judging_artifacts"} <= columns
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 5
    store.close()


def test_judging_artifacts_are_content_addressed_and_idempotent(store):
    run = _started_judging_run(store)
    payload = {"chunk_id": "chunk-1", "text": "cited digest"}
    artifact = {
        "schema_version": "1",
        "kind": "chunk_digest",
        "content_digest": _payload_digest(payload),
        "payload": payload,
    }
    artifact_id = "judge-1/session-1/digest/chunk-1"

    first = store.record_judging_artifact(run.run_id, artifact_id, artifact)
    second = store.record_judging_artifact(run.run_id, artifact_id, artifact)

    assert first.judging_artifacts == {artifact_id: artifact}
    assert second.judging_artifacts == first.judging_artifacts

    changed_payload = {"chunk_id": "chunk-1", "text": "changed"}
    changed = {
        **artifact,
        "content_digest": _payload_digest(changed_payload),
        "payload": changed_payload,
    }
    with pytest.raises(
        RunStoreConflictError,
        match="artifact already exists with different content",
    ):
        store.record_judging_artifact(run.run_id, artifact_id, changed)


@pytest.mark.parametrize(
    "artifact_id",
    ["", " ", "artifact", "/session/digest/chunk-1", "judge-1//digest/chunk-1"],
)
def test_judging_artifact_ids_must_be_nonblank_slash_delimited(store, artifact_id):
    run = _started_judging_run(store)
    payload = {"chunk_id": "chunk-1"}
    artifact = {
        "schema_version": "1",
        "kind": "chunk_digest",
        "content_digest": _payload_digest(payload),
        "payload": payload,
    }

    with pytest.raises(ValueError, match="artifact ID"):
        store.record_judging_artifact(run.run_id, artifact_id, artifact)


def test_judging_artifacts_require_exact_valid_content_digest(store):
    run = _started_judging_run(store)
    payload = {"chunk_id": "chunk-1"}
    artifact = {
        "schema_version": "1",
        "kind": "chunk_digest",
        "content_digest": _payload_digest(payload),
        "payload": payload,
    }

    with pytest.raises(ValueError, match="exactly"):
        store.record_judging_artifact(
            run.run_id,
            "judge-1/session-1/digest/chunk-1",
            {**artifact, "extra": True},
        )
    with pytest.raises(ValueError, match="content digest"):
        store.record_judging_artifact(
            run.run_id,
            "judge-1/session-1/digest/chunk-1",
            {**artifact, "content_digest": "sha256:" + ("0" * 64)},
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"schema_version": " "}, "schema_version"),
        ({"kind": ""}, "kind"),
        ({"content_digest": 1}, "content_digest"),
        ({"payload": {"value": float("nan")}}, "JSON serializable"),
    ],
)
def test_judging_artifact_body_fields_are_strict(store, changes, match):
    run = _started_judging_run(store)
    payload = {"chunk_id": "chunk-1"}
    artifact = {
        "schema_version": "1",
        "kind": "chunk_digest",
        "content_digest": _payload_digest(payload),
        "payload": payload,
        **changes,
    }

    with pytest.raises(ValueError, match=match):
        store.record_judging_artifact(
            run.run_id,
            "judge-1/session-1/digest/chunk-1",
            artifact,
        )


def test_judging_artifacts_can_only_be_written_while_judging(store):
    run, _, _ = _start(store)
    payload = {"chunk_id": "chunk-1"}
    artifact = {
        "schema_version": "1",
        "kind": "chunk_digest",
        "content_digest": _payload_digest(payload),
        "payload": payload,
    }

    with pytest.raises(RunStoreConflictError, match="only record.*while judging"):
        store.record_judging_artifact(
            run.run_id,
            "judge-1/session-1/digest/chunk-1",
            artifact,
        )


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
