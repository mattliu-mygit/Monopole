from __future__ import annotations

import json
from concurrent.futures import Future

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.routes.models import ReflectingProgressResponse, ReviewAttemptResponse
from weave_agent_signals.routes.runs import create_runs_router
from weave_agent_signals.run_config import RunConfig
from weave_agent_signals.runs.service import RunService
from weave_agent_signals.runs.store import (
    RunStatus,
    RunStore,
    RunSummarySource,
)


class SynchronousExecutor:
    def submit(self, function, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


def test_route_attempt_model_exposes_skipped_disposition() -> None:
    attempt = ReviewAttemptResponse.model_validate(
        {
            "position": 1,
            "role": "judge",
            "trigger": "panel",
            "requested_model": "small",
            "requested_family": "family",
            "requested_backend": "cli",
            "status": "skipped",
            "skip_reason": "insufficient_context_capacity",
            "resolved_model": None,
            "resolved_family": None,
            "score": None,
            "rationale": None,
            "evidence_ids": [],
            "usage": {},
            "output_mode": None,
            "schema_name": None,
            "schema_fallback_reason": None,
            "transport_request_count": 0,
            "verdict_schema_version": None,
            "raw_output_digest": None,
            "error_type": None,
            "message": None,
            "behavioral_feedback": None,
            "steps": [],
        }
    )

    assert attempt.status == "skipped"
    assert attempt.skip_reason == "insufficient_context_capacity"


def test_route_attempt_exposes_ordered_response_diagnostics() -> None:
    attempt = ReviewAttemptResponse.model_validate(
        {
            "position": 1,
            "role": "judge",
            "trigger": "panel",
            "requested_model": "wandb:qwen",
            "requested_family": "qwen",
            "requested_backend": "wandb",
            "status": "failed",
            "resolved_model": "Qwen/test",
            "resolved_family": "qwen",
            "score": None,
            "rationale": None,
            "evidence_ids": [],
            "usage": {"completion_tokens": 20_000},
            "output_mode": "json_schema",
            "schema_name": "window_findings",
            "schema_fallback_reason": None,
            "transport_request_count": 2,
            "verdict_schema_version": None,
            "raw_output_digest": "a" * 64,
            "error_type": "InferenceOutputExceeded",
            "message": "provider output limit exhausted",
            "behavioral_feedback": None,
            "steps": [
                {
                    "phase": "window",
                    "artifact_id": "sha256:" + "b" * 64,
                    "requested_model": "wandb:qwen",
                    "resolved_model": "Qwen/test",
                    "usage": {"completion_tokens": 20_000},
                    "output_mode": "json_schema",
                    "schema_name": "window_findings",
                    "transport_request_count": 2,
                    "raw_output_digest": "a" * 64,
                    "response_diagnostics": [
                        {
                            "finish_reason": "length",
                            "usage": {"completion_tokens": 10_000},
                            "completion_details": {"reasoning_tokens": 9_500},
                            "content_characters": 0,
                        }
                    ],
                    "reused": False,
                }
            ],
        }
    )

    assert attempt.steps[0].response_diagnostics[0].finish_reason == "length"
    assert attempt.steps[0].response_diagnostics[0].completion_details == {
        "reasoning_tokens": 9_500
    }


def test_reflecting_progress_exposes_pinned_model_context() -> None:
    progress = ReflectingProgressResponse.model_validate(
        {
            "phase": "candidate_started",
            "status_message": "Generating candidate 1",
            "started_at": "2026-07-14T12:00:00+00:00",
            "attempted": 0,
            "valid": 0,
            "rejected": 0,
            "scored": 0,
            "total_attempts": 3,
            "proposal_writer": "gpt-5.6-sol",
            "proposal_evaluator": "claude-sonnet-5",
            "no_improvement_patience": 2,
            "events": [],
        }
    )

    assert progress.proposal_writer == "gpt-5.6-sol"
    assert progress.proposal_evaluator == "claude-sonnet-5"
    assert progress.no_improvement_patience == 2


@pytest.fixture
def route_context(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    models = build_model_catalog(which=lambda name: f"/bin/{name}")
    rubrics = build_rubric_catalog()

    def cohort(selection):
        conversation_id = selection.session_ids[0]
        return {
            "schema_version": 1,
            "pinned_at": "2026-07-14T12:00:00+00:00",
            "cohort_id": "sha256:test-cohort",
            "turn_count": 1,
            "session_count": 1,
            "turns": [
                {
                    "trace_id": "trace-1",
                    "weave_ref": ("weave:///weave-team/agent-sessions/agent_turn/trace-1"),
                    "conversation_id": conversation_id,
                    "started_at": "2026-07-14T11:00:00+00:00",
                    "model": None,
                    "model_family": "unknown",
                }
            ],
            "sessions": [
                {
                    "conversation_id": conversation_id,
                    "weave_ref": (
                        f"weave:///weave-team/agent-sessions/agent_conversation/{conversation_id}"
                    ),
                    "turn_count": 1,
                }
            ],
        }

    def stage(run, _config, _cancel):
        if run.status is RunStatus.JUDGING:
            store.record_stage_result(
                run.run_id,
                stage=run.status,
                result={
                    "plan_id": "plan-1",
                    "planned_rubrics": 0,
                    "rubrics_completed": 0,
                    "rated_rubrics": 0,
                    "not_evaluable_rubrics": 0,
                    "minimum_reviewer_attempts": 0,
                    "maximum_reviewer_attempts": 0,
                    "reviewer_attempts_completed": 0,
                    "digest_steps_completed": 0,
                    "maximum_digest_steps": 0,
                    "window_steps_completed": 0,
                    "maximum_window_steps": 0,
                    "merge_steps_completed": 0,
                    "maximum_merge_steps": 0,
                    "scores_written": 0,
                    "failure_count": 0,
                    "write_failure_count": 0,
                    "coverage_complete": True,
                    "phase": "judging_complete",
                    "status_message": "Judging complete",
                    "started_at": "2026-07-14T12:00:00+00:00",
                    "events": [
                        {
                            "id": 1,
                            "at": "2026-07-14T12:00:00+00:00",
                            "phase": "judging_complete",
                            "message": "Judging complete",
                        }
                    ],
                    "attempt_summary_count": 0,
                    "attempt_summaries_truncated": False,
                    "attempt_summaries": [],
                    "failure_detail_count": 0,
                    "failure_details_truncated": False,
                    "failure_details": [],
                },
            )
            return
        store.record_stage_result(
            run.run_id,
            stage=run.status,
            result={
                "turns_scored": 0,
                "sessions_scored": 0,
                "scores_written": 0,
                "errors": 0,
                "turn_details": [],
            },
        )

    service = RunService(
        store=store,
        build_model_catalog=lambda: models,
        build_rubric_catalog=lambda: rubrics,
        discover_cohort=cohort,
        scoring_stage=stage,
        judging_stage=stage,
        reflection_stage=stage,
        executor=SynchronousExecutor(),
    )
    app = FastAPI()
    app.include_router(create_runs_router(service))
    with TestClient(app) as client:
        yield client, service, models, rubrics
    store.close()


def _config(models, rubrics, **updates) -> dict:
    value = RunConfig(
        model_catalog_version=models.catalog_version,
        rubric_catalog_version=rubrics.catalog_version,
        judge_models=("claude:claude-sonnet-5", "codex:gpt-5.6-sol"),
        challenge_judge_models=("codex:gpt-5.6-sol",),
        proposal_model="codex:gpt-5.6-sol",
        proposal_evaluator_model="claude:claude-sonnet-5",
        rubrics=("judge.verification", "judge.session_outcome"),
        candidate_budget=3,
        force=False,
    ).model_dump(mode="json")
    value.update(updates)
    return value


def test_run_routes_accept_exact_setup_contract_and_return_pinned_run(route_context):
    client, _service, models, rubrics = route_context
    created = client.post("/api/runs").json()

    selected = client.put(
        f"/api/runs/{created['run_id']}/selection",
        json={
            "since": "2026-07-01T00:00:00+00:00",
            "until": "2026-07-14T00:00:00+00:00",
            "timezone": "UTC",
            "session_ids": ["session-1"],
        },
    )
    configured = client.put(
        f"/api/runs/{created['run_id']}/config",
        json=_config(models, rubrics),
    )
    automatic = client.put(
        f"/api/runs/{created['run_id']}/auto_run",
        json={"auto_run": False},
    )
    advanced = client.post(f"/api/runs/{created['run_id']}/advance")

    assert selected.status_code == configured.status_code == automatic.status_code == 200
    assert advanced.status_code == 200
    body = advanced.json()
    assert body["status"] == "scoring"
    assert body["current_stage_succeeded"] is True
    assert body["effective_config"]["pipeline_version"]
    assert body["effective_config"]["rubrics"][0]["id"] == "judge.verification"
    assert body["scoring_result"] == {
        "turns_scored": 0,
        "sessions_scored": 0,
        "scores_written": 0,
        "errors": 0,
        "turn_details": [],
    }
    assert client.get("/api/runs").json()["runs"] == [
        {
            "run_id": created["run_id"],
            "status": "scoring",
            "current_stage_succeeded": True,
            "created_at": body["created_at"],
            "selection": {
                "session_count": 1,
                "since": "2026-07-01T00:00:00+00:00",
                "until": "2026-07-14T00:00:00+00:00",
                "timezone": "UTC",
            },
            "review_state": "none",
        }
    ]
    assert client.get(f"/api/runs/{created['run_id']}").json() == body


def test_run_detail_hides_internal_recovery_state(route_context):
    client, _service, models, rubrics = route_context
    created = client.post("/api/runs").json()
    client.put(
        f"/api/runs/{created['run_id']}/selection",
        json={"session_ids": ["session-1"]},
    )
    client.put(
        f"/api/runs/{created['run_id']}/config",
        json=_config(models, rubrics),
    )
    client.post(f"/api/runs/{created['run_id']}/advance")
    response = client.get(f"/api/runs/{created['run_id']}")

    assert response.status_code == 200
    assert "judging_artifacts" not in response.json()
    assert "reflection_input" not in response.json()


def test_run_list_uses_scalar_projection_without_evidence_or_live_review_overlay(
    tmp_path,
    monkeypatch,
):
    sentinel = "DO-NOT-LEAK-FULL-BUNDLE-CONTENT"
    large_content = sentinel + ("x" * 250_000)
    store = RunStore(tmp_path / "summary.db")
    run = store.create()
    store._conn.execute(
        """
        UPDATE runs
        SET run_id = ?, status = ?, created_at = ?, data_selection = ?, run_config = ?,
            reflection_input = ?, reflecting_result = ?, reflecting_succeeded = 1
        WHERE run_id = ?
        """,
        (
            "run-summary",
            RunStatus.COMPLETE.value,
            "2026-07-14T12:00:00+00:00",
            json.dumps(
                {
                    "since": "2026-07-01T00:00:00+00:00",
                    "until": None,
                    "timezone": "America/Los_Angeles",
                    "session_ids": ["session-1", "session-2"],
                }
            ),
            f"invalid full config {large_content}",
            json.dumps({"baseline": {"content": large_content}}),
            json.dumps(
                {
                    "baseline": {"content": large_content},
                    "baseline_won": True,
                    "reason": None,
                }
            ),
            run.run_id,
        ),
    )
    store._conn.commit()
    monkeypatch.setattr(
        store,
        "_row_to_run",
        lambda _row: pytest.fail("list route must not hydrate a full run"),
    )

    class Runs:
        def list_summaries(self, limit):
            return store.list_summaries(limit)

    class Reviews:
        def overlay(self, _run):
            raise AssertionError("list reads must not capture live review drift")

    app = FastAPI()
    app.include_router(create_runs_router(Runs(), Reviews()))
    with TestClient(app) as client:
        response = client.get("/api/runs")
    store.close()

    assert response.status_code == 200
    summary = response.json()["runs"][0]
    assert summary == {
        "run_id": "run-summary",
        "status": "complete",
        "current_stage_succeeded": True,
        "created_at": "2026-07-14T12:00:00+00:00",
        "selection": {
            "session_count": 2,
            "since": "2026-07-01T00:00:00+00:00",
            "until": None,
            "timezone": "America/Los_Angeles",
        },
        "review_state": "no-change",
    }
    assert sentinel not in response.text
    assert not {
        "auto_run",
        "data_selection",
        "run_config",
        "effective_config",
        "turn_cohort",
        "judging_plan",
        "judging_artifacts",
        "reflection_input",
        "scoring_progress",
        "scoring_result",
        "judging_progress",
        "judging_result",
        "reflecting_progress",
        "reflecting_result",
        "reflection_review",
        "reflection_review_revision",
        "error",
        "bundle",
        "baseline",
        "candidates",
        "generation_attempts",
        "attempts",
        "draft",
        "receipt",
    }.intersection(summary)


@pytest.mark.parametrize(
    ("review_status", "baseline_won", "reason", "expected"),
    [
        (None, None, None, "none"),
        ("pending", None, None, "review-needed"),
        ("promoted", None, None, "promoted"),
        ("dismissed", None, None, "dismissed"),
        (None, True, None, "no-change"),
        (None, None, "No evaluation feedback was found", "none"),
        (None, False, None, "none"),
        (None, False, "No valid proposal generated", "no-valid-proposal"),
    ],
)
def test_run_list_derives_all_review_states(
    review_status,
    baseline_won,
    reason,
    expected,
):
    summary_source = RunSummarySource(
        run_id="run-state",
        status=RunStatus.COMPLETE,
        created_at="2026-07-14T12:00:00+00:00",
        scoring_succeeded=True,
        judging_succeeded=True,
        reflecting_succeeded=True,
        session_count=None,
        selection_since=None,
        selection_until=None,
        selection_timezone=None,
        review_status=review_status,
        baseline_won=baseline_won,
        reflection_reason=reason,
    )

    class Runs:
        def list_summaries(self, _limit):
            return [summary_source]

    app = FastAPI()
    app.include_router(create_runs_router(Runs()))
    with TestClient(app) as client:
        summary = client.get("/api/runs").json()["runs"][0]

    assert summary["review_state"] == expected
    assert summary["selection"] is None


def test_run_routes_translate_typed_conflicts(route_context):
    client, _service, models, rubrics = route_context
    created = client.post("/api/runs").json()
    run_id = created["run_id"]

    assert client.post(f"/api/runs/{run_id}/advance").status_code == 400
    assert client.get("/api/runs/missing").status_code == 404

    client.put(
        f"/api/runs/{run_id}/selection",
        json={"session_ids": ["session-1"]},
    )
    client.put(f"/api/runs/{run_id}/config", json=_config(models, rubrics))
    client.post(f"/api/runs/{run_id}/advance")
    conflict = client.put(
        f"/api/runs/{run_id}/config",
        json=_config(models, rubrics),
    )

    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "run_lifecycle_conflict"
    assert conflict.json()["detail"]["current_status"] == "scoring"


def test_run_actions_are_bodyless_and_cancel_is_service_owned(route_context):
    client, _service, _models, _rubrics = route_context
    paths = client.get("/openapi.json").json()["paths"]
    assert "requestBody" not in paths["/api/runs"]["post"]
    assert "requestBody" not in paths["/api/runs/{run_id}"]["delete"]
    assert "requestBody" not in paths["/api/runs/{run_id}/advance"]["post"]
    assert "requestBody" not in paths["/api/runs/{run_id}/cancel"]["post"]

    created = client.post("/api/runs").json()
    cancelled = client.post(f"/api/runs/{created['run_id']}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == RunStatus.CANCELLED.value
    repeated = client.post(f"/api/runs/{created['run_id']}/cancel")
    assert repeated.status_code == 409
    assert repeated.json()["detail"]["code"] == "run_cancellation_conflict"


def test_run_delete_removes_inactive_runs_and_rejects_active_runs(route_context):
    client, _service, models, rubrics = route_context
    inactive = client.post("/api/runs").json()

    deleted = client.delete(f"/api/runs/{inactive['run_id']}")

    assert deleted.status_code == 204
    assert deleted.content == b""
    assert client.get(f"/api/runs/{inactive['run_id']}").status_code == 404
    assert client.delete("/api/runs/missing").status_code == 404

    active = client.post("/api/runs").json()
    client.put(
        f"/api/runs/{active['run_id']}/selection",
        json={"session_ids": ["session-1"]},
    )
    client.put(f"/api/runs/{active['run_id']}/config", json=_config(models, rubrics))
    client.post(f"/api/runs/{active['run_id']}/advance")

    conflict = client.delete(f"/api/runs/{active['run_id']}")
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "run_conflict"
