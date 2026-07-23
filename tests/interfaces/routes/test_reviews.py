from __future__ import annotations

from dataclasses import replace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from weave_agent_signals.routes.reviews import create_reviews_router
from weave_agent_signals.routes.runs import create_runs_router
from weave_agent_signals.runs.review import (
    ReviewConflictError,
    ReviewNotFoundError,
    ReviewOperationError,
    ReviewRequestError,
)
from weave_agent_signals.runs.store import Run, RunStatus, RunSummarySource


class StubService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.error: Exception | None = None
        self.run = Run(
            run_id="run-1",
            status=RunStatus.COMPLETE,
            created_at="2026-07-14T12:00:00+00:00",
            reflection_review={
                "status": "pending",
                "selected_candidate_id": "candidate-1",
                "draft": None,
            },
            reflection_review_revision=1,
        )

    def _call(self, name: str, **values) -> Run:
        if self.error is not None:
            raise self.error
        self.calls.append((name, values))
        return self.run

    def select_candidate(self, run_id: str, **values) -> Run:
        return self._call("select_candidate", run_id=run_id, **values)

    def save_draft(self, run_id: str, **values) -> Run:
        return self._call("save_draft", run_id=run_id, **values)

    def reset_draft(self, run_id: str, **values) -> Run:
        return self._call("reset_draft", run_id=run_id, **values)

    def promote(self, run_id: str, **values) -> Run:
        return self._call("promote", run_id=run_id, **values)

    def dismiss(self, run_id: str, **values) -> Run:
        return self._call("dismiss", run_id=run_id, **values)


def _client(service: StubService) -> TestClient:
    app = FastAPI()
    app.include_router(create_reviews_router(service))
    return TestClient(app)


def test_review_routes_delegate_one_service_operation_per_request():
    service = StubService()
    contents = {"CLAUDE.md": "edited\n"}
    with _client(service) as client:
        responses = [
            client.put(
                "/api/runs/run-1/reflection_selection",
                json={
                    "candidate_id": "candidate-2",
                    "expected_revision": 1,
                    "discard_draft": True,
                },
            ),
            client.put(
                "/api/runs/run-1/reflection_draft",
                json={
                    "contents": contents,
                    "expected_revision": 2,
                    "expected_draft_revision": None,
                },
            ),
            client.request(
                "DELETE",
                "/api/runs/run-1/reflection_draft",
                json={
                    "expected_revision": 3,
                    "expected_draft_revision": "sha256:d",
                },
            ),
            client.post(
                "/api/runs/run-1/promote",
                json={
                    "expected_revision": 4,
                    "expected_draft_revision": "sha256:d",
                    "idempotency_key": "promotion-1",
                    "acknowledge_unevaluated": True,
                    "acknowledge_unverified": True,
                },
            ),
            client.post("/api/runs/run-1/dismiss", json={"expected_revision": 5}),
        ]

    assert [response.status_code for response in responses] == [200] * 5
    assert service.calls == [
        (
            "select_candidate",
            {
                "run_id": "run-1",
                "candidate_id": "candidate-2",
                "expected_revision": 1,
                "discard_draft": True,
            },
        ),
        (
            "save_draft",
            {
                "run_id": "run-1",
                "contents": contents,
                "expected_revision": 2,
                "expected_draft_revision": None,
            },
        ),
        (
            "reset_draft",
            {
                "run_id": "run-1",
                "expected_revision": 3,
                "expected_draft_revision": "sha256:d",
            },
        ),
        (
            "promote",
            {
                "run_id": "run-1",
                "promotion_id": "promotion-1",
                "expected_revision": 4,
                "expected_draft_revision": "sha256:d",
                "acknowledge_unevaluated": True,
                "acknowledge_unverified": True,
            },
        ),
        ("dismiss", {"run_id": "run-1", "expected_revision": 5}),
    ]


def test_review_routes_translate_typed_domain_errors():
    service = StubService()
    cases = [
        (ReviewNotFoundError("missing"), 404, "run_not_found"),
        (
            ReviewRequestError("invalid_reflection_draft", "bad draft"),
            400,
            "invalid_reflection_draft",
        ),
        (
            ReviewConflictError("baseline_stale", "stale", changed_targets=["CLAUDE.md"]),
            409,
            "baseline_stale",
        ),
        (
            ReviewOperationError(
                "promotion_receipt_persist_failed",
                "failed",
                manual_inspection_required=True,
            ),
            500,
            "promotion_receipt_persist_failed",
        ),
    ]
    with _client(service) as client:
        for error, status, code in cases:
            service.error = error
            response = client.post("/api/runs/run-1/dismiss", json={"expected_revision": 1})
            assert response.status_code == status
            assert response.json()["detail"]["code"] == code


def test_review_route_models_reject_unknown_or_coerced_fields():
    service = StubService()
    with _client(service) as client:
        unknown = client.post(
            "/api/runs/run-1/dismiss",
            json={"expected_revision": 1, "retired_option": True},
        )
        coerced = client.post("/api/runs/run-1/dismiss", json={"expected_revision": "1"})

    assert unknown.status_code == 422
    assert coerced.status_code == 422
    assert service.calls == []


def test_run_detail_delegates_recovery_and_drift_but_list_does_not_overlay():
    run = StubService().run

    class Runs:
        def list_summaries(self, _limit):
            return [
                RunSummarySource(
                    run_id=run.run_id,
                    status=run.status,
                    created_at=run.created_at,
                    scoring_succeeded=run.scoring_succeeded,
                    judging_succeeded=run.judging_succeeded,
                    reflecting_succeeded=run.reflecting_succeeded,
                    session_count=None,
                    selection_since=None,
                    selection_until=None,
                    selection_timezone=None,
                    review_status="pending",
                    baseline_won=None,
                    reflection_reason=None,
                )
            ]

    class Reviews:
        overlay_calls = 0

        def read(self, run_id):
            assert run_id == "run-1"
            return replace(
                run,
                reflection_review={
                    "status": "pending",
                    "selected_candidate_id": "candidate-1",
                    "draft": None,
                    "stale": True,
                },
            )

        def overlay(self, listed):
            self.overlay_calls += 1
            raise AssertionError("list must not derive live filesystem drift")

    reviews = Reviews()
    app = FastAPI()
    app.include_router(create_runs_router(Runs(), reviews))
    with TestClient(app) as client:
        listed = client.get("/api/runs").json()["runs"][0]
        detail = client.get("/api/runs/run-1").json()

    assert listed["review_state"] == "review-needed"
    assert "reflection_review" not in listed
    assert reviews.overlay_calls == 0
    assert detail["reflection_review"]["stale"] is True
