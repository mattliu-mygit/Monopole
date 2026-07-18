from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from weave_agent_signals.catalogs import build_model_catalog
from weave_agent_signals.runs.bundles import BundleSnapshot, TargetSnapshot
from weave_agent_signals.runs.promotion import TargetPromoter
from weave_agent_signals.runs.reflection_records import (
    EvaluatorRecord,
    GenerationAttemptRecord,
    ReflectionResultRecord,
)
from weave_agent_signals.runs.review import (
    ReviewConflictError,
    ReviewRequestError,
    ReviewService,
)
from weave_agent_signals.runs.store import (
    ReflectionReviewLifecycleConflictError,
    ReflectionReviewRevisionConflictError,
    Run,
    RunStatus,
)
from weave_agent_signals.runs.targets import load_target_registry


class FakeStore:
    def __init__(self, run: Run) -> None:
        self.run = run
        self.update_error: Exception | None = None

    def get(self, run_id: str) -> Run | None:
        return self.run if self.run.run_id == run_id else None

    def update_reflection_review(
        self,
        run_id: str,
        review: dict,
        *,
        expected_revision: int,
    ) -> Run:
        if self.update_error is not None:
            error, self.update_error = self.update_error, None
            raise error
        if expected_revision != self.run.reflection_review_revision:
            raise ReflectionReviewRevisionConflictError(
                run_id,
                expected_revision,
                self.run.reflection_review_revision,
                self.run.reflection_review,
            )
        if self.run.status is not RunStatus.COMPLETE:
            current = self.run.reflection_review or {}
            raise ReflectionReviewLifecycleConflictError(
                run_id,
                "review mutations require a completed run",
                current_run_status=self.run.status,
                current_review_status=current.get("status"),
                current_revision=self.run.reflection_review_revision,
            )
        self.run = replace(
            self.run,
            reflection_review=dict(review),
            reflection_review_revision=expected_revision + 1,
        )
        return self.run


def _contents(bundle: BundleSnapshot) -> dict[str, str]:
    return {
        target.locator: target.content
        for target in bundle.targets
        if target.exists and target.content is not None
    }


def _candidate(baseline: BundleSnapshot, locator: str, content: str) -> BundleSnapshot:
    targets = [
        TargetSnapshot(
            kind=target.kind,
            locator=target.locator,
            exists=True,
            content=content if target.locator == locator else target.content,
            display_name=target.display_name,
            path=target.path,
        )
        for target in baseline.targets
    ]
    return BundleSnapshot(tuple(targets), baseline.scope)


def _context(tmp_path, *, status: RunStatus = RunStatus.COMPLETE):
    (tmp_path / "skills" / "audit").mkdir(parents=True)
    (tmp_path / "CLAUDE.md").write_text("past\n")
    (tmp_path / "skills" / "audit" / "SKILL.md").write_text("audit past\n")
    registry_path = tmp_path / "targets.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "targets": [
                    {"kind": "file", "id": "claude", "path": "CLAUDE.md"},
                    {"kind": "skill_collection", "id": "skills", "root": "skills"},
                ],
            }
        )
    )
    adapter = TargetPromoter(load_target_registry(registry_path))
    baseline = adapter.capture()
    candidate_one = _candidate(baseline, "file:claude", "candidate one\n")
    candidate_two = _candidate(baseline, "file:claude", "candidate two\n")
    writer = next(
        model
        for model in build_model_catalog().available_models
        if "proposal_writer" in model.supported_roles
    )
    attempts = tuple(
        GenerationAttemptRecord(
            attempt_id=f"attempt-{index}",
            number=index,
            status="succeeded",
            requested_writer=writer,
            resolved_model=writer.provider_model,
            resolved_family=writer.family,
            resolved_backend=writer.provider,
            candidate_id=f"candidate-{index}",
            bundle=bundle,
            changed_paths=("file:claude",),
        )
        for index, bundle in enumerate((candidate_one, candidate_two), start=1)
    )
    evaluations = tuple(
        EvaluatorRecord(
            evaluation_id=f"evaluation-{index}",
            target_id=target_id,
            requested_model="evaluator",
            requested_family="evaluator-family",
            requested_backend="test",
            resolved_model="evaluator",
            resolved_family="evaluator-family",
            resolved_backend="test",
            score=score,
            rationale="test",
        )
        for index, (target_id, score) in enumerate(
            (("baseline", 0.1), ("candidate-1", 0.8), ("candidate-2", 0.7)), start=1
        )
    )
    result = ReflectionResultRecord(
        baseline=baseline,
        attempts=attempts,
        evaluations=evaluations,
        recommended_candidate_id="candidate-1",
        baseline_won=False,
    )
    run = Run(
        run_id="run-1",
        status=status,
        created_at="2026-07-14T12:00:00+00:00",
        reflecting_result=result,
        reflection_review={
            "status": "pending",
            "selected_candidate_id": "candidate-1",
            "draft": None,
        },
        reflection_review_revision=1,
    )
    store = FakeStore(run)
    service = ReviewService(
        store=store,
        adapter_factory=lambda: TargetPromoter(load_target_registry(registry_path)),
        clock=lambda: datetime(2026, 7, 14, 12, tzinfo=timezone.utc),
    )
    return service, store, adapter, baseline, candidate_one, candidate_two


def test_selects_c_and_requires_explicit_draft_discard(tmp_path):
    service, store, _adapter, _baseline, candidate, _other = _context(tmp_path)
    saved = service.save_draft(
        "run-1",
        contents={
            **_contents(candidate),
            "file:claude": "edited D\n",
        },
        expected_revision=1,
        expected_draft_revision=None,
    )

    with pytest.raises(ReviewConflictError, match="Discard") as caught:
        service.select_candidate(
            "run-1",
            candidate_id="candidate-2",
            expected_revision=saved.reflection_review_revision,
        )
    assert caught.value.code == "dirty_draft"

    selected = service.select_candidate(
        "run-1",
        candidate_id="candidate-2",
        expected_revision=saved.reflection_review_revision,
        discard_draft=True,
    )
    assert selected.reflection_review == {
        "status": "pending",
        "selected_candidate_id": "candidate-2",
        "draft": None,
    }
    assert store.run.reflection_review_revision == 3


def test_saves_and_resets_content_only_d_with_two_revisions(tmp_path):
    service, _store, adapter, _baseline, candidate, _other = _context(tmp_path)
    draft = _candidate(candidate, "file:claude", "edited D\n")
    saved = service.save_draft(
        "run-1",
        contents={
            **_contents(candidate),
            "file:claude": "edited D\n",
        },
        expected_revision=1,
        expected_draft_revision=None,
    )
    assert saved.reflection_review["draft"] == {
        "candidate_id": "candidate-1",
        "bundle": draft.to_dict(),
        "revision": draft.revision,
    }

    with pytest.raises(ReviewConflictError) as caught:
        service.reset_draft(
            "run-1",
            expected_revision=2,
            expected_draft_revision="sha256:stale",
        )
    assert caught.value.code == "reflection_draft_conflict"

    reset = service.reset_draft(
        "run-1",
        expected_revision=2,
        expected_draft_revision=draft.revision,
    )
    assert reset.reflection_review["draft"] is None
    assert reset.reflection_review_revision == 3


def test_rejects_d_content_that_changes_target_membership(tmp_path):
    service, _store, _adapter, _baseline, candidate, _other = _context(tmp_path)
    incomplete = _contents(candidate)
    incomplete.pop("skills:skills/audit/SKILL.md")

    with pytest.raises(ReviewRequestError) as caught:
        service.save_draft(
            "run-1",
            contents=incomplete,
            expected_revision=1,
            expected_draft_revision=None,
        )
    assert caught.value.code == "invalid_reflection_draft"


def test_reflection_finalizing_and_store_cas_are_typed_conflicts(tmp_path):
    service, store, _adapter, _baseline, _candidate, _other = _context(
        tmp_path, status=RunStatus.REFLECTING
    )
    with pytest.raises(ReviewConflictError) as finalizing:
        service.select_candidate("run-1", candidate_id="candidate-2", expected_revision=1)
    assert finalizing.value.code == "reflection_review_run_incomplete"

    store.run = replace(store.run, status=RunStatus.COMPLETE)
    store.update_error = ReflectionReviewRevisionConflictError(
        "run-1", 1, 2, store.run.reflection_review
    )
    with pytest.raises(ReviewConflictError) as raced:
        service.select_candidate("run-1", candidate_id="candidate-2", expected_revision=1)
    assert raced.value.code == "reflection_review_conflict"
    assert raced.value.context["current_revision"] == 2


def test_any_managed_baseline_drift_blocks_review_edits(tmp_path):
    service, _store, _adapter, _baseline, _candidate, _other = _context(tmp_path)
    (tmp_path / "skills" / "audit" / "SKILL.md").write_text("changed outside C\n")

    with pytest.raises(ReviewConflictError) as caught:
        service.select_candidate("run-1", candidate_id="candidate-2", expected_revision=1)
    assert caught.value.code == "baseline_stale"
    assert caught.value.context["changed_targets"] == ["skills:skills/audit/SKILL.md"]
    assert caught.value.context["current"] is not None


def test_read_derives_stale_overlay_without_persisting_it(tmp_path):
    service, store, _adapter, _baseline, _candidate, _other = _context(tmp_path)
    persisted_review = dict(store.run.reflection_review)
    (tmp_path / "skills" / "audit" / "SKILL.md").write_text("live change\n")

    view = service.read("run-1")

    assert view.reflection_review["stale"] is True
    assert view.reflection_review["changed_targets"] == ["skills:skills/audit/SKILL.md"]
    assert view.reflection_review["current"] is not None
    assert store.run.reflection_review == persisted_review
    assert store.run.reflection_review_revision == 1


def test_promotes_evaluated_c_and_persists_exact_receipt(tmp_path):
    service, store, _adapter, baseline, candidate, _other = _context(tmp_path)
    promoted = service.promote(
        "run-1",
        promotion_id="promotion-c",
        expected_revision=1,
        expected_draft_revision=None,
        acknowledge_unevaluated=False,
    )

    receipt = promoted.reflection_review["receipt"]
    assert (tmp_path / "CLAUDE.md").read_text() == "candidate one\n"
    assert receipt["past"] == baseline.to_dict()
    assert receipt["evaluated_candidate"] == candidate.to_dict()
    assert receipt["promoted"] == candidate.to_dict()
    assert receipt["promoted_was_evaluated"] is True
    assert receipt["unevaluated_d_acknowledged"] is False
    assert "proposed" not in receipt
    assert store.run.reflection_review_revision == 2

    assert (
        service.promote(
            "run-1",
            promotion_id="promotion-c",
            expected_revision=1,
            expected_draft_revision=None,
            acknowledge_unevaluated=False,
        )
        == promoted
    )
    with pytest.raises(ReviewConflictError) as retry_conflict:
        service.promote(
            "run-1",
            promotion_id="promotion-c",
            expected_revision=2,
            expected_draft_revision=None,
            acknowledge_unevaluated=False,
        )
    assert retry_conflict.value.code == "promotion_idempotency_conflict"


def test_d_requires_exact_acknowledgement_then_promotes(tmp_path):
    service, _store, adapter, _baseline, candidate, _other = _context(tmp_path)
    draft = _candidate(candidate, "file:claude", "edited D\n")
    service.save_draft(
        "run-1",
        contents={
            **_contents(candidate),
            "file:claude": "edited D\n",
        },
        expected_revision=1,
        expected_draft_revision=None,
    )

    with pytest.raises(ReviewRequestError) as missing_ack:
        service.promote(
            "run-1",
            promotion_id="promotion-d",
            expected_revision=2,
            expected_draft_revision=draft.revision,
            acknowledge_unevaluated=False,
        )
    assert missing_ack.value.code == "unevaluated_d_acknowledgement_required"

    promoted = service.promote(
        "run-1",
        promotion_id="promotion-d",
        expected_revision=2,
        expected_draft_revision=draft.revision,
        acknowledge_unevaluated=True,
    )
    receipt = promoted.reflection_review["receipt"]
    assert receipt["evaluated_candidate"] == candidate.to_dict()
    assert receipt["promoted"] == draft.to_dict()
    assert receipt["promoted_was_evaluated"] is False
    assert receipt["unevaluated_d_acknowledged"] is True

    with pytest.raises(ReviewConflictError) as retry_conflict:
        service.promote(
            "run-1",
            promotion_id="promotion-d",
            expected_revision=2,
            expected_draft_revision=None,
            acknowledge_unevaluated=True,
        )
    assert retry_conflict.value.code == "promotion_idempotency_conflict"
    assert (
        service.promote(
            "run-1",
            promotion_id="promotion-d",
            expected_revision=2,
            expected_draft_revision=draft.revision,
            acknowledge_unevaluated=True,
        )
        == promoted
    )


def test_partial_receipt_is_persisted_as_terminal_review(tmp_path, monkeypatch: pytest.MonkeyPatch):
    service, store, _adapter, baseline, candidate, _other = _context(tmp_path)
    candidate = _candidate(candidate, "skills:skills/audit/SKILL.md", "audit new\n")
    result = store.run.reflecting_result
    attempts = (
        result.attempts[0].model_copy(update={"bundle": candidate}),
        *result.attempts[1:],
    )
    store.run = replace(
        store.run,
        reflecting_result=result.model_copy(update={"attempts": attempts}),
    )
    real_publish = TargetPromoter._publish

    def fail_skill(self, item):
        if item.destination.name == "SKILL.md":
            raise OSError("disk full")
        real_publish(self, item)

    monkeypatch.setattr(TargetPromoter, "_publish", fail_skill)

    promoted = service.promote(
        "run-1",
        promotion_id="promotion-partial",
        expected_revision=1,
        expected_draft_revision=None,
        acknowledge_unevaluated=False,
    )

    assert promoted.reflection_review["status"] == "partial"
    assert promoted.reflection_review["receipt"]["outcomes"] == [
        {
            "locator": "file:claude",
            "action": "update",
            "status": "applied",
            "reason": None,
            "message": None,
        },
        {
            "locator": "skills:skills/audit/SKILL.md",
            "action": "update",
            "status": "not_applied",
            "reason": "write_failed",
            "message": "disk full",
        },
    ]
    with pytest.raises(ReviewConflictError) as caught:
        service.dismiss("run-1", expected_revision=2)
    assert caught.value.code == "reflection_review_resolved"
    assert baseline.target("file:claude").content == "past\n"


def test_dismisses_without_mutating_files(tmp_path):
    service, _store, _adapter, _baseline, _candidate, _other = _context(tmp_path)
    dismissed = service.dismiss("run-1", expected_revision=1)

    assert dismissed.reflection_review["status"] == "dismissed"
    assert dismissed.reflection_review["dismissed_at"] == "2026-07-14T12:00:00+00:00"
    assert (tmp_path / "CLAUDE.md").read_text() == "past\n"


def test_receipt_persistence_failure_does_not_roll_back_completed_files(tmp_path):
    service, store, _adapter, _baseline, _candidate, _other = _context(tmp_path)
    store.update_error = RuntimeError("database unavailable")

    with pytest.raises(Exception, match="receipt could not be persisted") as caught:
        service.promote(
            "run-1",
            promotion_id="promotion-unrecorded",
            expected_revision=1,
            expected_draft_revision=None,
            acknowledge_unevaluated=False,
        )
    assert getattr(caught.value, "code") == "promotion_receipt_persist_failed"
    assert (tmp_path / "CLAUDE.md").read_text() == "candidate one\n"
    assert not (tmp_path / ".weave-agent-signals").exists()
