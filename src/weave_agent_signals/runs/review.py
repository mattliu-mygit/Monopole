"""Reflection review decisions over canonical A/B/C instruction bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from weave_agent_signals.runs.bundles import (
    BundleSnapshot,
    BundleValidationError,
    TargetSnapshot,
    changed_locators,
    validate_edited_bundle,
)
from weave_agent_signals.runs.promotion import (
    PromotionError,
    PromotionReceipt,
    PromotionValidationError,
    StaleBaseError,
    TargetPromoter,
)
from weave_agent_signals.runs.store import (
    Run,
    RunStatus,
    RunStore,
    RunStoreConflictError,
)


class ReviewServiceError(Exception):
    """Base typed error translated by the HTTP boundary."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        self.code = code
        self.context = context
        super().__init__(message)

    def to_detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), **self.context}


class ReviewNotFoundError(ReviewServiceError):
    def __init__(self, run_id: str) -> None:
        super().__init__("run_not_found", f"Run {run_id} was not found", run_id=run_id)


class ReviewRequestError(ReviewServiceError):
    """The requested review decision is invalid (HTTP 400)."""


class ReviewConflictError(ReviewServiceError):
    """The request lost a revision/lifecycle/drift race (HTTP 409)."""


class ReviewOperationError(ReviewServiceError):
    """A filesystem or persistence transaction failed (HTTP 500)."""


def _operation(
    code: str,
    source: str | Exception,
    manual_inspection: bool,
    changed: tuple[str, ...] | None = None,
) -> ReviewOperationError:
    return ReviewOperationError(
        code,
        str(source),
        changed_targets=list(changed or getattr(source, "changed_locators", ())),
        manual_inspection_required=manual_inspection,
    )


@dataclass
class ReviewService:
    """Own mutable review state while adapters own target transactions."""

    store: RunStore
    adapter_factory: Callable[[], TargetPromoter]
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    def select_candidate(
        self,
        run_id: str,
        *,
        candidate_id: str,
        expected_revision: int,
        discard_draft: bool = False,
    ) -> Run:
        run, review = self._pending(run_id, expected_revision)
        baseline, candidates = self._evidence(run)
        if candidate_id not in candidates:
            raise ReviewRequestError(
                "unknown_reflection_candidate",
                f"Reflection candidate {candidate_id!r} does not exist",
                candidate_id=candidate_id,
            )
        self._assert_fresh(run, baseline)

        selected = review.get("selected_candidate_id")
        draft = review.get("draft")
        if selected == candidate_id:
            if draft is None or not discard_draft:
                return run
            return self._store(run, {**review, "draft": None})
        if draft is not None and not discard_draft:
            raise ReviewConflictError(
                "dirty_draft",
                "Discard the edited draft before selecting another candidate",
                current_revision=run.reflection_review_revision,
                current_review=review,
            )
        return self._store(
            run,
            {**review, "selected_candidate_id": candidate_id, "draft": None},
        )

    def save_draft(
        self,
        run_id: str,
        *,
        contents: Mapping[str, str | None],
        expected_revision: int,
        expected_draft_revision: str | None,
    ) -> Run:
        run, review = self._pending(run_id, expected_revision)
        self._check_draft_revision(run, review, expected_draft_revision)
        baseline, candidate_id, candidate = self._decision_candidate(run, review)
        self._assert_fresh(run, baseline)
        draft = self._draft_from_contents(baseline, candidate, contents)
        stored_draft = (
            None
            if draft == candidate
            else {
                "candidate_id": candidate_id,
                "bundle": draft.to_dict(),
                "revision": draft.revision,
            }
        )
        if review.get("draft") == stored_draft:
            return run
        return self._store(run, {**review, "draft": stored_draft})

    def reset_draft(
        self,
        run_id: str,
        *,
        expected_revision: int,
        expected_draft_revision: str | None,
    ) -> Run:
        run, review = self._pending(run_id, expected_revision)
        self._check_draft_revision(run, review, expected_draft_revision)
        baseline, _candidates = self._evidence(run)
        self._assert_fresh(run, baseline)
        if review.get("draft") is None:
            return run
        return self._store(run, {**review, "draft": None})

    def dismiss(self, run_id: str, *, expected_revision: int) -> Run:
        run, review = self._pending(run_id, expected_revision)
        return self._store(
            run,
            {**review, "status": "dismissed", "dismissed_at": self._timestamp()},
        )

    def promote(
        self,
        run_id: str,
        *,
        promotion_id: str,
        expected_revision: int,
        expected_draft_revision: str | None,
        acknowledge_unevaluated: bool = False,
        acknowledge_unverified: bool = False,
    ) -> Run:
        run = self._load(run_id)
        if self._is_idempotent_promotion(
            run,
            promotion_id=promotion_id,
            expected_revision=expected_revision,
            expected_draft_revision=expected_draft_revision,
            acknowledge_unevaluated=acknowledge_unevaluated,
            acknowledge_unverified=acknowledge_unverified,
        ):
            return run

        run, review = self._pending_run(run, expected_revision)
        self._check_draft_revision(run, review, expected_draft_revision)
        baseline, candidate_id, candidate = self._decision_candidate(run, review)
        promoted = self._draft_bundle(review, candidate_id) or candidate
        edited = promoted != candidate
        self._check_acknowledgement(edited, acknowledge_unevaluated)
        sandbox_verified = candidate_id == run.reflecting_result.recommended_candidate_id
        self._check_verification_acknowledgement(
            sandbox_verified,
            acknowledge_unverified,
        )
        challenge = run.reflecting_result.challenge
        challenge_value = challenge.to_dict() if hasattr(challenge, "to_dict") else challenge or {}
        try:
            adapter = self.adapter_factory()
        except PromotionError as error:
            raise self._stale_error(
                run,
                baseline,
                None,
                error.changed_locators or baseline.locators,
            ) from error
        try:
            receipt = adapter.promote(
                promotion_id=promotion_id,
                run_id=run.run_id,
                candidate_id=candidate_id,
                past=baseline,
                evaluated_candidate=candidate,
                promoted=promoted,
                review_revision=run.reflection_review_revision,
                acknowledge_unevaluated=acknowledge_unevaluated,
                sandbox_verified=sandbox_verified,
                challenge_id=challenge_value.get("challenge_id"),
                challenge_status=challenge_value.get("status"),
                challenge_reason=challenge_value.get("reason"),
                acknowledge_unverified=acknowledge_unverified,
            )
        except StaleBaseError as error:
            raise self._stale_error(
                run,
                error.expected,
                error.current,
                error.changed_locators,
            ) from error
        except PromotionValidationError as error:
            raise ReviewRequestError(
                "invalid_promotion",
                str(error),
                changed_targets=list(error.changed_locators),
            ) from error
        except PromotionError as error:
            raise _operation(
                "promotion_not_applied",
                error,
                False,
            ) from error
        if receipt.status == "not_applied":
            raise _operation(
                "promotion_not_applied",
                receipt.outcomes[0].message or "Promotion was not applied",
                False,
                receipt.not_applied_locators,
            )
        return self._persist_receipt(run, review, receipt, adapter)

    def read(self, run_id: str) -> Run:
        """Load one run and derive live target drift for pending review."""

        return self.overlay(self._load(run_id))

    def overlay(self, run: Run) -> Run:
        """Return a nonpersistent live-drift view of a pending review."""

        review = run.reflection_review
        if not isinstance(review, Mapping) or review.get("status") != "pending":
            return run
        derived = {
            key: value
            for key, value in review.items()
            if key not in {"stale", "changed_targets", "current", "stale_reason"}
        }
        result = run.reflecting_result
        if result is None:
            derived = {
                **derived,
                "stale": True,
                "changed_targets": [],
                "current": None,
                "stale_reason": "Past baseline snapshot is unavailable",
            }
            return replace(run, reflection_review=derived)
        baseline = result.baseline
        current, changed, reason = self._drift(baseline)
        derived.update(stale=reason is not None, changed_targets=list(changed))
        if reason is not None:
            derived.update(
                current=current.to_dict() if current is not None else None,
                stale_reason=reason,
            )
        return replace(run, reflection_review=derived)

    def _load(self, run_id: str) -> Run:
        run = self.store.get(run_id)
        if run is None:
            raise ReviewNotFoundError(run_id)
        return run

    def _pending(self, run_id: str, expected_revision: int) -> tuple[Run, dict[str, Any]]:
        return self._pending_run(self._load(run_id), expected_revision)

    def _pending_run(
        self,
        run: Run,
        expected_revision: int,
    ) -> tuple[Run, dict[str, Any]]:
        context = {
            "current_revision": run.reflection_review_revision,
            "current_review": run.reflection_review,
        }
        if type(expected_revision) is not int or expected_revision < 0:
            raise ReviewRequestError(
                "invalid_review_revision",
                "expected_revision must be a non-negative integer",
            )
        if expected_revision != run.reflection_review_revision:
            raise ReviewConflictError(
                "reflection_review_conflict",
                "Reflection review revision is stale",
                expected_revision=expected_revision,
                **context,
            )
        if run.status is not RunStatus.COMPLETE:
            raise ReviewConflictError(
                "reflection_review_run_incomplete",
                "Reflection review actions require a completed run; "
                f"current status is {run.status.value}",
                current_status=run.status.value,
                **context,
            )
        review = run.reflection_review
        if not isinstance(review, dict):
            raise ReviewConflictError(
                "review_unavailable",
                "Run has no persisted reflection review",
                **context,
            )
        if review.get("status") != "pending":
            raise ReviewConflictError(
                "reflection_review_resolved",
                "Reflection review is no longer pending",
                **context,
            )
        return run, dict(review)

    def _evidence(self, run: Run) -> tuple[BundleSnapshot, dict[str, BundleSnapshot]]:
        result = run.reflecting_result
        if result is None:
            raise self._unavailable("Finalized reflection evidence is unavailable")
        baseline = result.baseline
        candidates = {}
        for attempt in result.attempts:
            if attempt.status != "succeeded":
                continue
            if attempt.candidate_id is None or attempt.bundle is None:
                raise self._unavailable("A reflection candidate is invalid")
            if attempt.bundle.scope != baseline.scope:
                raise self._unavailable("Reflection candidate scope does not match baseline")
            candidates[attempt.candidate_id] = attempt.bundle
        return baseline, candidates

    def _decision_candidate(
        self,
        run: Run,
        review: Mapping[str, Any],
    ) -> tuple[BundleSnapshot, str, BundleSnapshot]:
        baseline, candidates = self._evidence(run)
        candidate_id = review.get("selected_candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in candidates:
            raise self._unavailable("Selected candidate is not in reflection evidence")
        return baseline, candidate_id, candidates[candidate_id]

    def _stored_bundle(self, value: Any, label: str) -> BundleSnapshot:
        if not isinstance(value, Mapping):
            raise self._unavailable(f"{label} snapshot is unavailable")
        try:
            return BundleSnapshot.from_dict(value)
        except BundleValidationError as error:
            raise self._unavailable(
                f"{label} snapshot is invalid: {error}",
                changed_targets=list(error.changed_locators),
            ) from error

    @staticmethod
    def _unavailable(message: str, **context: Any) -> ReviewConflictError:
        return ReviewConflictError("review_unavailable", message, **context)

    def _assert_fresh(self, run: Run, baseline: BundleSnapshot) -> None:
        current, changed, reason = self._drift(baseline)
        if reason is None:
            return
        raise self._stale_error(run, baseline, current, changed)

    def _drift(
        self,
        baseline: BundleSnapshot,
    ) -> tuple[BundleSnapshot | None, tuple[str, ...], str | None]:
        try:
            adapter = self.adapter_factory()
            current = adapter.capture(include_missing=baseline.locators)
        except PromotionError as error:
            changed = error.changed_locators or baseline.locators
            return None, tuple(changed), str(error)
        if baseline.scope == current.scope and baseline == current:
            return current, (), None
        if baseline.scope != current.scope:
            changed = tuple(sorted(set(baseline.locators) | set(current.locators)))
            return current, changed, "target_changed"
        return current, changed_locators(baseline, current), "baseline_changed"

    @staticmethod
    def _stale_error(
        run: Run,
        baseline: BundleSnapshot,
        current: BundleSnapshot | None,
        changed: tuple[str, ...] | list[str],
    ) -> ReviewConflictError:
        return ReviewConflictError(
            "baseline_stale",
            "The complete managed scope changed after baseline evaluation",
            changed_targets=list(changed),
            expected=baseline.to_dict(),
            current=current.to_dict() if current is not None else None,
            review_revision=run.reflection_review_revision,
        )

    @staticmethod
    def _draft_from_contents(
        baseline: BundleSnapshot,
        candidate: BundleSnapshot,
        contents: Mapping[str, str | None],
    ) -> BundleSnapshot:
        if not isinstance(contents, Mapping) or any(
            not isinstance(locator, str) for locator in contents
        ):
            raise ReviewRequestError(
                "invalid_reflection_draft",
                "Draft contents must map target locators to content",
            )
        submitted = set(contents)
        expected = set(candidate.locators)
        if submitted != expected:
            raise ReviewRequestError(
                "invalid_reflection_draft",
                "Draft contents must include exactly the evaluated targets",
                changed_targets=sorted(submitted ^ expected),
            )
        targets: list[TargetSnapshot] = []
        try:
            for target in candidate.targets:
                content = contents[target.locator]
                targets.append(
                    TargetSnapshot(
                        kind=target.kind,
                        locator=target.locator,
                        exists=target.exists,
                        content=content,
                        display_name=target.display_name,
                        path=target.path,
                    )
                )
            draft = BundleSnapshot(tuple(targets), candidate.scope)
            validate_edited_bundle(baseline, candidate, draft)
        except BundleValidationError as error:
            raise ReviewRequestError(
                "invalid_reflection_draft",
                str(error),
                changed_targets=list(error.changed_locators),
            ) from error
        return draft

    def _draft_bundle(
        self,
        review: Mapping[str, Any],
        candidate_id: str,
    ) -> BundleSnapshot | None:
        value = review.get("draft")
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {
            "candidate_id",
            "bundle",
            "revision",
        }:
            raise self._unavailable("Stored reflection draft is invalid")
        bundle = self._stored_bundle(value.get("bundle"), "Edited draft")
        if value.get("candidate_id") != candidate_id or value.get("revision") != bundle.revision:
            raise self._unavailable("Stored reflection draft identity is invalid")
        return bundle

    @staticmethod
    def _draft_revision(review: Mapping[str, Any]) -> str | None:
        value = review.get("draft")
        if not isinstance(value, Mapping):
            return None
        revision = value.get("revision")
        return revision if isinstance(revision, str) else None

    def _check_draft_revision(
        self,
        run: Run,
        review: Mapping[str, Any],
        expected: str | None,
    ) -> None:
        current = self._draft_revision(review)
        if current != expected:
            raise ReviewConflictError(
                "reflection_draft_conflict",
                "Reflection draft revision is stale",
                expected_draft_revision=expected,
                current_draft_revision=current,
                current_revision=run.reflection_review_revision,
                current_review=dict(review),
            )

    @staticmethod
    def _check_acknowledgement(edited: bool, acknowledged: bool) -> None:
        if edited and not acknowledged:
            raise ReviewRequestError(
                "unevaluated_d_acknowledgement_required",
                "Acknowledge that edited draft C was not evaluated before promotion",
            )
        if not edited and acknowledged:
            raise ReviewRequestError(
                "unexpected_unevaluated_acknowledgement",
                "Evaluated proposal B does not require an unevaluated-C acknowledgement",
            )

    @staticmethod
    def _check_verification_acknowledgement(verified: bool, acknowledged: bool) -> None:
        if not verified and not acknowledged:
            raise ReviewRequestError(
                "unverified_b_acknowledgement_required",
                "Acknowledge that proposal B did not pass paired sandbox verification",
            )
        if verified and acknowledged:
            raise ReviewRequestError(
                "unexpected_unverified_acknowledgement",
                "Verified proposal B does not require an unverified-B acknowledgement",
            )

    def _store(self, run: Run, review: Mapping[str, Any]) -> Run:
        try:
            return self.store.update_reflection_review(
                run.run_id,
                review,
                expected_revision=run.reflection_review_revision,
            )
        except RunStoreConflictError as error:
            names = ("expected_revision", "current_revision", "current_review")
            context = {name: getattr(error, name) for name in names if hasattr(error, name)}
            raise ReviewConflictError(
                "reflection_review_conflict", str(error), **context
            ) from error
        except Exception as error:
            raise _operation(
                "review_persistence_failed",
                "Reflection review could not be persisted",
                False,
            ) from error

    def _persist_receipt(
        self,
        run: Run,
        review: Mapping[str, Any],
        receipt: PromotionReceipt,
        adapter: TargetPromoter,
    ) -> Run:
        del adapter
        receipt_dict = receipt.to_dict()
        try:
            return self._store(
                run,
                {
                    **review,
                    "status": "partial" if receipt.status == "partial" else "promoted",
                    "receipt": receipt_dict,
                },
            )
        except ReviewServiceError as persistence_error:
            raise _operation(
                "promotion_receipt_persist_failed",
                "Promotion changed one or more files but its receipt could not be persisted",
                True,
                receipt.applied_locators,
            ) from persistence_error

    def _is_idempotent_promotion(
        self,
        run: Run,
        *,
        promotion_id: str,
        expected_revision: int,
        expected_draft_revision: str | None,
        acknowledge_unevaluated: bool,
        acknowledge_unverified: bool,
    ) -> bool:
        review = run.reflection_review
        if not isinstance(review, Mapping) or review.get("status") not in {
            "promoted",
            "partial",
        }:
            return False
        raw_receipt = review.get("receipt")
        try:
            if not isinstance(raw_receipt, Mapping):
                raise PromotionValidationError("missing persisted promotion receipt")
            receipt = PromotionReceipt.from_dict(raw_receipt)
        except PromotionValidationError as error:
            raise _operation(
                "invalid_promotion_receipt",
                error,
                True,
            ) from error
        if receipt.promotion_id != promotion_id:
            raise ReviewConflictError(
                "reflection_review_resolved",
                "Reflection review is already promoted",
                current_revision=run.reflection_review_revision,
                current_review=dict(review),
            )
        committed_draft_revision = (
            None if receipt.promoted_was_evaluated else receipt.promoted.revision
        )
        if (
            receipt.review_revision != expected_revision
            or committed_draft_revision != expected_draft_revision
            or receipt.unevaluated_d_acknowledged != acknowledge_unevaluated
            or receipt.unverified_b_acknowledged != acknowledge_unverified
        ):
            raise ReviewConflictError(
                "promotion_idempotency_conflict",
                "Promotion retry does not match the committed review decision",
                expected_revision=expected_revision,
                committed_review_revision=receipt.review_revision,
                expected_draft_revision=expected_draft_revision,
                committed_draft_revision=committed_draft_revision,
            )
        return True

    def _timestamp(self) -> str:
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise _operation(
                "review_clock_invalid",
                "Review clock must return an aware datetime",
                False,
            )
        return value.astimezone(timezone.utc).isoformat()
