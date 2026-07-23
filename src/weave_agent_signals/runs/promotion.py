"""Registry-bounded promotion with atomic, independent file outcomes."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from weave_agent_signals.runs import bundles
from weave_agent_signals.runs.targets import TargetRegistry

OutcomeStatus = Literal["applied", "not_applied"]
OutcomeReason = Literal["source_drift", "write_failed"]
ReceiptStatus = Literal["applied", "partial", "not_applied"]


class PromotionError(Exception):
    """Base promotion error carrying affected stable target locators."""

    def __init__(self, message: str, *, changed_locators: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.changed_locators = tuple(sorted(set(changed_locators)))


class PromotionValidationError(PromotionError):
    """The requested promotion does not satisfy the pinned contract."""


class PromotionApplyError(PromotionError):
    """Promotion could not stage all files before publication began."""


class StaleBaseError(PromotionError):
    """The current registry scope no longer equals evaluated baseline A."""

    def __init__(
        self,
        expected: bundles.BundleSnapshot,
        current: bundles.BundleSnapshot | None,
        changed_locators: Iterable[str] = (),
    ) -> None:
        changed = (
            bundles.changed_locators(expected, current)
            if current is not None
            else tuple(changed_locators)
        )
        super().__init__(
            "the complete managed scope changed after baseline evaluation",
            changed_locators=changed,
        )
        self.expected = expected
        self.current = current


@dataclass(frozen=True)
class PromotionTargetOutcome:
    locator: str
    action: bundles.ActionKind
    status: OutcomeStatus
    reason: OutcomeReason | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.locator, str) or not self.locator:
            raise PromotionValidationError("outcome locator must be nonblank")
        if self.action not in {"create", "update"}:
            raise PromotionValidationError("outcome action must be create or update")
        if self.status not in {"applied", "not_applied"}:
            raise PromotionValidationError("outcome status is invalid")
        if self.status == "applied" and (self.reason is not None or self.message is not None):
            raise PromotionValidationError("applied outcomes cannot contain a failure")
        if self.status == "not_applied" and self.reason not in {
            "source_drift",
            "write_failed",
        }:
            raise PromotionValidationError("not-applied outcomes require a reason")
        if self.message is not None and (
            not isinstance(self.message, str) or not self.message or len(self.message) > 500
        ):
            raise PromotionValidationError("outcome message must be 1 to 500 characters")

    def to_dict(self) -> dict[str, Any]:
        return {
            "locator": self.locator,
            "action": self.action,
            "status": self.status,
            "reason": self.reason,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromotionTargetOutcome:
        if set(value) != {"locator", "action", "status", "reason", "message"}:
            raise PromotionValidationError("invalid promotion target outcome")
        return cls(
            locator=value["locator"],
            action=value["action"],
            status=value["status"],
            reason=value["reason"],
            message=value["message"],
        )


@dataclass(frozen=True)
class PromotionReceipt:
    """Auditable A/B/C decision plus the outcome of each complete file action."""

    promotion_id: str
    run_id: str
    candidate_id: str
    past: bundles.BundleSnapshot
    evaluated_candidate: bundles.BundleSnapshot
    promoted: bundles.BundleSnapshot
    review_revision: int
    outcomes: tuple[PromotionTargetOutcome, ...]
    decided_at: str
    promoted_was_evaluated: bool
    unevaluated_d_acknowledged: bool
    sandbox_verified: bool
    challenge_id: str | None
    challenge_status: str | None
    challenge_reason: str | None
    unverified_b_acknowledged: bool

    def __post_init__(self) -> None:
        for value in (self.promotion_id, self.run_id, self.candidate_id, self.decided_at):
            if not isinstance(value, str) or not value:
                raise PromotionValidationError("receipt identifiers must be nonblank")
        if type(self.review_revision) is not int or self.review_revision < 0:
            raise PromotionValidationError("receipt review revision is invalid")
        try:
            actions = bundles.compare_bundles(self.past, self.promoted).actions
            if self.evaluated_candidate != self.promoted:
                bundles.validate_edited_bundle(
                    self.past,
                    self.evaluated_candidate,
                    self.promoted,
                )
        except bundles.BundleValidationError as exc:
            raise PromotionValidationError(str(exc), changed_locators=exc.changed_locators) from exc
        if not actions:
            raise PromotionValidationError("receipt must contain at least one action")
        outcomes = tuple(self.outcomes)
        if tuple((item.locator, item.action) for item in outcomes) != tuple(
            (item.locator, item.action) for item in actions
        ):
            raise PromotionValidationError("receipt outcomes do not match requested actions")
        failure_seen = False
        for outcome in outcomes:
            if outcome.status == "not_applied":
                failure_seen = True
            elif failure_seen:
                raise PromotionValidationError(
                    "receipt cannot apply a file after a not-applied outcome"
                )
        expected_evaluated = self.evaluated_candidate == self.promoted
        if self.promoted_was_evaluated is not expected_evaluated:
            raise PromotionValidationError("receipt evaluation marker is inconsistent")
        if self.unevaluated_d_acknowledged is not (not expected_evaluated):
            raise PromotionValidationError("receipt acknowledgement is inconsistent")
        if type(self.sandbox_verified) is not bool:
            raise PromotionValidationError("receipt sandbox verification marker is invalid")
        if self.challenge_status not in {None, "complete", "incomplete", "invalid_task"}:
            raise PromotionValidationError("receipt challenge status is invalid")
        for value in (self.challenge_id, self.challenge_reason):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise PromotionValidationError("receipt challenge provenance is invalid")
        if self.unverified_b_acknowledged is not (not self.sandbox_verified):
            raise PromotionValidationError("receipt verification acknowledgement is inconsistent")
        object.__setattr__(self, "outcomes", outcomes)

    @property
    def status(self) -> ReceiptStatus:
        applied = len(self.applied_locators)
        if applied == len(self.outcomes):
            return "applied"
        return "partial" if applied else "not_applied"

    @property
    def applied_locators(self) -> tuple[str, ...]:
        return tuple(item.locator for item in self.outcomes if item.status == "applied")

    @property
    def not_applied_locators(self) -> tuple[str, ...]:
        return tuple(item.locator for item in self.outcomes if item.status == "not_applied")

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotion_id": self.promotion_id,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "past": self.past.to_dict(),
            "evaluated_candidate": self.evaluated_candidate.to_dict(),
            "promoted": self.promoted.to_dict(),
            "review_revision": self.review_revision,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "decided_at": self.decided_at,
            "promoted_was_evaluated": self.promoted_was_evaluated,
            "unevaluated_d_acknowledged": self.unevaluated_d_acknowledged,
            "sandbox_verified": self.sandbox_verified,
            "challenge_id": self.challenge_id,
            "challenge_status": self.challenge_status,
            "challenge_reason": self.challenge_reason,
            "unverified_b_acknowledged": self.unverified_b_acknowledged,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromotionReceipt:
        expected = {
            "promotion_id",
            "run_id",
            "candidate_id",
            "past",
            "evaluated_candidate",
            "promoted",
            "review_revision",
            "outcomes",
            "decided_at",
            "promoted_was_evaluated",
            "unevaluated_d_acknowledged",
            "sandbox_verified",
            "challenge_id",
            "challenge_status",
            "challenge_reason",
            "unverified_b_acknowledged",
        }
        try:
            if set(value) != expected or not isinstance(value["outcomes"], list):
                raise TypeError
            return cls(
                promotion_id=value["promotion_id"],
                run_id=value["run_id"],
                candidate_id=value["candidate_id"],
                past=bundles.BundleSnapshot.from_dict(value["past"]),
                evaluated_candidate=bundles.BundleSnapshot.from_dict(value["evaluated_candidate"]),
                promoted=bundles.BundleSnapshot.from_dict(value["promoted"]),
                review_revision=value["review_revision"],
                outcomes=tuple(
                    PromotionTargetOutcome.from_dict(item) for item in value["outcomes"]
                ),
                decided_at=value["decided_at"],
                promoted_was_evaluated=value["promoted_was_evaluated"],
                unevaluated_d_acknowledged=value["unevaluated_d_acknowledged"],
                sandbox_verified=value["sandbox_verified"],
                challenge_id=value["challenge_id"],
                challenge_status=value["challenge_status"],
                challenge_reason=value["challenge_reason"],
                unverified_b_acknowledged=value["unverified_b_acknowledged"],
            )
        except (KeyError, TypeError, bundles.BundleValidationError) as exc:
            raise PromotionValidationError("invalid promotion receipt") from exc


@dataclass
class _StagedAction:
    action: bundles.PromotionAction
    destination: Path
    temporary: Path


class TargetPromoter:
    """Apply one reviewed registry bundle as independent complete-file writes."""

    def __init__(self, registry: TargetRegistry) -> None:
        if not isinstance(registry, TargetRegistry):
            raise PromotionValidationError("TargetPromoter requires a target registry")
        self.registry = registry

    def contract_manifest(self) -> dict[str, object]:
        return self.registry.contract_manifest()

    def resolve_locator(self, locator: str, *, require_absent_for_create: bool = False) -> Path:
        return self.registry.resolve_locator(
            locator, require_absent_for_create=require_absent_for_create
        )

    def capture(self, *, include_missing: Iterable[str] = ()) -> bundles.BundleSnapshot:
        captured = self.registry.capture()
        missing = set(include_missing) - set(captured.locators)
        if not missing:
            return captured
        targets = list(captured.targets)
        for locator in sorted(missing):
            try:
                self.registry.resolve_locator(locator)
            except (KeyError, ValueError) as exc:
                raise PromotionValidationError(
                    f"target is no longer admitted by the registry: {locator}",
                    changed_locators=(locator,),
                ) from exc
            targets.append(
                bundles.TargetSnapshot(kind="file", locator=locator, exists=False, content=None)
            )
        return bundles.BundleSnapshot(tuple(targets), captured.scope)

    def promote(
        self,
        *,
        promotion_id: str,
        run_id: str,
        candidate_id: str,
        past: bundles.BundleSnapshot,
        evaluated_candidate: bundles.BundleSnapshot,
        promoted: bundles.BundleSnapshot,
        review_revision: int,
        acknowledge_unevaluated: bool,
        sandbox_verified: bool = True,
        challenge_id: str | None = None,
        challenge_status: str | None = None,
        challenge_reason: str | None = None,
        acknowledge_unverified: bool = False,
    ) -> PromotionReceipt:
        with self.registry.promotion_guard():
            return self._promote_locked(
                promotion_id=promotion_id,
                run_id=run_id,
                candidate_id=candidate_id,
                past=past,
                evaluated_candidate=evaluated_candidate,
                promoted=promoted,
                review_revision=review_revision,
                acknowledge_unevaluated=acknowledge_unevaluated,
                sandbox_verified=sandbox_verified,
                challenge_id=challenge_id,
                challenge_status=challenge_status,
                challenge_reason=challenge_reason,
                acknowledge_unverified=acknowledge_unverified,
            )

    def _promote_locked(
        self,
        *,
        promotion_id: str,
        run_id: str,
        candidate_id: str,
        past: bundles.BundleSnapshot,
        evaluated_candidate: bundles.BundleSnapshot,
        promoted: bundles.BundleSnapshot,
        review_revision: int,
        acknowledge_unevaluated: bool,
        sandbox_verified: bool,
        challenge_id: str | None,
        challenge_status: str | None,
        challenge_reason: str | None,
        acknowledge_unverified: bool,
    ) -> PromotionReceipt:
        try:
            actions = bundles.compare_bundles(past, promoted).actions
            if evaluated_candidate != promoted:
                bundles.validate_edited_bundle(past, evaluated_candidate, promoted)
        except bundles.BundleValidationError as exc:
            raise PromotionValidationError(str(exc), changed_locators=exc.changed_locators) from exc
        edited = evaluated_candidate != promoted
        if acknowledge_unevaluated is not edited:
            raise PromotionValidationError("unevaluated draft acknowledgement is inconsistent")
        if acknowledge_unverified is not (not sandbox_verified):
            raise PromotionValidationError("unverified proposal acknowledgement is inconsistent")
        if not actions:
            raise PromotionValidationError("promotion must contain at least one action")

        try:
            current = self.capture(include_missing=past.locators)
        except (ValueError, PromotionError) as exc:
            changed = getattr(exc, "changed_locators", past.locators)
            raise StaleBaseError(past, None, changed) from exc
        if current != past:
            raise StaleBaseError(past, current)

        staged: list[_StagedAction] = []
        registered_creates: tuple[str, ...] = ()
        try:
            for action in actions:
                destination = self._resolve(action)
                try:
                    temporary = self._stage(destination, action.after.content or "", action.action)
                except OSError as exc:
                    raise PromotionApplyError(
                        f"could not stage {action.locator}: {exc}",
                        changed_locators=(action.locator,),
                    ) from exc
                staged.append(_StagedAction(action, destination, temporary))

            try:
                registered_creates = self.registry.register_creates(
                    tuple(item.action.locator for item in staged if item.action.action == "create")
                )
            except (KeyError, OSError, ValueError) as exc:
                raise PromotionApplyError(
                    f"could not register create targets: {exc}",
                    changed_locators=tuple(
                        item.action.locator for item in staged if item.action.action == "create"
                    ),
                ) from exc

            outcomes: list[PromotionTargetOutcome] = []
            failed = False
            for item in staged:
                if failed:
                    outcomes.append(
                        PromotionTargetOutcome(
                            item.action.locator,
                            item.action.action,
                            "not_applied",
                            "write_failed",
                            "not attempted after an earlier target failed",
                        )
                    )
                    continue
                if not self._matches_past(item.action, past):
                    failed = True
                    outcomes.append(
                        PromotionTargetOutcome(
                            item.action.locator,
                            item.action.action,
                            "not_applied",
                            "source_drift",
                            "source changed after promotion preflight",
                        )
                    )
                    continue
                try:
                    self._publish(item)
                except OSError as exc:
                    failed = True
                    outcomes.append(
                        PromotionTargetOutcome(
                            item.action.locator,
                            item.action.action,
                            "not_applied",
                            "write_failed",
                            str(exc)[:500] or type(exc).__name__,
                        )
                    )
                else:
                    outcomes.append(
                        PromotionTargetOutcome(item.action.locator, item.action.action, "applied")
                    )
            if registered_creates and not any(outcome.status == "applied" for outcome in outcomes):
                try:
                    self.registry.unregister_absent_creates(registered_creates)
                except (KeyError, OSError, ValueError) as exc:
                    raise PromotionApplyError(
                        f"could not restore create registrations: {exc}",
                        changed_locators=registered_creates,
                    ) from exc
        finally:
            for item in staged:
                item.temporary.unlink(missing_ok=True)

        return PromotionReceipt(
            promotion_id=promotion_id,
            run_id=run_id,
            candidate_id=candidate_id,
            past=past,
            evaluated_candidate=evaluated_candidate,
            promoted=promoted,
            review_revision=review_revision,
            outcomes=tuple(outcomes),
            decided_at=datetime.now(timezone.utc).isoformat(),
            promoted_was_evaluated=not edited,
            unevaluated_d_acknowledged=edited,
            sandbox_verified=sandbox_verified,
            challenge_id=challenge_id,
            challenge_status=challenge_status,
            challenge_reason=challenge_reason,
            unverified_b_acknowledged=acknowledge_unverified,
        )

    def _resolve(self, action: bundles.PromotionAction) -> Path:
        try:
            return self.registry.resolve_locator(
                action.locator,
                require_absent_for_create=action.action == "create",
            )
        except (KeyError, ValueError) as exc:
            raise PromotionValidationError(
                f"target is not admitted for {action.action}: {action.locator}",
                changed_locators=(action.locator,),
            ) from exc

    def _stage(self, destination: Path, content: str, action: bundles.ActionKind) -> Path:
        staging_directory = destination.parent
        if action == "create":
            while not staging_directory.exists():
                parent = staging_directory.parent
                if parent == staging_directory:
                    raise OSError("create target has no existing ancestor")
                staging_directory = parent
        descriptor, name = tempfile.mkstemp(prefix=".monopole-", dir=staging_directory)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if action == "update":
                os.chmod(temporary, destination.stat().st_mode & 0o7777)
            return temporary
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _matches_past(self, action: bundles.PromotionAction, past: bundles.BundleSnapshot) -> bool:
        expected = past.by_locator.get(action.locator)
        try:
            current = self.registry.capture_locator(action.locator)
        except (KeyError, OSError, ValueError):
            return False
        if action.action == "create" and expected is None:
            return not current.exists
        return current == expected

    def _publish(self, item: _StagedAction) -> None:
        if item.action.action == "create":
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(item.temporary, item.destination)
        else:
            os.replace(item.temporary, item.destination)
