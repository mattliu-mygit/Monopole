"""Canonical immutable instruction bundles and exact B/C/D comparison rules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal


class BundleValidationError(ValueError):
    """A bundle, target, comparison, or edited-bundle identity is invalid."""

    def __init__(self, message: str, *, changed_locators: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.changed_locators = tuple(sorted(set(changed_locators)))


def revision_hash(value: Any) -> str:
    """Return the canonical content digest used by snapshot and intent identities."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _string(value: Any, name: str, locator: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise BundleValidationError(
            f"{name} must be a non-empty string",
            changed_locators=(locator,) if locator else (),
        )
    return value


@dataclass(frozen=True)
class ScopeDescriptor:
    """Persisted instructions for rediscovering one complete managed scope."""

    kind: str
    target_id: str
    patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _string(self.kind, "scope kind")
        _string(self.target_id, "scope target_id")
        patterns = tuple(self.patterns)
        if any(not isinstance(pattern, str) or not pattern for pattern in patterns):
            raise BundleValidationError("scope patterns must be non-empty strings")
        object.__setattr__(self, "patterns", patterns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_id": self.target_id,
            "patterns": list(self.patterns),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScopeDescriptor:
        try:
            if set(value) != {"kind", "target_id", "patterns"}:
                raise TypeError
            patterns = value["patterns"]
            if not isinstance(patterns, list):
                raise TypeError
            return cls(value["kind"], value["target_id"], tuple(patterns))
        except (KeyError, TypeError) as exc:
            raise BundleValidationError("invalid scope descriptor") from exc


@dataclass(frozen=True)
class TargetSnapshot:
    """Exact immutable state for one adapter-neutral target."""

    kind: str
    locator: str
    exists: bool
    content: str | None
    display_name: str | None = None
    path: str | None = None
    revision: str = field(init=False)

    def __post_init__(self) -> None:
        _string(self.kind, "target kind", self.locator)
        _string(self.locator, "target locator")
        if type(self.exists) is not bool:
            raise BundleValidationError(
                "target exists must be boolean",
                changed_locators=(self.locator,),
            )
        if self.exists and not isinstance(self.content, str):
            raise BundleValidationError(
                "existing target requires content",
                changed_locators=(self.locator,),
            )
        if not self.exists and self.content is not None:
            raise BundleValidationError(
                "missing target has content",
                changed_locators=(self.locator,),
            )
        display_name = self.display_name or PurePosixPath(self.locator).name
        _string(display_name, "display name", self.locator)
        if self.path is not None:
            _string(self.path, "target path", self.locator)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(
            self,
            "revision",
            revision_hash(
                {
                    "kind": self.kind,
                    "locator": self.locator,
                    "exists": self.exists,
                    "content": self.content,
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "locator": self.locator,
            "display_name": self.display_name,
            "path": self.path,
            "exists": self.exists,
            "content": self.content,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TargetSnapshot:
        expected = {
            "kind",
            "locator",
            "display_name",
            "path",
            "exists",
            "content",
            "revision",
        }
        try:
            if set(value) != expected:
                raise TypeError
            result = cls(
                kind=value["kind"],
                locator=value["locator"],
                display_name=value["display_name"],
                path=value["path"],
                exists=value["exists"],
                content=value["content"],
            )
        except (KeyError, TypeError) as exc:
            raise BundleValidationError("invalid target snapshot") from exc
        if value["revision"] != result.revision:
            raise BundleValidationError(
                "target revision mismatch",
                changed_locators=(result.locator,),
            )
        return result


@dataclass(frozen=True)
class BundleSnapshot:
    """Canonical immutable target bundle with a content-derived revision."""

    targets: tuple[TargetSnapshot, ...]
    scope: ScopeDescriptor | None = None
    revision: str = field(init=False)

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        if any(not isinstance(target, TargetSnapshot) for target in targets):
            raise BundleValidationError("bundle contains a non-snapshot target")
        seen: set[str] = set()
        duplicates: set[str] = set()
        for target in targets:
            if target.locator in seen:
                duplicates.add(target.locator)
            else:
                seen.add(target.locator)
        if duplicates:
            raise BundleValidationError(
                "bundle contains duplicate locators",
                changed_locators=duplicates,
            )
        if self.scope is not None and not isinstance(self.scope, ScopeDescriptor):
            raise BundleValidationError("invalid bundle scope")
        targets = tuple(sorted(targets, key=lambda target: target.locator))
        object.__setattr__(self, "targets", targets)
        object.__setattr__(
            self,
            "revision",
            revision_hash([(target.kind, target.locator, target.revision) for target in targets]),
        )

    @property
    def by_locator(self) -> Mapping[str, TargetSnapshot]:
        return MappingProxyType({target.locator: target for target in self.targets})

    @property
    def locators(self) -> tuple[str, ...]:
        return tuple(target.locator for target in self.targets)

    def target(self, locator: str) -> TargetSnapshot:
        """Return one exact target or raise ``KeyError`` when it is not captured."""

        return self.by_locator[locator]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.to_dict() if self.scope else None,
            "targets": [target.to_dict() for target in self.targets],
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BundleSnapshot:
        try:
            if set(value) != {"scope", "targets", "revision"}:
                raise TypeError
            raw_targets = value["targets"]
            if not isinstance(raw_targets, list):
                raise TypeError
            raw_scope = value["scope"]
            result = cls(
                tuple(TargetSnapshot.from_dict(target) for target in raw_targets),
                ScopeDescriptor.from_dict(raw_scope) if raw_scope is not None else None,
            )
        except (KeyError, TypeError) as exc:
            raise BundleValidationError("invalid bundle snapshot") from exc
        if value["revision"] != result.revision:
            raise BundleValidationError("bundle revision mismatch")
        return result


def bundle_from_content_map(
    contents: Mapping[str, str],
    *,
    scope: ScopeDescriptor | None = None,
    include_missing: Iterable[str] = (),
) -> BundleSnapshot:
    """Convert a complete content map plus explicit missing targets to a bundle."""

    if not isinstance(contents, Mapping):
        raise BundleValidationError("content map must be a mapping")
    kind = scope.kind if scope else "file"
    targets: list[TargetSnapshot] = []
    for locator, content in contents.items():
        locator = _string(locator, "content-map locator")
        if not isinstance(content, str):
            raise BundleValidationError(
                "content-map values must be strings",
                changed_locators=(locator,),
            )
        targets.append(
            TargetSnapshot(
                kind=kind,
                locator=locator,
                exists=True,
                content=content,
                path=locator if kind == "file" else None,
            )
        )
    _append_missing(targets, include_missing, kind=kind)
    return BundleSnapshot(tuple(targets), scope)


def _append_missing(
    targets: list[TargetSnapshot],
    include_missing: Iterable[str],
    *,
    kind: str,
) -> None:
    included: set[str] = set()
    present = {target.locator for target in targets}
    for locator in include_missing:
        locator = _string(locator, "missing target locator")
        if locator in included:
            raise BundleValidationError(
                "duplicate missing locator",
                changed_locators=(locator,),
            )
        included.add(locator)
        if locator not in present:
            targets.append(
                TargetSnapshot(
                    kind=kind,
                    locator=locator,
                    exists=False,
                    content=None,
                    path=locator if kind == "file" else None,
                )
            )


def _missing_like(target: TargetSnapshot) -> TargetSnapshot:
    return TargetSnapshot(
        kind=target.kind,
        locator=target.locator,
        exists=False,
        content=None,
        display_name=target.display_name,
        path=target.path,
    )


ActionKind = Literal["create", "update"]


@dataclass(frozen=True)
class PromotionAction:
    action: ActionKind
    locator: str
    before: TargetSnapshot
    after: TargetSnapshot

    def __post_init__(self) -> None:
        if self.action not in {"create", "update"}:
            raise BundleValidationError(
                "invalid promotion action",
                changed_locators=(self.locator,),
            )
        if self.before.locator != self.locator or self.after.locator != self.locator:
            raise BundleValidationError(
                "action locator mismatch",
                changed_locators=(self.locator,),
            )
        valid_state = {
            "create": (False, True),
            "update": (True, True),
        }[self.action]
        if (self.before.exists, self.after.exists) != valid_state:
            raise BundleValidationError(
                "action state mismatch",
                changed_locators=(self.locator,),
            )
        if self.action == "update" and self.before.revision == self.after.revision:
            raise BundleValidationError(
                "update action has no content change",
                changed_locators=(self.locator,),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "locator": self.locator,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromotionAction:
        try:
            if set(value) != {"action", "locator", "before", "after"}:
                raise TypeError
            return cls(
                value["action"],
                value["locator"],
                TargetSnapshot.from_dict(value["before"]),
                TargetSnapshot.from_dict(value["after"]),
            )
        except (KeyError, TypeError) as exc:
            raise BundleValidationError("invalid promotion action") from exc


@dataclass(frozen=True)
class PromotionPreview:
    past: BundleSnapshot
    proposed: BundleSnapshot
    actions: tuple[PromotionAction, ...]

    @property
    def changed_locators(self) -> tuple[str, ...]:
        return tuple(action.locator for action in self.actions)

    def _locators(self, action: ActionKind) -> tuple[str, ...]:
        return tuple(item.locator for item in self.actions if item.action == action)

    @property
    def created_locators(self) -> tuple[str, ...]:
        return self._locators("create")

    @property
    def updated_locators(self) -> tuple[str, ...]:
        return self._locators("update")


def compare_bundles(
    past: BundleSnapshot,
    proposed: BundleSnapshot,
) -> PromotionPreview:
    """Compare exact snapshots; synthesize missing targets only in action sides."""

    if not isinstance(past, BundleSnapshot) or not isinstance(proposed, BundleSnapshot):
        raise BundleValidationError("expected BundleSnapshot")
    if past.scope != proposed.scope:
        raise BundleValidationError(
            "bundle scopes do not match",
            changed_locators=set(past.locators) | set(proposed.locators),
        )
    actions: list[PromotionAction] = []
    for locator in sorted(set(past.locators) | set(proposed.locators)):
        before = past.by_locator.get(locator)
        after = proposed.by_locator.get(locator)
        if before is None:
            before = _missing_like(after)  # type: ignore[arg-type]
        if after is None:
            after = _missing_like(before)
        if before.kind != after.kind:
            raise BundleValidationError(
                "one locator has multiple target kinds",
                changed_locators=(locator,),
            )
        action: ActionKind | None = None
        if not before.exists and after.exists:
            action = "create"
        elif before.exists and not after.exists:
            raise BundleValidationError(
                "delete actions are not supported",
                changed_locators=(locator,),
            )
        elif before.exists and after.exists and before.revision != after.revision:
            action = "update"
        if action is not None:
            actions.append(PromotionAction(action, locator, before, after))
    return PromotionPreview(past, proposed, tuple(actions))


def changed_locators(
    expected: BundleSnapshot,
    current: BundleSnapshot,
) -> tuple[str, ...]:
    """Return state differences while treating absent and explicit-missing as equal."""

    changed: list[str] = []
    for locator in sorted(set(expected.locators) | set(current.locators)):
        before = expected.by_locator.get(locator)
        after = current.by_locator.get(locator)
        before_state = (before.exists, before.content) if before is not None else (False, None)
        after_state = (after.exists, after.content) if after is not None else (False, None)
        if before_state != after_state:
            changed.append(locator)
    return tuple(changed)


def validate_edited_bundle(
    baseline: BundleSnapshot,
    evaluated: BundleSnapshot,
    edited: BundleSnapshot,
) -> None:
    """Require unevaluated D to preserve C target identity and B-to-C actions."""

    for bundle in (baseline, evaluated, edited):
        if not isinstance(bundle, BundleSnapshot):
            raise BundleValidationError("expected BundleSnapshot")
    if not (baseline.scope == evaluated.scope == edited.scope):
        raise BundleValidationError("edited bundle scope must match evaluated evidence")
    if evaluated.locators != edited.locators:
        raise BundleValidationError(
            "edited bundle must preserve evaluated target membership",
            changed_locators=set(evaluated.locators) ^ set(edited.locators),
        )

    identity_changes: list[str] = []
    for locator in evaluated.locators:
        candidate = evaluated.by_locator[locator]
        draft = edited.by_locator[locator]
        if (
            candidate.kind,
            candidate.locator,
            candidate.display_name,
            candidate.path,
            candidate.exists,
        ) != (
            draft.kind,
            draft.locator,
            draft.display_name,
            draft.path,
            draft.exists,
        ):
            identity_changes.append(locator)
    if identity_changes:
        raise BundleValidationError(
            "edited bundle may change only evaluated target content",
            changed_locators=identity_changes,
        )

    evaluated_actions = {
        (action.action, action.locator) for action in compare_bundles(baseline, evaluated).actions
    }
    edited_actions = {
        (action.action, action.locator) for action in compare_bundles(baseline, edited).actions
    }
    if evaluated_actions != edited_actions:
        raise BundleValidationError(
            "edited bundle must preserve evaluated create/update actions",
            changed_locators={
                locator for _, locator in evaluated_actions.symmetric_difference(edited_actions)
            },
        )
