"""Authoritative reflection proposal contract and candidate materialization."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from weave_agent_signals.judges.inference import JsonSchemaSpec
from weave_agent_signals.runs.bundles import BundleSnapshot, TargetSnapshot

REFLECTION_PROPOSAL_SCHEMA_VERSION = 3

CandidateAction = Literal["create", "update"]


@dataclass(frozen=True)
class CandidateChange:
    action: CandidateAction
    locator: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or self.action not in {"create", "update"}:
            raise ValueError("proposal action must be create or update")
        if not isinstance(self.locator, str) or not self.locator:
            raise ValueError("proposal locator must be a non-empty string")
        if not isinstance(self.content, str):
            raise ValueError(f"{self.action} requires string content")


@dataclass(frozen=True)
class CandidateProposal:
    schema_version: Literal[3]
    changes: tuple[CandidateChange, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 3:
            raise ValueError("reflection proposal schema_version must be integer 3")
        if not isinstance(self.changes, tuple) or any(
            not isinstance(change, CandidateChange) for change in self.changes
        ):
            raise ValueError("proposal changes must be CandidateChange values")
        locators = [change.locator for change in self.changes]
        if len(locators) != len(set(locators)):
            raise ValueError("proposal locators must be unique")
        object.__setattr__(
            self, "changes", tuple(sorted(self.changes, key=lambda item: item.locator))
        )


_CHANGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["create", "update"]},
        "locator": {"type": "string", "minLength": 1},
        "content": {"type": "string"},
    },
    "required": ["action", "locator", "content"],
}

REFLECTION_PROPOSAL_SCHEMA = JsonSchemaSpec(
    name="reflection_proposal",
    schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 3, "type": "integer"},
            "changes": {"type": "array", "items": _CHANGE_SCHEMA, "minItems": 1},
        },
        "required": ["schema_version", "changes"],
    },
)


def parse_candidate_proposal(value: str | Mapping[str, Any]) -> CandidateProposal:
    """Parse the closed versioned writer payload into its sole value object."""

    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "changes"}:
        raise ValueError("reflection proposal must contain exactly schema_version and changes")
    if type(value["schema_version"]) is not int or value["schema_version"] != 3:
        raise ValueError("reflection proposal schema_version must be integer 3")
    raw_changes = value["changes"]
    if not isinstance(raw_changes, list):
        raise ValueError("reflection proposal changes must be a list")

    changes: list[CandidateChange] = []
    for raw_change in raw_changes:
        if not isinstance(raw_change, Mapping) or set(raw_change) != {
            "action",
            "locator",
            "content",
        }:
            raise ValueError("proposal changes must contain exactly action, locator, and content")
        changes.append(
            CandidateChange(
                action=raw_change["action"],
                locator=raw_change["locator"],
                content=raw_change["content"],
            )
        )
    return CandidateProposal(3, tuple(sorted(changes, key=lambda change: change.locator)))


def serialize_candidate_proposal(proposal: CandidateProposal) -> str:
    """Return the deterministic JSON transport form of one proposal."""

    return json.dumps(
        {
            "schema_version": proposal.schema_version,
            "changes": [
                {
                    "action": change.action,
                    "locator": change.locator,
                    "content": change.content,
                }
                for change in sorted(proposal.changes, key=lambda change: change.locator)
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def materialize_candidate_proposal(
    proposal: CandidateProposal,
    *,
    baseline: BundleSnapshot,
    resolve_locator: Callable[..., object],
) -> BundleSnapshot:
    """Apply one complete proposal to B and build its immutable C snapshot."""

    if not isinstance(proposal, CandidateProposal):
        raise ValueError("expected CandidateProposal")
    if not isinstance(baseline, BundleSnapshot):
        raise ValueError("baseline must be a BundleSnapshot")
    if not proposal.changes:
        raise ValueError("writer proposal must contain at least one effective change")

    by_locator = dict(baseline.by_locator)
    for change in proposal.changes:
        target = by_locator.get(change.locator)
        exists = target is not None and target.exists
        if change.action == "create" and exists:
            raise ValueError(f"create locator already exists in baseline B: {change.locator}")
        if change.action == "update" and not exists:
            raise ValueError(f"update locator is absent from baseline B: {change.locator}")
        if change.action == "update" and target is not None and target.content == change.content:
            raise ValueError(f"update must change baseline B content: {change.locator}")
        resolve_locator(
            change.locator,
            require_absent_for_create=change.action == "create",
        )

    candidate_targets = dict(by_locator)
    for change in proposal.changes:
        target = by_locator.get(change.locator)
        if target is None:
            candidate_targets[change.locator] = TargetSnapshot(
                kind="file",
                locator=change.locator,
                exists=True,
                content=change.content,
            )
        else:
            candidate_targets[change.locator] = TargetSnapshot(
                kind=target.kind,
                locator=target.locator,
                exists=True,
                content=change.content,
                display_name=target.display_name,
                path=target.path,
            )

    candidate = BundleSnapshot(tuple(candidate_targets.values()), baseline.scope)
    if candidate.revision == baseline.revision:
        raise ValueError("writer proposal must produce a changed candidate bundle")
    return candidate
