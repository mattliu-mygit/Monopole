"""Authoritative reflection proposal contract and candidate materialization."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from weave_agent_signals.judges.inference import JsonSchemaSpec
from weave_agent_signals.runs.bundles import BundleSnapshot

REFLECTION_PROPOSAL_SCHEMA_VERSION = 2

CandidateAction = Literal["create", "update", "delete"]


@dataclass(frozen=True)
class CandidateChange:
    action: CandidateAction
    path: str
    content: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or self.action not in {"create", "update", "delete"}:
            raise ValueError("proposal action must be create, update, or delete")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("proposal path must be a non-empty string")
        canonical = PurePosixPath(self.path).as_posix()
        if canonical != self.path:
            raise ValueError("proposal paths must use canonical relative POSIX spelling")
        if self.action in {"create", "update"} and not isinstance(self.content, str):
            raise ValueError(f"{self.action} requires string content")
        if self.action == "delete" and self.content is not None:
            raise ValueError("delete requires null content")


@dataclass(frozen=True)
class CandidateProposal:
    schema_version: Literal[2]
    changes: tuple[CandidateChange, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("reflection proposal schema_version must be integer 2")
        if not isinstance(self.changes, tuple) or any(
            not isinstance(change, CandidateChange) for change in self.changes
        ):
            raise ValueError("proposal changes must be CandidateChange values")
        paths = [change.path for change in self.changes]
        if len(paths) != len(set(paths)):
            raise ValueError("proposal paths must be unique")
        object.__setattr__(self, "changes", tuple(sorted(self.changes, key=lambda item: item.path)))


_CHANGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["create", "update", "delete"]},
        "path": {"type": "string", "minLength": 1},
        "content": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["action", "path", "content"],
}

REFLECTION_PROPOSAL_SCHEMA = JsonSchemaSpec(
    name="reflection_proposal",
    schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": 2, "type": "integer"},
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
    if type(value["schema_version"]) is not int or value["schema_version"] != 2:
        raise ValueError("reflection proposal schema_version must be integer 2")
    raw_changes = value["changes"]
    if not isinstance(raw_changes, list):
        raise ValueError("reflection proposal changes must be a list")

    changes: list[CandidateChange] = []
    for raw_change in raw_changes:
        if not isinstance(raw_change, Mapping) or set(raw_change) != {
            "action",
            "path",
            "content",
        }:
            raise ValueError("proposal changes must contain exactly action, path, and content")
        changes.append(
            CandidateChange(
                action=raw_change["action"],
                path=raw_change["path"],
                content=raw_change["content"],
            )
        )
    return CandidateProposal(2, tuple(sorted(changes, key=lambda change: change.path)))


def serialize_candidate_proposal(proposal: CandidateProposal) -> str:
    """Return the deterministic JSON transport form of one proposal."""

    return json.dumps(
        {
            "schema_version": proposal.schema_version,
            "changes": [
                {
                    "action": change.action,
                    "path": change.path,
                    "content": change.content,
                }
                for change in sorted(proposal.changes, key=lambda change: change.path)
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
    build_candidate: Callable[[Mapping[str, str]], BundleSnapshot],
) -> BundleSnapshot:
    """Apply one complete proposal to B and build its immutable C snapshot."""

    if not isinstance(proposal, CandidateProposal):
        raise ValueError("expected CandidateProposal")
    if not isinstance(baseline, BundleSnapshot):
        raise ValueError("baseline must be a BundleSnapshot")
    if not proposal.changes:
        raise ValueError("writer proposal must contain at least one effective change")

    baseline_contents = {
        target.locator: target.content
        for target in baseline.targets
        if target.exists and target.content is not None
    }
    for change in proposal.changes:
        exists = change.path in baseline_contents
        if change.action == "create" and exists:
            raise ValueError(f"create path already exists in baseline B: {change.path}")
        if change.action in {"update", "delete"} and not exists:
            raise ValueError(f"{change.action} path is absent from baseline B: {change.path}")
        if change.action == "update" and baseline_contents[change.path] == change.content:
            raise ValueError(f"update must change baseline B content: {change.path}")

    candidate_contents = dict(baseline_contents)
    for change in proposal.changes:
        if change.action == "delete":
            del candidate_contents[change.path]
        else:
            candidate_contents[change.path] = change.content  # type: ignore[assignment]

    candidate = build_candidate(candidate_contents)
    if not isinstance(candidate, BundleSnapshot):
        raise ValueError("candidate builder must return BundleSnapshot")
    if candidate.revision == baseline.revision:
        raise ValueError("writer proposal must produce a changed candidate bundle")
    return candidate
