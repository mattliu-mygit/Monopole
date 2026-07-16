from __future__ import annotations

import json
from pathlib import Path

import pytest

from weave_agent_signals.runs.bundles import compare_bundles
from weave_agent_signals.runs.proposals import (
    REFLECTION_PROPOSAL_SCHEMA,
    REFLECTION_PROPOSAL_SCHEMA_VERSION,
    CandidateProposal,
    materialize_candidate_proposal,
    parse_candidate_proposal,
    serialize_candidate_proposal,
)
from weave_agent_signals.runs.targets import load_target_registry


def _registry(tmp_path: Path):
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "targets": [
                    {"kind": "file", "id": "agents", "path": "AGENTS.md"},
                    {"kind": "file", "id": "soul", "path": "SOUL.md"},
                    {"kind": "skill_collection", "id": "skills", "root": "skills"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return load_target_registry(path)


def test_proposal_schema_is_closed_create_update_only_and_canonical() -> None:
    assert REFLECTION_PROPOSAL_SCHEMA_VERSION == 3
    schema = REFLECTION_PROPOSAL_SCHEMA.schema
    change_schema = schema["properties"]["changes"]["items"]
    assert change_schema["properties"]["action"]["enum"] == ["create", "update"]

    proposal = parse_candidate_proposal(
        {
            "schema_version": 3,
            "changes": [
                {"action": "create", "locator": "skills:skills/new/SKILL.md", "content": "new"},
                {"action": "update", "locator": "file:agents", "content": "updated"},
            ],
        }
    )

    assert isinstance(proposal, CandidateProposal)
    assert [change.locator for change in proposal.changes] == [
        "file:agents",
        "skills:skills/new/SKILL.md",
    ]
    assert serialize_candidate_proposal(proposal) == (
        '{"changes":['
        '{"action":"update","content":"updated","locator":"file:agents"},'
        '{"action":"create","content":"new","locator":"skills:skills/new/SKILL.md"}'
        '],"schema_version":3}'
    )


def test_proposal_materializes_update_and_admitted_skill_create(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("old", encoding="utf-8")
    registry = _registry(tmp_path)
    baseline = registry.capture()
    proposal = parse_candidate_proposal(
        {
            "schema_version": 3,
            "changes": [
                {"action": "update", "locator": "file:agents", "content": "updated"},
                {"action": "create", "locator": "skills:skills/new/SKILL.md", "content": "new"},
            ],
        }
    )

    candidate = materialize_candidate_proposal(
        proposal,
        baseline=baseline,
        resolve_locator=registry.resolve_locator,
    )

    assert [
        (item.action, item.locator) for item in compare_bundles(baseline, candidate).actions
    ] == [
        ("update", "file:agents"),
        ("create", "skills:skills/new/SKILL.md"),
    ]


@pytest.mark.parametrize("action", ["delete", "archive", "rename"])
def test_proposal_rejects_retired_actions(action: str) -> None:
    with pytest.raises(ValueError):
        parse_candidate_proposal(
            {
                "schema_version": 3,
                "changes": [{"action": action, "locator": "file:agents", "content": None}],
            }
        )


def test_proposal_rejects_duplicate_unknown_and_action_inconsistent_locators(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("old", encoding="utf-8")
    registry = _registry(tmp_path)
    baseline = registry.capture()
    with pytest.raises(ValueError, match="unique"):
        parse_candidate_proposal(
            {
                "schema_version": 3,
                "changes": [
                    {"action": "update", "locator": "file:agents", "content": "one"},
                    {"action": "update", "locator": "file:agents", "content": "two"},
                ],
            }
        )

    invalid = [
        {"action": "create", "locator": "file:agents", "content": "new"},
        {"action": "update", "locator": "file:soul", "content": "new"},
        {"action": "update", "locator": "file:agents", "content": "old"},
        {"action": "create", "locator": "skills:skills/nested/child/SKILL.md", "content": "new"},
        {"action": "create", "locator": "unknown:new", "content": "new"},
    ]
    for change in invalid:
        proposal = parse_candidate_proposal({"schema_version": 3, "changes": [change]})
        with pytest.raises((ValueError, KeyError)):
            materialize_candidate_proposal(
                proposal,
                baseline=baseline,
                resolve_locator=registry.resolve_locator,
            )
