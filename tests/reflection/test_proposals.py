from __future__ import annotations

import json

import pytest

from weave_agent_signals.runs.bundles import compare_bundles
from weave_agent_signals.runs.promotion import ProjectFileAdapter
from weave_agent_signals.runs.proposals import (
    materialize_candidate_proposal,
    parse_candidate_proposal,
)


def test_proposal_schema_is_closed_versioned_and_canonical():
    from weave_agent_signals.runs.proposals import (
        REFLECTION_PROPOSAL_SCHEMA,
        REFLECTION_PROPOSAL_SCHEMA_VERSION,
        CandidateProposal,
        parse_candidate_proposal,
        serialize_candidate_proposal,
    )

    assert REFLECTION_PROPOSAL_SCHEMA_VERSION == 2
    assert REFLECTION_PROPOSAL_SCHEMA.name == "reflection_proposal"
    schema = REFLECTION_PROPOSAL_SCHEMA.schema
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"] == {"const": 2, "type": "integer"}
    assert schema["properties"]["changes"]["minItems"] == 1
    change_schema = schema["properties"]["changes"]["items"]
    assert change_schema["additionalProperties"] is False
    assert set(change_schema["required"]) == {"action", "path", "content"}

    proposal = parse_candidate_proposal(
        json.dumps(
            {
                "schema_version": 2,
                "changes": [
                    {"action": "create", "path": "b.md", "content": "new B"},
                    {"action": "update", "path": "a.md", "content": "new A"},
                ],
            }
        )
    )

    assert isinstance(proposal, CandidateProposal)
    assert [change.path for change in proposal.changes] == ["a.md", "b.md"]
    assert serialize_candidate_proposal(proposal) == (
        '{"changes":['
        '{"action":"update","content":"new A","path":"a.md"},'
        '{"action":"create","content":"new B","path":"b.md"}'
        '],"schema_version":2}'
    )


def test_proposal_materializes_atomic_create_update_delete(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "AGENTS.md").write_text("old agents")
    (tmp_path / "docs" / "obsolete.md").write_text("obsolete")
    (tmp_path / "docs" / "keep.md").write_text("keep")
    adapter = ProjectFileAdapter(tmp_path, target_id="project")
    baseline = adapter.capture()
    builder_calls: list[dict[str, str]] = []

    def build_candidate(contents):
        builder_calls.append(dict(contents))
        return adapter.bundle_from_content_map(contents)

    proposal = parse_candidate_proposal(
        {
            "schema_version": 2,
            "changes": [
                {"action": "delete", "path": "docs/obsolete.md", "content": None},
                {"action": "create", "path": "docs/new.md", "content": "new"},
                {"action": "update", "path": "AGENTS.md", "content": "new agents"},
            ],
        }
    )

    candidate = materialize_candidate_proposal(
        proposal,
        baseline=baseline,
        build_candidate=build_candidate,
    )

    assert builder_calls == [
        {
            "AGENTS.md": "new agents",
            "docs/keep.md": "keep",
            "docs/new.md": "new",
        }
    ]
    assert [
        (action.action, action.locator) for action in compare_bundles(baseline, candidate).actions
    ] == [
        ("update", "AGENTS.md"),
        ("create", "docs/new.md"),
        ("delete", "docs/obsolete.md"),
    ]


def test_proposal_rejects_duplicates_extra_keys_and_invalid_content():
    duplicate = {
        "schema_version": 2,
        "changes": [
            {"action": "update", "path": "AGENTS.md", "content": "one"},
            {"action": "delete", "path": "AGENTS.md", "content": None},
        ],
    }
    with pytest.raises(ValueError, match="unique"):
        parse_candidate_proposal(duplicate)

    with pytest.raises(ValueError, match="exactly"):
        parse_candidate_proposal({**duplicate, "extra": True})

    extra_change_key = {
        "schema_version": 2,
        "changes": [
            {
                "action": "update",
                "path": "AGENTS.md",
                "content": "new",
                "reason": "not part of the contract",
            }
        ],
    }
    with pytest.raises(ValueError, match="exactly"):
        parse_candidate_proposal(extra_change_key)

    invalid_changes = [
        {"action": "create", "path": "new.md", "content": None},
        {"action": "update", "path": "AGENTS.md", "content": None},
        {"action": "delete", "path": "AGENTS.md", "content": "not null"},
        {"action": "create", "path": "new.md", "content": 1},
        {"action": "rename", "path": "AGENTS.md", "content": "new"},
        {"action": [], "path": "AGENTS.md", "content": "new"},
        {"action": "create", "path": "docs//new.md", "content": "new"},
    ]
    for change in invalid_changes:
        with pytest.raises(ValueError):
            parse_candidate_proposal({"schema_version": 2, "changes": [change]})


def test_proposal_rejects_action_inconsistent_with_baseline(tmp_path):
    (tmp_path / "AGENTS.md").write_text("old")
    adapter = ProjectFileAdapter(tmp_path, target_id="project")
    baseline = adapter.capture()
    builder_calls: list[dict[str, str]] = []

    def build_candidate(contents):
        builder_calls.append(dict(contents))
        return adapter.bundle_from_content_map(contents)

    invalid_changes = [
        {"action": "create", "path": "AGENTS.md", "content": "new"},
        {"action": "update", "path": "missing.md", "content": "new"},
        {"action": "update", "path": "AGENTS.md", "content": "old"},
        {"action": "delete", "path": "missing.md", "content": None},
    ]
    for change in invalid_changes:
        proposal = parse_candidate_proposal({"schema_version": 2, "changes": [change]})
        with pytest.raises(ValueError):
            materialize_candidate_proposal(
                proposal,
                baseline=baseline,
                build_candidate=build_candidate,
            )

    assert builder_calls == []
