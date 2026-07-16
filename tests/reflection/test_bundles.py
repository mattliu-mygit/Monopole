from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from weave_agent_signals.runs.bundles import (
    BundleSnapshot,
    BundleValidationError,
    ScopeDescriptor,
    TargetSnapshot,
    bundle_from_content_map,
    changed_locators,
    compare_bundles,
    validate_edited_bundle,
)


def _scope(target_id: str = "project-1") -> ScopeDescriptor:
    return ScopeDescriptor(
        kind="file",
        target_id=target_id,
        patterns=("CLAUDE.md", ".claude/commands/*.md", ".claude/skills/*.md"),
    )


def _bundle(
    contents: dict[str, str],
    *,
    missing: tuple[str, ...] = (),
    scope: ScopeDescriptor | None = None,
) -> BundleSnapshot:
    return bundle_from_content_map(
        contents,
        scope=scope or _scope(),
        include_missing=missing,
    )


def test_snapshots_have_stable_revisions_round_trip_and_are_immutable():
    scope = _scope()
    claude = TargetSnapshot(
        kind="file",
        locator="CLAUDE.md",
        display_name="CLAUDE.md",
        path="CLAUDE.md",
        exists=True,
        content="instructions\n",
    )
    missing = TargetSnapshot(
        kind="file",
        locator=".claude/skills/new.md",
        display_name="new.md",
        path=".claude/skills/new.md",
        exists=False,
        content=None,
    )

    original = BundleSnapshot(targets=(claude, missing), scope=scope)
    reordered = BundleSnapshot(targets=(missing, claude), scope=scope)

    assert reordered == original
    assert reordered.revision == original.revision
    assert TargetSnapshot.from_dict(claude.to_dict()) == claude
    assert BundleSnapshot.from_dict(original.to_dict()) == original
    assert (
        TargetSnapshot(kind="file", locator="CLAUDE.md", exists=True, content="changed").revision
        != claude.revision
    )
    with pytest.raises(FrozenInstanceError):
        claude.content = "mutated"  # type: ignore[misc]
    with pytest.raises(BundleValidationError, match="revision mismatch"):
        BundleSnapshot.from_dict({**original.to_dict(), "revision": "sha256:tampered"})
    assert original.target("CLAUDE.md") is claude
    with pytest.raises(KeyError):
        original.target("missing.md")


def test_compare_bundles_supports_create_update_and_preserves_exact_bundles():
    baseline = _bundle(
        {
            "CLAUDE.md": "old",
            ".claude/commands/keep.md": "keep me",
        }
    )
    candidate = _bundle(
        {
            "CLAUDE.md": "new",
            ".claude/commands/keep.md": "keep me",
            ".claude/skills/new.md": "created",
        }
    )
    baseline_revision = baseline.revision
    candidate_revision = candidate.revision
    baseline_targets = baseline.targets
    candidate_targets = candidate.targets

    preview = compare_bundles(baseline, candidate)

    assert preview.past is baseline
    assert preview.proposed is candidate
    assert preview.past.revision == baseline_revision
    assert preview.proposed.revision == candidate_revision
    assert preview.past.targets == baseline_targets
    assert preview.proposed.targets == candidate_targets
    assert preview.created_locators == (".claude/skills/new.md",)
    assert preview.updated_locators == ("CLAUDE.md",)
    assert ".claude/skills/new.md" not in preview.past.by_locator
    create = next(action for action in preview.actions if action.action == "create")
    assert create.before.exists is False
    assert create.after is candidate.by_locator[create.locator]


def test_compare_bundles_rejects_delete_actions():
    baseline = _bundle({"CLAUDE.md": "old", "obsolete.md": "remove"})
    candidate = _bundle({"CLAUDE.md": "new"})

    with pytest.raises(BundleValidationError, match="delete"):
        compare_bundles(baseline, candidate)


def test_changed_locators_is_union_aware_without_conflating_missing_state():
    absent = _bundle({"CLAUDE.md": "same"})
    explicit_missing = _bundle(
        {"CLAUDE.md": "same"},
        missing=(".claude/skills/new.md",),
    )
    created = _bundle(
        {
            "CLAUDE.md": "same",
            ".claude/skills/new.md": "new",
        }
    )

    assert changed_locators(absent, explicit_missing) == ()
    assert changed_locators(absent, created) == (".claude/skills/new.md",)


def test_content_map_conversion_rejects_duplicate_or_invalid_values():
    with pytest.raises(BundleValidationError, match="duplicate"):
        bundle_from_content_map(
            {"CLAUDE.md": "one"},
            scope=_scope(),
            include_missing=(".claude/skills/new.md", ".claude/skills/new.md"),
        )
    with pytest.raises(BundleValidationError):
        bundle_from_content_map({"CLAUDE.md": object()}, scope=_scope())


def test_validate_edited_bundle_allows_content_only_for_the_same_action_identity():
    baseline = _bundle(
        {
            "CLAUDE.md": "old",
            ".claude/commands/remove.md": "remove",
        }
    )
    evaluated = _bundle(
        {
            "CLAUDE.md": "candidate",
            ".claude/commands/remove.md": "remove",
            ".claude/skills/new.md": "candidate skill",
        }
    )
    edited = _bundle(
        {
            "CLAUDE.md": "edited",
            ".claude/commands/remove.md": "remove",
            ".claude/skills/new.md": "edited skill",
        }
    )
    revisions = baseline.revision, evaluated.revision, edited.revision

    validate_edited_bundle(baseline, evaluated, edited)

    assert (baseline.revision, evaluated.revision, edited.revision) == revisions
    assert baseline.targets[0] is baseline.targets[0]


@pytest.mark.parametrize(
    "edited",
    [
        _bundle({"CLAUDE.md": "edited"}),
        _bundle(
            {
                "CLAUDE.md": "edited",
                ".claude/skills/new.md": "candidate skill",
                ".claude/commands/extra.md": "extra",
            }
        ),
        _bundle(
            {
                "CLAUDE.md": "old",
                ".claude/skills/new.md": "candidate skill",
            }
        ),
    ],
)
def test_validate_edited_bundle_rejects_target_or_action_changes(edited):
    baseline = _bundle({"CLAUDE.md": "old"})
    evaluated = _bundle(
        {
            "CLAUDE.md": "candidate",
            ".claude/skills/new.md": "candidate skill",
        }
    )

    with pytest.raises(BundleValidationError):
        validate_edited_bundle(baseline, evaluated, edited)


@pytest.mark.parametrize("field", ["kind", "display_name", "path", "exists", "scope"])
def test_validate_edited_bundle_rejects_non_content_identity_changes(field: str):
    baseline = _bundle({"CLAUDE.md": "old"})
    evaluated = _bundle({"CLAUDE.md": "candidate"})
    target = evaluated.targets[0]
    values = {
        "kind": target.kind,
        "locator": target.locator,
        "exists": target.exists,
        "content": "edited",
        "display_name": target.display_name,
        "path": target.path,
    }
    if field == "scope":
        edited = BundleSnapshot((target,), _scope("other-project"))
    else:
        values[field] = {
            "kind": "prompt",
            "display_name": "renamed.md",
            "path": None,
            "exists": False,
        }[field]
        if field == "exists":
            values["content"] = None
        edited = BundleSnapshot((TargetSnapshot(**values),), evaluated.scope)

    with pytest.raises(BundleValidationError):
        validate_edited_bundle(baseline, evaluated, edited)
