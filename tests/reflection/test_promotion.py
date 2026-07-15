"""Focused safety tests for exact-bundle project-file promotion."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

import weave_agent_signals.runs.promotion as promotion_module
from weave_agent_signals.runs.bundles import (
    BundleSnapshot,
    TargetSnapshot,
    compare_bundles,
)
from weave_agent_signals.runs.promotion import (
    ProjectFileAdapter,
    ProjectMarkdownPolicy,
    PromotionReceipt,
    PromotionTransactionError,
    PromotionValidationError,
    StaleBaseError,
    target_adapter_contract_manifest,
)


def _write(root: Path, locator: str, content: str) -> None:
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _promote(
    adapter: ProjectFileAdapter,
    past: BundleSnapshot,
    evaluated_candidate: BundleSnapshot,
    promoted: BundleSnapshot,
    *,
    promotion_id: str = "promotion-1",
    git_metadata: dict | None = None,
) -> PromotionReceipt:
    edited = promoted != evaluated_candidate
    return adapter.promote(
        promotion_id=promotion_id,
        run_id="run-1",
        candidate_id="candidate-1",
        past=past,
        evaluated_candidate=evaluated_candidate,
        promoted=promoted,
        review_revision=1,
        acknowledge_unevaluated=edited,
        git_metadata=git_metadata,
    )


def test_capture_includes_all_project_owned_markdown_recursively(tmp_path: Path):
    expected = (
        "AGENTS.md",
        "README.md",
        "docs/guide.md",
        "specs/nested/design.md",
    )
    for locator in expected:
        _write(tmp_path, locator, locator)
    _write(tmp_path, "docs/ignored.MD", "wrong suffix")
    _write(tmp_path, "specs/nested/ignored.txt", "wrong suffix")

    adapter = ProjectFileAdapter(tmp_path)

    assert adapter.capture().locators == expected


def test_capture_recurses_through_directory_named_with_markdown_suffix(tmp_path: Path):
    _write(tmp_path, "archive.md/nested.md", "nested")

    assert ProjectFileAdapter(tmp_path).capture().locators == ("archive.md/nested.md",)


def test_capture_includes_hidden_instruction_directories(tmp_path: Path):
    expected = (
        ".agents/review/check.md",
        ".claude/commands/run.md",
        ".codex/instructions.md",
        ".hidden/project-note.md",
    )
    for locator in expected:
        _write(tmp_path, locator, locator)

    assert ProjectFileAdapter(tmp_path).capture().locators == expected


def test_capture_excludes_dependencies_caches_builds_and_temp_plans(tmp_path: Path):
    policy = ProjectMarkdownPolicy()
    assert policy.suffix == ".md"
    assert (policy.max_files, policy.max_file_bytes, policy.max_total_bytes) == (
        500,
        256 * 1024,
        512 * 1024,
    )
    assert policy.excluded_directories == (
        ".git",
        ".weave-agent-signals",
        ".superpowers",
        ".venv",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "__pycache__",
        "dist",
        "build",
        "coverage",
    )
    for directory in policy.excluded_directories:
        _write(tmp_path, f"{directory}/ignored.md", directory)
        _write(tmp_path, f"docs/{directory}/nested-ignored.md", directory)
    _write(tmp_path, "docs/kept.md", "kept")

    adapter = ProjectFileAdapter(tmp_path, policy=policy)

    assert adapter.capture().locators == ("docs/kept.md",)
    assert adapter.contract_manifest() == target_adapter_contract_manifest(policy)


def test_capture_rejects_symlinked_markdown_and_directory_scope(tmp_path: Path):
    outside_file = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside_file.write_text("outside", encoding="utf-8")
    file_root = tmp_path / "file-symlink"
    file_root.mkdir()
    (file_root / "linked.md").symlink_to(outside_file)

    with pytest.raises(PromotionValidationError, match="symlink"):
        ProjectFileAdapter(file_root).capture()

    outside_directory = tmp_path.parent / f"{tmp_path.name}-outside-directory"
    outside_directory.mkdir()
    _write(outside_directory, "hidden.md", "outside")
    directory_root = tmp_path / "directory-symlink"
    directory_root.mkdir()
    (directory_root / "docs").symlink_to(outside_directory, target_is_directory=True)

    with pytest.raises(PromotionValidationError, match="symlink"):
        ProjectFileAdapter(directory_root).capture()

    excluded_root = tmp_path / "excluded-symlink"
    excluded_root.mkdir()
    (excluded_root / ".git").symlink_to(outside_directory, target_is_directory=True)
    assert ProjectFileAdapter(excluded_root).capture().targets == ()


def test_parent_symlink_swap_during_capture_cannot_read_outside_scope(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "docs/target.md", "inside")
    outside = tmp_path.parent / f"{tmp_path.name}-outside-capture"
    outside.mkdir()
    (outside / "target.md").write_text("outside secret", encoding="utf-8")
    adapter = ProjectFileAdapter(tmp_path)
    real_path = adapter._path
    managed = tmp_path / "docs"
    moved = tmp_path / "docs-before-capture-swap"
    swapped = False

    def raced_path(locator: str) -> Path:
        nonlocal swapped
        result = real_path(locator)
        if locator == "docs/target.md" and not swapped:
            managed.rename(moved)
            managed.symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(adapter, "_path", raced_path)

    with pytest.raises(PromotionValidationError, match="changed during capture"):
        adapter.capture()


def test_directory_swap_during_discovery_cannot_add_outside_locators(
    tmp_path: Path,
    monkeypatch,
):
    managed = tmp_path / "docs"
    managed.mkdir()
    (managed / "ignored.txt").write_text("inside", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside-discovery"
    outside.mkdir()
    (outside / "secret.md").write_text("outside", encoding="utf-8")
    adapter = ProjectFileAdapter(tmp_path)
    moved = tmp_path / "docs-before-discovery-swap"
    real_iterdir = Path.iterdir
    real_validate = adapter._validate_casefold_locators
    swapped = False

    def raced_iterdir(path: Path):
        nonlocal swapped
        if path == managed and not swapped:
            managed.rename(moved)
            managed.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_iterdir(path)

    def restore_before_read(locators) -> None:
        nonlocal swapped
        if swapped and managed.is_symlink():
            managed.unlink()
            moved.rename(managed)
            swapped = False
        real_validate(locators)

    monkeypatch.setattr(Path, "iterdir", raced_iterdir)
    monkeypatch.setattr(adapter, "_validate_casefold_locators", restore_before_read)

    assert adapter.capture().locators == ()


def test_capture_rejects_casefold_collisions_and_non_utf8_content(
    tmp_path: Path,
    monkeypatch,
):
    collision_root = tmp_path / "collision"
    collision_root.mkdir()
    _write(collision_root, "A.md", "one")
    adapter = ProjectFileAdapter(collision_root)
    monkeypatch.setattr(adapter, "_discover", lambda: {"A.md", "a.md"})

    with pytest.raises(PromotionValidationError, match="case-fold") as exc_info:
        adapter.capture()

    assert exc_info.value.changed_locators == ("A.md", "a.md")

    invalid_root = tmp_path / "invalid-utf8"
    invalid_root.mkdir()
    (invalid_root / "invalid.md").write_bytes(b"\xff")
    with pytest.raises(PromotionValidationError, match="valid UTF-8"):
        ProjectFileAdapter(invalid_root).capture()


@pytest.mark.parametrize(
    ("policy", "contents", "message"),
    [
        (
            ProjectMarkdownPolicy(max_files=1),
            {"one.md": "1", "two.md": "2"},
            "file count limit",
        ),
        (
            ProjectMarkdownPolicy(max_file_bytes=3),
            {"large.md": "1234"},
            "per-file byte limit",
        ),
        (
            ProjectMarkdownPolicy(max_total_bytes=3),
            {"one.md": "12", "two.md": "34"},
            "total byte limit",
        ),
    ],
)
def test_capture_fails_cleanly_at_each_scope_limit(
    tmp_path: Path,
    policy: ProjectMarkdownPolicy,
    contents: dict[str, str],
    message: str,
):
    for locator, content in contents.items():
        _write(tmp_path, locator, content)

    with pytest.raises(PromotionValidationError, match=message):
        ProjectFileAdapter(tmp_path, policy=policy).capture()


def test_capture_rejects_cross_device_and_unreadable_entries(tmp_path: Path, monkeypatch):
    cross_device_root = tmp_path / "cross-device"
    cross_device_root.mkdir()
    _write(cross_device_root, "docs/guide.md", "guide")
    cross_device = cross_device_root / "docs"
    cross_device_identity = cross_device.stat()
    real_fstat = os.fstat

    def cross_device_fstat(descriptor: int):
        result = real_fstat(descriptor)
        if (result.st_dev, result.st_ino) == (
            cross_device_identity.st_dev,
            cross_device_identity.st_ino,
        ):
            values = list(result)
            values[2] += 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(os, "fstat", cross_device_fstat)
    with pytest.raises(PromotionValidationError, match="another filesystem"):
        ProjectFileAdapter(cross_device_root).capture()
    monkeypatch.setattr(os, "fstat", real_fstat)

    unreadable_root = tmp_path / "unreadable"
    unreadable_root.mkdir()
    blocked = unreadable_root / "docs"
    blocked.mkdir()
    _write(unreadable_root, "docs/guide.md", "guide")
    blocked_identity = blocked.stat()
    real_scandir = os.scandir

    def unreadable_scandir(path):
        if isinstance(path, int):
            identity = real_fstat(path)
        else:
            identity = os.stat(path)
        if (identity.st_dev, identity.st_ino) == (
            blocked_identity.st_dev,
            blocked_identity.st_ino,
        ):
            raise PermissionError("denied")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", unreadable_scandir)
    with pytest.raises(PromotionValidationError, match="unreadable"):
        ProjectFileAdapter(unreadable_root).capture()


def test_multi_file_markdown_promotion_preserves_journal_and_rollback(tmp_path: Path):
    _write(tmp_path, "AGENTS.md", "old agents")
    _write(tmp_path, "docs/remove.md", "remove")
    _write(tmp_path, ".agents/review.md", "old review")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map(
        {
            "AGENTS.md": "new agents",
            ".agents/review.md": "new review",
            "specs/new/design.md": "new design",
        }
    )

    receipt = _promote(adapter, past, promoted, promoted)

    journal = adapter.transaction_path("promotion-1") / "journal.json"
    assert journal.is_file()
    assert json.loads(journal.read_text(encoding="utf-8"))["state"] == "committed"
    assert adapter.capture() == promoted
    assert adapter.rollback_committed("promotion-1") == receipt
    assert adapter.capture() == past


def test_content_map_rejects_every_unsafe_or_excluded_markdown_path(tmp_path: Path):
    adapter = ProjectFileAdapter(tmp_path)
    invalid = (
        "/absolute.md",
        "dir\\windows.md",
        "../traversal.md",
        "dir/../traversal.md",
        "./dot.md",
        "dir//empty.md",
        "nul\x00.md",
        "uppercase.MD",
        ".git/internal.md",
        "docs/node_modules/dependency.md",
    )

    for locator in invalid:
        with pytest.raises(PromotionValidationError, match="outside the managed scope"):
            adapter.bundle_from_content_map({locator: "invalid"})


def test_content_map_boundary_rejects_unmanaged_symlink_collision_and_non_utf8(
    tmp_path: Path,
):
    adapter = ProjectFileAdapter(tmp_path)
    with pytest.raises(PromotionValidationError, match="outside the managed scope"):
        adapter.bundle_from_content_map({"../escape.md": "nope"})
    with pytest.raises(PromotionValidationError, match="UTF-8"):
        adapter.bundle_from_content_map({"CLAUDE.md": "bad\udcff"})

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / ".claude").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PromotionValidationError, match="symlink"):
        adapter.bundle_from_content_map({".claude/skills/new.md": "skill"})
    (tmp_path / ".claude").unlink()

    (tmp_path / ".claude/skills/collision.md").mkdir(parents=True)
    with pytest.raises(PromotionValidationError, match="collision"):
        adapter.bundle_from_content_map({".claude/skills/collision.md": "skill"})


def test_capture_is_complete_and_rejects_non_utf8_or_symlinked_scope(tmp_path: Path):
    _write(tmp_path, "CLAUDE.md", "root")
    _write(tmp_path, ".claude/commands/one.md", "command")
    _write(tmp_path, ".claude/skills/two.md", "skill")
    _write(tmp_path, ".claude/skills/ignored.txt", "ignored")
    adapter = ProjectFileAdapter(tmp_path)

    captured = adapter.capture()

    assert captured.locators == (
        ".claude/commands/one.md",
        ".claude/skills/two.md",
        "CLAUDE.md",
    )
    (tmp_path / "CLAUDE.md").write_bytes(b"\xff")
    with pytest.raises(PromotionValidationError, match="valid UTF-8"):
        adapter.capture()


def test_multi_file_receipt_preserves_exact_b_c_and_promoted_c(tmp_path: Path):
    _write(tmp_path, "CLAUDE.md", "old")
    _write(tmp_path, ".claude/commands/remove.md", "remove")
    adapter = ProjectFileAdapter(
        tmp_path,
        target_id="project-1",
        clock=lambda: datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
    )
    past = adapter.capture()
    evaluated = adapter.bundle_from_content_map(
        {"CLAUDE.md": "new", ".claude/skills/new.md": "created"}
    )
    revisions = past.revision, evaluated.revision
    targets = past.targets, evaluated.targets

    receipt = _promote(
        adapter,
        past,
        evaluated,
        evaluated,
        git_metadata={"commit": "abc123", "dirty": False},
    )

    assert receipt.past is past
    assert receipt.evaluated_candidate is evaluated
    assert receipt.promoted is evaluated
    assert (past.revision, evaluated.revision) == revisions
    assert (past.targets, evaluated.targets) == targets
    assert receipt.actions == compare_bundles(past, evaluated).actions
    assert receipt.created_locators == (".claude/skills/new.md",)
    assert receipt.updated_locators == ("CLAUDE.md",)
    assert receipt.deleted_locators == (".claude/commands/remove.md",)
    assert receipt.promoted_was_evaluated is True
    assert receipt.unevaluated_d_acknowledged is False
    assert receipt.decided_at == "2026-07-14T12:00:00+00:00"
    assert dict(receipt.git_metadata or {}) == {"commit": "abc123", "dirty": False}
    serialized = receipt.to_dict()
    assert set(serialized) == {
        "promotion_id",
        "run_id",
        "candidate_id",
        "target_kind",
        "target_id",
        "past",
        "evaluated_candidate",
        "promoted",
        "review_revision",
        "actions",
        "decided_at",
        "promoted_was_evaluated",
        "unevaluated_d_acknowledged",
        "git_metadata",
    }
    assert PromotionReceipt.from_dict(serialized) == receipt
    with pytest.raises(PromotionValidationError):
        PromotionReceipt.from_dict({**serialized, "proposed": serialized["promoted"]})
    assert adapter.capture() == receipt.promoted
    assert adapter.acknowledge("promotion-1") is True
    assert adapter.acknowledge("promotion-1") is False


def test_edited_d_receipt_preserves_exact_b_c_d_and_requires_acknowledgement(
    tmp_path: Path,
):
    _write(tmp_path, "CLAUDE.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    evaluated = adapter.bundle_from_content_map({"CLAUDE.md": "candidate"})
    edited = adapter.bundle_from_content_map({"CLAUDE.md": "edited"})

    with pytest.raises(PromotionValidationError, match="acknowledgement"):
        adapter.promote(
            promotion_id="missing-ack",
            run_id="run-1",
            candidate_id="candidate-1",
            past=past,
            evaluated_candidate=evaluated,
            promoted=edited,
            review_revision=1,
        )

    receipt = _promote(adapter, past, evaluated, edited)

    assert receipt.past is past
    assert receipt.evaluated_candidate is evaluated
    assert receipt.promoted is edited
    assert receipt.actions == compare_bundles(past, edited).actions
    assert receipt.promoted_was_evaluated is False
    assert receipt.unevaluated_d_acknowledged is True
    assert receipt.evaluated_candidate.revision != receipt.promoted.revision


def test_evaluated_c_rejects_unevaluated_acknowledgement(tmp_path: Path):
    _write(tmp_path, "CLAUDE.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    evaluated = adapter.bundle_from_content_map({"CLAUDE.md": "candidate"})

    with pytest.raises(PromotionValidationError, match="must not claim"):
        adapter.promote(
            promotion_id="bad-ack",
            run_id="run-1",
            candidate_id="candidate-1",
            past=past,
            evaluated_candidate=evaluated,
            promoted=evaluated,
            review_revision=1,
            acknowledge_unevaluated=True,
        )


@pytest.mark.parametrize("change", ["add", "remove", "path", "action"])
def test_promotion_rejects_edited_bundle_identity_or_action_changes(
    tmp_path: Path,
    change: str,
):
    created = ".claude/skills/new.md"
    _write(tmp_path, "CLAUDE.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    evaluated = adapter.bundle_from_content_map(
        {"CLAUDE.md": "candidate", created: "candidate skill"}
    )
    if change == "add":
        edited = adapter.bundle_from_content_map(
            {
                "CLAUDE.md": "edited",
                created: "candidate skill",
                ".claude/commands/extra.md": "extra",
            }
        )
    elif change == "remove":
        edited = adapter.bundle_from_content_map({"CLAUDE.md": "edited"})
    elif change == "action":
        edited = adapter.bundle_from_content_map({"CLAUDE.md": "old", created: "candidate skill"})
    else:
        edited = BundleSnapshot(
            tuple(
                TargetSnapshot(
                    kind=target.kind,
                    locator=target.locator,
                    exists=target.exists,
                    content="edited" if target.locator == "CLAUDE.md" else target.content,
                    display_name=target.display_name,
                    path=None if target.locator == "CLAUDE.md" else target.path,
                )
                for target in evaluated.targets
            ),
            evaluated.scope,
        )

    with pytest.raises(PromotionValidationError):
        _promote(adapter, past, evaluated, edited)

    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "old"
    assert not adapter.journal_root.exists()


def test_whole_scope_drift_blocks_before_journal_or_write(tmp_path: Path):
    _write(tmp_path, "CLAUDE.md", "old")
    _write(tmp_path, ".claude/skills/existing.md", "existing")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    evaluated = adapter.bundle_from_content_map(
        {"CLAUDE.md": "new", ".claude/skills/existing.md": "existing"}
    )
    _write(tmp_path, ".claude/commands/unrelated.md", "drift")

    with pytest.raises(StaleBaseError) as exc_info:
        _promote(adapter, past, evaluated, evaluated)

    assert exc_info.value.expected is past
    assert exc_info.value.changed_locators == (".claude/commands/unrelated.md",)
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "old"
    assert not adapter.journal_root.exists()


@pytest.mark.parametrize("collision_kind", ["directory", "symlink"])
def test_managed_path_collision_after_snapshot_is_reported_as_drift(
    tmp_path: Path,
    collision_kind: str,
):
    _write(tmp_path, "CLAUDE.md", "old")
    created = ".claude/skills/new.md"
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    evaluated = adapter.bundle_from_content_map({"CLAUDE.md": "old", created: "new skill"})
    collision = tmp_path / created
    collision.parent.mkdir(parents=True)
    if collision_kind == "directory":
        collision.mkdir()
    else:
        outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
        outside.write_text("outside", encoding="utf-8")
        collision.symlink_to(outside)

    with pytest.raises(StaleBaseError) as exc_info:
        _promote(adapter, past, evaluated, evaluated)

    assert exc_info.value.expected is past
    assert exc_info.value.changed_locators == (created,)
    if collision_kind == "directory":
        assert collision.is_dir()
    else:
        assert collision.is_symlink()
    assert not adapter.journal_root.exists()


@pytest.mark.parametrize("action", ["create", "update", "delete"])
def test_parent_symlink_swap_during_apply_cannot_mutate_outside_scope(
    tmp_path: Path,
    monkeypatch,
    action: str,
):
    managed = tmp_path / "docs"
    managed.mkdir()
    (managed / "ignored.txt").write_text("keeps parent present", encoding="utf-8")
    if action in {"update", "delete"}:
        _write(tmp_path, "docs/target.md", "old")

    outside = tmp_path.parent / f"{tmp_path.name}-outside-{action}"
    outside.mkdir()
    outside_target = outside / "target.md"
    if action in {"update", "delete"}:
        outside_target.write_text("outside", encoding="utf-8")

    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    if action == "create":
        promoted = adapter.bundle_from_content_map({"docs/target.md": "new"})
    elif action == "update":
        promoted = adapter.bundle_from_content_map({"docs/target.md": "new"})
    else:
        promoted = adapter.bundle_from_content_map({}, include_missing=past.locators)

    moved = tmp_path / "docs-before-swap"
    swapped = False

    def swap_parent() -> None:
        nonlocal swapped
        if swapped:
            return
        managed.rename(moved)
        managed.symlink_to(outside, target_is_directory=True)
        swapped = True

    if action == "delete":
        real_unlink = adapter._unlink_live

        def raced_unlink(destination: Path) -> None:
            swap_parent()
            real_unlink(destination)

        monkeypatch.setattr(adapter, "_unlink_live", raced_unlink)
    else:
        real_replace = adapter._replace_staged

        def raced_replace(staged: Path, destination: Path) -> None:
            swap_parent()
            real_replace(staged, destination)

        monkeypatch.setattr(adapter, "_replace_staged", raced_replace)

    with pytest.raises(PromotionTransactionError):
        _promote(adapter, past, promoted, promoted)

    if action == "create":
        assert not outside_target.exists()
    else:
        assert outside_target.read_text(encoding="utf-8") == "outside"


def test_parent_symlink_swap_during_backup_cannot_capture_outside_content(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "docs/target.md", "old")
    outside = tmp_path.parent / f"{tmp_path.name}-outside-backup"
    outside.mkdir()
    (outside / "target.md").write_text("outside secret", encoding="utf-8")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"docs/target.md": "new"})
    real_path = adapter._path
    managed = tmp_path / "docs"
    moved = tmp_path / "docs-before-backup-swap"
    swapped = False

    def raced_path(locator: str) -> Path:
        nonlocal swapped
        result = real_path(locator)
        transaction = adapter.transaction_path("promotion-1")
        if locator == "docs/target.md" and transaction.exists() and not swapped:
            managed.rename(moved)
            managed.symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(adapter, "_path", raced_path)

    with pytest.raises((PromotionTransactionError, StaleBaseError)):
        _promote(adapter, past, promoted, promoted)

    assert not adapter.transaction_path("promotion-1").exists()
    assert (outside / "target.md").read_text(encoding="utf-8") == "outside secret"


def test_metadata_symlink_swap_cannot_redirect_promotion_journal(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "CLAUDE.md", "old")
    metadata = tmp_path / ".weave-agent-signals"
    (metadata / "promotions").mkdir(parents=True)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-journal"
    (outside / "promotions").mkdir(parents=True)
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"CLAUDE.md": "new"})
    moved = tmp_path / ".weave-agent-signals-before-swap"
    real_journal_parents = adapter._journal_parents
    swapped = False

    def raced_journal_parents(*, create: bool = False) -> None:
        nonlocal swapped
        real_journal_parents(create=create)
        if create and not swapped:
            metadata.rename(moved)
            metadata.symlink_to(outside, target_is_directory=True)
            swapped = True

    monkeypatch.setattr(adapter, "_journal_parents", raced_journal_parents)

    with pytest.raises(PromotionTransactionError):
        _promote(adapter, past, promoted, promoted)

    assert not (outside / "promotions" / adapter.transaction_path("promotion-1").name).exists()
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "old"


def test_failure_after_partial_apply_rolls_back_every_file(tmp_path: Path, monkeypatch):
    _write(tmp_path, "CLAUDE.md", "old root")
    _write(tmp_path, ".claude/commands/update.md", "old command")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map(
        {
            "CLAUDE.md": "new root",
            ".claude/commands/update.md": "new command",
            ".claude/skills/new.md": "new skill",
        }
    )
    real_replace = adapter._replace_staged

    def fail_on_root(staged: Path, destination: Path) -> None:
        if destination.name == "CLAUDE.md":
            raise OSError("injected replace failure")
        real_replace(staged, destination)

    monkeypatch.setattr(adapter, "_replace_staged", fail_on_root)

    with pytest.raises(PromotionTransactionError) as exc_info:
        _promote(adapter, past, promoted, promoted)

    assert exc_info.value.changed_locators == (
        ".claude/commands/update.md",
        ".claude/skills/new.md",
        "CLAUDE.md",
    )
    assert adapter.capture() == past
    assert adapter.recover("promotion-1") is None


def test_apply_failure_preserves_directory_created_by_another_actor(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "AGENTS.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"AGENTS.md": "old", "docs/new.md": "new"})
    external = tmp_path / "docs"

    def external_create_then_fail(*_args) -> None:
        external.mkdir()
        raise OSError("injected apply failure")

    monkeypatch.setattr(adapter, "_apply", external_create_then_fail)

    with pytest.raises(PromotionTransactionError):
        _promote(adapter, past, promoted, promoted)

    assert external.is_dir()
    assert tuple(external.iterdir()) == ()


def test_recovery_preserves_replacement_for_transaction_created_directory(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "AGENTS.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"AGENTS.md": "old", "docs/new.md": "new"})
    real_replace = adapter._replace_staged

    def crash_before_target_replace(staged: Path, destination: Path) -> None:
        if destination == tmp_path / "docs/new.md":
            raise SystemExit("crash")
        real_replace(staged, destination)

    monkeypatch.setattr(adapter, "_replace_staged", crash_before_target_replace)
    with pytest.raises(SystemExit):
        _promote(adapter, past, promoted, promoted)
    monkeypatch.undo()

    created = tmp_path / "docs"
    created_inode = created.stat().st_ino
    created.rmdir()
    for index in range(4):
        (tmp_path / f"inode-occupier-{index}").mkdir()
    created.mkdir()
    assert created.stat().st_ino != created_inode

    assert adapter.recover("promotion-1") is None
    assert created.is_dir()
    assert tuple(created.iterdir()) == ()


def test_final_scope_mismatch_rolls_back_without_removing_user_file(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "CLAUDE.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"CLAUDE.md": "new"})
    real_replace = adapter._replace_staged

    def add_unexpected_file(staged: Path, destination: Path) -> None:
        real_replace(staged, destination)
        if destination == tmp_path / "CLAUDE.md":
            _write(tmp_path, ".claude/commands/unexpected.md", "unexpected")

    monkeypatch.setattr(adapter, "_replace_staged", add_unexpected_file)

    with pytest.raises(PromotionTransactionError):
        _promote(adapter, past, promoted, promoted)

    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "old"
    assert (tmp_path / ".claude/commands/unexpected.md").read_text() == "unexpected"


def test_committed_journal_is_idempotent_recoverable_and_rollback_safe(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "CLAUDE.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"CLAUDE.md": "new"})
    first = _promote(adapter, past, promoted, promoted)

    monkeypatch.setattr(
        adapter,
        "_replace_staged",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not rewrite")),
    )
    second = _promote(adapter, past, promoted, promoted)

    assert second == first
    assert adapter.recover("promotion-1") == first
    assert adapter.find_committed_receipt("run-1") == first
    monkeypatch.undo()
    assert adapter.rollback_committed("promotion-1") == first
    assert adapter.capture() == past


def test_committed_rollback_refuses_to_overwrite_later_user_edit(tmp_path: Path):
    _write(tmp_path, "CLAUDE.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"CLAUDE.md": "new"})
    _promote(adapter, past, promoted, promoted)
    _write(tmp_path, "CLAUDE.md", "later user edit")

    with pytest.raises(PromotionTransactionError) as exc_info:
        adapter.rollback_committed("promotion-1")

    assert exc_info.value.changed_locators == ("CLAUDE.md",)
    assert (tmp_path / "CLAUDE.md").read_text() == "later user edit"


def test_parent_symlink_swap_during_committed_rollback_cannot_mutate_outside_scope(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "docs/target.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"docs/target.md": "new"})
    _promote(adapter, past, promoted, promoted)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-rollback"
    outside.mkdir()
    outside_target = outside / "target.md"
    outside_target.write_text("outside", encoding="utf-8")
    managed = tmp_path / "docs"
    moved = tmp_path / "docs-before-rollback-swap"
    real_replace = promotion_module.os.replace
    swapped = False

    def raced_replace(source, destination, *args, **kwargs):
        nonlocal swapped
        is_target = destination == tmp_path / "docs/target.md" or (
            destination == "target.md" and kwargs.get("dst_dir_fd") is not None
        )
        if is_target and not swapped:
            managed.rename(moved)
            managed.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(promotion_module.os, "replace", raced_replace)

    with pytest.raises(PromotionTransactionError):
        adapter.rollback_committed("promotion-1")

    assert outside_target.read_text(encoding="utf-8") == "outside"


def test_committed_rollback_retries_after_crash_with_restore_stage_present(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "CLAUDE.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"CLAUDE.md": "new"})
    receipt = _promote(adapter, past, promoted, promoted)
    real_replace = adapter._replace_staged

    def crash_after_restore_staging(staged: Path, destination: Path) -> None:
        if staged.parent.name == "restore":
            raise SystemExit("crash")
        real_replace(staged, destination)

    monkeypatch.setattr(adapter, "_replace_staged", crash_after_restore_staging)
    with pytest.raises(SystemExit):
        adapter.rollback_committed("promotion-1")
    monkeypatch.undo()

    assert adapter.rollback_committed("promotion-1") == receipt
    assert adapter.capture() == past


def test_recovery_refuses_to_overwrite_content_changed_after_interruption(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "CLAUDE.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"CLAUDE.md": "new"})
    monkeypatch.setattr(
        adapter,
        "_apply",
        lambda *_args: (_ for _ in ()).throw(SystemExit("crash")),
    )
    with pytest.raises(SystemExit):
        _promote(adapter, past, promoted, promoted)
    _write(tmp_path, "CLAUDE.md", "later user edit")

    with pytest.raises(PromotionTransactionError) as exc_info:
        adapter.recover("promotion-1")

    assert exc_info.value.changed_locators == ("CLAUDE.md",)
    assert (tmp_path / "CLAUDE.md").read_text() == "later user edit"


def test_parent_symlink_swap_during_recovery_cannot_mutate_outside_scope(
    tmp_path: Path,
    monkeypatch,
):
    _write(tmp_path, "docs/a.md", "old a")
    _write(tmp_path, "docs/b.md", "old b")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"docs/a.md": "new a", "docs/b.md": "new b"})

    def crash_after_first_apply(transaction, preview, _directories, applied_indexes):
        first = preview.actions[0]
        adapter._replace_staged(
            transaction / "stage" / "0000",
            tmp_path / first.locator,
        )
        applied_indexes.append(0)
        raise SystemExit("crash")

    monkeypatch.setattr(adapter, "_apply", crash_after_first_apply)
    with pytest.raises(SystemExit):
        _promote(adapter, past, promoted, promoted)
    monkeypatch.undo()

    outside = tmp_path.parent / f"{tmp_path.name}-outside-recovery"
    outside.mkdir()
    outside_target = outside / "a.md"
    outside_target.write_text("outside", encoding="utf-8")
    managed = tmp_path / "docs"
    moved = tmp_path / "docs-before-recovery-swap"
    real_replace = promotion_module.os.replace
    swapped = False

    def raced_replace(source, destination, *args, **kwargs):
        nonlocal swapped
        is_target = destination == tmp_path / "docs/a.md" or (
            destination == "a.md" and kwargs.get("dst_dir_fd") is not None
        )
        if is_target and not swapped:
            managed.rename(moved)
            managed.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(promotion_module.os, "replace", raced_replace)

    with pytest.raises(PromotionTransactionError):
        adapter.recover("promotion-1")

    assert outside_target.read_text(encoding="utf-8") == "outside"


def test_project_lock_serializes_competing_promotions(tmp_path: Path, monkeypatch):
    _write(tmp_path, "CLAUDE.md", "old")
    first_adapter = ProjectFileAdapter(tmp_path)
    second_adapter = ProjectFileAdapter(tmp_path)
    past = first_adapter.capture()
    first_bundle = first_adapter.bundle_from_content_map({"CLAUDE.md": "first"})
    second_bundle = second_adapter.bundle_from_content_map({"CLAUDE.md": "second"})
    first_entered = threading.Event()
    release_first = threading.Event()
    second_prepared = threading.Event()
    real_apply = first_adapter._apply
    real_prepare = second_adapter._prepare

    def blocked_apply(*args):
        first_entered.set()
        assert release_first.wait(timeout=2)
        return real_apply(*args)

    def observe_prepare(*args):
        second_prepared.set()
        return real_prepare(*args)

    monkeypatch.setattr(first_adapter, "_apply", blocked_apply)
    monkeypatch.setattr(second_adapter, "_prepare", observe_prepare)
    outcomes: list[PromotionReceipt | Exception] = []

    def run(adapter, bundle, promotion_id):
        try:
            outcomes.append(_promote(adapter, past, bundle, bundle, promotion_id=promotion_id))
        except Exception as exc:  # pragma: no branch - asserted below
            outcomes.append(exc)

    first = threading.Thread(target=run, args=(first_adapter, first_bundle, "first"))
    second = threading.Thread(target=run, args=(second_adapter, second_bundle, "second"))
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    second_was_blocked = not second_prepared.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert second_was_blocked
    assert sum(isinstance(item, PromotionReceipt) for item in outcomes) == 1
    assert sum(isinstance(item, StaleBaseError) for item in outcomes) == 1


def test_update_preserves_file_mode_and_transaction_uses_project_filesystem(tmp_path: Path):
    _write(tmp_path, "CLAUDE.md", "old")
    (tmp_path / "CLAUDE.md").chmod(0o600)
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map({"CLAUDE.md": "new"})

    _promote(adapter, past, promoted, promoted)

    assert (tmp_path / "CLAUDE.md").stat().st_mode & 0o777 == 0o600
    assert os.stat(adapter.transaction_path("promotion-1")).st_dev == os.stat(tmp_path).st_dev


def test_recovery_rejects_tampered_created_directory_escape(tmp_path: Path, monkeypatch):
    _write(tmp_path, "CLAUDE.md", "old")
    adapter = ProjectFileAdapter(tmp_path)
    past = adapter.capture()
    promoted = adapter.bundle_from_content_map(
        {"CLAUDE.md": "new", ".claude/skills/new.md": "skill"}
    )
    monkeypatch.setattr(
        adapter,
        "_apply",
        lambda *_args: (_ for _ in ()).throw(SystemExit("crash")),
    )
    with pytest.raises(SystemExit):
        _promote(adapter, past, promoted, promoted)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-dir"
    outside.mkdir()
    journal = adapter.transaction_path("promotion-1") / "journal.json"
    record = json.loads(journal.read_text(encoding="utf-8"))
    outside_identity = outside.stat()
    record["created_directories"] = [
        {
            "locator": "../" + outside.name,
            "device": outside_identity.st_dev,
            "inode": outside_identity.st_ino,
        }
    ]
    journal.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(PromotionTransactionError):
        adapter.recover("promotion-1")

    assert outside.is_dir()
    outside.rmdir()
