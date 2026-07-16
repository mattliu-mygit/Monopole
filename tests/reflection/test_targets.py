from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weave_agent_signals.runs.targets import load_target_registry


def _registry(tmp_path: Path, targets: list[dict] | None = None) -> Path:
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "targets": targets
                or [
                    {"kind": "file", "id": "global-agents", "path": "AGENTS.md"},
                    {"kind": "file", "id": "soul", "path": "SOUL.md"},
                    {"kind": "skill_collection", "id": "repo-skills", "root": "skills"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_registry_captures_exact_files_and_one_level_skills(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")
    (tmp_path / "skills" / "alpha").mkdir(parents=True)
    (tmp_path / "skills" / "alpha" / "SKILL.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "skills" / "alpha" / "extra.md").write_text("ignored", encoding="utf-8")
    (tmp_path / "skills" / "nested" / "child").mkdir(parents=True)
    (tmp_path / "skills" / "nested" / "child" / "SKILL.md").write_text(
        "ignored",
        encoding="utf-8",
    )

    registry = load_target_registry(_registry(tmp_path))
    captured = registry.capture()

    assert captured.locators == (
        "file:global-agents",
        "file:soul",
        "skills:repo-skills/alpha/SKILL.md",
    )
    assert captured.target("file:soul").exists is False
    assert captured.target("skills:repo-skills/alpha/SKILL.md").content == "alpha"
    assert registry.contract_manifest()["schema_version"] == "1"
    assert registry.contract_manifest()["digest"].startswith("sha256:")


def test_registry_resolves_admitted_create_without_exposing_absolute_paths(tmp_path: Path) -> None:
    registry = load_target_registry(_registry(tmp_path))

    destination = registry.resolve_locator(
        "skills:repo-skills/new-skill/SKILL.md",
        require_absent_for_create=True,
    )

    assert destination == tmp_path / "skills" / "new-skill" / "SKILL.md"
    manifest = json.dumps(registry.contract_manifest())
    assert str(tmp_path) not in manifest


@pytest.mark.parametrize(
    "target",
    [
        {"kind": "file", "id": "bad", "path": "notes.txt"},
        {"kind": "file", "id": "bad", "path": "../outside.md"},
        {"kind": "skill_collection", "id": "bad", "root": "../skills"},
    ],
)
def test_registry_rejects_invalid_target_paths(tmp_path: Path, target: dict) -> None:
    with pytest.raises((ValueError, ValidationError)):
        load_target_registry(_registry(tmp_path, [target]))


def test_registry_rejects_extra_keys_duplicates_overlap_and_symlinks(tmp_path: Path) -> None:
    path = _registry(tmp_path)
    path.write_text('{"schema_version":"1","targets":[],"extra":true}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_target_registry(path)

    with pytest.raises(ValueError, match="duplicate"):
        load_target_registry(
            _registry(
                tmp_path,
                [
                    {"kind": "file", "id": "same", "path": "A.md"},
                    {"kind": "file", "id": "same", "path": "B.md"},
                ],
            )
        )

    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "linked").symlink_to(tmp_path, target_is_directory=True)
    registry = load_target_registry(_registry(tmp_path))
    with pytest.raises(ValueError, match="symlink"):
        registry.capture()


def test_registry_rejects_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_bytes(b"\xff")
    registry = load_target_registry(_registry(tmp_path))
    with pytest.raises(ValueError, match="UTF-8"):
        registry.capture()


def test_registry_rejects_overlap_in_either_order_and_invalid_skill_names(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="overlapping"):
        load_target_registry(
            _registry(
                tmp_path,
                [
                    {"kind": "file", "id": "nested", "path": "skills/a/SKILL.md"},
                    {"kind": "skill_collection", "id": "skills", "root": "skills"},
                ],
            )
        )

    (tmp_path / "skills" / "bad name").mkdir(parents=True)
    (tmp_path / "skills" / "bad name" / "SKILL.md").write_text("bad", encoding="utf-8")
    registry = load_target_registry(_registry(tmp_path))
    with pytest.raises(ValueError, match="skill directory name"):
        registry.capture()


def test_markdown_root_captures_only_listed_files_and_resolves_recursive_create(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    (root / "docs" / "nested").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("guide", encoding="utf-8")
    (root / "docs" / "nested" / "notes.md").write_text("notes", encoding="utf-8")
    (root / "docs" / "ignored.txt").write_text("ignored", encoding="utf-8")
    registry = load_target_registry(
        _registry(
            tmp_path,
            [
                {
                    "kind": "markdown_root",
                    "id": "repo",
                    "root": str(root),
                    "files": ["docs/guide.md"],
                    "allow_create": True,
                }
            ],
        )
    )

    captured = registry.capture()

    assert captured.locators == ("markdown:repo/docs/guide.md",)
    with pytest.raises(KeyError):
        registry.resolve_locator("markdown:repo/docs/nested/notes.md")
    destination = registry.resolve_locator(
        "markdown:repo/new/deep/AGENTS.md",
        require_absent_for_create=True,
    )
    assert destination == root / "new" / "deep" / "AGENTS.md"


@pytest.mark.parametrize(
    "locator",
    [
        "markdown:repo/../outside.md",
        "markdown:repo//absolute.md",
        "markdown:repo/docs/notes.txt",
        "markdown:repo/node_modules/package/README.md",
    ],
)
def test_markdown_root_rejects_escaped_non_markdown_and_excluded_locators(
    tmp_path: Path,
    locator: str,
) -> None:
    registry = load_target_registry(
        _registry(
            tmp_path,
            [
                {
                    "kind": "markdown_root",
                    "id": "repo",
                    "root": "repo",
                    "files": [],
                    "allow_create": True,
                }
            ],
        )
    )

    with pytest.raises((KeyError, ValueError)):
        registry.resolve_locator(locator)


def test_registry_normalizes_paths_before_rejecting_overlap(tmp_path: Path) -> None:
    root = tmp_path / "repo"

    with pytest.raises(ValueError, match="overlapping"):
        load_target_registry(
            _registry(
                tmp_path,
                [
                    {
                        "kind": "markdown_root",
                        "id": "repo",
                        "root": str(root),
                        "files": [],
                        "allow_create": True,
                    },
                    {
                        "kind": "file",
                        "id": "alias",
                        "path": str(root / "detour" / ".." / "AGENTS.md"),
                    },
                ],
            )
        )


def test_markdown_root_rejects_symlinked_descendants(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "SECRET.md").write_text("secret", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    registry = load_target_registry(
        _registry(
            tmp_path,
            [
                {
                    "kind": "markdown_root",
                    "id": "repo",
                    "root": str(root),
                    "files": [],
                    "allow_create": True,
                }
            ],
        )
    )

    with pytest.raises(ValueError, match="symlink"):
        registry.resolve_locator(
            "markdown:repo/linked/SECRET.md",
            require_absent_for_create=True,
        )


def test_markdown_root_rejects_create_when_disabled(tmp_path: Path) -> None:
    registry = load_target_registry(
        _registry(
            tmp_path,
            [
                {
                    "kind": "markdown_root",
                    "id": "repo",
                    "root": "repo",
                    "files": [],
                    "allow_create": False,
                }
            ],
        )
    )

    with pytest.raises(KeyError):
        registry.resolve_locator(
            "markdown:repo/new/AGENTS.md",
            require_absent_for_create=True,
        )
