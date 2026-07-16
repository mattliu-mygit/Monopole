"""Explicit instruction target registry and bounded filesystem capture."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, model_validator

from weave_agent_signals.runs.bundles import (
    BundleSnapshot,
    ScopeDescriptor,
    TargetSnapshot,
    revision_hash,
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SKILL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EXCLUDED_MARKDOWN_DIRECTORIES = frozenset(
    {
        ".cache",
        ".flox",
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "cache",
        "coverage",
        "dist",
        "node_modules",
        "vendor",
    }
)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileTarget(_ClosedModel):
    kind: Literal["file"] = "file"
    id: StrictStr
    path: StrictStr

    @model_validator(mode="after")
    def validate_target(self) -> FileTarget:
        _validate_id(self.id)
        _validate_configured_path(self.path, markdown=True)
        return self


class SkillCollection(_ClosedModel):
    kind: Literal["skill_collection"] = "skill_collection"
    id: StrictStr
    root: StrictStr

    @model_validator(mode="after")
    def validate_target(self) -> SkillCollection:
        _validate_id(self.id)
        _validate_configured_path(self.root, markdown=False)
        return self


class MarkdownRoot(_ClosedModel):
    kind: Literal["markdown_root"] = "markdown_root"
    id: StrictStr
    root: StrictStr
    files: tuple[StrictStr, ...]
    allow_create: StrictBool

    @model_validator(mode="after")
    def validate_target(self) -> MarkdownRoot:
        _validate_id(self.id)
        _validate_configured_path(self.root, markdown=False)
        files = tuple(_markdown_relative(value) for value in self.files)
        if len(files) != len(set(files)):
            raise ValueError("Markdown root files must be unique")
        object.__setattr__(self, "files", tuple(sorted(files)))
        return self


RegistryTarget = Annotated[
    FileTarget | SkillCollection | MarkdownRoot,
    Field(discriminator="kind"),
]


class _RegistryDocument(_ClosedModel):
    schema_version: Literal["1"]
    targets: tuple[RegistryTarget, ...]


def _validate_id(value: str) -> None:
    if not _ID_RE.fullmatch(value):
        raise ValueError("target IDs must be nonblank stable identifiers")


def _validate_configured_path(value: str, *, markdown: bool) -> None:
    if not value or "\x00" in value:
        raise ValueError("target path must be nonblank")
    path = Path(value).expanduser()
    if not path.is_absolute() and ".." in PurePosixPath(value).parts:
        raise ValueError("relative target paths cannot escape the registry directory")
    if markdown and path.suffix != ".md":
        raise ValueError("file targets must use the .md suffix")


def _markdown_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Markdown paths must be nonblank")
    raw_parts = value.split("/")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(part in EXCLUDED_MARKDOWN_DIRECTORIES for part in raw_parts[:-1])
        or pure.suffix != ".md"
    ):
        raise ValueError("Markdown paths must be safe relative .md paths")
    return pure.as_posix()


def _resolve(base: Path, value: str) -> Path:
    expanded = Path(value).expanduser()
    return expanded if expanded.is_absolute() else base / expanded


def _normalized(base: Path, value: str) -> Path:
    return _resolve(base, value).resolve(strict=False)


def _reject_symlinks(path: Path) -> None:
    current = path
    while True:
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise ValueError(f"target path contains a symlink: {path}")
        if current.parent == current:
            return
        current = current.parent


class TargetRegistry:
    """Resolved registry whose public identities never expose filesystem paths."""

    def __init__(self, document: _RegistryDocument, registry_path: Path) -> None:
        self.schema_version = document.schema_version
        self._registry_path = registry_path
        self._base = registry_path.parent
        self._entries = document.targets
        self._source_text = registry_path.read_text(encoding="utf-8")
        self._promotion_lock = threading.RLock()
        ids = [entry.id for entry in self._entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate target IDs")
        resolved: list[tuple[Path, bool]] = []
        for entry in self._entries:
            path = _normalized(
                self._base,
                entry.path if isinstance(entry, FileTarget) else entry.root,
            )
            container = not isinstance(entry, FileTarget)
            for other, other_container in resolved:
                overlaps = path == other
                if container and other_container:
                    overlaps = overlaps or path in other.parents or other in path.parents
                elif container:
                    overlaps = overlaps or path in other.parents
                elif other_container:
                    overlaps = overlaps or other in path.parents
                if overlaps:
                    raise ValueError("overlapping registry targets")
            resolved.append((path, container))

    def contract_manifest(self) -> dict[str, object]:
        public = [{"kind": entry.kind, "id": entry.id} for entry in self._entries]
        return {
            "schema_version": self.schema_version,
            "targets": public,
            "digest": revision_hash(
                {
                    "schema_version": self.schema_version,
                    "targets": [entry.model_dump(mode="json") for entry in self._entries],
                }
            ),
        }

    def promotion_guard(self) -> threading.RLock:
        """Serialize drift checks and publication for this loaded registry."""

        return self._promotion_lock

    def capture(self) -> BundleSnapshot:
        targets: list[TargetSnapshot] = []
        for entry in self._entries:
            if isinstance(entry, FileTarget):
                locator = f"file:{entry.id}"
                targets.append(self._capture_path(locator, _resolve(self._base, entry.path)))
                continue
            root = _resolve(self._base, entry.root)
            _reject_symlinks(root)
            if isinstance(entry, MarkdownRoot):
                if root.exists() and not root.is_dir():
                    raise ValueError(f"Markdown root is not a directory: {entry.id}")
                for relative in entry.files:
                    targets.append(
                        self._capture_path(
                            f"markdown:{entry.id}/{relative}",
                            root.joinpath(*PurePosixPath(relative).parts),
                        )
                    )
                continue
            if not root.exists():
                continue
            if not root.is_dir():
                raise ValueError(f"skill collection is not a directory: {entry.id}")
            for child in sorted(root.iterdir(), key=lambda item: item.name):
                _reject_symlinks(child)
                if not child.is_dir():
                    continue
                skill = child / "SKILL.md"
                if skill.exists() or skill.is_symlink():
                    if not _SKILL_RE.fullmatch(child.name):
                        raise ValueError(f"invalid skill directory name in collection {entry.id}")
                    locator = f"skills:{entry.id}/{child.name}/SKILL.md"
                    targets.append(self._capture_path(locator, skill))
        return BundleSnapshot(
            tuple(targets),
            ScopeDescriptor(kind="registry", target_id=self.contract_manifest()["digest"]),
        )

    def _capture_path(self, locator: str, path: Path) -> TargetSnapshot:
        _reject_symlinks(path)
        if not path.exists():
            return TargetSnapshot(kind="file", locator=locator, exists=False, content=None)
        if not path.is_file():
            raise ValueError(f"target is not a file: {locator}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"target is not valid UTF-8: {locator}") from exc
        return TargetSnapshot(kind="file", locator=locator, exists=True, content=content)

    def resolve_locator(self, locator: str, *, require_absent_for_create: bool = False) -> Path:
        for entry in self._entries:
            if isinstance(entry, FileTarget) and locator == f"file:{entry.id}":
                path = _resolve(self._base, entry.path)
                break
            if isinstance(entry, SkillCollection) and locator.startswith(f"skills:{entry.id}/"):
                relative = locator.removeprefix(f"skills:{entry.id}/")
                parts = PurePosixPath(relative).parts
                if len(parts) != 2 or parts[1] != "SKILL.md" or not _SKILL_RE.fullmatch(parts[0]):
                    raise ValueError("skill locators must be <skill>/SKILL.md")
                path = _resolve(self._base, entry.root) / parts[0] / "SKILL.md"
                break
            if isinstance(entry, MarkdownRoot) and locator.startswith(f"markdown:{entry.id}/"):
                relative = locator.removeprefix(f"markdown:{entry.id}/")
                relative = _markdown_relative(relative)
                if require_absent_for_create:
                    if not entry.allow_create:
                        raise KeyError(locator)
                elif relative not in entry.files:
                    raise KeyError(locator)
                path = _resolve(self._base, entry.root).joinpath(*PurePosixPath(relative).parts)
                break
        else:
            raise KeyError(locator)
        _reject_symlinks(path)
        if require_absent_for_create and path.exists():
            raise ValueError(f"create target already exists: {locator}")
        return path

    def capture_locator(self, locator: str) -> TargetSnapshot:
        """Capture one admitted locator for an immediate source comparison."""

        return self._capture_path(locator, self.resolve_locator(locator))

    def register_creates(self, locators: tuple[str, ...]) -> tuple[str, ...]:
        """Atomically add admitted create paths before target publication begins."""

        if not locators:
            return ()
        additions: dict[str, set[str]] = {}
        for locator in locators:
            for entry in self._entries:
                prefix = f"markdown:{entry.id}/"
                if isinstance(entry, MarkdownRoot) and locator.startswith(prefix):
                    if not entry.allow_create:
                        raise KeyError(locator)
                    relative = _markdown_relative(locator.removeprefix(prefix))
                    additions.setdefault(entry.id, set()).add(relative)
                    break
            else:
                continue
        if not additions:
            return ()
        updated: list[RegistryTarget] = []
        registered: list[str] = []
        for entry in self._entries:
            if not isinstance(entry, MarkdownRoot) or entry.id not in additions:
                updated.append(entry)
                continue
            registered.extend(
                f"markdown:{entry.id}/{relative}"
                for relative in sorted(additions[entry.id] - set(entry.files))
            )
            files = tuple(sorted(set(entry.files) | additions[entry.id]))
            updated.append(entry.model_copy(update={"files": files}))
        if not registered:
            return ()
        self._replace_entries(tuple(updated))
        return tuple(registered)

    def unregister_absent_creates(self, locators: tuple[str, ...]) -> None:
        """Remove registrations added by an unsuccessful create attempt."""

        removals: dict[str, set[str]] = {}
        for locator in locators:
            target_id, relative = locator.removeprefix("markdown:").split("/", 1)
            path = self.resolve_locator(locator)
            if path.exists():
                raise ValueError(f"cannot unregister an existing target: {locator}")
            removals.setdefault(target_id, set()).add(relative)
        updated: list[RegistryTarget] = []
        for entry in self._entries:
            if not isinstance(entry, MarkdownRoot) or entry.id not in removals:
                updated.append(entry)
                continue
            updated.append(
                entry.model_copy(
                    update={
                        "files": tuple(
                            item for item in entry.files if item not in removals[entry.id]
                        )
                    }
                )
            )
        self._replace_entries(tuple(updated))

    def _replace_entries(self, entries: tuple[RegistryTarget, ...]) -> None:
        current = self._registry_path.read_text(encoding="utf-8")
        if current != self._source_text:
            raise ValueError("target registry changed after it was loaded")
        document = _RegistryDocument(schema_version=self.schema_version, targets=entries)
        content = json.dumps(document.model_dump(mode="json"), indent=2) + "\n"
        descriptor, name = tempfile.mkstemp(
            prefix=".monopole-targets-",
            dir=self._registry_path.parent,
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._registry_path)
        finally:
            temporary.unlink(missing_ok=True)
        self._entries = entries
        self._source_text = content


def load_target_registry(path: str | Path) -> TargetRegistry:
    registry_path = Path(path).expanduser().resolve(strict=True)
    document = _RegistryDocument.model_validate_json(registry_path.read_text(encoding="utf-8"))
    return TargetRegistry(document, registry_path)
