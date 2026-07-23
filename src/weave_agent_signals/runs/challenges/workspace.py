"""Deterministic, mount-free workspace capture and A/B arm materialization."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import tarfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path, PurePosixPath

from weave_agent_signals.runs.bundles import BundleSnapshot, compare_bundles
from weave_agent_signals.runs.challenges.contracts import ArtifactChange, canonical_digest
from weave_agent_signals.runs.challenges.environment import RuntimeFile
from weave_agent_signals.runs.targets import EXCLUDED_MARKDOWN_DIRECTORIES

DEFAULT_WORKSPACE_EXCLUSIONS = EXCLUDED_MARKDOWN_DIRECTORIES | frozenset({".git"})
MAX_WORKSPACE_FILES = 10_000
MAX_WORKSPACE_FILE_BYTES = 16 * 1024 * 1024
MAX_WORKSPACE_TOTAL_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_DIFF_CHARACTERS = 200_000
MAX_ARTIFACT_FILE_DIFF_CHARACTERS = 20_000


def _content_digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _relative_path(value: str) -> str:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("workspace file path must be a safe relative path")
    return pure.as_posix()


@dataclass(frozen=True)
class WorkspaceFile:
    path: str
    content: bytes
    mode: int = 0o644
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        if not isinstance(self.content, bytes):
            raise ValueError("workspace file content must be bytes")
        if len(self.content) > MAX_WORKSPACE_FILE_BYTES:
            raise ValueError("workspace file exceeds the maximum captured size")
        if type(self.mode) is not int or self.mode < 0 or self.mode > 0o777:
            raise ValueError("workspace file mode must be a Unix permission mode")
        object.__setattr__(self, "digest", _content_digest(self.content))

    def manifest_entry(self) -> dict[str, object]:
        return {"path": self.path, "digest": self.digest, "mode": self.mode}


@dataclass(frozen=True)
class WorkspaceSnapshot:
    files: tuple[WorkspaceFile, ...]
    digest: str = field(init=False)
    archive: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        files = tuple(sorted(self.files, key=lambda item: item.path))
        paths = tuple(item.path for item in files)
        if len(paths) != len(set(paths)):
            raise ValueError("workspace snapshot contains duplicate paths")
        if len(files) > MAX_WORKSPACE_FILES:
            raise ValueError("workspace snapshot exceeds the maximum file count")
        if sum(len(item.content) for item in files) > MAX_WORKSPACE_TOTAL_BYTES:
            raise ValueError("workspace snapshot exceeds the maximum total size")
        object.__setattr__(self, "files", files)
        object.__setattr__(
            self,
            "digest",
            canonical_digest([item.manifest_entry() for item in files]),
        )
        object.__setattr__(self, "archive", _archive(files))

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files)

    @property
    def manifest(self) -> tuple[dict[str, object], ...]:
        return tuple(item.manifest_entry() for item in self.files)

    def file(self, path: str) -> WorkspaceFile:
        normalized = _relative_path(path)
        for item in self.files:
            if item.path == normalized:
                return item
        raise KeyError(path)

    def read_text(self, path: str) -> str:
        return self.file(path).content.decode("utf-8")


@dataclass(frozen=True)
class ArmSnapshots:
    baseline: WorkspaceSnapshot
    candidate: WorkspaceSnapshot
    authoring: WorkspaceSnapshot
    baseline_runtime_files: tuple[RuntimeFile, ...] = ()
    candidate_runtime_files: tuple[RuntimeFile, ...] = ()

    def __post_init__(self) -> None:
        for name in ("baseline_runtime_files", "candidate_runtime_files"):
            files = tuple(sorted(getattr(self, name), key=lambda item: item.path))
            paths = tuple(item.path for item in files)
            if len(paths) != len(set(paths)):
                raise ValueError("arm runtime instruction paths must be unique")
            object.__setattr__(self, name, files)


def _archive(files: tuple[WorkspaceFile, ...]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for item in files:
            info = tarfile.TarInfo(item.path)
            info.size = len(item.content)
            info.mode = item.mode
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(item.content))
    return output.getvalue()


def capture_workspace(
    root: str | Path,
    *,
    excluded_names: Iterable[str] = DEFAULT_WORKSPACE_EXCLUSIONS,
) -> WorkspaceSnapshot:
    """Capture one workspace without following symlinks or retaining host paths."""

    source = Path(root).expanduser()
    if source.is_symlink():
        raise ValueError("sandbox workspace root cannot be a symlink")
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("sandbox workspace root must be a directory")
    excluded = frozenset(excluded_names)
    if any(not isinstance(name, str) or not name for name in excluded):
        raise ValueError("workspace exclusions must be nonblank names")

    captured: list[WorkspaceFile] = []
    for directory, dirnames, filenames in os.walk(source, topdown=True, followlinks=False):
        current = Path(directory)
        kept_dirs: list[str] = []
        for name in sorted(dirnames):
            child = current / name
            if name in excluded:
                continue
            if child.is_symlink():
                raise ValueError(f"sandbox workspace contains a symlink: {child}")
            kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            if name in excluded:
                continue
            path = current / name
            if path.is_symlink():
                raise ValueError(f"sandbox workspace contains a symlink: {path}")
            if not path.is_file():
                raise ValueError(f"sandbox workspace contains a non-file entry: {path}")
            if path.stat().st_size > MAX_WORKSPACE_FILE_BYTES:
                raise ValueError(f"sandbox workspace file exceeds the maximum size: {path}")
            relative = path.relative_to(source).as_posix()
            mode = stat.S_IMODE(path.stat().st_mode)
            captured.append(WorkspaceFile(relative, path.read_bytes(), mode))
    return WorkspaceSnapshot(tuple(captured))


def _artifact_diff(path: str, before: WorkspaceFile | None, after: WorkspaceFile | None) -> str:
    try:
        before_text = "" if before is None else before.content.decode("utf-8")
        after_text = "" if after is None else after.content.decode("utf-8")
    except UnicodeDecodeError:
        return "Binary artifact changed; semantic text diff is unavailable."
    value = "".join(
        unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    if not value:
        return "Artifact metadata changed without a textual content change."
    if len(value) <= MAX_ARTIFACT_FILE_DIFF_CHARACTERS:
        return value
    return value[:MAX_ARTIFACT_FILE_DIFF_CHARACTERS] + "\n[diff truncated]\n"


def describe_workspace_changes(
    before: WorkspaceSnapshot,
    after: WorkspaceSnapshot,
) -> tuple[ArtifactChange, ...]:
    """Create bounded, persisted semantic evidence for final workspace changes."""

    before_files = {item.path: item for item in before.files}
    after_files = {item.path: item for item in after.files}
    changes: list[ArtifactChange] = []
    remaining = MAX_ARTIFACT_DIFF_CHARACTERS
    for path in sorted(set(before_files) | set(after_files)):
        previous = before_files.get(path)
        current = after_files.get(path)
        if previous == current:
            continue
        if previous is None:
            action = "added"
        elif current is None:
            action = "deleted"
        else:
            action = "modified"
        diff = _artifact_diff(path, previous, current)
        if remaining <= 0:
            diff = "Artifact changed; aggregate diff budget exhausted."
        elif len(diff) > remaining:
            diff = diff[:remaining] + "\n[aggregate diff truncated]\n"
        remaining -= min(len(diff), remaining)
        changes.append(
            ArtifactChange(
                path=path,
                action=action,
                before_digest=None if previous is None else previous.digest,
                after_digest=None if current is None else current.digest,
                diff=diff,
            )
        )
    return tuple(changes)


def _target_location(
    locator: str,
    *,
    workspace_root: Path,
    resolve_locator: Callable[[str], str | Path],
    host_home: Path,
    guest_home: PurePosixPath,
) -> tuple[str, str, Path]:
    source = Path(resolve_locator(locator)).expanduser()
    if source.is_symlink():
        raise ValueError(f"managed target cannot be a symlink: {locator}")
    resolved = source.resolve(strict=False)
    try:
        relative = resolved.relative_to(workspace_root)
        return "workspace", _relative_path(relative.as_posix()), resolved
    except ValueError:
        pass
    try:
        relative = resolved.relative_to(host_home)
    except ValueError as exc:
        raise ValueError(
            f"managed target is outside the sandbox workspace and host home: {locator}"
        ) from exc
    guest_path = guest_home.joinpath(*relative.parts)
    RuntimeFile(guest_path.as_posix(), b"")
    return "runtime", guest_path.as_posix(), resolved


def _require_baseline_matches(
    snapshot: WorkspaceSnapshot,
    baseline: BundleSnapshot,
    *,
    workspace_root: Path,
    resolve_locator: Callable[[str], str | Path],
    host_home: Path,
    guest_home: PurePosixPath,
) -> dict[str, tuple[str, str, Path]]:
    files = {item.path: item for item in snapshot.files}
    locations: dict[str, tuple[str, str, Path]] = {}
    guest_paths: set[str] = set()
    for target in baseline.targets:
        location = _target_location(
            target.locator,
            workspace_root=workspace_root,
            resolve_locator=resolve_locator,
            host_home=host_home,
            guest_home=guest_home,
        )
        kind, path, source = location
        guest_path = f"/workspace/{path}" if kind == "workspace" else path
        if guest_path in guest_paths:
            raise ValueError("managed target locators resolve to the same workspace path")
        guest_paths.add(guest_path)
        locations[target.locator] = location
        if kind == "workspace":
            captured = files.get(path)
            matches = (
                captured is not None
                and target.exists
                and captured.content == (target.content or "").encode("utf-8")
            ) or (captured is None and not target.exists)
        else:
            if source.exists() and (source.is_symlink() or not source.is_file()):
                raise ValueError(f"managed target is not a regular file: {target.locator}")
            matches = (
                source.is_file()
                and target.exists
                and source.read_bytes() == (target.content or "").encode("utf-8")
            ) or (not source.exists() and not target.exists)
        if not matches:
            raise ValueError(f"workspace does not match baseline A at {target.locator}")
    return locations


def _runtime_instruction_files(
    bundle: BundleSnapshot,
    locations: dict[str, tuple[str, str, Path]],
) -> tuple[RuntimeFile, ...]:
    files: list[RuntimeFile] = []
    for target in bundle.targets:
        if not target.exists:
            continue
        kind, path, source = locations[target.locator]
        if kind != "runtime":
            continue
        mode = stat.S_IMODE(source.stat().st_mode) if source.is_file() else 0o644
        files.append(RuntimeFile(path, (target.content or "").encode("utf-8"), mode))
    return tuple(sorted(files, key=lambda item: item.path))


def materialize_arms(
    snapshot: WorkspaceSnapshot,
    *,
    baseline: BundleSnapshot,
    candidate: BundleSnapshot,
    workspace_root: str | Path,
    resolve_locator: Callable[[str], str | Path],
    host_home: str | Path | None = None,
    guest_home: str = "/root",
    verify_baseline: bool = True,
) -> ArmSnapshots:
    """Build private A/B workspace archives and authenticate their sole differences."""

    if not isinstance(snapshot, WorkspaceSnapshot):
        raise ValueError("snapshot must be a WorkspaceSnapshot")
    if not isinstance(baseline, BundleSnapshot) or not isinstance(candidate, BundleSnapshot):
        raise ValueError("baseline and candidate must be BundleSnapshot values")
    root = Path(workspace_root).expanduser().resolve(strict=True)
    home = (
        Path.home().resolve(strict=True)
        if host_home is None
        else Path(host_home).resolve(strict=True)
    )
    guest = PurePosixPath(guest_home)
    if not guest.is_absolute() or guest == PurePosixPath("/workspace"):
        raise ValueError("guest home must be an absolute path outside /workspace")
    if verify_baseline:
        locations = _require_baseline_matches(
            snapshot,
            baseline,
            workspace_root=root,
            resolve_locator=resolve_locator,
            host_home=home,
            guest_home=guest,
        )
    else:
        locations = {
            target.locator: _target_location(
                target.locator,
                workspace_root=root,
                resolve_locator=resolve_locator,
                host_home=home,
                guest_home=guest,
            )
            for target in baseline.targets
        }
    baseline_files = {item.path: item for item in snapshot.files}
    if not verify_baseline:
        for target in baseline.targets:
            kind, path, _source = locations[target.locator]
            if kind != "workspace":
                continue
            current = baseline_files.get(path)
            if target.exists:
                baseline_files[path] = WorkspaceFile(
                    path,
                    (target.content or "").encode("utf-8"),
                    current.mode if current is not None else 0o644,
                )
            else:
                baseline_files.pop(path, None)
    baseline_snapshot = WorkspaceSnapshot(tuple(baseline_files.values()))
    preview = compare_bundles(baseline, candidate)
    candidate_files = dict(baseline_files)
    expected_paths: list[str] = []
    for action in preview.actions:
        location = locations.get(action.locator)
        if location is None:
            location = _target_location(
                action.locator,
                workspace_root=root,
                resolve_locator=resolve_locator,
                host_home=home,
                guest_home=guest,
            )
            locations[action.locator] = location
        kind, path, _source = location
        expected_paths.append(path)
        if kind == "workspace":
            current = candidate_files.get(path)
            if action.after.exists:
                mode = current.mode if current is not None else 0o644
                candidate_files[path] = WorkspaceFile(
                    path,
                    (action.after.content or "").encode("utf-8"),
                    mode,
                )
            else:
                candidate_files.pop(path, None)

    for target in candidate.targets:
        if target.locator not in locations:
            locations[target.locator] = _target_location(
                target.locator,
                workspace_root=root,
                resolve_locator=resolve_locator,
                host_home=home,
                guest_home=guest,
            )

    candidate_snapshot = WorkspaceSnapshot(tuple(candidate_files.values()))
    baseline_runtime_files = _runtime_instruction_files(baseline, locations)
    candidate_runtime_files = _runtime_instruction_files(candidate, locations)
    workspace_changes = tuple(
        sorted(
            path
            for path in set(baseline_files) | set(candidate_files)
            if baseline_files.get(path) != candidate_files.get(path)
        )
    )
    baseline_runtime = {item.path: item for item in baseline_runtime_files}
    candidate_runtime = {item.path: item for item in candidate_runtime_files}
    runtime_changes = tuple(
        sorted(
            path
            for path in set(baseline_runtime) | set(candidate_runtime)
            if baseline_runtime.get(path) != candidate_runtime.get(path)
        )
    )
    actual_paths = tuple(sorted((*workspace_changes, *runtime_changes)))
    if actual_paths != tuple(sorted(expected_paths)):
        raise ValueError("A/B workspace delta does not match declared candidate actions")
    managed_paths = frozenset(
        path for kind, path, _source in locations.values() if kind == "workspace"
    )
    authoring_snapshot = WorkspaceSnapshot(
        tuple(item for item in snapshot.files if item.path not in managed_paths)
    )
    return ArmSnapshots(
        baseline_snapshot,
        candidate_snapshot,
        authoring_snapshot,
        baseline_runtime_files,
        candidate_runtime_files,
    )
