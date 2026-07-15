"""Journaled, locked promotion of canonical instruction bundles."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
import threading
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from weave_agent_signals.runs import bundles

TARGET_ADAPTER_CONTRACT_VERSION = "2.0.0"
DEFAULT_EXCLUDED_MARKDOWN_DIRECTORIES = (
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


@dataclass(frozen=True)
class ProjectMarkdownPolicy:
    """Immutable discovery and validation policy for project Markdown."""

    suffix: str = ".md"
    excluded_directories: tuple[str, ...] = DEFAULT_EXCLUDED_MARKDOWN_DIRECTORIES
    max_files: int = 500
    max_file_bytes: int = 256 * 1024
    max_total_bytes: int = 512 * 1024

    def __post_init__(self) -> None:
        if self.suffix != ".md":
            raise PromotionValidationError("project Markdown suffix must be .md")
        if not isinstance(self.excluded_directories, tuple):
            raise PromotionValidationError("excluded directories must be an immutable tuple")
        if len(set(self.excluded_directories)) != len(self.excluded_directories) or any(
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or "\x00" in name
            for name in self.excluded_directories
        ):
            raise PromotionValidationError("excluded directory names are invalid")
        for name in ("max_files", "max_file_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise PromotionValidationError(f"{name} must be a positive integer")

    def to_manifest(self) -> dict[str, object]:
        return {
            "suffix": self.suffix,
            "excluded_directories": list(self.excluded_directories),
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "max_total_bytes": self.max_total_bytes,
            "traversal": "filesystem_recursive_sorted",
            "path_format": "relative_posix",
            "symlink_policy": "reject_nonexcluded",
            "cross_filesystem_policy": "reject",
            "content_encoding": "utf-8",
        }


def target_adapter_contract_manifest(
    policy: ProjectMarkdownPolicy | None = None,
) -> dict[str, object]:
    """Return deterministic managed-scope and project-file adapter semantics."""

    if policy is None:
        policy = ProjectMarkdownPolicy()
    if not isinstance(policy, ProjectMarkdownPolicy):
        raise PromotionValidationError("invalid project Markdown policy")
    return {
        "version": TARGET_ADAPTER_CONTRACT_VERSION,
        "adapter_kind": "file",
        "markdown_policy": policy.to_manifest(),
        "scope_capture": "complete_manifest",
        "promotion_journal_directory": ".weave-agent-signals/promotions",
    }


class PromotionError(Exception):
    """Base promotion error carrying the affected target locators."""

    def __init__(self, message: str, *, changed_locators: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.changed_locators = tuple(sorted(set(changed_locators)))


class PromotionValidationError(PromotionError):
    """A snapshot, target, or action is invalid."""


class StaleBaseError(PromotionError):
    """The complete current scope no longer equals evaluated baseline B."""

    def __init__(
        self,
        expected: bundles.BundleSnapshot,
        current: bundles.BundleSnapshot | None,
        changed_locators: Iterable[str] = (),
    ) -> None:
        super().__init__(
            "the complete managed scope changed after baseline evaluation",
            changed_locators=(
                bundles.changed_locators(expected, current)
                if current is not None
                else changed_locators
            ),
        )
        self.expected = expected
        self.current = current


class PromotionTransactionError(PromotionError):
    """A prepared transaction failed and was rolled back or needs recovery."""

    def __init__(
        self,
        message: str,
        *,
        changed_locators: Iterable[str],
        cause: Exception | None = None,
        rollback_error: Exception | None = None,
    ) -> None:
        super().__init__(message, changed_locators=changed_locators)
        self.cause = cause
        self.rollback_error = rollback_error


@dataclass(frozen=True)
class _CreatedDirectory:
    locator: str
    device: int
    inode: int

    def to_dict(self) -> dict[str, object]:
        return {
            "locator": self.locator,
            "device": self.device,
            "inode": self.inode,
        }


def _string(value: Any, name: str, locator: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise PromotionValidationError(
            f"{name} must be a non-empty string",
            changed_locators=(locator,) if locator else (),
        )
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PromotionValidationError("Git metadata keys must be strings")
        return MappingProxyType({key: _freeze(value[key]) for key in sorted(value)})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise PromotionValidationError("Git metadata must contain only JSON values")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class PromotionReceipt:
    """Audit receipt distinguishing Past B, evaluated C, and promoted C-or-D."""

    promotion_id: str
    run_id: str
    candidate_id: str
    target_kind: str
    target_id: str
    past: bundles.BundleSnapshot
    evaluated_candidate: bundles.BundleSnapshot
    promoted: bundles.BundleSnapshot
    review_revision: int
    actions: tuple[bundles.PromotionAction, ...]
    decided_at: str
    promoted_was_evaluated: bool
    unevaluated_d_acknowledged: bool
    git_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in (
            "promotion_id",
            "run_id",
            "candidate_id",
            "target_kind",
            "target_id",
            "decided_at",
        ):
            _string(getattr(self, name), name)
        actions = tuple(self.actions)
        if type(self.review_revision) is not int or self.review_revision < 0:
            raise PromotionValidationError("receipt review revision is invalid")
        try:
            expected_actions = bundles.compare_bundles(
                self.past,
                self.promoted,
            ).actions
            if self.evaluated_candidate != self.promoted:
                bundles.validate_edited_bundle(
                    self.past,
                    self.evaluated_candidate,
                    self.promoted,
                )
        except bundles.BundleValidationError as exc:
            raise PromotionValidationError(
                str(exc),
                changed_locators=exc.changed_locators,
            ) from exc
        if actions != expected_actions:
            raise PromotionValidationError("receipt diff does not match its snapshots")
        expected_evaluated = self.evaluated_candidate == self.promoted
        if type(self.promoted_was_evaluated) is not bool or (
            self.promoted_was_evaluated != expected_evaluated
        ):
            raise PromotionValidationError("receipt evaluation marker is inconsistent")
        if type(self.unevaluated_d_acknowledged) is not bool or self.unevaluated_d_acknowledged != (
            not expected_evaluated
        ):
            raise PromotionValidationError("receipt unevaluated-d acknowledgement is inconsistent")
        object.__setattr__(self, "actions", actions)
        if self.git_metadata is not None:
            object.__setattr__(self, "git_metadata", _freeze(self.git_metadata))

    def _locators(self, action: bundles.ActionKind) -> tuple[str, ...]:
        return tuple(item.locator for item in self.actions if item.action == action)

    @property
    def created_locators(self) -> tuple[str, ...]:
        return self._locators("create")

    @property
    def updated_locators(self) -> tuple[str, ...]:
        return self._locators("update")

    @property
    def deleted_locators(self) -> tuple[str, ...]:
        return self._locators("delete")

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotion_id": self.promotion_id,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "past": self.past.to_dict(),
            "evaluated_candidate": self.evaluated_candidate.to_dict(),
            "promoted": self.promoted.to_dict(),
            "review_revision": self.review_revision,
            "promoted_was_evaluated": self.promoted_was_evaluated,
            "unevaluated_d_acknowledged": self.unevaluated_d_acknowledged,
            "actions": [action.to_dict() for action in self.actions],
            "decided_at": self.decided_at,
            "git_metadata": _thaw(self.git_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromotionReceipt:
        expected_keys = {
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
        try:
            if set(value) != expected_keys or not isinstance(value["actions"], list):
                raise TypeError
            result = cls(
                promotion_id=value["promotion_id"],
                run_id=value["run_id"],
                candidate_id=value["candidate_id"],
                target_kind=value["target_kind"],
                target_id=value["target_id"],
                past=bundles.BundleSnapshot.from_dict(value["past"]),
                evaluated_candidate=bundles.BundleSnapshot.from_dict(value["evaluated_candidate"]),
                promoted=bundles.BundleSnapshot.from_dict(value["promoted"]),
                review_revision=value["review_revision"],
                actions=tuple(bundles.PromotionAction.from_dict(item) for item in value["actions"]),
                decided_at=value["decided_at"],
                promoted_was_evaluated=value["promoted_was_evaluated"],
                unevaluated_d_acknowledged=value["unevaluated_d_acknowledged"],
                git_metadata=value["git_metadata"],
            )
        except (KeyError, TypeError, bundles.BundleValidationError) as exc:
            raise PromotionValidationError("invalid promotion receipt") from exc
        return result


class ProjectFileAdapter:
    """Capture and transactionally promote the reflector's managed file scope."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        target_id: str | None = None,
        policy: ProjectMarkdownPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        try:
            self.root = Path(project_root).expanduser().resolve(strict=True)
        except OSError as exc:
            raise PromotionValidationError("project root is not readable") from exc
        if not self.root.is_dir():
            raise PromotionValidationError("project root is not a directory")
        root_stat = self.root.lstat()
        self._root_identity = (root_stat.st_dev, root_stat.st_ino)
        self.target_id = _string(target_id or str(self.root), "target_id")
        if policy is None:
            policy = ProjectMarkdownPolicy()
        if not isinstance(policy, ProjectMarkdownPolicy):
            raise PromotionValidationError("invalid project Markdown policy")
        self.policy = policy
        self.scope = bundles.ScopeDescriptor(
            "file",
            self.target_id,
            (f"**/*{self.policy.suffix}",),
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.journal_root = self.root / ".weave-agent-signals" / "promotions"
        self._descriptor_state = threading.local()

    def contract_manifest(self) -> dict[str, object]:
        """Return the exact canonical contract pinned with reflection input."""

        return target_adapter_contract_manifest(self.policy)

    def _assert_root(self) -> None:
        try:
            root_stat = self.root.lstat()
        except OSError as exc:
            raise PromotionValidationError("project root disappeared") from exc
        if (
            stat.S_ISLNK(root_stat.st_mode)
            or not stat.S_ISDIR(root_stat.st_mode)
            or (root_stat.st_dev, root_stat.st_ino) != self._root_identity
        ):
            raise PromotionValidationError("project root identity changed")

    @contextmanager
    def _project_lock(self):
        """Serialize mutating operations without creating a lock file."""

        self._assert_root()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.root, flags)
        try:
            identity = os.fstat(descriptor)
            if (identity.st_dev, identity.st_ino) != self._root_identity:
                raise PromotionValidationError("project root identity changed")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._assert_root()
            self._descriptor_state.root = descriptor
            try:
                yield
            finally:
                del self._descriptor_state.root
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @contextmanager
    def _root_descriptor(self):
        active = getattr(self._descriptor_state, "root", None)
        if active is not None:
            yield active
            return
        self._assert_root()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.root, flags)
        try:
            identity = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(identity.st_mode)
                or (identity.st_dev, identity.st_ino) != self._root_identity
            ):
                raise PromotionValidationError("project root identity changed")
            yield descriptor
        finally:
            os.close(descriptor)

    def _relative_parts(self, path: Path) -> tuple[str, ...]:
        try:
            parts = path.relative_to(self.root).parts
        except ValueError as exc:
            raise OSError("path is outside the project root") from exc
        if any(part in {"", ".", ".."} or "\x00" in part for part in parts):
            raise OSError("path is outside the project root")
        return parts

    @contextmanager
    def _open_directory_descriptor(self, path: Path):
        parts = self._relative_parts(path)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with self._root_descriptor() as root_descriptor:
            current = os.dup(root_descriptor)
            try:
                for part in parts:
                    following = os.open(part, flags, dir_fd=current)
                    os.close(current)
                    current = following
                    identity = os.fstat(current)
                    if (
                        not stat.S_ISDIR(identity.st_mode)
                        or identity.st_dev != self._root_identity[0]
                    ):
                        raise OSError("directory escaped the project filesystem")
                yield current
            finally:
                os.close(current)

    @contextmanager
    def _open_parent_descriptor(self, path: Path):
        parts = self._relative_parts(path)
        if not parts:
            raise OSError("project root has no parent inside itself")
        with self._open_directory_descriptor(self.root.joinpath(*parts[:-1])) as parent:
            yield parent, parts[-1]

    @contextmanager
    def _open_regular_descriptor(
        self,
        path: Path,
        flags: int = os.O_RDONLY,
        mode: int = 0o600,
    ):
        with self._open_parent_descriptor(path) as (parent, name):
            descriptor = os.open(
                name,
                flags | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=parent,
            )
            try:
                identity = os.fstat(descriptor)
                if not stat.S_ISREG(identity.st_mode) or identity.st_dev != self._root_identity[0]:
                    raise OSError("file escaped the project filesystem")
                yield descriptor, identity
            finally:
                os.close(descriptor)

    def _read_rooted_file(
        self,
        path: Path,
        *,
        max_bytes: int | None = None,
    ) -> tuple[bytes, os.stat_result]:
        with self._open_regular_descriptor(path) as (descriptor, identity):
            if max_bytes is not None and identity.st_size > max_bytes:
                raise OverflowError
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                content = handle.read(None if max_bytes is None else max_bytes + 1)
            if max_bytes is not None and len(content) > max_bytes:
                raise OverflowError
            return content, identity

    def _write_new_rooted_file(
        self,
        path: Path,
        content: bytes,
        *,
        source_stat: os.stat_result | None = None,
        mode: int | None = None,
        preserve_times: bool = False,
    ) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        create_mode = (
            stat.S_IMODE(source_stat.st_mode)
            if source_stat is not None
            else (mode if mode is not None else 0o666)
        )
        with self._open_parent_descriptor(path) as (parent, name):
            descriptor = os.open(
                name,
                flags | getattr(os, "O_NOFOLLOW", 0),
                create_mode,
                dir_fd=parent,
            )
            try:
                identity = os.fstat(descriptor)
                if not stat.S_ISREG(identity.st_mode) or identity.st_dev != self._root_identity[0]:
                    raise OSError("file escaped the project filesystem")
                if source_stat is not None or mode is not None:
                    os.fchmod(descriptor, create_mode)
                remaining = memoryview(content)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("short file write")
                    remaining = remaining[written:]
                if source_stat is not None and preserve_times:
                    os.utime(
                        name,
                        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
                        dir_fd=parent,
                        follow_symlinks=False,
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _copy_rooted_file(
        self,
        source: Path,
        destination: Path,
        *,
        expected_content: bytes | None = None,
    ) -> os.stat_result:
        content, source_stat = self._read_rooted_file(source)
        if expected_content is not None and content != expected_content:
            raise OSError("live target changed before backup")
        self._write_new_rooted_file(
            destination,
            content,
            source_stat=source_stat,
            preserve_times=True,
        )
        return source_stat

    def _mkdir_rooted(self, path: Path) -> os.stat_result:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with self._open_parent_descriptor(path) as (parent, name):
            os.mkdir(name, dir_fd=parent)
            descriptor = os.open(name, flags, dir_fd=parent)
            try:
                identity = os.fstat(descriptor)
                if not stat.S_ISDIR(identity.st_mode) or identity.st_dev != self._root_identity[0]:
                    raise OSError("created directory escaped the project filesystem")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(parent)
            return identity

    def _rmdir_rooted(self, path: Path) -> None:
        with self._open_parent_descriptor(path) as (parent, name):
            os.rmdir(name, dir_fd=parent)
            os.fsync(parent)

    def _rooted_lstat(self, path: Path) -> os.stat_result:
        if path == self.root:
            with self._root_descriptor() as descriptor:
                return os.fstat(descriptor)
        with self._open_parent_descriptor(path) as (parent, name):
            return os.stat(name, dir_fd=parent, follow_symlinks=False)

    def _rooted_exists(self, path: Path) -> bool:
        try:
            self._rooted_lstat(path)
        except FileNotFoundError:
            return False
        return True

    def _ensure_rooted_directory(self, path: Path, *, create: bool) -> bool:
        try:
            identity = self._rooted_lstat(path)
        except FileNotFoundError:
            if not create:
                return False
            self._mkdir_rooted(path)
            return True
        if (
            stat.S_ISLNK(identity.st_mode)
            or not stat.S_ISDIR(identity.st_mode)
            or identity.st_dev != self._root_identity[0]
        ):
            raise OSError("journal directory escaped the project filesystem")
        return True

    def _list_rooted_directory(self, path: Path) -> tuple[Path, ...]:
        with self._open_directory_descriptor(path) as descriptor:
            with os.scandir(descriptor) as entries:
                names = tuple(sorted(entry.name for entry in entries))
        return tuple(path / name for name in names)

    def _unlink_rooted(self, path: Path) -> None:
        with self._open_parent_descriptor(path) as (parent, name):
            os.unlink(name, dir_fd=parent)
            os.fsync(parent)

    def _remove_tree_rooted(self, path: Path) -> None:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

        def clear(descriptor: int) -> None:
            with os.scandir(descriptor) as iterator:
                entries = tuple(iterator)
            for entry in entries:
                identity = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(identity.st_mode):
                    child = os.open(entry.name, directory_flags, dir_fd=descriptor)
                    try:
                        opened = os.fstat(child)
                        if (
                            not stat.S_ISDIR(opened.st_mode)
                            or opened.st_dev != self._root_identity[0]
                            or (opened.st_dev, opened.st_ino) != (identity.st_dev, identity.st_ino)
                        ):
                            raise OSError("journal tree changed during cleanup")
                        clear(child)
                    finally:
                        os.close(child)
                    os.rmdir(entry.name, dir_fd=descriptor)
                else:
                    os.unlink(entry.name, dir_fd=descriptor)
            os.fsync(descriptor)

        with self._open_parent_descriptor(path) as (parent, name):
            descriptor = os.open(name, directory_flags, dir_fd=parent)
            try:
                identity = os.fstat(descriptor)
                if not stat.S_ISDIR(identity.st_mode) or identity.st_dev != self._root_identity[0]:
                    raise OSError("journal tree escaped the project filesystem")
                clear(descriptor)
            finally:
                os.close(descriptor)
            os.rmdir(name, dir_fd=parent)
            os.fsync(parent)

    def bundle_from_content_map(
        self,
        contents: Mapping[str, str],
        *,
        include_missing: Iterable[str] = (),
    ) -> bundles.BundleSnapshot:
        """Validate a reflector content map and return its exact candidate snapshot."""

        try:
            bundle = bundles.bundle_from_content_map(
                contents,
                scope=self.scope,
                include_missing=include_missing,
            )
        except bundles.BundleValidationError as exc:
            raise PromotionValidationError(
                str(exc),
                changed_locators=exc.changed_locators,
            ) from exc
        bundle = self._validate_bundle(bundle)
        self._validate_live_paths(bundle)
        return bundle

    def _locator_parts(self, locator: str) -> tuple[str, ...]:
        _string(locator, "file locator")
        parts = tuple(locator.split("/"))
        unsafe = (
            "\x00" in locator
            or "\\" in locator
            or locator.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        )
        managed = locator.endswith(self.policy.suffix) and not any(
            part in self.policy.excluded_directories for part in parts[:-1]
        )
        if unsafe or not managed:
            raise PromotionValidationError(
                "file target is outside the managed scope", changed_locators=(locator,)
            )
        return parts

    @staticmethod
    def _validate_casefold_locators(locators: Iterable[str]) -> None:
        spellings: dict[str, str] = {}
        collisions: set[str] = set()
        for locator in sorted(set(locators)):
            folded = locator.casefold()
            existing = spellings.get(folded)
            if existing is not None and existing != locator:
                collisions.update((existing, locator))
            else:
                spellings[folded] = locator
        if collisions:
            raise PromotionValidationError(
                "case-fold locator collision in managed Markdown scope",
                changed_locators=collisions,
            )

    def _path(self, locator: str) -> Path:
        self._assert_root()
        parts = self._locator_parts(locator)
        path = self.root
        for index, part in enumerate(parts):
            path /= part
            try:
                path_stat = self._rooted_lstat(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise PromotionValidationError(
                    "managed path is unreadable",
                    changed_locators=(locator,),
                ) from exc
            mode = path_stat.st_mode
            leaf = index == len(parts) - 1
            valid = stat.S_ISREG(mode) if leaf else stat.S_ISDIR(mode)
            if path_stat.st_dev != self._root_identity[0]:
                raise PromotionValidationError(
                    "managed target is on another filesystem",
                    changed_locators=(locator,),
                )
            if stat.S_ISLNK(mode) or not valid:
                reason = "symlink traversal" if stat.S_ISLNK(mode) else "non-file collision"
                raise PromotionValidationError(reason, changed_locators=(locator,))
        return self.root.joinpath(*parts)

    def _validate_bundle(
        self,
        bundle: bundles.BundleSnapshot,
    ) -> bundles.BundleSnapshot:
        if not isinstance(bundle, bundles.BundleSnapshot):
            raise PromotionValidationError("expected BundleSnapshot")
        if bundle.scope != self.scope:
            raise PromotionValidationError("bundle scope does not match adapter target")
        self._validate_casefold_locators(bundle.locators)
        existing_count = 0
        total_bytes = 0
        for target in bundle.targets:
            if target.kind != "file":
                raise PromotionValidationError(
                    "unsupported target kind", changed_locators=(target.locator,)
                )
            self._locator_parts(target.locator)
            if target.path != target.locator:
                raise PromotionValidationError(
                    "file path must equal locator", changed_locators=(target.locator,)
                )
            if target.exists:
                try:
                    content_bytes = target.content.encode("utf-8")  # type: ignore[union-attr]
                except UnicodeEncodeError as exc:
                    raise PromotionValidationError(
                        "invalid UTF-8 content", changed_locators=(target.locator,)
                    ) from exc
                existing_count += 1
                if existing_count > self.policy.max_files:
                    raise PromotionValidationError(
                        "managed Markdown file count limit exceeded",
                        changed_locators=(target.locator,),
                    )
                if len(content_bytes) > self.policy.max_file_bytes:
                    raise PromotionValidationError(
                        "managed Markdown per-file byte limit exceeded",
                        changed_locators=(target.locator,),
                    )
                total_bytes += len(content_bytes)
                if total_bytes > self.policy.max_total_bytes:
                    raise PromotionValidationError(
                        "managed Markdown total byte limit exceeded",
                        changed_locators=(target.locator,),
                    )
        return bundle

    def _validate_live_paths(self, bundle: bundles.BundleSnapshot) -> None:
        self._validate_casefold_locators((*bundle.locators, *self._discover()))
        for target in bundle.targets:
            self._path(target.locator)

    def _discover(self) -> set[str]:
        discovered: set[str] = set()
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)

        def visit(directory: int, parent_parts: tuple[str, ...]) -> None:
            locator = "/".join(parent_parts)
            try:
                with os.scandir(directory) as iterator:
                    entries = tuple(sorted(iterator, key=lambda entry: entry.name))
            except OSError as exc:
                raise PromotionValidationError(
                    "managed directory is unreadable",
                    changed_locators=(locator,) if locator else (),
                ) from exc
            for entry in entries:
                name = entry.name
                entry_parts = (*parent_parts, name)
                entry_locator = "/".join(entry_parts)
                if name in self.policy.excluded_directories:
                    continue
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise PromotionValidationError(
                        "managed entry is unreadable",
                        changed_locators=(entry_locator,),
                    ) from exc
                mode = entry_stat.st_mode
                if stat.S_ISLNK(mode):
                    raise PromotionValidationError(
                        "non-excluded symlink makes managed Markdown scope ambiguous",
                        changed_locators=(entry_locator,),
                    )
                if entry_stat.st_dev != self._root_identity[0]:
                    raise PromotionValidationError(
                        "managed entry is on another filesystem",
                        changed_locators=(entry_locator,),
                    )
                if stat.S_ISDIR(mode):
                    if "\\" in name or "\x00" in name:
                        raise PromotionValidationError(
                            "directory path is outside the managed scope",
                            changed_locators=(entry_locator,),
                        )
                    try:
                        child = os.open(name, directory_flags, dir_fd=directory)
                    except OSError as exc:
                        raise PromotionValidationError(
                            "managed directory changed during capture",
                            changed_locators=(entry_locator,),
                        ) from exc
                    try:
                        opened = os.fstat(child)
                        if opened.st_dev != self._root_identity[0]:
                            raise PromotionValidationError(
                                "managed entry is on another filesystem",
                                changed_locators=(entry_locator,),
                            )
                        if not stat.S_ISDIR(opened.st_mode) or (
                            opened.st_dev,
                            opened.st_ino,
                        ) != (entry_stat.st_dev, entry_stat.st_ino):
                            raise PromotionValidationError(
                                "managed directory changed during capture",
                                changed_locators=(entry_locator,),
                            )
                        visit(child, entry_parts)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(mode):
                    if entry_locator.endswith(self.policy.suffix):
                        self._locator_parts(entry_locator)
                        discovered.add(entry_locator)
                        if len(discovered) > self.policy.max_files:
                            raise PromotionValidationError(
                                "managed Markdown file count limit exceeded",
                                changed_locators=(entry_locator,),
                            )
                elif entry_locator.endswith(self.policy.suffix):
                    raise PromotionValidationError(
                        "managed Markdown non-file collision",
                        changed_locators=(entry_locator,),
                    )

        with self._root_descriptor() as root_descriptor:
            visit(root_descriptor, ())
        self._validate_casefold_locators(discovered)
        return discovered

    def _read(self, locator: str) -> bundles.TargetSnapshot:
        path = self._path(locator)
        try:
            raw_content, _path_stat = self._read_rooted_file(
                path,
                max_bytes=self.policy.max_file_bytes,
            )
            content = raw_content.decode("utf-8")
        except FileNotFoundError:
            return bundles.TargetSnapshot("file", locator, False, None, path=locator)
        except OverflowError as exc:
            raise PromotionValidationError(
                "managed Markdown per-file byte limit exceeded",
                changed_locators=(locator,),
            ) from exc
        except PromotionValidationError:
            raise
        except UnicodeDecodeError as exc:
            raise PromotionValidationError(
                "managed file is not valid UTF-8", changed_locators=(locator,)
            ) from exc
        except OSError as exc:
            raise PromotionValidationError(
                "managed Markdown path changed during capture",
                changed_locators=(locator,),
            ) from exc
        return bundles.TargetSnapshot("file", locator, True, content, path=locator)

    def capture(self, *, include_missing: Iterable[str] = ()) -> bundles.BundleSnapshot:
        """Capture one coherent scope snapshot under the promotion lock."""

        with self._project_lock():
            return self._capture_unlocked(include_missing=include_missing)

    def _capture_unlocked(
        self,
        *,
        include_missing: Iterable[str] = (),
    ) -> bundles.BundleSnapshot:
        self._assert_root()
        included: set[str] = set()
        for locator in include_missing:
            self._locator_parts(locator)
            if locator in included:
                raise PromotionValidationError(
                    "duplicate capture locator", changed_locators=(locator,)
                )
            included.add(locator)
        locators = self._discover() | included
        self._validate_casefold_locators(locators)
        targets: list[bundles.TargetSnapshot] = []
        total_bytes = 0
        for locator in sorted(locators):
            target = self._read(locator)
            targets.append(target)
            if target.exists:
                total_bytes += len(target.content.encode("utf-8"))  # type: ignore[union-attr]
                if total_bytes > self.policy.max_total_bytes:
                    raise PromotionValidationError(
                        "managed Markdown total byte limit exceeded",
                        changed_locators=(locator,),
                    )
        return self._validate_bundle(
            bundles.BundleSnapshot(
                tuple(targets),
                self.scope,
            )
        )

    def _capture_for_drift(
        self,
        baseline: bundles.BundleSnapshot,
    ) -> bundles.BundleSnapshot:
        try:
            return self._capture_unlocked(include_missing=baseline.locators)
        except PromotionValidationError as exc:
            raise StaleBaseError(baseline, None, exc.changed_locators) from exc

    def transaction_path(self, promotion_id: str) -> Path:
        promotion_id = _string(promotion_id, "promotion_id")
        return self.journal_root / hashlib.sha256(promotion_id.encode()).hexdigest()

    def _fsync_directory(self, path: Path) -> None:
        with self._open_directory_descriptor(path) as descriptor:
            os.fsync(descriptor)

    def _journal_parents(self, *, create: bool = False) -> None:
        metadata_root = self.root / ".weave-agent-signals"
        metadata_exists = self._ensure_rooted_directory(metadata_root, create=create)
        if not metadata_exists:
            return
        self._ensure_rooted_directory(self.journal_root, create=create)

    @staticmethod
    def _journal_path(transaction: Path) -> Path:
        return transaction / "journal.json"

    def _write_journal(self, transaction: Path, record: Mapping[str, Any]) -> None:
        temporary = transaction / f"journal-{secrets.token_hex(8)}.tmp"
        data = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        try:
            self._write_new_rooted_file(temporary, data, mode=0o600)
            self._replace_staged(temporary, self._journal_path(transaction))
            self._fsync_directory(transaction)
        finally:
            try:
                self._unlink_rooted(temporary)
            except OSError:
                pass

    def _read_journal(self, transaction: Path) -> dict[str, Any]:
        try:
            raw, _identity = self._read_rooted_file(self._journal_path(transaction))
            record = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PromotionTransactionError(
                "promotion journal is corrupt", changed_locators=(), cause=exc
            ) from exc
        if not isinstance(record, dict) or record.get("version") != 1:
            raise PromotionTransactionError("unsupported promotion journal", changed_locators=())
        return record

    def _set_state(self, transaction: Path, record: dict[str, Any], state: str) -> None:
        record["state"] = state
        self._write_journal(transaction, record)

    def _created_directories(
        self,
        preview: bundles.PromotionPreview,
    ) -> tuple[str, ...]:
        missing: set[str] = set()
        for action in preview.actions:
            if not action.after.exists:
                continue
            path = self.root
            parent_missing = False
            for part in action.locator.split("/")[:-1]:
                path /= part
                if parent_missing:
                    missing.add(str(path.relative_to(self.root)))
                    continue
                try:
                    identity = self._rooted_lstat(path)
                except FileNotFoundError:
                    parent_missing = True
                    missing.add(str(path.relative_to(self.root)))
                    continue
                if (
                    stat.S_ISLNK(identity.st_mode)
                    or not stat.S_ISDIR(identity.st_mode)
                    or identity.st_dev != self._root_identity[0]
                ):
                    raise PromotionValidationError(
                        "managed directory changed before promotion",
                        changed_locators=(action.locator,),
                    )
        return tuple(sorted(missing, key=lambda value: (value.count("/"), value)))

    def _prepare(
        self,
        promotion_id: str,
        intent: str,
        receipt: PromotionReceipt,
        preview: bundles.PromotionPreview,
    ) -> tuple[Path, dict[str, Any]]:
        self._journal_parents(create=True)
        transaction = self.transaction_path(promotion_id)
        changed = preview.changed_locators
        try:
            transaction_exists = self._rooted_exists(transaction)
        except OSError as exc:
            raise PromotionTransactionError(
                "promotion journal path changed",
                changed_locators=changed,
                cause=exc,
            ) from exc
        if transaction_exists:
            raise PromotionTransactionError(
                "promotion transaction already exists", changed_locators=changed
            )
        record = {
            "version": 1,
            "state": "preparing",
            "promotion_id": promotion_id,
            "intent": intent,
            "created_directories": [],
            "receipt": receipt.to_dict(),
        }
        try:
            self._mkdir_rooted(transaction)
            if self._rooted_lstat(transaction).st_dev != self._root_identity[0]:
                raise OSError("staging is not on the project filesystem")
            self._write_journal(transaction, record)
            stage, backup = transaction / "stage", transaction / "backup"
            self._mkdir_rooted(stage)
            self._mkdir_rooted(backup)
            self._fsync_directory(transaction)
            for index, action in enumerate(preview.actions):
                destination = self._path(action.locator)
                source_stat: os.stat_result | None = None
                if action.before.exists:
                    backup_file = backup / f"{index:04d}"
                    source_stat = self._copy_rooted_file(
                        destination,
                        backup_file,
                        expected_content=action.before.content.encode("utf-8"),  # type: ignore[union-attr]
                    )
                if action.after.exists:
                    staged = stage / f"{index:04d}"
                    self._write_new_rooted_file(
                        staged,
                        action.after.content.encode("utf-8"),  # type: ignore[union-attr]
                        source_stat=source_stat,
                    )
            self._fsync_directory(stage)
            self._fsync_directory(backup)
            self._set_state(transaction, record, "prepared")
            return transaction, record
        except Exception as exc:
            try:
                if self._rooted_exists(transaction):
                    self._remove_tree_rooted(transaction)
                if self._rooted_exists(self.journal_root):
                    self._fsync_directory(self.journal_root)
                self._prune_journal_parents()
            except OSError:
                pass
            raise PromotionTransactionError(
                "could not prepare promotion", changed_locators=changed, cause=exc
            ) from exc

    def _replace_staged(self, staged: Path, destination: Path) -> None:
        """Small fault-injection seam for transaction tests."""

        with self._open_parent_descriptor(staged) as (staged_parent, staged_name):
            with self._open_parent_descriptor(destination) as (
                destination_parent,
                destination_name,
            ):
                os.replace(
                    staged_name,
                    destination_name,
                    src_dir_fd=staged_parent,
                    dst_dir_fd=destination_parent,
                )

    def _unlink_live(self, destination: Path) -> None:
        """Small fault-injection seam for transaction tests."""

        with self._open_parent_descriptor(destination) as (parent, name):
            os.unlink(name, dir_fd=parent)

    def _apply(
        self,
        transaction: Path,
        preview: bundles.PromotionPreview,
        record: dict[str, Any],
        applied_indexes: list[int],
    ) -> None:
        created = record.get("created_directories")
        if not isinstance(created, list):
            raise OSError("journal directory ownership is invalid")
        for locator in self._created_directories(preview):
            directory = self.root / locator
            identity = self._mkdir_rooted(directory)
            created.append(
                _CreatedDirectory(
                    locator=locator,
                    device=identity.st_dev,
                    inode=identity.st_ino,
                ).to_dict()
            )
            self._write_journal(transaction, record)
        for index, action in enumerate(preview.actions):
            destination = self._path(action.locator)
            live = self._read(action.locator)
            if not self._same_target(live, action.before):
                raise OSError(f"target changed during promotion: {action.locator}")
            if action.after.exists:
                self._replace_staged(transaction / "stage" / f"{index:04d}", destination)
            else:
                self._unlink_live(destination)
            applied_indexes.append(index)
            self._fsync_directory(destination.parent)

    def _rollback(
        self,
        transaction: Path,
        preview: bundles.PromotionPreview,
        created_directories: Iterable[_CreatedDirectory],
        applied_indexes: Iterable[int] | None = None,
    ) -> None:
        restore = transaction / "restore"
        if not self._rooted_exists(restore):
            self._mkdir_rooted(restore)
        indexes = (
            tuple(range(len(preview.actions)))
            if applied_indexes is None
            else tuple(applied_indexes)
        )
        divergent: list[str] = []
        for index in indexes:
            action = preview.actions[index]
            live = self._read(action.locator)
            if not self._same_target(live, action.after):
                divergent.append(action.locator)
        if divergent:
            raise PromotionTransactionError(
                "rollback refused to overwrite targets changed after apply",
                changed_locators=divergent,
            )
        for index in reversed(indexes):
            action = preview.actions[index]
            destination = self.root / action.locator
            if action.before.exists:
                backup = transaction / "backup" / f"{index:04d}"
                try:
                    backup_content, backup_stat = self._read_rooted_file(backup)
                except (FileNotFoundError, NotADirectoryError):
                    raise OSError(f"missing backup for {action.locator}")
                staged = restore / f"{index:04d}-{secrets.token_hex(8)}"
                self._write_new_rooted_file(
                    staged,
                    backup_content,
                    source_stat=backup_stat,
                    preserve_times=True,
                )
                self._fsync_directory(restore)
                self._replace_staged(staged, destination)
                self._fsync_directory(destination.parent)
            elif self._rooted_exists(destination):
                self._unlink_live(destination)
                self._fsync_directory(destination.parent)
        self._remove_created_directories(created_directories)

    def _remove_created_directories(
        self,
        created_directories: Iterable[_CreatedDirectory],
    ) -> None:
        for created in sorted(
            created_directories,
            key=lambda value: value.locator.count("/"),
            reverse=True,
        ):
            try:
                directory = self.root / created.locator
                identity = self._rooted_lstat(directory)
                if not stat.S_ISDIR(identity.st_mode) or (identity.st_dev, identity.st_ino) != (
                    created.device,
                    created.inode,
                ):
                    continue
                self._rmdir_rooted(directory)
            except OSError:
                continue

    def _receipt(self, record: Mapping[str, Any]) -> PromotionReceipt:
        try:
            return PromotionReceipt.from_dict(record["receipt"])
        except (KeyError, TypeError, PromotionValidationError) as exc:
            changed = getattr(exc, "changed_locators", ())
            raise PromotionTransactionError(
                "journal receipt is invalid", changed_locators=changed, cause=exc
            ) from exc

    @staticmethod
    def _same_target(
        left: bundles.TargetSnapshot,
        right: bundles.TargetSnapshot,
    ) -> bool:
        return (left.kind, left.exists, left.content) == (
            right.kind,
            right.exists,
            right.content,
        )

    @staticmethod
    def _bundle_matches(
        expected: bundles.BundleSnapshot,
        current: bundles.BundleSnapshot,
    ) -> bool:
        try:
            return not bundles.compare_bundles(expected, current).actions
        except bundles.BundleValidationError:
            return False

    def _capture_union(
        self,
        *snapshots: bundles.BundleSnapshot,
    ) -> bundles.BundleSnapshot:
        return self._capture_unlocked(
            include_missing={locator for snapshot in snapshots for locator in snapshot.locators}
        )

    def _recovery_divergence(
        self,
        receipt: PromotionReceipt,
        current: bundles.BundleSnapshot,
    ) -> tuple[str, ...]:
        past = receipt.past.by_locator
        promoted = receipt.promoted.by_locator
        live = current.by_locator
        actions = {
            action.locator: action
            for action in bundles.compare_bundles(
                receipt.past,
                receipt.promoted,
            ).actions
        }
        changed: list[str] = []
        for locator in sorted(set(past) | set(promoted) | set(live)):
            target = live.get(locator)
            action = actions.get(locator)
            if action is not None:
                allowed = (action.before, action.after)
            else:
                unchanged = past.get(locator) or promoted.get(locator)
                allowed = (unchanged,) if unchanged is not None else ()
            if target is None or not allowed:
                changed.append(locator)
            elif not any(self._same_target(target, item) for item in allowed):
                changed.append(locator)
        return tuple(changed)

    def _journal_directories(
        self,
        record: Mapping[str, Any],
        preview: bundles.PromotionPreview,
    ) -> tuple[_CreatedDirectory, ...]:
        raw = record.get("created_directories")
        if not isinstance(raw, list):
            raise PromotionTransactionError(
                "journal directories are invalid", changed_locators=preview.changed_locators
            )
        allowed: set[str] = set()
        for action in preview.actions:
            if not action.after.exists:
                continue
            parts = action.locator.split("/")[:-1]
            allowed.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
        parsed: list[_CreatedDirectory] = []
        for item in raw:
            if (
                not isinstance(item, dict)
                or set(item) != {"locator", "device", "inode"}
                or not isinstance(item["locator"], str)
                or type(item["device"]) is not int
                or type(item["inode"]) is not int
                or item["device"] != self._root_identity[0]
                or item["inode"] <= 0
            ):
                raise PromotionTransactionError(
                    "journal directories are invalid",
                    changed_locators=preview.changed_locators,
                )
            parsed.append(
                _CreatedDirectory(
                    locator=item["locator"],
                    device=item["device"],
                    inode=item["inode"],
                )
            )
        locators = [item.locator for item in parsed]
        if len(locators) != len(set(locators)) or not set(locators).issubset(allowed):
            raise PromotionTransactionError(
                "journal directory escapes the managed scope",
                changed_locators=preview.changed_locators,
            )
        return tuple(parsed)

    def _valid_orphan_journal_temp(self, path: Path) -> bool:
        prefix, suffix = "journal-", ".tmp"
        token = path.name[len(prefix) : -len(suffix)]
        if (
            not path.name.startswith(prefix)
            or not path.name.endswith(suffix)
            or len(token) != 16
            or any(character not in "0123456789abcdef" for character in token)
        ):
            return False
        try:
            path_stat = self._rooted_lstat(path)
        except OSError:
            return False
        return (
            stat.S_ISREG(path_stat.st_mode)
            and path_stat.st_dev == self._root_identity[0]
            and path_stat.st_nlink == 1
        )

    def _existing(self, promotion_id: str) -> tuple[PromotionReceipt | None, str | None]:
        transaction = self.transaction_path(promotion_id)
        try:
            if not self._rooted_exists(transaction):
                return None, None
        except OSError as exc:
            raise PromotionTransactionError(
                "promotion transaction cannot be inspected",
                changed_locators=(),
                cause=exc,
            ) from exc
        self._journal_parents()
        try:
            entries = self._list_rooted_directory(transaction)
        except OSError as exc:
            raise PromotionTransactionError(
                "promotion transaction cannot be inspected",
                changed_locators=(),
                cause=exc,
            ) from exc
        if not entries:
            self._rmdir_rooted(transaction)
            self._prune_journal_parents()
            return None, None
        journal = self._journal_path(transaction)
        if not self._rooted_exists(journal):
            if len(entries) != 1 or not self._valid_orphan_journal_temp(entries[0]):
                raise PromotionTransactionError(
                    "promotion transaction has no canonical journal",
                    changed_locators=(),
                )
            self._unlink_rooted(entries[0])
            self._rmdir_rooted(transaction)
            self._prune_journal_parents()
            return None, None
        record = self._read_journal(transaction)
        if record.get("promotion_id") != promotion_id or not isinstance(record.get("intent"), str):
            raise PromotionTransactionError("journal identity is invalid", changed_locators=())
        receipt, state = self._receipt(record), record.get("state")
        if (
            receipt.target_kind != "file"
            or receipt.target_id != self.target_id
            or receipt.past.scope != self.scope
            or receipt.evaluated_candidate.scope != self.scope
            or receipt.promoted.scope != self.scope
            or record["intent"]
            != self._intent(
                receipt.promotion_id,
                receipt.run_id,
                receipt.candidate_id,
                receipt.past,
                receipt.evaluated_candidate,
                receipt.promoted,
                receipt.review_revision,
                receipt.unevaluated_d_acknowledged,
                receipt.git_metadata,
            )
        ):
            raise PromotionTransactionError(
                "journal receipt does not belong to this target", changed_locators=()
            )
        if state == "committed":
            return receipt, record["intent"]
        if state == "rolled_back":
            return None, record["intent"]
        if state not in {"preparing", "prepared"}:
            raise PromotionTransactionError(
                "journal state is invalid",
                changed_locators=receipt.promoted.locators,
            )
        current = self._capture_union(receipt.past, receipt.promoted)
        if state == "prepared" and self._bundle_matches(receipt.promoted, current):
            self._set_state(transaction, record, "committed")
            return receipt, record["intent"]
        preview = bundles.compare_bundles(receipt.past, receipt.promoted)
        if state == "preparing":
            if not self._bundle_matches(receipt.past, current):
                raise PromotionTransactionError(
                    "interrupted preparation no longer matches baseline",
                    changed_locators=bundles.changed_locators(receipt.past, current),
                )
            self._set_state(transaction, record, "rolled_back")
            return None, record["intent"]
        divergent = self._recovery_divergence(receipt, current)
        if divergent:
            raise PromotionTransactionError(
                "managed targets diverged after interrupted promotion",
                changed_locators=divergent,
            )
        directories = self._journal_directories(record, preview)
        if self._bundle_matches(receipt.past, current):
            self._remove_created_directories(directories)
            self._set_state(transaction, record, "rolled_back")
            return None, record["intent"]
        applied_indexes = tuple(
            index
            for index, action in enumerate(preview.actions)
            if self._same_target(current.by_locator[action.locator], action.after)
        )
        try:
            self._rollback(
                transaction,
                preview,
                directories,
                applied_indexes,
            )
            restored = self._capture_union(receipt.past, receipt.promoted)
            if not self._bundle_matches(receipt.past, restored):
                raise OSError("recovery did not restore baseline")
            self._set_state(transaction, record, "rolled_back")
        except Exception as exc:
            raise PromotionTransactionError(
                "could not recover promotion",
                changed_locators=preview.changed_locators,
                cause=exc,
                rollback_error=exc,
            ) from exc
        return None, record["intent"]

    def recover(self, promotion_id: str) -> PromotionReceipt | None:
        """Return a committed receipt or roll an interrupted promotion back."""

        with self._project_lock():
            return self._existing(promotion_id)[0]

    def find_committed_receipt(self, run_id: str) -> PromotionReceipt | None:
        """Find an unacknowledged committed promotion for one run.

        This is the restart-recovery path for the narrow crash window between
        committing target changes and persisting their receipt in the run DB.
        """

        run_id = _string(run_id, "run_id")
        with self._project_lock():
            try:
                if not self._rooted_exists(self.journal_root):
                    return None
            except OSError as exc:
                raise PromotionTransactionError(
                    "promotion journals cannot be inspected",
                    changed_locators=(),
                    cause=exc,
                ) from exc
            self._journal_parents()
            matches: list[PromotionReceipt] = []
            try:
                transactions = self._list_rooted_directory(self.journal_root)
            except OSError as exc:
                raise PromotionTransactionError(
                    "promotion journals cannot be inspected",
                    changed_locators=(),
                    cause=exc,
                ) from exc
            for transaction in transactions:
                try:
                    identity = self._rooted_lstat(transaction)
                except OSError as exc:
                    raise PromotionTransactionError(
                        "promotion journal identity is invalid",
                        changed_locators=(),
                        cause=exc,
                    ) from exc
                if not stat.S_ISDIR(identity.st_mode) or identity.st_dev != self._root_identity[0]:
                    raise PromotionTransactionError(
                        "promotion journal identity is invalid",
                        changed_locators=(),
                    )
                record = self._read_journal(transaction)
                promotion_id = record.get("promotion_id")
                if not isinstance(promotion_id, str) or transaction != self.transaction_path(
                    promotion_id
                ):
                    raise PromotionTransactionError(
                        "promotion journal identity is invalid",
                        changed_locators=(),
                    )
                receipt, _ = self._existing(promotion_id)
                if receipt is not None and receipt.run_id == run_id:
                    matches.append(receipt)
            if len(matches) > 1:
                raise PromotionTransactionError(
                    "multiple committed promotions require recovery",
                    changed_locators={
                        locator for receipt in matches for locator in receipt.promoted.locators
                    },
                )
            return matches[0] if matches else None

    def rollback_committed(self, promotion_id: str) -> PromotionReceipt:
        """Undo a committed promotion whose durable receipt could not be saved.

        Rollback is deliberately conditional: it proceeds only while the live
        managed scope still equals the promotion's proposed snapshot. This
        keeps recovery from overwriting a user edit made after the file commit.
        """

        promotion_id = _string(promotion_id, "promotion_id")
        with self._project_lock():
            receipt, _ = self._existing(promotion_id)
            transaction = self.transaction_path(promotion_id)
            if receipt is None or not self._rooted_exists(transaction):
                raise PromotionTransactionError(
                    "promotion is not committed",
                    changed_locators=(),
                )
            record = self._read_journal(transaction)
            if record.get("state") != "committed":
                raise PromotionTransactionError(
                    "promotion is not committed",
                    changed_locators=receipt.promoted.locators,
                )
            current = self._capture_union(receipt.past, receipt.promoted)
            if not self._bundle_matches(receipt.promoted, current):
                raise PromotionTransactionError(
                    "committed promotion no longer matches the live target",
                    changed_locators=bundles.changed_locators(
                        receipt.promoted,
                        current,
                    ),
                )
            preview = bundles.compare_bundles(receipt.past, receipt.promoted)
            directories = self._journal_directories(record, preview)
            try:
                self._rollback(transaction, preview, directories)
                restored = self._capture_union(receipt.past, receipt.promoted)
                if not self._bundle_matches(receipt.past, restored):
                    raise OSError("rollback did not restore baseline")
                self._set_state(transaction, record, "rolled_back")
            except PromotionTransactionError:
                raise
            except Exception as exc:
                raise PromotionTransactionError(
                    "could not roll back committed promotion",
                    changed_locators=preview.changed_locators,
                    cause=exc,
                    rollback_error=exc,
                ) from exc
            return receipt

    def _prune_journal_parents(self) -> None:
        for directory in (self.journal_root, self.root / ".weave-agent-signals"):
            try:
                self._rmdir_rooted(directory)
            except OSError:
                pass

    def acknowledge(self, promotion_id: str) -> bool:
        """Clean a stable committed or rolled-back journal after persistence."""

        with self._project_lock():
            return self._acknowledge(promotion_id)

    def _acknowledge(self, promotion_id: str) -> bool:
        transaction = self.transaction_path(promotion_id)
        if not self._rooted_exists(transaction):
            return False
        self._journal_parents()
        record = self._read_journal(transaction)
        if record.get("state") not in {"committed", "rolled_back"}:
            raise PromotionTransactionError("transaction is not stable", changed_locators=())
        self._remove_tree_rooted(transaction)
        self._prune_journal_parents()
        return True

    def _intent(
        self,
        promotion_id: str,
        run_id: str,
        candidate_id: str,
        past: bundles.BundleSnapshot,
        evaluated_candidate: bundles.BundleSnapshot,
        promoted: bundles.BundleSnapshot,
        review_revision: int,
        acknowledge_unevaluated: bool,
        git_metadata: Mapping[str, Any] | None,
    ) -> str:
        return bundles.revision_hash(
            {
                "promotion_id": promotion_id,
                "run_id": run_id,
                "candidate_id": candidate_id,
                "target_id": self.target_id,
                "past": past.revision,
                "evaluated_candidate": evaluated_candidate.revision,
                "promoted": promoted.revision,
                "review_revision": review_revision,
                "acknowledge_unevaluated": acknowledge_unevaluated,
                "git": _thaw(_freeze(git_metadata or {})),
            }
        )

    def _decided_at(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise PromotionValidationError("clock must return an aware datetime")
        return value.astimezone(timezone.utc).isoformat()

    def promote(
        self,
        *,
        promotion_id: str,
        run_id: str,
        candidate_id: str,
        past: bundles.BundleSnapshot,
        evaluated_candidate: bundles.BundleSnapshot,
        promoted: bundles.BundleSnapshot,
        review_revision: int,
        acknowledge_unevaluated: bool = False,
        git_metadata: Mapping[str, Any] | None = None,
    ) -> PromotionReceipt:
        """Promote exact C-or-D iff the current complete scope still equals B."""

        promotion_id = _string(promotion_id, "promotion_id")
        run_id, candidate_id = _string(run_id, "run_id"), _string(candidate_id, "candidate_id")
        past = self._validate_bundle(past)
        evaluated_candidate = self._validate_bundle(evaluated_candidate)
        promoted = self._validate_bundle(promoted)
        try:
            bundles.validate_edited_bundle(
                past,
                evaluated_candidate,
                promoted,
            )
            preview = bundles.compare_bundles(past, promoted)
        except bundles.BundleValidationError as exc:
            raise PromotionValidationError(
                str(exc),
                changed_locators=exc.changed_locators,
            ) from exc
        promoted_was_evaluated = evaluated_candidate == promoted
        if type(review_revision) is not int or review_revision < 0:
            raise PromotionValidationError("review_revision must be a non-negative integer")
        if type(acknowledge_unevaluated) is not bool or acknowledge_unevaluated != (
            not promoted_was_evaluated
        ):
            raise PromotionValidationError(
                "unevaluated D promotion requires an explicit acknowledgement"
                if not promoted_was_evaluated
                else "evaluated C promotion must not claim an unevaluated acknowledgement"
            )
        intent = self._intent(
            promotion_id,
            run_id,
            candidate_id,
            past,
            evaluated_candidate,
            promoted,
            review_revision,
            acknowledge_unevaluated,
            git_metadata,
        )
        with self._project_lock():
            return self._promote_locked(
                promotion_id,
                run_id,
                candidate_id,
                past,
                evaluated_candidate,
                promoted,
                review_revision,
                preview,
                intent,
                acknowledge_unevaluated,
                git_metadata,
            )

    def _promote_locked(
        self,
        promotion_id: str,
        run_id: str,
        candidate_id: str,
        past: bundles.BundleSnapshot,
        evaluated_candidate: bundles.BundleSnapshot,
        promoted: bundles.BundleSnapshot,
        review_revision: int,
        preview: bundles.PromotionPreview,
        intent: str,
        acknowledge_unevaluated: bool,
        git_metadata: Mapping[str, Any] | None,
    ) -> PromotionReceipt:
        existing, existing_intent = self._existing(promotion_id)
        if existing_intent is not None and existing_intent != intent:
            raise PromotionValidationError(
                "promotion_id was used for another request",
                changed_locators=preview.changed_locators,
            )
        if existing is not None:
            return existing
        if self._rooted_exists(self.transaction_path(promotion_id)):
            self._acknowledge(promotion_id)

        current = self._capture_for_drift(past)
        if not self._bundle_matches(past, current):
            raise StaleBaseError(past, current)
        for action in preview.actions:
            try:
                self._path(action.locator)
            except PromotionValidationError as exc:
                raise StaleBaseError(past, None, exc.changed_locators) from exc
        self._journal_parents()
        receipt = PromotionReceipt(
            promotion_id=promotion_id,
            run_id=run_id,
            candidate_id=candidate_id,
            target_kind="file",
            target_id=self.target_id,
            past=past,
            evaluated_candidate=evaluated_candidate,
            promoted=promoted,
            review_revision=review_revision,
            actions=preview.actions,
            decided_at=self._decided_at(),
            promoted_was_evaluated=evaluated_candidate == promoted,
            unevaluated_d_acknowledged=acknowledge_unevaluated,
            git_metadata=git_metadata,
        )
        transaction, record = self._prepare(promotion_id, intent, receipt, preview)

        try:
            restaged = self._capture_for_drift(past)
        except StaleBaseError:
            self._set_state(transaction, record, "rolled_back")
            raise
        if not self._bundle_matches(past, restaged):
            self._set_state(transaction, record, "rolled_back")
            raise StaleBaseError(past, restaged)
        applied_indexes: list[int] = []
        try:
            self._apply(transaction, preview, record, applied_indexes)
            final = self._capture_union(past, promoted)
            if not self._bundle_matches(promoted, final):
                raise OSError("final digest mismatch")
            self._set_state(transaction, record, "committed")
            return receipt
        except Exception as exc:
            rollback_error: Exception | None = None
            try:
                directories = self._journal_directories(record, preview)
                self._rollback(
                    transaction,
                    preview,
                    directories,
                    applied_indexes,
                )
                restored = self._capture_union(past, promoted)
                if not self._bundle_matches(past, restored):
                    raise OSError("rollback digest mismatch")
                self._set_state(transaction, record, "rolled_back")
            except Exception as rollback_exc:
                rollback_error = rollback_exc
            raise PromotionTransactionError(
                "promotion failed and was rolled back"
                if rollback_error is None
                else "promotion failed and rollback requires recovery",
                changed_locators=preview.changed_locators,
                cause=exc,
                rollback_error=rollback_error,
            ) from exc
