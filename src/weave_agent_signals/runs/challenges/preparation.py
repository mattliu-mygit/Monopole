"""Deterministic preparation of public task materials before paired execution."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from weave_agent_signals.runs.challenges.contracts import (
    AuthoredTask,
    TaskMaterial,
    TaskMaterialPlan,
    TaskPreflightError,
    validate_public_git_url,
)
from weave_agent_signals.runs.challenges.workspace import (
    WorkspaceFile,
    WorkspaceSnapshot,
    capture_workspace,
)

RepositoryFetcher = Callable[[TaskMaterial], WorkspaceSnapshot]


def _git_environment() -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }


def resolve_repository_head(url: str) -> str:
    """Resolve a public repository's advertised HEAD to an immutable commit SHA."""

    url = validate_public_git_url(url)
    result = subprocess.run(
        [
            "git",
            "-c",
            "credential.helper=",
            "-c",
            "core.askPass=",
            "ls-remote",
            "--exit-code",
            url,
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=_git_environment(),
    )
    if result.returncode != 0:
        message = " ".join(result.stderr.split())[:500]
        raise RuntimeError(f"public Git HEAD resolution failed: {message}")
    fields = result.stdout.split()
    if len(fields) != 2 or fields[1] != "HEAD" or not re.fullmatch(r"[0-9a-fA-F]{40}", fields[0]):
        raise RuntimeError("public Git HEAD resolution returned an invalid revision")
    return fields[0].lower()


def _fetch_repository(material: TaskMaterial) -> WorkspaceSnapshot:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        commands = (
            ("init", ["git", "init", "--quiet"]),
            ("remote", ["git", "remote", "add", "origin", material.url]),
            (
                "fetch",
                [
                    "git",
                    "-c",
                    "credential.helper=",
                    "-c",
                    "core.askPass=",
                    "fetch",
                    "--quiet",
                    "--depth",
                    "1",
                    "origin",
                    material.revision,
                ],
            ),
            ("checkout", ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"]),
        )
        for stage, command in commands:
            result = subprocess.run(
                command,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
                env=_git_environment(),
            )
            if result.returncode != 0:
                message = " ".join(result.stderr.split())[:500]
                raise RuntimeError(f"public Git material preparation failed at {stage}: {message}")
        return capture_workspace(root)


def prepare_task_context(
    seed: WorkspaceSnapshot,
    task: AuthoredTask | TaskMaterialPlan,
    *,
    fetch_repository: RepositoryFetcher = _fetch_repository,
) -> WorkspaceSnapshot:
    """Overlay pinned public material for grounded task authoring."""

    files = {item.path: item for item in seed.files}
    for material in task.materials:
        fetched = fetch_repository(material)
        destination = PurePosixPath(material.destination)
        for item in fetched.files:
            path = destination.joinpath(item.path).as_posix()
            if path in files:
                raise ValueError(f"prepared workspace material collision at {path}")
            files[path] = WorkspaceFile(path, item.content, item.mode)
    return WorkspaceSnapshot(tuple(files.values()))


def prepare_task_workspace(
    seed: WorkspaceSnapshot,
    task: AuthoredTask | TaskMaterialPlan,
    *,
    fetch_repository: RepositoryFetcher = _fetch_repository,
) -> WorkspaceSnapshot:
    """Return the one common workspace snapshot from which A and B are forked."""

    if task.setup_mode == "agent_bootstrap":
        return seed
    return prepare_task_context(seed, task, fetch_repository=fetch_repository)


def validate_task_workspace(workspace: WorkspaceSnapshot, task: AuthoredTask) -> None:
    """Reject authored file assumptions that the prepared snapshot cannot satisfy."""

    available = frozenset(workspace.paths)
    missing = tuple(path for path in task.required_files if path not in available)
    if missing:
        raise TaskPreflightError(
            f"prepared workspace is missing required files: {', '.join(missing)}"
        )
