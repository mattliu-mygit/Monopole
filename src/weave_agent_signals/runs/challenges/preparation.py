"""Deterministic preparation of public task materials before paired execution."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from weave_agent_signals.runs.challenges.contracts import (
    AuthoredTask,
    TaskMaterial,
    TaskMaterialPlan,
    TaskPreflightError,
)
from weave_agent_signals.runs.challenges.workspace import (
    WorkspaceFile,
    WorkspaceSnapshot,
    capture_workspace,
)

RepositoryFetcher = Callable[[TaskMaterial], WorkspaceSnapshot]


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
        environment = {
            "PATH": os.defpath,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        for stage, command in commands:
            result = subprocess.run(
                command,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
                env=environment,
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
    """Return the one common workspace snapshot from which B and C are forked."""

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
