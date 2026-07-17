"""Capture one exact, secret-safe execution identity for both comparison arms."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType

from weave_agent_signals.runs.challenges.contracts import (
    ExecutionIdentity,
    NamedDigest,
)

VersionReader = Callable[[list[str]], str]


@dataclass(frozen=True)
class RuntimeFile:
    path: str
    content: bytes
    mode: int = 0o600

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("runtime file target must be an absolute guest path")
        if path == PurePosixPath("/workspace") or PurePosixPath("/workspace") in path.parents:
            raise ValueError("runtime files must remain outside /workspace")
        if not isinstance(self.content, bytes):
            raise ValueError("runtime file content must be bytes")
        if type(self.mode) is not int or not 0 <= self.mode <= 0o777:
            raise ValueError("runtime file mode must be a Unix permission mode")

    @property
    def digest(self) -> str:
        return f"sha256:{hashlib.sha256(self.content).hexdigest()}"


@dataclass(frozen=True)
class CapturedEnvironment:
    identity: ExecutionIdentity
    runtime_env: Mapping[str, str]
    runtime_files: tuple[RuntimeFile, ...] = ()

    def __post_init__(self) -> None:
        values = dict(self.runtime_env)
        if any(not isinstance(key, str) or not key for key in values):
            raise ValueError("runtime environment names must be nonblank strings")
        if any(not isinstance(value, str) for value in values.values()):
            raise ValueError("runtime environment values must be strings")
        files = tuple(sorted(self.runtime_files, key=lambda item: item.path))
        if any(not isinstance(item, RuntimeFile) for item in files):
            raise ValueError("runtime files must be RuntimeFile values")
        paths = tuple(item.path for item in files)
        if len(paths) != len(set(paths)):
            raise ValueError("runtime file targets must be unique")
        object.__setattr__(self, "runtime_env", MappingProxyType(values))
        object.__setattr__(self, "runtime_files", files)


def _read_version(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} --version exited {result.returncode}")
    return result.stdout.strip()


def _one_value(values: set[str | None], label: str) -> str | None:
    if len(values) != 1:
        raise ValueError(f"paired verification requires one {label}")
    return next(iter(values))


def _command(
    harness: str,
    model: str,
    effort: str | None,
    *,
    network_enabled: bool,
) -> tuple[str, ...]:
    if harness == "codex":
        command = ["codex"]
        if network_enabled:
            command.append("--search")
        command.extend(("exec", "--model", model))
        if effort is not None:
            command.extend(("-c", f'model_reasoning_effort="{effort}"'))
        command.extend(
            (
                "--sandbox",
                "workspace-write",
                "--skip-git-repo-check",
                "--ephemeral",
                "--json",
                "{task}",
            )
        )
        return tuple(command)
    command = [
        "claude",
        "--print",
        "--model",
        model,
    ]
    if effort is not None:
        command.extend(("--effort", effort))
    command.extend(("--output-format", "stream-json", "--no-session-persistence", "{task}"))
    return tuple(command)


def capture_execution_environment(
    *,
    cohort: Mapping[str, object],
    workspace_digest: str,
    image: str,
    image_digest: str,
    timeout_seconds: int,
    runtime_env: Mapping[str, str],
    runtime_files: tuple[RuntimeFile, ...] = (),
    network_enabled: bool = True,
    version_reader: VersionReader = _read_version,
) -> CapturedEnvironment:
    """Resolve the single original agent identity and hash its runtime values."""

    turns = cohort.get("turns")
    if (
        not isinstance(turns, list)
        or not turns
        or any(not isinstance(item, Mapping) for item in turns)
    ):
        raise ValueError("pinned cohort must contain turn execution identities")
    models = {item.get("model") for item in turns}
    model = _one_value(models, "evaluated model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("paired verification requires one nonblank evaluated model")
    families = {item.get("model_family") for item in turns}
    family = _one_value(families, "evaluated model family")
    if family == "openai":
        harness = "codex"
    elif family == "anthropic":
        harness = "claude"
    else:
        raise ValueError(f"unsupported evaluated model family: {family}")
    efforts = {item.get("effort_level") for item in turns}
    effort = _one_value(efforts, "effort level")
    if effort is not None and (not isinstance(effort, str) or not effort.strip()):
        raise ValueError("effort level must be nonblank when present")

    version = version_reader([harness, "--version"])
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"{harness} version could not be resolved")
    values = dict(runtime_env)
    if any(not isinstance(key, str) or not key for key in values):
        raise ValueError("runtime environment names must be nonblank strings")
    if any(not isinstance(value, str) for value in values.values()):
        raise ValueError("runtime environment values must be strings")
    environment = tuple(
        NamedDigest(
            name=name,
            digest=f"sha256:{hashlib.sha256(value.encode()).hexdigest()}",
        )
        for name, value in sorted(values.items())
    )
    files = tuple(sorted(runtime_files, key=lambda item: item.path))
    runtime_file_digests = tuple(NamedDigest(name=item.path, digest=item.digest) for item in files)
    identity = ExecutionIdentity(
        model=model,
        model_family=family,
        harness=harness,
        harness_version=version.strip(),
        effort=effort,
        image=image,
        image_digest=image_digest,
        command=_command(harness, model, effort, network_enabled=network_enabled),
        timeout_seconds=timeout_seconds,
        network_enabled=network_enabled,
        environment=environment,
        runtime_files=runtime_file_digests,
        workspace_digest=workspace_digest,
    )
    return CapturedEnvironment(identity, values, files)
