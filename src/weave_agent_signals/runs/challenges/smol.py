"""Mount-free local Smol Machines execution for paired A/B agent arms."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import plistlib
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

from weave_agent_signals.runs.challenges.contracts import (
    ArmName,
    ArmResult,
    AuthoredTask,
    TaskPreflightError,
)
from weave_agent_signals.runs.challenges.environment import CapturedEnvironment, RuntimeFile
from weave_agent_signals.runs.challenges.workspace import (
    DEFAULT_WORKSPACE_EXCLUSIONS,
    MAX_WORKSPACE_FILE_BYTES,
    MAX_WORKSPACE_FILES,
    MAX_WORKSPACE_TOTAL_BYTES,
    WorkspaceFile,
    WorkspaceSnapshot,
    describe_workspace_changes,
)

_WORKSPACE = "/workspace"
_SOURCE_ARCHIVE = "/tmp/workspace.tar"
_RUNTIME_ARCHIVE = "/tmp/runtime-files.tar"
_FINAL_ARCHIVE = "/root/.monopole-final-workspace.tar"
_MAX_TRANSCRIPT_CHARACTERS = 200_000
_MAX_PROCESS_OUTPUT_BYTES = 200_000
# Smol CLI 1.6.6 held each finished guest command open for seconds; 1.6.13 returns at once
# (docs/upstream-smol-issues.md).
_MIN_SMOL_CLI_VERSION = (1, 6, 13)
# Files go into a machine as base64 in secret env vars and come back out through dd, in 90 KiB
# chunks, each step retried: Smol CLI 1.6.13's stdin upload and `machine cp` can report success
# without writing the file, and an exec can fail intermittently without a diagnostic
# (docs/upstream-smol-issues.md).
_FILE_CHUNK_BYTES = 90 * 1024
_FILE_WRITE_BATCH_CHUNKS = 6
_FILE_TRANSFER_ATTEMPTS = 3
_TRANSFER_GUEST_ENV = "MONOPOLE_TRANSFER_CHUNK"
# The CLI enforces `--timeout` itself; the host stops waiting this long after it, in case the
# CLI hangs.
_HOST_TIMEOUT_GRACE_SECONDS = 60
_MAX_PROCESS_CAPTURE_BYTES = 32 * 1024 * 1024
_MAX_PROCESS_CAPTURE_FILE_BYTES = _MAX_PROCESS_CAPTURE_BYTES // 2
_MAX_WORKSPACE_ARCHIVE_BYTES = 320 * 1024 * 1024
# Each harness's own session files: Codex rollouts, and Claude Code's <project>/<session>.jsonl,
# beside which older versions keep subagent transcripts named agent-*.jsonl.
_SESSION_FILES = {
    "codex": 'find "$HOME/.codex/sessions" -name "rollout-*.jsonl"',
    "claude": (
        'find "$HOME/.claude/projects" -mindepth 2 -maxdepth 2 -name "*.jsonl" ! -name "agent-*"'
    ),
}
_MAX_SESSION_BYTES = 16 * 1024 * 1024
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionOptions:
    env: dict[str, str]
    workdir: str
    timeout: int
    cancel_requested: Callable[[], bool] = lambda: False


class ChallengeCancelled(RuntimeError):
    """The owning evaluation run cancelled paired agent execution."""


class ExecResult(Protocol):
    exit_code: int
    stdout: str
    stderr: str


class Machine(Protocol):
    def write_file(
        self,
        path: str,
        data: bytes,
        mode: int | None = None,
        *,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> None: ...

    def exec(self, command: list[str], options: ExecutionOptions | None = None) -> ExecResult: ...

    def read_file(
        self, path: str, *, cancel_requested: Callable[[], bool] = lambda: False
    ) -> bytes: ...

    def delete(self) -> None: ...


class MachineFactory(Protocol):
    def __call__(self, name: str, image: str, network_enabled: bool) -> Machine: ...


@dataclass(frozen=True)
class _CliExecResult:
    exit_code: int
    stdout: str
    stderr: str


class _CliMachine:
    """Small adapter over the official self-contained local Smol CLI."""

    def __init__(
        self,
        executable: str,
        name: str,
        image: str,
        network_enabled: bool,
        memory_mib: int | None = None,
    ) -> None:
        self._executable = executable
        self._name = name
        self._data_dir: Path | None = None
        image_flag = "--from" if Path(image).is_file() else "--image"
        command = [executable, "machine", "create", "--name", name, image_flag, image]
        if network_enabled:
            command.append("--net")
        if memory_mib is not None:
            command.extend(("--mem", str(memory_mib)))
        try:
            created = self._run(command)
        except BaseException:
            self._cleanup_after_init_failure()  # a create that timed out may have made the machine
            raise
        if created.returncode != 0:
            self._cleanup_after_init_failure()
            raise RuntimeError(f"Smol machine creation failed: {_safe_process_error(created)}")
        try:
            self._data_dir = self._resolve_data_dir()
            started = self._run([executable, "machine", "start", "--name", name, "--local"])
        except BaseException:
            self._cleanup_after_init_failure()
            raise
        if started.returncode != 0:
            self._cleanup_after_init_failure()
            raise RuntimeError(f"Smol machine startup failed: {_safe_process_error(started)}")

    def _resolve_data_dir(self) -> Path | None:
        """The machine's cache directory on macOS, which `machine rm` can leave behind with its
        disk image still mounted. Anything but the expected cache path fails closed."""
        if not _macos():
            return None
        result = self._run([self._executable, "machine", "data-dir", "--name", self._name])
        if result.returncode != 0:
            raise RuntimeError(f"Smol machine data directory failed: {_safe_process_error(result)}")
        path = Path(result.stdout.strip()).resolve()
        root = (Path.home() / "Library" / "Caches" / "smolvm" / "vms").resolve()
        if path.parent != root or re.fullmatch(r"[0-9a-f]{16}", path.name) is None:
            raise RuntimeError("Smol returned an unexpected machine data directory")
        return path

    def _cleanup_after_init_failure(self) -> None:
        if self._data_dir is None:
            try:
                self._data_dir = self._resolve_data_dir()
            except Exception:
                pass
        try:
            self.delete()
        except Exception:
            pass

    @staticmethod
    def _run(command: list[str], *, env: dict[str, str] | None = None):
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

    _popen = staticmethod(subprocess.Popen)

    def _transfer(
        self,
        command: list[str],
        operation: str,
        *,
        secrets: dict[str, str] | None = None,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> ExecResult:
        """Run one file-transfer step, retrying a failed attempt."""
        for attempt in range(_FILE_TRANSFER_ATTEMPTS):
            _require_active(cancel_requested)
            try:
                result = self.exec(
                    command,
                    ExecutionOptions(
                        env=secrets or {},
                        workdir="/",
                        timeout=30,
                        cancel_requested=cancel_requested,
                    ),
                )
            except TimeoutError:
                result = _CliExecResult(124, "", "timed out")  # retried like any failed step
            _require_active(cancel_requested)
            if result.exit_code == 0:
                return result
            if attempt < _FILE_TRANSFER_ATTEMPTS - 1:
                time.sleep(0.1)
        raise RuntimeError(f"Smol file {operation} exited {result.exit_code}: {result.stderr}")

    def write_file(
        self,
        path: str,
        data: bytes,
        mode: int | None = None,
        *,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        self._transfer(
            ["sh", "-c", 'umask 077; : > "$1"', "sh", path],
            "creation",
            cancel_requested=cancel_requested,
        )
        batch = _FILE_CHUNK_BYTES * _FILE_WRITE_BATCH_CHUNKS
        for batch_start in range(0, len(data), batch):
            commands, secrets = ["set -e"], {}
            for offset in range(
                batch_start, min(len(data), batch_start + batch), _FILE_CHUNK_BYTES
            ):
                name = f"{_TRANSFER_GUEST_ENV}_{len(secrets)}"
                secrets[name] = base64.b64encode(data[offset : offset + _FILE_CHUNK_BYTES]).decode()
                commands.append(
                    f'printf %s "${name}" | base64 -d | dd of="$1" bs={_FILE_CHUNK_BYTES} '
                    f"seek={offset // _FILE_CHUNK_BYTES} conv=notrunc status=none"
                )
            self._transfer(
                ["sh", "-c", "; ".join(commands), "sh", path],
                "copy",
                secrets=secrets,
                cancel_requested=cancel_requested,
            )
        if mode is not None:
            self._transfer(["chmod", f"{mode:o}", path], "mode", cancel_requested=cancel_requested)

    def exec(self, command: list[str], options: ExecutionOptions | None = None) -> ExecResult:
        argv = [self._executable, "machine", "exec", "--name", self._name, "--local"]
        child_env = dict(os.environ)
        if options is not None:
            for index, (name, value) in enumerate(sorted(options.env.items())):
                host_name = f"MONOPOLE_SMOL_ENV_{index}"
                child_env[host_name] = value
                argv.extend(("--secret-env", f"{name}={host_name}"))
            argv.extend(("--workdir", options.workdir, "--timeout", str(options.timeout)))
        argv.append("--")
        argv.extend(command)
        if options is None:
            result = self._run(argv, env=child_env)
        else:
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                file_limit_blocks = max(1, _MAX_PROCESS_CAPTURE_FILE_BYTES // 512)
                limited_argv = [
                    "/bin/sh",
                    "-c",
                    f'ulimit -f {file_limit_blocks}; exec "$@"',
                    "sh",
                    *argv,
                ]
                process = self._popen(
                    limited_argv,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=child_env,
                )
                deadline = time.monotonic() + options.timeout + _HOST_TIMEOUT_GRACE_SECONDS
                while True:
                    try:
                        process.communicate(timeout=0.25)
                        if _capture_limit_reached(stdout_file, stderr_file):
                            raise RuntimeError("Smol agent process exceeded the host capture limit")
                        result = subprocess.CompletedProcess(
                            limited_argv,
                            process.returncode,
                            _read_tail(stdout_file),
                            _read_tail(stderr_file),
                        )
                        break
                    except subprocess.TimeoutExpired:
                        if _capture_limit_reached(stdout_file, stderr_file):
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            raise RuntimeError("Smol agent process exceeded the host capture limit")
                        cancelled = options.cancel_requested()
                        if not cancelled and time.monotonic() < deadline:
                            continue
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        if cancelled:
                            raise ChallengeCancelled()
                        raise TimeoutError("Smol CLI did not return after its timeout")
        if result.returncode == 124:
            raise TimeoutError("Smol agent execution reached its timeout")
        return _CliExecResult(result.returncode, result.stdout, result.stderr)

    def read_file(
        self, path: str, *, cancel_requested: Callable[[], bool] = lambda: False
    ) -> bytes:
        size = self._transfer(["stat", "-c", "%s", path], "size", cancel_requested=cancel_requested)
        try:
            byte_count = int(size.stdout.strip())
        except ValueError as exc:
            raise RuntimeError("Smol file size could not be determined") from exc
        if byte_count < 0:
            raise RuntimeError("Smol file size cannot be negative")
        if byte_count > _MAX_WORKSPACE_ARCHIVE_BYTES:
            raise RuntimeError("Smol file exceeds the maximum captured archive size")
        value = bytearray()
        for index in range(0, byte_count, _FILE_CHUNK_BYTES):
            chunk = self._transfer(
                [
                    "sh",
                    "-c",
                    f'dd if="$1" bs={_FILE_CHUNK_BYTES} skip="$2" count=1 status=none | base64',
                    "sh",
                    path,
                    str(index // _FILE_CHUNK_BYTES),
                ],
                "read",
                cancel_requested=cancel_requested,
            )
            try:
                value.extend(base64.b64decode("".join(chunk.stdout.split()), validate=True))
            except ValueError as exc:
                raise RuntimeError("Smol file read returned invalid base64") from exc
        if len(value) != byte_count:
            raise RuntimeError("Smol file read returned a truncated archive")
        return bytes(value)

    def delete(self) -> None:
        try:
            self._run([self._executable, "machine", "stop", "--name", self._name, "--local"])
        except Exception:
            pass  # removal still runs; a machine that won't stop is removed anyway
        deleted = self._run(
            [
                self._executable,
                "machine",
                "rm",
                "--name",
                self._name,
                "--yes",
                "--local",
            ]
        )
        if deleted.returncode != 0:
            raise RuntimeError(f"Smol machine cleanup failed: {_safe_process_error(deleted)}")
        if self._data_dir is not None and self._data_dir.exists():
            _cleanup_orphaned_macos_data_dir(self._data_dir)


def _macos() -> bool:
    return sys.platform == "darwin"


def _cleanup_orphaned_macos_data_dir(data_dir: Path) -> None:
    """Detach the deleted machine's disk image, which Smol CLI 1.6.13 leaves mounted on macOS,
    then remove its cache directory."""
    image_path = (data_dir / "pack" / "layers-cs.sparseimage").resolve()
    info = subprocess.run(
        ["hdiutil", "info", "-plist"], check=True, capture_output=True, timeout=30
    )
    for image in plistlib.loads(info.stdout).get("images", []):
        if Path(image.get("image-path", "")).resolve() != image_path:
            continue
        devices = [
            entity["dev-entry"]
            for entity in image.get("system-entities", [])
            if entity.get("dev-entry")
        ]
        if not devices:
            raise RuntimeError("Smol orphaned disk image has no device")
        for attempt in range(10):
            detached = subprocess.run(
                ["hdiutil", "detach", devices[0], "-force"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if detached.returncode == 0:
                break
            if attempt < 9:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"Smol disk cleanup failed: {_safe_process_error(detached)}")
        break
    shutil.rmtree(data_dir)


def _safe_process_error(result) -> str:
    value = result.stderr or result.stdout
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return " ".join(value.split())[:500]


def _read_tail(stream) -> str:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - _MAX_PROCESS_OUTPUT_BYTES))
    return stream.read().decode("utf-8", errors="replace")


def _capture_limit_reached(*streams) -> bool:
    sizes = tuple(os.fstat(stream.fileno()).st_size for stream in streams)
    return any(size >= _MAX_PROCESS_CAPTURE_FILE_BYTES for size in sizes) or (
        sum(sizes) > _MAX_PROCESS_CAPTURE_BYTES
    )


def _smol_executable() -> str:
    executable = shutil.which("smol") or os.path.expanduser("~/.local/bin/smol")
    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        raise RuntimeError("Smol CLI is unavailable; install it from smolmachines.com")
    version = subprocess.run(
        [executable, "--version"], check=False, capture_output=True, text=True, timeout=10
    )
    if version.returncode != 0:
        raise RuntimeError("Smol CLI version could not be resolved")
    _validate_smol_cli_version(version.stdout.strip())
    return executable


def _validate_smol_cli_version(value: str) -> None:
    match = re.match(r"smol (\d+)\.(\d+)\.(\d+)\b", value)  # 1.32 appends "(engine smolvm ...)"
    if match is None or tuple(map(int, match.groups())) < _MIN_SMOL_CLI_VERSION:
        raise RuntimeError("Smol CLI 1.6.13 or newer is required")


def _create_machine(
    name: str, image: str, network_enabled: bool, memory_mib: int | None = None
) -> Machine:
    return _CliMachine(_smol_executable(), name, image, network_enabled, memory_mib)


def _task_text(task: AuthoredTask) -> str:
    package = {
        "setup_mode": task.setup_mode,
        "materials": [item.to_dict() for item in task.materials],
        "start_checks": list(task.start_checks),
        "judging_criteria": list(task.judging_criteria),
        "required_files": list(task.required_files),
        "required_executables": list(task.required_executables),
        "requires_git_metadata": task.requires_git_metadata,
    }
    return (
        f"PROMPT:\n{task.prompt}\n\nGOAL:\n{task.goal}\n\n"
        f"TASK PACKAGE:\n{json.dumps(package, sort_keys=True)}\n\n"
        "Work autonomously in the supplied environment until you have achieved the goal. "
        "Verify the result, summarize the evidence, and then exit."
    )


def _command(template: tuple[str, ...], task: AuthoredTask) -> list[str]:
    rendered = _task_text(task)
    if template.count("{task}") != 1:
        raise ValueError("harness command must contain exactly one task placeholder")
    return [rendered if item == "{task}" else item for item in template]


def _safe_error(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return (text or type(error).__name__)[:500]


def _require_active(cancel_requested: Callable[[], bool]) -> None:
    if cancel_requested():
        raise ChallengeCancelled()


def _is_execution_timeout(error: BaseException) -> bool:
    return isinstance(error, TimeoutError) or getattr(error, "code", None) == "TIMEOUT"


def _session_transcript(machine: Machine, environment: CapturedEnvironment) -> str | None:
    """The arm's harness session file, read so the arm can be logged to CES.

    Each arm runs one session in a fresh machine; the newest session file is the arm's. A
    missing, oversized, or unreadable file leaves the arm unlogged, never failed."""
    try:
        found = machine.exec(
            [
                "/bin/sh",
                "-c",
                f"{_SESSION_FILES[environment.identity.harness]} -exec ls -t {{}} + 2>/dev/null"
                " | head -1",
            ],
            ExecutionOptions(env=dict(environment.runtime_env), workdir=_WORKSPACE, timeout=60),
        )
        path = found.stdout.strip()
        if found.exit_code != 0 or not path:
            return None
        data = machine.read_file(path)
    except Exception as error:
        _LOG.warning("arm session file could not be read: %s", _safe_error(error))
        return None
    if len(data) > _MAX_SESSION_BYTES:
        _LOG.warning("arm session file is %d bytes, over the logging limit", len(data))
        return None
    return data.decode("utf-8", errors="replace")


def _transcript(stdout: str, stderr: str) -> tuple[str, str]:
    combined = stdout
    if stderr:
        combined += ("\n" if combined else "") + stderr
    digest = f"sha256:{hashlib.sha256(combined.encode()).hexdigest()}"
    return combined[-_MAX_TRANSCRIPT_CHARACTERS:], digest


def _snapshot_from_archive(value: bytes) -> WorkspaceSnapshot:
    if len(value) > _MAX_WORKSPACE_ARCHIVE_BYTES:
        raise ValueError("final workspace archive exceeds the maximum captured size")
    files: list[WorkspaceFile] = []
    total_bytes = 0
    with tarfile.open(fileobj=io.BytesIO(value), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError("final workspace archive contains a non-file entry")
            if len(files) >= MAX_WORKSPACE_FILES:
                raise ValueError("final workspace archive exceeds the maximum file count")
            if member.size > MAX_WORKSPACE_FILE_BYTES:
                raise ValueError("final workspace archive contains an oversized file")
            total_bytes += member.size
            if total_bytes > MAX_WORKSPACE_TOTAL_BYTES:
                raise ValueError("final workspace archive exceeds the maximum total size")
            name = member.name.removeprefix("./")
            if not name or name.startswith("/"):
                raise ValueError("final workspace archive contains an unsafe path")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("final workspace archive file is unreadable")
            files.append(
                WorkspaceFile(
                    name,
                    source.read(),
                    stat.S_IMODE(member.mode),
                )
            )
    return WorkspaceSnapshot(tuple(files))


def _runtime_archive(files: tuple[RuntimeFile, ...]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        for runtime_file in sorted(files, key=lambda item: item.path):
            info = tarfile.TarInfo(runtime_file.path.removeprefix("/"))
            info.size = len(runtime_file.content)
            info.mode = runtime_file.mode
            archive.addfile(info, io.BytesIO(runtime_file.content))
    return buffer.getvalue()


class SmolMachineRunner:
    """Run both arms concurrently and classify agent versus infrastructure outcomes."""

    def __init__(
        self, *, machine_factory: MachineFactory | None = None, memory_mib: int | None = None
    ) -> None:
        self._machine_factory = machine_factory or partial(_create_machine, memory_mib=memory_mib)

    def preflight(
        self,
        *,
        workspace: WorkspaceSnapshot,
        task: AuthoredTask,
        environment: CapturedEnvironment,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        """Verify the common image and declared capabilities before either arm runs."""

        if task.workspace_digest != environment.identity.workspace_digest:
            raise ValueError("task and execution workspace digests do not match")
        machine: Machine | None = None
        name = f"preflight-{task.task_id.removeprefix('sha256:')[:12]}-{secrets.token_hex(4)}"
        try:
            _require_active(cancel_requested)
            machine = self._machine_factory(
                name,
                environment.identity.image,
                environment.identity.network_enabled,
            )
            machine.write_file(
                _SOURCE_ARCHIVE, workspace.archive, 0o600, cancel_requested=cancel_requested
            )
            if environment.runtime_files:
                machine.write_file(
                    _RUNTIME_ARCHIVE,
                    _runtime_archive(environment.runtime_files),
                    0o600,
                    cancel_requested=cancel_requested,
                )
                extracted = machine.exec(["tar", "-xf", _RUNTIME_ARCHIVE, "-C", "/"])
                if extracted.exit_code != 0:
                    raise RuntimeError(
                        f"runtime file setup exited {extracted.exit_code}: {extracted.stderr}"
                    )
            setup = machine.exec(["mkdir", "-p", _WORKSPACE])
            if setup.exit_code != 0:
                raise RuntimeError(f"workspace setup exited {setup.exit_code}: {setup.stderr}")
            unpack = machine.exec(["tar", "-xf", _SOURCE_ARCHIVE, "-C", _WORKSPACE])
            if unpack.exit_code != 0:
                raise RuntimeError(f"workspace unpack exited {unpack.exit_code}: {unpack.stderr}")
            version = machine.exec([environment.identity.harness, "--version"])
            if (
                version.exit_code != 0
                or version.stdout.strip() != environment.identity.harness_version
            ):
                raise RuntimeError("guest harness version does not match the pinned host version")
            options = ExecutionOptions(
                env=dict(environment.runtime_env),
                workdir=_WORKSPACE,
                timeout=min(60, environment.identity.timeout_seconds),
                cancel_requested=cancel_requested,
            )
            for executable in task.required_executables:
                available = machine.exec(
                    ["/bin/sh", "-c", 'command -v "$1" >/dev/null', "sh", executable],
                    options,
                )
                if available.exit_code != 0:
                    raise TaskPreflightError(
                        f"required sandbox executable is unavailable: {executable}"
                    )
        finally:
            if machine is not None:
                try:
                    machine.delete()
                except Exception as exc:
                    _LOG.warning("Smol preflight cleanup failed for %s: %s", name, _safe_error(exc))

    def run_pair(
        self,
        *,
        baseline: WorkspaceSnapshot,
        candidate: WorkspaceSnapshot,
        task: AuthoredTask,
        environment: CapturedEnvironment,
        baseline_runtime_files: tuple[RuntimeFile, ...] = (),
        candidate_runtime_files: tuple[RuntimeFile, ...] = (),
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> tuple[ArmResult, ArmResult]:
        if task.workspace_digest != environment.identity.workspace_digest:
            raise ValueError("task and execution workspace digests do not match")
        pair_name = f"paired-{task.task_id.removeprefix('sha256:')[:12]}-{secrets.token_hex(4)}"
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="paired-sandbox") as executor:
            baseline_future = executor.submit(
                self._run_arm,
                "baseline",
                pair_name,
                baseline,
                task,
                environment,
                baseline_runtime_files,
                cancel_requested,
            )
            candidate_future = executor.submit(
                self._run_arm,
                "candidate",
                pair_name,
                candidate,
                task,
                environment,
                candidate_runtime_files,
                cancel_requested,
            )
            return baseline_future.result(), candidate_future.result()

    def _run_arm(
        self,
        arm: ArmName,
        pair_name: str,
        snapshot: WorkspaceSnapshot,
        task: AuthoredTask,
        environment: CapturedEnvironment,
        instruction_runtime_files: tuple[RuntimeFile, ...],
        cancel_requested: Callable[[], bool],
    ) -> ArmResult:
        started = time.monotonic()
        machine: Machine | None = None
        try:
            _require_active(cancel_requested)
            name = f"{pair_name}-{arm}"
            machine = self._machine_factory(
                name,
                environment.identity.image,
                environment.identity.network_enabled,
            )
            _require_active(cancel_requested)
            machine.write_file(
                _SOURCE_ARCHIVE, snapshot.archive, 0o600, cancel_requested=cancel_requested
            )
            _require_active(cancel_requested)
            common_paths = {item.path for item in environment.runtime_files}
            instruction_paths = {item.path for item in instruction_runtime_files}
            if len(instruction_paths) != len(instruction_runtime_files):
                raise ValueError("arm runtime instruction paths must be unique")
            if common_paths & instruction_paths:
                raise ValueError("arm instructions cannot replace common runtime files")
            runtime_files = (*environment.runtime_files, *instruction_runtime_files)
            if runtime_files:
                machine.write_file(
                    _RUNTIME_ARCHIVE,
                    _runtime_archive(runtime_files),
                    0o600,
                    cancel_requested=cancel_requested,
                )
                extracted = machine.exec(["tar", "-xf", _RUNTIME_ARCHIVE, "-C", "/"])
                if extracted.exit_code != 0:
                    raise RuntimeError(
                        f"runtime file setup exited {extracted.exit_code}: {extracted.stderr}"
                    )
                _require_active(cancel_requested)
            setup = machine.exec(["mkdir", "-p", _WORKSPACE])
            if setup.exit_code != 0:
                raise RuntimeError(f"workspace setup exited {setup.exit_code}: {setup.stderr}")
            unpack = machine.exec(["tar", "-xf", _SOURCE_ARCHIVE, "-C", _WORKSPACE])
            if unpack.exit_code != 0:
                raise RuntimeError(f"workspace unpack exited {unpack.exit_code}: {unpack.stderr}")
            _require_active(cancel_requested)
            version = machine.exec([environment.identity.harness, "--version"])
            if (
                version.exit_code != 0
                or version.stdout.strip() != environment.identity.harness_version
            ):
                raise RuntimeError("guest harness version does not match the pinned host version")
            _require_active(cancel_requested)

            status = "exited"
            exit_code: int | None
            stdout = ""
            stderr = ""
            try:
                result = machine.exec(
                    _command(environment.identity.command, task),
                    ExecutionOptions(
                        env=dict(environment.runtime_env),
                        workdir=_WORKSPACE,
                        timeout=environment.identity.timeout_seconds,
                        cancel_requested=cancel_requested,
                    ),
                )
                exit_code = result.exit_code
                stdout = result.stdout
                stderr = result.stderr
            except ChallengeCancelled:
                raise
            except Exception as exc:
                if not _is_execution_timeout(exc):
                    raise
                status = "timed_out"
                exit_code = None
                stderr = "Agent process reached the pinned execution timeout."

            _require_active(cancel_requested)
            exclusions = [
                pattern
                for name in sorted(DEFAULT_WORKSPACE_EXCLUSIONS)
                for pattern in (f"--exclude={name}", f"--exclude=*/{name}")
            ]
            packed = machine.exec(
                ["tar", "-cf", _FINAL_ARCHIVE, "-C", _WORKSPACE, *exclusions, "."]
            )
            if packed.exit_code != 0:
                raise RuntimeError(f"artifact capture exited {packed.exit_code}: {packed.stderr}")
            final = _snapshot_from_archive(
                machine.read_file(_FINAL_ARCHIVE, cancel_requested=cancel_requested)
            )
            transcript, transcript_digest = _transcript(stdout, stderr)
            session_transcript = _session_transcript(machine, environment)
            return ArmResult(
                arm=arm,
                status=status,
                exit_code=exit_code,
                duration_seconds=time.monotonic() - started,
                initial_workspace_digest=snapshot.digest,
                final_workspace_digest=final.digest,
                transcript=transcript,
                transcript_digest=transcript_digest,
                artifact_digests={item.path: item.digest for item in final.files},
                artifact_changes=describe_workspace_changes(snapshot, final),
                session_transcript=session_transcript,
            )
        except ChallengeCancelled:
            raise
        except Exception as exc:
            transcript, transcript_digest = _transcript("", _safe_error(exc))
            return ArmResult(
                arm=arm,
                status="infrastructure_failed",
                exit_code=None,
                duration_seconds=time.monotonic() - started,
                initial_workspace_digest=snapshot.digest,
                final_workspace_digest=None,
                transcript=transcript,
                transcript_digest=transcript_digest,
                artifact_digests={},
                infrastructure_error=_safe_error(exc),
            )
        finally:
            if machine is not None:
                try:
                    machine.delete()
                except Exception as exc:
                    _LOG.warning("Smol machine cleanup failed for %s: %s", name, _safe_error(exc))
