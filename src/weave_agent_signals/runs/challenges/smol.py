"""Mount-free local Smol Machines execution for paired B/C agent arms."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import secrets
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from weave_agent_signals.runs.challenges.contracts import ArmName, ArmResult, AuthoredTask
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
_MAX_PROCESS_CAPTURE_BYTES = 32 * 1024 * 1024
_MAX_PROCESS_CAPTURE_FILE_BYTES = _MAX_PROCESS_CAPTURE_BYTES // 2
_MAX_WORKSPACE_ARCHIVE_BYTES = 320 * 1024 * 1024
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
    def write_file(self, path: str, data: bytes, mode: int | None = None) -> None: ...

    def exec(self, command: list[str], options: ExecutionOptions | None = None) -> ExecResult: ...

    def read_file(self, path: str) -> bytes: ...

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

    def __init__(self, executable: str, name: str, image: str, network_enabled: bool) -> None:
        self._executable = executable
        self._name = name
        image_flag = "--from" if Path(image).is_file() else "--image"
        command = [executable, "machine", "create", "--name", name, image_flag, image]
        if network_enabled:
            command.append("--net")
        created = self._run(command)
        if created.returncode != 0:
            raise RuntimeError(f"Smol machine creation failed: {_safe_process_error(created)}")
        try:
            started = self._run([executable, "machine", "start", "--name", name, "--local"])
        except BaseException:
            try:
                self._run([executable, "machine", "rm", "--name", name, "--yes", "--local"])
            except Exception:
                pass
            raise
        if started.returncode != 0:
            self._run([executable, "machine", "rm", "--name", name, "--yes", "--local"])
            raise RuntimeError(f"Smol machine startup failed: {_safe_process_error(started)}")

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

    @staticmethod
    def _run_binary(command: list[str], *, input_data: bytes | None = None):
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            input=input_data,
            timeout=300,
        )

    _popen = staticmethod(subprocess.Popen)

    def write_file(self, path: str, data: bytes, mode: int | None = None) -> None:
        copied = self._run_binary(
            [
                self._executable,
                "machine",
                "exec",
                "--name",
                self._name,
                "--local",
                "--interactive",
                "--",
                "dd",
                f"of={path}",
                "status=none",
            ],
            input_data=data,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"Smol file copy failed: {_safe_process_error(copied)}")
        if mode is not None:
            changed = self.exec(["chmod", f"{mode:o}", path])
            if changed.exit_code != 0:
                raise RuntimeError(f"Smol file mode failed: {changed.stderr}")

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
                        if not options.cancel_requested():
                            continue
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise ChallengeCancelled()
        if result.returncode == 124:
            raise TimeoutError("Smol agent execution reached its timeout")
        return _CliExecResult(result.returncode, result.stdout, result.stderr)

    def read_file(self, path: str) -> bytes:
        size = self.exec(["stat", "-c", "%s", path])
        if size.exit_code != 0:
            raise RuntimeError(f"Smol file size failed: {size.stderr}")
        try:
            byte_count = int(size.stdout.strip())
        except ValueError as exc:
            raise RuntimeError("Smol file size could not be determined") from exc
        if byte_count < 0:
            raise RuntimeError("Smol file size cannot be negative")
        if byte_count > _MAX_WORKSPACE_ARCHIVE_BYTES:
            raise RuntimeError("Smol file exceeds the maximum captured archive size")
        with tempfile.TemporaryDirectory(prefix="monopole-smol-copy-") as directory:
            destination = Path(directory) / "captured"
            copied = self._run(
                [
                    self._executable,
                    "machine",
                    "cp",
                    "--local",
                    f"{self._name}:{path}",
                    str(destination),
                ]
            )
            if copied.returncode != 0:
                raise RuntimeError(f"Smol file copy failed: {_safe_process_error(copied)}")
            value = destination.read_bytes()
        if len(value) != byte_count:
            raise RuntimeError("Smol file read returned a truncated archive")
        return value

    def delete(self) -> None:
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
    discovered = shutil.which("smol")
    if discovered is not None:
        return discovered
    installed = os.path.expanduser("~/.local/bin/smol")
    if os.path.isfile(installed) and os.access(installed, os.X_OK):
        return installed
    raise RuntimeError("Smol CLI is unavailable; install it from smolmachines.com")


def _create_machine(name: str, image: str, network_enabled: bool) -> Machine:
    return _CliMachine(_smol_executable(), name, image, network_enabled)


def _task_text(task: AuthoredTask) -> str:
    package = {
        "setup_mode": task.setup_mode,
        "materials": [item.to_dict() for item in task.materials],
        "start_checks": list(task.start_checks),
        "judging_criteria": list(task.judging_criteria),
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

    def __init__(self, *, machine_factory: MachineFactory = _create_machine) -> None:
        self._machine_factory = machine_factory

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
            machine.write_file(_SOURCE_ARCHIVE, snapshot.archive, 0o600)
            _require_active(cancel_requested)
            common_paths = {item.path for item in environment.runtime_files}
            instruction_paths = {item.path for item in instruction_runtime_files}
            if len(instruction_paths) != len(instruction_runtime_files):
                raise ValueError("arm runtime instruction paths must be unique")
            if common_paths & instruction_paths:
                raise ValueError("arm instructions cannot replace common runtime files")
            runtime_files = (*environment.runtime_files, *instruction_runtime_files)
            if runtime_files:
                machine.write_file(_RUNTIME_ARCHIVE, _runtime_archive(runtime_files), 0o600)
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
            final = _snapshot_from_archive(machine.read_file(_FINAL_ARCHIVE))
            transcript, transcript_digest = _transcript(stdout, stderr)
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
