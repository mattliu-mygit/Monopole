from __future__ import annotations

import base64
import io
import plistlib
import re
import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired

import pytest

from weave_agent_signals.runs.challenges import smol as smol_module
from weave_agent_signals.runs.challenges.contracts import AuthoredTask, TaskMaterial
from weave_agent_signals.runs.challenges.environment import (
    RuntimeFile,
    capture_execution_environment,
)
from weave_agent_signals.runs.challenges.smol import (
    ChallengeCancelled,
    ExecutionOptions,
    SmolMachineRunner,
    _CliMachine,
    _task_text,
)
from weave_agent_signals.runs.challenges.workspace import WorkspaceFile, WorkspaceSnapshot


@pytest.fixture(autouse=True)
def _not_macos(monkeypatch):
    """Keep the CLI adapter's macOS data-directory handling out of tests that don't ask for it."""
    monkeypatch.setattr(smol_module, "_macos", lambda: False)


@dataclass
class _ExecResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


class _FakeMachine:
    def __init__(
        self,
        name: str,
        archive: bytes,
        barrier: threading.Barrier,
        *,
        task_error: Exception | None = None,
        setup_exit: int = 0,
        missing_executables: frozenset[str] = frozenset(),
        session_path: str | None = "/root/.codex/sessions/rollout-arm.jsonl",
    ) -> None:
        self.name = name
        self.session_path = session_path
        self.archive = archive
        self.barrier = barrier
        self.task_error = task_error
        self.setup_exit = setup_exit
        self.missing_executables = missing_executables
        self.deleted = False
        self.writes: dict[str, bytes] = {}
        self.calls: list[tuple[list[str], object]] = []
        self.cancel_checks: list[tuple[str, object]] = []

    def write_file(self, path: str, data: bytes, mode=None, *, cancel_requested=None) -> None:
        self.writes[path] = data
        self.cancel_checks.append((path, cancel_requested))

    def exec(self, command: list[str], options=None):
        self.calls.append((command, options))
        if command[0] == "mkdir":
            return _ExecResult(exit_code=self.setup_exit, stderr="setup failed")
        if command == ["codex", "--version"]:
            return _ExecResult(stdout="codex-cli 0.144.1\n")
        if command[:4] == ["/bin/sh", "-c", 'command -v "$1" >/dev/null', "sh"]:
            return _ExecResult(exit_code=int(command[4] in self.missing_executables))
        if command[0] == "codex":
            self.barrier.wait(timeout=2)
            if options is not None and options.cancel_requested():
                raise ChallengeCancelled()
            if self.task_error is not None:
                raise self.task_error
            return _ExecResult(stdout=f"{self.name} completed and verified the goal")
        if command[0] == "tar":
            return _ExecResult()
        if command[:2] == ["/bin/sh", "-c"] and command[2].startswith(
            'find "$HOME/.codex/sessions"'
        ):
            return _ExecResult(stdout=f"{self.session_path}\n" if self.session_path else "")
        raise AssertionError(command)

    def read_file(self, path: str, *, cancel_requested=None) -> bytes:
        self.cancel_checks.append((path, cancel_requested))
        if path == self.session_path:
            return f'{{"session": "{self.name}"}}\n'.encode()
        assert path == "/root/.monopole-final-workspace.tar"
        return self.archive

    def delete(self) -> None:
        self.deleted = True


class _SdkTimeout(Exception):
    code = "TIMEOUT"


def _environment():
    return capture_execution_environment(
        cohort={
            "turns": [
                {
                    "model": "gpt-5.6-sol",
                    "model_family": "openai",
                    "effort_level": "high",
                }
            ]
        },
        workspace_digest="sha256:source",
        image="runtime:latest",
        image_digest=f"sha256:{'a' * 64}",
        timeout_seconds=60,
        runtime_env={"TOKEN": "secret", "PATH": "/usr/bin"},
        runtime_files=(RuntimeFile("/root/.codex/config.toml", b"same-config", 0o600),),
        version_reader=lambda _argv: "codex-cli 0.144.1",
    )


def _task() -> AuthoredTask:
    return AuthoredTask(
        prompt="Fix the parser.",
        goal="The parser regression test passes.",
        workspace_digest="sha256:source",
        author_model="claude-sonnet-5",
        author_backend="cli",
    )


def test_agent_task_text_includes_bootstrap_material_and_start_checks() -> None:
    task = AuthoredTask(
        prompt="Clone the fixture and fix the parser.",
        goal="Parser tests pass.",
        setup_mode="agent_bootstrap",
        materials=(
            TaskMaterial(
                url="https://github.com/example/parser.git",
                revision="a" * 40,
                destination="task",
            ),
        ),
        judging_criteria=("Parser tests pass.",),
        start_checks=("The workspace is writable.",),
        required_files=("README.md",),
        required_executables=("git",),
        requires_git_metadata=True,
        workspace_digest="sha256:source",
        author_model="author",
        author_backend="cli",
    )

    rendered = _task_text(task)

    assert "agent_bootstrap" in rendered
    assert "https://github.com/example/parser.git" in rendered
    assert "a" * 40 in rendered
    assert "task" in rendered
    assert "The workspace is writable." in rendered
    assert "required_files" in rendered
    assert "README.md" in rendered
    assert "required_executables" in rendered
    assert "requires_git_metadata" in rendered


def _snapshots() -> tuple[WorkspaceSnapshot, WorkspaceSnapshot]:
    baseline = WorkspaceSnapshot((WorkspaceFile("AGENTS.md", b"baseline\n"),))
    candidate = WorkspaceSnapshot((WorkspaceFile("AGENTS.md", b"candidate\n"),))
    return baseline, candidate


def _archive_files(value: bytes) -> dict[str, tuple[bytes, int]]:
    with tarfile.open(fileobj=io.BytesIO(value), mode="r:") as archive:
        return {
            member.name: (archive.extractfile(member).read(), member.mode)
            for member in archive.getmembers()
        }


def test_runner_executes_both_arms_concurrently_with_identical_runtime() -> None:
    baseline, candidate = _snapshots()
    barrier = threading.Barrier(2)
    machines: list[_FakeMachine] = []

    def factory(name: str, image: str, network_enabled: bool):
        assert image == "runtime:latest"
        assert network_enabled is True
        snapshot = baseline if name.endswith("baseline") else candidate
        machine = _FakeMachine(name, snapshot.archive, barrier)
        machines.append(machine)
        return machine

    results = SmolMachineRunner(machine_factory=factory).run_pair(
        baseline=baseline,
        candidate=candidate,
        task=_task(),
        environment=_environment(),
        baseline_runtime_files=(
            RuntimeFile("/root/.codex/AGENTS.md", b"baseline instructions", 0o600),
        ),
        candidate_runtime_files=(
            RuntimeFile("/root/.codex/AGENTS.md", b"candidate instructions", 0o600),
        ),
    )

    assert [result.arm for result in results] == ["baseline", "candidate"]
    assert all(result.status == "exited" for result in results)
    assert [result.session_transcript for result in results] == [
        f'{{"session": "{machine.name}"}}\n' for machine in machines
    ]
    assert all("session_transcript" not in result.to_dict() for result in results)
    assert all(machine.deleted for machine in machines)
    assert all(machine.writes["/tmp/workspace.tar"] for machine in machines)
    assert {
        machine.name.rsplit("-", 1)[-1]: _archive_files(machine.writes["/tmp/runtime-files.tar"])
        for machine in machines
    } == {
        "baseline": {
            "root/.codex/AGENTS.md": (b"baseline instructions", 0o600),
            "root/.codex/config.toml": (b"same-config", 0o600),
        },
        "candidate": {
            "root/.codex/AGENTS.md": (b"candidate instructions", 0o600),
            "root/.codex/config.toml": (b"same-config", 0o600),
        },
    }
    task_calls = [
        (command, options)
        for machine in machines
        for command, options in machine.calls
        if command and command[0] == "codex" and command != ["codex", "--version"]
    ]
    assert len(task_calls) == 2
    assert task_calls[0][0] == task_calls[1][0]
    assert "Fix the parser." in task_calls[0][0][-1]
    assert "The parser regression test passes." in task_calls[0][0][-1]
    assert task_calls[0][1].env == task_calls[1][1].env
    assert task_calls[0][1].workdir == task_calls[1][1].workdir == "/workspace"
    packed_calls = [
        command
        for machine in machines
        for command, _options in machine.calls
        if command[:2] == ["tar", "-cf"]
    ]
    assert len(packed_calls) == 2
    assert all("--exclude=*/.venv" in command for command in packed_calls)
    assert all("--exclude=*/node_modules" in command for command in packed_calls)


def test_runner_leaves_an_arm_without_a_session_file_unlogged() -> None:
    baseline, candidate = _snapshots()
    barrier = threading.Barrier(2)

    def factory(name: str, _image: str, _network: bool):
        snapshot = baseline if name.endswith("baseline") else candidate
        return _FakeMachine(name, snapshot.archive, barrier, session_path=None)

    results = SmolMachineRunner(machine_factory=factory).run_pair(
        baseline=baseline,
        candidate=candidate,
        task=_task(),
        environment=_environment(),
    )

    assert [(result.status, result.session_transcript) for result in results] == [
        ("exited", None),
        ("exited", None),
    ]


def test_runner_preflight_rejects_missing_required_executable_before_arms() -> None:
    snapshot = WorkspaceSnapshot((WorkspaceFile("task/app.py", b"print('ready')\n"),))
    machines: list[_FakeMachine] = []

    def factory(name: str, _image: str, _network: bool):
        machine = _FakeMachine(
            name,
            snapshot.archive,
            threading.Barrier(1),
            missing_executables=frozenset({"node"}),
        )
        machines.append(machine)
        return machine

    task = AuthoredTask(
        prompt="Update task/app.py.",
        goal="The update is verified.",
        required_files=("task/app.py",),
        required_executables=("node",),
        workspace_digest="sha256:source",
        author_model="author",
        author_backend="cli",
    )

    with pytest.raises(RuntimeError, match="node"):
        SmolMachineRunner(machine_factory=factory).preflight(
            workspace=snapshot,
            task=task,
            environment=_environment(),
        )

    assert len(machines) == 1
    assert machines[0].deleted is True


def test_runner_uses_unique_machine_names_for_repeated_challenges() -> None:
    baseline, candidate = _snapshots()
    names: list[str] = []

    def factory(name: str, _image: str, _network: bool):
        names.append(name)
        snapshot = baseline if name.endswith("baseline") else candidate
        return _FakeMachine(name, snapshot.archive, threading.Barrier(1))

    runner = SmolMachineRunner(machine_factory=factory)
    for _ in range(2):
        runner.run_pair(
            baseline=baseline,
            candidate=candidate,
            task=_task(),
            environment=_environment(),
        )

    assert set(names[:2]).isdisjoint(names[2:])


def test_runner_classifies_timeout_as_behavioral_and_still_extracts_artifacts() -> None:
    baseline, candidate = _snapshots()
    barrier = threading.Barrier(2)

    def factory(name: str, _image: str, _network: bool):
        snapshot = baseline if name.endswith("baseline") else candidate
        error = TimeoutError("budget expired") if name.endswith("candidate") else None
        return _FakeMachine(name, snapshot.archive, barrier, task_error=error)

    baseline_result, candidate_result = SmolMachineRunner(machine_factory=factory).run_pair(
        baseline=baseline,
        candidate=candidate,
        task=_task(),
        environment=_environment(),
    )

    assert baseline_result.status == "exited"
    assert candidate_result.status == "timed_out"
    assert candidate_result.infrastructure_error is None
    assert candidate_result.final_workspace_digest == candidate.digest


def test_runner_classifies_smol_timeout_code_as_behavioral() -> None:
    baseline, candidate = _snapshots()
    barrier = threading.Barrier(2)

    def factory(name: str, _image: str, _network: bool):
        snapshot = baseline if name.endswith("baseline") else candidate
        error = _SdkTimeout("deadline reached") if name.endswith("candidate") else None
        return _FakeMachine(name, snapshot.archive, barrier, task_error=error)

    _, candidate_result = SmolMachineRunner(machine_factory=factory).run_pair(
        baseline=baseline,
        candidate=candidate,
        task=_task(),
        environment=_environment(),
    )

    assert candidate_result.status == "timed_out"
    assert candidate_result.infrastructure_error is None


def test_runner_classifies_setup_failure_as_infrastructure_and_cleans_up() -> None:
    baseline, candidate = _snapshots()
    barrier = threading.Barrier(1)
    machines: list[_FakeMachine] = []

    def factory(name: str, _image: str, _network: bool):
        snapshot = baseline if name.endswith("baseline") else candidate
        machine = _FakeMachine(name, snapshot.archive, barrier, setup_exit=9)
        machines.append(machine)
        return machine

    results = SmolMachineRunner(machine_factory=factory).run_pair(
        baseline=baseline,
        candidate=candidate,
        task=_task(),
        environment=_environment(),
    )

    assert all(result.status == "infrastructure_failed" for result in results)
    assert all(result.exit_code is None for result in results)
    assert all(machine.deleted for machine in machines)
    assert "setup" in results[0].infrastructure_error


def test_runner_cancels_both_agent_processes_and_cleans_up() -> None:
    baseline, candidate = _snapshots()
    barrier = threading.Barrier(2)
    machines: list[_FakeMachine] = []

    def factory(name: str, _image: str, _network: bool):
        snapshot = baseline if name.endswith("baseline") else candidate
        machine = _FakeMachine(name, snapshot.archive, barrier)
        machines.append(machine)
        return machine

    with pytest.raises(ChallengeCancelled):
        SmolMachineRunner(machine_factory=factory).run_pair(
            baseline=baseline,
            candidate=candidate,
            task=_task(),
            environment=_environment(),
            cancel_requested=lambda: True,
        )

    assert machines == []


def test_workspace_archive_has_no_host_paths() -> None:
    snapshot, _ = _snapshots()
    with tarfile.open(fileobj=io.BytesIO(snapshot.archive), mode="r:") as archive:
        assert archive.getnames() == ["AGENTS.md"]


def test_cli_adapter_creates_the_machine_with_the_requested_memory(monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(command, *, env=None):
        calls.append(command)
        return CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr(_CliMachine, "_run", staticmethod(run))
    _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False, memory_mib=4096)

    assert calls[0][-2:] == ["--mem", "4096"]


def test_cli_adapter_uses_local_machine_and_secret_environment_references(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    popen_calls: list[tuple[list[str], dict[str, str]]] = []

    def run(command, *, env=None):
        calls.append((command, env))
        return CompletedProcess(command, 0, stdout="done", stderr="")

    class Process:
        returncode = 0

        def communicate(self, timeout):
            return "done", ""

    def popen(command, **kwargs):
        popen_calls.append((command, kwargs["env"]))
        return Process()

    monkeypatch.setattr(_CliMachine, "_run", staticmethod(run))
    monkeypatch.setattr(_CliMachine, "_popen", staticmethod(popen))
    machine = _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", True)
    result = machine.exec(
        ["codex", "exec"],
        ExecutionOptions(env={"TOKEN": "secret-value"}, workdir="/workspace", timeout=60),
    )
    machine.delete()

    assert calls[0][0] == [
        "/bin/smol",
        "machine",
        "create",
        "--name",
        "pair-baseline",
        "--image",
        "runtime@sha256:digest",
        "--net",
    ]
    exec_argv, exec_env = popen_calls[0]
    host_name = "MONOPOLE_SMOL_ENV_0"
    assert "--local" in exec_argv
    assert f"TOKEN={host_name}" in exec_argv
    assert "secret-value" not in exec_argv
    assert exec_env[host_name] == "secret-value"
    assert result.exit_code == 0


def _machine_with_exec(monkeypatch, exec_):
    monkeypatch.setattr(
        _CliMachine,
        "_run",
        staticmethod(lambda command, *, env=None: CompletedProcess(command, 0, "", "")),
    )
    monkeypatch.setattr(_CliMachine, "exec", exec_)
    return _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)


def _is_copy(command: list[str]) -> bool:
    return command[:2] == ["sh", "-c"] and smol_module._TRANSFER_GUEST_ENV in command[2]


def test_cli_adapter_transfers_files_in_chunks_through_secret_environment(monkeypatch) -> None:
    calls: list[tuple[list[str], ExecutionOptions]] = []
    files: dict[str, bytearray] = {}
    chunk = smol_module._FILE_CHUNK_BYTES

    def exec_(self, command, options=None):
        calls.append((command, options))
        if command[:2] == ["sh", "-c"] and command[2].startswith("umask"):
            files[command[4]] = bytearray()
        elif _is_copy(command):
            offsets = [int(value) for value in re.findall(r"seek=(\d+)", command[2])]
            for encoded, offset in zip(options.env.values(), offsets, strict=True):
                value = base64.b64decode(encoded)
                files[command[4]][offset * chunk : offset * chunk + len(value)] = value
        elif command[:3] == ["stat", "-c", "%s"]:
            return _ExecResult(stdout=str(len(files[command[3]])))
        elif command[:2] == ["sh", "-c"] and command[2].startswith("dd"):
            start = int(command[5]) * chunk
            return _ExecResult(
                stdout=base64.encodebytes(bytes(files[command[4]][start : start + chunk])).decode()
            )
        return _ExecResult()

    machine = _machine_with_exec(monkeypatch, exec_)
    data = bytes(range(256)) * (chunk * (smol_module._FILE_WRITE_BATCH_CHUNKS + 1) // 256 + 1)

    machine.write_file("/tmp/workspace.tar", data, 0o600)
    captured = machine.read_file("/tmp/workspace.tar")

    assert captured == data
    copies = [(command, options) for command, options in calls if _is_copy(command)]
    assert len(copies) == 2
    secrets = [secret for _, options in copies for secret in options.env.values()]
    assert all(secret not in " ".join(command) for command, _ in copies for secret in secrets)
    assert all(len(secret) < 128 * 1024 for secret in secrets)
    assert all(command[2].startswith("set -e; ") for command, _ in copies)
    assert (["chmod", "600", "/tmp/workspace.tar"], "/") in [
        (command, options.workdir) for command, options in calls
    ]


def test_cli_transfer_retries_a_failed_chunk_at_the_same_offset(monkeypatch) -> None:
    attempts: list[list[str]] = []

    def exec_(self, command, options=None):
        if _is_copy(command):
            attempts.append(command)
            return _ExecResult(exit_code=1 if len(attempts) == 1 else 0, stderr="transient")
        return _ExecResult()

    monkeypatch.setattr(smol_module.time, "sleep", lambda _seconds: None)
    machine = _machine_with_exec(monkeypatch, exec_)

    machine.write_file("/tmp/workspace.tar", b"archive")

    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert "seek=0" in attempts[0][2]


def test_cli_transfer_gives_up_after_its_attempts(monkeypatch) -> None:
    def exec_(self, command, options=None):
        if command[:3] == ["stat", "-c", "%s"]:
            return _ExecResult(stdout="8")
        return _ExecResult(exit_code=1, stderr="read unavailable")

    monkeypatch.setattr(smol_module.time, "sleep", lambda _seconds: None)
    machine = _machine_with_exec(monkeypatch, exec_)

    with pytest.raises(RuntimeError, match="Smol file read exited 1: read unavailable"):
        machine.read_file("/tmp/result.tar")


def test_cli_transfer_stops_between_chunks_when_cancelled(monkeypatch) -> None:
    copied = 0

    def exec_(self, command, options=None):
        nonlocal copied
        copied += _is_copy(command)
        return _ExecResult()

    machine = _machine_with_exec(monkeypatch, exec_)

    with pytest.raises(ChallengeCancelled):
        machine.write_file(
            "/tmp/workspace.tar",
            b"x" * (smol_module._FILE_CHUNK_BYTES * smol_module._FILE_WRITE_BATCH_CHUNKS * 2),
            cancel_requested=lambda: copied >= 1,
        )

    assert copied == 1


def test_cli_adapter_does_not_treat_agent_stderr_text_as_harness_timeout(monkeypatch) -> None:
    def run(command, *, env=None):
        del env
        if "exec" in command:
            return CompletedProcess(command, 2, stdout="", stderr="one test timed out")
        return CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr(_CliMachine, "_run", staticmethod(run))
    machine = _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)

    result = machine.exec(["pytest"])

    assert result.exit_code == 2
    assert result.stderr == "one test timed out"


def test_cli_adapter_stops_agent_when_host_capture_limit_is_exceeded(monkeypatch) -> None:
    calls: list[str] = []

    def run(command, *, env=None):
        del env
        return CompletedProcess(command, 0, stdout="done", stderr="")

    class Process:
        returncode = None

        def __init__(self, stdout, stderr):
            self.stdout = stdout
            self.stderr = stderr

        def communicate(self, timeout):
            del timeout
            self.stdout.write(b"x" * 17)
            self.stdout.flush()
            raise TimeoutExpired("smol", 0.25)

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout=None):
            del timeout
            self.returncode = -15

        def kill(self):
            calls.append("kill")

    def popen(_command, **kwargs):
        return Process(kwargs["stdout"], kwargs["stderr"])

    monkeypatch.setattr(_CliMachine, "_run", staticmethod(run))
    monkeypatch.setattr(_CliMachine, "_popen", staticmethod(popen))
    monkeypatch.setattr(
        "weave_agent_signals.runs.challenges.smol._MAX_PROCESS_CAPTURE_BYTES",
        16,
    )
    machine = _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)

    with pytest.raises(RuntimeError, match="capture limit"):
        machine.exec(
            ["codex", "exec"],
            ExecutionOptions(env={}, workdir="/workspace", timeout=60),
        )

    assert calls == ["terminate"]


def test_cli_adapter_removes_created_machine_when_start_raises(monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(command, *, env=None):
        del env
        calls.append(command)
        if "start" in command:
            raise TimeoutExpired(command, 300)
        return CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr(_CliMachine, "_run", staticmethod(run))

    with pytest.raises(TimeoutExpired):
        _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)

    assert calls[-1] == [
        "/bin/smol",
        "machine",
        "rm",
        "--name",
        "pair-baseline",
        "--yes",
        "--local",
    ]


@pytest.mark.parametrize("value", ["smol 1.6.12", "smol 1.5.99", "1.6.13", "smol 1.6", ""])
def test_an_old_or_unreadable_smol_cli_version_is_rejected(value):
    from weave_agent_signals.runs.challenges import smol

    with pytest.raises(RuntimeError, match="1.6.13 or newer"):
        smol._validate_smol_cli_version(value)


@pytest.mark.parametrize(
    "value", ["smol 1.6.13", "smol 1.7.0", "smol 2.0.0", "smol 1.32.2 (engine smolvm 1.32.2)"]
)
def test_a_current_smol_cli_version_is_accepted(value):
    from weave_agent_signals.runs.challenges import smol

    smol._validate_smol_cli_version(value)


def _recording_run(calls, *, fail=lambda command: None):
    def run(command, *, env=None):
        calls.append(command)
        failure = fail(command)
        if isinstance(failure, BaseException):
            raise failure
        return failure or CompletedProcess(command, 0, stdout="", stderr="")

    return run


def test_cli_adapter_stops_the_machine_before_removing_it(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(_CliMachine, "_run", staticmethod(_recording_run(calls)))
    machine = _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)

    machine.delete()

    assert [call[1:3] for call in calls[-2:]] == [["machine", "stop"], ["machine", "rm"]]


def test_cli_adapter_removes_the_machine_when_stop_raises(monkeypatch) -> None:
    calls: list[list[str]] = []

    def stop_times_out(command):
        return TimeoutExpired(command, 300) if "stop" in command else None

    monkeypatch.setattr(
        _CliMachine, "_run", staticmethod(_recording_run(calls, fail=stop_times_out))
    )
    machine = _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)

    machine.delete()

    assert calls[-1][1:3] == ["machine", "rm"]


def test_cli_adapter_removes_a_machine_whose_create_timed_out(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    cleaned: list[Path] = []
    data_dir = tmp_path / "0123456789abcdef"
    data_dir.mkdir()

    def create_times_out(command):
        return TimeoutExpired(command, 300) if "create" in command else None

    monkeypatch.setattr(
        _CliMachine, "_run", staticmethod(_recording_run(calls, fail=create_times_out))
    )
    monkeypatch.setattr(_CliMachine, "_resolve_data_dir", lambda _self: data_dir)
    monkeypatch.setattr(smol_module, "_cleanup_orphaned_macos_data_dir", cleaned.append)

    with pytest.raises(TimeoutExpired):
        _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)

    assert calls[-1][1:3] == ["machine", "rm"]
    assert cleaned == [data_dir]


def test_cli_adapter_resolves_the_exact_macos_data_directory(monkeypatch) -> None:
    calls: list[list[str]] = []
    expected = (
        Path.home() / "Library" / "Caches" / "smolvm" / "vms" / "0123456789abcdef"
    ).resolve()

    def data_dir(command):
        return CompletedProcess(command, 0, str(expected), "") if "data-dir" in command else None

    monkeypatch.setattr(smol_module, "_macos", lambda: True)
    monkeypatch.setattr(_CliMachine, "_run", staticmethod(_recording_run(calls, fail=data_dir)))

    machine = _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)

    assert ["/bin/smol", "machine", "data-dir", "--name", "pair-baseline"] in calls
    assert machine._data_dir == expected


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "/tmp/0123456789abcdef",
        str(Path.home() / "Library" / "Caches" / "smolvm" / "vms" / "not-hex"),
    ],
)
def test_cli_adapter_fails_closed_and_removes_the_machine_for_an_unexpected_data_dir(
    monkeypatch, stdout
) -> None:
    calls: list[list[str]] = []

    def data_dir(command):
        return CompletedProcess(command, 0, stdout, "") if "data-dir" in command else None

    monkeypatch.setattr(smol_module, "_macos", lambda: True)
    monkeypatch.setattr(_CliMachine, "_run", staticmethod(_recording_run(calls, fail=data_dir)))

    with pytest.raises(RuntimeError, match="data directory"):
        _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)

    assert not any("start" in call for call in calls)
    assert calls[-1][1:3] == ["machine", "rm"]


def test_macos_orphan_cleanup_detaches_only_the_machines_image(monkeypatch, tmp_path) -> None:
    data_dir = tmp_path / "0123456789abcdef"
    image_path = data_dir / "pack" / "layers-cs.sparseimage"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"disk")
    removed: list[Path] = []
    calls: list[list[str]] = []
    info = plistlib.dumps(
        {
            "images": [
                {
                    "image-path": str(tmp_path / "other" / "layers-cs.sparseimage"),
                    "system-entities": [{"dev-entry": "/dev/disk41"}],
                },
                {"image-path": str(image_path), "system-entities": [{"dev-entry": "/dev/disk42"}]},
            ]
        }
    )

    def run(command, **kwargs):
        calls.append(command)
        stdout = info if command[:2] == ["hdiutil", "info"] else ""
        return CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(smol_module.subprocess, "run", run)
    monkeypatch.setattr(smol_module.shutil, "rmtree", removed.append)

    smol_module._cleanup_orphaned_macos_data_dir(data_dir)

    assert ["hdiutil", "detach", "/dev/disk42", "-force"] in calls
    assert ["hdiutil", "detach", "/dev/disk41", "-force"] not in calls
    assert removed == [data_dir]


def _keep_running() -> bool:
    return False


def test_runner_passes_its_cancel_check_to_every_pair_file_transfer() -> None:
    baseline, candidate = _snapshots()
    barrier = threading.Barrier(2)
    machines: list[_FakeMachine] = []

    def factory(name: str, _image: str, _network: bool):
        snapshot = baseline if name.endswith("baseline") else candidate
        machines.append(_FakeMachine(name, snapshot.archive, barrier))
        return machines[-1]

    SmolMachineRunner(machine_factory=factory).run_pair(
        baseline=baseline,
        candidate=candidate,
        task=_task(),
        environment=_environment(),
        baseline_runtime_files=(RuntimeFile("/root/.codex/AGENTS.md", b"baseline", 0o600),),
        candidate_runtime_files=(RuntimeFile("/root/.codex/AGENTS.md", b"candidate", 0o600),),
        cancel_requested=_keep_running,
    )

    for machine in machines:
        transfers = dict(machine.cancel_checks)
        assert set(transfers) >= {
            "/tmp/workspace.tar",
            "/tmp/runtime-files.tar",
            "/root/.monopole-final-workspace.tar",
        }
        assert all(
            check is _keep_running
            for path, check in transfers.items()
            if path != machine.session_path
        )


def test_runner_preflight_passes_its_cancel_check_to_file_transfers() -> None:
    snapshot = WorkspaceSnapshot((WorkspaceFile("task/app.py", b"print('ready')\n"),))
    machines: list[_FakeMachine] = []

    def factory(name: str, _image: str, _network: bool):
        machines.append(
            _FakeMachine(
                name,
                snapshot.archive,
                threading.Barrier(1),
                missing_executables=frozenset({"node"}),
            )
        )
        return machines[-1]

    task = AuthoredTask(
        prompt="Update task/app.py.",
        goal="The update is verified.",
        required_files=("task/app.py",),
        required_executables=("node",),
        workspace_digest="sha256:source",
        author_model="author",
        author_backend="cli",
    )

    with pytest.raises(RuntimeError, match="node"):
        SmolMachineRunner(machine_factory=factory).preflight(
            workspace=snapshot,
            task=task,
            environment=_environment(),
            cancel_requested=_keep_running,
        )

    [machine] = machines
    assert machine.cancel_checks
    assert all(check is _keep_running for _, check in machine.cancel_checks)


@pytest.mark.parametrize(
    ("version", "accepted"), [("smol 1.6.12\n", False), ("smol 1.6.13\n", True)]
)
def test_smol_executable_enforces_the_minimum_cli_version(
    monkeypatch, tmp_path, version, accepted
) -> None:
    executable = tmp_path / "smol"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(smol_module.shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(
        smol_module.subprocess,
        "run",
        lambda command, **_: CompletedProcess(command, 0, version, ""),
    )

    if accepted:
        assert smol_module._smol_executable() == str(executable)
    else:
        with pytest.raises(RuntimeError, match="1.6.13 or newer"):
            smol_module._smol_executable()


@pytest.mark.parametrize(
    ("chunk", "error"),
    [(base64.b64encode(b"half").decode(), "truncated"), ("aGFsZg==!!", "invalid base64")],
)
def test_cli_read_rejects_a_short_or_corrupt_chunk(monkeypatch, chunk, error) -> None:
    def exec_(self, command, options=None):
        if command[:3] == ["stat", "-c", "%s"]:
            return _ExecResult(stdout="8")
        return _ExecResult(stdout=chunk)

    machine = _machine_with_exec(monkeypatch, exec_)

    with pytest.raises(RuntimeError, match=error):
        machine.read_file("/tmp/result.tar")


def test_cli_transfer_retries_a_timed_out_chunk(monkeypatch) -> None:
    attempts = 0

    def exec_(self, command, options=None):
        nonlocal attempts
        if _is_copy(command):
            attempts += 1
            if attempts == 1:
                raise TimeoutError("Smol agent execution reached its timeout")
        return _ExecResult()

    monkeypatch.setattr(smol_module.time, "sleep", lambda _seconds: None)
    machine = _machine_with_exec(monkeypatch, exec_)

    machine.write_file("/tmp/workspace.tar", b"archive")

    assert attempts == 2


def test_cli_exec_stops_waiting_when_the_cli_outlives_its_timeout(monkeypatch) -> None:
    ticks = iter(range(0, 1_000_000, 50))

    class Process:
        returncode = None
        terminated = False
        polls = 0

        def communicate(self, timeout):
            self.polls += 1
            assert self.polls < 1_000, "exec kept waiting past its deadline"
            raise TimeoutExpired("smol", timeout)

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    process = Process()
    monkeypatch.setattr(
        _CliMachine,
        "_run",
        staticmethod(lambda command, *, env=None: CompletedProcess(command, 0, "", "")),
    )
    monkeypatch.setattr(_CliMachine, "_popen", staticmethod(lambda command, **_: process))
    machine = _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)
    monkeypatch.setattr(smol_module.time, "monotonic", lambda: next(ticks))

    with pytest.raises(TimeoutError, match="did not return"):
        machine.exec(["true"], ExecutionOptions(env={}, workdir="/", timeout=30))

    assert process.terminated
