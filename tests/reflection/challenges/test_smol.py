from __future__ import annotations

import io
import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired

import pytest

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
    ) -> None:
        self.name = name
        self.archive = archive
        self.barrier = barrier
        self.task_error = task_error
        self.setup_exit = setup_exit
        self.missing_executables = missing_executables
        self.deleted = False
        self.writes: dict[str, bytes] = {}
        self.calls: list[tuple[list[str], object]] = []

    def write_file(self, path: str, data: bytes, mode: int | None = None) -> None:
        self.writes[path] = data

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
        raise AssertionError(command)

    def read_file(self, path: str) -> bytes:
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
    assert "--local" in exec_argv
    assert "TOKEN=MONOPOLE_SMOL_ENV_0" in exec_argv
    assert "secret-value" not in exec_argv
    assert exec_env["MONOPOLE_SMOL_ENV_0"] == "secret-value"
    assert result.exit_code == 0


def test_cli_adapter_uses_native_copy_for_bounded_machine_files(monkeypatch) -> None:
    calls: list[list[str]] = []
    binary_calls: list[tuple[list[str], bytes | None]] = []

    def run(command, *, env=None):
        del env
        calls.append(command)
        if command[-4:] == ["stat", "-c", "%s", "/tmp/result.tar"]:
            return CompletedProcess(command, 0, stdout="8", stderr="")
        if command[1:3] == ["machine", "cp"]:
            Path(command[-1]).write_bytes(b"captured")
        return CompletedProcess(command, 0, stdout="done", stderr="")

    def run_binary(command, *, input_data=None):
        binary_calls.append((command, input_data))
        return CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(_CliMachine, "_run", staticmethod(run))
    monkeypatch.setattr(_CliMachine, "_run_binary", staticmethod(run_binary))
    machine = _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)

    machine.write_file("/tmp/workspace.tar", b"archive", 0o600)
    captured = machine.read_file("/tmp/result.tar")

    assert binary_calls == [
        (
            [
                "/bin/smol",
                "machine",
                "exec",
                "--name",
                "pair-baseline",
                "--local",
                "--interactive",
                "--",
                "dd",
                "of=/tmp/workspace.tar",
                "status=none",
            ],
            b"archive",
        ),
    ]
    assert captured == b"captured"
    transfer_call = next(call for call in calls if call[1:3] == ["machine", "cp"])
    assert transfer_call[-2] == "pair-baseline:/tmp/result.tar"
    assert ["chmod", "600", "/tmp/workspace.tar"] in [call[-3:] for call in calls]


def test_cli_adapter_rejects_failed_native_copy(monkeypatch) -> None:
    def run(command, *, env=None):
        del env
        if command[-4:] == ["stat", "-c", "%s", "/tmp/result.tar"]:
            return CompletedProcess(command, 0, stdout="8", stderr="")
        if command[1:3] == ["machine", "cp"]:
            return CompletedProcess(command, 1, stdout="", stderr="copy unavailable")
        return CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr(_CliMachine, "_run", staticmethod(run))
    machine = _CliMachine("/bin/smol", "pair-baseline", "runtime@sha256:digest", False)

    with pytest.raises(RuntimeError, match="copy failed"):
        machine.read_file("/tmp/result.tar")


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
