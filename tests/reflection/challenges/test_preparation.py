from pathlib import Path

import pytest

from weave_agent_signals.runs.challenges.contracts import (
    AuthoredTask,
    TaskMaterial,
    TaskPreflightError,
)
from weave_agent_signals.runs.challenges.preparation import (
    prepare_task_context,
    prepare_task_workspace,
    validate_task_workspace,
)
from weave_agent_signals.runs.challenges.workspace import WorkspaceFile, WorkspaceSnapshot


def _task(mode: str, *materials: TaskMaterial) -> AuthoredTask:
    return AuthoredTask(
        prompt="Fix the parser.",
        goal="Parser tests pass.",
        setup_mode=mode,
        materials=materials,
        judging_criteria=("Parser tests pass.",),
        start_checks=("The workspace is writable.",),
        workspace_digest="sha256:seed",
        author_model="author",
        author_backend="cli",
    )


def _material(destination: str = "task") -> TaskMaterial:
    return TaskMaterial(
        url="https://github.com/example/parser.git",
        revision="a" * 40,
        destination=destination,
    )


def test_agent_bootstrap_keeps_identical_seed_and_does_not_fetch() -> None:
    seed = WorkspaceSnapshot((WorkspaceFile("README.md", b"seed\n"),))

    result = prepare_task_workspace(
        seed,
        _task("agent_bootstrap", _material()),
        fetch_repository=lambda _material: (_ for _ in ()).throw(AssertionError("fetched")),
    )

    assert result is seed


def test_agent_bootstrap_fetches_public_material_for_task_authoring_context() -> None:
    seed = WorkspaceSnapshot((WorkspaceFile("README.md", b"seed\n"),))

    context = prepare_task_context(
        seed,
        _task("agent_bootstrap", _material()),
        fetch_repository=lambda _material: WorkspaceSnapshot(
            (WorkspaceFile("src/parser.py", b"def parse(): ...\n"),)
        ),
    )

    assert context.paths == ("README.md", "task/src/parser.py")


def test_prepared_workspace_fetches_each_pinned_repo_once_and_overlays_destination() -> None:
    seed = WorkspaceSnapshot((WorkspaceFile("README.md", b"seed\n"),))
    fetched: list[TaskMaterial] = []

    def fetch(material: TaskMaterial) -> WorkspaceSnapshot:
        fetched.append(material)
        return WorkspaceSnapshot(
            (
                WorkspaceFile("pyproject.toml", b"[project]\n"),
                WorkspaceFile("src/parser.py", b"def parse(): ...\n"),
            )
        )

    result = prepare_task_workspace(
        seed,
        _task("prepared_workspace", _material()),
        fetch_repository=fetch,
    )

    assert fetched == [_material()]
    assert result.paths == ("README.md", "task/pyproject.toml", "task/src/parser.py")
    assert result.read_text("README.md") == "seed\n"


def test_prepared_workspace_rejects_overlay_collision() -> None:
    seed = WorkspaceSnapshot((WorkspaceFile("task/README.md", b"seed\n"),))
    fetched = WorkspaceSnapshot((WorkspaceFile("README.md", b"remote\n"),))

    with pytest.raises(ValueError, match="collision"):
        prepare_task_workspace(
            seed,
            _task("prepared_workspace", _material()),
            fetch_repository=lambda _material: fetched,
        )


def test_task_workspace_preflight_rejects_missing_required_files() -> None:
    task = AuthoredTask.model_validate(
        {
            **_task("prepared_workspace", _material()).model_dump(mode="json"),
            "task_id": "pending",
            "required_files": ["task/src/parser.py", "task/tests/test_parser.py"],
        }
    )
    workspace = WorkspaceSnapshot((WorkspaceFile("task/src/parser.py", b"pass\n"),))

    with pytest.raises(TaskPreflightError, match="task/tests/test_parser.py"):
        validate_task_workspace(workspace, task)


def test_task_workspace_preflight_accepts_exact_required_files() -> None:
    task = AuthoredTask.model_validate(
        {
            **_task("prepared_workspace", _material()).model_dump(mode="json"),
            "task_id": "pending",
            "required_files": ["task/src/parser.py"],
        }
    )
    workspace = WorkspaceSnapshot((WorkspaceFile("task/src/parser.py", b"pass\n"),))

    validate_task_workspace(workspace, task)


def test_default_git_fetcher_uses_detached_pinned_revision_and_excludes_git_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []
    run_kwargs: list[dict[str, object]] = []

    def run(command, **kwargs):
        calls.append(command)
        run_kwargs.append(kwargs)
        destination = Path(kwargs["cwd"])
        if command[:2] == ["git", "init"]:
            (destination / ".git").mkdir()
            (destination / "app.py").write_text("print('ready')\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("weave_agent_signals.runs.challenges.preparation.subprocess.run", run)
    monkeypatch.setattr(
        "weave_agent_signals.runs.challenges.preparation.tempfile.TemporaryDirectory",
        lambda: _TempDirectory(tmp_path),
    )

    result = prepare_task_workspace(
        WorkspaceSnapshot(()),
        _task("prepared_workspace", _material()),
    )

    assert calls == [
        ["git", "init", "--quiet"],
        ["git", "remote", "add", "origin", _material().url],
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
            "a" * 40,
        ],
        ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"],
    ]
    assert all(kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0" for kwargs in run_kwargs)
    assert result.paths == ("task/app.py",)


class _TempDirectory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, *_args) -> None:
        return None
