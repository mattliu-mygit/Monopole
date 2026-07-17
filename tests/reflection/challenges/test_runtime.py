from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from weave_agent_signals.judges.rubrics import VERIFICATION_DISCIPLINE
from weave_agent_signals.run_config import ModelDescriptor, PositionedJudge, RubricDescriptor
from weave_agent_signals.runs.bundles import BundleSnapshot, TargetSnapshot
from weave_agent_signals.runs.challenges.contracts import AuthoredTask, TaskMaterial
from weave_agent_signals.runs.challenges.runtime import (
    create_challenge_runner,
    load_sandbox_runtime,
)
from weave_agent_signals.runs.challenges.workspace import WorkspaceFile, WorkspaceSnapshot


def _document(tmp_path, **updates):
    value = {
        "schema_version": "1",
        "workspace_root": str(tmp_path),
        "image": f"agent-runtime@sha256:{'a' * 64}",
        "image_digest": f"sha256:{'a' * 64}",
        "timeout_seconds": 900,
        "network_enabled": True,
        "environment_overrides": {"HOME": "/root", "PWD": "/workspace"},
        "runtime_files": [],
    }
    value.update(updates)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_runtime_inherits_every_parent_variable_and_applies_guest_overrides(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setenv("HOST_ONLY_SETTING", "value")

    runtime = load_sandbox_runtime(_document(tmp_path))
    environment = runtime.environment()

    assert environment["WANDB_API_KEY"] == "secret"
    assert environment["HOST_ONLY_SETTING"] == "value"
    assert environment["HOME"] == "/root"
    assert environment["PWD"] == "/workspace"
    assert runtime.workspace_root == tmp_path.resolve()


def test_runtime_document_is_closed_and_requires_guest_environment(tmp_path):
    with pytest.raises(ValueError):
        load_sandbox_runtime(_document(tmp_path, extra=True))

    with pytest.raises(ValueError, match="environment_overrides"):
        load_sandbox_runtime(_document(tmp_path, environment_overrides={}))


def test_runtime_requires_workspace_and_content_addressed_image(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="workspace_root"):
        load_sandbox_runtime(_document(tmp_path, workspace_root=str(missing)))

    with pytest.raises(ValueError, match="content-addressed"):
        load_sandbox_runtime(_document(tmp_path, image="mutable-latest"))


def test_runtime_accepts_a_digest_verified_local_smolmachine(tmp_path, monkeypatch):
    image = tmp_path / "agent.smolmachine"
    image.write_bytes(b"portable-runtime")
    digest = f"sha256:{hashlib.sha256(image.read_bytes()).hexdigest()}"
    monkeypatch.setattr(Path, "read_bytes", lambda _path: (_ for _ in ()).throw(AssertionError()))

    runtime = load_sandbox_runtime(_document(tmp_path, image=str(image), image_digest=digest))

    assert runtime.image == str(image)
    assert runtime.image_digest == digest


def test_runtime_captures_declared_cli_config_without_persisting_its_value(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('approval_policy = "never"\n', encoding="utf-8")
    runtime = load_sandbox_runtime(
        _document(
            tmp_path,
            runtime_files=[
                {
                    "source": str(config),
                    "target": "/root/.codex/config.toml",
                    "mode": 384,
                }
            ],
        )
    )

    captured = runtime.capture_files()

    assert captured[0].path == "/root/.codex/config.toml"
    assert captured[0].content == b'approval_policy = "never"\n'
    assert captured[0].digest.startswith("sha256:")


@pytest.mark.parametrize("target", ["workspace/config.toml", "/workspace/config.toml"])
def test_runtime_files_cannot_change_the_paired_workspace(tmp_path, target):
    config = tmp_path / "config.toml"
    config.write_text("config", encoding="utf-8")
    with pytest.raises(ValueError, match="absolute|outside /workspace"):
        load_sandbox_runtime(
            _document(
                tmp_path,
                runtime_files=[{"source": str(config), "target": target, "mode": 384}],
            )
        )


def test_challenge_runtime_prepares_public_material_once_before_forking_arms(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "AGENTS.md").write_text("baseline\n", encoding="utf-8")
    runtime = load_sandbox_runtime(_document(tmp_path))
    baseline = BundleSnapshot(
        (TargetSnapshot(kind="file", locator="agents", exists=True, content="baseline\n"),)
    )
    candidate_bundle = BundleSnapshot(
        (TargetSnapshot(kind="file", locator="agents", exists=True, content="candidate\n"),)
    )
    candidate = SimpleNamespace(candidate_id="candidate-1", bundle=candidate_bundle)
    fetched: list[TaskMaterial] = []

    def fetch(material):
        fetched.append(material)
        return WorkspaceSnapshot((WorkspaceFile("app.py", b"print('fixture')\n"),))

    captured = {}

    def fake_run_challenge(**kwargs):
        task = AuthoredTask(
            prompt="Fix task/app.py.",
            goal="The fixture works.",
            setup_mode="prepared_workspace",
            materials=(
                TaskMaterial(
                    url="https://github.com/example/fixture.git",
                    revision="a" * 40,
                    destination="task",
                ),
            ),
            judging_criteria=("The fixture works.",),
            start_checks=("task/app.py exists.",),
            workspace_digest=kwargs["environment"].identity.workspace_digest,
            author_model="author",
            author_backend="cli",
        )
        captured["prepared"] = kwargs["prepare_pair"](task)
        return "challenge"

    monkeypatch.setattr(
        "weave_agent_signals.runs.challenges.runtime.run_challenge",
        fake_run_challenge,
    )
    config = SimpleNamespace(
        models=SimpleNamespace(
            proposal_evaluator=ModelDescriptor(
                id="author",
                label="author",
                provider="codex",
                provider_model="author",
                family="openai",
                supported_roles=("proposal_evaluator",),
            ),
            challenge_judges=(
                PositionedJudge(
                    id="judge",
                    label="judge",
                    provider="codex",
                    provider_model="judge",
                    family="openai",
                    supported_roles=("judge",),
                    position=1,
                ),
            ),
        ),
        rubrics=(
            RubricDescriptor(
                id=VERIFICATION_DISCIPLINE.scorer_name,
                label="Verification",
                evaluation_unit="session",
                version="1",
                content_digest="sha256:rubric",
                pass_threshold=0.5,
            ),
        ),
    )
    adapter = SimpleNamespace(resolve_locator=lambda _locator: tmp_path / "AGENTS.md")
    challenge_runner = create_challenge_runner(
        runtime,
        pair_runner=object(),
        version_reader=lambda _argv: "codex-cli 0.144.1",
        repository_fetcher=fetch,
    )

    result = challenge_runner(
        baseline=baseline,
        candidate=candidate,
        coaching_text="verification was weak",
        config=config,
        cohort={
            "turns": [
                {
                    "model": "gpt-5.6-sol",
                    "model_family": "openai",
                    "effort_level": "high",
                }
            ]
        },
        adapter=adapter,
        author_client=object(),
        judge_clients={"judge": object()},
        cancel_requested=lambda: False,
    )

    prepared = captured["prepared"]
    assert result == "challenge"
    assert len(fetched) == 1
    assert prepared.task.workspace_digest == prepared.environment.identity.workspace_digest
    assert prepared.arms.baseline.read_text("task/app.py") == "print('fixture')\n"
    assert prepared.arms.baseline.read_text("AGENTS.md") == "baseline\n"
    assert prepared.arms.candidate.read_text("AGENTS.md") == "candidate\n"
