import json

import pytest

from weave_agent_signals.runs.challenges.environment import (
    RuntimeFile,
    capture_execution_environment,
)


def _cohort(*turns: dict) -> dict:
    return {"turns": list(turns)}


def _turn(model: str = "gpt-5.6-sol", effort: str = "high") -> dict:
    return {
        "model": model,
        "model_family": "openai" if model.startswith("gpt") else "anthropic",
        "effort_level": effort,
    }


def test_execution_environment_pins_exact_harness_and_hashes_all_values() -> None:
    captured = capture_execution_environment(
        cohort=_cohort(_turn(), _turn()),
        workspace_digest="sha256:workspace",
        image="agent-runtime:codex-0.144.1",
        image_digest=f"sha256:{'a' * 64}",
        timeout_seconds=1800,
        runtime_env={"TOKEN": "super-secret", "PATH": "/usr/bin"},
        runtime_files=(RuntimeFile("/root/.codex/config.toml", b"secret config"),),
        version_reader=lambda argv: "codex-cli 0.144.1" if argv == ["codex", "--version"] else "",
    )

    assert captured.identity.harness == "codex"
    assert captured.identity.harness_version == "codex-cli 0.144.1"
    assert captured.identity.command == (
        "codex",
        "--search",
        "exec",
        "--model",
        "gpt-5.6-sol",
        "-c",
        'model_reasoning_effort="high"',
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "{task}",
    )
    assert [item.name for item in captured.identity.environment] == ["PATH", "TOKEN"]
    assert [item.name for item in captured.identity.runtime_files] == ["/root/.codex/config.toml"]
    serialized = json.dumps(captured.identity.to_dict())
    assert "super-secret" not in serialized
    assert "secret config" not in serialized
    assert captured.runtime_env["TOKEN"] == "super-secret"


def test_execution_environment_disables_codex_search_without_network() -> None:
    captured = capture_execution_environment(
        cohort=_cohort(_turn()),
        workspace_digest="sha256:workspace",
        image="agent-runtime:codex-0.144.1",
        image_digest=f"sha256:{'a' * 64}",
        timeout_seconds=1800,
        runtime_env={"PATH": "/usr/bin"},
        network_enabled=False,
        version_reader=lambda _argv: "codex-cli 0.144.1",
    )

    assert "--search" not in captured.identity.command


def test_execution_environment_rejects_mixed_models() -> None:
    with pytest.raises(ValueError, match="one evaluated model"):
        capture_execution_environment(
            cohort=_cohort(_turn(), _turn("claude-sonnet-5")),
            workspace_digest="sha256:workspace",
            image="runtime",
            image_digest=f"sha256:{'a' * 64}",
            timeout_seconds=60,
            runtime_env={},
            version_reader=lambda _argv: "version",
        )


def test_execution_environment_rejects_mixed_effort() -> None:
    with pytest.raises(ValueError, match="one effort level"):
        capture_execution_environment(
            cohort=_cohort(_turn(effort="high"), _turn(effort="medium")),
            workspace_digest="sha256:workspace",
            image="runtime",
            image_digest=f"sha256:{'a' * 64}",
            timeout_seconds=60,
            runtime_env={},
            version_reader=lambda _argv: "version",
        )


def test_execution_environment_rejects_unsupported_model_family() -> None:
    with pytest.raises(ValueError, match="unsupported evaluated model family"):
        capture_execution_environment(
            cohort=_cohort({"model": "llama", "model_family": "meta", "effort_level": None}),
            workspace_digest="sha256:workspace",
            image="runtime",
            image_digest=f"sha256:{'a' * 64}",
            timeout_seconds=60,
            runtime_env={},
            version_reader=lambda _argv: "version",
        )


def test_execution_environment_requires_harness_version() -> None:
    with pytest.raises(ValueError, match="version"):
        capture_execution_environment(
            cohort=_cohort(_turn()),
            workspace_digest="sha256:workspace",
            image="runtime",
            image_digest=f"sha256:{'a' * 64}",
            timeout_seconds=60,
            runtime_env={},
            version_reader=lambda _argv: "",
        )
