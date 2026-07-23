"""One explicit runtime description shared by both paired sandbox arms."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, model_validator

from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.runs.challenges.contracts import TaskMaterialPlan
from weave_agent_signals.runs.challenges.environment import (
    RuntimeFile,
    capture_execution_environment,
)
from weave_agent_signals.runs.challenges.preparation import (
    RepositoryFetcher,
    prepare_task_context,
)
from weave_agent_signals.runs.challenges.service import PreparedPair, run_challenge
from weave_agent_signals.runs.challenges.smol import SmolMachineRunner
from weave_agent_signals.runs.challenges.workspace import capture_workspace, materialize_arms


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class RuntimeFileSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Path
    target: StrictStr
    mode: Annotated[int, Field(strict=True, ge=0, le=0o777)] = 0o600

    @model_validator(mode="after")
    def validate_source(self) -> RuntimeFileSource:
        source = self.source.expanduser()
        if not source.is_absolute():
            raise ValueError("runtime file source must be an absolute host path")
        if source.is_symlink() or not source.is_file():
            raise ValueError("runtime file source must be an existing non-symlink file")
        object.__setattr__(self, "source", source.resolve(strict=True))
        RuntimeFile(self.target, b"", self.mode)
        return self


class SandboxRuntime(BaseModel):
    """Platform runtime inputs that are identical for A and B."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    workspace_root: Path
    image: StrictStr
    image_digest: StrictStr
    timeout_seconds: Annotated[int, Field(strict=True, ge=1)]
    network_enabled: StrictBool
    environment_overrides: dict[StrictStr, StrictStr]
    runtime_files: tuple[RuntimeFileSource, ...] = ()

    @model_validator(mode="after")
    def validate_runtime(self) -> SandboxRuntime:
        root = self.workspace_root.expanduser().resolve(strict=False)
        if not root.is_dir():
            raise ValueError("workspace_root must be an existing directory")
        object.__setattr__(self, "workspace_root", root)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_digest):
            raise ValueError("image_digest must be a complete sha256 digest")
        local_image = Path(self.image).expanduser()
        if local_image.is_file():
            local_image = local_image.resolve(strict=True)
            actual = _file_digest(local_image)
            if actual != self.image_digest:
                raise ValueError("local Smol image does not match image_digest")
            object.__setattr__(self, "image", str(local_image))
        elif f"@{self.image_digest}" not in self.image:
            raise ValueError("image must be a content-addressed OCI reference or local file")
        if not self.environment_overrides or any(
            not key.strip() for key in self.environment_overrides
        ):
            raise ValueError("environment_overrides must contain nonblank names")
        if self.environment_overrides.get("HOME") is None:
            raise ValueError("environment_overrides must define guest HOME")
        if self.environment_overrides.get("PWD") != "/workspace":
            raise ValueError("environment_overrides must set PWD to /workspace")
        targets = tuple(item.target for item in self.runtime_files)
        if len(targets) != len(set(targets)):
            raise ValueError("runtime file targets must be unique")
        return self

    def environment(
        self,
        parent: Mapping[str, str] | None = None,
    ) -> Mapping[str, str]:
        """Copy every parent variable, then apply declared guest path corrections."""

        values = dict(os.environ if parent is None else parent)
        values.update(self.environment_overrides)
        return MappingProxyType(values)

    def capture_files(self) -> tuple[RuntimeFile, ...]:
        return tuple(
            RuntimeFile(item.target, item.source.read_bytes(), item.mode)
            for item in self.runtime_files
        )


def load_sandbox_runtime(path: str | Path) -> SandboxRuntime:
    source = Path(path).expanduser().resolve(strict=True)
    return SandboxRuntime.model_validate_json(source.read_text(encoding="utf-8"))


def create_challenge_runner(
    runtime: SandboxRuntime,
    *,
    pair_runner=None,
    version_reader: Callable[[list[str]], str] | None = None,
    repository_fetcher: RepositoryFetcher | None = None,
):
    """Bind the platform runtime to the reflection-stage challenge contract."""

    machine_runner = pair_runner or SmolMachineRunner()

    def challenge_runner(
        *,
        baseline,
        candidate,
        coaching_text,
        config,
        cohort,
        adapter,
        author_client,
        judge_clients,
        cancel_requested,
    ):
        frozen = capture_workspace(runtime.workspace_root)
        arms = materialize_arms(
            frozen,
            baseline=baseline,
            candidate=candidate.bundle,
            workspace_root=runtime.workspace_root,
            resolve_locator=adapter.resolve_locator,
            guest_home=runtime.environment_overrides["HOME"],
        )
        environment_kwargs = {
            "cohort": cohort,
            "workspace_digest": frozen.digest,
            "image": runtime.image,
            "image_digest": runtime.image_digest,
            "timeout_seconds": runtime.timeout_seconds,
            "runtime_env": runtime.environment(),
            "runtime_files": runtime.capture_files(),
            "network_enabled": runtime.network_enabled,
        }
        if version_reader is not None:
            environment_kwargs["version_reader"] = version_reader
        environment = capture_execution_environment(**environment_kwargs)

        def prepare_pair(plan: TaskMaterialPlan) -> PreparedPair:
            preparation_kwargs = {}
            if repository_fetcher is not None:
                preparation_kwargs["fetch_repository"] = repository_fetcher
            authoring_context = prepare_task_context(
                arms.authoring,
                plan,
                **preparation_kwargs,
            )
            prepared_workspace = (
                authoring_context if plan.setup_mode == "prepared_workspace" else arms.authoring
            )
            prepared_arms = materialize_arms(
                prepared_workspace,
                baseline=baseline,
                candidate=candidate.bundle,
                workspace_root=runtime.workspace_root,
                resolve_locator=adapter.resolve_locator,
                guest_home=runtime.environment_overrides["HOME"],
                verify_baseline=False,
            )
            prepared_environment_kwargs = {
                **environment_kwargs,
                "workspace_digest": prepared_workspace.digest,
            }
            return PreparedPair(
                authoring=authoring_context,
                arms=prepared_arms,
                environment=capture_execution_environment(**prepared_environment_kwargs),
            )

        try:
            rubrics = tuple(SESSION_RUBRICS[item.id] for item in config.rubrics)
        except KeyError as exc:
            raise ValueError(f"unknown configured challenge rubric: {exc.args[0]}") from exc
        return run_challenge(
            candidate_id=candidate.candidate_id,
            coaching_digest=coaching_text,
            authoring_snapshot=arms.authoring,
            baseline_snapshot=arms.baseline,
            candidate_snapshot=arms.candidate,
            environment=environment,
            author=config.models.proposal_evaluator,
            judges=config.models.challenge_judges,
            rubrics=rubrics,
            author_client=author_client,
            judge_clients=judge_clients,
            runner=machine_runner,
            baseline_runtime_files=arms.baseline_runtime_files,
            candidate_runtime_files=arms.candidate_runtime_files,
            prepare_pair=prepare_pair,
            cancel_requested=cancel_requested,
        )

    return challenge_runner
