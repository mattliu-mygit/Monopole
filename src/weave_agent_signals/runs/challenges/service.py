"""Orchestrate one authored task, paired execution, and configured judge panel."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from weave_agent_signals.judges.inference import ChatClient, InferenceCancelled
from weave_agent_signals.judges.rubrics import Rubric
from weave_agent_signals.run_config import ModelDescriptor, PositionedJudge
from weave_agent_signals.runs.challenges.contracts import (
    ArmName,
    ArmResult,
    AuthoredTask,
    ChallengeResult,
    TaskMaterialPlan,
    TaskPreflightError,
    Winner,
)
from weave_agent_signals.runs.challenges.environment import CapturedEnvironment, RuntimeFile
from weave_agent_signals.runs.challenges.inference import (
    author_task,
    judge_pair,
    plan_task_materials,
)
from weave_agent_signals.runs.challenges.preparation import validate_task_workspace
from weave_agent_signals.runs.challenges.workspace import ArmSnapshots, WorkspaceSnapshot


@dataclass(frozen=True)
class PreparedPair:
    authoring: WorkspaceSnapshot
    arms: ArmSnapshots
    environment: CapturedEnvironment


class PairRunner(Protocol):
    def preflight(
        self,
        *,
        workspace: WorkspaceSnapshot,
        task: AuthoredTask,
        environment: CapturedEnvironment,
        cancel_requested: Callable[[], bool],
    ) -> None: ...

    def run_pair(
        self,
        *,
        baseline: WorkspaceSnapshot,
        candidate: WorkspaceSnapshot,
        task,
        environment: CapturedEnvironment,
        baseline_runtime_files: tuple[RuntimeFile, ...],
        candidate_runtime_files: tuple[RuntimeFile, ...],
        cancel_requested: Callable[[], bool],
    ) -> tuple[ArmResult, ArmResult]: ...


def _safe_error(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return (text or type(error).__name__)[:500]


def _require_active(cancel_requested: Callable[[], bool]) -> None:
    if cancel_requested():
        raise InferenceCancelled("paired challenge cancelled")


def _presentation_order(candidate_id: str) -> tuple[ArmName, ArmName]:
    first_byte = hashlib.sha256(candidate_id.encode()).digest()[0]
    return ("candidate", "baseline") if first_byte % 2 == 0 else ("baseline", "candidate")


def _panel_winner(verdicts) -> Winner:
    counts = Counter(verdict.winner for verdict in verdicts)
    majority = len(verdicts) // 2 + 1
    if counts["candidate"] >= majority:
        return "candidate"
    if counts["baseline"] >= majority:
        return "baseline"
    return "tie"


def _blinded_instruction_evidence(
    baseline: WorkspaceSnapshot,
    candidate: WorkspaceSnapshot,
    baseline_runtime_files: tuple[RuntimeFile, ...],
    candidate_runtime_files: tuple[RuntimeFile, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    baseline_files = {item.path: item.content for item in baseline.files}
    candidate_files = {item.path: item.content for item in candidate.files}
    baseline_files.update({item.path: item.content for item in baseline_runtime_files})
    candidate_files.update({item.path: item.content for item in candidate_runtime_files})
    paths = tuple(
        sorted(
            path
            for path in set(baseline_files) | set(candidate_files)
            if baseline_files.get(path) != candidate_files.get(path)
        )
    )
    contents: list[str] = []
    for path in paths:
        for files in (baseline_files, candidate_files):
            content = files.get(path)
            if content is None:
                continue
            try:
                contents.append(content.decode("utf-8"))
            except UnicodeDecodeError:
                continue
    return paths, tuple(contents)


def run_challenge(
    *,
    candidate_id: str,
    coaching_digest: str,
    authoring_snapshot: WorkspaceSnapshot,
    baseline_snapshot: WorkspaceSnapshot,
    candidate_snapshot: WorkspaceSnapshot,
    environment: CapturedEnvironment,
    author: ModelDescriptor,
    judges: Sequence[PositionedJudge],
    rubrics: Sequence[Rubric],
    author_client: ChatClient,
    judge_clients: Mapping[str, ChatClient],
    runner: PairRunner,
    baseline_runtime_files: tuple[RuntimeFile, ...] = (),
    candidate_runtime_files: tuple[RuntimeFile, ...] = (),
    prepare_pair: Callable[[TaskMaterialPlan], PreparedPair] | None = None,
    cancel_requested: Callable[[], bool] = lambda: False,
) -> ChallengeResult:
    """Run one full comparison; every non-clear-B outcome conservatively keeps A."""

    positions = tuple(judge.position for judge in judges)
    if positions != tuple(range(1, len(judges) + 1)) or not 1 <= len(judges) <= 3:
        raise ValueError("paired challenge judges must be ordered one through three")
    _require_active(cancel_requested)
    try:
        material_plan = plan_task_materials(
            author_client,
            author=author,
            coaching_digest=coaching_digest,
            workspace_digest=environment.identity.workspace_digest,
            workspace=authoring_snapshot,
        )
    except InferenceCancelled:
        raise
    except Exception as exc:
        return ChallengeResult(
            candidate_id=candidate_id,
            status="invalid_task",
            winner="tie",
            reason=f"Task material planning failed: {_safe_error(exc)}",
            task=None,
            execution=environment.identity,
            baseline=None,
            candidate=None,
            judges=(),
        )

    prepared_authoring_snapshot = authoring_snapshot
    initial_workspace = authoring_snapshot
    if prepare_pair is not None:
        try:
            prepared = prepare_pair(material_plan)
        except InferenceCancelled:
            raise
        except Exception as exc:
            return ChallengeResult(
                candidate_id=candidate_id,
                status="invalid_task",
                winner="tie",
                reason=f"Task preparation failed: {_safe_error(exc)}",
                task=None,
                execution=environment.identity,
                baseline=None,
                candidate=None,
                judges=(),
            )
        prepared_authoring_snapshot = prepared.authoring
        initial_workspace = prepared.arms.authoring
        baseline_snapshot = prepared.arms.baseline
        candidate_snapshot = prepared.arms.candidate
        baseline_runtime_files = prepared.arms.baseline_runtime_files
        candidate_runtime_files = prepared.arms.candidate_runtime_files
        environment = prepared.environment
        _require_active(cancel_requested)
    elif material_plan.materials:
        return ChallengeResult(
            candidate_id=candidate_id,
            status="invalid_task",
            winner="tie",
            reason="Task preparation failed: public materials require a preparation boundary.",
            task=None,
            execution=environment.identity,
            baseline=None,
            candidate=None,
            judges=(),
        )

    task: AuthoredTask | None = None
    try:
        task = author_task(
            author_client,
            author=author,
            coaching_digest=coaching_digest,
            material_plan=material_plan,
            workspace_digest=environment.identity.workspace_digest,
            workspace=prepared_authoring_snapshot,
            initial_workspace=initial_workspace,
        )
        validate_task_workspace(initial_workspace, task)
    except InferenceCancelled:
        raise
    except TaskPreflightError as exc:
        return ChallengeResult(
            candidate_id=candidate_id,
            status="invalid_task",
            winner="tie",
            reason=f"Task authoring or workspace preflight failed: {_safe_error(exc)}",
            task=task,
            execution=environment.identity,
            baseline=None,
            candidate=None,
            judges=(),
        )
    except Exception as exc:
        return ChallengeResult(
            candidate_id=candidate_id,
            status="invalid_task",
            winner="tie",
            reason=f"Task authoring failed: {_safe_error(exc)}",
            task=task,
            execution=environment.identity,
            baseline=None,
            candidate=None,
            judges=(),
        )

    if task is None:
        raise AssertionError("successful task authoring did not return a task")
    _require_active(cancel_requested)
    try:
        runner.preflight(
            workspace=initial_workspace,
            task=task,
            environment=environment,
            cancel_requested=cancel_requested,
        )
    except InferenceCancelled:
        raise
    except TaskPreflightError as exc:
        return ChallengeResult(
            candidate_id=candidate_id,
            status="invalid_task",
            winner="tie",
            reason=f"Task capability preflight failed: {_safe_error(exc)}",
            task=task,
            execution=environment.identity,
            baseline=None,
            candidate=None,
            judges=(),
        )
    except Exception as exc:
        return ChallengeResult(
            candidate_id=candidate_id,
            status="incomplete",
            winner="tie",
            reason=f"Sandbox preflight infrastructure failed: {_safe_error(exc)}",
            task=task,
            execution=environment.identity,
            baseline=None,
            candidate=None,
            judges=(),
        )

    baseline_result, candidate_result = runner.run_pair(
        baseline=baseline_snapshot,
        candidate=candidate_snapshot,
        task=task,
        environment=environment,
        baseline_runtime_files=baseline_runtime_files,
        candidate_runtime_files=candidate_runtime_files,
        cancel_requested=cancel_requested,
    )
    _require_active(cancel_requested)
    if any(arm.status == "infrastructure_failed" for arm in (baseline_result, candidate_result)):
        return ChallengeResult(
            candidate_id=candidate_id,
            status="incomplete",
            winner="tie",
            reason="Paired sandbox infrastructure did not complete both arms.",
            task=task,
            execution=environment.identity,
            baseline=baseline_result,
            candidate=candidate_result,
            judges=(),
        )

    order = _presentation_order(candidate_id)
    instruction_paths, instruction_contents = _blinded_instruction_evidence(
        baseline_snapshot,
        candidate_snapshot,
        baseline_runtime_files,
        candidate_runtime_files,
    )
    try:
        verdicts = []
        for judge in judges:
            _require_active(cancel_requested)
            verdicts.append(
                judge_pair(
                    judge_clients[judge.id],
                    judge=judge,
                    task=task,
                    baseline=baseline_result,
                    candidate=candidate_result,
                    rubrics=rubrics,
                    presentation_order=order,
                    blinded_instruction_paths=instruction_paths,
                    blinded_instruction_contents=instruction_contents,
                )
            )
        verdicts = tuple(verdicts)
    except InferenceCancelled:
        raise
    except Exception as exc:
        return ChallengeResult(
            candidate_id=candidate_id,
            status="incomplete",
            winner="tie",
            reason=f"Paired judging did not complete: {_safe_error(exc)}",
            task=task,
            execution=environment.identity,
            baseline=baseline_result,
            candidate=candidate_result,
            judges=(),
        )
    invalid_count = sum(not verdict.task_valid for verdict in verdicts)
    if invalid_count >= len(verdicts) // 2 + 1:
        reasons = sorted(
            {
                verdict.task_invalid_reason
                for verdict in verdicts
                if verdict.task_invalid_reason is not None
            }
        )
        return ChallengeResult(
            candidate_id=candidate_id,
            status="invalid_task",
            winner="tie",
            reason="Task was not comparable: " + " ".join(reasons),
            task=task,
            execution=environment.identity,
            baseline=baseline_result,
            candidate=candidate_result,
            judges=verdicts,
        )
    return ChallengeResult(
        candidate_id=candidate_id,
        status="complete",
        winner=_panel_winner(verdicts),
        reason=None,
        task=task,
        execution=environment.identity,
        baseline=baseline_result,
        candidate=candidate_result,
        judges=verdicts,
    )
