"""Closed, content-authenticated evidence contracts for paired verification."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

ArmName = Literal["baseline", "candidate"]
Winner = Literal["baseline", "candidate", "tie"]
ChallengeStatus = Literal["complete", "incomplete", "invalid_task"]
ArmStatus = Literal["exited", "timed_out", "infrastructure_failed"]
_PUBLIC_GIT_HOSTS = frozenset({"github.com", "gitlab.com", "bitbucket.org", "codeberg.org"})


class TaskPreflightError(RuntimeError):
    """An authored task requirement is unavailable before paired execution."""


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _nonblank(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must be nonblank")
    return value


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        return cls.model_validate(value)


class NamedDigest(_Contract):
    name: StrictStr
    digest: StrictStr

    @model_validator(mode="after")
    def validate_value(self) -> NamedDigest:
        _nonblank(self.name, "digest name")
        _nonblank(self.digest, "digest")
        return self


class ExecutionIdentity(_Contract):
    schema_version: Literal["1"] = "1"
    execution_id: StrictStr = "pending"
    model: StrictStr
    model_family: StrictStr
    harness: Literal["codex", "claude"]
    harness_version: StrictStr
    effort: StrictStr | None
    image: StrictStr
    image_digest: StrictStr
    command: tuple[StrictStr, ...]
    timeout_seconds: Annotated[int, Field(strict=True, ge=1)]
    network_enabled: bool
    environment: tuple[NamedDigest, ...]
    runtime_files: tuple[NamedDigest, ...]
    workspace_digest: StrictStr

    @model_validator(mode="after")
    def validate_identity(self) -> ExecutionIdentity:
        for name in (
            "model",
            "model_family",
            "harness_version",
            "image",
            "image_digest",
            "workspace_digest",
        ):
            _nonblank(getattr(self, name), name)
        if self.effort is not None:
            _nonblank(self.effort, "effort")
        if not self.command or any(not item.strip() for item in self.command):
            raise ValueError("command must contain nonblank arguments")
        for name, values in (
            ("environment variables", self.environment),
            ("runtime files", self.runtime_files),
        ):
            names = tuple(item.name for item in values)
            if names != tuple(sorted(names)) or len(names) != len(set(names)):
                raise ValueError(f"{name} must be uniquely sorted by name")
        expected = canonical_digest(self.model_dump(mode="json", exclude={"execution_id"}))
        if self.execution_id not in {"pending", expected}:
            raise ValueError("execution identity digest mismatch")
        object.__setattr__(self, "execution_id", expected)
        return self


class TaskMaterial(_Contract):
    kind: Literal["git_repository"] = "git_repository"
    url: StrictStr
    revision: StrictStr
    destination: StrictStr

    @model_validator(mode="after")
    def validate_material(self) -> TaskMaterial:
        parsed = urlparse(self.url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
            raise ValueError("material url must be a public https URL without credentials")
        hostname = parsed.hostname
        if hostname is None or hostname.lower() not in _PUBLIC_GIT_HOSTS:
            raise ValueError("material url must use a supported public Git host")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            raise ValueError("material revision must be a full 40-character Git commit SHA")
        destination = PurePosixPath(self.destination)
        if (
            destination.is_absolute()
            or not destination.parts
            or any(part in {"", ".", ".."} for part in destination.parts)
        ):
            raise ValueError("material destination must be a safe relative directory")
        object.__setattr__(self, "revision", self.revision.lower())
        object.__setattr__(self, "destination", destination.as_posix())
        return self


class TaskMaterialPlan(_Contract):
    schema_version: Literal["1"] = "1"
    plan_id: StrictStr = "pending"
    setup_mode: Literal["prepared_workspace", "agent_bootstrap"]
    materials: Annotated[tuple[TaskMaterial, ...], Field(max_length=3)] = ()

    @model_validator(mode="after")
    def validate_plan(self) -> TaskMaterialPlan:
        destinations = tuple(item.destination for item in self.materials)
        if len(destinations) != len(set(destinations)):
            raise ValueError("material destinations must be unique")
        expected = canonical_digest(self.model_dump(mode="json", exclude={"plan_id"}))
        if self.plan_id not in {"pending", expected}:
            raise ValueError("material plan identity digest mismatch")
        object.__setattr__(self, "plan_id", expected)
        return self


class AuthoredTask(_Contract):
    schema_version: Literal["3"] = "3"
    task_id: StrictStr = "pending"
    prompt: StrictStr
    goal: StrictStr
    setup_mode: Literal["prepared_workspace", "agent_bootstrap"] = "agent_bootstrap"
    materials: Annotated[tuple[TaskMaterial, ...], Field(max_length=3)] = ()
    judging_criteria: Annotated[tuple[StrictStr, ...], Field(min_length=1, max_length=8)] = (
        "The stated goal is achieved.",
    )
    start_checks: Annotated[tuple[StrictStr, ...], Field(min_length=1, max_length=8)] = (
        "The supplied workspace is available.",
    )
    required_files: Annotated[tuple[StrictStr, ...], Field(max_length=32)] = ()
    required_executables: Annotated[tuple[StrictStr, ...], Field(max_length=16)] = ()
    requires_git_metadata: bool = False
    workspace_digest: StrictStr
    author_model: StrictStr
    author_backend: StrictStr

    @model_validator(mode="after")
    def validate_task(self) -> AuthoredTask:
        for name in ("prompt", "goal", "workspace_digest", "author_model", "author_backend"):
            _nonblank(getattr(self, name), name)
        if not self.judging_criteria or any(not item.strip() for item in self.judging_criteria):
            raise ValueError("judging criteria must contain nonblank values")
        if not self.start_checks or any(not item.strip() for item in self.start_checks):
            raise ValueError("start checks must contain nonblank values")
        normalized_files: list[str] = []
        for value in self.required_files:
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or not path.parts
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError("required files must use safe relative paths")
            if ".git" in path.parts:
                raise ValueError("required files cannot depend on Git metadata")
            normalized_files.append(path.as_posix())
        executables = tuple(self.required_executables)
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", item) for item in executables):
            raise ValueError("required executables must use a bare executable name")
        if self.setup_mode == "prepared_workspace" and self.requires_git_metadata:
            raise ValueError("prepared workspaces cannot require Git metadata")
        for name, values in (
            ("material destinations", tuple(item.destination for item in self.materials)),
            ("judging criteria", self.judging_criteria),
            ("start checks", self.start_checks),
            ("required files", tuple(normalized_files)),
            ("required executables", executables),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        object.__setattr__(self, "required_files", tuple(normalized_files))
        expected = canonical_digest(self.model_dump(mode="json", exclude={"task_id"}))
        if self.task_id not in {"pending", expected}:
            raise ValueError("task identity digest mismatch")
        object.__setattr__(self, "task_id", expected)
        return self


class ArtifactChange(_Contract):
    path: StrictStr
    action: Literal["added", "modified", "deleted"]
    before_digest: StrictStr | None
    after_digest: StrictStr | None
    diff: Annotated[StrictStr, Field(max_length=40_000)]

    @model_validator(mode="after")
    def validate_change(self) -> ArtifactChange:
        _nonblank(self.path, "artifact path")
        _nonblank(self.diff, "artifact diff")
        if self.action == "added" and (self.before_digest is not None or self.after_digest is None):
            raise ValueError("added artifact requires only an after digest")
        if self.action == "deleted" and (
            self.before_digest is None or self.after_digest is not None
        ):
            raise ValueError("deleted artifact requires only a before digest")
        if self.action == "modified" and (self.before_digest is None or self.after_digest is None):
            raise ValueError("modified artifact requires before and after digests")
        if self.before_digest is not None:
            _nonblank(self.before_digest, "artifact before digest")
        if self.after_digest is not None:
            _nonblank(self.after_digest, "artifact after digest")
        return self


class ArmResult(_Contract):
    arm: ArmName
    status: ArmStatus
    exit_code: Annotated[int, Field(strict=True)] | None
    duration_seconds: Annotated[float, Field(ge=0)]
    initial_workspace_digest: StrictStr
    final_workspace_digest: StrictStr | None
    transcript: StrictStr
    transcript_digest: StrictStr
    artifact_digests: dict[StrictStr, StrictStr]
    artifact_changes: tuple[ArtifactChange, ...] = ()
    infrastructure_error: StrictStr | None = None

    @model_validator(mode="after")
    def validate_result(self) -> ArmResult:
        _nonblank(self.initial_workspace_digest, "initial workspace digest")
        _nonblank(self.transcript_digest, "transcript digest")
        if self.status == "infrastructure_failed":
            if self.exit_code is not None or self.final_workspace_digest is not None:
                raise ValueError("infrastructure failure cannot contain agent exit evidence")
            if self.artifact_digests:
                raise ValueError("infrastructure failure cannot contain agent artifacts")
            if self.artifact_changes:
                raise ValueError("infrastructure failure cannot contain artifact changes")
            if self.infrastructure_error is None:
                raise ValueError("infrastructure failure requires a safe error")
            _nonblank(self.infrastructure_error, "infrastructure error")
            return self
        if self.infrastructure_error is not None:
            raise ValueError("behavioral arm result cannot contain an infrastructure error")
        if self.final_workspace_digest is None:
            raise ValueError("behavioral arm result requires a final workspace digest")
        _nonblank(self.final_workspace_digest, "final workspace digest")
        if self.status == "exited" and self.exit_code is None:
            raise ValueError("exited arm requires an exit code")
        if self.status == "timed_out" and self.exit_code is not None:
            raise ValueError("timed out arm cannot contain an exit code")
        if any(
            not path.strip() or not digest.strip() for path, digest in self.artifact_digests.items()
        ):
            raise ValueError("artifact digests must contain nonblank paths and values")
        changed_paths = tuple(item.path for item in self.artifact_changes)
        if len(changed_paths) != len(set(changed_paths)):
            raise ValueError("artifact change paths must be unique")
        for change in self.artifact_changes:
            if change.after_digest is not None and self.artifact_digests.get(change.path) != (
                change.after_digest
            ):
                raise ValueError("artifact change after digest must match final artifacts")
        return self


class RubricVerdict(_Contract):
    rubric_id: StrictStr
    baseline_score: Annotated[float, Field(ge=0, le=1)]
    candidate_score: Annotated[float, Field(ge=0, le=1)]
    winner: Winner
    rationale: StrictStr
    delta: Annotated[float, Field(ge=-1, le=1)] | None = None

    @model_validator(mode="after")
    def validate_verdict(self) -> RubricVerdict:
        _nonblank(self.rubric_id, "rubric_id")
        _nonblank(self.rationale, "rationale")
        expected: Winner
        if math.isclose(self.baseline_score, self.candidate_score, abs_tol=1e-12):
            expected = "tie"
        elif self.candidate_score > self.baseline_score:
            expected = "candidate"
        else:
            expected = "baseline"
        if self.winner != expected:
            raise ValueError("rubric winner does not match its scores")
        expected_delta = self.candidate_score - self.baseline_score
        if self.delta is not None and not math.isclose(
            self.delta,
            expected_delta,
            abs_tol=1e-12,
        ):
            raise ValueError("rubric delta does not match its scores")
        object.__setattr__(self, "delta", expected_delta)
        return self


class JudgeVerdict(_Contract):
    position: Annotated[int, Field(strict=True, ge=1, le=3)]
    requested_model: StrictStr
    resolved_model: StrictStr
    family: StrictStr
    backend: StrictStr
    baseline_label: StrictStr
    candidate_label: StrictStr
    task_valid: bool = True
    task_invalid_reason: StrictStr | None = None
    winner: Winner
    rationale: StrictStr
    rubrics: tuple[RubricVerdict, ...]
    usage: dict[StrictStr, Annotated[int, Field(strict=True, ge=0)]]

    @model_validator(mode="after")
    def validate_verdict(self) -> JudgeVerdict:
        for name in (
            "requested_model",
            "resolved_model",
            "family",
            "backend",
            "baseline_label",
            "candidate_label",
            "rationale",
        ):
            _nonblank(getattr(self, name), name)
        if self.baseline_label == self.candidate_label:
            raise ValueError("blinded arm labels must be distinct")
        if self.task_valid:
            if self.task_invalid_reason is not None:
                raise ValueError("valid task verdict cannot contain an invalid reason")
        else:
            if self.task_invalid_reason is None:
                raise ValueError("invalid task verdict requires a reason")
            _nonblank(self.task_invalid_reason, "task invalid reason")
            if self.winner != "tie":
                raise ValueError("invalid task verdict cannot select an arm")
        if not self.rubrics:
            raise ValueError("judge verdict must contain at least one rubric")
        rubric_ids = tuple(item.rubric_id for item in self.rubrics)
        if len(rubric_ids) != len(set(rubric_ids)):
            raise ValueError("judge rubric IDs must be unique")
        return self


class ChallengeResult(_Contract):
    schema_version: Literal["1"] = "1"
    challenge_id: StrictStr = "pending"
    candidate_id: StrictStr
    status: ChallengeStatus
    winner: Winner
    reason: StrictStr | None
    task: AuthoredTask | None
    execution: ExecutionIdentity
    baseline: ArmResult | None
    candidate: ArmResult | None
    judges: tuple[JudgeVerdict, ...]

    @model_validator(mode="after")
    def validate_result(self) -> ChallengeResult:
        _nonblank(self.candidate_id, "candidate_id")
        positions = tuple(item.position for item in self.judges)
        if positions != tuple(range(1, len(self.judges) + 1)):
            raise ValueError("judge positions must be ordered, contiguous, and 1-based")
        if self.status == "complete":
            if self.reason is not None:
                raise ValueError("complete challenge cannot contain a failure reason")
            if self.task is None or self.baseline is None or self.candidate is None:
                raise ValueError("complete challenge requires task and both arm results")
            if self.baseline.arm != "baseline" or self.candidate.arm != "candidate":
                raise ValueError("challenge arm roles are inconsistent")
            if any(
                arm.status == "infrastructure_failed" for arm in (self.baseline, self.candidate)
            ):
                raise ValueError("complete challenge cannot contain infrastructure failure")
            if not self.judges:
                raise ValueError("complete challenge requires judge evidence")
            counts = Counter(judge.winner for judge in self.judges)
            majority = len(self.judges) // 2 + 1
            expected: Winner = "tie"
            for arm in ("baseline", "candidate"):
                if counts[arm] >= majority:
                    expected = arm
            if self.winner != expected:
                raise ValueError("challenge winner does not match the strict panel majority")
        else:
            if self.reason is None:
                raise ValueError("non-complete challenge requires a reason")
            _nonblank(self.reason, "reason")
            if self.winner != "tie":
                raise ValueError("non-complete challenge cannot declare an arm winner")
            if self.status == "incomplete" and self.judges:
                raise ValueError("incomplete challenge cannot contain judge verdicts")
            if self.status == "invalid_task" and self.judges:
                if self.task is None or self.baseline is None or self.candidate is None:
                    raise ValueError("judged invalid task requires task and both arm results")
                if not any(not judge.task_valid for judge in self.judges):
                    raise ValueError("invalid task challenge requires invalid-task judge evidence")
        expected_id = canonical_digest(self.model_dump(mode="json", exclude={"challenge_id"}))
        if self.challenge_id not in {"pending", expected_id}:
            raise ValueError("challenge identity digest mismatch")
        object.__setattr__(self, "challenge_id", expected_id)
        return self

    def recommendation(self) -> str | None:
        if self.status == "complete" and self.winner == "candidate":
            return self.candidate_id
        return None
