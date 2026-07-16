"""GEPA reflection over one exact managed instruction bundle."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal

from weave_agent_signals.judges.families import model_family
from weave_agent_signals.judges.inference import (
    ChatClient,
    InferenceCancelled,
    JudgeResponse,
)
from weave_agent_signals.patterns import is_evaluation_feedback_eligible
from weave_agent_signals.run_config import MAX_CANDIDATE_BUDGET, ModelDescriptor
from weave_agent_signals.runs.bundles import BundleSnapshot
from weave_agent_signals.runs.proposals import (
    REFLECTION_PROPOSAL_SCHEMA,
    CandidateProposal,
    materialize_candidate_proposal,
    parse_candidate_proposal,
    serialize_candidate_proposal,
)

log = logging.getLogger(__name__)

PREDICTED_EVALUATOR_SCORE_BASIS = "predicted_evaluator"
NO_IMPROVEMENT_PATIENCE = 2
NO_VALID_PROPOSAL_REASON = "No valid proposal generated"
MAX_REJECTED_RESPONSE_EXCERPT = 1_000
_CORRECTION_CONTEXT_LIMIT = 500
_GEPA_FENCE = "```"

_PROPOSAL_LAYOUT = """{
  "schema_version": 3,
  "changes": [
    {
      "action": "update",
      "locator": "file:project-agents",
      "content": "Complete replacement content for AGENTS.md"
    }
  ]
}"""
_GEPA_OBJECTIVE_TEMPLATE = (
    "Improve the complete managed instruction bundle to increase the agent's "
    "evaluation scores. Treat every target as part of one coherent configuration. "
    "The bundle may contain project instructions, commands, skills, and other "
    "adapter-managed prompt files.\n\n"
    "Current evaluation summary:\n{coaching_text}\n\n"
    "Propose specific, actionable edits to the lowest-scoring dimensions. Use "
    "this exact JSON layout:\n{proposal_layout}\n\n"
    "Contract rules:\n"
    "- `changes` contains only files changed by the proposal, and each locator "
    "appears once.\n"
    "- create requires a locator admitted by the pinned target registry and absent "
    "from baseline B, with complete string content.\n"
    "- update requires a locator present in baseline B and changed complete "
    "replacement string content.\n"
    "- The complete change list is one atomic multi-file proposal; every change "
    "must be valid together.\n"
    "- An empty or no-op writer proposal is invalid.\n"
    "- Return JSON only, with exactly `schema_version` and `changes` at the top "
    "level and exactly `action`, `locator`, and `content` for every change.\n"
    "- Do not change unrelated documentation merely to improve an evaluator score.\n"
)
_GEPA_BACKGROUND_TEMPLATE = (
    "The target adapter defines the complete managed instruction scope. "
    "Instructions should be clear, specific, non-redundant, and focused on "
    "verification, error handling, and effective tool use. Human review gates "
    "every promotion.\n\n"
    "Pinned baseline B complete inventory and contents:\n{baseline_contents}\n\n"
    "Pinned scope policy:\n{scope_policy}\n"
)
_EVALUATOR_SYSTEM = (
    "Predict how well a complete proposed managed instruction bundle will improve "
    "the coding agent's measured behavior. Score the bundle from 0.0 to 1.0 and "
    "briefly justify the score. Do not reward changes to unrelated documentation "
    "that cannot plausibly affect the evaluated agent behavior."
)
_EVALUATOR_USER = (
    "## Evaluation evidence\n\n{coaching_text}\n\n"
    "## Complete managed instruction bundle\n\n{bundle_text}\n\n"
    "Respond with JSON: "
    '{{"score": <float 0.0-1.0>, "rationale": "<brief explanation>"}}'
)

ProgressCallback = Callable[[dict[str, Any]], None]
LocatorResolver = Callable[..., object]
Optimizer = Callable[..., Any]
AttemptStatus = Literal["succeeded", "failed", "cancelled"]


class ReflectionEvaluationError(RuntimeError):
    """A generated bundle could not be assigned trustworthy evaluation evidence."""


class ReflectionCancelled(RuntimeError):
    """Reflection stopped because its owning run was cancelled."""


class _ProposalRejected(ReflectionEvaluationError):
    """GEPA may continue after one invalid writer response."""


class _DuplicateRevisionError(ValueError):
    """A writer response materialized to an already-seen immutable revision."""


def _gepa_safe_candidate(candidate: str) -> str:
    """Protect Markdown fences from GEPA's generic instruction extractor."""

    if _GEPA_FENCE not in candidate:
        return candidate
    return f"{_GEPA_FENCE}\n{candidate}\n{_GEPA_FENCE}"


def _unwrap_gepa_candidate(candidate: str) -> str:
    prefix = f"{_GEPA_FENCE}\n"
    suffix = f"\n{_GEPA_FENCE}"
    if candidate.startswith(prefix) and candidate.endswith(suffix):
        return candidate[len(prefix) : -len(suffix)]
    return candidate


def _nonblank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")
    return value


def _score(value: object, field_name: str = "score") -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field_name} must be a finite number from 0.0 to 1.0")
    return float(value)


def _usage(value: Mapping[str, Any] | None) -> Mapping[str, int]:
    """Keep portable top-level token counters from provider-specific usage."""

    copied: dict[str, int] = {}
    for key, count in (value or {}).items():
        if not isinstance(key, str) or not key:
            raise ValueError("usage keys must be nonblank strings")
        if isinstance(count, Mapping):
            continue
        if type(count) is not int or count < 0:
            raise ValueError("usage values must be non-negative integers")
        copied[key] = count
    return MappingProxyType(copied)


def _strict_data(value: Mapping[str, Any], model: type[Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(model)}:
        raise ValueError(f"invalid {label}")
    return dict(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, BundleSnapshot):
        return value.to_dict()
    if isinstance(value, ModelDescriptor):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _model_dict(value)
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _model_dict(value: Any) -> dict[str, Any]:
    return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:[a-z0-9_.-]*(?:api[_-]?key|token|secret|password|authorization|credential)"
    r"[a-z0-9_.-]*)\b\s*[:=]\s*)(?:\\?[\"']?)([^\s,}\]\\\"']+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[^\s,}\]\\\"']+")
_URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@")
_JUDGE_PROCESS_FAILURE = re.compile(r"(?i)\b[\w.-]+\s+judge exited\s+(-?\d+)\b")
_PROPOSAL_FIELD_NAMES = ("schema_version", "changes", "action", "locator", "content")
_SAFE_PROPOSAL_STRING_FIELDS = frozenset({"action", "locator"})


def _sanitized_text(value: object, *, limit: int) -> str:
    """Bound untrusted diagnostics and redact common credential forms."""

    text = "".join(
        character if character.isprintable() or character in "\n\r\t" else " "
        for character in str(value)
    )
    text = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    text = _BEARER_SECRET.sub("Bearer [REDACTED]", text)
    text = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", text)
    return text[:limit]


def _redacted_proposal_value(value: Any, *, field_name: str | None = None) -> Any:
    """Preserve proposal structure while removing all non-audit string bodies."""

    if isinstance(value, Mapping):
        return {
            key: (
                "[REDACTED]"
                if isinstance(key, str) and key.casefold() == "content"
                else _redacted_proposal_value(
                    item,
                    field_name=key.casefold() if isinstance(key, str) else None,
                )
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redacted_proposal_value(item) for item in value]
    if isinstance(value, str) and field_name not in _SAFE_PROPOSAL_STRING_FIELDS:
        return "[REDACTED]"
    return value


def _malformed_proposal_excerpt(response: str) -> str:
    detected_fields = [
        field
        for field in _PROPOSAL_FIELD_NAMES
        if re.search(
            rf"(?i)(?:[\"']\s*)?{re.escape(field)}(?:\s*[\"'])?\s*:",
            response,
        )
    ]
    fields = ",".join(detected_fields) if detected_fields else "unknown"
    return (
        "[malformed proposal output redacted; "
        f"characters={len(response)}; lines={response.count(chr(10)) + 1}; "
        f"detected_fields={fields}]"
    )[:MAX_REJECTED_RESPONSE_EXCERPT]


def _sanitized_proposal_excerpt(response: str) -> str:
    """Return bounded structural audit evidence without retaining proposal bodies."""

    try:
        parsed = json.loads(response)
        redacted = _redacted_proposal_value(parsed)
        structural = json.dumps(
            redacted,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
        return _malformed_proposal_excerpt(response)
    return _sanitized_text(structural, limit=MAX_REJECTED_RESPONSE_EXCERPT)


def _error_text(error: BaseException) -> str:
    return (
        _sanitized_text(error, limit=MAX_REJECTED_RESPONSE_EXCERPT).strip() or type(error).__name__
    )


def _actionable_failure(error: BaseException, *, role: str) -> str:
    """Keep the failure category actionable without persisting process output."""

    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return f"{role} transport returned HTTP {status_code}"
    if type(error).__name__ == "TimeoutExpired":
        return f"{role} invocation timed out"
    process_failure = _JUDGE_PROCESS_FAILURE.search(str(error))
    if process_failure is not None:
        return f"{role} process exited {process_failure.group(1)}"
    return _error_text(error)


def _response_digest(response: str) -> str:
    return f"sha256:{hashlib.sha256(response.encode()).hexdigest()}"


def _best_effort_changed_paths(response: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError, ValueError):
        return ()
    changes = parsed.get("changes") if isinstance(parsed, Mapping) else None
    if not isinstance(changes, list):
        return ()
    return tuple(
        sorted(
            {
                locator
                for item in changes
                if isinstance(item, Mapping)
                and isinstance((locator := item.get("locator")), str)
                and locator
            }
        )
    )


def _is_proposal_validation_error(error: Exception) -> bool:
    if isinstance(error, (json.JSONDecodeError, TypeError, ValueError)):
        return True
    return type(error).__name__.endswith("ValidationError") and isinstance(
        getattr(error, "changed_locators", None), tuple
    )


def _error_changed_paths(error: Exception, fallback: tuple[str, ...]) -> tuple[str, ...]:
    paths = getattr(error, "changed_locators", ())
    if isinstance(paths, tuple) and all(isinstance(path, str) and path for path in paths):
        return tuple(sorted(set((*fallback, *paths))))
    return fallback


@dataclass(frozen=True)
class EvaluatorRecord:
    evaluation_id: str
    target_revision: str
    requested_model: str
    requested_family: str
    requested_backend: str
    resolved_model: str
    resolved_family: str
    resolved_backend: str
    score: float
    rationale: str
    usage: Mapping[str, int]

    def __post_init__(self) -> None:
        for name in (
            "evaluation_id",
            "target_revision",
            "requested_model",
            "requested_family",
            "requested_backend",
            "resolved_model",
            "resolved_family",
            "resolved_backend",
        ):
            _nonblank(getattr(self, name), name)
        if not isinstance(self.rationale, str):
            raise ValueError("rationale must be a string")
        object.__setattr__(self, "score", _score(self.score))
        object.__setattr__(self, "usage", _usage(self.usage))

    def to_dict(self) -> dict[str, Any]:
        return _model_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluatorRecord:
        return cls(**_strict_data(value, cls, "evaluator record"))


def _evaluator_identity(record: EvaluatorRecord) -> tuple[str, ...]:
    return (
        record.requested_model,
        record.requested_family,
        record.requested_backend,
        record.resolved_model,
        record.resolved_family,
        record.resolved_backend,
    )


@dataclass(frozen=True)
class GenerationAttempt:
    attempt_id: str
    number: int
    status: AttemptStatus
    requested_writer: ModelDescriptor
    resolved_model: str | None
    resolved_family: str | None
    resolved_backend: str | None
    usage: Mapping[str, int]
    candidate_revision: str | None
    changed_paths: tuple[str, ...]
    response_digest: str | None
    response_excerpt: str | None
    error_type: str | None
    error: str | None

    def __post_init__(self) -> None:
        _nonblank(self.attempt_id, "attempt_id")
        if type(self.number) is not int or self.number < 1:
            raise ValueError("attempt number must be a positive integer")
        if self.status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("invalid generation-attempt status")
        if not isinstance(self.requested_writer, ModelDescriptor):
            raise ValueError("requested_writer must be a ModelDescriptor")
        if "proposal_writer" not in self.requested_writer.supported_roles:
            raise ValueError("requested_writer does not support proposal generation")
        for name in (
            "resolved_model",
            "resolved_family",
            "resolved_backend",
            "candidate_revision",
        ):
            value = getattr(self, name)
            if value is not None:
                _nonblank(value, name)
        changed_paths = tuple(self.changed_paths)
        if any(not isinstance(path, str) or not path for path in changed_paths):
            raise ValueError("changed_paths must contain nonblank strings")
        if len(changed_paths) != len(set(changed_paths)):
            raise ValueError("changed_paths must be unique")
        object.__setattr__(self, "changed_paths", tuple(sorted(changed_paths)))
        if self.response_digest is not None and not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.response_digest
        ):
            raise ValueError("response_digest must be a SHA-256 digest or null")
        if self.response_excerpt is not None and (
            not isinstance(self.response_excerpt, str)
            or len(self.response_excerpt) > MAX_REJECTED_RESPONSE_EXCERPT
        ):
            raise ValueError("response_excerpt must be a bounded string or null")
        for name in ("error_type", "error"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null")
        if self.status == "succeeded" and (
            self.resolved_model is None
            or self.resolved_family is None
            or self.resolved_backend is None
            or self.candidate_revision is None
            or self.error_type is not None
            or self.error is not None
            or self.response_digest is not None
            or self.response_excerpt is not None
        ):
            raise ValueError("successful attempts require resolved candidate provenance")
        if self.status != "succeeded" and (not self.error_type or not self.error):
            raise ValueError("unsuccessful attempts require error_type and error")
        object.__setattr__(self, "usage", _usage(self.usage))

    def to_dict(self) -> dict[str, Any]:
        return _model_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationAttempt:
        data = _strict_data(value, cls, "generation attempt")
        data["requested_writer"] = ModelDescriptor.model_validate(data["requested_writer"])
        return cls(**data)


@dataclass(frozen=True)
class ReflectionCandidate:
    candidate_id: str
    bundle: BundleSnapshot
    score: float
    score_delta: float
    rationale: str
    generation_attempt_id: str
    requested_writer: ModelDescriptor
    resolved_writer_model: str
    resolved_writer_family: str
    resolved_writer_backend: str
    evaluation: EvaluatorRecord

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "generation_attempt_id",
            "resolved_writer_model",
            "resolved_writer_family",
            "resolved_writer_backend",
        ):
            _nonblank(getattr(self, name), name)
        if not isinstance(self.bundle, BundleSnapshot):
            raise ValueError("candidate bundle must be a BundleSnapshot")
        if not isinstance(self.rationale, str):
            raise ValueError("candidate rationale must be a string")
        if not isinstance(self.requested_writer, ModelDescriptor):
            raise ValueError("requested_writer must be a ModelDescriptor")
        object.__setattr__(self, "score", _score(self.score))
        if (
            isinstance(self.score_delta, bool)
            or not isinstance(self.score_delta, (int, float))
            or not math.isfinite(float(self.score_delta))
        ):
            raise ValueError("score_delta must be finite")
        object.__setattr__(self, "score_delta", float(self.score_delta))
        if not isinstance(self.evaluation, EvaluatorRecord):
            raise ValueError("candidate must retain one evaluator record")
        if self.evaluation.target_revision != self.bundle.revision:
            raise ValueError("candidate evaluator record targets another revision")

    def to_dict(self) -> dict[str, Any]:
        return _model_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReflectionCandidate:
        data = _strict_data(value, cls, "reflection candidate")
        data["bundle"] = BundleSnapshot.from_dict(data["bundle"])
        data["requested_writer"] = ModelDescriptor.model_validate(data["requested_writer"])
        data["evaluation"] = EvaluatorRecord.from_dict(data["evaluation"])
        return cls(**data)


@dataclass(frozen=True)
class ReflectionResult:
    baseline: BundleSnapshot
    baseline_score: float
    baseline_evaluation: EvaluatorRecord
    candidates: tuple[ReflectionCandidate, ...]
    generation_attempts: tuple[GenerationAttempt, ...]
    recommended_candidate_id: str | None
    baseline_won: bool
    reason: str | None = None
    score_basis: str = PREDICTED_EVALUATOR_SCORE_BASIS

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, BundleSnapshot):
            raise ValueError("baseline must be a BundleSnapshot")
        object.__setattr__(self, "baseline_score", _score(self.baseline_score))
        candidates = tuple(self.candidates)
        attempts = tuple(self.generation_attempts)
        if not isinstance(self.baseline_evaluation, EvaluatorRecord):
            raise ValueError("baseline must retain one evaluator record")
        if self.baseline_evaluation.target_revision != self.baseline.revision:
            raise ValueError("baseline evaluator record targets another revision")
        if not math.isclose(
            self.baseline_evaluation.score,
            self.baseline_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("baseline score has no matching evaluator record")
        candidate_ids = tuple(item.candidate_id for item in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique")
        attempt_ids = tuple(item.attempt_id for item in attempts)
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("generation attempt IDs must be unique")
        attempts_by_id = {item.attempt_id: item for item in attempts}
        for candidate in candidates:
            attempt = attempts_by_id.get(candidate.generation_attempt_id)
            if attempt is None or attempt.status != "succeeded":
                raise ValueError("candidate has no successful generation attempt")
            if attempt.candidate_revision != candidate.bundle.revision:
                raise ValueError("candidate revision does not match its attempt")
            if candidate.requested_writer != attempt.requested_writer:
                raise ValueError("candidate writer does not match its attempt")
            if (
                candidate.resolved_writer_model,
                candidate.resolved_writer_family,
                candidate.resolved_writer_backend,
            ) != (
                attempt.resolved_model,
                attempt.resolved_family,
                attempt.resolved_backend,
            ):
                raise ValueError("candidate writer provenance does not match its attempt")
            if not math.isclose(
                candidate.evaluation.score,
                candidate.score,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("candidate score has no matching evaluator record")
            if candidate.rationale != candidate.evaluation.rationale:
                raise ValueError("candidate has no matching evaluator rationale")
            if _evaluator_identity(candidate.evaluation) != _evaluator_identity(
                self.baseline_evaluation
            ):
                raise ValueError(
                    "candidate evaluation does not match the authoritative "
                    "baseline evaluator identity"
                )
            if not math.isclose(
                candidate.score_delta,
                candidate.score - self.baseline_score,
                abs_tol=1e-12,
            ):
                raise ValueError("candidate score delta does not match baseline")
        if self.recommended_candidate_id is not None:
            _nonblank(self.recommended_candidate_id, "recommended_candidate_id")
            if self.recommended_candidate_id not in candidate_ids:
                raise ValueError("recommended candidate is not present")
        if type(self.baseline_won) is not bool:
            raise ValueError("baseline_won must be boolean")
        if self.reason is not None:
            if self.reason != NO_VALID_PROPOSAL_REASON:
                raise ValueError("unsupported reflection result reason")
            if candidates or self.recommended_candidate_id is not None or self.baseline_won:
                raise ValueError("no-valid-proposal result has inconsistent selection evidence")
        elif self.baseline_won != (self.recommended_candidate_id is None):
            raise ValueError("baseline winner and recommendation disagree")
        if self.score_basis != PREDICTED_EVALUATOR_SCORE_BASIS:
            raise ValueError("unsupported reflection score basis")
        successful_attempt_ids = {
            attempt.attempt_id for attempt in attempts if attempt.status == "succeeded"
        }
        if successful_attempt_ids != {candidate.generation_attempt_id for candidate in candidates}:
            raise ValueError("successful attempts and evaluated candidates disagree")
        revisions = [self.baseline.revision]
        revisions.extend(candidate.bundle.revision for candidate in candidates)
        if len(revisions) != len(set(revisions)):
            raise ValueError("baseline and candidate revisions must be unique")
        evaluation_ids = [self.baseline_evaluation.evaluation_id]
        evaluation_ids.extend(candidate.evaluation.evaluation_id for candidate in candidates)
        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise ValueError("evaluator record IDs must be unique")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "generation_attempts", attempts)

    def to_dict(self) -> dict[str, Any]:
        return _model_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReflectionResult:
        data = _strict_data(value, cls, "reflection result")
        data["baseline"] = BundleSnapshot.from_dict(data["baseline"])
        data["baseline_evaluation"] = EvaluatorRecord.from_dict(data["baseline_evaluation"])
        data["candidates"] = tuple(
            ReflectionCandidate.from_dict(item) for item in data["candidates"]
        )
        data["generation_attempts"] = tuple(
            GenerationAttempt.from_dict(item) for item in data["generation_attempts"]
        )
        return cls(**data)


def _has_eligible_signal(feedback: Sequence[Mapping[str, Any]]) -> bool:
    return any(is_evaluation_feedback_eligible(item) for item in feedback)


def _emit(callback: ProgressCallback | None, phase: str, message: str, **details: Any) -> None:
    if callback is None:
        return
    try:
        callback({"phase": phase, "message": message, **details})
    except Exception as exc:
        log.warning("Reflection progress callback failed during %s: %s", phase, exc)


def _content_map(bundle: BundleSnapshot) -> dict[str, str]:
    return {
        target.locator: target.content
        for target in bundle.targets
        if target.exists and target.content is not None
    }


def _prompt_json(value: Mapping[str, Any], label: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    try:
        return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain JSON values") from exc


def _candidate_id(attempt_id: str, revision: str) -> str:
    digest = hashlib.sha256(f"{attempt_id}\n{revision}".encode()).hexdigest()
    return f"candidate-{digest}"


def _messages(prompt: str | list[dict[str, Any]]) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    messages: list[dict[str, str]] = []
    for item in prompt:
        if not isinstance(item, Mapping):
            raise ValueError("writer prompt messages must be objects")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("writer prompt messages require string role and content")
        messages.append({"role": role, "content": content})
    return messages


@dataclass
class _AttemptState:
    attempt_id: str
    number: int
    requested_writer: ModelDescriptor
    status: AttemptStatus | None = None
    resolved_model: str | None = None
    resolved_family: str | None = None
    resolved_backend: str | None = None
    usage: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    bundle: BundleSnapshot | None = None
    changed_paths: tuple[str, ...] = ()
    response_digest: str | None = None
    response_excerpt: str | None = None
    error_type: str | None = None
    error: str | None = None
    rejected: bool = False

    def freeze(self) -> GenerationAttempt:
        status = self.status or "failed"
        error_type = self.error_type
        error = self.error
        if self.status is None:
            error_type = "writer_attempt_incomplete"
            error = "Writer attempt did not finish"
        return GenerationAttempt(
            attempt_id=self.attempt_id,
            number=self.number,
            status=status,
            requested_writer=self.requested_writer,
            resolved_model=self.resolved_model,
            resolved_family=self.resolved_family,
            resolved_backend=self.resolved_backend,
            usage=self.usage,
            candidate_revision=self.bundle.revision if self.bundle else None,
            changed_paths=self.changed_paths,
            response_digest=self.response_digest,
            response_excerpt=self.response_excerpt,
            error_type=error_type,
            error=error,
        )


@dataclass(frozen=True)
class _CachedEvaluation:
    record: EvaluatorRecord
    side_info: dict[str, Any]


class _AttemptLedger:
    def __init__(
        self,
        baseline: BundleSnapshot,
        writer: ModelDescriptor,
        evaluator: ModelDescriptor,
        resolve_locator: LocatorResolver,
        progress: ProgressCallback | None,
        total_attempts: int,
    ) -> None:
        self.baseline = baseline
        self.writer = writer
        self.evaluator = evaluator
        self.resolve_locator = resolve_locator
        self.progress = progress
        self.total_attempts = total_attempts
        self.attempts: list[_AttemptState] = []
        self.candidates_by_transport: dict[str, _AttemptState] = {}
        self.candidates_by_revision: dict[str, _AttemptState] = {}
        self.evaluations_by_revision: dict[str, _CachedEvaluation] = {}
        self.evaluation_order: list[str] = []
        self._evaluation_count = 0
        self.last_rejection: str | None = None
        self.terminal_error: Exception | None = None

    @property
    def attempted_count(self) -> int:
        return len(self.attempts)

    @property
    def valid_count(self) -> int:
        return sum(attempt.status == "succeeded" for attempt in self.attempts)

    @property
    def rejected_count(self) -> int:
        return sum(attempt.rejected for attempt in self.attempts)

    @property
    def scored_count(self) -> int:
        return sum(revision != self.baseline.revision for revision in self.evaluation_order)

    def start(self) -> _AttemptState:
        self.raise_terminal()
        if self.attempted_count >= self.total_attempts:
            raise ReflectionEvaluationError("Proposal attempt limit exhausted")
        number = len(self.attempts) + 1
        attempt = _AttemptState(f"attempt-{number}", number, self.writer)
        self.attempts.append(attempt)
        return attempt

    def set_writer_provenance(
        self,
        attempt: _AttemptState,
        response: JudgeResponse,
        backend: str,
    ) -> None:
        attempt.resolved_model = _nonblank(response.model, "resolved writer model")
        attempt.resolved_family = model_family(attempt.resolved_model)
        attempt.resolved_backend = _nonblank(backend, "resolved writer backend")
        attempt.usage = _usage(response.usage)

    def fail(self, attempt: _AttemptState, error_type: str, error: str) -> None:
        attempt.status = "failed"
        attempt.error_type = error_type
        attempt.error = error

    def cancel(self, attempt: _AttemptState, error: str) -> None:
        attempt.status = "cancelled"
        attempt.error_type = "cancelled"
        attempt.error = error

    def record_terminal(self, error: Exception) -> None:
        if self.terminal_error is None:
            self.terminal_error = error

    def raise_terminal(self) -> None:
        if self.terminal_error is not None:
            raise self.terminal_error

    @property
    def stop_requested(self) -> bool:
        return self.terminal_error is not None

    def next_evaluation_id(self) -> str:
        self._evaluation_count += 1
        return f"evaluation-{self._evaluation_count}"

    def reject_writer_response(
        self,
        attempt: _AttemptState,
        response: str,
        error: Exception,
        changed_paths: tuple[str, ...],
    ) -> None:
        message = _error_text(error)
        attempt.changed_paths = changed_paths
        attempt.response_digest = _response_digest(response)
        attempt.response_excerpt = _sanitized_proposal_excerpt(response)
        attempt.rejected = True
        self.fail(attempt, type(error).__name__, message)
        self.last_rejection = f"{type(error).__name__}: {message}"[:_CORRECTION_CONTEXT_LIMIT]
        self.emit(
            "candidate_rejected",
            f"Rejected candidate {attempt.number}: {message}",
            role="proposal_writer",
            model=attempt.resolved_model or self.writer.id,
            attempt=attempt,
            error_type=type(error).__name__,
            error=message,
            changed_paths=changed_paths,
            response_digest=attempt.response_digest,
            response_excerpt=attempt.response_excerpt,
        )

    def accept_writer_response(self, attempt: _AttemptState, response: object) -> str:
        if not isinstance(response, str):
            error = TypeError("proposal writer response must be a JSON string")
            self.reject_writer_response(attempt, str(response), error, ())
            raise _ProposalRejected(f"Reflection proposal rejected: {error}") from error

        changed_paths = _best_effort_changed_paths(response)
        try:
            proposal = parse_candidate_proposal(response)
            changed_paths = tuple(change.locator for change in proposal.changes)
            bundle = materialize_candidate_proposal(
                proposal,
                baseline=self.baseline,
                resolve_locator=self.resolve_locator,
            )
            if bundle.revision in self.candidates_by_revision:
                attempt.bundle = bundle
                raise _DuplicateRevisionError(
                    f"duplicate candidate bundle revision: {bundle.revision}"
                )
        except Exception as exc:
            if not _is_proposal_validation_error(exc):
                raise
            changed_paths = _error_changed_paths(exc, changed_paths)
            self.reject_writer_response(attempt, response, exc, changed_paths)
            raise _ProposalRejected(f"Reflection proposal rejected: {_error_text(exc)}") from exc

        canonical = serialize_candidate_proposal(proposal)
        attempt.bundle = bundle
        attempt.changed_paths = tuple(sorted(changed_paths))
        attempt.status = "succeeded"
        self.candidates_by_transport[canonical] = attempt
        self.candidates_by_revision[bundle.revision] = attempt
        self.emit(
            "candidate_valid",
            f"Validated candidate {attempt.number} with {len(changed_paths)} changed paths",
            role="proposal_writer",
            model=attempt.resolved_model or self.writer.id,
            attempt=attempt,
            changed_paths=attempt.changed_paths,
        )
        return canonical

    def emit(
        self,
        phase: str,
        message: str,
        *,
        role: Literal["proposal_writer", "proposal_evaluator"],
        model: str,
        attempt: _AttemptState | None = None,
        evaluation_id: str | None = None,
        score: float | None = None,
        error_type: str | None = None,
        error: str | None = None,
        changed_paths: tuple[str, ...] | None = None,
        response_digest: str | None = None,
        response_excerpt: str | None = None,
    ) -> None:
        _emit(
            self.progress,
            phase,
            message,
            candidate=attempt.number if attempt else None,
            attempt_id=attempt.attempt_id if attempt else None,
            evaluation_id=evaluation_id,
            model=model,
            acting_role=role,
            score=score,
            error_type=error_type,
            error=error,
            changed_paths=changed_paths,
            response_digest=response_digest,
            response_excerpt=response_excerpt,
            attempted=self.attempted_count,
            valid=self.valid_count,
            rejected=self.rejected_count,
            scored=self.scored_count,
            total_attempts=self.total_attempts,
        )

    def claim(self, candidate: str) -> tuple[BundleSnapshot, _AttemptState | None]:
        candidate = _unwrap_gepa_candidate(candidate)
        if candidate == serialize_candidate_proposal(CandidateProposal(3, ())):
            return self.baseline, None
        attempt = self.candidates_by_transport.get(candidate)
        if attempt is None or attempt.bundle is None:
            raise ReflectionEvaluationError(
                "Proposal evaluator received a candidate that was not validated "
                "by the writer boundary"
            )
        return attempt.bundle, attempt

    def cached_evaluation(self, revision: str) -> _CachedEvaluation | None:
        return self.evaluations_by_revision.get(revision)

    def add_evaluation(self, record: EvaluatorRecord, side_info: dict[str, Any]) -> None:
        if record.target_revision in self.evaluations_by_revision:
            raise ReflectionEvaluationError(
                f"Bundle revision was evaluated more than once: {record.target_revision}"
            )
        self.evaluations_by_revision[record.target_revision] = _CachedEvaluation(record, side_info)
        self.evaluation_order.append(record.target_revision)


class _WriterLM:
    def __init__(
        self,
        client: ChatClient,
        ledger: _AttemptLedger,
        cancel_requested: Callable[[], bool] | None,
    ) -> None:
        self.client = client
        self.ledger = ledger
        self.cancel_requested = cancel_requested

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        _check_cancel(self.cancel_requested)
        attempt = self.ledger.start()
        self.ledger.emit(
            "generating_candidate",
            f"Starting proposal attempt {attempt.number} of up to "
            f"{self.ledger.total_attempts} with {self.ledger.writer.id}",
            role="proposal_writer",
            model=self.ledger.writer.id,
            attempt=attempt,
        )
        try:
            messages = _messages(prompt)
            correction = self.ledger.last_rejection
            self.ledger.last_rejection = None
            if correction:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Correct the previous proposal rejection: "
                            f"{correction}. Return a complete valid JSON proposal."
                        ),
                    }
                )
            _, response = self.client.chat_json(
                model=self.ledger.writer.id,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
                response_schema=REFLECTION_PROPOSAL_SCHEMA,
            )
            self.ledger.set_writer_provenance(
                attempt,
                response,
                getattr(self.client, "backend", self.ledger.writer.backend),
            )
            _check_cancel(self.cancel_requested)
            canonical = self.ledger.accept_writer_response(attempt, response.content)
            return _gepa_safe_candidate(canonical)
        except _ProposalRejected:
            raise
        except (InferenceCancelled, ReflectionCancelled) as exc:
            cancelled = ReflectionCancelled("Reflection cancelled")
            self.ledger.cancel(attempt, str(exc) or "Reflection cancelled")
            self.ledger.record_terminal(cancelled)
            self.ledger.emit(
                "candidate_generation_cancelled",
                f"Candidate {attempt.number} generation was cancelled",
                role="proposal_writer",
                model=self.ledger.writer.id,
                attempt=attempt,
                error_type="cancelled",
                error="Reflection cancelled",
            )
            raise cancelled from exc
        except Exception as exc:
            message = f"Proposal writer failed: {_actionable_failure(exc, role='proposal writer')}"
            terminal = ReflectionEvaluationError(message)
            self.ledger.fail(attempt, type(exc).__name__, message)
            self.ledger.record_terminal(terminal)
            self.ledger.emit(
                "candidate_generation_failed",
                message,
                role="proposal_writer",
                model=self.ledger.writer.id,
                attempt=attempt,
                error_type=type(exc).__name__,
                error=message,
            )
            raise terminal from exc


def _check_cancel(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise ReflectionCancelled("Reflection cancelled")


def _evaluator(
    client: ChatClient,
    ledger: _AttemptLedger,
    coaching_text: str,
    cancel_requested: Callable[[], bool] | None,
):
    def evaluate(candidate: str) -> tuple[float, dict[str, Any]]:
        _check_cancel(cancel_requested)
        try:
            bundle, attempt = ledger.claim(candidate)
        except ReflectionEvaluationError as exc:
            ledger.record_terminal(exc)
            ledger.emit(
                "evaluation_failed",
                str(exc),
                role="proposal_evaluator",
                model=ledger.evaluator.id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        cached = ledger.cached_evaluation(bundle.revision)
        if cached is not None:
            return cached.record.score, cached.side_info

        evaluation_id = ledger.next_evaluation_id()
        candidate_number = attempt.number if attempt else None
        phase = "evaluating_baseline" if attempt is None else "evaluating_candidate"
        label = "current bundle" if attempt is None else f"candidate {candidate_number}"
        ledger.emit(
            phase,
            f"Evaluating {label} with {ledger.evaluator.id}",
            role="proposal_evaluator",
            model=ledger.evaluator.id,
            attempt=attempt,
            evaluation_id=evaluation_id,
        )
        bundle_text = "\n\n".join(
            f"## {target.locator}\n{target.content or ''}"
            for target in bundle.targets
            if target.exists
        )
        try:
            parsed, response = client.chat_json(
                model=ledger.evaluator.id,
                messages=[
                    {"role": "system", "content": _EVALUATOR_SYSTEM},
                    {
                        "role": "user",
                        "content": _EVALUATOR_USER.format(
                            coaching_text=coaching_text,
                            bundle_text=bundle_text,
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=512,
            )
            quality = _score(
                parsed.get("score") if isinstance(parsed, Mapping) else None,
                "evaluator score",
            )
            rationale = parsed.get("rationale", "") if isinstance(parsed, Mapping) else ""
            if not isinstance(rationale, str):
                raise ValueError("evaluator rationale must be a string")
            record = EvaluatorRecord(
                evaluation_id=evaluation_id,
                target_revision=bundle.revision,
                requested_model=ledger.evaluator.id,
                requested_family=ledger.evaluator.family,
                requested_backend=ledger.evaluator.backend,
                resolved_model=response.model,
                resolved_family=model_family(response.model),
                resolved_backend=_nonblank(
                    getattr(client, "backend", ledger.evaluator.backend),
                    "resolved evaluator backend",
                ),
                score=quality,
                rationale=rationale,
                usage=response.usage,
            )
        except (InferenceCancelled, ReflectionCancelled) as exc:
            terminal = ReflectionCancelled("Reflection cancelled")
            ledger.record_terminal(terminal)
            raise terminal from exc
        except Exception as exc:
            message = (
                f"Proposal evaluator failed: {_actionable_failure(exc, role='proposal evaluator')}"
            )
            terminal = ReflectionEvaluationError(message)
            ledger.record_terminal(terminal)
            ledger.emit(
                "evaluation_failed",
                message,
                role="proposal_evaluator",
                model=ledger.evaluator.id,
                attempt=attempt,
                evaluation_id=evaluation_id,
                error_type=type(exc).__name__,
                error=message,
            )
            raise terminal from exc

        side_info = {
            "predicted_quality": quality,
            "evaluator_rationale": rationale,
            "evaluation_id": evaluation_id,
            "target_revision": bundle.revision,
            "complete_bundle": _content_map(bundle),
        }
        ledger.add_evaluation(record, side_info)
        ledger.emit(
            "baseline_evaluated" if attempt is None else "candidate_evaluated",
            f"Evaluated {label}: {quality:.2f} with {response.model}",
            role="proposal_evaluator",
            model=response.model,
            attempt=attempt,
            evaluation_id=evaluation_id,
            score=quality,
        )
        return quality, side_info

    return evaluate


class _ScoredNoImprovementStopper:
    """Count only uniquely scored candidate revisions toward reflection patience."""

    def __init__(self, ledger: _AttemptLedger, max_iterations_without_improvement: int) -> None:
        self.ledger = ledger
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.iterations_without_improvement = 0
        self.best_score = float("-inf")
        self._seen: set[str] = set()

    def __call__(self, _state: object) -> bool:
        for revision in self.ledger.evaluation_order:
            if revision in self._seen:
                continue
            self._seen.add(revision)
            score = self.ledger.evaluations_by_revision[revision].record.score
            if score > self.best_score:
                self.best_score = score
                self.iterations_without_improvement = 0
            elif revision != self.ledger.baseline.revision:
                self.iterations_without_improvement += 1
        return self.iterations_without_improvement >= self.max_iterations_without_improvement


def run_reflection(
    *,
    baseline: BundleSnapshot,
    feedback: Sequence[Mapping[str, Any]],
    coaching_text: str,
    scope_policy: Mapping[str, Any],
    requested_writer: ModelDescriptor,
    requested_evaluator: ModelDescriptor,
    writer_client: ChatClient,
    evaluator_client: ChatClient,
    resolve_locator: LocatorResolver,
    candidate_budget: int = 3,
    progress_callback: ProgressCallback | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    optimizer: Optimizer | None = None,
) -> ReflectionResult:
    """Optimize and evaluate exact bundle revisions with separate pinned model roles."""

    if not isinstance(baseline, BundleSnapshot):
        raise ValueError("baseline must be a BundleSnapshot")
    if not _content_map(baseline):
        raise ValueError("baseline must contain at least one existing managed target")
    if not isinstance(requested_writer, ModelDescriptor) or (
        "proposal_writer" not in requested_writer.supported_roles
    ):
        raise ValueError("requested writer must support proposal_writer")
    if not isinstance(requested_evaluator, ModelDescriptor) or (
        "proposal_evaluator" not in requested_evaluator.supported_roles
    ):
        raise ValueError("requested evaluator must support proposal_evaluator")
    if type(candidate_budget) is not int or not 1 <= candidate_budget <= MAX_CANDIDATE_BUDGET:
        raise ValueError("candidate_budget must be an integer between 1 and 10")
    if not isinstance(coaching_text, str):
        raise ValueError("coaching_text must be a string")
    baseline_contents_text = _prompt_json(_content_map(baseline), "baseline contents")
    scope_policy_text = _prompt_json(scope_policy, "scope policy")
    if not _has_eligible_signal(feedback):
        raise ReflectionEvaluationError("Reflection requires at least one valid rated signal")
    _check_cancel(cancel_requested)

    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig
    from gepa.optimize_anything import optimize_anything as gepa_optimize

    ledger = _AttemptLedger(
        baseline,
        requested_writer,
        requested_evaluator,
        resolve_locator,
        progress_callback,
        candidate_budget,
    )
    writer_lm = _WriterLM(writer_client, ledger, cancel_requested)
    evaluator = _evaluator(
        evaluator_client,
        ledger,
        coaching_text,
        cancel_requested,
    )
    patience = _ScoredNoImprovementStopper(ledger, NO_IMPROVEMENT_PATIENCE)
    stop_callbacks: list[Callable[..., bool]] = [lambda _state: ledger.stop_requested]
    if cancel_requested is not None:
        stop_callbacks.append(lambda _state: cancel_requested())
    stop_callbacks.append(patience)
    config = GEPAConfig(
        engine=EngineConfig(
            max_candidate_proposals=candidate_budget,
            display_progress_bar=False,
        ),
        reflection=ReflectionConfig(reflection_lm=writer_lm),
        stop_callbacks=tuple(stop_callbacks),
    )
    optimize = optimizer or gepa_optimize
    optimize(
        seed_candidate=serialize_candidate_proposal(CandidateProposal(3, ())),
        evaluator=evaluator,
        objective=_GEPA_OBJECTIVE_TEMPLATE.format(
            coaching_text=coaching_text,
            proposal_layout=_PROPOSAL_LAYOUT,
        ),
        background=_GEPA_BACKGROUND_TEMPLATE.format(
            baseline_contents=baseline_contents_text,
            scope_policy=scope_policy_text,
        ),
        config=config,
    )
    _check_cancel(cancel_requested)
    ledger.raise_terminal()

    baseline_cached = ledger.cached_evaluation(baseline.revision)
    if baseline_cached is None:
        raise ReflectionEvaluationError("Reflection completed without baseline evaluator evidence")
    baseline_score = baseline_cached.record.score
    candidates: list[ReflectionCandidate] = []
    for attempt in ledger.attempts:
        if attempt.status != "succeeded":
            continue
        bundle = attempt.bundle
        if (
            bundle is None
            or attempt.resolved_model is None
            or attempt.resolved_family is None
            or attempt.resolved_backend is None
        ):
            raise ReflectionEvaluationError("Candidate is missing writer provenance")
        cached = ledger.cached_evaluation(bundle.revision)
        if cached is None:
            raise ReflectionEvaluationError(
                f"Validated candidate {attempt.number} was not evaluated"
            )
        score = cached.record.score
        candidates.append(
            ReflectionCandidate(
                candidate_id=_candidate_id(attempt.attempt_id, bundle.revision),
                bundle=bundle,
                score=score,
                score_delta=score - baseline_score,
                rationale=cached.record.rationale,
                generation_attempt_id=attempt.attempt_id,
                requested_writer=attempt.requested_writer,
                resolved_writer_model=attempt.resolved_model,
                resolved_writer_family=attempt.resolved_family,
                resolved_writer_backend=attempt.resolved_backend,
                evaluation=cached.record,
            )
        )

    reason = NO_VALID_PROPOSAL_REASON if not candidates else None
    selected: ReflectionCandidate | None = None
    best_score = baseline_score
    for candidate in candidates:
        if candidate.score > best_score:
            best_score = candidate.score
            selected = candidate
    baseline_won = bool(candidates) and selected is None
    recommended_candidate_id = selected.candidate_id if selected is not None else None
    if (
        patience.iterations_without_improvement >= NO_IMPROVEMENT_PATIENCE
        and ledger.attempted_count < candidate_budget
    ):
        _emit(
            progress_callback,
            "early_stop",
            "Stopped after two consecutive attempts without a better predicted score",
            attempted=ledger.attempted_count,
            valid=ledger.valid_count,
            rejected=ledger.rejected_count,
            scored=ledger.scored_count,
            total_attempts=candidate_budget,
        )
    if reason is not None:
        selection_message = reason
    elif baseline_won:
        selection_message = "Kept the evaluated baseline over evaluated candidates"
    else:
        assert selected is not None
        selection_message = f"Selected candidate {selected.generation_attempt_id}"
    _emit(
        progress_callback,
        "selection_complete",
        selection_message,
        candidate=(
            None
            if selected is None
            else next(
                attempt.number
                for attempt in ledger.attempts
                if attempt.attempt_id == selected.generation_attempt_id
            )
        ),
        score=best_score,
        attempted=ledger.attempted_count,
        valid=ledger.valid_count,
        rejected=ledger.rejected_count,
        scored=ledger.scored_count,
        total_attempts=candidate_budget,
    )
    return ReflectionResult(
        baseline=baseline,
        baseline_score=baseline_score,
        baseline_evaluation=baseline_cached.record,
        candidates=tuple(candidates),
        generation_attempts=tuple(attempt.freeze() for attempt in ledger.attempts),
        recommended_candidate_id=recommended_candidate_id,
        baseline_won=baseline_won,
        reason=reason,
    )
