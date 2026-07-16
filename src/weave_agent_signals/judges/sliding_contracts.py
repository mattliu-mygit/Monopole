"""Strict structured-output contracts for sliding judge inference phases."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from weave_agent_signals.judges.inference import JsonSchemaSpec
from weave_agent_signals.judges.windowing import estimate_tokens

SLIDING_CONTRACT_SCHEMA_VERSION = 1
MAX_WINDOW_FINDINGS = 4
MAX_FINDING_OBSERVATION_CHARACTERS = 350
MAX_BEHAVIORAL_FEEDBACK_CHARACTERS = 500
SCORE_ANCHORS = (0.0, 0.25, 0.5, 0.75, 1.0)

_MANAGED_FILE_PATTERN = re.compile(
    r"\b(?:agents\.md|claude\.md|skill\.md|instruction\s+files?|prompt\s+files?)\b",
    re.IGNORECASE,
)
_IMPERATIVE_FILE_EDIT_PATTERN = re.compile(
    r"\b(?:add|create|delete|edit|modify|remove|rename|replace|rewrite|update|write)\b"
    r"[^.!?\n]{0,100}\b(?:files?|[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)\b",
    re.IGNORECASE,
)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validation_data(value: object) -> object:
    if isinstance(value, BaseModel):
        data = dict(vars(value))
        if value.__pydantic_extra__:
            data.update(value.__pydantic_extra__)
        return {key: _validation_data(item) for key, item in data.items()}
    if isinstance(value, Mapping):
        return {key: _validation_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_validation_data(item) for item in value)
    if isinstance(value, list):
        return [_validation_data(item) for item in value]
    return value


def _validate_score_anchor(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be one of {SCORE_ANCHORS}")
    score = float(value)
    if not math.isfinite(score) or score not in SCORE_ANCHORS:
        raise ValueError(f"{field} must be one of {SCORE_ANCHORS}")
    return score


class ChunkDigest(_ClosedModel):
    schema_version: Literal[1]
    chunk_id: StrictStr
    text: StrictStr
    evidence_ids: tuple[StrictStr, ...]


class WindowFinding(_ClosedModel):
    finding_id: StrictStr
    polarity: Literal["positive", "negative"]
    observation: StrictStr
    evidence_ids: tuple[StrictStr, ...]


class WindowFindings(_ClosedModel):
    schema_version: Literal[1]
    window_id: StrictStr
    findings: tuple[WindowFinding, ...] = Field(max_length=MAX_WINDOW_FINDINGS)


class BehavioralFeedback(_ClosedModel):
    success: StrictStr | None
    problem: StrictStr | None
    desired_behavior: StrictStr | None


class MergedVerdict(_ClosedModel):
    schema_version: Literal[1]
    status: Literal["scored", "insufficient_evidence"]
    score: StrictFloat | StrictInt | None
    rationale: StrictStr
    evidence_ids: tuple[StrictStr, ...]
    feedback: BehavioralFeedback | None


CHUNK_DIGEST_SCHEMA = JsonSchemaSpec(
    name="chunk_digest",
    schema=ChunkDigest.model_json_schema(),
)
WINDOW_FINDING_SCHEMA = JsonSchemaSpec(
    name="window_finding",
    schema=WindowFinding.model_json_schema(),
)
WINDOW_FINDINGS_SCHEMA = JsonSchemaSpec(
    name="window_findings",
    schema=WindowFindings.model_json_schema(),
)
BEHAVIORAL_FEEDBACK_SCHEMA = JsonSchemaSpec(
    name="behavioral_feedback",
    schema=BehavioralFeedback.model_json_schema(),
)
MERGED_VERDICT_SCHEMA = JsonSchemaSpec(
    name="merged_verdict",
    schema=MergedVerdict.model_json_schema(),
)


def render_window_findings(value: WindowFindings) -> str:
    """Render normalized window findings as deterministic merge-input JSON."""

    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalized_text(value: str, *, field: str, max_characters: int | None = None) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{field} must be nonblank")
    if max_characters is not None and len(normalized) > max_characters:
        raise ValueError(f"{field} must contain at most {max_characters} normalized characters")
    return normalized


def _nonblank_id(value: str, *, field: str) -> str:
    if not value.strip():
        raise ValueError(f"{field} must be nonblank")
    return value


def _allowed_id_set(allowed_evidence_ids: Sequence[str]) -> set[str]:
    normalized = [
        _nonblank_id(evidence_id, field="allowed evidence ID")
        if isinstance(evidence_id, str)
        else ""
        for evidence_id in allowed_evidence_ids
    ]
    if any(not evidence_id for evidence_id in normalized):
        raise ValueError("allowed evidence IDs must be nonblank strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("allowed evidence IDs must be unique")
    return set(normalized)


def _validated_evidence_ids(
    evidence_ids: Sequence[str],
    *,
    allowed_ids: set[str],
    required: bool,
) -> tuple[str, ...]:
    normalized = tuple(
        _nonblank_id(evidence_id, field="evidence ID") for evidence_id in evidence_ids
    )
    if required and not normalized:
        raise ValueError("at least one evidence citation is required")
    if len(normalized) != len(set(normalized)):
        raise ValueError("evidence IDs must be unique")
    for evidence_id in normalized:
        if evidence_id not in allowed_ids:
            raise ValueError("unknown evidence ID")
    return normalized


def parse_chunk_digest(
    value: Mapping[str, object] | ChunkDigest,
    *,
    allowed_evidence_ids: Sequence[str],
    max_tokens: int,
    expected_chunk_id: str | None = None,
) -> ChunkDigest:
    """Validate one bounded digest against its exact source evidence."""

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    digest = ChunkDigest.model_validate(_validation_data(value))
    chunk_id = _nonblank_id(digest.chunk_id, field="chunk ID")
    if expected_chunk_id is not None and chunk_id != _nonblank_id(
        expected_chunk_id, field="expected chunk ID"
    ):
        raise ValueError("unexpected chunk ID")
    text = _normalized_text(digest.text, field="digest text")
    if estimate_tokens(text) > max_tokens:
        raise ValueError("digest text exceeds the configured token limit")
    evidence_ids = _validated_evidence_ids(
        digest.evidence_ids,
        allowed_ids=_allowed_id_set(allowed_evidence_ids),
        required=True,
    )
    return digest.model_copy(
        update={"chunk_id": chunk_id, "text": text, "evidence_ids": evidence_ids}
    )


def parse_window_finding(
    value: Mapping[str, object] | WindowFinding,
    *,
    allowed_evidence_ids: Sequence[str],
) -> WindowFinding:
    """Validate one bounded, evidence-cited positive or negative finding."""

    finding = WindowFinding.model_validate(_validation_data(value))
    return finding.model_copy(
        update={
            "finding_id": _nonblank_id(finding.finding_id, field="finding ID"),
            "observation": _normalized_text(
                finding.observation,
                field="finding observation",
                max_characters=MAX_FINDING_OBSERVATION_CHARACTERS,
            ),
            "evidence_ids": _validated_evidence_ids(
                finding.evidence_ids,
                allowed_ids=_allowed_id_set(allowed_evidence_ids),
                required=True,
            ),
        }
    )


def parse_window_findings(
    value: Mapping[str, object] | WindowFindings,
    *,
    allowed_evidence_ids: Sequence[str],
    max_tokens: int,
    expected_window_id: str | None = None,
) -> WindowFindings:
    """Validate one window response containing no more than four unique findings."""

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    parsed = WindowFindings.model_validate(_validation_data(value))
    window_id = _nonblank_id(parsed.window_id, field="window ID")
    if expected_window_id is not None and window_id != _nonblank_id(
        expected_window_id, field="expected window ID"
    ):
        raise ValueError("unexpected window ID")
    findings = tuple(
        parse_window_finding(finding, allowed_evidence_ids=allowed_evidence_ids)
        for finding in parsed.findings
    )
    finding_ids = [finding.finding_id for finding in findings]
    if len(finding_ids) != len(set(finding_ids)):
        raise ValueError("finding IDs must be unique within a window")
    finding_keys = [
        (finding.polarity, finding.observation, tuple(sorted(finding.evidence_ids)))
        for finding in findings
    ]
    if len(finding_keys) != len(set(finding_keys)):
        raise ValueError("duplicate findings are not allowed within a window")
    normalized = parsed.model_copy(update={"window_id": window_id, "findings": findings})
    if estimate_tokens(render_window_findings(normalized)) > max_tokens:
        raise ValueError("complete window findings artifact exceeds the configured token limit")
    return normalized


def parse_behavioral_feedback(
    value: Mapping[str, object] | BehavioralFeedback,
) -> BehavioralFeedback:
    """Validate bounded feedback that describes agent behavior, not file edits."""

    feedback = BehavioralFeedback.model_validate(_validation_data(value))
    normalized: dict[str, str | None] = {}
    for field in ("success", "problem", "desired_behavior"):
        text = getattr(feedback, field)
        normalized[field] = (
            None
            if text is None
            else _normalized_text(
                text,
                field=f"feedback {field}",
                max_characters=MAX_BEHAVIORAL_FEEDBACK_CHARACTERS,
            )
        )
    if not any(text is not None for text in normalized.values()):
        raise ValueError("behavioral feedback requires at least one non-null field")

    desired_behavior = normalized["desired_behavior"]
    if desired_behavior is not None and (
        _MANAGED_FILE_PATTERN.search(desired_behavior)
        or _IMPERATIVE_FILE_EDIT_PATTERN.search(desired_behavior)
    ):
        raise ValueError(
            "desired behavior must describe agent behavior rather than instruction edits"
        )
    return feedback.model_copy(update=normalized)


def parse_merged_verdict(
    value: Mapping[str, object] | MergedVerdict,
    *,
    allowed_evidence_ids: Sequence[str],
) -> MergedVerdict:
    """Validate the final anchored verdict and behavioral-feedback combination."""

    verdict = MergedVerdict.model_validate(_validation_data(value))
    rationale = _normalized_text(verdict.rationale, field="rationale")
    allowed_ids = _allowed_id_set(allowed_evidence_ids)

    if verdict.status == "insufficient_evidence":
        if verdict.score is not None:
            raise ValueError("insufficient_evidence requires a null score")
        evidence_ids = _validated_evidence_ids(
            verdict.evidence_ids,
            allowed_ids=allowed_ids,
            required=False,
        )
        if evidence_ids:
            raise ValueError("insufficient_evidence requires empty evidence")
        if verdict.feedback is not None:
            raise ValueError("insufficient_evidence requires null feedback")
        return verdict.model_copy(update={"rationale": rationale, "evidence_ids": evidence_ids})

    if verdict.score is None:
        raise ValueError("scored verdict requires a score")
    _validate_score_anchor(verdict.score, field="scored verdict score")
    evidence_ids = _validated_evidence_ids(
        verdict.evidence_ids,
        allowed_ids=allowed_ids,
        required=True,
    )
    if verdict.feedback is None:
        raise ValueError("scored verdict requires behavioral feedback")
    feedback = parse_behavioral_feedback(verdict.feedback)
    return verdict.model_copy(
        update={"rationale": rationale, "evidence_ids": evidence_ids, "feedback": feedback}
    )
