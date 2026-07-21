"""Strict structured-output contracts for sliding judge inference phases."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from weave_agent_signals.judges.inference import JsonSchemaSpec
from weave_agent_signals.judges.tokens import count_tokens

SLIDING_CONTRACT_SCHEMA_VERSION = 1
MAX_WINDOW_FINDINGS = 4
MAX_REPORTED_UNKNOWN_EVIDENCE_IDS = 3
MAX_REPORTED_EVIDENCE_ID_CHARACTERS = 64
SCORE_ANCHORS = (0.0, 0.25, 0.5, 0.75, 1.0)


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


class WindowFinding(_ClosedModel):
    finding_id: StrictStr
    polarity: Literal["positive", "negative"]
    observation: StrictStr
    evidence_ids: tuple[StrictStr, ...] = Field(min_length=1)
    quote: StrictStr | None


class WindowFindings(_ClosedModel):
    schema_version: Literal[1]
    window_id: StrictStr
    findings: tuple[WindowFinding, ...] = Field(max_length=MAX_WINDOW_FINDINGS)


def window_finding_semantic_key(finding: WindowFinding) -> tuple[object, ...]:
    return (
        finding.polarity,
        finding.observation,
        tuple(sorted(set(finding.evidence_ids))),
    )


class BehavioralFeedback(_ClosedModel):
    success: StrictStr | None
    problem: StrictStr | None
    desired_behavior: StrictStr | None


class MergedVerdict(_ClosedModel):
    schema_version: Literal[1]
    status: Literal["scored", "insufficient_evidence"]
    score: Annotated[
        StrictFloat | StrictInt | None,
        Field(json_schema_extra={"enum": [*SCORE_ANCHORS, None]}),
    ]
    rationale: StrictStr
    evidence_ids: tuple[StrictStr, ...]
    feedback: BehavioralFeedback | None


class _MergedVerdictOutput(_ClosedModel):
    schema_version: Literal[1]
    status: Literal["scored", "insufficient_evidence"]
    score: Annotated[
        StrictFloat | StrictInt | None,
        Field(json_schema_extra={"enum": [*SCORE_ANCHORS, None]}),
    ]
    rationale: StrictStr
    finding_ids: tuple[StrictStr, ...]
    feedback: BehavioralFeedback | None


def _model_output_json_schema(
    model: type[BaseModel],
    *host_fields: str,
) -> dict:
    schema = model.model_json_schema()
    for field in host_fields:
        schema["properties"].pop(field)
        schema["required"].remove(field)
    return schema


def _merged_verdict_json_schema() -> dict:
    schema = _model_output_json_schema(_MergedVerdictOutput, "schema_version")
    score = schema["properties"]["score"]
    score["anyOf"] = [{"type": "number"}, {"type": "null"}]
    return schema


CHUNK_DIGEST_SCHEMA = JsonSchemaSpec(
    name="chunk_digest",
    schema=_model_output_json_schema(ChunkDigest, "schema_version", "chunk_id"),
)
WINDOW_FINDING_SCHEMA = JsonSchemaSpec(
    name="window_finding",
    schema=WindowFinding.model_json_schema(),
)
WINDOW_FINDINGS_SCHEMA = JsonSchemaSpec(
    name="window_findings",
    schema=_model_output_json_schema(WindowFindings, "schema_version", "window_id"),
)
BEHAVIORAL_FEEDBACK_SCHEMA = JsonSchemaSpec(
    name="behavioral_feedback",
    schema=BehavioralFeedback.model_json_schema(),
)
MERGED_VERDICT_SCHEMA = JsonSchemaSpec(
    name="merged_verdict",
    schema=_merged_verdict_json_schema(),
)


def _bound_evidence_ids(allowed_evidence_ids: Sequence[str]) -> list[str]:
    values = list(allowed_evidence_ids)
    _allowed_id_set(values)
    if not values:
        raise ValueError("allowed evidence IDs must not be empty")
    return values


def bind_window_findings_schema(
    allowed_evidence_ids: Sequence[str],
) -> JsonSchemaSpec:
    """Bind one window response to its raw-window evidence."""

    schema = deepcopy(dict(WINDOW_FINDINGS_SCHEMA.schema))
    finding = schema["$defs"]["WindowFinding"]["properties"]
    finding["evidence_ids"]["items"]["enum"] = _bound_evidence_ids(allowed_evidence_ids)
    return JsonSchemaSpec(name=WINDOW_FINDINGS_SCHEMA.name, schema=schema)


def bind_merged_verdict_schema(
    allowed_finding_ids: Sequence[str],
) -> JsonSchemaSpec:
    """Bind one merged verdict to the findings supplied for this rubric."""

    schema = deepcopy(dict(MERGED_VERDICT_SCHEMA.schema))
    values = list(allowed_finding_ids)
    _allowed_id_set(values)
    if values:
        schema["properties"]["finding_ids"]["items"]["enum"] = values
    return JsonSchemaSpec(name=MERGED_VERDICT_SCHEMA.name, schema=schema)


def render_window_findings(value: WindowFindings) -> str:
    """Render normalized window findings as deterministic merge-input JSON."""

    return json.dumps(
        value.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalized_text(value: str, *, field: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{field} must be nonblank")
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


def _validated_aliases(evidence_aliases: Mapping[str, str]) -> dict[str, str]:
    aliases = dict(evidence_aliases)
    _allowed_id_set(tuple(aliases))
    canonical_ids = tuple(aliases.values())
    _allowed_id_set(canonical_ids)
    if not aliases:
        raise ValueError("evidence aliases must not be empty")
    return aliases


def _verified_quote(quote: str | None, *, raw_text: str) -> str | None:
    if quote is None:
        return None
    normalized = " ".join(quote.split())
    if not normalized:
        return None
    return normalized if normalized in " ".join(raw_text.split()) else None


def _validated_finding_evidence(
    finding_evidence: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    validated: dict[str, tuple[str, ...]] = {}
    for finding_id, evidence_ids in finding_evidence.items():
        key = _nonblank_id(finding_id, field="finding ID")
        canonical = tuple(
            _nonblank_id(evidence_id, field="evidence ID") for evidence_id in evidence_ids
        )
        if not canonical:
            raise ValueError("finding evidence must not be empty")
        validated[key] = canonical
    return validated


def _validated_evidence_ids(
    evidence_ids: Sequence[str],
    *,
    allowed_ids: set[str],
    required: bool,
    citation_kind: str = "evidence",
) -> tuple[str, ...]:
    normalized = tuple(
        _nonblank_id(evidence_id, field=f"{citation_kind} ID") for evidence_id in evidence_ids
    )
    if required and not normalized:
        raise ValueError(f"at least one {citation_kind} citation is required")
    unknown = [evidence_id for evidence_id in normalized if evidence_id not in allowed_ids]
    if unknown:
        reported = [
            evidence_id
            if len(evidence_id) <= MAX_REPORTED_EVIDENCE_ID_CHARACTERS
            else evidence_id[: MAX_REPORTED_EVIDENCE_ID_CHARACTERS - 3] + "..."
            for evidence_id in unknown[:MAX_REPORTED_UNKNOWN_EVIDENCE_IDS]
        ]
        remainder = len(unknown) - len(reported)
        suffix = f" (+{remainder} more)" if remainder else ""
        raise ValueError(f"unknown {citation_kind} IDs: {json.dumps(reported)}{suffix}")
    return normalized


def parse_chunk_digest(
    value: Mapping[str, object] | ChunkDigest,
    *,
    expected_chunk_id: str | None = None,
) -> ChunkDigest:
    """Validate one digest against its exact source evidence."""

    data = _validation_data(value)
    if expected_chunk_id is not None and isinstance(data, Mapping):
        data = {
            **data,
            "schema_version": SLIDING_CONTRACT_SCHEMA_VERSION,
            "chunk_id": _nonblank_id(expected_chunk_id, field="expected chunk ID"),
        }
    digest = ChunkDigest.model_validate(data)
    chunk_id = _nonblank_id(digest.chunk_id, field="chunk ID")
    text = _normalized_text(digest.text, field="digest text")
    return digest.model_copy(update={"chunk_id": chunk_id, "text": text})


def parse_window_finding(
    value: Mapping[str, object] | WindowFinding,
    *,
    evidence_aliases: Mapping[str, str],
    raw_text: str,
) -> WindowFinding:
    """Validate one bounded, evidence-cited positive or negative finding."""

    finding = WindowFinding.model_validate(_validation_data(value))
    aliases = _validated_aliases(evidence_aliases)
    alias_ids = _validated_evidence_ids(
        finding.evidence_ids,
        allowed_ids=set(aliases),
        required=True,
    )
    quote = _verified_quote(finding.quote, raw_text=raw_text)
    return finding.model_copy(
        update={
            "finding_id": _nonblank_id(finding.finding_id, field="finding ID"),
            "observation": _normalized_text(finding.observation, field="finding observation"),
            "evidence_ids": tuple(aliases[alias] for alias in alias_ids),
            "quote": quote,
        }
    )


def parse_window_findings(
    value: Mapping[str, object] | WindowFindings,
    *,
    evidence_aliases: Mapping[str, str],
    raw_text: str,
    max_tokens: int,
    expected_window_id: str | None = None,
) -> WindowFindings:
    """Validate one window response containing no more than four unique findings."""

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    data = _validation_data(value)
    if expected_window_id is not None and isinstance(data, Mapping):
        data = {
            **data,
            "schema_version": SLIDING_CONTRACT_SCHEMA_VERSION,
            "window_id": _nonblank_id(expected_window_id, field="expected window ID"),
        }
    parsed = WindowFindings.model_validate(data)
    window_id = _nonblank_id(parsed.window_id, field="window ID")
    findings: list[WindowFinding] = []
    seen: set[tuple[object, ...]] = set()
    for value in parsed.findings:
        finding = parse_window_finding(
            value,
            evidence_aliases=evidence_aliases,
            raw_text=raw_text,
        )
        key = window_finding_semantic_key(finding)
        if key in seen:
            continue
        seen.add(key)
        findings.append(finding.model_copy(update={"finding_id": f"finding-{len(findings) + 1}"}))
    normalized = parsed.model_copy(update={"window_id": window_id, "findings": tuple(findings)})
    if count_tokens(render_window_findings(normalized), "utf8_bytes_div_3") > max_tokens:
        raise ValueError("complete window findings artifact exceeds the configured token limit")
    return normalized


def parse_behavioral_feedback(
    value: Mapping[str, object] | BehavioralFeedback,
) -> BehavioralFeedback:
    """Validate behavioral feedback without interpreting its wording."""

    feedback = BehavioralFeedback.model_validate(_validation_data(value))
    normalized: dict[str, str | None] = {}
    for field in ("success", "problem", "desired_behavior"):
        text = getattr(feedback, field)
        normalized[field] = (
            None if text is None else _normalized_text(text, field=f"feedback {field}")
        )
    if not any(text is not None for text in normalized.values()):
        raise ValueError("behavioral feedback requires at least one non-null field")
    return feedback.model_copy(update=normalized)


def parse_merged_verdict(
    value: Mapping[str, object],
    *,
    finding_evidence: Mapping[str, Sequence[str]],
) -> MergedVerdict:
    """Validate the final anchored verdict and behavioral-feedback combination."""

    data = _validation_data(value)
    if isinstance(data, Mapping):
        data = {**data, "schema_version": SLIDING_CONTRACT_SCHEMA_VERSION}
    evidence_by_finding = _validated_finding_evidence(finding_evidence)
    output = _MergedVerdictOutput.model_validate(data)
    rationale = _normalized_text(output.rationale, field="rationale")

    if output.status == "insufficient_evidence":
        return MergedVerdict(
            schema_version=SLIDING_CONTRACT_SCHEMA_VERSION,
            status=output.status,
            score=None,
            rationale=rationale,
            evidence_ids=(),
            feedback=None,
        )

    if output.score is None:
        raise ValueError("scored verdict requires a score")
    _validate_score_anchor(output.score, field="scored verdict score")
    findings = _validated_evidence_ids(
        output.finding_ids,
        allowed_ids=set(evidence_by_finding),
        required=True,
        citation_kind="finding",
    )
    evidence_ids = tuple(
        evidence_id for finding_id in findings for evidence_id in evidence_by_finding[finding_id]
    )
    if output.feedback is None:
        raise ValueError("scored verdict requires behavioral feedback")
    feedback = parse_behavioral_feedback(output.feedback)
    return MergedVerdict(
        schema_version=SLIDING_CONTRACT_SCHEMA_VERSION,
        status=output.status,
        score=output.score,
        rationale=rationale,
        evidence_ids=evidence_ids,
        feedback=feedback,
    )
