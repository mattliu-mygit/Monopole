"""Authoritative schema and validation for judge verdicts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, StrictStr

from weave_agent_signals.judges.inference import JsonSchemaSpec

JUDGE_VERDICT_SCHEMA_VERSION = 3
JUDGE_SCORE_ANCHORS = (0.0, 0.25, 0.5, 0.75, 1.0)


class JudgeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StrictStr
    observations: tuple[StrictStr, ...]


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3]
    status: Literal["scored", "insufficient_evidence"]
    score: StrictFloat | StrictInt | None
    rationale: StrictStr
    evidence: tuple[JudgeEvidence, ...]


JUDGE_VERDICT_SCHEMA = JsonSchemaSpec(
    name="judge_verdict",
    schema=JudgeVerdict.model_json_schema(),
)


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


def validate_score_anchor(value: object, *, field: str = "score") -> float:
    """Return one finite score on the shared five-anchor judge scale."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field} must be one of " + ", ".join(str(anchor) for anchor in JUDGE_SCORE_ANCHORS)
        )
    score = float(value)
    if not math.isfinite(score) or score not in JUDGE_SCORE_ANCHORS:
        raise ValueError(
            f"{field} must be one of " + ", ".join(str(anchor) for anchor in JUDGE_SCORE_ANCHORS)
        )
    return score


def parse_judge_verdict(
    value: Mapping[str, object] | JudgeVerdict,
    evidence_ids: Sequence[str],
) -> JudgeVerdict:
    """Validate one judge verdict against the exact supplied evidence allowlist."""

    verdict = JudgeVerdict.model_validate(_validation_data(value))
    if not verdict.rationale.strip():
        raise ValueError("rationale must be nonblank")

    allowed_ids = set(evidence_ids)
    grouped_observations: dict[str, list[str]] = {}
    for evidence in verdict.evidence:
        if evidence.id not in allowed_ids:
            raise ValueError(f"unknown evidence ID: {evidence.id}")
        if not evidence.observations:
            raise ValueError("evidence observations must not be empty")
        observations = grouped_observations.setdefault(evidence.id, [])
        for observation in evidence.observations:
            if not observation.strip():
                raise ValueError("evidence observation must be nonblank")
            if observation not in observations:
                observations.append(observation)

    canonical_evidence = tuple(
        JudgeEvidence(id=evidence_id, observations=tuple(observations))
        for evidence_id, observations in grouped_observations.items()
    )
    verdict = verdict.model_copy(update={"evidence": canonical_evidence})

    if verdict.status == "insufficient_evidence":
        if verdict.score is not None:
            raise ValueError("insufficient_evidence requires a null score")
        if verdict.evidence:
            raise ValueError("insufficient_evidence requires empty evidence")
        return verdict

    if verdict.score is None:
        raise ValueError("scored verdict requires a score")
    validate_score_anchor(verdict.score, field="scored verdict score")
    if not verdict.evidence:
        raise ValueError("scored verdict requires at least one evidence citation")
    return verdict
