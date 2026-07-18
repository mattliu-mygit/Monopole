"""Canonical persisted records for judge inference and panel outcomes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from math import isfinite
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from weave_agent_signals.judges.inference import JsonSchemaSpec

_REQUEST_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_ERROR_MESSAGE = 2_000


def judge_request_id(
    *,
    requested_model_id: str,
    provider_model: str,
    messages: Sequence[Mapping[str, str]],
    response_schema: JsonSchemaSpec,
    temperature: float,
    max_tokens: int,
    protocol_version: str,
) -> str:
    """Return the exact identity of a model request that may be safely reused."""

    request = {
        "requested_model_id": requested_model_id,
        "provider_model": provider_model,
        "messages": [dict(message) for message in messages],
        "response_schema": {
            "name": response_schema.name,
            "schema": dict(response_schema.schema),
        },
        "temperature": temperature,
        "max_tokens": max_tokens,
        "protocol_version": protocol_version,
    }
    encoded = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JudgeCallAudit(_Record):
    resolved_model: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    output_mode: str | None = None
    schema_name: str = Field(min_length=1)
    schema_fallback_reason: str | None = None
    transport_request_count: int = Field(ge=0)
    raw_output_digest: str | None = None
    error_type: str | None = None
    message: str | None = Field(default=None, max_length=_MAX_ERROR_MESSAGE)

    @field_validator("usage")
    @classmethod
    def _valid_usage(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key.strip() for key in value):
            raise ValueError("usage keys must be nonblank")
        if any(type(count) is not int or count < 0 for count in value.values()):
            raise ValueError("usage values must be nonnegative integers")
        return value

    @field_validator("raw_output_digest")
    @classmethod
    def _valid_output_digest(cls, value: str | None) -> str | None:
        if value is not None and _HEX_DIGEST.fullmatch(value) is None:
            raise ValueError("raw_output_digest must be a lowercase SHA-256 digest")
        return value


class JudgeCallRecord(_Record):
    request_id: str
    phase: Literal["digest", "window", "merge"]
    conversation_id: str = Field(min_length=1)
    reviewer_position: int = Field(ge=1, le=3)
    requested_model_id: str = Field(min_length=1)
    rubric_id: str | None = None
    status: Literal["succeeded", "failed"]
    reusable: bool
    result: dict[str, Any] | None
    audit: JudgeCallAudit
    created_at: datetime

    @field_validator("request_id")
    @classmethod
    def _valid_request_id(cls, value: str) -> str:
        if _REQUEST_ID.fullmatch(value) is None:
            raise ValueError("request_id must be a lowercase SHA-256 identity")
        return value

    @model_validator(mode="after")
    def _valid_scope_and_result(self) -> Self:
        if self.phase == "digest" and self.rubric_id is not None:
            raise ValueError("digest call cannot have a rubric_id")
        if self.phase != "digest" and not self.rubric_id:
            raise ValueError("window and merge calls require rubric_id")
        if self.status == "failed":
            if self.reusable or self.result is not None:
                raise ValueError("failed call cannot be reusable or contain a result")
            if not self.audit.error_type:
                raise ValueError("failed call requires an error_type")
        elif self.reusable and self.result is None:
            raise ValueError("reusable success requires a result")
        elif self.audit.error_type is not None or self.audit.message is not None:
            raise ValueError("successful call cannot contain an error")
        return self


class ReviewerOutcome(_Record):
    position: int = Field(ge=1, le=3)
    requested_model_id: str = Field(min_length=1)
    status: Literal["succeeded", "abstained", "failed", "skipped"]
    score: float | None = None
    rationale: str | None = None
    evidence_ids: tuple[str, ...] = ()
    skip_reason: Literal["insufficient_context_capacity"] | None = None
    behavioral_feedback: dict[str, str | None] | None = None
    call_ids: tuple[str, ...] = ()

    @field_validator("score")
    @classmethod
    def _valid_score(cls, value: float | None) -> float | None:
        if value is not None and (not isfinite(value) or not 0 <= value <= 1):
            raise ValueError("score must be finite and between 0 and 1")
        return value

    @field_validator("call_ids")
    @classmethod
    def _valid_call_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(_REQUEST_ID.fullmatch(value) is None for value in values):
            raise ValueError("call_ids must contain SHA-256 request identities")
        return values


class PanelResult(_Record):
    conversation_id: str = Field(min_length=1)
    rubric_id: str = Field(min_length=1)
    status: Literal["complete", "degraded", "not_evaluable", "failed"]
    rating: float | None
    attempts: tuple[ReviewerOutcome, ...]
    successful_count: int = Field(ge=0, le=3)
    minimum: float | None
    maximum: float | None
    spread: float | None

    @field_validator("rating", "minimum", "maximum", "spread")
    @classmethod
    def _valid_metric(cls, value: float | None) -> float | None:
        if value is not None and (not isfinite(value) or not 0 <= value <= 1):
            raise ValueError("panel metrics must be finite and between 0 and 1")
        return value

    @model_validator(mode="after")
    def _valid_attempt_order(self) -> Self:
        positions = tuple(attempt.position for attempt in self.attempts)
        if positions != tuple(sorted(positions)) or len(set(positions)) != len(positions):
            raise ValueError("attempts must have unique positions in order")
        if self.successful_count != sum(attempt.status == "succeeded" for attempt in self.attempts):
            raise ValueError("successful_count must match attempts")
        return self


__all__ = [
    "JudgeCallAudit",
    "JudgeCallRecord",
    "PanelResult",
    "ReviewerOutcome",
    "judge_request_id",
]
