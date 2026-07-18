"""Shared sanitized activity-event contracts for evaluation runs."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

RunEventStage = Literal["scoring", "judging", "reflecting"]
_PHASE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:[a-z0-9_.-]*(?:api[_-]?key|token|secret|password|authorization|credential)"
    r"[a-z0-9_.-]*)\b\s*[:=]\s*)(?:\\?[\"']?)([^\s,}\]\\\"']+)"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[^\s,;}]+")
_URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@")

_SCORING_FIELDS = frozenset(
    {
        "trace_id",
        "conversation_id",
        "model",
        "total",
        "scored",
        "written",
        "errors",
        "tokens",
        "tool_count",
        "steering",
        "denials",
    }
)
_JUDGING_FIELDS = frozenset(
    {
        "model",
        "conversation_id",
        "rubric",
        "artifact_id",
        "item_index",
        "item_total",
        "request_attempt",
        "max_attempts",
        "elapsed_seconds",
        "error_category",
        "retry_reason",
        "provider_status",
        "provider_error_code",
        "provider_error_message",
        "output_sha256",
        "exit_code",
        "stdout_chars",
        "stderr_chars",
        "prompt_characters",
        "output_mode",
        "estimated_input_tokens",
        "max_output_tokens",
        "model_context_tokens",
        "status",
    }
)
_REFLECTION_FIELDS = frozenset(
    {
        "candidate",
        "model",
        "acting_role",
        "attempt_id",
        "evaluation_id",
        "score",
        "status",
        "error_type",
        "error",
        "changed_paths",
        "response_digest",
        "response_excerpt",
    }
)
_TEXT_FIELDS = frozenset(
    {
        "trace_id",
        "conversation_id",
        "model",
        "rubric",
        "artifact_id",
        "error_category",
        "retry_reason",
        "provider_error_code",
        "provider_error_message",
        "output_mode",
        "status",
        "acting_role",
        "attempt_id",
        "evaluation_id",
        "error_type",
        "error",
        "response_excerpt",
    }
)
_POSITIVE_INTS = frozenset(
    {
        "item_index",
        "item_total",
        "request_attempt",
        "max_attempts",
        "prompt_characters",
        "estimated_input_tokens",
        "max_output_tokens",
        "model_context_tokens",
        "candidate",
    }
)
_NONNEGATIVE_INTS = frozenset(
    {
        "total",
        "scored",
        "written",
        "errors",
        "tokens",
        "tool_count",
        "steering",
        "denials",
        "stdout_chars",
        "stderr_chars",
    }
)


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunEventDraft(_Event):
    stage: RunEventStage
    at: datetime
    phase: str
    message: str
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("event timestamp must include a timezone")
        return value.astimezone(UTC)


class RunEvent(RunEventDraft):
    sequence: int = Field(ge=1)


def _text(value: object, *, limit: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError("event text must be a string")
    text = " ".join(value.split())
    text = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    text = _BEARER_SECRET.sub("Bearer [REDACTED]", text)
    text = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", text)
    if not text:
        raise ValueError("event text must be nonblank")
    return text[:limit]


def _details(stage: RunEventStage, value: Mapping[str, Any]) -> dict[str, JsonValue]:
    allowed = {
        "scoring": _SCORING_FIELDS,
        "judging": _JUDGING_FIELDS,
        "reflecting": _REFLECTION_FIELDS,
    }[stage]
    unexpected = set(value) - allowed
    if unexpected:
        raise ValueError(f"unexpected {stage.removesuffix('ing')}ing event fields")
    result: dict[str, JsonValue] = {}
    for key, item in value.items():
        if item is None:
            continue
        if key in _TEXT_FIELDS:
            result[key] = _text(item, limit=1_000 if key == "response_excerpt" else 500)
        elif key in _POSITIVE_INTS:
            if type(item) is not int or item <= 0:
                raise ValueError(f"{key} must be a positive integer")
            result[key] = item
        elif key in _NONNEGATIVE_INTS:
            if type(item) is not int or item < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
            result[key] = item
        elif key in {"elapsed_seconds", "score"}:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{key} must be finite")
            number = float(item)
            maximum = 1.0 if key == "score" else math.inf
            if not math.isfinite(number) or not 0 <= number <= maximum:
                raise ValueError(f"{key} must be finite and nonnegative")
            result[key] = number
        elif key == "provider_status":
            if type(item) is not int or not 100 <= item <= 599:
                raise ValueError("provider_status must be an HTTP status")
            result[key] = item
        elif key == "exit_code":
            if type(item) is not int:
                raise ValueError("exit_code must be an integer")
            result[key] = item
        elif key in {"output_sha256", "response_digest"}:
            prefix = "sha256:" if key == "response_digest" else ""
            digest = item.removeprefix(prefix) if isinstance(item, str) else ""
            if _SHA256.fullmatch(digest) is None:
                raise ValueError(f"{key} must be a SHA-256 digest")
            result[key] = item
        elif key == "changed_paths":
            if isinstance(item, (str, bytes)) or not isinstance(item, Sequence):
                raise ValueError("changed_paths must be a sequence")
            paths = [_text(path) for path in item]
            if len(paths) > 50 or len(paths) != len(set(paths)):
                raise ValueError("changed_paths must be bounded and unique")
            result[key] = paths
        else:  # The allowlist and normalizers must evolve together.
            raise ValueError(f"unsupported {stage} event field: {key}")
    return result


def normalize_scoring_details(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    return _details("scoring", value)


def normalize_judging_details(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    return _details("judging", value)


def normalize_reflection_details(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    return _details("reflecting", value)


def sanitize_event(
    stage: RunEventStage,
    phase: str,
    message: str,
    details: Mapping[str, Any],
    at: datetime,
) -> RunEventDraft:
    if _PHASE.fullmatch(phase) is None:
        raise ValueError("event phase is invalid")
    return RunEventDraft(
        stage=stage,
        at=at,
        phase=phase,
        message=_text(message),
        details=_details(stage, details),
    )


__all__ = [
    "RunEvent",
    "RunEventDraft",
    "normalize_judging_details",
    "normalize_reflection_details",
    "normalize_scoring_details",
    "sanitize_event",
]
