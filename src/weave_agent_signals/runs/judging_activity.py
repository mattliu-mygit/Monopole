"""Bounded, sanitized activity history for long-running judging."""

from __future__ import annotations

import math
import re
import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

MAX_JUDGING_EVENTS = 1_000
MAX_ACTIVITY_TEXT_CHARACTERS = 500

_DETAIL_FIELDS = frozenset(
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
_TEXT_FIELDS = frozenset(
    {
        "model",
        "conversation_id",
        "rubric",
        "artifact_id",
        "error_category",
        "retry_reason",
        "provider_error_code",
        "provider_error_message",
        "output_mode",
        "status",
    }
)
_POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "item_index",
        "item_total",
        "request_attempt",
        "max_attempts",
        "prompt_characters",
        "estimated_input_tokens",
        "max_output_tokens",
        "model_context_tokens",
    }
)
_NONNEGATIVE_INTEGER_FIELDS = frozenset({"stdout_chars", "stderr_chars"})
_PHASE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|secret|password|authorization|credential)\b\s*[:=]\s*)"
    r"[^\s,;}]+"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[^\s,;}]+")

Clock = Callable[[], datetime]


class JudgingActivityLog:
    """Build safe activity events without owning run persistence."""

    def __init__(
        self,
        *,
        initial_snapshot: Mapping[str, Any] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        initial = dict(initial_snapshot or {})
        started_at = initial.get("started_at")
        self.started_at = (
            _timestamp(started_at) if isinstance(started_at, str) else _timestamp(self._clock())
        )
        raw_events = initial.get("events")
        self.events = (
            [_validate_existing_event(event) for event in raw_events][-MAX_JUDGING_EVENTS:]
            if isinstance(raw_events, list)
            else []
        )
        self._next_id = max((event["id"] for event in self.events), default=0) + 1
        self.phase = str(initial.get("phase") or "waiting")
        self.status_message = _sanitize_text(
            str(initial.get("status_message") or "Waiting for judging activity")
        )

    def record(self, value: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            raw = dict(value)
            phase = _sanitize_phase(raw.get("phase"))
            message = _sanitize_text(raw.get("message"))
            event: dict[str, Any] = {
                "id": self._next_id,
                "at": _timestamp(self._clock()),
                "phase": phase,
                "message": message,
            }
            for key in _DETAIL_FIELDS:
                if key in raw and raw[key] is not None:
                    event[key] = _normalize_detail(key, raw[key])
            self.events = (self.events + [event])[-MAX_JUDGING_EVENTS:]
            self._next_id += 1
            self.phase = phase
            self.status_message = message
            return deepcopy(event)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "phase": self.phase,
                "status_message": self.status_message,
                "started_at": self.started_at,
                "events": deepcopy(self.events),
            }


def _sanitize_phase(value: object) -> str:
    if not isinstance(value, str) or _PHASE.fullmatch(value) is None:
        raise ValueError("judging activity phase is invalid")
    return value


def _sanitize_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("judging activity text must be a string")
    text = " ".join(value.split())
    text = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    text = _BEARER_SECRET.sub("Bearer [REDACTED]", text)
    if not text:
        raise ValueError("judging activity text must be nonblank")
    return text[:MAX_ACTIVITY_TEXT_CHARACTERS]


def _timestamp(value: datetime | str) -> str:
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    except ValueError as error:
        raise ValueError("judging activity timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError("judging activity timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _normalize_detail(key: str, value: object) -> object:
    if key in _TEXT_FIELDS:
        return _sanitize_text(value)
    if key in _POSITIVE_INTEGER_FIELDS:
        if type(value) is not int or value <= 0:
            raise ValueError(f"judging activity {key} must be a positive integer")
        return value
    if key in _NONNEGATIVE_INTEGER_FIELDS:
        if type(value) is not int or value < 0:
            raise ValueError(f"judging activity {key} must be a nonnegative integer")
        return value
    if key == "exit_code":
        if type(value) is not int:
            raise ValueError("judging activity exit_code must be an integer")
        return value
    if key == "provider_status":
        if type(value) is not int or not 100 <= value <= 599:
            raise ValueError("judging activity provider_status must be an HTTP status")
        return value
    if key == "elapsed_seconds":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("judging activity elapsed_seconds must be nonnegative")
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError("judging activity elapsed_seconds must be nonnegative")
        return result
    if key == "output_sha256":
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("judging activity output_sha256 is invalid")
        return value
    raise ValueError(f"unsupported judging activity detail: {key}")


def _validate_existing_event(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("persisted judging activity event must be an object")
    raw = dict(value)
    if set(raw) - ({"id", "at", "phase", "message"} | _DETAIL_FIELDS):
        raise ValueError("persisted judging activity event contains unexpected fields")
    event_id = raw.get("id")
    if type(event_id) is not int or event_id <= 0:
        raise ValueError("persisted judging activity event ID is invalid")
    event = {
        "id": event_id,
        "at": _timestamp(raw.get("at")),
        "phase": _sanitize_phase(raw.get("phase")),
        "message": _sanitize_text(raw.get("message")),
    }
    for key in _DETAIL_FIELDS:
        if key in raw and raw[key] is not None:
            event[key] = _normalize_detail(key, raw[key])
    return event
