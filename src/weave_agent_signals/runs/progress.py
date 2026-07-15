"""Strict, persistable progress snapshots for long-running reflection work."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from weave_agent_signals.run_config import MAX_CANDIDATE_BUDGET

MAX_REFLECTION_EVENTS = 100
MAX_REFLECTION_ATTEMPTS = MAX_CANDIDATE_BUDGET
MAX_PROGRESS_MESSAGE_CHARS = 500
MAX_PROGRESS_ERROR_CHARS = 500
MAX_PROGRESS_PATH_CHARS = 500
MAX_PROGRESS_EXCERPT_CHARS = 1_000
MAX_PROGRESS_PATHS = 50

_EVENT_DETAIL_KEYS = frozenset(
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
_COUNTER_KEYS = ("attempted", "valid", "rejected", "scored")
_REQUIRED_EVENT_KEYS = frozenset({"id", "at", "phase", "message"})
_ALLOWED_EVENT_KEYS = _REQUIRED_EVENT_KEYS | _EVENT_DETAIL_KEYS
_RECORD_DETAIL_KEYS = _EVENT_DETAIL_KEYS | frozenset(_COUNTER_KEYS) | {"total_attempts"}
_REQUIRED_SNAPSHOT_KEYS = frozenset(
    {
        "phase",
        "status_message",
        "started_at",
        *_COUNTER_KEYS,
        "total_attempts",
        "events",
    }
)
_SNAPSHOT_CONTEXT_KEYS = frozenset(
    {
        "proposal_writer",
        "proposal_evaluator",
        "no_improvement_patience",
    }
)
_ALLOWED_SNAPSHOT_KEYS = _REQUIRED_SNAPSHOT_KEYS | _SNAPSHOT_CONTEXT_KEYS
_PHASE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|secret|password|authorization|credential)\b\s*[:=]\s*)"
    r"[^\s,;}]+"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[^\s,;}]+")
_URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@")
log = logging.getLogger(__name__)

ProgressSnapshot = dict[str, Any]
ProgressSink = Callable[[ProgressSnapshot], None]
Clock = Callable[[], datetime]


class ReflectionProgressRecorder:
    """Build and persist complete snapshots through one strict boundary."""

    def __init__(
        self,
        persist: ProgressSink,
        *,
        total_attempts: int = 0,
        context: Mapping[str, Any] | None = None,
        initial_snapshot: Mapping[str, Any] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._persist = persist
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        requested_total = _strict_counter(
            total_attempts,
            "total_attempts",
            maximum=MAX_REFLECTION_ATTEMPTS,
        )
        supplied_context = _sanitize_context(context or {})
        initial = _validated_snapshot(initial_snapshot) if initial_snapshot is not None else None
        initial_context = (
            {key: initial[key] for key in _SNAPSHOT_CONTEXT_KEYS if key in initial}
            if initial is not None
            else {}
        )
        for key in supplied_context.keys() & initial_context.keys():
            if supplied_context[key] != initial_context[key]:
                raise ValueError(f"Persisted reflection snapshot has conflicting {key}")

        self._context = {**initial_context, **supplied_context}
        self._started_at = (
            initial["started_at"]
            if initial is not None
            else _timestamp(self._clock(), "started_at")
        )
        self._events = deepcopy(initial["events"]) if initial is not None else []
        self._next_id = max((event["id"] for event in self._events), default=0) + 1
        self._counts = {key: initial[key] if initial is not None else 0 for key in _COUNTER_KEYS}
        self._total_attempts = max(
            requested_total,
            initial["total_attempts"] if initial is not None else 0,
        )

    def record(self, phase: str, message: str, **details: Any) -> None:
        """Append one validated semantic event and persist a complete snapshot."""
        normalized = _normalize_record_details(details)
        next_counts = dict(self._counts)
        for key in _COUNTER_KEYS:
            if key in normalized:
                next_counts[key] = max(next_counts[key], normalized[key])
        next_total = self._total_attempts
        if "total_attempts" in normalized:
            next_total = max(next_total, normalized["total_attempts"])
        _validate_counter_relationships(next_counts, next_total)

        event: dict[str, Any] = {
            "id": self._next_id,
            "at": _timestamp(self._clock(), "event timestamp"),
            "phase": _sanitize_phase(phase),
            "message": _sanitize_text(
                message,
                "message",
                MAX_PROGRESS_MESSAGE_CHARS,
            ),
        }
        event.update({key: normalized[key] for key in _EVENT_DETAIL_KEYS if key in normalized})
        event = _validate_event(event)
        events = (self._events + [event])[-MAX_REFLECTION_EVENTS:]
        snapshot = _validated_snapshot(
            {
                **deepcopy(self._context),
                "phase": event["phase"],
                "status_message": event["message"],
                "started_at": self._started_at,
                **next_counts,
                "total_attempts": next_total,
                "events": events,
            }
        )

        self._next_id += 1
        self._events = deepcopy(snapshot["events"])
        self._counts = {key: snapshot[key] for key in _COUNTER_KEYS}
        self._total_attempts = snapshot["total_attempts"]
        try:
            self._persist(deepcopy(snapshot))
        except Exception as exc:
            # Progress is observational. Never expose sink text or abort work.
            log.warning(
                "Could not persist reflection progress: error_type=%s",
                type(exc).__name__,
            )

    def handle(self, event: Mapping[str, Any]) -> None:
        """Adapt a reflector event dictionary to :meth:`record`."""
        if not isinstance(event, Mapping):
            raise ValueError("reflection progress event must be an object")
        payload = dict(event)
        if "phase" not in payload or "message" not in payload:
            raise ValueError("reflection progress event requires phase and message")
        phase = payload.pop("phase")
        message = payload.pop("message")
        self.record(phase, message, **payload)


def _sanitize_context(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("reflection progress context must be an object")
    context = dict(value)
    unexpected = set(context) - _SNAPSHOT_CONTEXT_KEYS
    if unexpected:
        raise ValueError("reflection progress context contains unexpected fields")
    for key in ("proposal_writer", "proposal_evaluator"):
        if key in context:
            context[key] = _sanitize_text(context[key], key, 200)
    if "no_improvement_patience" in context:
        patience = context["no_improvement_patience"]
        if type(patience) is not int or not 1 <= patience <= MAX_REFLECTION_ATTEMPTS:
            raise ValueError("no_improvement_patience must be an integer between 1 and 10")
    return context


def _normalize_record_details(details: Mapping[str, Any]) -> dict[str, Any]:
    unexpected = set(details) - _RECORD_DETAIL_KEYS
    if unexpected:
        raise ValueError("unexpected progress event fields")
    normalized: dict[str, Any] = {}
    for key, value in details.items():
        if value is None and key in _EVENT_DETAIL_KEYS:
            continue
        if key in _COUNTER_KEYS or key == "total_attempts":
            normalized[key] = _strict_counter(
                value,
                key,
                maximum=MAX_REFLECTION_ATTEMPTS,
            )
        elif key in _EVENT_DETAIL_KEYS:
            normalized[key] = _normalize_event_detail(key, value, sanitize=True)
    return normalized


def _normalize_event_detail(key: str, value: Any, *, sanitize: bool) -> Any:
    if key == "candidate":
        if type(value) is not int or not 1 <= value <= MAX_REFLECTION_ATTEMPTS:
            raise ValueError("candidate must be an integer between 1 and 10")
        return value
    if key == "score":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score must be a finite number between 0 and 1")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be a finite number between 0 and 1")
        return score
    if key == "acting_role":
        if value not in {"proposal_writer", "proposal_evaluator"}:
            raise ValueError("acting_role is invalid")
        return value
    if key == "response_digest":
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            raise ValueError("response_digest must be a SHA-256 digest")
        return value
    if key == "changed_paths":
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) > MAX_PROGRESS_PATHS
        ):
            raise ValueError("changed_paths must be a bounded list of paths")
        paths = [
            _validated_or_sanitized_text(
                path,
                "changed_paths",
                MAX_PROGRESS_PATH_CHARS,
                sanitize=sanitize,
            )
            for path in value
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("changed_paths must be unique")
        return paths

    limits = {
        "model": 200,
        "attempt_id": 200,
        "evaluation_id": 200,
        "status": 100,
        "error_type": 100,
        "error": MAX_PROGRESS_ERROR_CHARS,
        "response_excerpt": MAX_PROGRESS_EXCERPT_CHARS,
    }
    if key not in limits:
        raise ValueError(f"unsupported progress event field: {key}")
    return _validated_or_sanitized_text(
        value,
        key,
        limits[key],
        sanitize=sanitize,
    )


def _validated_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Persisted reflection snapshot must be an object")
    snapshot = deepcopy(dict(value))
    if not set(snapshot).issubset(_ALLOWED_SNAPSHOT_KEYS):
        raise ValueError("Persisted reflection snapshot contains retired progress fields")
    if not _REQUIRED_SNAPSHOT_KEYS.issubset(snapshot):
        raise ValueError(
            "Persisted reflection snapshot must satisfy the complete current progress contract"
        )

    snapshot["phase"] = _validated_phase(snapshot["phase"])
    snapshot["status_message"] = _validated_or_sanitized_text(
        snapshot["status_message"],
        "status_message",
        MAX_PROGRESS_MESSAGE_CHARS,
        sanitize=False,
    )
    snapshot["started_at"] = _validated_timestamp(snapshot["started_at"], "started_at")
    for key in _COUNTER_KEYS:
        snapshot[key] = _strict_counter(
            snapshot[key],
            key,
            maximum=MAX_REFLECTION_ATTEMPTS,
        )
    snapshot["total_attempts"] = _strict_counter(
        snapshot["total_attempts"],
        "total_attempts",
        maximum=MAX_REFLECTION_ATTEMPTS,
    )
    _validate_counter_relationships(
        {key: snapshot[key] for key in _COUNTER_KEYS},
        snapshot["total_attempts"],
    )

    raw_events = snapshot["events"]
    if not isinstance(raw_events, list) or len(raw_events) > MAX_REFLECTION_EVENTS:
        raise ValueError("Persisted reflection snapshot has invalid progress events")
    snapshot["events"] = [_validate_event(event) for event in raw_events]
    event_ids = [event["id"] for event in snapshot["events"]]
    if event_ids != sorted(set(event_ids)):
        raise ValueError("Persisted reflection snapshot event IDs must be unique and increasing")
    if snapshot["events"]:
        latest = snapshot["events"][-1]
        if snapshot["phase"] != latest["phase"] or snapshot["status_message"] != latest["message"]:
            raise ValueError("Persisted reflection snapshot does not match its latest event")

    raw_context = {key: snapshot[key] for key in _SNAPSHOT_CONTEXT_KEYS if key in snapshot}
    validated_context = _sanitize_context(raw_context)
    if validated_context != raw_context:
        raise ValueError("Persisted reflection snapshot has unsafe progress context")
    snapshot.update(validated_context)
    return snapshot


def _validate_event(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Persisted reflection snapshot has invalid progress events")
    event = deepcopy(dict(value))
    if not _REQUIRED_EVENT_KEYS.issubset(event) or not set(event).issubset(_ALLOWED_EVENT_KEYS):
        raise ValueError("Persisted reflection snapshot has unexpected progress event fields")
    if type(event["id"]) is not int or event["id"] < 1:
        raise ValueError("Persisted reflection snapshot has invalid progress event ID")
    event["at"] = _validated_timestamp(event["at"], "event timestamp")
    event["phase"] = _validated_phase(event["phase"])
    event["message"] = _validated_or_sanitized_text(
        event["message"],
        "message",
        MAX_PROGRESS_MESSAGE_CHARS,
        sanitize=False,
    )
    for key in set(event) - _REQUIRED_EVENT_KEYS:
        event[key] = _normalize_event_detail(key, event[key], sanitize=False)
    return event


def _validate_counter_relationships(counts: Mapping[str, int], total_attempts: int) -> None:
    if counts["valid"] + counts["rejected"] > counts["attempted"]:
        raise ValueError("completed attempts cannot exceed attempted proposals")
    if counts["scored"] > counts["valid"]:
        raise ValueError("scored proposals cannot exceed valid proposals")
    if counts["attempted"] > total_attempts:
        raise ValueError("attempted proposals cannot exceed total_attempts")


def _strict_counter(value: Any, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}")
    return value


def _sanitize_phase(value: Any) -> str:
    if not isinstance(value, str) or _PHASE.fullmatch(value) is None:
        raise ValueError("phase must be a lowercase progress identifier")
    return value


def _validated_phase(value: Any) -> str:
    try:
        return _sanitize_phase(value)
    except ValueError as exc:
        raise ValueError("Persisted reflection snapshot has invalid progress phase") from exc


def _sanitize_text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    text = "".join(
        character if character.isprintable() or character in "\n\r\t" else " "
        for character in value
    )
    text = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    text = _BEARER_SECRET.sub("Bearer [REDACTED]", text)
    text = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", text)
    return text[:limit]


def _validated_or_sanitized_text(
    value: Any,
    name: str,
    limit: int,
    *,
    sanitize: bool,
) -> str:
    text = _sanitize_text(value, name, limit)
    if not sanitize and text != value:
        raise ValueError(f"Persisted reflection snapshot has unsafe or oversized {name}")
    return text


def _timestamp(value: Any, name: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.isoformat()


def _validated_timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Persisted reflection snapshot has invalid {name}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Persisted reflection snapshot has invalid {name}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Persisted reflection snapshot has invalid {name}")
    return value
