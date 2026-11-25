"""Exact discovery and hydration for immutable evaluation-run cohorts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from weave_agent_signals.ces_client import CesSession, session_or_none
from weave_agent_signals.judges.families import model_family
from weave_agent_signals.models import SessionView, TurnSpan, session_is_evaluable
from weave_agent_signals.runs.store import DataSelection

_COHORT_SCHEMA_VERSION = 3
_COHORT_KEYS = {
    "schema_version",
    "cohort_id",
    "pinned_at",
    "turn_count",
    "session_count",
    "turns",
    "sessions",
}
_TURN_KEYS = {
    "trace_id",
    "case_ids",
    "conversation_id",
    "started_at",
    "model",
    "model_family",
    "effort_level",
}
# A turn's trace_id is its first CES Case, and a session's conversation_id is its session
# Trajectory. Cases never change, so pinned Case IDs always read back the same content.
_SESSION_KEYS = {"conversation_id", "turn_count"}


def _selection_boundary(value: str | None) -> datetime | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Cannot parse selection datetime: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("Selection boundaries must include a timezone offset")
    return parsed


def _selection_bounds(selection: DataSelection) -> tuple[datetime | None, datetime | None]:
    if not isinstance(selection, DataSelection):
        raise ValueError("selection must be a DataSelection")
    if selection.timezone is not None:
        try:
            ZoneInfo(selection.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {selection.timezone}") from exc
    if not selection.session_ids:
        raise ValueError("selection must include at least one session")
    if any(not session_id.strip() for session_id in selection.session_ids):
        raise ValueError("selection session IDs must be nonblank")
    if len(selection.session_ids) != len(set(selection.session_ids)):
        raise ValueError("selection session IDs must be unique")

    since = _selection_boundary(selection.since)
    until = _selection_boundary(selection.until)
    if since is not None and until is not None and since > until:
        raise ValueError("selection since must not be after until")
    return since, until


def _require_aware_turn(turn: TurnSpan) -> None:
    if turn.started_at.tzinfo is None or turn.started_at.utcoffset() is None:
        raise ValueError(f"turn {turn.trace_id} started_at must include a timezone")


def _session_views(turns: list[TurnSpan]) -> dict[str, SessionView]:
    grouped: dict[str, list[TurnSpan]] = {}
    for turn in turns:
        grouped.setdefault(turn.conversation_id, []).append(turn)

    sessions: dict[str, SessionView] = {}
    for conversation_id in sorted(grouped):
        ordered = sorted(
            grouped[conversation_id],
            key=lambda turn: (turn.started_at, turn.trace_id),
        )
        sessions[conversation_id] = SessionView(
            conversation_id=conversation_id,
            turns=ordered,
            config_version=ordered[0].config_version,
            git_branch=ordered[0].git_branch,
        )
    return sessions


def _with_inferred_agent(turns: list[TurnSpan]) -> list[TurnSpan]:
    """Give each pinned turn with no model, such as a /compact, the model and effort of the closest
    earlier pinned turn in its session, else the first later one. Turns must be in time order. The
    adapter records the same values for turns it writes now; older Cases keep them empty."""

    first: dict[str, TurnSpan] = {}
    for turn in turns:
        if turn.model:
            first.setdefault(turn.conversation_id, turn)
    last: dict[str, TurnSpan] = {}
    filled: list[TurnSpan] = []
    for turn in turns:
        if turn.model:
            last[turn.conversation_id] = turn
        elif source := last.get(turn.conversation_id) or first.get(turn.conversation_id):
            turn = replace(turn, model=source.model, effort_level=source.effort_level)
        filled.append(turn)
    return filled


def _turn_entry(turn: TurnSpan) -> dict[str, Any]:
    return {
        "trace_id": turn.trace_id,
        "case_ids": list(turn.case_ids),
        "conversation_id": turn.conversation_id,
        "started_at": turn.started_at.isoformat(),
        "model": turn.model,
        "model_family": model_family(turn.model or ""),
        "effort_level": turn.effort_level,
    }


def _session_entries(sessions: Mapping[str, SessionView]) -> list[dict[str, Any]]:
    return [
        {"conversation_id": conversation_id, "turn_count": len(session.turns)}
        for conversation_id, session in sessions.items()
    ]


def _build_cohort(turns: list[TurnSpan]) -> dict[str, Any]:
    if not turns:
        raise ValueError("selection matched no turns")

    ordered = sorted(turns, key=lambda turn: (turn.started_at, turn.trace_id))
    trace_ids: set[str] = set()
    case_ids: set[str] = set()
    for turn in ordered:
        _require_aware_turn(turn)
        if not turn.trace_id or not turn.conversation_id:
            raise ValueError("cohort turns require trace and conversation IDs")
        if not turn.case_ids or turn.case_ids[0] != turn.trace_id:
            raise ValueError("cohort turns require their CES Case IDs")
        if turn.trace_id in trace_ids or case_ids & set(turn.case_ids):
            raise ValueError("cohort turn identities must be unique")
        trace_ids.add(turn.trace_id)
        case_ids.update(turn.case_ids)

    turn_entries = [_turn_entry(turn) for turn in _with_inferred_agent(ordered)]
    session_entries = _session_entries(_session_views(ordered))
    identity = {
        "schema_version": _COHORT_SCHEMA_VERSION,
        "turn_count": len(turn_entries),
        "session_count": len(session_entries),
        "turns": turn_entries,
        "sessions": session_entries,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return {
        **identity,
        "cohort_id": f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}",
        "pinned_at": datetime.now(timezone.utc).isoformat(),
    }


def discover_turn_cohort(
    selection: DataSelection,
    *,
    ces_factory: Callable[[], CesSession],
) -> dict[str, Any]:
    """Resolve a mutable selection of session Trajectories to one content-addressed cohort."""

    since, until = _selection_bounds(selection)
    with ces_factory() as ces:
        sessions = {sid: session_or_none(ces.client, sid) for sid in selection.session_ids}

    filtered: list[TurnSpan] = []
    for session in sessions.values():
        for turn in session.turns if session is not None else ():
            _require_aware_turn(turn)
            if since is not None and turn.started_at < since:
                continue
            if until is not None and turn.started_at > until:
                continue
            filtered.append(turn)
    if not filtered:
        raise ValueError("selection matched no turns")
    discovered_sessions = {turn.conversation_id for turn in filtered}
    missing_sessions = sorted(set(selection.session_ids) - discovered_sessions)
    if missing_sessions:
        raise ValueError("missing selected sessions: " + ", ".join(missing_sessions))
    ineligible = sorted(
        sid for sid, session in sessions.items() if not session_is_evaluable(session.turns)
    )
    if ineligible:
        raise ValueError("selected sessions are not evaluable: " + ", ".join(ineligible))

    return _build_cohort(filtered)


def _parse_aware_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RuntimeError("invalid pinned turn cohort: " + label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("invalid pinned turn cohort: " + label) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError("invalid pinned turn cohort: " + label)
    return parsed


def _validated_cohort(
    cohort: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(cohort, Mapping) or set(cohort) != _COHORT_KEYS:
        raise RuntimeError("invalid pinned turn cohort: top-level shape")
    if cohort.get("schema_version") != _COHORT_SCHEMA_VERSION:
        raise RuntimeError("invalid pinned turn cohort: schema version")
    _parse_aware_timestamp(cohort.get("pinned_at"), "pinned_at")

    turn_entries = cohort.get("turns")
    session_entries = cohort.get("sessions")
    if not isinstance(turn_entries, list) or not turn_entries:
        raise RuntimeError("invalid pinned turn cohort: turns")
    if not isinstance(session_entries, list) or not session_entries:
        raise RuntimeError("invalid pinned turn cohort: sessions")
    if cohort.get("turn_count") != len(turn_entries):
        raise RuntimeError("invalid pinned turn cohort: turn count")
    if cohort.get("session_count") != len(session_entries):
        raise RuntimeError("invalid pinned turn cohort: session count")

    seen_traces: set[str] = set()
    seen_cases: set[str] = set()
    parsed_turns: list[tuple[datetime, str]] = []
    session_counts: dict[str, int] = {}
    for entry in turn_entries:
        if not isinstance(entry, dict) or set(entry) != _TURN_KEYS:
            raise RuntimeError("invalid pinned turn cohort: turn metadata")
        trace_id = entry.get("trace_id")
        case_ids = entry.get("case_ids")
        conversation_id = entry.get("conversation_id")
        model = entry.get("model")
        family = entry.get("model_family")
        effort_level = entry.get("effort_level")
        if not all(isinstance(item, str) and item for item in (trace_id, conversation_id, family)):
            raise RuntimeError("invalid pinned turn cohort: turn metadata")
        if (
            not isinstance(case_ids, list)
            or not case_ids
            or not all(isinstance(case_id, str) and case_id for case_id in case_ids)
            or case_ids[0] != trace_id
        ):
            raise RuntimeError("invalid pinned turn cohort: case IDs")
        if model is not None and (not isinstance(model, str) or not model):
            raise RuntimeError("invalid pinned turn cohort: model")
        if effort_level is not None and (not isinstance(effort_level, str) or not effort_level):
            raise RuntimeError("invalid pinned turn cohort: effort level")
        if family != model_family(model or ""):
            raise RuntimeError("invalid pinned turn cohort: model family")
        if (
            trace_id in seen_traces
            or seen_cases & set(case_ids)
            or len(set(case_ids)) != len(case_ids)
        ):
            raise RuntimeError("invalid pinned turn cohort: duplicate turn identity")
        seen_traces.add(trace_id)
        seen_cases.update(case_ids)
        parsed_turns.append(
            (_parse_aware_timestamp(entry.get("started_at"), "turn started_at"), trace_id)
        )
        session_counts[conversation_id] = session_counts.get(conversation_id, 0) + 1
    if parsed_turns != sorted(parsed_turns):
        raise RuntimeError("invalid pinned turn cohort: turn ordering")

    seen_sessions: set[str] = set()
    for entry in session_entries:
        if not isinstance(entry, dict) or set(entry) != _SESSION_KEYS:
            raise RuntimeError("invalid pinned turn cohort: session metadata")
        conversation_id = entry.get("conversation_id")
        turn_count = entry.get("turn_count")
        if (
            not isinstance(conversation_id, str)
            or not conversation_id
            or type(turn_count) is not int
            or turn_count < 1
            or conversation_id in seen_sessions
            or session_counts.get(conversation_id) != turn_count
        ):
            raise RuntimeError("invalid pinned turn cohort: session metadata")
        seen_sessions.add(conversation_id)
    if seen_sessions != set(session_counts):
        raise RuntimeError("invalid pinned turn cohort: session membership")
    if [entry["conversation_id"] for entry in session_entries] != sorted(seen_sessions):
        raise RuntimeError("invalid pinned turn cohort: session ordering")

    identity = {
        "schema_version": cohort["schema_version"],
        "turn_count": cohort["turn_count"],
        "session_count": cohort["session_count"],
        "turns": turn_entries,
        "sessions": session_entries,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    expected_id = f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
    if cohort.get("cohort_id") != expected_id:
        raise RuntimeError("invalid pinned turn cohort: cohort ID")
    return turn_entries, session_entries


def hydrate_turn_cohort(
    cohort: Mapping[str, Any],
    *,
    ces_factory: Callable[[], CesSession],
) -> tuple[list[TurnSpan], dict[str, SessionView]]:
    """Read exactly the pinned turns back from CES and reject any identity drift.

    Turns added to a pinned session after pinning are ignored, because the pin lists
    Case IDs. A pinned session or turn that CES no longer returns fails closed."""

    turn_entries, session_entries = _validated_cohort(cohort)
    with ces_factory() as ces:
        live = {
            entry["conversation_id"]: session_or_none(ces.client, entry["conversation_id"])
            for entry in session_entries
        }
    gone = sorted(sid for sid, session in live.items() if session is None)
    if gone:
        raise RuntimeError("Pinned turn cohort sessions are missing: " + ", ".join(gone))

    by_trace_id = {turn.trace_id: turn for session in live.values() for turn in session.turns}
    missing = [entry["trace_id"] for entry in turn_entries if entry["trace_id"] not in by_trace_id]
    if missing:
        raise RuntimeError(
            "Pinned turn cohort is incomplete; missing trace IDs: " + ", ".join(missing)
        )
    ordered = _with_inferred_agent([by_trace_id[entry["trace_id"]] for entry in turn_entries])
    sessions = _session_views(ordered)
    ineligible = sorted(
        conversation_id
        for conversation_id, session in sessions.items()
        if not session_is_evaluable(session.turns)
    )
    if ineligible:
        raise RuntimeError(
            "Pinned turn cohort contains sessions that are not evaluable: " + ", ".join(ineligible)
        )

    changed = [
        turn.trace_id
        for entry, turn in zip(turn_entries, ordered, strict=True)
        if _turn_entry(turn) != entry
    ]
    if changed:
        raise RuntimeError(
            "Pinned turn cohort metadata changed for trace IDs: " + ", ".join(changed)
        )
    if _session_entries(sessions) != session_entries:
        raise RuntimeError("Pinned turn cohort session metadata changed")

    return ordered, sessions
