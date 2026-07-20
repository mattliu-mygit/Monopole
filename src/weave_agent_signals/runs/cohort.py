"""Exact discovery and hydration for immutable evaluation-run cohorts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from weave_agent_signals.client import WeaveClient
from weave_agent_signals.judges.families import model_family
from weave_agent_signals.models import SessionView, TurnSpan, session_is_evaluable
from weave_agent_signals.runs.store import DataSelection

_COHORT_SCHEMA_VERSION = 2
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
    "weave_ref",
    "conversation_id",
    "started_at",
    "model",
    "model_family",
    "effort_level",
}
_SESSION_KEYS = {"conversation_id", "weave_ref", "turn_count"}


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


def _build_cohort(
    turns: list[TurnSpan],
    *,
    entity: str,
    project: str,
) -> dict[str, Any]:
    if not turns:
        raise ValueError("selection matched no turns")
    if not entity.strip() or not project.strip():
        raise ValueError("entity and project must be nonblank")

    ordered = sorted(turns, key=lambda turn: (turn.started_at, turn.trace_id))
    trace_ids: set[str] = set()
    weave_refs: set[str] = set()
    for turn in ordered:
        _require_aware_turn(turn)
        if not turn.trace_id or not turn.conversation_id:
            raise ValueError("cohort turns require trace and conversation IDs")
        weave_ref = turn.ref_for(entity, project)
        if turn.trace_id in trace_ids or weave_ref in weave_refs:
            raise ValueError("cohort turn identities must be unique")
        trace_ids.add(turn.trace_id)
        weave_refs.add(weave_ref)

    sessions = _session_views(ordered)
    turn_entries = [
        {
            "trace_id": turn.trace_id,
            "weave_ref": turn.ref_for(entity, project),
            "conversation_id": turn.conversation_id,
            "started_at": turn.started_at.isoformat(),
            "model": turn.model,
            "model_family": model_family(turn.model or ""),
            "effort_level": turn.effort_level,
        }
        for turn in ordered
    ]
    session_entries = [
        {
            "conversation_id": conversation_id,
            "weave_ref": session.ref_for(entity, project),
            "turn_count": len(session.turns),
        }
        for conversation_id, session in sessions.items()
    ]
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
    client_factory: Callable[[], WeaveClient],
    entity: str,
    project: str,
) -> dict[str, Any]:
    """Resolve a mutable selection to one content-addressed hydrated cohort."""

    since, until = _selection_bounds(selection)
    query_since = since.astimezone(timezone.utc) if since is not None else None
    selected_sessions = set(selection.session_ids)

    with client_factory() as client:
        turns = client.query_turns_paginated(page_size=500, since=query_since)
        filtered: list[TurnSpan] = []
        for turn in turns:
            _require_aware_turn(turn)
            if since is not None and turn.started_at < since:
                continue
            if until is not None and turn.started_at > until:
                continue
            if turn.conversation_id not in selected_sessions:
                continue
            filtered.append(turn)
        if not filtered:
            raise ValueError("selection matched no turns")
        discovered_sessions = {turn.conversation_id for turn in filtered}
        missing_sessions = sorted(selected_sessions - discovered_sessions)
        if missing_sessions:
            raise ValueError("missing selected sessions: " + ", ".join(missing_sessions))
        ineligible = [
            conversation_id
            for conversation_id in sorted(selected_sessions)
            if not session_is_evaluable(client.query_session(conversation_id).turns)
        ]
        if ineligible:
            raise ValueError("selected sessions are not evaluable: " + ", ".join(ineligible))
        client.hydrate_turns_batch(filtered)

    return _build_cohort(filtered, entity=entity, project=project)


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
    seen_refs: set[str] = set()
    parsed_turns: list[tuple[datetime, str]] = []
    session_counts: dict[str, int] = {}
    for entry in turn_entries:
        if not isinstance(entry, dict) or set(entry) != _TURN_KEYS:
            raise RuntimeError("invalid pinned turn cohort: turn metadata")
        trace_id = entry.get("trace_id")
        weave_ref = entry.get("weave_ref")
        conversation_id = entry.get("conversation_id")
        model = entry.get("model")
        family = entry.get("model_family")
        effort_level = entry.get("effort_level")
        if not all(
            isinstance(item, str) and item
            for item in (trace_id, weave_ref, conversation_id, family)
        ):
            raise RuntimeError("invalid pinned turn cohort: turn metadata")
        if model is not None and (not isinstance(model, str) or not model):
            raise RuntimeError("invalid pinned turn cohort: model")
        if effort_level is not None and (not isinstance(effort_level, str) or not effort_level):
            raise RuntimeError("invalid pinned turn cohort: effort level")
        if family != model_family(model or ""):
            raise RuntimeError("invalid pinned turn cohort: model family")
        if trace_id in seen_traces or weave_ref in seen_refs:
            raise RuntimeError("invalid pinned turn cohort: duplicate turn identity")
        seen_traces.add(trace_id)
        seen_refs.add(weave_ref)
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
        weave_ref = entry.get("weave_ref")
        turn_count = entry.get("turn_count")
        if (
            not isinstance(conversation_id, str)
            or not conversation_id
            or not isinstance(weave_ref, str)
            or not weave_ref
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
    client_factory: Callable[[], WeaveClient],
) -> tuple[list[TurnSpan], dict[str, SessionView]]:
    """Hydrate exactly the pinned traces and reject any identity drift."""

    turn_entries, session_entries = _validated_cohort(cohort)
    trace_ids = [entry["trace_id"] for entry in turn_entries]

    with client_factory() as client:
        fetched = client.query_turns_by_trace_ids(trace_ids)
        fetched_ids = [turn.trace_id for turn in fetched]
        duplicate_ids = sorted(
            {trace_id for trace_id in fetched_ids if fetched_ids.count(trace_id) > 1}
        )
        if duplicate_ids:
            raise RuntimeError(
                "Pinned turn cohort returned duplicate trace IDs: " + ", ".join(duplicate_ids)
            )
        unexpected = sorted(set(fetched_ids) - set(trace_ids))
        if unexpected:
            raise RuntimeError(
                "Pinned turn cohort returned unexpected trace IDs: " + ", ".join(unexpected)
            )
        by_trace_id = {turn.trace_id: turn for turn in fetched}
        missing = [trace_id for trace_id in trace_ids if trace_id not in by_trace_id]
        if missing:
            raise RuntimeError(
                "Pinned turn cohort is incomplete; missing trace IDs: " + ", ".join(missing)
            )
        ordered = [by_trace_id[trace_id] for trace_id in trace_ids]
        ineligible = sorted(
            conversation_id
            for conversation_id, session in _session_views(ordered).items()
            if not session_is_evaluable(session.turns)
        )
        if ineligible:
            raise RuntimeError(
                "Pinned turn cohort contains sessions that are not evaluable: "
                + ", ".join(ineligible)
            )
        client.hydrate_turns_batch(ordered)

        entity = getattr(client, "entity", None)
        project = getattr(client, "project", None)
        if not isinstance(entity, str) or not entity or not isinstance(project, str) or not project:
            raise RuntimeError("cohort client must expose its entity and project")

        changed: list[str] = []
        for entry, turn in zip(turn_entries, ordered, strict=True):
            if (
                turn.conversation_id != entry["conversation_id"]
                or turn.ref_for(entity, project) != entry["weave_ref"]
                or turn.started_at.isoformat() != entry["started_at"]
                or turn.model != entry["model"]
                or model_family(turn.model or "") != entry["model_family"]
                or turn.effort_level != entry["effort_level"]
            ):
                changed.append(turn.trace_id)
        if changed:
            raise RuntimeError(
                "Pinned turn cohort metadata changed for trace IDs: " + ", ".join(changed)
            )

        sessions = _session_views(ordered)
        live_session_entries = [
            {
                "conversation_id": conversation_id,
                "weave_ref": session.ref_for(entity, project),
                "turn_count": len(session.turns),
            }
            for conversation_id, session in sessions.items()
        ]
        if live_session_entries != session_entries:
            raise RuntimeError("Pinned turn cohort session metadata changed")

    return ordered, sessions
