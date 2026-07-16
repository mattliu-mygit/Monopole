"""Read-only session and evaluation-analysis routes."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, HTTPException, Query

from weave_agent_signals.client import WeaveClient
from weave_agent_signals.models import TurnSpan
from weave_agent_signals.patterns import (
    ab_leaderboard,
    aggregate_scores,
    coaching_digest,
    detect_regressions,
)
from weave_agent_signals.routes._time import DateFilterError, parse_selection_bounds

_SYNTHETIC_SESSION_PREFIXES = (
    "## Scoring criteria",
    "You are an expert optimization assistant.",
)
_SIGNAL_SCORER_RE = re.compile(
    r"^agent-signal-(?P<signal>.+)-(?P<version>v[1-9][0-9]*)-scorer(?::.+)?$"
)
_SIGNAL_ANCHORS = {0.0, 0.25, 0.5, 0.75, 1.0}
_SIGNAL_THRESHOLD = 0.5


def _signal_identity(runnable_ref: object) -> tuple[str, str] | None:
    if not isinstance(runnable_ref, str):
        return None
    parts = [unquote(part) for part in urlparse(runnable_ref).path.split("/") if part]
    if not parts:
        return None
    match = _SIGNAL_SCORER_RE.fullmatch(parts[-1])
    if match is None:
        return None
    return match.group("signal"), match.group("version")


def _signal_rating(row: dict) -> tuple[float, str, datetime]:
    payload = row.get("payload")
    output = payload.get("output") if isinstance(payload, dict) else None
    if not isinstance(output, dict):
        raise ValueError("eligible Signal feedback has no output object")
    rating_value = output.get("rating")
    value_value = output.get("value")
    if rating_value is not None and value_value is not None and rating_value != value_value:
        raise ValueError("eligible Signal feedback has conflicting rating aliases")
    rating = rating_value if rating_value is not None else value_value
    if isinstance(rating, bool) or not isinstance(rating, (int, float)):
        raise ValueError("eligible Signal feedback has a non-numeric rating")
    rating = float(rating)
    if rating not in _SIGNAL_ANCHORS:
        raise ValueError("eligible Signal feedback has an invalid rating anchor")
    scorer_ratings = row.get("scorer_ratings")
    if not isinstance(scorer_ratings, dict) or scorer_ratings.get("_rating_") != rating:
        raise ValueError("eligible Signal feedback has inconsistent typed rating")
    reason = output.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("eligible Signal feedback has no grounded reason")
    created_at = row.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("eligible Signal feedback has invalid score time") from error
    if not isinstance(created_at, datetime) or created_at.tzinfo is None:
        raise ValueError("eligible Signal feedback has invalid score time")
    return rating, reason.strip(), created_at.astimezone(timezone.utc)


def _signal_evidence(
    turns: list[TurnSpan],
    feedback_by_ref: dict[str, list[dict]],
    *,
    entity: str,
    project: str,
) -> list[dict]:
    selected: dict[tuple[str, str, str], tuple[datetime, str, dict]] = {}
    for turn in turns:
        turn_ref = turn.ref_for(entity, project)
        for row in feedback_by_ref.get(turn_ref, []):
            if row.get("feedback_type") != "wandb.agent_monitor":
                continue
            identity = _signal_identity(row.get("runnable_ref"))
            if identity is None:
                continue
            typed_ratings = row.get("scorer_ratings")
            if not isinstance(typed_ratings, dict) or "_rating_" not in typed_ratings:
                continue
            rating, reason, created_at = _signal_rating(row)
            feedback_id = row.get("id")
            if not isinstance(feedback_id, str) or not feedback_id:
                raise ValueError("eligible Signal feedback has no id")
            signal, version = identity
            evidence = {
                "signal": signal,
                "version": version,
                "rating": rating,
                "reason": reason,
                "turn_id": turn.trace_id,
                "turn_started_at": turn.started_at.isoformat(),
            }
            key = (signal, version, turn.trace_id)
            existing = selected.get(key)
            if existing is None or (created_at, feedback_id) > existing[:2]:
                selected[key] = (created_at, feedback_id, evidence)
    return sorted(
        (item[2] for item in selected.values() if item[2]["rating"] <= _SIGNAL_THRESHOLD),
        key=lambda item: (
            item["turn_started_at"],
            item["signal"],
            item["version"],
            item["turn_id"],
        ),
    )


def _turn_json(turn: TurnSpan, *, include_children: bool = False) -> dict:
    value = {
        "trace_id": turn.trace_id,
        "conversation_id": turn.conversation_id,
        "started_at": turn.started_at.isoformat(),
        "ended_at": turn.ended_at.isoformat() if turn.ended_at else None,
        "model": turn.model,
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "cache_read_tokens": turn.cache_read_tokens,
        "status_code": turn.status_code,
        "config_version": turn.config_version,
        "git_branch": turn.git_branch,
        "effort_level": turn.effort_level,
        "session_id": turn.session_id,
        "steering_count": turn.steering_count,
        "denial_count": turn.denial_count,
        "tool_error_count": turn.tool_error_count,
        "tool_call_count": len(turn.tool_calls),
        "chat_span_count": len(turn.chat_spans),
        "subagent_count": len(turn.subagents),
        "user_input": turn.user_input,
    }
    if include_children:
        value.update(
            {
                "tool_calls": [
                    {
                        "span_id": item.span_id,
                        "tool_name": item.tool_name,
                        "arguments": item.arguments or "",
                        "result": item.result or "",
                        "status_code": item.status_code,
                        "started_at": (item.started_at.isoformat() if item.started_at else None),
                        "ended_at": item.ended_at.isoformat() if item.ended_at else None,
                    }
                    for item in turn.tool_calls
                ],
                "chat_spans": [
                    {
                        "span_id": item.span_id,
                        "model": item.model,
                        "input_tokens": item.input_tokens,
                        "output_tokens": item.output_tokens,
                        "cache_read_tokens": item.cache_read_tokens,
                        "finish_reason": item.finish_reason,
                    }
                    for item in turn.chat_spans
                ],
                "subagents": [
                    {
                        "span_id": item.span_id,
                        "agent_type": item.agent_type,
                        "tool_call_count": len(item.tool_calls),
                    }
                    for item in turn.subagents
                ],
            }
        )
    return value


def _session_summary(conversation_id: str, turns: list[TurnSpan]) -> dict:
    ordered = sorted(turns, key=lambda turn: (turn.started_at, turn.trace_id))
    first, last = ordered[0], ordered[-1]
    models = sorted({turn.model for turn in ordered if turn.model})
    return {
        "conversation_id": conversation_id,
        "session_id": first.session_id,
        "turn_count": len(ordered),
        "started_at": first.started_at.isoformat(),
        "ended_at": last.ended_at.isoformat() if last.ended_at else None,
        "last_activity": last.started_at.isoformat(),
        "model": models[0] if len(models) == 1 else ", ".join(models) or None,
        "effort_level": first.effort_level,
        "config_version": first.config_version,
        "git_branch": first.git_branch,
        "total_tokens": sum(turn.input_tokens + turn.output_tokens for turn in ordered),
        "total_tool_calls": sum(len(turn.tool_calls) for turn in ordered),
        "input_preview": first.user_input,
    }


def _group_sessions(turns: list[TurnSpan]) -> dict[str, list[TurnSpan]]:
    grouped: dict[str, list[TurnSpan]] = {}
    for turn in turns:
        grouped.setdefault(turn.conversation_id, []).append(turn)
    sessions: dict[str, list[TurnSpan]] = {}
    for conversation_id, session_turns in grouped.items():
        first = min(session_turns, key=lambda turn: turn.started_at)
        synthetic = isinstance(first.user_input, str) and first.user_input.startswith(
            _SYNTHETIC_SESSION_PREFIXES
        )
        if not synthetic:
            sessions[conversation_id] = session_turns
    return sessions


def _date_error(error: DateFilterError) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"code": error.code, "message": str(error)},
    )


def create_inspection_router(
    client_factory: Callable[[], WeaveClient],
) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["inspection"])

    @router.get("/sessions")
    def get_sessions(
        since: str | None = None,
        until: str | None = None,
        timezone_name: str | None = Query(default=None, alias="timezone"),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        try:
            since_value, until_value = parse_selection_bounds(
                since,
                until,
                timezone_name,
            )
        except DateFilterError as error:
            raise _date_error(error) from error
        since_value = since_value or (datetime.now(timezone.utc) - timedelta(hours=168))
        query_since = since_value.astimezone(timezone.utc)
        with client_factory() as client:
            turns = client.query_turns_paginated(
                page_size=500,
                since=query_since,
                include_details=True,
            )
        filtered = [turn for turn in turns if turn.started_at >= since_value]
        if until_value is not None:
            filtered = [turn for turn in filtered if turn.started_at <= until_value]
        grouped = _group_sessions(filtered)
        sessions = [
            (_session_summary(conversation_id, session_turns), session_turns)
            for conversation_id, session_turns in grouped.items()
        ]
        sessions.sort(key=lambda item: item[0]["last_activity"], reverse=True)
        visible = sessions[:limit]
        with client_factory() as feedback_client:
            refs = [
                turn.ref_for(feedback_client.entity, feedback_client.project)
                for _summary, session_turns in visible
                for turn in session_turns
            ]
            feedback_by_ref = feedback_client.query_all_feedback_batch(refs)
            for summary, session_turns in visible:
                summary["signal_evidence"] = _signal_evidence(
                    session_turns,
                    feedback_by_ref,
                    entity=feedback_client.entity,
                    project=feedback_client.project,
                )
        return {
            "sessions": [summary for summary, _turns in visible],
            "total": len(sessions),
            "truncated": len(sessions) > limit,
            "limit": limit,
        }

    @router.get("/sessions/{conversation_id:path}")
    def get_session(conversation_id: str) -> dict:
        try:
            with client_factory() as client:
                session = client.query_session(conversation_id)
                if not session.turns:
                    raise ValueError("session has no turns")
                client.hydrate_turns_batch(session.turns)
                session_ref = session.ref_for(client.entity, client.project)
                turn_refs = {
                    turn.trace_id: turn.ref_for(client.entity, client.project)
                    for turn in session.turns
                }
                feedback_by_ref = client.query_all_feedback_batch(
                    [session_ref, *turn_refs.values()]
                )
                session_feedback = feedback_by_ref[session_ref]
                turn_feedback = {
                    trace_id: feedback_by_ref[ref]
                    for trace_id, ref in turn_refs.items()
                    if feedback_by_ref[ref]
                }
        except ValueError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "session_not_found", "message": str(error)},
            ) from error
        return {
            "conversation_id": session.conversation_id,
            "config_version": session.config_version,
            "git_branch": session.git_branch,
            "total_tokens": session.total_tokens,
            "turn_count": len(session.turns),
            "turns": [_turn_json(turn, include_children=True) for turn in session.turns],
            "session_feedback": session_feedback,
            "turn_feedback": turn_feedback,
        }

    @router.get("/analyze")
    def get_analysis(limit: int = Query(default=1000, ge=1, le=5000)) -> dict:
        with client_factory() as client:
            feedback = client.query_project_feedback(limit=limit)
        if not feedback:
            return {
                "summary": [],
                "ab_leaderboard": [],
                "trends": [],
                "coaching_markdown": "",
            }

        summaries = aggregate_scores(feedback)
        summary = [
            {
                "scorer": scorer,
                "count": value.count,
                "mean": round(value.mean, 4),
                "ci": [round(value.ci[0], 4), round(value.ci[1], 4)],
                "binary": value.binary,
                "pass_rate": (round(value.pass_rate, 4) if value.pass_rate is not None else None),
                "tag_counts": dict(value.tag_counts),
                "confident": value.confident,
            }
            for scorer, value in sorted(summaries.items())
        ]
        leaderboard = [
            {
                "config_version": item.config_version,
                "evaluated_target_count": item.evaluated_target_count,
                "scores": {
                    scorer: {"mean": round(value.mean, 4), "count": value.count}
                    for scorer, value in item.scores.items()
                },
            }
            for item in ab_leaderboard(feedback)
        ]
        trends = [
            {
                **item,
                "older_mean": round(item["older_mean"], 4),
                "recent_mean": round(item["recent_mean"], 4),
                "delta": round(item["delta"], 4),
            }
            for item in detect_regressions(feedback)
        ]
        return {
            "summary": summary,
            "ab_leaderboard": leaderboard,
            "trends": trends,
            "coaching_markdown": coaching_digest(feedback),
        }

    return router
