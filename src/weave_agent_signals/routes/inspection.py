"""Read-only session and evaluation-analysis routes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

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
        sessions = [
            _session_summary(conversation_id, session_turns)
            for conversation_id, session_turns in _group_sessions(filtered).items()
        ]
        sessions.sort(key=lambda item: item["last_activity"], reverse=True)
        return {
            "sessions": sessions[:limit],
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
