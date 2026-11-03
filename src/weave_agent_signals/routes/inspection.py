"""Read-only session and evaluation-analysis routes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import get_args

import httpx
from fastapi import APIRouter, HTTPException, Query

from weave_agent_signals.ces_client import CesSession
from weave_agent_signals.judges.tokens import TokenCounterName, count_tokens
from weave_agent_signals.judges.windowing import render_raw_turn
from weave_agent_signals.models import FEEDBACK_PREFIX, SessionView, TurnSpan, session_is_evaluable
from weave_agent_signals.patterns import (
    ab_leaderboard,
    aggregate_scores,
    coaching_digest,
    detect_regressions,
)
from weave_agent_signals.routes._time import DateFilterError, parse_selection_bounds
from weave_agent_signals.routes.models import (
    AnalysisResponse,
    SessionDetailResponse,
    SessionListResponse,
)

_SIGNAL_THRESHOLD = 0.5


def _by_target(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["target_id"], []).append(record)
    return grouped


def _signal_evidence(
    turns: list[TurnSpan], feedback_by_target: dict[str, list[dict]]
) -> list[dict]:
    """Turn Scores at or below the threshold that carry a reason, in turn order."""
    evidence = []
    for turn in turns:
        for record in feedback_by_target.get(turn.trace_id, []):
            payload = record["payload"]
            rating, reason = payload.get("rating"), payload.get("reason")
            if not isinstance(rating, (int, float)) or rating > _SIGNAL_THRESHOLD or not reason:
                continue
            evidence.append(
                {
                    "signal": record["feedback_type"].removeprefix(FEEDBACK_PREFIX),
                    "version": payload.get("scorer_version") or "v1",
                    "rating": float(rating),
                    "reason": reason,
                    "turn_id": turn.trace_id,
                    "turn_started_at": turn.started_at.isoformat(),
                }
            )
    return sorted(
        evidence,
        key=lambda item: (
            item["turn_started_at"],
            item["signal"],
            item["version"],
            item["turn_id"],
        ),
    )


def _read_session(ces: CesSession, trajectory_id: str) -> SessionView | None:
    try:
        return ces.client.read_session(trajectory_id)
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 404:
            return None
        raise


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


def _judging_token_estimates(turns: list[TurnSpan]) -> dict:
    rendered_turns = [
        render_raw_turn(turn, position) for position, turn in enumerate(turns, start=1)
    ]
    judging_token_estimates = {}
    for counter in get_args(TokenCounterName):
        counts = [count_tokens(rendered, counter) for rendered in rendered_turns]
        judging_token_estimates[counter] = {
            "total_tokens": sum(counts),
            "largest_turn_tokens": max(counts),
            "turn_tokens": counts,
        }
    return judging_token_estimates


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


def _date_error(error: DateFilterError) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"code": error.code, "message": str(error)},
    )


def create_inspection_router(ces_factory: Callable[[], CesSession]) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["inspection"])

    @router.get("/sessions")
    def get_sessions(
        since: str | None = None,
        until: str | None = None,
        timezone_name: str | None = Query(default=None, alias="timezone"),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> SessionListResponse:
        try:
            since_value, until_value = parse_selection_bounds(
                since,
                until,
                timezone_name,
            )
        except DateFilterError as error:
            raise _date_error(error) from error
        since_value = since_value or (datetime.now(timezone.utc) - timedelta(hours=168))
        with ces_factory() as ces:
            # Newest activity first, so reading stops once `limit` sessions are found.
            session_ids = ces.client.find_sessions(
                ces.catalog, since=since_value, until=until_value
            )
            visible: list[tuple[dict, list[TurnSpan]]] = []
            for sid in session_ids:
                if len(visible) == limit:
                    break
                session = _read_session(ces, sid)
                if session is None or not session_is_evaluable(session.turns):
                    continue
                in_window = [
                    turn
                    for turn in session.turns
                    if turn.started_at >= since_value
                    and (until_value is None or turn.started_at <= until_value)
                ]
                if in_window:
                    visible.append((_session_summary(sid, in_window), in_window))
            feedback = _by_target(
                ces.client.cohort_feedback(ces.catalog, [s["conversation_id"] for s, _ in visible])
                if visible
                else []
            )
        for summary, turns in visible:
            summary["signal_evidence"] = _signal_evidence(turns, feedback)
        return SessionListResponse.model_validate(
            {
                "sessions": [summary for summary, _turns in visible],
                "total": len(session_ids),
                "truncated": len(session_ids) > len(visible),
                "limit": limit,
            }
        )

    @router.get("/sessions/{conversation_id:path}")
    def get_session(conversation_id: str) -> SessionDetailResponse:
        with ces_factory() as ces:
            session = _read_session(ces, conversation_id)
            if session is None or not session_is_evaluable(session.turns):
                message = (
                    "session not found"
                    if session is None
                    else "session is not an evaluable agent session"
                )
                raise HTTPException(
                    status_code=404,
                    detail={"code": "session_not_found", "message": message},
                )
            feedback = _by_target(ces.client.cohort_feedback(ces.catalog, [conversation_id]))
        return SessionDetailResponse.model_validate(
            {
                "conversation_id": session.conversation_id,
                "config_version": session.config_version,
                "git_branch": session.git_branch,
                "total_tokens": session.total_tokens,
                "turn_count": len(session.turns),
                "judging_token_estimates": _judging_token_estimates(session.turns),
                "turns": [_turn_json(turn, include_children=True) for turn in session.turns],
                "session_feedback": feedback.get(conversation_id, []),
                "turn_feedback": {
                    turn.trace_id: feedback[turn.trace_id]
                    for turn in session.turns
                    if feedback.get(turn.trace_id)
                },
                "signal_evidence": _signal_evidence(session.turns, feedback),
            }
        )

    @router.get("/analyze")
    def get_analysis(limit: int = Query(default=1000, ge=1, le=5000)) -> AnalysisResponse:
        with ces_factory() as ces:
            feedback = ces.client.read_all_scores(ces.catalog)[:limit]
        if not feedback:
            return AnalysisResponse.model_validate(
                {
                    "summary": [],
                    "ab_leaderboard": [],
                    "trends": [],
                    "coaching_markdown": "",
                }
            )

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
        return AnalysisResponse.model_validate(
            {
                "summary": summary,
                "ab_leaderboard": leaderboard,
                "trends": trends,
                "coaching_markdown": coaching_digest(feedback),
            }
        )

    return router
