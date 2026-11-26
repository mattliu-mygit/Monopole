"""Read-only session and evaluation-analysis routes."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Any, get_args

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from weave_agent_signals import ces_queries
from weave_agent_signals.ces_client import CesSession, session_or_none
from weave_agent_signals.ces_sessions import turns_from_rows
from weave_agent_signals.judges.tokens import TokenCounterName, count_tokens
from weave_agent_signals.judges.windowing import render_raw_turn
from weave_agent_signals.models import FEEDBACK_PREFIX, TurnSpan, session_is_evaluable
from weave_agent_signals.patterns import (
    coaching_digest,
    stat_leaderboard,
    stat_summaries,
    trends,
)
from weave_agent_signals.routes._time import DateFilterError, parse_selection_bounds
from weave_agent_signals.routes.models import (
    AnalysisResponse,
    SessionDetailResponse,
    SessionEstimatesResponse,
    SessionListResponse,
    TurnSubagentsResponse,
)

_SIGNAL_THRESHOLD = 0.5


SESSION_SCAN_FACTOR = 3
# The most sessions one estimates request may name: the list's own limit.
ESTIMATE_REQUEST_LIMIT = 100
# How many sessions' judging estimates the API keeps: two of the largest requests, so one
# request never evicts its own sessions.
ESTIMATE_CACHE_SIZE = 2 * ESTIMATE_REQUEST_LIMIT
# How many distinct list queries (window and limit) keep their last list.
LISTING_CACHE_SIZE = 32
# An estimates request's turn rows: each row's Case ID is its turn's identity, so the request
# asks CES only for the session each row belongs to.
_TURN_ID_COLUMNS = {"input": ["session_trajectory_id"]}
_LOG = logging.getLogger(__name__)


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


def _subagent_json(item, *, read: bool = True) -> dict:
    return {
        "span_id": item.span_id,
        "agent_type": item.agent_type,
        "tool_call_count": len(item.tool_calls) if read else None,
    }


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
                "assistant_output": turn.assistant_output,
                "tool_calls": [
                    {
                        "span_id": item.span_id,
                        "tool_name": item.tool_name,
                        "arguments": item.arguments or "",
                        "result": item.result or "",
                        "status_code": item.status_code,
                        "started_at": (item.started_at.isoformat() if item.started_at else None),
                        "ended_at": item.ended_at.isoformat() if item.ended_at else None,
                        "exit_code": item.exit_code,
                        "permission_decision": item.permission_decision,
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
                # The session view doesn't read subagents; a turn's are read on request.
                "subagents": [_subagent_json(item, read=False) for item in turn.subagents],
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
        "input_preview": first.user_input,
    }


class SessionEstimatesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_ids: list[StrictStr] = Field(min_length=1, max_length=ESTIMATE_REQUEST_LIMIT)


def _date_error(error: DateFilterError) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"code": error.code, "message": str(error)},
    )


class _LruCache:
    """A dict that keeps only its `size` most recently used entries, safe across threads."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._entries: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key]

    def update(self, entries: dict[Any, Any]) -> None:
        with self._lock:
            for key, value in entries.items():
                self._entries[key] = value
                self._entries.move_to_end(key)
            while len(self._entries) > self._size:
                self._entries.popitem(last=False)


def create_inspection_router(ces_factory: Callable[[], CesSession]) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["inspection"])
    # Query IDs and whether CES has a stddev aggregation, learned once per process.
    query_state = ces_queries.QueryState()
    # Estimates read a whole session, subagents included. Stored turns never change, so each
    # session's estimate, kept with its turn IDs, is reused until its turn list changes.
    estimates = _LruCache(ESTIMATE_CACHE_SIZE)
    # Reading turn rows from CES takes 10 s or more per list load, mostly in 503 retries and
    # small pages, and CES can't ask for only newer turns. So the route serves each query's
    # last list and rebuilds it in the background: a just-finished turn appears one load
    # later. Runs never read this cache.
    listings = _LruCache(LISTING_CACHE_SIZE)
    rebuilding: set[tuple] = set()
    rebuild_lock = threading.Lock()

    def build_listing(since: datetime | None, until: datetime | None, limit: int) -> dict:
        with ces_factory() as ces:
            # Newest activity first, so reading stops once `limit` sessions are found. Sessions
            # that can't be evaluated are skipped, and the scan reads at most SESSION_SCAN_FACTOR
            # times `limit` of them, so one request's CES reads stay bounded.
            scan = limit * SESSION_SCAN_FACTOR
            scanned, complete = ces.client.recent_turn_rows(
                ces.catalog, since=since, until=until, limit=scan
            )
            session_ids = list(scanned)
            visible: list[tuple[dict, list[TurnSpan]]] = []
            visible_rows: dict[str, list[dict]] = {}
            read = 0
            # Turn rows only: a session's steps and subagents can take hundreds of reads. The
            # rows also carry the turn Scores, so they're read once.
            while read < len(session_ids) and len(visible) < limit:
                batch = session_ids[read : read + limit - len(visible)]
                read += len(batch)
                # A scan that reached the oldest turn already read every turn row of its sessions.
                batch_rows = (
                    {sid: scanned[sid] for sid in batch}
                    if complete
                    else ces.client.session_turn_rows(ces.catalog, batch)
                )
                for sid, rows in batch_rows.items():
                    turns = turns_from_rows(sid, rows)
                    if not session_is_evaluable(turns):
                        continue
                    in_window = [
                        turn
                        for turn in turns
                        if (since is None or turn.started_at >= since)
                        and (until is None or turn.started_at <= until)
                    ]
                    if in_window:
                        visible.append((_session_summary(sid, in_window), in_window))
                        visible_rows[sid] = rows
            # Signal evidence comes from turn Scores only, so session Scores aren't read.
            feedback = _by_target(ces.client.turn_row_scores(visible_rows))
        for summary, turns in visible:
            summary["signal_evidence"] = _signal_evidence(turns, feedback)
        return {
            "sessions": [summary for summary, _turns in visible],
            "total": len(session_ids),
            "truncated": read < len(session_ids) or len(session_ids) == scan,
            "limit": limit,
        }

    def rebuild(key: tuple) -> None:
        try:
            listings.update({key: build_listing(*key)})
        except Exception:
            _LOG.warning("background session-list rebuild failed", exc_info=True)
        finally:
            with rebuild_lock:
                rebuilding.discard(key)

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
        key = (since_value, until_value, limit)
        listing = listings.get(key)
        if listing is None:
            listing = build_listing(*key)
            listings.update({key: listing})
        else:
            with rebuild_lock:
                stale = key not in rebuilding
                rebuilding.add(key)
            if stale:
                threading.Thread(target=rebuild, args=(key,), daemon=True).start()
        return SessionListResponse.model_validate(listing)

    @router.post("/sessions/estimates")
    def get_session_estimates(request: SessionEstimatesRequest) -> SessionEstimatesResponse:
        # Judging token estimates for the named sessions, by session; unknown ones are left out.
        # Run setup asks only for the sessions the user selects, since each reads a whole session.
        found: dict[str, dict] = {}
        with ces_factory() as ces:
            # One small search finds each session's current turns, so a kept estimate is
            # reused only while they are unchanged, and an unknown session costs no read.
            rows = ces.client.session_turn_rows(
                ces.catalog, list(dict.fromkeys(request.session_ids)), columns=_TURN_ID_COLUMNS
            )
            for sid, session_rows in rows.items():
                turn_ids = tuple(sorted(row["case_id"] for row in session_rows))
                if not turn_ids:
                    continue
                known = estimates.get(sid)
                if known is None or known[0] != turn_ids:
                    # The judge renders each subagent's type and tool-call count, so the
                    # estimate reads subagents too.
                    session = session_or_none(ces.client, sid)
                    if session is None or not session.turns:
                        continue
                    known = (turn_ids, _judging_token_estimates(session.turns))
                    estimates.update({sid: known})
                found[sid] = known[1]
        return SessionEstimatesResponse.model_validate(found)

    @router.get("/sessions/{conversation_id}")
    def get_session(conversation_id: str) -> SessionDetailResponse:
        with ces_factory() as ces:
            session = session_or_none(ces.client, conversation_id, subagents=False)
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
            feedback = _by_target(ces.client.cohort_scores(ces.catalog, [conversation_id]))
        return SessionDetailResponse.model_validate(
            {
                "conversation_id": session.conversation_id,
                "config_version": session.config_version,
                "git_branch": session.git_branch,
                "total_tokens": session.total_tokens,
                "turn_count": len(session.turns),
                "turns": [_turn_json(turn, include_children=True) for turn in session.turns],
                "session_scores": feedback.get(conversation_id, []),
                "turn_scores": {
                    turn.trace_id: feedback[turn.trace_id]
                    for turn in session.turns
                    if feedback.get(turn.trace_id)
                },
                "signal_evidence": _signal_evidence(session.turns, feedback),
            }
        )

    @router.get("/sessions/{conversation_id}/turns/{trace_id}/subagents")
    def get_turn_subagents(conversation_id: str, trace_id: str) -> TurnSubagentsResponse:
        with ces_factory() as ces:
            subagents = ces.client.turn_subagents(conversation_id, trace_id)
        if subagents is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "turn_not_found", "message": "turn not found"},
            )
        return TurnSubagentsResponse.model_validate(
            {"subagents": [_subagent_json(item) for item in subagents]}
        )

    @router.get("/analyze")
    def get_analysis(limit: int = Query(default=1000, ge=1, le=5000)) -> AnalysisResponse:
        # Summaries, the A/B leaderboard, and session and turn trends cover every Score, from
        # CES queries. Coaching examples and tag counts cover the Scores on the newest `limit`
        # turns and their sessions.
        with ces_factory() as ces:
            # Read per request: a judge run can add a panel Scorer at any time.
            scorers = ces_queries.session_scorers(ces.client, ces.catalog)
            stats = ces_queries.config_stats(ces.client, ces.catalog, query_state, scorers)
            days = ces_queries.session_day_stats(ces.client, ces.catalog, query_state, scorers)
            try:
                turn_days = ces_queries.turn_day_stats(ces.client, ces.catalog, query_state)
            except ces_queries.DatasetWithoutDay as error:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "dataset_without_day", "message": str(error)},
                ) from error
            stats = replace(stats, days=days, turn_days=turn_days)
            feedback = ces.client.read_all_scores(ces.catalog, limit=limit, outcome_details=False)
        if not stats.by_config and not feedback:
            return AnalysisResponse.model_validate(
                {
                    "summary": [],
                    "ab_leaderboard": [],
                    "trends": [],
                    "coaching_markdown": "",
                }
            )

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
            for scorer, value in sorted(stat_summaries(stats, feedback).items())
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
            for item in stat_leaderboard(stats)
        ]
        trend = [
            {
                **item,
                "older_mean": round(item["older_mean"], 4),
                "recent_mean": round(item["recent_mean"], 4),
                "delta": round(item["delta"], 4),
            }
            for item in trends(feedback, stats)
        ]
        return AnalysisResponse.model_validate(
            {
                "summary": summary,
                "ab_leaderboard": leaderboard,
                "trends": trend,
                "coaching_markdown": coaching_digest(feedback, stats),
            }
        )

    return router
