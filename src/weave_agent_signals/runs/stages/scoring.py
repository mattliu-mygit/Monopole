"""Deterministic scoring for one immutable evaluation-run cohort."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any

from weave_agent_signals.ces_client import CesSession
from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.run_config import EffectiveRunConfig
from weave_agent_signals.runs.events import sanitize_event
from weave_agent_signals.runs.stages import (
    record_progress,
    record_result,
    require_active,
    translate_cancellation,
)
from weave_agent_signals.runs.store import (
    Run,
    RunStatus,
    RunStore,
    RunStoreConflictError,
)
from weave_agent_signals.scorers import score_session, score_turn

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoringDependencies:
    store: RunStore
    ces_factory: Callable[[], CesSession]
    hydrate_cohort: Callable[
        [Mapping[str, Any]],
        tuple[list[TurnSpan], dict[str, SessionView]],
    ]


_require_active = partial(require_active, stage=RunStatus.SCORING)
_translate_cancellation = translate_cancellation
_record_progress = partial(record_progress, stage=RunStatus.SCORING)
_record_result = partial(record_result, stage=RunStatus.SCORING)


def _record_event(
    store: RunStore,
    run_id: str,
    phase: str,
    message: str,
    **details: object,
) -> None:
    """Persist sanitized scoring activity without affecting stage correctness."""

    try:
        store.append_run_event(
            run_id,
            sanitize_event(
                "scoring",
                phase,
                message,
                details,
                datetime.now(timezone.utc),
            ),
        )
    except Exception:
        log.warning("Run %s scoring activity could not be recorded", run_id, exc_info=True)


def _progress(
    turns: list[TurnSpan],
    *,
    scored: int,
    written: int,
    status_message: str,
    turn_details: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "total": len(turns),
        "scored": scored,
        "written": written,
        "status_message": status_message,
        "turn_details": turn_details,
    }


def _turn_detail(turn: TurnSpan, scores: list[Score]) -> dict[str, Any]:
    all_tools = turn.tool_calls + [
        tool_call for subagent in turn.subagents for tool_call in subagent.tool_calls
    ]
    user_input = turn.user_input or ""
    return {
        "trace_id": turn.trace_id[:12],
        "conversation_id": turn.conversation_id[:12],
        "model": turn.model,
        "user_input": user_input[:120] + ("..." if len(user_input) > 120 else ""),
        "tokens": turn.input_tokens + turn.output_tokens,
        "tool_count": len(all_tools),
        "tools_used": sorted({call.tool_name for call in all_tools})[:8],
        "errors": turn.tool_error_count,
        "steering": turn.steering_count,
        "denials": turn.denial_count,
        "scores": {
            score.scorer: round(float(score.value), 4)
            for score in scores
            if isinstance(score.value, (int, float))
        },
    }


Target = tuple[str, str]  # (CES target kind, target ID)


def _queue_scores(
    scores: list[Score],
    *,
    target: Target,
    config_version: str | None,
    git_branch: str | None,
    run_time: Any,
    existing: set[str],
    force: bool,
    pending_writes: dict[Target, list[Score]],
) -> None:
    for score in scores:
        score.stamp(
            config_version=config_version,
            git_branch=git_branch,
            run_time=run_time,
        )
        if score.scorer in existing and not force:
            continue
        pending_writes.setdefault(target, []).append(score)


def _write_scores(
    ces: CesSession,
    pending_writes: dict[Target, list[Score]],
    *,
    run_id: str,
) -> tuple[int, int]:
    """Write each target's Scores in one request keyed by the run, so a retried stage
    replays them instead of adding copies. Reads return the newest Score per scorer,
    so a forced rerun needs no purge."""
    if not pending_writes:
        return 0, 0

    def write(item: tuple[Target, list[Score]]) -> tuple[int, int]:
        (kind, target_id), scores = item
        try:
            ces.client.write_scores(ces.catalog, run_id, kind, target_id, scores)
        except Exception as exc:
            log.warning("Run %s score write failed: %s", run_id, exc)
            return 0, len(scores)
        return len(scores), 0

    with ThreadPoolExecutor(max_workers=min(10, len(pending_writes))) as pool:
        results = list(pool.map(write, pending_writes.items()))
    return sum(written for written, _ in results), sum(errors for _, errors in results)


def run_scoring_stage(
    run: Run,
    config: EffectiveRunConfig,
    cancel: threading.Event,
    *,
    dependencies: ScoringDependencies,
) -> None:
    """Score the exact pinned cohort and guard writes against cancellation."""

    store = dependencies.store
    _require_active(store, run.run_id, cancel)
    if run.turn_cohort is None:
        raise ValueError(f"Run {run.run_id} has no pinned turn cohort")

    turns, sessions = dependencies.hydrate_cohort(run.turn_cohort)
    _require_active(store, run.run_id, cancel)
    _record_event(
        store,
        run.run_id,
        "scoring_started",
        f"Scoring {len(turns)} turns",
        total=len(turns),
    )
    turn_details: list[dict[str, Any]] = []
    _record_progress(
        store,
        run.run_id,
        cancel,
        _progress(
            turns,
            scored=0,
            written=0,
            status_message=f"Scoring {len(turns)} turns...",
            turn_details=turn_details,
        ),
    )

    with dependencies.ces_factory() as ces:
        existing = {} if config.force else _existing_scores(ces, sessions)
        _require_active(store, run.run_id, cancel)
        pending_writes: dict[Target, list[Score]] = {}
        errors = _score_turns(
            store,
            run.run_id,
            cancel,
            turns,
            existing=existing,
            force=config.force,
            pending_writes=pending_writes,
            turn_details=turn_details,
        )
        errors += _score_sessions(
            store,
            run.run_id,
            cancel,
            sessions,
            existing=existing,
            force=config.force,
            pending_writes=pending_writes,
        )
        _require_active(store, run.run_id, cancel)
        _record_progress(
            store,
            run.run_id,
            cancel,
            _progress(
                turns,
                scored=len(turns),
                written=0,
                status_message=f"Writing {sum(map(len, pending_writes.values()))} scores...",
                turn_details=turn_details,
            ),
        )
        written, write_errors = _write_pending(store, run.run_id, cancel, ces, pending_writes)
        errors += write_errors

    _finish_scoring(
        store,
        run.run_id,
        cancel,
        turns,
        sessions_scored=len(sessions),
        written=written,
        errors=errors,
        turn_details=turn_details,
    )


def _existing_scores(
    ces: CesSession, sessions: Mapping[str, SessionView]
) -> dict[Target, set[str]]:
    """The Scorers that already scored each target, so a rerun skips them unless forced."""
    existing: dict[Target, set[str]] = {}
    for sid in sessions:
        for case_id, rows in ces.client.turn_scores(ces.catalog, sid).items():
            existing[("case", case_id)] = ces.catalog.scored(rows)
    for sid, rows in ces.client.session_scores(ces.catalog, list(sessions)).items():
        existing[("trajectory", sid)] = ces.catalog.scored(rows)
    return existing


def _score_turns(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    turns: list[TurnSpan],
    *,
    existing: dict[Target, set[str]],
    force: bool,
    pending_writes: dict[Target, list[Score]],
    turn_details: list[dict[str, Any]],
) -> int:
    """Score each turn, queue its Scores, and record its event. Returns the error count."""
    errors = 0
    for index, turn in enumerate(turns):
        _require_active(store, run_id, cancel)
        try:
            scores = score_turn(turn)
            turn_details.append(_turn_detail(turn, scores))
            _queue_scores(
                scores,
                target=("case", turn.trace_id),
                config_version=turn.config_version,
                git_branch=turn.git_branch,
                run_time=turn.started_at,
                existing=existing.get(("case", turn.trace_id), set()),
                force=force,
                pending_writes=pending_writes,
            )
        except Exception as exc:
            errors += 1
            log.warning("Run %s score error on %s: %s", run_id, turn.trace_id[:12], exc)
            _record_event(
                store,
                run_id,
                "turn_failed",
                f"Turn {index + 1} could not be scored",
                trace_id=turn.trace_id,
                conversation_id=turn.conversation_id,
                model=turn.model,
                scored=index + 1,
                total=len(turns),
                errors=errors,
            )
        else:
            detail = turn_details[-1]
            _record_event(
                store,
                run_id,
                "turn_scored",
                f"Scored turn {index + 1} of {len(turns)}",
                trace_id=turn.trace_id,
                conversation_id=turn.conversation_id,
                model=turn.model,
                scored=index + 1,
                total=len(turns),
                errors=errors,
                tokens=detail["tokens"],
                tool_count=detail["tool_count"],
                steering=detail["steering"],
                denials=detail["denials"],
            )
        _record_progress(
            store,
            run_id,
            cancel,
            _progress(
                turns,
                scored=index + 1,
                written=0,
                status_message=f"Scored turn {index + 1} of {len(turns)}",
                turn_details=turn_details,
            ),
        )
    return errors


def _score_sessions(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    sessions: Mapping[str, SessionView],
    *,
    existing: dict[Target, set[str]],
    force: bool,
    pending_writes: dict[Target, list[Score]],
) -> int:
    """Score each session and queue its Scores. Returns the error count."""
    errors = 0
    for session in sessions.values():
        _require_active(store, run_id, cancel)
        try:
            _queue_scores(
                score_session(session),
                target=("trajectory", session.conversation_id),
                config_version=session.config_version,
                git_branch=session.git_branch,
                run_time=session.turns[0].started_at if session.turns else None,
                existing=existing.get(("trajectory", session.conversation_id), set()),
                force=force,
                pending_writes=pending_writes,
            )
        except Exception as exc:
            errors += 1
            log.warning(
                "Run %s session score error on %s: %s", run_id, session.conversation_id[:12], exc
            )
    return errors


def _write_pending(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    ces: CesSession,
    pending_writes: dict[Target, list[Score]],
) -> tuple[int, int]:
    """Write the queued Scores inside the cancellation barrier. Returns (written, errors)."""
    if not pending_writes:
        return 0, 0
    _require_active(store, run_id, cancel)
    try:
        with store.external_write_barrier(run_id, RunStatus.SCORING):
            return _write_scores(ces, pending_writes, run_id=run_id)
    except RunStoreConflictError as exc:
        _translate_cancellation(store, run_id, cancel, exc)
        raise


def _finish_scoring(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    turns: list[TurnSpan],
    *,
    sessions_scored: int,
    written: int,
    errors: int,
    turn_details: list[dict[str, Any]],
) -> None:
    """Record the final progress, result, and event. Raises when any Score failed."""
    _require_active(store, run_id, cancel)
    error_label = "error" if errors == 1 else "errors"
    status_message = (
        f"Scoring failed: {errors} {error_label}; {written} scores written."
        if errors
        else f"Scoring complete: {written} scores written."
    )
    _record_progress(
        store,
        run_id,
        cancel,
        _progress(
            turns,
            scored=len(turns),
            written=written,
            status_message=status_message,
            turn_details=turn_details,
        ),
    )
    _record_result(
        store,
        run_id,
        cancel,
        {
            "turns_scored": len(turns),
            "sessions_scored": sessions_scored,
            "scores_written": written,
            "errors": errors,
            "turn_details": turn_details,
            "status_message": status_message,
        },
    )
    _record_event(
        store,
        run_id,
        "scoring_complete" if not errors else "scoring_failed",
        status_message,
        total=len(turns),
        scored=len(turns),
        written=written,
        errors=errors,
    )
    if errors:
        raise RuntimeError(f"Scoring completed with {errors} {error_label}")
