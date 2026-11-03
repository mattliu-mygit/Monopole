"""Deterministic scoring for one immutable evaluation-run cohort."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from weave_agent_signals.ces_client import CesSession
from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.run_config import EffectiveRunConfig
from weave_agent_signals.runs.events import sanitize_event
from weave_agent_signals.runs.stages import StageCancelled
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


def _require_active(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
) -> Run:
    if cancel.is_set():
        raise StageCancelled()
    current = store.get(run_id)
    if current is None:
        raise ValueError(f"Run {run_id} not found")
    if current.status is RunStatus.CANCELLED:
        raise StageCancelled()
    if current.status is not RunStatus.SCORING:
        raise RunStoreConflictError(
            f"Run {run_id} is no longer scoring; current status is {current.status.value}"
        )
    return current


def _translate_cancellation(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    conflict: RunStoreConflictError,
) -> None:
    current = store.get(run_id)
    if cancel.is_set() or (current is not None and current.status is RunStatus.CANCELLED):
        raise StageCancelled() from conflict


def _record_progress(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    progress: Mapping[str, Any],
) -> None:
    try:
        store.record_stage_progress(
            run_id,
            stage=RunStatus.SCORING,
            progress=progress,
        )
    except RunStoreConflictError as exc:
        _translate_cancellation(store, run_id, cancel, exc)
        raise


def _record_result(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    result: Mapping[str, Any],
) -> None:
    try:
        store.record_stage_result(
            run_id,
            stage=RunStatus.SCORING,
            result=result,
        )
    except RunStoreConflictError as exc:
        _translate_cancellation(store, run_id, cancel, exc)
        raise


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

    _require_active(dependencies.store, run.run_id, cancel)
    if run.turn_cohort is None:
        raise ValueError(f"Run {run.run_id} has no pinned turn cohort")

    turns, sessions = dependencies.hydrate_cohort(run.turn_cohort)
    _require_active(dependencies.store, run.run_id, cancel)
    _record_event(
        dependencies.store,
        run.run_id,
        "scoring_started",
        f"Scoring {len(turns)} turns",
        total=len(turns),
    )
    turn_details: list[dict[str, Any]] = []
    _record_progress(
        dependencies.store,
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
        existing: dict[Target, set[str]] = {}
        if not config.force:
            for sid in sessions:
                for case_id, rows in ces.client.turn_scores(ces.catalog, sid).items():
                    existing[("case", case_id)] = ces.catalog.scored(rows)
            for sid, rows in ces.client.session_scores(ces.catalog, list(sessions)).items():
                existing[("trajectory", sid)] = ces.catalog.scored(rows)
        _require_active(dependencies.store, run.run_id, cancel)

        errors = 0
        pending_writes: dict[Target, list[Score]] = {}
        for index, turn in enumerate(turns):
            _require_active(dependencies.store, run.run_id, cancel)
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
                    force=config.force,
                    pending_writes=pending_writes,
                )
            except Exception as exc:
                errors += 1
                log.warning(
                    "Run %s score error on %s: %s",
                    run.run_id,
                    turn.trace_id[:12],
                    exc,
                )
                _record_event(
                    dependencies.store,
                    run.run_id,
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
                    dependencies.store,
                    run.run_id,
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
                dependencies.store,
                run.run_id,
                cancel,
                _progress(
                    turns,
                    scored=index + 1,
                    written=0,
                    status_message=(f"Scored turn {index + 1} of {len(turns)}"),
                    turn_details=turn_details,
                ),
            )

        for session in sessions.values():
            _require_active(dependencies.store, run.run_id, cancel)
            try:
                scores = score_session(session)
                _queue_scores(
                    scores,
                    target=("trajectory", session.conversation_id),
                    config_version=session.config_version,
                    git_branch=session.git_branch,
                    run_time=session.turns[0].started_at if session.turns else None,
                    existing=existing.get(("trajectory", session.conversation_id), set()),
                    force=config.force,
                    pending_writes=pending_writes,
                )
            except Exception as exc:
                errors += 1
                log.warning(
                    "Run %s session score error on %s: %s",
                    run.run_id,
                    session.conversation_id[:12],
                    exc,
                )

        _require_active(dependencies.store, run.run_id, cancel)
        _record_progress(
            dependencies.store,
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

        written = 0
        if pending_writes:
            _require_active(dependencies.store, run.run_id, cancel)
            try:
                with dependencies.store.external_write_barrier(
                    run.run_id,
                    RunStatus.SCORING,
                ):
                    written, write_errors = _write_scores(
                        ces,
                        pending_writes,
                        run_id=run.run_id,
                    )
                    errors += write_errors
            except RunStoreConflictError as exc:
                _translate_cancellation(
                    dependencies.store,
                    run.run_id,
                    cancel,
                    exc,
                )
                raise

    _require_active(dependencies.store, run.run_id, cancel)
    error_label = "error" if errors == 1 else "errors"
    status_message = (
        f"Scoring failed: {errors} {error_label}; {written} scores written."
        if errors
        else f"Scoring complete: {written} scores written."
    )
    final_progress = _progress(
        turns,
        scored=len(turns),
        written=written,
        status_message=status_message,
        turn_details=turn_details,
    )
    _record_progress(
        dependencies.store,
        run.run_id,
        cancel,
        final_progress,
    )
    _record_result(
        dependencies.store,
        run.run_id,
        cancel,
        {
            "turns_scored": len(turns),
            "sessions_scored": len(sessions),
            "scores_written": written,
            "errors": errors,
            "turn_details": turn_details,
        },
    )
    _record_event(
        dependencies.store,
        run.run_id,
        "scoring_complete" if not errors else "scoring_failed",
        status_message,
        total=len(turns),
        scored=len(turns),
        written=written,
        errors=errors,
    )
    if errors:
        raise RuntimeError(f"Scoring completed with {errors} {error_label}")
