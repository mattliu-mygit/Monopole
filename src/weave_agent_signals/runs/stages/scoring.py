"""Deterministic scoring for one immutable evaluation-run cohort."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from weave_agent_signals.client import WeaveClient
from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.run_config import EffectiveRunConfig
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
    client_factory: Callable[[], WeaveClient]
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


def _queue_scores(
    scores: list[Score],
    *,
    ref: str,
    config_version: str | None,
    git_branch: str | None,
    run_time: Any,
    existing_feedback: Mapping[tuple[str, str], list[dict]],
    force: bool,
    pending_writes: list[tuple[Score, str, list[dict]]],
) -> None:
    for score in scores:
        score.stamp(
            config_version=config_version,
            git_branch=git_branch,
            run_time=run_time,
        )
        feedback_type = f"weave_agent_signals.{score.scorer}"
        existing = existing_feedback.get((ref, feedback_type), [])
        if existing and not force:
            continue
        pending_writes.append((score, ref, existing))


def _write_scores(
    client: WeaveClient,
    pending_writes: list[tuple[Score, str, list[dict]]],
    *,
    run_id: str,
) -> tuple[int, int]:
    if not pending_writes:
        return 0, 0

    def write(item: tuple[Score, str, list[dict]]) -> tuple[int, int]:
        score, ref, prior = item
        try:
            client.write_score(score, ref)
        except Exception as exc:
            log.warning("Run %s score write failed: %s", run_id, exc)
            return 0, 1
        try:
            if prior:
                client.delete_feedback_ids(prior)
        except Exception as exc:
            log.warning("Run %s prior feedback cleanup failed: %s", run_id, exc)
            return 1, 1
        return 1, 0

    with ThreadPoolExecutor(max_workers=min(10, len(pending_writes))) as pool:
        results = list(pool.map(write, pending_writes))
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

    with dependencies.client_factory() as client:
        refs = [turn.ref_for(client.entity, client.project) for turn in turns]
        refs.extend(session.ref_for(client.entity, client.project) for session in sessions.values())
        existing_feedback = client.query_existing_feedback_batch(refs)
        _require_active(dependencies.store, run.run_id, cancel)

        errors = 0
        pending_writes: list[tuple[Score, str, list[dict]]] = []
        for index, turn in enumerate(turns):
            _require_active(dependencies.store, run.run_id, cancel)
            try:
                scores = score_turn(turn)
                turn_details.append(_turn_detail(turn, scores))
                _queue_scores(
                    scores,
                    ref=turn.ref_for(client.entity, client.project),
                    config_version=turn.config_version,
                    git_branch=turn.git_branch,
                    run_time=turn.started_at,
                    existing_feedback=existing_feedback,
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
                    ref=session.ref_for(client.entity, client.project),
                    config_version=session.config_version,
                    git_branch=session.git_branch,
                    run_time=session.turns[0].started_at if session.turns else None,
                    existing_feedback=existing_feedback,
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
                status_message=f"Writing {len(pending_writes)} scores...",
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
                        client,
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
    if errors:
        raise RuntimeError(f"Scoring completed with {errors} {error_label}")
