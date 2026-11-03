"""Shared evaluation-stage contracts: cancellation and the stage's store writes."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from weave_agent_signals.runs.store import Run, RunStatus, RunStore, RunStoreConflictError


class StageCancelled(Exception):
    """Signal cooperative cancellation without treating it as a stage failure."""


def require_active(
    store: RunStore, run_id: str, cancel: threading.Event, *, stage: RunStatus
) -> Run:
    """The run while it is still in `stage`. Raises StageCancelled once it was cancelled."""
    if cancel.is_set():
        raise StageCancelled()
    current = store.get(run_id)
    if current is None:
        raise ValueError(f"Run {run_id} not found")
    if current.status is RunStatus.CANCELLED:
        raise StageCancelled()
    if current.status is not stage:
        raise RunStoreConflictError(
            f"Run {run_id} is no longer {stage.value}; current status is {current.status.value}"
        )
    return current


def translate_cancellation(
    store: RunStore, run_id: str, cancel: threading.Event, conflict: RunStoreConflictError
) -> None:
    """Raise StageCancelled when a store conflict came from the run being cancelled."""
    current = store.get(run_id)
    if cancel.is_set() or (current is not None and current.status is RunStatus.CANCELLED):
        raise StageCancelled() from conflict


def record_progress(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    progress: Mapping[str, Any],
    *,
    stage: RunStatus,
) -> Run:
    try:
        return store.record_stage_progress(run_id, stage=stage, progress=progress)
    except RunStoreConflictError as conflict:
        translate_cancellation(store, run_id, cancel, conflict)
        raise


def record_result(
    store: RunStore,
    run_id: str,
    cancel: threading.Event,
    result: Mapping[str, Any],
    *,
    stage: RunStatus,
) -> Run:
    try:
        return store.record_stage_result(run_id, stage=stage, result=result)
    except RunStoreConflictError as conflict:
        translate_cancellation(store, run_id, cancel, conflict)
        raise
