"""Lifecycle coordination for evaluation runs."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any, Protocol

from weave_agent_signals.run_config import (
    PIPELINE_VERSION,
    EffectiveRunConfig,
    EvaluatedModelIdentity,
    ModelCatalog,
    RubricCatalog,
    RunConfig,
    resolve_run_config,
)
from weave_agent_signals.runs.stages import StageCancelled
from weave_agent_signals.runs.store import (
    DataSelection,
    Run,
    RunStatus,
    RunStore,
    RunStoreConflictError,
    RunSummarySource,
)

log = logging.getLogger(__name__)


class StageRunner(Protocol):
    def __call__(
        self,
        run: Run,
        config: EffectiveRunConfig,
        cancel: threading.Event,
    ) -> None:
        """Execute one active stage using only pinned configuration."""


class RunNotFoundError(LookupError):
    """The requested evaluation run does not exist."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"Run {run_id} not found")


class RunAdvanceError(ValueError):
    """The run cannot move to its next lifecycle stage yet."""


_NEXT_STATUS = {
    RunStatus.SCORING: RunStatus.JUDGING,
    RunStatus.JUDGING: RunStatus.REFLECTING,
    RunStatus.REFLECTING: RunStatus.COMPLETE,
}
_SUCCESS_FIELD = {
    RunStatus.SCORING: "scoring_succeeded",
    RunStatus.JUDGING: "judging_succeeded",
    RunStatus.REFLECTING: "reflecting_succeeded",
}
_RESULT_FIELD = {
    RunStatus.SCORING: "scoring_result",
    RunStatus.JUDGING: "judging_result",
    RunStatus.REFLECTING: "reflecting_result",
}


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _persisted_result_is_successful(run: Run) -> bool:
    """Validate a completed result that may have missed its final status write."""

    result = getattr(run, _RESULT_FIELD[run.status])
    if not isinstance(result, dict):
        return False
    if run.status is RunStatus.SCORING:
        cohort = run.turn_cohort or {}
        return (
            _nonnegative_int(result.get("turns_scored"))
            and result["turns_scored"] == cohort.get("turn_count")
            and _nonnegative_int(result.get("sessions_scored"))
            and result["sessions_scored"] == cohort.get("session_count")
            and _nonnegative_int(result.get("scores_written"))
            and _nonnegative_int(result.get("errors"))
            and result.get("errors") == 0
            and isinstance(result.get("turn_details"), list)
        )
    if run.status is RunStatus.JUDGING:
        plan = run.judging_plan
        return (
            plan is not None
            and isinstance(result.get("plan_id"), str)
            and result["plan_id"] == plan.plan_id
            and result.get("coverage_complete") is True
            and _nonnegative_int(result.get("planned_rubrics"))
            and _nonnegative_int(result.get("rubrics_completed"))
            and result["rubrics_completed"] == result["planned_rubrics"]
            and result.get("failure_count") == 0
            and result.get("write_failure_count") == 0
        )
    return False


def _configuration_error(run: Run) -> str | None:
    if run.effective_config is None:
        return "Run has no pinned effective configuration"
    if run.effective_config.pipeline_version != PIPELINE_VERSION:
        return "Pinned pipeline version does not match this server; start a new run."
    return None


class RunService:
    """Coordinate immutable run inputs, stages, and cancellation."""

    def __init__(
        self,
        *,
        store: RunStore,
        build_model_catalog: Callable[[], ModelCatalog],
        build_rubric_catalog: Callable[[], RubricCatalog],
        discover_cohort: Callable[[DataSelection], dict[str, Any]],
        scoring_stage: StageRunner,
        judging_stage: StageRunner,
        reflection_stage: StageRunner,
        executor: Executor,
    ) -> None:
        self.store = store
        self._build_model_catalog = build_model_catalog
        self._build_rubric_catalog = build_rubric_catalog
        self._discover_cohort = discover_cohort
        self._stages = {
            RunStatus.SCORING: scoring_stage,
            RunStatus.JUDGING: judging_stage,
            RunStatus.REFLECTING: reflection_stage,
        }
        self._executor = executor
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()

    def create(self) -> Run:
        return self.store.create()

    def get(self, run_id: str) -> Run:
        run = self.store.get(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        return run

    def list_summaries(self, limit: int = 50) -> list[RunSummarySource]:
        return self.store.list_summaries(limit)

    def delete(self, run_id: str) -> None:
        self.get(run_id)
        self.store.delete(run_id)

    def recover_interrupted_runs(self) -> None:
        """Resolve active rows left by a stopped server without repeating paid work."""

        for run in self.store.list_active():
            stage = run.status
            configuration_error = _configuration_error(run)
            if configuration_error is not None:
                self._fail_active(run.run_id, stage, configuration_error)
                continue
            if run.current_stage_succeeded:
                continue
            result = getattr(run, _RESULT_FIELD[stage])
            if stage is RunStatus.REFLECTING and result is not None:
                # The reflection stage validates the evidence and finishes any
                # missing review initialization without running inference again.
                self._submit(run.run_id, stage)
                continue
            if result is None:
                self._fail_active(
                    run.run_id,
                    stage,
                    f"Server stopped before {stage.value} completed. Start a new run.",
                )
                continue
            if not _persisted_result_is_successful(run):
                message = result.get("status_message")
                reason = (
                    message.strip()
                    if isinstance(message, str) and message.strip()
                    else f"Persisted {stage.value} result is incomplete"
                )
                self._fail_active(run.run_id, stage, f"{reason}. Start a new run.")
                continue

            advance = run.auto_run
            finalized = self.store.finalize_stage_success(
                run.run_id,
                stage=stage,
                advance=advance,
            )
            if advance and finalized.status in self._stages:
                self._submit(run.run_id, finalized.status)

    def save_selection(self, run_id: str, selection: DataSelection) -> Run:
        self.get(run_id)
        return self.store.save_selection(run_id, selection)

    def save_config(self, run_id: str, config: RunConfig) -> Run:
        self.get(run_id)
        resolve_run_config(
            config,
            model_catalog=self._build_model_catalog(),
            rubric_catalog=self._build_rubric_catalog(),
        )
        return self.store.save_config(run_id, config)

    def set_auto_run(self, run_id: str, auto_run: bool) -> Run:
        self.get(run_id)
        return self.store.save_auto_run(run_id, auto_run)

    def cancel(self, run_id: str) -> Run:
        self.get(run_id)
        cancelled = self.store.cancel_if_safe(run_id)
        with self._cancel_lock:
            event = self._cancel_events.get(run_id)
            if event is not None:
                event.set()
        return cancelled

    def advance(self, run_id: str) -> Run:
        run = self.get(run_id)
        if run.status is RunStatus.CREATED:
            started = self._start(run)
            self._submit(started.run_id, RunStatus.SCORING)
            return self.get(run_id)

        if run.status not in _NEXT_STATUS:
            raise RunAdvanceError(f"Cannot advance from {run.status.value}")
        if run.auto_run:
            raise RunStoreConflictError("Auto-run advances completed stages automatically")
        if not getattr(run, _SUCCESS_FIELD[run.status]):
            raise RunStoreConflictError(
                f"Run {run_id} {run.status.value} has not finalized successfully"
            )

        next_status = _NEXT_STATUS[run.status]
        transitioned = self.store.transition(
            run_id,
            expected_status=run.status,
            new_status=next_status,
        )
        if next_status in self._stages:
            self._submit(run_id, next_status)
        return self.get(transitioned.run_id)

    def execute_stage(self, run_id: str, stage: RunStatus) -> None:
        if stage not in self._stages:
            raise ValueError("stage must be scoring, judging, or reflecting")
        run = self.get(run_id)
        if run.status is RunStatus.CANCELLED:
            return
        if run.status is not stage:
            raise RunStoreConflictError(
                f"Run {run_id} expected {stage.value}; current status is {run.status.value}"
            )

        cancel = threading.Event()
        with self._cancel_lock:
            previous = self._cancel_events.get(run_id)
            if previous is not None and previous.is_set():
                cancel.set()
            self._cancel_events[run_id] = cancel

        next_stage: RunStatus | None = None
        try:
            configuration_error = _configuration_error(run)
            if configuration_error is not None:
                self._fail_active(run_id, stage, configuration_error)
                return
            config = run.effective_config
            assert config is not None

            self._stages[stage](run, config, cancel)
            advance = stage is RunStatus.REFLECTING or run.auto_run
            finalized = self.store.finalize_stage_success(
                run_id,
                stage=stage,
                advance=advance,
            )
            if advance and finalized.status in self._stages:
                next_stage = finalized.status
        except StageCancelled:
            log.info("Run %s cancelled during %s", run_id, stage.value)
        except Exception as error:
            log.warning("Run %s stage %s failed: %s", run_id, stage.value, error)
            self._fail_active(run_id, stage, str(error))
        finally:
            with self._cancel_lock:
                if self._cancel_events.get(run_id) is cancel:
                    self._cancel_events.pop(run_id, None)

        if next_stage is not None:
            self._submit(run_id, next_stage)

    def _start(self, run: Run) -> Run:
        if run.data_selection is None:
            raise RunAdvanceError("Data selection required before starting")
        if run.run_config is None:
            raise RunAdvanceError("Run configuration required before starting")

        cohort = self._discover_cohort(run.data_selection)
        evaluated_models = tuple(
            EvaluatedModelIdentity(
                id=turn.get("model"),
                family=turn.get("model_family", ""),
            )
            for turn in cohort.get("turns", ())
            if isinstance(turn, dict)
        )
        model_catalog = self._build_model_catalog()
        rubric_catalog = self._build_rubric_catalog()
        effective = resolve_run_config(
            run.run_config,
            model_catalog=model_catalog,
            rubric_catalog=rubric_catalog,
            evaluated_models=evaluated_models,
        )
        return self.store.start(
            run.run_id,
            expected_selection=run.data_selection,
            expected_config=run.run_config,
            turn_cohort=cohort,
            effective_config=effective,
        )

    def _submit(self, run_id: str, stage: RunStatus) -> None:
        self._executor.submit(self.execute_stage, run_id, stage)

    def _fail_active(self, run_id: str, stage: RunStatus, error: str) -> None:
        current = self.store.get(run_id)
        if current is None or current.status is not stage:
            return
        try:
            self.store.fail(run_id, expected_status=stage, error=error)
        except RunStoreConflictError:
            latest = self.store.get(run_id)
            if latest is not None and latest.status is not RunStatus.CANCELLED:
                raise
