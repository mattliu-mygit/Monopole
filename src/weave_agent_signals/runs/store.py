"""Typed SQLite persistence for the evaluation-run lifecycle."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from weave_agent_signals.judges.plan import JudgingPlan
from weave_agent_signals.judges.records import JudgeCallAudit, JudgeCallRecord
from weave_agent_signals.run_config import EffectiveRunConfig, RunConfig
from weave_agent_signals.runs.events import RunEvent, RunEventDraft, RunEventStage
from weave_agent_signals.runs.migrations import RunMigrationError, migrate_v9_to_v10
from weave_agent_signals.runs.reflection_records import (
    ReflectionInputRecord,
    ReflectionResultRecord,
)

_DEFAULT_DB_DIR = Path.home() / ".weave-agent-signals"
RUN_DB_SCHEMA_VERSION = 10


def _default_db_path() -> Path:
    _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_DB_DIR / "runs.db"


class RunStatus(str, Enum):
    CREATED = "created"
    SCORING = "scoring"
    JUDGING = "judging"
    REFLECTING = "reflecting"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset({RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.SCORING: frozenset({RunStatus.JUDGING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.JUDGING: frozenset({RunStatus.REFLECTING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.REFLECTING: frozenset({RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.COMPLETE: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}
_ACTIVE_STAGES = frozenset({RunStatus.SCORING, RunStatus.JUDGING, RunStatus.REFLECTING})
_STAGE_FIELDS = {
    RunStatus.SCORING: ("scoring_progress", "scoring_result"),
    RunStatus.JUDGING: ("judging_progress", "judging_result"),
    RunStatus.REFLECTING: ("reflecting_progress", "reflecting_result"),
}
_STAGE_SUCCESS_FIELDS = {
    RunStatus.SCORING: "scoring_succeeded",
    RunStatus.JUDGING: "judging_succeeded",
    RunStatus.REFLECTING: "reflecting_succeeded",
}
_STAGE_SUCCESSORS = {
    RunStatus.SCORING: RunStatus.JUDGING,
    RunStatus.JUDGING: RunStatus.REFLECTING,
    RunStatus.REFLECTING: RunStatus.COMPLETE,
}
_REFLECTION_REVIEW_STATUSES = frozenset({"pending", "promoted", "partial", "dismissed"})
_RESOLVED_REFLECTION_REVIEW_STATUSES = frozenset({"promoted", "partial", "dismissed"})


class RunStoreConflictError(ValueError):
    """Base class for optimistic-concurrency and lifecycle conflicts."""


class RunLifecycleConflictError(RunStoreConflictError):
    def __init__(self, run_id: str, current_status: RunStatus):
        self.run_id = run_id
        self.current_status = current_status
        super().__init__(
            f"Run {run_id} can only be configured while created; "
            f"current status is {current_status.value}"
        )


class RunCancellationConflictError(RunStoreConflictError):
    def __init__(
        self,
        run_id: str,
        current_status: RunStatus,
        *,
        reflection_finalizing: bool,
    ):
        self.run_id = run_id
        self.current_status = current_status
        self.reflection_finalizing = reflection_finalizing
        reason = (
            "reflection is finalizing"
            if reflection_finalizing
            else f"current status is {current_status.value}"
        )
        super().__init__(f"Run {run_id} cannot be cancelled because {reason}")


class ReflectionReviewRevisionConflictError(RunStoreConflictError):
    def __init__(
        self,
        run_id: str,
        expected_revision: int,
        current_revision: int,
        current_review: dict[str, Any] | None,
    ):
        self.run_id = run_id
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        self.current_review = current_review
        super().__init__(
            f"Reflection review revision conflict for {run_id}: "
            f"expected {expected_revision}, current {current_revision}"
        )


class ReflectionReviewLifecycleConflictError(RunStoreConflictError):
    def __init__(
        self,
        run_id: str,
        message: str,
        *,
        current_run_status: RunStatus,
        current_review_status: str | None,
        current_revision: int,
    ):
        self.run_id = run_id
        self.current_run_status = current_run_status
        self.current_review_status = current_review_status
        self.current_revision = current_revision
        super().__init__(f"Reflection review conflict for {run_id}: {message}")


@dataclass(frozen=True)
class DataSelection:
    since: str | None = None
    until: str | None = None
    timezone: str | None = None
    session_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_ids", tuple(self.session_ids))


def _current_stage_succeeded(
    status: RunStatus,
    *,
    scoring: bool,
    judging: bool,
    reflecting: bool,
) -> bool:
    if status is RunStatus.SCORING:
        return scoring
    if status is RunStatus.JUDGING:
        return judging
    if status in {RunStatus.REFLECTING, RunStatus.COMPLETE}:
        return reflecting
    return False


@dataclass(frozen=True)
class RunSummarySource:
    """Scalar run-list fields projected without hydrating full evidence."""

    run_id: str
    status: RunStatus
    created_at: str
    scoring_succeeded: bool
    judging_succeeded: bool
    reflecting_succeeded: bool
    session_count: int | None
    selection_since: str | None
    selection_until: str | None
    selection_timezone: str | None
    review_status: str | None
    baseline_won: bool | None
    reflection_reason: str | None

    @property
    def current_stage_succeeded(self) -> bool:
        return _current_stage_succeeded(
            self.status,
            scoring=self.scoring_succeeded,
            judging=self.judging_succeeded,
            reflecting=self.reflecting_succeeded,
        )


@dataclass(frozen=True)
class Run:
    run_id: str
    status: RunStatus
    created_at: str
    auto_run: bool = False
    data_selection: DataSelection | None = None
    run_config: RunConfig | None = None
    effective_config: EffectiveRunConfig | None = None
    turn_cohort: dict[str, Any] | None = None
    judging_plan: JudgingPlan | None = None
    reflection_input: ReflectionInputRecord | None = None
    scoring_progress: dict[str, Any] | None = None
    scoring_result: dict[str, Any] | None = None
    scoring_succeeded: bool = False
    judging_progress: dict[str, Any] | None = None
    judging_result: dict[str, Any] | None = None
    judging_succeeded: bool = False
    reflecting_progress: dict[str, Any] | None = None
    reflecting_result: ReflectionResultRecord | None = None
    reflecting_succeeded: bool = False
    reflection_review: dict[str, Any] | None = None
    reflection_review_revision: int = 0
    error: str | None = None

    @property
    def current_stage_succeeded(self) -> bool:
        """Whether the active stage worker returned and finalized successfully."""

        return _current_stage_succeeded(
            self.status,
            scoring=self.scoring_succeeded,
            judging=self.judging_succeeded,
            reflecting=self.reflecting_succeeded,
        )


def _encode_data_selection(selection: DataSelection) -> str:
    if not isinstance(selection, DataSelection):
        raise ValueError("data selection must be a DataSelection")
    if not selection.session_ids:
        raise ValueError("selection must include at least one session")
    if any(not session_id.strip() for session_id in selection.session_ids):
        raise ValueError("selection session IDs must be nonblank")
    if len(selection.session_ids) != len(set(selection.session_ids)):
        raise ValueError("selection session IDs must be unique")
    return json.dumps(asdict(selection), sort_keys=True)


def _encode_run_config(config: RunConfig) -> str:
    if not isinstance(config, RunConfig):
        raise ValueError("run configuration must be a RunConfig")
    return config.model_dump_json()


def _encode_effective_config(config: EffectiveRunConfig) -> str:
    if not isinstance(config, EffectiveRunConfig):
        raise ValueError("effective configuration must be an EffectiveRunConfig")
    return config.model_dump_json()


def _encode_json_object(value: Mapping[str, Any], label: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    try:
        return json.dumps(dict(value), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc


def _encode_turn_cohort(turn_cohort: Mapping[str, Any]) -> str:
    if not isinstance(turn_cohort, Mapping):
        raise ValueError("turn cohort must be a JSON object")
    cohort = dict(turn_cohort)
    turns = cohort.get("turns")
    sessions = cohort.get("sessions")
    if not isinstance(turns, list) or not turns:
        raise ValueError("turn cohort must contain at least one turn")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("turn cohort must contain at least one session")
    if cohort.get("turn_count") != len(turns):
        raise ValueError("turn cohort turn_count does not match turns")
    if cohort.get("session_count") != len(sessions):
        raise ValueError("turn cohort session_count does not match sessions")
    if not isinstance(cohort.get("cohort_id"), str) or not cohort["cohort_id"]:
        raise ValueError("turn cohort must have a cohort_id")

    trace_ids: set[str] = set()
    refs: set[str] = set()
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("turn cohort entries must be JSON objects")
        trace_id = turn.get("trace_id")
        weave_ref = turn.get("weave_ref")
        conversation_id = turn.get("conversation_id")
        if not all(
            isinstance(value, str) and value for value in (trace_id, weave_ref, conversation_id)
        ):
            raise ValueError("turn cohort entries require trace_id, weave_ref, and conversation_id")
        if trace_id in trace_ids or weave_ref in refs:
            raise ValueError("turn cohort entries must be unique")
        evaluated_model = turn.get("model")
        evaluated_family = turn.get("model_family")
        if evaluated_model is not None and (
            not isinstance(evaluated_model, str) or not evaluated_model
        ):
            raise ValueError("turn cohort model must be null or a nonblank string")
        if not isinstance(evaluated_family, str) or not evaluated_family.strip():
            raise ValueError("turn cohort model family must be a nonblank string")
        trace_ids.add(trace_id)
        refs.add(weave_ref)
    return _encode_json_object(cohort, "turn cohort")


def _encode_reflection_input(reflection_input: ReflectionInputRecord) -> str:
    if not isinstance(reflection_input, ReflectionInputRecord):
        raise TypeError("reflection_input must be a ReflectionInputRecord")
    return ReflectionInputRecord.model_validate(
        reflection_input.model_dump(mode="json")
    ).model_dump_json()


def _encode_judging_plan(judging_plan: JudgingPlan) -> str:
    if not isinstance(judging_plan, JudgingPlan):
        raise TypeError("judging_plan must be a JudgingPlan")
    return JudgingPlan.model_validate(judging_plan.model_dump(mode="json")).model_dump_json()


def _judging_plan_matches_cohort(
    judging_plan: JudgingPlan,
    turn_cohort: Mapping[str, Any],
) -> bool:
    cohort_sessions = turn_cohort.get("sessions")
    cohort_turns = turn_cohort.get("turns")
    if not isinstance(cohort_sessions, list) or not isinstance(cohort_turns, list):
        return False

    session_turn_counts: dict[str, int] = {}
    for session in cohort_sessions:
        if not isinstance(session, dict):
            return False
        conversation_id = session.get("conversation_id")
        turn_count = session.get("turn_count")
        if (
            not isinstance(conversation_id, str)
            or conversation_id in session_turn_counts
            or type(turn_count) is not int
        ):
            return False
        session_turn_counts[conversation_id] = turn_count

    turn_sessions: dict[str, str] = {}
    for turn in cohort_turns:
        if not isinstance(turn, dict):
            return False
        trace_id = turn.get("trace_id")
        conversation_id = turn.get("conversation_id")
        if (
            not isinstance(trace_id, str)
            or trace_id in turn_sessions
            or not isinstance(conversation_id, str)
        ):
            return False
        turn_sessions[trace_id] = conversation_id

    if sum(len(session.turns) for session in judging_plan.sessions) != len(cohort_turns):
        return False

    planned_session_ids: set[str] = set()
    for session in judging_plan.sessions:
        conversation_id = session.conversation_id
        if conversation_id in planned_session_ids or len(session.turns) != session_turn_counts.get(
            conversation_id
        ):
            return False
        planned_session_ids.add(conversation_id)
        coverage = [turn.trace_id for turn in session.turns]
        expected_coverage = [
            trace_id for trace_id, owner in turn_sessions.items() if owner == conversation_id
        ]
        if coverage != expected_coverage:
            return False
        for reviewer in session.reviewers:
            if reviewer.status == "skipped":
                if reviewer.window_plan is not None:
                    return False
            elif reviewer.status != "planned" or reviewer.window_plan is None:
                return False
        if [reviewer.position for reviewer in session.reviewers] != list(
            range(1, len(session.reviewers) + 1)
        ):
            return False
    return planned_session_ids == set(session_turn_counts)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'created',
    created_at TEXT NOT NULL,
    auto_run INTEGER NOT NULL DEFAULT 0,
    data_selection TEXT,
    run_config TEXT,
    effective_config TEXT,
    turn_cohort TEXT,
    judging_plan TEXT,
    reflection_input TEXT,
    scoring_progress TEXT,
    scoring_result TEXT,
    scoring_succeeded INTEGER NOT NULL DEFAULT 0,
    judging_progress TEXT,
    judging_result TEXT,
    judging_succeeded INTEGER NOT NULL DEFAULT 0,
    reflecting_progress TEXT,
    reflecting_result TEXT,
    reflecting_succeeded INTEGER NOT NULL DEFAULT 0,
    reflection_review TEXT,
    reflection_review_revision INTEGER NOT NULL DEFAULT 0,
    error TEXT
)
"""

_JUDGE_CALLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS judge_calls (
    run_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    reviewer_position INTEGER NOT NULL,
    requested_model_id TEXT NOT NULL,
    rubric_id TEXT,
    status TEXT NOT NULL,
    reusable INTEGER NOT NULL,
    result_json TEXT,
    audit_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, request_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
)
"""

_RUN_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_events (
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    stage TEXT NOT NULL,
    at TEXT NOT NULL,
    phase TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
)
"""

_EVENT_LIMITS: dict[RunEventStage, int] = {
    "scoring": 100,
    "judging": 1_000,
    "reflecting": 100,
}

_JSON_FIELDS = {
    "turn_cohort",
    "scoring_progress",
    "scoring_result",
    "judging_progress",
    "judging_result",
    "reflecting_progress",
    "reflection_review",
}


class RunStore:
    """SQLite store whose writes enforce the evaluation-run lifecycle."""

    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path or _default_db_path())
        self._conn = sqlite3.connect(
            str(path),
            check_same_thread=False,
        )
        self._lock = threading.Lock()
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA busy_timeout=5000")
            stored_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            has_runs = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
            ).fetchone()
            if stored_version == 9:
                migrate_v9_to_v10(
                    self._conn,
                    path,
                    lambda: datetime.now(timezone.utc),
                )
            elif stored_version == 0 and has_runs is None:
                pass
            elif stored_version != RUN_DB_SCHEMA_VERSION:
                raise RunMigrationError(f"unsupported run database schema: {stored_version}")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(_SCHEMA)
            self._conn.execute(_JUDGE_CALLS_SCHEMA)
            self._conn.execute(_RUN_EVENTS_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {RUN_DB_SCHEMA_VERSION}")
            self._conn.commit()
        except BaseException:
            try:
                self._conn.rollback()
            finally:
                self._conn.close()
            raise

    def create(self) -> Run:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, status, created_at) VALUES (?, ?, ?)",
                (run_id, RunStatus.CREATED.value, created_at),
            )
            self._conn.commit()
            row = self._get_row_locked(run_id)
        return self._row_to_run(row)

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._row_to_run(row) if row is not None else None

    def list_summaries(self, limit: int = 50) -> list[RunSummarySource]:
        """Project scalar list fields without loading full run payloads."""

        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    run_id,
                    status,
                    created_at,
                    scoring_succeeded,
                    judging_succeeded,
                    reflecting_succeeded,
                    json_array_length(data_selection, '$.session_ids') AS session_count,
                    json_extract(data_selection, '$.since') AS selection_since,
                    json_extract(data_selection, '$.until') AS selection_until,
                    json_extract(data_selection, '$.timezone') AS selection_timezone,
                    json_extract(reflection_review, '$.status') AS review_status,
                    json_extract(reflecting_result, '$.baseline_won') AS baseline_won,
                    json_extract(reflecting_result, '$.reason') AS reflection_reason
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            RunSummarySource(
                run_id=row["run_id"],
                status=RunStatus(row["status"]),
                created_at=row["created_at"],
                scoring_succeeded=bool(row["scoring_succeeded"]),
                judging_succeeded=bool(row["judging_succeeded"]),
                reflecting_succeeded=bool(row["reflecting_succeeded"]),
                session_count=row["session_count"],
                selection_since=row["selection_since"],
                selection_until=row["selection_until"],
                selection_timezone=row["selection_timezone"],
                review_status=row["review_status"],
                baseline_won=(
                    bool(row["baseline_won"]) if row["baseline_won"] is not None else None
                ),
                reflection_reason=row["reflection_reason"],
            )
            for row in rows
        ]

    def list_active(self) -> list[Run]:
        """Return every run whose pipeline stage has not reached a terminal state."""

        statuses = tuple(stage.value for stage in _ACTIVE_STAGES)
        placeholders = ", ".join("?" for _ in statuses)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM runs WHERE status IN ({placeholders}) ORDER BY created_at ASC",
                statuses,
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def save_selection(self, run_id: str, selection: DataSelection) -> Run:
        encoded = _encode_data_selection(selection)
        with self._lock:
            return self._write_created_only_locked(
                run_id,
                "UPDATE runs SET data_selection = ? WHERE run_id = ? AND status = ?",
                (encoded,),
            )

    def save_config(self, run_id: str, config: RunConfig) -> Run:
        encoded = _encode_run_config(config)
        with self._lock:
            return self._write_created_only_locked(
                run_id,
                "UPDATE runs SET run_config = ? WHERE run_id = ? AND status = ?",
                (encoded,),
            )

    def save_auto_run(self, run_id: str, auto_run: bool) -> Run:
        if type(auto_run) is not bool:
            raise ValueError("auto_run must be a boolean")
        with self._lock:
            return self._write_created_only_locked(
                run_id,
                "UPDATE runs SET auto_run = ? WHERE run_id = ? AND status = ?",
                (int(auto_run),),
            )

    def start(
        self,
        run_id: str,
        *,
        expected_selection: DataSelection,
        expected_config: RunConfig,
        turn_cohort: Mapping[str, Any],
        effective_config: EffectiveRunConfig,
    ) -> Run:
        """Atomically pin immutable inputs and transition created to scoring."""

        encoded_selection = _encode_data_selection(expected_selection)
        encoded_config = _encode_run_config(expected_config)
        encoded_cohort = _encode_turn_cohort(turn_cohort)
        encoded_effective = _encode_effective_config(effective_config)
        if (
            effective_config.model_catalog_version != expected_config.model_catalog_version
            or effective_config.rubric_catalog_version != expected_config.rubric_catalog_version
        ):
            raise ValueError("effective configuration does not match requested catalogs")

        with self._lock:
            row = self._get_row_locked(run_id)
            current = RunStatus(row["status"])
            if current is not RunStatus.CREATED:
                raise RunLifecycleConflictError(run_id, current)
            if row["data_selection"] != encoded_selection:
                raise RunStoreConflictError(
                    f"Run {run_id} data selection changed while pinning inputs"
                )
            if row["run_config"] is None:
                raise ValueError("run configuration required before start")
            if row["run_config"] != encoded_config:
                raise RunStoreConflictError(
                    f"Run {run_id} configuration changed while pinning inputs"
                )
            if row["turn_cohort"] is not None or row["effective_config"] is not None:
                raise RunStoreConflictError(f"Run {run_id} immutable inputs are already pinned")

            cursor = self._conn.execute(
                "UPDATE runs SET status = ?, turn_cohort = ?, effective_config = ? "
                "WHERE run_id = ? AND status = ? AND data_selection = ? "
                "AND run_config = ? AND turn_cohort IS NULL AND effective_config IS NULL",
                (
                    RunStatus.SCORING.value,
                    encoded_cohort,
                    encoded_effective,
                    run_id,
                    RunStatus.CREATED.value,
                    encoded_selection,
                    encoded_config,
                ),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                latest = self._get_row_locked(run_id)
                latest_status = RunStatus(latest["status"])
                if latest_status is not RunStatus.CREATED:
                    raise RunLifecycleConflictError(run_id, latest_status)
                raise RunStoreConflictError(f"Run {run_id} inputs changed while starting")
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def record_stage_progress(
        self,
        run_id: str,
        *,
        stage: RunStatus,
        progress: Mapping[str, Any],
    ) -> Run:
        return self._record_stage_value(
            run_id,
            stage=stage,
            value=progress,
            result=False,
        )

    def record_stage_result(
        self,
        run_id: str,
        *,
        stage: RunStatus,
        result: Mapping[str, Any] | ReflectionResultRecord,
    ) -> Run:
        return self._record_stage_value(
            run_id,
            stage=stage,
            value=result,
            result=True,
        )

    def finalize_stage_success(
        self,
        run_id: str,
        *,
        stage: RunStatus,
        advance: bool,
    ) -> Run:
        """Atomically mark a returned worker successful and optionally advance it."""

        if stage not in _ACTIVE_STAGES:
            raise ValueError("stage must be scoring, judging, or reflecting")
        if type(advance) is not bool:
            raise ValueError("advance must be a boolean")
        success_field = _STAGE_SUCCESS_FIELDS[stage]
        result_field = _STAGE_FIELDS[stage][1]
        new_status = _STAGE_SUCCESSORS[stage] if advance else stage
        with self._lock:
            row = self._get_row_locked(run_id)
            current = RunStatus(row["status"])
            if current is not stage:
                raise RunStoreConflictError(
                    f"Run {run_id} expected {stage.value}; current status is {current.value}"
                )
            if row[result_field] is None:
                raise RunStoreConflictError(
                    f"Run {run_id} {stage.value} cannot finalize without a result"
                )
            cursor = self._conn.execute(
                f"UPDATE runs SET {success_field} = 1, status = ? "
                f"WHERE run_id = ? AND status = ? AND {result_field} IS NOT NULL",
                (new_status.value, run_id, stage.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise RunStoreConflictError(f"Run {run_id} changed while finalizing {stage.value}")
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def transition(
        self,
        run_id: str,
        *,
        expected_status: RunStatus,
        new_status: RunStatus,
    ) -> Run:
        if not isinstance(expected_status, RunStatus) or not isinstance(new_status, RunStatus):
            raise ValueError("transition statuses must be RunStatus values")
        if expected_status is RunStatus.CREATED and new_status is RunStatus.SCORING:
            raise RunStoreConflictError(
                f"Run {run_id} must use start() to transition from created to scoring"
            )
        if new_status not in _TRANSITIONS[expected_status]:
            raise ValueError(f"Invalid transition: {expected_status.value} -> {new_status.value}")
        success_field = (
            _STAGE_SUCCESS_FIELDS[expected_status]
            if _STAGE_SUCCESSORS.get(expected_status) is new_status
            else None
        )
        success_guard = f" AND {success_field} = 1" if success_field is not None else ""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ? AND status = ?" + success_guard,
                (new_status.value, run_id, expected_status.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                latest = self._get_row_locked(run_id)
                current = RunStatus(latest["status"])
                if (
                    current is expected_status
                    and success_field is not None
                    and not bool(latest[success_field])
                ):
                    raise RunStoreConflictError(
                        f"Run {run_id} {expected_status.value} has not finalized successfully"
                    )
                raise RunStoreConflictError(
                    f"Run {run_id} expected {expected_status.value}; current status is "
                    f"{current.value}"
                )
            self._conn.commit()
            row = self._get_row_locked(run_id)
        return self._row_to_run(row)

    def fail(
        self,
        run_id: str,
        *,
        expected_status: RunStatus,
        error: str,
    ) -> Run:
        if (
            expected_status not in _TRANSITIONS
            or RunStatus.FAILED not in _TRANSITIONS[expected_status]
        ):
            raise ValueError(f"Run status {expected_status.value} cannot fail")
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be a nonblank string")
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runs SET status = ?, error = ? WHERE run_id = ? AND status = ?",
                (
                    RunStatus.FAILED.value,
                    error,
                    run_id,
                    expected_status.value,
                ),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                current = RunStatus(self._get_row_locked(run_id)["status"])
                raise RunStoreConflictError(
                    f"Run {run_id} expected {expected_status.value}; current status is "
                    f"{current.value}"
                )
            self._conn.commit()
            row = self._get_row_locked(run_id)
        return self._row_to_run(row)

    def pin_reflection_input(
        self,
        run_id: str,
        reflection_input: ReflectionInputRecord,
    ) -> Run:
        encoded = _encode_reflection_input(reflection_input)
        with self._lock:
            row = self._get_row_locked(run_id)
            if RunStatus(row["status"]) is not RunStatus.REFLECTING:
                raise RunStoreConflictError(
                    f"Run {run_id} can only pin reflection input while reflecting; "
                    f"current status is {row['status']}"
                )
            current = (
                ReflectionInputRecord.model_validate_json(row["reflection_input"])
                if row["reflection_input"]
                else None
            )
            if current is not None:
                if current == reflection_input:
                    return self._row_to_run(row)
                raise RunStoreConflictError(f"Run {run_id} reflection input is already pinned")
            cursor = self._conn.execute(
                "UPDATE runs SET reflection_input = ? "
                "WHERE run_id = ? AND status = ? AND reflection_input IS NULL",
                (encoded, run_id, RunStatus.REFLECTING.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise RunStoreConflictError(f"Run {run_id} changed while pinning reflection input")
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def pin_judging_plan(
        self,
        run_id: str,
        judging_plan: JudgingPlan,
    ) -> Run:
        encoded = _encode_judging_plan(judging_plan)
        with self._lock:
            row = self._get_row_locked(run_id)
            if RunStatus(row["status"]) is not RunStatus.JUDGING:
                raise RunStoreConflictError(
                    f"Run {run_id} can only pin a judging plan while judging; "
                    f"current status is {row['status']}"
                )
            cohort = json.loads(row["turn_cohort"]) if row["turn_cohort"] else None
            if not isinstance(cohort, dict) or cohort.get("cohort_id") != judging_plan.cohort_id:
                raise RunStoreConflictError(
                    f"Run {run_id} judging plan cohort does not match its pinned cohort"
                )
            if not _judging_plan_matches_cohort(judging_plan, cohort):
                raise RunStoreConflictError(
                    f"Run {run_id} judging plan contents do not match its pinned cohort"
                )
            current = (
                JudgingPlan.model_validate_json(row["judging_plan"])
                if row["judging_plan"]
                else None
            )
            if current is not None:
                if current == judging_plan:
                    return self._row_to_run(row)
                raise RunStoreConflictError(f"Run {run_id} judging plan is already pinned")
            cursor = self._conn.execute(
                "UPDATE runs SET judging_plan = ? "
                "WHERE run_id = ? AND status = ? AND judging_plan IS NULL",
                (encoded, run_id, RunStatus.JUDGING.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise RunStoreConflictError(f"Run {run_id} changed while pinning judging plan")
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    @staticmethod
    def _judge_call_from_row(row: sqlite3.Row) -> JudgeCallRecord:
        return JudgeCallRecord(
            request_id=row["request_id"],
            phase=row["phase"],
            conversation_id=row["conversation_id"],
            reviewer_position=row["reviewer_position"],
            requested_model_id=row["requested_model_id"],
            rubric_id=row["rubric_id"],
            status=row["status"],
            reusable=bool(row["reusable"]),
            result=json.loads(row["result_json"]) if row["result_json"] is not None else None,
            audit=JudgeCallAudit.model_validate_json(row["audit_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def record_judge_call(self, run_id: str, record: JudgeCallRecord) -> JudgeCallRecord:
        """Insert one immutable inference call or return identical existing content."""

        if not isinstance(record, JudgeCallRecord):
            raise TypeError("record must be a JudgeCallRecord")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                run = self._get_row_locked(run_id)
                if RunStatus(run["status"]) is not RunStatus.JUDGING:
                    raise RunStoreConflictError(
                        f"Run {run_id} can only record judge calls while judging"
                    )
                existing = self._conn.execute(
                    "SELECT * FROM judge_calls WHERE run_id = ? AND request_id = ?",
                    (run_id, record.request_id),
                ).fetchone()
                if existing is not None:
                    stored = self._judge_call_from_row(existing)
                    if stored != record:
                        raise RunStoreConflictError(
                            f"Run {run_id} judge call already exists with different content: "
                            f"{record.request_id}"
                        )
                    self._conn.commit()
                    return stored
                self._conn.execute(
                    """
                    INSERT INTO judge_calls (
                        run_id, request_id, phase, conversation_id, reviewer_position,
                        requested_model_id, rubric_id, status, reusable, result_json,
                        audit_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        record.request_id,
                        record.phase,
                        record.conversation_id,
                        record.reviewer_position,
                        record.requested_model_id,
                        record.rubric_id,
                        record.status,
                        int(record.reusable),
                        (
                            json.dumps(record.result, sort_keys=True)
                            if record.result is not None
                            else None
                        ),
                        record.audit.model_dump_json(),
                        record.created_at.isoformat(),
                    ),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return record

    def get_judge_call(self, run_id: str, request_id: str) -> JudgeCallRecord | None:
        """Return an exact reusable success, never a historical audit-only call."""

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM judge_calls WHERE run_id = ? AND request_id = ? AND reusable = 1",
                (run_id, request_id),
            ).fetchone()
        return self._judge_call_from_row(row) if row is not None else None

    def list_judge_calls(self, run_id: str) -> list[JudgeCallRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM judge_calls WHERE run_id = ? ORDER BY created_at, rowid",
                (run_id,),
            ).fetchall()
        return [self._judge_call_from_row(row) for row in rows]

    def append_run_event(self, run_id: str, draft: RunEventDraft) -> RunEvent:
        """Append one bounded event with a store-assigned monotonic sequence."""

        if not isinstance(draft, RunEventDraft):
            raise TypeError("draft must be a RunEventDraft")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._get_row_locked(run_id)
                sequence = self._conn.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
                self._conn.execute(
                    "INSERT INTO run_events "
                    "(run_id, sequence, stage, at, phase, message, details_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        sequence,
                        draft.stage,
                        draft.at.isoformat(),
                        draft.phase,
                        draft.message,
                        json.dumps(draft.details, sort_keys=True),
                    ),
                )
                limit = _EVENT_LIMITS[draft.stage]
                self._conn.execute(
                    "DELETE FROM run_events WHERE run_id = ? AND stage = ? AND sequence NOT IN "
                    "(SELECT sequence FROM run_events WHERE run_id = ? AND stage = ? "
                    "ORDER BY sequence DESC LIMIT ?)",
                    (run_id, draft.stage, run_id, draft.stage, limit),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return RunEvent(sequence=sequence, **draft.model_dump())

    def list_run_events(
        self,
        run_id: str,
        *,
        stage: RunEventStage | None = None,
    ) -> list[RunEvent]:
        query = "SELECT * FROM run_events WHERE run_id = ?"
        parameters: tuple[object, ...] = (run_id,)
        if stage is not None:
            query += " AND stage = ?"
            parameters += (stage,)
        query += " ORDER BY sequence"
        with self._lock:
            rows = self._conn.execute(query, parameters).fetchall()
        return [
            RunEvent(
                sequence=row["sequence"],
                stage=row["stage"],
                at=datetime.fromisoformat(row["at"]),
                phase=row["phase"],
                message=row["message"],
                details=json.loads(row["details_json"]),
            )
            for row in rows
        ]

    @contextmanager
    def external_write_barrier(
        self,
        run_id: str,
        expected_status: RunStatus,
    ) -> Iterator[None]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._get_row_locked(run_id)
                current = RunStatus(row["status"])
                if current is not expected_status:
                    raise RunStoreConflictError(
                        f"Run {run_id} is no longer {expected_status.value}; "
                        f"current status is {current.value}"
                    )
                yield
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def initialize_reflection_review(
        self,
        run_id: str,
        review: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> Run:
        return self._write_reflection_review(
            run_id,
            review,
            expected_revision=expected_revision,
            initialize=True,
        )

    def update_reflection_review(
        self,
        run_id: str,
        review: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> Run:
        return self._write_reflection_review(
            run_id,
            review,
            expected_revision=expected_revision,
            initialize=False,
        )

    def cancel_if_safe(self, run_id: str) -> Run:
        cancellable = tuple(status.value for status in _TRANSITIONS if _TRANSITIONS[status])
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ? "
                "AND status IN (?, ?, ?, ?) "
                "AND reflecting_result IS NULL AND reflection_review IS NULL",
                (RunStatus.CANCELLED.value, run_id, *cancellable),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                row = self._get_row_locked(run_id)
                raise RunCancellationConflictError(
                    run_id,
                    RunStatus(row["status"]),
                    reflection_finalizing=(
                        row["reflecting_result"] is not None or row["reflection_review"] is not None
                    ),
                )
            self._conn.commit()
            row = self._get_row_locked(run_id)
        return self._row_to_run(row)

    def _record_stage_value(
        self,
        run_id: str,
        *,
        stage: RunStatus,
        value: Mapping[str, Any] | ReflectionResultRecord,
        result: bool,
    ) -> Run:
        if stage not in _ACTIVE_STAGES:
            raise ValueError("stage must be scoring, judging, or reflecting")
        field_name = _STAGE_FIELDS[stage][1 if result else 0]
        if field_name == "reflecting_result":
            if not isinstance(value, ReflectionResultRecord):
                raise TypeError("reflecting result must be a ReflectionResultRecord")
            encoded = ReflectionResultRecord.model_validate(
                value.model_dump(mode="json")
            ).model_dump_json()
        else:
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            encoded = _encode_json_object(value, field_name)
        with self._lock:
            row = self._get_row_locked(run_id)
            current = RunStatus(row["status"])
            if current is not stage:
                raise RunStoreConflictError(
                    f"Run {run_id} is no longer {stage.value}; current status is {current.value}"
                )
            if field_name == "reflecting_result" and row["reflection_review"] is not None:
                current_review = json.loads(row["reflection_review"])
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    "reflection evidence is immutable once review is initialized",
                    current_run_status=current,
                    current_review_status=current_review.get("status"),
                    current_revision=int(row["reflection_review_revision"]),
                )
            review_guard = (
                " AND reflection_review IS NULL" if field_name == "reflecting_result" else ""
            )
            cursor = self._conn.execute(
                f"UPDATE runs SET {field_name} = ? WHERE run_id = ? AND status = ?{review_guard}",
                (encoded, run_id, stage.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                latest = self._get_row_locked(run_id)
                latest_review = (
                    json.loads(latest["reflection_review"])
                    if latest["reflection_review"] is not None
                    else None
                )
                if field_name == "reflecting_result" and latest_review is not None:
                    raise ReflectionReviewLifecycleConflictError(
                        run_id,
                        "reflection evidence is immutable once review is initialized",
                        current_run_status=RunStatus(latest["status"]),
                        current_review_status=latest_review.get("status"),
                        current_revision=int(latest["reflection_review_revision"]),
                    )
                raise RunStoreConflictError(f"Run {run_id} changed while writing {field_name}")
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def _write_created_only_locked(
        self,
        run_id: str,
        statement: str,
        values: tuple[Any, ...],
    ) -> Run:
        row = self._get_row_locked(run_id)
        current = RunStatus(row["status"])
        if current is not RunStatus.CREATED:
            raise RunLifecycleConflictError(run_id, current)
        cursor = self._conn.execute(
            statement,
            (*values, run_id, RunStatus.CREATED.value),
        )
        if cursor.rowcount != 1:
            self._conn.rollback()
            latest = self._get_row_locked(run_id)
            raise RunLifecycleConflictError(run_id, RunStatus(latest["status"]))
        self._conn.commit()
        return self._row_to_run(self._get_row_locked(run_id))

    def _write_reflection_review(
        self,
        run_id: str,
        review: Mapping[str, Any],
        *,
        expected_revision: int,
        initialize: bool,
    ) -> Run:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        encoded = _encode_json_object(review, "reflection review")
        value = json.loads(encoded)
        new_status = value.get("status")
        if new_status not in _REFLECTION_REVIEW_STATUSES:
            allowed = ", ".join(sorted(_REFLECTION_REVIEW_STATUSES))
            raise ValueError(f"reflection review status must be one of: {allowed}")
        if initialize and new_status != "pending":
            raise ValueError("initial reflection review status must be pending")

        with self._lock:
            row = self._get_row_locked(run_id)
            run_status = RunStatus(row["status"])
            revision = int(row["reflection_review_revision"])
            current_review = (
                json.loads(row["reflection_review"])
                if row["reflection_review"] is not None
                else None
            )
            review_status = current_review.get("status") if current_review else None
            required_status = RunStatus.REFLECTING if initialize else RunStatus.COMPLETE
            if run_status is not required_status:
                message = (
                    f"run status {run_status.value} is not reviewable"
                    if initialize
                    else "review mutations require a completed run; "
                    f"current status is {run_status.value}"
                )
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    message,
                    current_run_status=run_status,
                    current_review_status=review_status,
                    current_revision=revision,
                )
            if revision != expected_revision:
                raise ReflectionReviewRevisionConflictError(
                    run_id,
                    expected_revision,
                    revision,
                    current_review,
                )
            if initialize and current_review is not None:
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    "review is already initialized",
                    current_run_status=run_status,
                    current_review_status=review_status,
                    current_revision=revision,
                )
            if not initialize and current_review is None:
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    "review is not initialized",
                    current_run_status=run_status,
                    current_review_status=None,
                    current_revision=revision,
                )
            if review_status in _RESOLVED_REFLECTION_REVIEW_STATUSES:
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    f"review is already resolved as {review_status}",
                    current_run_status=run_status,
                    current_review_status=review_status,
                    current_revision=revision,
                )
            if not initialize and review_status != "pending":
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    "stored review status is invalid",
                    current_run_status=run_status,
                    current_review_status=review_status,
                    current_revision=revision,
                )

            evidence_guard = ""
            expected_evidence: str | None = None
            if initialize:
                expected_evidence = row["reflecting_result"]
                if expected_evidence is None:
                    raise ReflectionReviewLifecycleConflictError(
                        run_id,
                        "reflection evidence is not finalized",
                        current_run_status=run_status,
                        current_review_status=review_status,
                        current_revision=revision,
                    )
                evidence = ReflectionResultRecord.model_validate_json(expected_evidence)
                selected_id = value.get("selected_candidate_id")
                if not (
                    isinstance(selected_id, str)
                    and selected_id
                    and any(
                        attempt.status == "succeeded" and attempt.candidate_id == selected_id
                        for attempt in evidence.attempts
                    )
                ):
                    raise ReflectionReviewLifecycleConflictError(
                        run_id,
                        "selected candidate is not present in finalized reflection evidence",
                        current_run_status=run_status,
                        current_review_status=review_status,
                        current_revision=revision,
                    )
                evidence_guard = " AND reflecting_result = ?"

            parameters: tuple[Any, ...] = (
                encoded,
                run_id,
                expected_revision,
                required_status.value,
            )
            if initialize:
                parameters = (*parameters, expected_evidence)
            cursor = self._conn.execute(
                "UPDATE runs SET reflection_review = ?, "
                "reflection_review_revision = reflection_review_revision + 1 "
                "WHERE run_id = ? AND reflection_review_revision = ? AND status = ?"
                f"{evidence_guard}",
                parameters,
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                latest = self._get_row_locked(run_id)
                latest_revision = int(latest["reflection_review_revision"])
                latest_review = (
                    json.loads(latest["reflection_review"])
                    if latest["reflection_review"] is not None
                    else None
                )
                if latest_revision != expected_revision:
                    raise ReflectionReviewRevisionConflictError(
                        run_id,
                        expected_revision,
                        latest_revision,
                        latest_review,
                    )
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    "run or reflection evidence changed while writing review",
                    current_run_status=RunStatus(latest["status"]),
                    current_review_status=(latest_review.get("status") if latest_review else None),
                    current_revision=latest_revision,
                )
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def _get_row_locked(self, run_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Run {run_id} not found")
        return row

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        data = dict(row)
        data["status"] = RunStatus(data["status"])
        data["auto_run"] = bool(data["auto_run"])
        for field_name in _STAGE_SUCCESS_FIELDS.values():
            data[field_name] = bool(data[field_name])
        raw_selection = data.pop("data_selection")
        raw_config = data.pop("run_config")
        raw_effective = data.pop("effective_config")
        raw_judging_plan = data.pop("judging_plan")
        raw_reflection_input = data.pop("reflection_input")
        raw_reflecting_result = data.pop("reflecting_result")
        for field_name in _JSON_FIELDS:
            if data.get(field_name) is not None:
                data[field_name] = json.loads(data[field_name])
        data["data_selection"] = (
            DataSelection(**json.loads(raw_selection)) if raw_selection is not None else None
        )
        data["run_config"] = (
            RunConfig.model_validate_json(raw_config) if raw_config is not None else None
        )
        data["effective_config"] = (
            EffectiveRunConfig.model_validate_json(raw_effective)
            if raw_effective is not None
            else None
        )
        data["judging_plan"] = (
            JudgingPlan.model_validate_json(raw_judging_plan)
            if raw_judging_plan is not None
            else None
        )
        data["reflection_input"] = (
            ReflectionInputRecord.model_validate_json(raw_reflection_input)
            if raw_reflection_input is not None
            else None
        )
        data["reflecting_result"] = (
            ReflectionResultRecord.model_validate_json(raw_reflecting_result)
            if raw_reflecting_result is not None
            else None
        )
        return Run(**data)

    def close(self) -> None:
        self._conn.close()
