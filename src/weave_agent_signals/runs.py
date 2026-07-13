"""Run model and SQLite-backed store for the evaluation run lifecycle.

An evaluation run moves through a fixed pipeline: created -> scoring ->
judging -> reflecting -> complete, with a failure exit from any non-terminal
state. See docs/plans/2026-07-12-evaluation-runs.md.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

_DEFAULT_DB_DIR = Path.home() / ".weave-agent-signals"


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


# Valid forward transitions. FAILED is reachable from every non-terminal
# state; COMPLETE and FAILED are terminal (no transitions out).
_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.CREATED: {RunStatus.SCORING, RunStatus.FAILED},
    RunStatus.SCORING: {RunStatus.JUDGING, RunStatus.FAILED},
    RunStatus.JUDGING: {RunStatus.REFLECTING, RunStatus.FAILED},
    RunStatus.REFLECTING: {RunStatus.COMPLETE, RunStatus.FAILED},
    RunStatus.COMPLETE: set(),
    RunStatus.FAILED: set(),
}


@dataclass
class DataSelection:
    since: str | None = None
    until: str | None = None
    session_ids: list[str] = field(default_factory=list)
    excluded_session_ids: list[str] = field(default_factory=list)


@dataclass
class Run:
    run_id: str
    status: RunStatus
    created_at: str
    config_version: str | None = None
    auto_run: bool = False
    data_selection: DataSelection | None = None
    scoring_progress: dict | None = None
    scoring_result: dict | None = None
    judging_progress: dict | None = None
    judging_result: dict | None = None
    reflecting_progress: dict | None = None
    reflecting_result: dict | None = None
    error: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'created',
    created_at TEXT NOT NULL,
    config_version TEXT,
    auto_run INTEGER NOT NULL DEFAULT 0,
    data_selection TEXT,
    scoring_progress TEXT,
    scoring_result TEXT,
    judging_progress TEXT,
    judging_result TEXT,
    reflecting_progress TEXT,
    reflecting_result TEXT,
    error TEXT
)
"""

_JSON_FIELDS = {
    "data_selection",
    "scoring_progress",
    "scoring_result",
    "judging_progress",
    "judging_result",
    "reflecting_progress",
    "reflecting_result",
}

# Fields that RunStore.update() is allowed to set. data_selection has its own
# dedicated setter (set_selection); run_id/created_at/config_version/auto_run
# are fixed at creation time.
_UPDATABLE_FIELDS = (_JSON_FIELDS - {"data_selection"}) | {"status", "error"}


class RunStore:
    """SQLite-backed store for evaluation run lifecycle tracking."""

    def __init__(self, db_path: str | Path | None = None):
        path = str(db_path or _default_db_path())
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def create(self, *, config_version: str | None = None, auto_run: bool = False) -> Run:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, status, created_at, config_version, auto_run)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, RunStatus.CREATED.value, created_at, config_version, int(auto_run)),
            )
            self._conn.commit()
        return Run(
            run_id=run_id,
            status=RunStatus.CREATED,
            created_at=created_at,
            config_version=config_version,
            auto_run=auto_run,
        )

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row is not None else None

    def list(self, limit: int = 50) -> list[Run]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def set_selection(
        self,
        run_id: str,
        *,
        since: str | None = None,
        until: str | None = None,
        session_ids: list[str] | None = None,
        excluded_session_ids: list[str] | None = None,
    ) -> None:
        selection = DataSelection(
            since=since,
            until=until,
            session_ids=session_ids or [],
            excluded_session_ids=excluded_session_ids or [],
        )
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise ValueError(f"Run {run_id} not found")
            self._conn.execute(
                "UPDATE runs SET data_selection = ? WHERE run_id = ?",
                (json.dumps(asdict(selection)), run_id),
            )
            self._conn.commit()

    def update(self, run_id: str, **fields: Any) -> None:
        unknown = set(fields) - _UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Unknown field(s) for update: {', '.join(sorted(unknown))}")

        with self._lock:
            row = self._conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise ValueError(f"Run {run_id} not found")
            if "status" in fields:
                fields["status"] = self._check_transition_locked(run_id, fields["status"])

            sets = [f"{key} = ?" for key in fields]
            values: list[Any] = [
                json.dumps(value) if key in _JSON_FIELDS and value is not None else value
                for key, value in fields.items()
            ]
            values.append(run_id)
            self._conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ?", values)
            self._conn.commit()

    def _check_transition_locked(self, run_id: str, status: RunStatus | str) -> str:
        """Validate a status transition. Caller must already hold self._lock."""
        new_status = RunStatus(status)
        row = self._conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Run {run_id} not found")
        current_status = RunStatus(row["status"])
        if new_status not in _TRANSITIONS[current_status]:
            raise ValueError(f"Invalid transition: {current_status.value} -> {new_status.value}")
        return new_status.value

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        data = dict(row)
        data["status"] = RunStatus(data["status"])
        data["auto_run"] = bool(data["auto_run"])
        for key in _JSON_FIELDS:
            if data.get(key) is not None:
                data[key] = json.loads(data[key])
        if data.get("data_selection") is not None:
            data["data_selection"] = DataSelection(**data["data_selection"])
        return Run(**data)

    def close(self) -> None:
        self._conn.close()
