"""One-way preservation migration for the pre-release epoch-9 run database."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from weave_agent_signals.judges.plan import canonical_plan_from_epoch9
from weave_agent_signals.judges.records import JudgeCallAudit, JudgeCallRecord
from weave_agent_signals.run_config import EffectiveRunConfig, RunConfig
from weave_agent_signals.runs.events import RunEventDraft, sanitize_event
from weave_agent_signals.runs.reflection_records import (
    ReflectionInputRecord,
    ReflectionResultRecord,
)

log = logging.getLogger(__name__)


class RunMigrationError(RuntimeError):
    """The database remains at its source epoch and a backup is available."""


_RUNS_V10 = """
CREATE TABLE runs (
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

_JUDGE_CALLS = """
CREATE TABLE judge_calls (
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

_RUN_EVENTS = """
CREATE TABLE run_events (
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

_RUN_COLUMNS = (
    "run_id",
    "status",
    "created_at",
    "auto_run",
    "data_selection",
    "run_config",
    "effective_config",
    "turn_cohort",
    "judging_plan",
    "reflection_input",
    "scoring_progress",
    "scoring_result",
    "scoring_succeeded",
    "judging_progress",
    "judging_result",
    "judging_succeeded",
    "reflecting_progress",
    "reflecting_result",
    "reflecting_succeeded",
    "reflection_review",
    "reflection_review_revision",
    "error",
)


def historical_request_id(identity: str) -> str:
    return "sha256:" + hashlib.sha256(f"epoch9:{identity}".encode()).hexdigest()


def _load_object(value: str | None, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must be an object")
    return loaded


def _attempt_contexts(result: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    summaries = result.get("attempt_summaries", []) if result is not None else []
    if not isinstance(summaries, list):
        raise ValueError("attempt_summaries must be a list")
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise ValueError("attempt summary must be an object")
        for attempt in summary.get("attempts", []):
            if not isinstance(attempt, Mapping):
                raise ValueError("review attempt must be an object")
            context = {
                "conversation_id": summary.get("conversation_id"),
                "rubric_id": summary.get("rubric"),
                "reviewer_position": attempt.get("position"),
                "requested_model_id": attempt.get("requested_model"),
            }
            for step in attempt.get("steps", []):
                if not isinstance(step, Mapping) or not isinstance(step.get("artifact_id"), str):
                    raise ValueError("attempt step must contain artifact_id")
                contexts.setdefault(step["artifact_id"], context)
    return contexts


def _audit(value: Mapping[str, Any]) -> JudgeCallAudit:
    return JudgeCallAudit(
        resolved_model=value.get("resolved_model"),
        usage=dict(value.get("usage") or {}),
        output_mode=value.get("output_mode"),
        schema_name=value.get("schema_name") or "historical_unknown",
        schema_fallback_reason=value.get("schema_fallback_reason"),
        transport_request_count=value.get("transport_request_count") or 0,
        raw_output_digest=value.get("raw_output_digest"),
    )


def _historical_call(
    *,
    identity: str,
    audit: Mapping[str, Any],
    context: Mapping[str, Any],
    result: dict[str, Any] | None,
    created_at: str,
) -> JudgeCallRecord:
    phase = audit.get("phase")
    if phase not in {"digest", "window", "merge"}:
        raise ValueError("historical call phase is invalid")
    conversation_id = context.get("conversation_id")
    position = context.get("reviewer_position")
    requested_model = context.get("requested_model_id") or audit.get("requested_model")
    rubric_id = None if phase == "digest" else context.get("rubric_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        conversation_id = "historical-unattributed"
    if type(position) is not int or not 1 <= position <= 3:
        position = 1
    if not isinstance(requested_model, str) or not requested_model:
        requested_model = "historical-unknown"
    if phase != "digest" and (not isinstance(rubric_id, str) or not rubric_id):
        rubric_id = "historical-unknown"
    return JudgeCallRecord(
        request_id=historical_request_id(identity),
        phase=phase,
        conversation_id=conversation_id,
        reviewer_position=position,
        requested_model_id=requested_model,
        rubric_id=rubric_id,
        status="succeeded",
        reusable=False,
        result=result,
        audit=_audit(audit),
        created_at=datetime.fromisoformat(created_at),
    )


def _artifact_calls(
    row: sqlite3.Row,
    result: dict[str, Any] | None,
) -> list[JudgeCallRecord]:
    contexts = _attempt_contexts(result)
    calls: dict[str, JudgeCallRecord] = {}
    artifacts = _load_object(row["judging_artifacts"], "judging_artifacts") or {}
    for artifact_id, artifact in artifacts.items():
        if not isinstance(artifact_id, str) or not isinstance(artifact, Mapping):
            raise ValueError("judging artifact map is invalid")
        if set(artifact) != {"schema_version", "kind", "content_digest", "payload"}:
            raise ValueError("judging artifact is invalid")
        payload = artifact["payload"]
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema_version", "audit", "result"}
            or payload.get("schema_version") != 1
        ):
            raise ValueError("judging artifact payload is invalid")
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        expected = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        if artifact["schema_version"] != "1" or artifact["content_digest"] != expected:
            raise ValueError("judging artifact identity is invalid")
        audit = payload["audit"]
        call_result = payload["result"]
        if not isinstance(audit, Mapping) or not isinstance(call_result, dict):
            raise ValueError("judging artifact evidence is invalid")
        call = _historical_call(
            identity=artifact_id,
            audit=audit,
            context=contexts.get(artifact_id, {}),
            result=call_result,
            created_at=row["created_at"],
        )
        calls[call.request_id] = call
    for artifact_id, context in contexts.items():
        request_id = historical_request_id(artifact_id)
        if request_id in calls:
            continue
        step = next(
            step
            for summary in result.get("attempt_summaries", [])
            for attempt in summary.get("attempts", [])
            for step in attempt.get("steps", [])
            if step.get("artifact_id") == artifact_id
        )
        calls[request_id] = _historical_call(
            identity=artifact_id,
            audit=step,
            context=context,
            result=None,
            created_at=row["created_at"],
        )
    return list(calls.values())


def _event_rows(row: sqlite3.Row) -> list[RunEventDraft]:
    events: list[RunEventDraft] = []
    for stage, result_name, progress_name in (
        ("judging", "judging_result", "judging_progress"),
        ("reflecting", "reflecting_result", "reflecting_progress"),
    ):
        result = _load_object(row[result_name], result_name)
        progress = _load_object(row[progress_name], progress_name)
        source = (
            result
            if isinstance(result, dict) and isinstance(result.get("events"), list)
            else progress
        )
        raw_events = source.get("events", []) if source is not None else []
        for event in raw_events:
            if not isinstance(event, Mapping):
                raise ValueError("historical event must be an object")
            details = {
                key: value
                for key, value in event.items()
                if key not in {"id", "at", "phase", "message"}
            }
            events.append(
                sanitize_event(
                    stage,
                    event.get("phase"),
                    event.get("message"),
                    details,
                    datetime.fromisoformat(event.get("at")),
                )
            )
    return sorted(events, key=lambda event: event.at)


def _without_events(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {key: item for key, item in value.items() if key != "events"}


def _run_config(value: str | None) -> str | None:
    config = _load_object(value, "run_config")
    if config is None:
        return None
    config.pop("judge_backend", None)
    return RunConfig.model_validate(config).model_dump_json()


def _effective_config(value: str | None) -> str | None:
    config = _load_object(value, "effective_config")
    if config is None:
        return None
    config.pop("judge_backend", None)
    config["schema_version"] = "5"
    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("effective_config models must be an object")
    descriptors = [models.get("proposal_writer"), models.get("proposal_evaluator")]
    descriptors.extend(models.get("judges", []))
    descriptors.extend(models.get("challenge_judges", []))
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("effective_config model descriptor must be an object")
        backend = descriptor.pop("backend", None)
        descriptor.setdefault("provider", backend or "historical")
        descriptor.setdefault("provider_model", descriptor.get("id"))
    return EffectiveRunConfig.model_validate(config).model_dump_json()


def _insert_call(connection: sqlite3.Connection, run_id: str, call: JudgeCallRecord) -> None:
    connection.execute(
        """
        INSERT INTO judge_calls (
            run_id, request_id, phase, conversation_id, reviewer_position,
            requested_model_id, rubric_id, status, reusable, result_json,
            audit_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            call.request_id,
            call.phase,
            call.conversation_id,
            call.reviewer_position,
            call.requested_model_id,
            call.rubric_id,
            call.status,
            int(call.reusable),
            json.dumps(call.result, sort_keys=True) if call.result is not None else None,
            call.audit.model_dump_json(),
            call.created_at.isoformat(),
        ),
    )


def _backup(connection: sqlite3.Connection, path: Path, now: datetime) -> Path:
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.name}.v9.{stamp}.backup")
    target = sqlite3.connect(backup_path)
    try:
        connection.backup(target)
    finally:
        target.close()
    return backup_path


def migrate_v9_to_v10(
    connection: sqlite3.Connection,
    database_path: str | Path,
    clock: Callable[[], datetime],
) -> Path:
    """Back up and transactionally convert every epoch-9 run to epoch 10."""

    path = Path(database_path)
    backup_path = _backup(connection, path, clock())
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != 9:
            raise ValueError("source database is not epoch 9")
        connection.row_factory = sqlite3.Row
        source_rows = connection.execute(
            "SELECT * FROM runs ORDER BY created_at, run_id"
        ).fetchall()
        expected = {
            row["run_id"]: (
                row["status"],
                bool(row["judging_succeeded"]),
                bool(row["reflecting_succeeded"]),
                row["reflection_review_revision"],
            )
            for row in source_rows
        }
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("ALTER TABLE runs RENAME TO runs_v9")
        connection.execute(_RUNS_V10)
        connection.execute(_JUDGE_CALLS)
        connection.execute(_RUN_EVENTS)
        for row in source_rows:
            data = dict(row)
            data["run_config"] = _run_config(row["run_config"])
            data["effective_config"] = _effective_config(row["effective_config"])
            judging_result = _load_object(row["judging_result"], "judging_result")
            events = _event_rows(row)
            calls = _artifact_calls(row, judging_result)
            plan = _load_object(row["judging_plan"], "judging_plan")
            if plan is not None:
                data["judging_plan"] = canonical_plan_from_epoch9(plan).model_dump_json()
            reflection_input = _load_object(row["reflection_input"], "reflection_input")
            if reflection_input is not None:
                data["reflection_input"] = ReflectionInputRecord.from_epoch9(
                    reflection_input
                ).model_dump_json()
            reflecting_result = _load_object(row["reflecting_result"], "reflecting_result")
            if reflecting_result is not None:
                data["reflecting_result"] = ReflectionResultRecord.from_epoch9(
                    reflecting_result
                ).model_dump_json()
            data["judging_result"] = (
                json.dumps(_without_events(judging_result), sort_keys=True)
                if judging_result is not None
                else None
            )
            for name in ("judging_progress", "reflecting_progress"):
                value = _load_object(row[name], name)
                data[name] = (
                    json.dumps(_without_events(value), sort_keys=True)
                    if value is not None
                    else None
                )
            placeholders = ", ".join("?" for _ in _RUN_COLUMNS)
            connection.execute(
                f"INSERT INTO runs ({', '.join(_RUN_COLUMNS)}) VALUES ({placeholders})",
                tuple(data[column] for column in _RUN_COLUMNS),
            )
            for call in calls:
                _insert_call(connection, row["run_id"], call)
            for sequence, event in enumerate(events, start=1):
                connection.execute(
                    "INSERT INTO run_events "
                    "(run_id, sequence, stage, at, phase, message, details_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["run_id"],
                        sequence,
                        event.stage,
                        event.at.isoformat(),
                        event.phase,
                        event.message,
                        json.dumps(event.details, sort_keys=True),
                    ),
                )
        connection.execute("DROP TABLE runs_v9")
        actual = {
            row["run_id"]: (
                row["status"],
                bool(row["judging_succeeded"]),
                bool(row["reflecting_succeeded"]),
                row["reflection_review_revision"],
            )
            for row in connection.execute("SELECT * FROM runs")
        }
        if actual != expected:
            raise ValueError("migrated run invariants do not match")
        connection.execute("PRAGMA user_version = 10")
        connection.commit()
    except BaseException as error:
        connection.rollback()
        log.warning("Epoch-9 run migration failed: error_type=%s", type(error).__name__)
        raise RunMigrationError(f"Run migration failed; backup: {backup_path}") from None
    return backup_path


__all__ = ["RunMigrationError", "historical_request_id", "migrate_v9_to_v10"]
