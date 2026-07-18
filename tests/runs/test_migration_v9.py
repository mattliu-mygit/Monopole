from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime

import pytest

from weave_agent_signals.run_config import EffectiveRunConfig, RunConfig
from weave_agent_signals.runs.migrations import RunMigrationError, migrate_v9_to_v10

_V9_SCHEMA = """
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    auto_run INTEGER NOT NULL DEFAULT 0,
    data_selection TEXT,
    run_config TEXT,
    effective_config TEXT,
    turn_cohort TEXT,
    judging_plan TEXT,
    judging_artifacts TEXT,
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


def _artifact() -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "result": {
            "schema_version": 1,
            "chunk_id": "sha256:" + "b" * 64,
            "text": "Historical digest",
            "evidence_ids": ["trace-1"],
        },
        "audit": {
            "phase": "digest",
            "artifact_id": "digest/historical",
            "requested_model": "gpt-5.6-sol",
            "resolved_model": "gpt-5.6-sol",
            "usage": {"total_tokens": 10},
            "output_mode": "json_schema",
            "schema_name": "chunk_digest",
            "schema_fallback_reason": None,
            "transport_request_count": 1,
            "raw_output_digest": "c" * 64,
            "reused": False,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "1",
        "kind": "chunk_digest",
        "content_digest": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
        "payload": payload,
    }


def _create_v9(path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(_V9_SCHEMA)
    rows = [
        (
            "run-complete",
            "complete",
            json.dumps({"digest/historical": _artifact()}),
            json.dumps(
                {
                    "attempt_summaries": [
                        {
                            "conversation_id": "conversation-1",
                            "rubric": "judge.correctness",
                            "review_status": "complete",
                            "rating": 0.8,
                            "successful_reviewer_count": 1,
                            "attempt_count": 1,
                            "attempts": [
                                {
                                    "position": 1,
                                    "requested_model": "gpt-5.6-sol",
                                    "status": "succeeded",
                                    "score": 0.8,
                                    "rationale": "Good result",
                                    "evidence_ids": ["trace-1"],
                                    "behavioral_feedback": {
                                        "success": "Good result",
                                        "problem": None,
                                        "desired_behavior": None,
                                    },
                                    "steps": [_artifact()["payload"]["audit"]],
                                }
                            ],
                        }
                    ],
                    "events": [
                        {
                            "id": 1,
                            "at": "2026-07-17T12:01:00+00:00",
                            "phase": "judge_completed",
                            "message": "Judge completed",
                            "model": "gpt-5.6-sol",
                        }
                    ],
                    "scores_written": 1,
                }
            ),
            1,
            3,
        ),
        ("run-audit-only", "complete", None, json.dumps({"attempt_summaries": []}), 1, 0),
        ("run-failed", "failed", None, None, 0, 0),
        ("run-review", "complete", None, None, 1, 7),
    ]
    for run_id, status, artifacts, result, judging_succeeded, review_revision in rows:
        connection.execute(
            """
            INSERT INTO runs (
                run_id, status, created_at, judging_artifacts, judging_result,
                judging_succeeded, reflection_review, reflection_review_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                status,
                "2026-07-17T12:00:00+00:00",
                artifacts,
                result,
                judging_succeeded,
                json.dumps({"status": "pending"}) if run_id == "run-review" else None,
                review_revision,
            ),
        )
    connection.execute("PRAGMA user_version = 9")
    connection.commit()
    return connection


def test_migration_backs_up_and_preserves_historical_runs(tmp_path) -> None:
    path = tmp_path / "runs.db"
    connection = _create_v9(path)

    backup = migrate_v9_to_v10(
        connection,
        path,
        lambda: datetime(2026, 7, 17, 12, 30, tzinfo=UTC),
    )

    assert backup.exists()
    assert sqlite3.connect(backup).execute("PRAGMA user_version").fetchone()[0] == 9
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 10
    assert [row[0] for row in connection.execute("SELECT run_id FROM runs ORDER BY run_id")] == [
        "run-audit-only",
        "run-complete",
        "run-failed",
        "run-review",
    ]
    columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
    assert "judging_artifacts" not in columns
    call = connection.execute(
        "SELECT reusable, result_json FROM judge_calls WHERE run_id = 'run-complete'"
    ).fetchone()
    assert tuple(call) == (0, json.dumps(_artifact()["payload"]["result"], sort_keys=True))
    event = connection.execute(
        "SELECT sequence, stage, phase FROM run_events WHERE run_id = 'run-complete'"
    ).fetchone()
    assert tuple(event) == (1, "judging", "judge_completed")
    review = connection.execute(
        "SELECT reflection_review_revision FROM runs WHERE run_id = 'run-review'"
    ).fetchone()
    assert review[0] == 7
    connection.close()


def test_migration_normalizes_legacy_run_configuration(tmp_path) -> None:
    path = tmp_path / "runs.db"
    connection = _create_v9(path)
    requested = {
        "model_catalog_version": "models-v1",
        "rubric_catalog_version": "rubrics-v1",
        "judge_backend": "cli",
        "judge_models": ["gpt-old"],
        "challenge_judge_models": ["gpt-old"],
        "proposal_model": "gpt-old",
        "proposal_evaluator_model": "gpt-old",
        "rubrics": ["judge.verification"],
        "candidate_budget": 1,
        "force": False,
    }
    descriptor = {
        "id": "gpt-old",
        "label": "Historical GPT",
        "family": "openai",
        "backend": "cli",
        "supported_roles": ["proposal_writer", "judge", "proposal_evaluator"],
        "max_input_tokens": 128000,
        "token_counter": "utf8_bytes_div_3",
    }
    effective = {
        "schema_version": "4",
        "pipeline_version": "7",
        "model_catalog_version": "models-v1",
        "rubric_catalog_version": "rubrics-v1",
        "judge_backend": "cli",
        "models": {
            "proposal_writer": dict(descriptor),
            "judges": [{**descriptor, "role": "judge", "position": 1}],
            "challenge_judges": [{**descriptor, "role": "judge", "position": 1}],
            "proposal_evaluator": dict(descriptor),
        },
        "rubrics": [
            {
                "id": "judge.verification",
                "label": "Verification",
                "evaluation_unit": "session",
                "version": "v1",
                "content_digest": "sha256:rubric",
                "pass_threshold": 0.5,
            }
        ],
        "selection_warnings": [],
        "judging_context": {},
        "candidate_budget": 1,
        "force": False,
    }
    connection.execute(
        "UPDATE runs SET run_config = ?, effective_config = ? WHERE run_id = 'run-complete'",
        (json.dumps(requested), json.dumps(effective)),
    )
    connection.commit()

    migrate_v9_to_v10(
        connection,
        path,
        lambda: datetime(2026, 7, 17, 12, 30, tzinfo=UTC),
    )

    row = connection.execute(
        "SELECT run_config, effective_config FROM runs WHERE run_id = 'run-complete'"
    ).fetchone()
    assert RunConfig.model_validate_json(row[0]).judge_models == ("gpt-old",)
    migrated = EffectiveRunConfig.model_validate_json(row[1])
    assert migrated.schema_version == "5"
    assert migrated.models.judges[0].provider == "cli"
    assert migrated.models.judges[0].provider_model == "gpt-old"
    connection.close()


def test_migration_rolls_back_every_schema_change_on_bad_row(tmp_path) -> None:
    path = tmp_path / "runs.db"
    connection = _create_v9(path)
    connection.execute(
        "UPDATE runs SET judging_artifacts = ? WHERE run_id = 'run-complete'",
        (json.dumps({"digest/bad": {"not": "an artifact"}}),),
    )
    connection.commit()

    with pytest.raises(RunMigrationError) as error:
        migrate_v9_to_v10(connection, path, lambda: datetime(2026, 7, 17, tzinfo=UTC))

    assert "backup" in str(error.value).lower()
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 9
    assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 4
    assert (
        connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'judge_calls'"
        ).fetchone()[0]
        == 0
    )
    connection.close()
