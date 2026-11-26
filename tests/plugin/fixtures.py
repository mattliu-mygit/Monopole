"""Runs written straight into a run database, in the shapes the stages store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from weave_agent_signals.runs.store import RunStore

JUDGES = [
    {"id": "wandb:DeepSeek-V4.1-Flash", "label": "DeepSeek V4.1 Flash", "provider": "wandb"},
    {"id": "codex:gpt-5.6-sol", "label": "GPT-5.6 Sol", "provider": "codex"},
]
RUBRICS = [
    {"id": "judge.verification", "label": "Verification Discipline"},
    {"id": "judge.session_outcome", "label": "Session Outcome Quality"},
]
SESSION = "01a1ad64-7d00-75f7-b10f-e2e0471fdf58"


def new_db(path: Path) -> Path:
    RunStore(path).close()
    return path


def add_run(path: Path, run_id: str, status: str, created_at: str, **columns: object) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO runs (run_id, status, created_at) VALUES (?, ?, ?)",
        (run_id, status, created_at),
    )
    for column, value in columns.items():
        stored = value if isinstance(value, (str, int)) or value is None else json.dumps(value)
        connection.execute(f"UPDATE runs SET {column} = ? WHERE run_id = ?", (stored, run_id))
    connection.commit()
    connection.close()


def started(prompt: str | None = "Fix the flaky login test", turns: int = 12) -> dict:
    """The columns of a run whose setup pinned one session and whose scoring finished."""

    return {
        "data_selection": {
            "session_ids": [SESSION],
            "since": None,
            "until": None,
            "timezone": None,
        },
        "effective_config": {
            "models": {
                "judges": JUDGES,
                "proposal_writer": {"label": "GLM 5.3"},
                "proposal_evaluator": {"label": "DeepSeek V4.1 Flash"},
            },
            "rubrics": RUBRICS,
        },
        "turn_cohort": {
            "session_count": 1,
            "turn_count": turns,
            "sessions": [{"conversation_id": SESSION, "turn_count": turns}],
        },
        "scoring_result": {
            "status_message": "Scoring complete: 12 scores written.",
            "turn_details": [{"user_input": prompt}] if prompt is not None else [],
        },
        "scoring_succeeded": 1,
    }


def judged(ratings: tuple[float, float] = (0.75, 0.5)) -> dict:
    return {
        "judging_result": {
            "status_message": "Judging complete",
            "attempt_summaries": [
                {
                    "rubric": rubric["id"],
                    "conversation_id": SESSION,
                    "rating": rating,
                    "review_status": "complete",
                }
                for rubric, rating in zip(RUBRICS, ratings, strict=True)
            ],
        },
        "judging_succeeded": 1,
    }


def failed_judging() -> dict:
    message = (
        "Session Outcome Quality failed: DeepSeek V4.1 Flash failed ValueError after 28 "
        'provider requests: unknown evidence IDs: ["e224"]'
    )
    return {
        "judging_result": {
            "status_message": message,
            "attempt_summaries": [
                {
                    "rubric": "judge.session_outcome",
                    "conversation_id": SESSION,
                    "rating": None,
                    "review_status": "failed",
                }
            ],
            "failure_details": [{"rubric": "judge.session_outcome", "message": message}],
        },
        "error": "1 rubric(s) failed and 0 were not evaluable",
    }


def reflected(*, challenge: dict | None, review: str | None = "pending") -> dict:
    """A reflection whose best of two candidates scored 0.35 against the current 0.15."""

    columns: dict = {
        "reflecting_result": {
            "evaluations": [
                {
                    "target_id": "baseline",
                    "score": 0.15,
                    "rationale": "The current bundle is long.",
                },
                {
                    "target_id": "candidate-a",
                    "score": 0.35,
                    "rationale": "Targets outcome.command and outcome.test. It also trims prose.",
                },
                {"target_id": "candidate-b", "score": 0.2, "rationale": "Small gain."},
            ],
            "recommended_candidate_id": None,
            "provisional_candidate_id": "candidate-a",
            "baseline_won": True,
            "reason": None,
            "challenge": challenge,
        },
        "reflecting_succeeded": 1,
    }
    if review is not None:
        columns["reflection_review"] = {"status": review, "draft": None}
    return columns
