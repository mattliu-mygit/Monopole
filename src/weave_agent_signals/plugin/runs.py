"""Runs as the plugin's page and Prompting Time's status list show them.

Each read opens the run database the server writes and projects only the fields a card needs
with SQLite's JSON functions, because a run's full row holds megabytes of judging evidence and
`status` runs again after every change. Nothing here writes to the database.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from weave_agent_signals.routes import review_state, run_response
from weave_agent_signals.runs.store import RUN_DB_SCHEMA_VERSION, RunDatabaseSchemaError, RunStore

_STAGES = ("scoring", "judging", "reflecting")
_ACTIVE = frozenset(_STAGES)
# The stage Continue starts from a run that finished `status` and waits, because it doesn't
# continue automatically. Reflection always runs on to complete.
_CONTINUES_TO = {"scoring": "judging", "judging": "reflecting"}
_WAITING_LABELS = {"judging": "Waiting to judge", "reflecting": "Waiting to reflect"}
# Why a waiting run needs the user, as Prompting Time's Needs you lists it.
WAITING_REASONS = {
    "judging": "Ready to judge — Continue to start",
    "reflecting": "Ready to reflect — Continue to start",
}
TITLE_LIMIT = 120
DETAIL_LIMIT = 240

_REVIEW_LABELS = {
    "review-needed": "Review needed",
    "promoted": "Promoted",
    "partial": "Partially applied",
    "dismissed": "Dismissed",
    "no-change": "No change",
    "no-valid-proposal": "No valid proposal",
}


def _stage_messages(stage: str) -> str:
    return (
        f"json_extract({stage}_result, '$.status_message') AS {stage}_result_message, "
        f"json_extract({stage}_progress, '$.status_message') AS {stage}_progress_message"
    )


_CARD_COLUMNS = f"""
    run_id, status, created_at, error, auto_run,
    data_selection IS NOT NULL AND run_config IS NOT NULL AS set_up,
    scoring_succeeded, judging_succeeded, reflecting_succeeded,
    coalesce(
        (SELECT max(at) FROM run_events WHERE run_events.run_id = runs.run_id), created_at
    ) AS updated_at,
    json_extract(scoring_result, '$.turn_details[0].user_input') AS first_prompt,
    coalesce(
        json_extract(turn_cohort, '$.sessions[0].conversation_id'),
        json_extract(data_selection, '$.session_ids[0]')
    ) AS first_session,
    coalesce(
        json_extract(turn_cohort, '$.session_count'),
        json_array_length(data_selection, '$.session_ids')
    ) AS session_count,
    json_extract(turn_cohort, '$.turn_count') AS turns,
    json((SELECT json_group_array(json_object(
        'id', json_extract(value, '$.id'),
        'label', json_extract(value, '$.label'),
        'provider', json_extract(value, '$.provider')))
        FROM json_each(effective_config, '$.models.judges'))) AS judges,
    json((SELECT json_group_object(json_extract(value, '$.id'), json_extract(value, '$.label'))
        FROM json_each(effective_config, '$.rubrics'))) AS rubric_labels,
    (SELECT avg(json_extract(value, '$.rating'))
        FROM json_each(judging_result, '$.attempt_summaries')) AS mean_score,
    (SELECT count(json_extract(value, '$.rating'))
        FROM json_each(judging_result, '$.attempt_summaries')) AS rated,
    json_extract(judging_result, '$.failure_details[0].rubric') AS failed_rubric,
    {", ".join(_stage_messages(stage) for stage in _STAGES)},
    json_extract(reflection_review, '$.status') AS review_status,
    reflecting_result IS NOT NULL AS reflected,
    json_extract(reflecting_result, '$.baseline_won') AS baseline_won,
    json_extract(reflecting_result, '$.reason') AS reflection_reason,
    json_extract(reflecting_result, '$.recommended_candidate_id') AS recommended_candidate_id,
    json_extract(reflecting_result, '$.provisional_candidate_id') AS provisional_candidate_id,
    json((SELECT json_group_array(json_object(
        'target_id', json_extract(value, '$.target_id'),
        'score', json_extract(value, '$.score'),
        'rationale', json_extract(value, '$.rationale')))
        FROM json_each(reflecting_result, '$.evaluations'))) AS evaluations,
    json_extract(reflecting_result, '$.challenge.status') AS challenge_status,
    json_extract(reflecting_result, '$.challenge.winner') AS challenge_winner,
    json_extract(reflecting_result, '$.challenge.reason') AS challenge_reason
"""


@contextmanager
def read_runs(path: Path) -> Iterator[sqlite3.Connection | None]:
    """A connection to the run database for reads, or None when no server has created it yet."""

    if not path.exists():
        yield None
        return
    connection = sqlite3.connect(path)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            yield None
            return
        if version != RUN_DB_SCHEMA_VERSION:
            raise RunDatabaseSchemaError(
                f"unsupported run database schema: {version}; this Monopole reads schema "
                f"{RUN_DB_SCHEMA_VERSION} ({path})"
            )
        yield connection
    finally:
        connection.close()


def one_line(text: str | None, limit: int) -> str | None:
    """`text` with its whitespace collapsed to single spaces, cut to `limit` characters."""

    if text is None:
        return None
    flat = " ".join(text.split())
    if not flat:
        return None
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _first_sentence(text: str | None) -> str | None:
    line = one_line(text, 10_000)
    if line is None:
        return None
    match = re.match(r"(.+?[.!?])(\s|$)", line)
    return one_line(match.group(1) if match else line, DETAIL_LIMIT)


def _failed_stage(row: sqlite3.Row) -> str:
    return next(
        (stage for stage in _STAGES if not row[f"{stage}_succeeded"]),
        "reflecting",
    )


def _failure(row: sqlite3.Row, rubric_labels: dict[str, str]) -> dict[str, Any]:
    stage = _failed_stage(row)
    cause = row[f"{stage}_result_message"] or row[f"{stage}_progress_message"] or row["error"] or ""
    rubric = row["failed_rubric"] if stage == "judging" else None
    return {
        "stage": stage,
        "stopped_at": rubric_labels.get(rubric, rubric) if rubric else stage.capitalize(),
        "cause": one_line(cause, 2_000) or "The run stopped without a message.",
    }


def _proposal(row: sqlite3.Row) -> dict[str, Any] | None:
    """How the best proposal compared with the current instructions, by the evaluator's
    predicted score, and whether the A/B check in the sandbox confirmed it."""

    if not row["reflected"]:
        return None
    evaluations = json.loads(row["evaluations"] or "[]")
    baseline = next((e for e in evaluations if e["target_id"] == "baseline"), None)
    candidates = [e for e in evaluations if e["target_id"] != "baseline"]
    chosen = row["recommended_candidate_id"] or row["provisional_candidate_id"]
    best = next((e for e in candidates if e["target_id"] == chosen), None) or max(
        candidates, key=lambda e: e["score"], default=None
    )
    current = baseline["score"] if baseline else None
    proposed = best["score"] if best else None
    check = None
    if row["challenge_status"]:
        check = {
            "status": row["challenge_status"],
            "winner": row["challenge_winner"],
            "reason": one_line(row["challenge_reason"], DETAIL_LIMIT),
        }
    note_source = best["rationale"] if best else row["reflection_reason"]
    return {
        "current": current,
        "proposed": proposed,
        "change": (
            round(proposed - current, 4) if current is not None and proposed is not None else None
        ),
        "candidates": len(candidates),
        "check": check,
        "recommended": row["recommended_candidate_id"] is not None,
        "note": _first_sentence(note_source),
    }


def _title(row: sqlite3.Row) -> str:
    prompt = one_line(row["first_prompt"], TITLE_LIMIT)
    if prompt:
        return prompt
    if row["first_session"]:
        return f"Session {row['first_session'][:12]}"
    return f"Run {row['run_id']}"


def _next_stage(row: sqlite3.Row) -> str | None:
    """The stage Continue starts, as the server's advance route would: scoring for a draft that
    is set up, and the next stage for a run that finished scoring or judging without Continue
    automatically. None when the run isn't waiting."""

    status = row["status"]
    if status == "created":
        return "scoring" if row["set_up"] else None
    if status in _CONTINUES_TO and row[f"{status}_succeeded"] and not row["auto_run"]:
        return _CONTINUES_TO[status]
    return None


def card(row: sqlite3.Row) -> dict[str, Any]:
    """One run as the page's card and Prompting Time's item read it."""

    status = row["status"]
    rubric_labels = json.loads(row["rubric_labels"] or "{}")
    baseline_won = None if row["baseline_won"] is None else bool(row["baseline_won"])
    review = review_state(status, row["review_status"], row["reflection_reason"], baseline_won)
    failure = _failure(row, rubric_labels) if status == "failed" else None
    progress = None
    next_stage = _next_stage(row)
    if status == "created":
        label, dot, group = "Draft", "idle", None
    elif next_stage:
        label, dot, group = _WAITING_LABELS[next_stage], "attention", "waiting"
    elif status in _ACTIVE:
        label, dot, group = status.capitalize(), "running", "running"
        progress = one_line(row[f"{status}_progress_message"], DETAIL_LIMIT)
    elif status == "failed":
        label, dot, group = f"Failed at {failure['stage']}", "error", "failed"
    elif status == "cancelled":
        label, dot, group = "Cancelled", "idle", "done"
    elif review == "review-needed":
        label, dot, group = "Review needed", "attention", "review"
    else:
        label, dot, group = _REVIEW_LABELS.get(review, "Complete"), "success", "done"
    return {
        "run_id": row["run_id"],
        "status": status,
        "label": label,
        "dot": dot,
        "group": group,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "title": _title(row),
        "session_count": row["session_count"],
        "turns": row["turns"],
        "judges": json.loads(row["judges"] or "[]"),
        "rated": row["rated"] or 0,
        "mean_score": round(row["mean_score"], 4) if row["mean_score"] is not None else None,
        "progress": progress,
        "failure": failure,
        "review_state": review,
        "proposal": _proposal(row),
        "deletable": status not in _ACTIVE,
        "next_stage": next_stage,
    }


def list_cards(path: Path, limit: int, *, drafts: bool = True) -> tuple[list[dict[str, Any]], bool]:
    """The newest `limit` runs as cards, and whether older runs were left out."""

    with read_runs(path) as connection:
        if connection is None:
            return [], False
        where = "" if drafts else "WHERE status != 'created'"
        rows = connection.execute(
            f"SELECT {_CARD_COLUMNS} FROM runs {where} ORDER BY created_at DESC LIMIT ?",
            (limit + 1,),
        ).fetchall()
    return [card(row) for row in rows[:limit]], len(rows) > limit


def stage_running(path: Path) -> bool:
    """Whether any run's stage is executing now, not waiting for Continue."""

    with read_runs(path) as connection:
        if connection is None:
            return False
        rows = connection.execute(
            f"SELECT {_CARD_COLUMNS} FROM runs WHERE status IN ({', '.join('?' * len(_STAGES))})",
            _STAGES,
        ).fetchall()
    return any(card(row)["group"] == "running" for row in rows)


def get_card(path: Path, run_id: str) -> dict[str, Any] | None:
    with read_runs(path) as connection:
        if connection is None:
            return None
        row = connection.execute(
            f"SELECT {_CARD_COLUMNS} FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return card(row) if row is not None else None


_DETAIL_COLUMNS = f"""
    auto_run,
    json((SELECT json_group_object(json_extract(value, '$.id'), json_extract(value, '$.label'))
        FROM json_each(effective_config, '$.rubrics'))) AS rubric_labels,
    json_extract(effective_config, '$.models.proposal_writer.label') AS proposal_writer,
    json_extract(effective_config, '$.models.proposal_evaluator.label') AS proposal_evaluator,
    json((SELECT json_group_array(json_object(
        'rubric', json_extract(value, '$.rubric'),
        'conversation_id', json_extract(value, '$.conversation_id'),
        'rating', json_extract(value, '$.rating'),
        'review_status', json_extract(value, '$.review_status')))
        FROM json_each(judging_result, '$.attempt_summaries'))) AS rubric_reviews,
    status, scoring_succeeded, judging_succeeded, reflecting_succeeded,
    {", ".join(_stage_messages(stage) for stage in _STAGES)}
"""


def run_detail(path: Path, run_id: str) -> dict[str, Any] | None:
    """One run's card plus its selection, stages, rubric ratings, and models."""

    found = get_card(path, run_id)
    if found is None:
        return None
    with read_runs(path) as connection:
        row = connection.execute(
            f"SELECT {_DETAIL_COLUMNS} FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    rubric_labels = json.loads(row["rubric_labels"] or "{}")
    failed_stage = _failed_stage(row) if row["status"] == "failed" else None
    stages = []
    for stage in _STAGES:
        if row[f"{stage}_succeeded"]:
            state = "done"
        elif row["status"] == stage:
            state = "running"
        elif stage == failed_stage:
            state = "failed"
        else:
            state = "not_started"
        message = row[f"{stage}_result_message"] or row[f"{stage}_progress_message"]
        stages.append({"stage": stage, "state": state, "message": one_line(message, 2_000)})
    return {
        **found,
        "auto_run": bool(row["auto_run"]),
        "stages": stages,
        "rubrics": [
            {**review, "label": rubric_labels.get(review["rubric"], review["rubric"])}
            for review in json.loads(row["rubric_reviews"] or "[]")
        ],
        "models": {
            "proposal_writer": row["proposal_writer"],
            "proposal_evaluator": row["proposal_evaluator"],
        },
    }


def run_page(path: Path, run_id: str) -> dict[str, Any] | None:
    """What the page's run view shows: `run_detail`, plus every field Monopole's web UI reads
    from GET /api/runs/{run_id}, serialized by the same code, so the page can show each stage's
    progress, events, judge attempts, and reflection evidence. Unlike that route, it doesn't
    check a pending review's managed files for drift; reviewing stays in the web UI."""

    detail = run_detail(path, run_id)
    if detail is None:
        return None
    store = RunStore(path, read_only=True)
    try:
        run = store.get(run_id)
        full = run_response(run, store).model_dump(mode="json") if run is not None else None
    finally:
        store.close()
    return {**detail, **full} if full is not None else None
