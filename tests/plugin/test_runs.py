from __future__ import annotations

import sqlite3

import pytest

from tests.plugin.fixtures import add_run, failed_judging, judged, reflected, started
from weave_agent_signals.plugin.runs import get_card, list_cards, run_detail
from weave_agent_signals.runs.store import RunDatabaseSchemaError


def test_a_run_that_failed_in_judging_names_the_rubric_and_the_judges_message(state):
    db = state / "runs.db"
    add_run(db, "run-1", "failed", "2026-11-05T22:00:56+00:00", **started(), **failed_judging())

    card = get_card(db, "run-1")

    assert card["label"] == "Failed at judging"
    assert (card["dot"], card["group"]) == ("error", "failed")
    assert card["title"] == "Fix the flaky login test"
    assert card["failure"] == {
        "stage": "judging",
        "stopped_at": "Session Outcome Quality",
        "cause": "Session Outcome Quality failed: DeepSeek V4.1 Flash failed ValueError after "
        '28 provider requests: unknown evidence IDs: ["e224"]',
    }
    assert [judge["label"] for judge in card["judges"]] == ["DeepSeek V4.1 Flash", "GPT-5.6 Sol"]
    assert (card["turns"], card["rated"], card["mean_score"]) == (12, 0, None)
    assert card["deletable"] is True


def test_a_failure_without_a_stage_message_shows_the_runs_error(state):
    db = state / "runs.db"
    error = "Server error '503 Service Unavailable' for url 'https://evaluations.wandb.ai/x'"
    add_run(
        db,
        "run-1",
        "failed",
        "2026-11-05T20:41:30+00:00",
        error=error,
        data_selection={"session_ids": ["01a1ad21-1a23-7c9c"]},
    )

    card = get_card(db, "run-1")

    assert card["failure"] == {"stage": "scoring", "stopped_at": "Scoring", "cause": error}
    # No scoring result means no first prompt, so the session names the run.
    assert card["title"] == "Session 01a1ad21-1a2"


def test_a_proposal_compares_the_best_candidate_with_the_current_instructions(state):
    db = state / "runs.db"
    challenge = {
        "status": "incomplete",
        "winner": "tie",
        "reason": "Sandbox preflight infrastructure failed",
    }
    add_run(
        db,
        "run-1",
        "complete",
        "2026-11-05T20:50:59+00:00",
        **started(),
        **judged(),
        **reflected(challenge=challenge),
    )

    card = get_card(db, "run-1")

    assert (card["label"], card["dot"], card["group"]) == ("Review needed", "attention", "review")
    assert (card["rated"], card["mean_score"]) == (2, 0.625)
    assert card["proposal"] == {
        "current": 0.15,
        "proposed": 0.35,
        "change": 0.2,
        "candidates": 2,
        "check": challenge,
        "recommended": False,
        "note": "Targets outcome.command and outcome.test.",
    }


def test_a_finished_run_without_a_pending_review_is_done(state):
    db = state / "runs.db"
    add_run(
        db,
        "run-1",
        "complete",
        "2026-11-05T20:50:59+00:00",
        **started(),
        **judged(),
        **reflected(challenge=None, review=None),
    )
    add_run(db, "run-2", "cancelled", "2026-11-05T21:00:00+00:00", **started())

    first, second = (get_card(db, run) for run in ("run-1", "run-2"))

    assert (first["label"], first["dot"], first["group"]) == ("No change", "success", "done")
    assert first["proposal"]["check"] is None
    assert (second["label"], second["group"]) == ("Cancelled", "done")


def test_running_and_draft_runs(state):
    db = state / "runs.db"
    add_run(
        db,
        "run-1",
        "judging",
        "2026-11-05T21:00:00+00:00",
        **started(),
        judging_progress={"status_message": "Judging 2 of 6 rubrics"},
    )
    add_run(db, "run-2", "created", "2026-11-05T22:00:00+00:00")

    running, draft = get_card(db, "run-1"), get_card(db, "run-2")

    assert (running["label"], running["dot"], running["group"]) == ("Judging", "running", "running")
    assert running["progress"] == "Judging 2 of 6 rubrics"
    assert running["deletable"] is False
    assert (draft["label"], draft["group"], draft["title"]) == ("Draft", None, "Run run-2")


def test_a_long_first_prompt_becomes_one_short_line(state):
    db = state / "runs.db"
    add_run(
        db,
        "run-1",
        "complete",
        "2026-11-05T21:00:00+00:00",
        **started(prompt="Review   this\nstack " + "x" * 200),
    )

    title = get_card(db, "run-1")["title"]

    assert title.startswith("Review this stack xxx")
    assert len(title) == 120 and title.endswith("…")


def test_list_cards_lists_newest_first_and_says_when_it_left_runs_out(state):
    db = state / "runs.db"
    for day in (1, 2, 3):
        add_run(db, f"run-{day}", "complete", f"2026-11-0{day}T00:00:00+00:00")

    cards, truncated = list_cards(db, 2)

    assert [card["run_id"] for card in cards] == ["run-3", "run-2"]
    assert truncated is True
    assert list_cards(db, 3)[1] is False


def test_run_detail_adds_stages_rubrics_and_models(state):
    db = state / "runs.db"
    add_run(db, "run-1", "failed", "2026-11-05T22:00:56+00:00", **started(), **failed_judging())

    detail = run_detail(db, "run-1")

    assert [(stage["stage"], stage["state"]) for stage in detail["stages"]] == [
        ("scoring", "done"),
        ("judging", "failed"),
        ("reflecting", "not_started"),
    ]
    assert detail["rubrics"][0]["label"] == "Session Outcome Quality"
    assert detail["models"] == {
        "proposal_writer": "GLM 5.3",
        "proposal_evaluator": "DeepSeek V4.1 Flash",
    }
    assert run_detail(db, "run-missing") is None


def test_no_database_yet_means_no_runs_and_creates_nothing(tmp_path):
    missing = tmp_path / "runs.db"

    assert list_cards(missing, 10) == ([], False)
    assert not missing.exists()


def test_a_database_of_another_schema_is_refused(tmp_path):
    db = tmp_path / "runs.db"
    connection = sqlite3.connect(db)
    connection.execute("PRAGMA user_version = 10")
    connection.close()

    with pytest.raises(RunDatabaseSchemaError, match="schema: 10"):
        list_cards(db, 10)
