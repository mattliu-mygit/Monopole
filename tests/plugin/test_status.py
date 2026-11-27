from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.plugin.fixtures import add_run, failed_judging, judged, reflected, started
from tests.plugin.schema_check import check_schema
from weave_agent_signals.plugin.commands import status
from weave_agent_signals.plugin.settings import state_dir

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "schemas" / "prompting-time-status.json").read_text())
MANIFEST = json.loads((ROOT / "prompting-time-plugin.json").read_text())
NOW = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)


def _save_check(checked_at: datetime, providers: list[dict]) -> None:
    folder = state_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "providers.json").write_text(
        json.dumps({"format": 1, "checked_at": checked_at.isoformat(), "providers": providers})
    )


def test_status_fits_prompting_times_schema_and_the_manifests_groups(state):
    db = state / "runs.db"
    add_run(
        db, "run-failed", "failed", "2026-11-05T22:00:00+00:00", **started(), **failed_judging()
    )
    add_run(
        db,
        "run-review",
        "complete",
        "2026-11-05T21:00:00+00:00",
        **started(),
        **judged(),
        **reflected(challenge=None),
    )
    add_run(
        db,
        "run-judging",
        "judging",
        "2026-11-05T20:00:00+00:00",
        **started(),
        judging_progress={"status_message": "Judging 2 of 6 rubrics"},
    )
    add_run(
        db,
        "run-done",
        "complete",
        "2026-11-05T19:00:00+00:00",
        **started(),
        **judged(),
        **reflected(challenge=None, review=None),
    )
    add_run(db, "run-draft", "created", "2026-11-05T23:00:00+00:00")

    printed = status(db, NOW)

    check_schema(printed, SCHEMA)
    groups = {group["id"] for group in MANIFEST["status"]["groups"]}
    allowed = set(MANIFEST["status"]["item_actions"])
    assert all(item["group"] in groups for item in printed["items"])
    assert all(set(item["actions"]) <= allowed for item in printed["items"])
    by_id = {item["id"]: item for item in printed["items"]}
    # A draft is still being set up in the web UI, so Prompting Time does not list it.
    assert set(by_id) == {"run-failed", "run-review", "run-judging", "run-done"}
    assert by_id["run-failed"]["detail"].startswith("Session Outcome Quality failed:")
    assert by_id["run-review"]["detail"] == (
        "A proposal waits for your review · predicted 0.35 vs 0.15 now · 2 ratings · mean 0.62"
    )
    assert by_id["run-judging"] | {"detail": None} == {
        "id": "run-judging",
        "group": "running",
        "title": "Fix the flaky login test",
        "detail": None,
        "dot": "running",
        "project": None,
        "updated_at": "2026-11-05T20:00:00+00:00",
        "route": "/runs/run-judging",
        "actions": ["rerun"],
    }
    assert by_id["run-judging"]["detail"] == "Judging 2 of 6 rubrics"
    assert (by_id["run-done"]["actions"], by_id["run-done"]["dot"]) == (
        ["rerun", "delete"],
        "success",
    )
    assert printed["agent"] == {"words": "Running 1 run", "dot": "running"}
    assert printed["blockers"] == []


def test_status_lists_the_newest_50_and_says_it_left_the_rest_out(state):
    db = state / "runs.db"
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    for index in range(51):
        add_run(db, f"run-{index:02}", "complete", (start + timedelta(minutes=index)).isoformat())

    printed = status(db, NOW)

    check_schema(printed, SCHEMA)
    assert len(printed["items"]) == 50
    assert printed["items"][0]["id"] == "run-50"
    assert printed["truncated"] is True
    assert printed["agent"]["dot"] == "idle"
    assert printed["agent"]["words"].startswith("Idle · last run ")


def test_a_recent_check_that_found_providers_signed_out_blocks_every_run(state):
    _save_check(
        NOW - timedelta(hours=1),
        [
            {"id": "codex", "title": "Codex", "state": "signed_out", "detail": "Not logged in"},
            {"id": "wandb", "title": "W&B Inference", "state": "no_key", "detail": None},
            {"id": "openai", "title": "OpenAI", "state": "no_key", "detail": None},
            {"id": "devin", "title": "Devin", "state": "not_installed", "detail": None},
        ],
    )

    printed = status(state / "runs.db", NOW)

    check_schema(printed, SCHEMA)
    assert printed["blockers"] == [
        {
            "id": "codex-signed-out",
            "title": "Codex is signed out",
            "dot": "error",
            "route": "/providers",
            "action": "check-providers",
        },
        {
            "id": "wandb-no-key",
            "title": "No W&B API key",
            "dot": "error",
            "route": "/providers",
            "action": "check-providers",
        },
    ]
    assert printed["agent"] == {"words": "Idle · no runs yet · 2 blockers", "dot": "error"}
    assert {blocker["action"] for blocker in printed["blockers"]} <= {
        action["name"] for action in MANIFEST["actions"]
    }


def test_a_check_older_than_a_day_blocks_nothing(state):
    _save_check(
        NOW - timedelta(hours=25),
        [
            {"id": "codex", "title": "Codex", "state": "signed_out", "detail": None},
        ],
    )

    assert status(state / "runs.db", NOW)["blockers"] == []


@pytest.mark.parametrize(
    ("name", "change"),
    [
        ("an unknown dot", {"agent": {"words": "x", "dot": "red"}}),
        ("more than 50 items", {"items": [{"id": "a"}] * 51}),
        ("an extra agent field", {"agent": {"words": "x", "dot": "idle", "extra": 1}}),
    ],
)
def test_the_schema_check_rejects_output_that_does_not_fit(state, name, change):
    printed = status(state / "runs.db", NOW)
    with pytest.raises(ValueError):
        check_schema({**printed, **change}, SCHEMA)
