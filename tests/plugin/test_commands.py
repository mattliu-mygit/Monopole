from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

from tests.plugin.fixtures import add_run, failed_judging, started
from weave_agent_signals.cli import main
from weave_agent_signals.plugin.commands import watch_run_database
from weave_agent_signals.plugin.settings import state_dir


def test_the_json_actions_print_what_the_page_reads(state, capsys):
    add_run(
        state / "runs.db",
        "run-1",
        "failed",
        "2026-11-05T22:00:00+00:00",
        **started(),
        **failed_judging(),
    )

    assert main(["plugin", "runs"]) == 0
    listed = json.loads(capsys.readouterr().out)
    # The run view validates a run's stored stages as the web UI's run route does, so it reads a
    # run in real shapes; the card fixtures above store only the fields cards read.
    add_run(state / "runs.db", "run-2", "failed", "2026-11-04T22:00:00+00:00", error="boom")
    assert main(["plugin", "run", "--item", "run-2"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert main(["plugin", "status"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert main(["plugin", "providers"]) == 0
    last = json.loads(capsys.readouterr().out)

    assert listed["format"] == 1 and listed["truncated"] is False
    assert [run["run_id"] for run in listed["runs"]] == ["run-1"]
    assert shown["run"]["run_id"] == "run-2" and len(shown["run"]["stages"]) == 3
    assert shown["run"]["error"] == "boom" and shown["run"]["judging_result"] is None
    assert printed["format"] == 2 and printed["items"][0]["group"] == "failed"
    assert last == {"format": 1, "checked_at": None, "providers": [], "sign_in": None}


def test_a_missing_run_exits_1_with_one_line_on_stderr(state, capsys):
    assert main(["plugin", "run", "--item", "run-9"]) == 1
    assert capsys.readouterr().err == "There is no run run-9.\n"


def test_the_watcher_reports_commits_from_other_connections(state):
    db = state / "runs.db"
    changed = state_dir() / "runs-changed"
    stop = threading.Event()
    watcher = threading.Thread(target=watch_run_database, args=(db, stop, 0.01))
    watcher.start()
    try:
        time.sleep(0.05)
        assert not changed.exists()
        add_run(db, "run-1", "created", "2026-11-05T22:00:00+00:00")
        deadline = time.monotonic() + 2
        while not changed.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert changed.exists()
    finally:
        stop.set()
        watcher.join()


def test_reads_leave_the_database_unchanged(state, capsys):
    db = state / "runs.db"
    add_run(db, "run-1", "complete", "2026-11-05T22:00:00+00:00")
    connection = sqlite3.connect(db)
    before = connection.execute("PRAGMA data_version").fetchone()[0]

    for argv in (["plugin", "status"], ["plugin", "runs"], ["plugin", "run", "--item", "run-1"]):
        assert main(argv) == 0

    assert connection.execute("PRAGMA data_version").fetchone()[0] == before
    connection.close()


@pytest.mark.parametrize(
    "argv",
    [
        ["plugin", "monitor", "--exp", "abc"],
        ["plugin", "monitor", "--pri"],
        ["plugin", "delete", "--item", "run-1", "--pri"],
        ["plugin", "run", "--ite", "run-1"],
        ["plugin", "--he"],
    ],
)
def test_abbreviated_flags_are_refused(state, capsys, argv):
    # Prompting Time blocks only the full --print and --expect spellings, so an abbreviation
    # must not reach them.
    with pytest.raises(SystemExit) as exited:
        main(argv)

    assert exited.value.code == 2
    assert "usage:" in capsys.readouterr().err
