from __future__ import annotations

import hashlib
import re
import sqlite3

import pytest

from tests.plugin.fixtures import add_run, started
from weave_agent_signals.cli import main
from weave_agent_signals.plugin.settings import state_dir


def _runs(db) -> list[str]:
    connection = sqlite3.connect(db)
    try:
        return [row[0] for row in connection.execute("SELECT run_id FROM runs")]
    finally:
        connection.close()


def _print(capsys) -> tuple[str, str]:
    assert main(["plugin", "delete", "--item", "run-1", "--print"]) == 0
    out = capsys.readouterr().out
    fingerprint = re.fullmatch(r"fingerprint: ([0-9a-f]{64})", out.strip().splitlines()[-1])
    assert fingerprint
    return out, fingerprint.group(1)


def test_print_saves_the_request_whose_sha256_is_the_fingerprint_and_deletes_nothing(state, capsys):
    db = state / "runs.db"
    add_run(db, "run-1", "failed", "2026-11-05T22:00:00+00:00", **started())

    out, fingerprint = _print(capsys)

    request = state_dir() / "delete-request.json"
    # Prompting Time finds the request by the absolute path the output names, read up to the
    # next whitespace without trailing punctuation.
    assert str(request) in [word.rstrip(";,:.") for word in out.split()]
    assert hashlib.sha256(request.read_bytes()).hexdigest() == fingerprint
    assert '"run_id": "run-1"' in request.read_text()
    assert _runs(db) == ["run-1"]


def test_expect_deletes_the_confirmed_run_and_reports_the_change(state, capsys):
    db = state / "runs.db"
    add_run(db, "run-1", "failed", "2026-11-05T22:00:00+00:00", **started())
    _, fingerprint = _print(capsys)

    assert main(["plugin", "delete", "--item", "run-1", "--expect", fingerprint]) == 0

    assert capsys.readouterr().out.strip() == "Deleted run-1."
    assert _runs(db) == []
    assert not (state_dir() / "delete-request.json").exists()
    assert (state_dir() / "runs-changed").exists()


def test_expect_refuses_when_the_run_changed_after_the_confirmation(state, capsys):
    db = state / "runs.db"
    add_run(db, "run-1", "complete", "2026-11-05T22:00:00+00:00", **started())
    _, fingerprint = _print(capsys)
    connection = sqlite3.connect(db)
    connection.execute('UPDATE runs SET reflection_review = \'{"status": "pending"}\'')
    connection.commit()
    connection.close()

    assert main(["plugin", "delete", "--item", "run-1", "--expect", fingerprint]) == 1

    assert "changed after you confirmed; nothing deleted" in capsys.readouterr().err
    assert _runs(db) == ["run-1"]


def test_a_running_or_missing_run_is_refused_before_anything_is_saved(state, capsys):
    add_run(state / "runs.db", "run-1", "judging", "2026-11-05T22:00:00+00:00")

    assert main(["plugin", "delete", "--item", "run-1", "--print"]) == 1
    assert "is judging; cancel it" in capsys.readouterr().err
    assert main(["plugin", "delete", "--item", "run-9", "--print"]) == 1
    assert "There is no run run-9" in capsys.readouterr().err
    assert not (state_dir() / "delete-request.json").exists()


def test_delete_runs_only_through_the_confirmation(state):
    with pytest.raises(SystemExit):
        main(["plugin", "delete", "--item", "run-1"])
