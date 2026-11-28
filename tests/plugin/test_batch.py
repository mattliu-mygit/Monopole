"""Delete, Rerun, and Continue given several --item: one confirmation covers every run."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3

import httpx
import pytest

from tests.plugin.fixtures import add_run, started
from tests.plugin.test_newrun import CONFIG, FakeServer
from weave_agent_signals.cli import main
from weave_agent_signals.plugin import server
from weave_agent_signals.plugin.settings import state_dir

CREATED_AT = "2026-11-05T22:00:00+00:00"


def _items(*run_ids: str) -> list[str]:
    return [arg for run_id in run_ids for arg in ("--item", run_id)]


def _print(capsys, action: str, *run_ids: str) -> tuple[dict, str]:
    assert main(["plugin", action, *_items(*run_ids), "--print"]) == 0
    out = capsys.readouterr().out
    found = re.fullmatch(r"fingerprint: ([0-9a-f]{64})", out.strip().splitlines()[-1])
    assert found
    request_file = state_dir() / f"{action}-request.json"
    assert hashlib.sha256(request_file.read_bytes()).hexdigest() == found.group(1)
    return json.loads(request_file.read_text()), found.group(1)


def _runs(db) -> list[str]:
    with sqlite3.connect(db) as connection:
        return [row[0] for row in connection.execute("SELECT run_id FROM runs ORDER BY run_id")]


def _server(monkeypatch, answer) -> None:
    monkeypatch.setattr(
        server,
        "open_server",
        lambda port: server.Server(port, httpx.Client(transport=httpx.MockTransport(answer))),
    )


def test_delete_confirms_several_runs_once_and_deletes_them_all(state, capsys):
    db = state / "runs.db"
    for run_id in ("run-1", "run-2", "run-3"):
        add_run(db, run_id, "failed", CREATED_AT, **started())

    request, fingerprint = _print(capsys, "delete", "run-1", "run-2")

    assert request["action"] == "delete"
    assert [run["run_id"] for run in request["runs"]] == ["run-1", "run-2"]
    assert _runs(db) == ["run-1", "run-2", "run-3"]

    assert main(["plugin", "delete", *_items("run-1", "run-2"), "--expect", fingerprint]) == 0

    assert capsys.readouterr().out.strip() == "Deleted run-1, run-2."
    assert _runs(db) == ["run-3"]


def test_one_fingerprint_covers_every_run_so_a_change_to_any_refuses_all(state, capsys):
    db = state / "runs.db"
    add_run(db, "run-1", "failed", CREATED_AT, **started())
    add_run(db, "run-2", "complete", CREATED_AT, **started())
    _, fingerprint = _print(capsys, "delete", "run-1", "run-2")
    with sqlite3.connect(db) as connection:
        connection.execute(
            "UPDATE runs SET reflection_review = '{\"status\": \"pending\"}' WHERE run_id = 'run-2'"
        )

    assert main(["plugin", "delete", *_items("run-1", "run-2"), "--expect", fingerprint]) == 1

    assert "delete 2 runs changed after you confirmed; nothing ran" in capsys.readouterr().err
    assert _runs(db) == ["run-1", "run-2"]


def test_a_run_that_cant_be_deleted_refuses_the_whole_request(state, capsys):
    db = state / "runs.db"
    add_run(db, "run-1", "failed", CREATED_AT, **started())
    add_run(db, "run-2", "judging", CREATED_AT, **started())

    assert main(["plugin", "delete", *_items("run-1", "run-2"), "--print"]) == 1

    assert "run-2 is judging; cancel it" in capsys.readouterr().err
    assert not (state_dir() / "delete-request.json").exists()


def test_a_run_given_twice_is_refused(state, capsys):
    add_run(state / "runs.db", "run-1", "failed", CREATED_AT, **started())

    assert main(["plugin", "delete", *_items("run-1", "run-1"), "--print"]) == 1

    assert "--item run-1 is given more than once" in capsys.readouterr().err


def test_rerun_reads_the_catalogs_once_and_starts_a_run_per_old_run(state, monkeypatch, capsys):
    fake = FakeServer([*CONFIG["judge_models"], CONFIG["proposal_model"]])
    _server(monkeypatch, fake)
    for run_id in ("run-a", "run-b"):
        add_run(state / "runs.db", run_id, "failed", CREATED_AT, **started(), run_config=CONFIG)

    request, fingerprint = _print(capsys, "rerun", "run-a", "run-b")

    assert [run["from_run"] for run in request["runs"]] == ["run-a", "run-b"]
    assert [r[1] for r in fake.requests] == ["/api/models", "/api/rubrics"]
    assert fake.writes() == []

    assert main(["plugin", "rerun", *_items("run-a", "run-b"), "--expect", fingerprint]) == 0

    printed = json.loads(capsys.readouterr().out)
    started_run = {"format": 1, "run_id": "run-new", "status": "scoring", "route": "/runs/run-new"}
    assert printed == {
        "format": 1,
        "results": [
            {"item": "run-a", "result": started_run},
            {"item": "run-b", "result": started_run},
        ],
    }
    assert [w[:2] for w in fake.writes()].count(("POST", "/api/runs")) == 2


@pytest.fixture
def advance(monkeypatch, state):
    """Monopole's advance route, which refuses the runs named in `refuse`."""

    refuse: set[str] = set()
    seen: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/advance"):
            run_id = path.split("/")[3]
            seen.append(run_id)
            if run_id in refuse:
                return httpx.Response(409, json={"detail": f"{run_id} is already running"})
            return httpx.Response(200, json={"run_id": run_id, "status": "judging"})
        if path == "/api/rubrics":
            return httpx.Response(200, json={"catalog_version": "v1", "rubrics": []})
        return httpx.Response(404, json={"detail": "Not Found"})

    _server(monkeypatch, answer)
    return refuse, seen


def test_continue_goes_on_past_a_run_the_server_refuses_and_reports_each(state, advance, capsys):
    refuse, seen = advance
    for run_id in ("run-1", "run-2"):
        add_run(state / "runs.db", run_id, "scoring", CREATED_AT, **started(), auto_run=0)
    request, fingerprint = _print(capsys, "continue", "run-1", "run-2")
    assert [run["run_id"] for run in request["runs"]] == ["run-1", "run-2"]
    refuse.add("run-1")

    assert main(["plugin", "continue", *_items("run-1", "run-2"), "--expect", fingerprint]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "format": 1,
        "results": [
            {
                "item": "run-1",
                "error": "POST /api/runs/run-1/advance: run-1 is already running",
            },
            {
                "item": "run-2",
                "result": {
                    "format": 1,
                    "run_id": "run-2",
                    "status": "judging",
                    "route": "/runs/run-2",
                },
            },
        ],
    }
    assert seen == ["run-1", "run-2"]


def test_continue_refuses_with_every_reason_when_every_run_was_refused(state, advance, capsys):
    refuse, _ = advance
    for run_id in ("run-1", "run-2"):
        add_run(state / "runs.db", run_id, "scoring", CREATED_AT, **started(), auto_run=0)
    _, fingerprint = _print(capsys, "continue", "run-1", "run-2")
    refuse.update({"run-1", "run-2"})

    assert main(["plugin", "continue", *_items("run-1", "run-2"), "--expect", fingerprint]) == 1

    err = capsys.readouterr().err
    assert "run-1: POST /api/runs/run-1/advance: run-1 is already running" in err
    assert "run-2: POST /api/runs/run-2/advance: run-2 is already running" in err
