from __future__ import annotations

import json
import re
import sqlite3

import httpx
import pytest

from tests.plugin.fixtures import add_run, judged, started
from weave_agent_signals.cli import main
from weave_agent_signals.plugin import server
from weave_agent_signals.plugin.settings import state_dir

CREATED_AT = "2026-11-05T22:00:00+00:00"


@pytest.fixture
def requests(monkeypatch, state):
    """Monopole's advance route, recording each request; it answers the run's next status."""

    seen: list[tuple[str, str]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/advance"):
            return httpx.Response(200, json={"run_id": "run-1", "status": "judging"})
        if request.url.path == "/api/rubrics":
            return httpx.Response(200, json={"catalog_version": "v1", "rubrics": []})
        return httpx.Response(404, json={"detail": "Not Found"})

    monkeypatch.setattr(
        server,
        "open_server",
        lambda port: server.Server(port, httpx.Client(transport=httpx.MockTransport(answer))),
    )
    return seen


def _print(capsys, run_id: str = "run-1") -> tuple[dict, str]:
    assert main(["plugin", "continue", "--item", run_id, "--print"]) == 0
    out = capsys.readouterr().out
    found = re.fullmatch(r"fingerprint: ([0-9a-f]{64})", out.strip().splitlines()[-1])
    assert found
    request = json.loads((state_dir() / "continue-request.json").read_text())
    return request, found.group(1)


def test_continue_starts_judging_a_scored_run_through_the_servers_advance_route(
    state, requests, capsys
):
    add_run(state / "runs.db", "run-1", "scoring", CREATED_AT, **started(), auto_run=0)

    request, fingerprint = _print(capsys)

    assert request["status"] == "Waiting to judge"
    assert request["starts"] == (
        "Judging, which calls the judge models: DeepSeek V4.1 Flash, GPT-5.6 Sol."
    )
    # --print read the run database and checked that the server answers.
    assert requests == [("GET", "/api/rubrics")]

    assert main(["plugin", "continue", "--item", "run-1", "--expect", fingerprint]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "format": 1,
        "run_id": "run-1",
        "status": "judging",
        "route": "/runs/run-1",
    }
    assert requests[1:] == [("POST", "/api/runs/run-1/advance")]


def test_continue_starts_reflection_after_judging_and_scoring_for_a_set_up_draft(
    state, requests, capsys
):
    db = state / "runs.db"
    add_run(db, "run-1", "judging", CREATED_AT, **started(), **judged(), auto_run=0)
    columns = started()
    add_run(
        db,
        "run-draft",
        "created",
        CREATED_AT,
        data_selection=columns["data_selection"],
        run_config={"judge_models": ["wandb:DeepSeek-V4.1-Flash"]},
    )

    request, _ = _print(capsys)
    assert request["starts"] == (
        "Reflection, which calls the proposal writer (GLM 5.3) and the evaluator "
        "(DeepSeek V4.1 Flash)."
    )
    request, _ = _print(capsys, "run-draft")
    assert request["starts"] == (
        "Scoring, which writes Scores to CES; then it waits for Continue again."
    )


@pytest.mark.parametrize(
    ("status", "columns"),
    [
        ("scoring", {**started(), "auto_run": 1}),
        ("judging", {**started(), "auto_run": 0}),
        ("complete", {**started(), **judged(), "auto_run": 0}),
        ("created", {}),
    ],
    ids=["continues-automatically", "still-judging", "complete", "draft-not-set-up"],
)
def test_continue_refuses_a_run_that_is_not_waiting(state, requests, capsys, status, columns):
    add_run(state / "runs.db", "run-1", status, CREATED_AT, **columns)

    assert main(["plugin", "continue", "--item", "run-1", "--print"]) == 1

    assert "isn't waiting for Continue" in capsys.readouterr().err
    assert requests == []


def test_continue_refuses_when_the_run_changed_after_the_confirmation(state, requests, capsys):
    db = state / "runs.db"
    add_run(db, "run-1", "scoring", CREATED_AT, **started(), auto_run=0)
    _, fingerprint = _print(capsys)
    # Continued from Monopole's web UI meanwhile.
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE runs SET status = 'judging' WHERE run_id = 'run-1'")

    assert main(["plugin", "continue", "--item", "run-1", "--expect", fingerprint]) == 1

    assert "isn't waiting for Continue" in capsys.readouterr().err
    # Only the preview's check reached the server; nothing advanced.
    assert requests == [("GET", "/api/rubrics")]


def test_continue_says_when_the_server_is_not_running(state, requests, monkeypatch, capsys):
    add_run(state / "runs.db", "run-1", "scoring", CREATED_AT, **started(), auto_run=0)
    _, fingerprint = _print(capsys)
    timeouts: list[object] = []

    def refuse(request):
        timeouts.append(request.extensions["timeout"]["connect"])
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        server,
        "open_server",
        lambda port: server.Server(port, httpx.Client(transport=httpx.MockTransport(refuse))),
    )

    # The preview refuses before Prompting Time asks the user to confirm, after a short wait.
    assert main(["plugin", "continue", "--item", "run-1", "--print"]) == 1
    err = capsys.readouterr().err
    assert "Monopole's server isn't answering on http://127.0.0.1:8787" in err
    assert "Start Monopole in Prompting Time" in err
    assert timeouts == [server.CHECK_TIMEOUT_SECONDS]

    # A confirmed request whose server stopped meanwhile says the same.
    assert main(["plugin", "continue", "--item", "run-1", "--expect", fingerprint]) == 1
    assert "Monopole's server isn't answering" in capsys.readouterr().err
