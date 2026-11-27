"""The page's run view and Trends tab: `plugin run` returns what the web UI's run route returns,
and `plugin trends` what its analysis route returns, read without writing."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import httpx
import pytest

from tests.interfaces.routes.test_runs import _config
from tests.interfaces.routes.test_runs import route_context as route_context  # noqa: F401
from weave_agent_signals.cli import main
from weave_agent_signals.plugin import server
from weave_agent_signals.plugin.runs import run_page
from weave_agent_signals.runs.events import RunEventDraft
from weave_agent_signals.runs.store import RunDatabaseSchemaError, RunStore


def _judged_run(client, service, models, rubrics) -> str:
    """A run set up through the web UI's routes that finished scoring and judging, with one
    judging event."""

    run_id = client.post("/api/runs").json()["run_id"]
    client.put(f"/api/runs/{run_id}/selection", json={"session_ids": ["session-1"]})
    client.put(f"/api/runs/{run_id}/config", json=_config(models, rubrics))
    client.put(f"/api/runs/{run_id}/auto_run", json={"auto_run": False})
    client.post(f"/api/runs/{run_id}/advance")
    client.post(f"/api/runs/{run_id}/advance")
    service.store.append_run_event(
        run_id,
        RunEventDraft(
            stage="judging",
            at=datetime(2026, 7, 14, 12, 1, tzinfo=UTC),
            phase="transport_retry",
            message="Retrying GPT-5.6 Sol",
            details={"model": "codex:gpt-5.6-sol", "request_attempt": 2},
        ),
    )
    return run_id


def test_the_run_view_has_every_field_the_web_uis_run_route_returns(route_context, tmp_path):  # noqa: F811
    client, service, models, rubrics = route_context
    run_id = _judged_run(client, service, models, rubrics)
    api = client.get(f"/api/runs/{run_id}").json()

    page = run_page(tmp_path / "runs.db", run_id)

    assert {key: page[key] for key in api} == api
    assert page["judging_result"]["events"][-1]["message"] == "Retrying GPT-5.6 Sol"
    assert page["effective_config"]["rubrics"][0]["id"] == "judge.verification"
    # The card and summary the list and header read stay beside them.
    assert (page["label"], page["group"], page["next_stage"]) == (
        "Waiting to reflect",
        "waiting",
        "reflecting",
    )
    assert [stage["state"] for stage in page["stages"]] == ["done", "done", "not_started"]
    assert run_page(tmp_path / "runs.db", "run-missing") is None


def test_the_run_view_reads_without_writing(route_context, tmp_path, monkeypatch, capsys):  # noqa: F811
    client, service, models, rubrics = route_context
    run_id = _judged_run(client, service, models, rubrics)
    db = tmp_path / "runs.db"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("WEAVE_AGENT_SIGNALS_RUN_DB", str(db))
    watcher = sqlite3.connect(db)
    before = watcher.execute("PRAGMA data_version").fetchone()[0]

    assert main(["plugin", "run", "--item", run_id]) == 0

    shown = json.loads(capsys.readouterr().out)["run"]
    assert shown["run_id"] == run_id and shown["judging_plan"] is None
    assert watcher.execute("PRAGMA data_version").fetchone()[0] == before
    watcher.close()


def test_a_read_only_store_refuses_writes_and_other_schemas(tmp_path):
    db = tmp_path / "runs.db"
    RunStore(db).close()
    reader = RunStore(db, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.create()
    finally:
        reader.close()
    other = tmp_path / "other.db"
    connection = sqlite3.connect(other)
    connection.execute("PRAGMA user_version = 10")
    connection.close()
    with pytest.raises(RunDatabaseSchemaError, match="schema: 10"):
        RunStore(other, read_only=True)


ANALYSIS = {
    "summary": [
        {
            "scorer": "outcome.tests_passed",
            "count": 40,
            "mean": 0.8,
            "ci": [0.7, 0.9],
            "binary": True,
            "pass_rate": 0.8,
            "tag_counts": {},
            "confident": True,
        }
    ],
    "ab_leaderboard": [],
    "trends": [
        {
            "scorer": "outcome.tests_passed",
            "direction": "regression",
            "older_mean": 0.9,
            "recent_mean": 0.7,
            "delta": -0.2,
            "significant": True,
            "sample_count": 40,
        }
    ],
    "coaching_markdown": "## Coaching\n",
}


def _serve(monkeypatch, handler) -> None:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "open_server", lambda port: server.Server(port, client))


def test_trends_prints_the_web_uis_analysis_from_the_server(state, monkeypatch, capsys):
    requests: list[tuple[str, str]] = []

    def analysis_route(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json=ANALYSIS)

    _serve(monkeypatch, analysis_route)

    assert main(["plugin", "trends"]) == 0

    assert json.loads(capsys.readouterr().out) == ANALYSIS
    assert requests == [("GET", "/api/analyze")]


def test_trends_without_the_server_says_so(state, monkeypatch, capsys):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _serve(monkeypatch, refuse)

    assert main(["plugin", "trends"]) == 1
    assert "Monopole's server isn't answering" in capsys.readouterr().err
