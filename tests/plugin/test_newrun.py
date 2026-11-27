from __future__ import annotations

import hashlib
import json
import re

import httpx
import pytest

from tests.plugin.fixtures import JUDGES, SESSION, add_run, started
from weave_agent_signals.cli import main
from weave_agent_signals.plugin import server
from weave_agent_signals.plugin.settings import state_dir

CONFIG = {
    "model_catalog_version": "models-1",
    "rubric_catalog_version": "rubrics-1",
    "judge_models": [judge["id"] for judge in JUDGES],
    "challenge_judge_models": [JUDGES[0]["id"]],
    "proposal_model": "wandb:GLM-5.3",
    "proposal_evaluator_model": JUDGES[0]["id"],
    "rubrics": ["judge.verification"],
    "candidate_budget": 3,
    "force": False,
}


class FakeServer:
    """Monopole's run and catalog routes, recording each request."""

    def __init__(self, models: list[str], *, refuse_config: str | None = None):
        self.models = models
        self.refuse_config = refuse_config
        self.requests: list[tuple[str, str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        route = (request.method, request.url.path)
        if route == ("GET", "/api/models"):
            models = [{"id": model, "label": model} for model in self.models]
            return httpx.Response(
                200, json={"catalog_version": "models-2", "available_models": models}
            )
        if route == ("GET", "/api/rubrics"):
            rubrics = [{"id": "judge.verification"}, {"id": "judge.session_outcome"}]
            return httpx.Response(200, json={"catalog_version": "rubrics-2", "rubrics": rubrics})
        if route == ("GET", "/api/sessions"):
            return httpx.Response(200, json={"sessions": [], "params": dict(request.url.params)})
        if route == ("POST", "/api/sessions/estimates"):
            estimate = {
                "turn_tokens": [900, 1200],
                "largest_turn_tokens": 1200,
                "total_tokens": 2100,
            }
            known = [sid for sid in body["session_ids"] if sid == SESSION]
            return httpx.Response(200, json={sid: {"utf8_bytes_div_3": estimate} for sid in known})
        if route == ("POST", "/api/runs"):
            return httpx.Response(200, json={"run_id": "run-new", "status": "created"})
        if route == ("PUT", "/api/runs/run-new/config") and self.refuse_config:
            detail = {"code": "invalid_run_request", "message": self.refuse_config}
            return httpx.Response(400, json={"detail": detail})
        if route == ("POST", "/api/runs/run-new/advance"):
            return httpx.Response(200, json={"run_id": "run-new", "status": "scoring"})
        if route == ("DELETE", "/api/runs/run-new"):
            return httpx.Response(204)
        if request.method == "PUT":
            return httpx.Response(200, json={"run_id": "run-new", "status": "created"})
        return httpx.Response(404, json={"detail": "Not Found"})

    def writes(self) -> list[tuple[str, str, object]]:
        return [request for request in self.requests if request[0] != "GET"]


@pytest.fixture
def fake(monkeypatch, state):
    fake = FakeServer([*CONFIG["judge_models"], CONFIG["proposal_model"]])
    monkeypatch.setattr(
        server,
        "open_server",
        lambda port: server.Server(port, httpx.Client(transport=httpx.MockTransport(fake))),
    )
    return fake


def _stored_run(state, run_id: str = "run-old", auto_run: int = 1) -> None:
    add_run(
        state / "runs.db",
        run_id,
        "failed",
        "2026-11-05T22:00:00+00:00",
        **started(),
        run_config=CONFIG,
        auto_run=auto_run,
    )


def _print(capsys, argv: list[str]) -> tuple[str, str]:
    assert main(argv + ["--print"]) == 0
    out = capsys.readouterr().out
    found = re.fullmatch(r"fingerprint: ([0-9a-f]{64})", out.strip().splitlines()[-1])
    assert found
    return out, found.group(1)


def test_rerun_starts_a_new_run_with_the_old_sessions_and_settings_under_current_catalogs(
    state, fake, capsys
):
    _stored_run(state)
    out, fingerprint = _print(capsys, ["plugin", "rerun", "--item", "run-old"])

    request_file = state_dir() / "rerun-request.json"
    assert str(request_file) in [word.rstrip(";,:.") for word in out.split()]
    assert hashlib.sha256(request_file.read_bytes()).hexdigest() == fingerprint
    request = json.loads(request_file.read_text())
    assert request["from_run"] == "run-old" and request["continue_automatically"] is True
    assert "judging and reflection follow" in request["starts"]
    # --print only read the catalogs.
    assert fake.writes() == []

    assert main(["plugin", "rerun", "--item", "run-old", "--expect", fingerprint]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "format": 1,
        "run_id": "run-new",
        "status": "scoring",
        "route": "/runs/run-new",
    }
    selection = {"session_ids": [SESSION], "since": None, "until": None, "timezone": None}
    config = CONFIG | {"model_catalog_version": "models-2", "rubric_catalog_version": "rubrics-2"}
    assert fake.writes() == [
        ("POST", "/api/runs", None),
        ("PUT", "/api/runs/run-new/selection", selection),
        ("PUT", "/api/runs/run-new/config", config),
        ("PUT", "/api/runs/run-new/auto_run", {"auto_run": True}),
        ("POST", "/api/runs/run-new/advance", None),
    ]
    assert not request_file.exists()


def test_rerun_refuses_a_judge_that_left_the_catalog_and_names_it(state, fake, capsys):
    _stored_run(state)
    fake.models.remove("codex:gpt-5.6-sol")

    assert main(["plugin", "rerun", "--item", "run-old", "--print"]) == 1

    err = capsys.readouterr().err
    assert "Can't rerun run-old: judge GPT-5.6 Sol (codex:gpt-5.6-sol) is no longer" in err
    assert "Edit and rerun" in err
    assert not (state_dir() / "rerun-request.json").exists()
    assert fake.writes() == []


def test_rerun_refuses_when_the_catalog_changed_after_the_confirmation(state, fake, capsys):
    _stored_run(state)
    _, fingerprint = _print(capsys, ["plugin", "rerun", "--item", "run-old"])
    fake.models.append("wandb:new-model")
    fake.models.remove(CONFIG["proposal_model"])

    assert main(["plugin", "rerun", "--item", "run-old", "--expect", fingerprint]) == 1

    assert "proposal writer wandb:GLM-5.3 is no longer in the catalog" in capsys.readouterr().err
    assert fake.writes() == []


def test_a_setup_the_server_refuses_deletes_the_draft_and_says_why(state, fake, capsys):
    _stored_run(state)
    fake.refuse_config = "stale model catalog version"
    _, fingerprint = _print(capsys, ["plugin", "rerun", "--item", "run-old"])

    assert main(["plugin", "rerun", "--item", "run-old", "--expect", fingerprint]) == 1

    assert "stale model catalog version" in capsys.readouterr().err
    assert fake.writes()[-1] == ("DELETE", "/api/runs/run-new", None)


def test_rerun_says_when_the_server_is_not_running(state, monkeypatch, capsys):
    _stored_run(state)

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        server,
        "open_server",
        lambda port: server.Server(port, httpx.Client(transport=httpx.MockTransport(refuse))),
    )

    assert main(["plugin", "rerun", "--item", "run-old", "--print"]) == 1

    assert "Monopole's server isn't answering on http://127.0.0.1:8787" in capsys.readouterr().err


def test_create_run_starts_new_runs_settings_after_the_confirmation(state, fake, capsys):
    setup = {
        "selection": {"session_ids": ["s1", "s2"], "since": None, "until": None, "timezone": None},
        "config": CONFIG,
        "auto_run": False,
    }
    argv = ["plugin", "create-run", json.dumps(setup)]
    _, fingerprint = _print(capsys, argv)
    request = json.loads((state_dir() / "create-run-request.json").read_text())
    assert request["sessions"] == ["s1", "s2"]
    assert "until you continue it" in request["starts"]
    assert fake.writes() == []

    assert main(argv + ["--expect", fingerprint]) == 0

    assert json.loads(capsys.readouterr().out)["run_id"] == "run-new"
    assert ("PUT", "/api/runs/run-new/auto_run", {"auto_run": False}) in fake.writes()


def test_create_run_refuses_settings_without_a_session(state, fake, capsys):
    setup = {"selection": {"session_ids": []}, "config": CONFIG, "auto_run": False}

    assert main(["plugin", "create-run", json.dumps(setup), "--print"]) == 1

    assert "Choose at least one session" in capsys.readouterr().err


def test_catalog_and_sessions_give_new_run_its_choices(state, fake, capsys):
    # Edit and rerun fills New run from `run`'s run_config and data_selection, which
    # test_run_page checks against the web UI's run route.
    assert main(["plugin", "catalog"]) == 0
    found = json.loads(capsys.readouterr().out)
    assert main(["plugin", "sessions", "--since", "2026-11-01", "--limit", "20"]) == 0
    listed = json.loads(capsys.readouterr().out)

    assert found["models"]["catalog_version"] == "models-2"
    assert found["rubrics"]["catalog_version"] == "rubrics-2"
    assert listed["params"] == {"since": "2026-11-01", "limit": "20"}


def test_session_estimates_ask_the_server_for_the_selected_sessions_only(state, fake, capsys):
    assert main(["plugin", "session-estimates", SESSION, "conv-unknown"]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        SESSION: {
            "utf8_bytes_div_3": {
                "turn_tokens": [900, 1200],
                "largest_turn_tokens": 1200,
                "total_tokens": 2100,
            }
        }
    }
    assert fake.requests == [
        ("POST", "/api/sessions/estimates", {"session_ids": [SESSION, "conv-unknown"]})
    ]
