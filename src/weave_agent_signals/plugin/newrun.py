"""New runs from the plugin: Rerun, which repeats an old run's sessions and settings, and New run,
which the page fills in, either blank or from an old run (Edit and rerun).

Both start the run through the server's run routes in the order Monopole's web UI uses: create,
save the selection, the configuration, and auto-run, then advance into scoring. A rerun reads the
old run's sessions again, so turns added to them since join the new run, and saves the old
configuration under the current catalog versions; a model or rubric that has left the catalog
stops the rerun with a message naming it. The old run stays.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from weave_agent_signals.plugin.confirm import Refused, request_text
from weave_agent_signals.plugin.runs import DETAIL_LIMIT, printed_bytes, read_runs, shorten
from weave_agent_signals.plugin.server import Server

_SETUP_KEYS = {"selection", "config", "auto_run"}
# The most `sessions` prints, well under the 8 MiB Prompting Time passes a page. Up to 100
# sessions each carry their first prompt and the reason of every low turn Score, either of which
# can be long; the page shows only the prompt, as a title.
SESSIONS_BYTES = 1_000_000
# The most signal evidence each session keeps, from most to least, while fitting.
_EVIDENCE_LIMITS = (1_000, 32, 8, 0)


def catalog(server: Server) -> dict[str, Any]:
    """The models and rubrics a new run can choose, as the server's catalogs list them."""

    return {
        "format": 1,
        "models": server.get("/api/models"),
        "rubrics": server.get("/api/rubrics"),
    }


def sessions(
    server: Server, since: str | None, until: str | None, timezone: str | None, limit: int
) -> dict[str, Any]:
    params = {"since": since, "until": until, "timezone": timezone, "limit": limit}
    found = server.get("/api/sessions", {key: value for key, value in params.items() if value})
    fitted = fit_sessions(found)
    if fitted is None:
        raise Refused("The session list is too large to show here; list fewer sessions.")
    return fitted


def fit_sessions(found: dict[str, Any]) -> dict[str, Any] | None:
    """The server's session list in at most SESSIONS_BYTES: every session's texts, such as its
    first prompt and its evidence reasons, cut to DETAIL_LIMIT characters, and its signal
    evidence to the fewest items that fit. `shortened` says whether anything was cut. None when
    even no evidence doesn't fit."""

    for evidence_limit in _EVIDENCE_LIMITS:
        fitted = [shorten(item, DETAIL_LIMIT, evidence_limit) for item in found["sessions"]]
        view = {**found, "sessions": fitted, "shortened": fitted != found["sessions"]}
        if printed_bytes(view) <= SESSIONS_BYTES:
            return view
    return None


def session_estimates(server: Server, session_ids: list[str]) -> dict[str, Any]:
    """Each named session's judging token estimates by token counter, which New run's capacity
    hints read, as the web app's run setup asks for them; a session CES doesn't know is left
    out."""

    return server.call("POST", "/api/sessions/estimates", body={"session_ids": session_ids})


def _old_run(path: Path, run_id: str) -> sqlite3.Row:
    with read_runs(path) as connection:
        row = None
        if connection is not None:
            row = connection.execute(
                "SELECT data_selection, run_config, auto_run, "
                "json_extract(effective_config, '$.models') AS models "
                "FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
    if row is None:
        raise Refused(f"There is no run {run_id} in {path}.")
    if not row["data_selection"] or not row["run_config"]:
        raise Refused(f"{run_id} was never set up, so there is nothing to rerun.")
    return row


def _labels(models_json: str | None) -> dict[str, str]:
    """Model labels by ID from the old run's pinned models, to name a model that left the
    catalog the way the old run showed it."""

    models = json.loads(models_json or "{}")
    found = [*models.get("judges", []), *models.get("challenge_judges", [])]
    found += [models[key] for key in ("proposal_writer", "proposal_evaluator") if key in models]
    return {model["id"]: model["label"] for model in found if "id" in model and "label" in model}


def _missing(config: dict[str, Any], found: dict[str, Any], labels: dict[str, str]) -> list[str]:
    models = {model["id"] for model in found["models"]["available_models"]}
    rubrics = {rubric["id"] for rubric in found["rubrics"]["rubrics"]}

    def name(model_id: str) -> str:
        label = labels.get(model_id)
        return f"{label} ({model_id})" if label else model_id

    missing = [f"judge {name(m)}" for m in config["judge_models"] if m not in models]
    missing += [
        f"A/B check judge {name(m)}"
        for m in config["challenge_judge_models"]
        if m not in models and m not in config["judge_models"]
    ]
    roles = {"proposal_model": "proposal writer", "proposal_evaluator_model": "evaluator"}
    missing += [
        f"{role} {name(config[key])}" for key, role in roles.items() if config[key] not in models
    ]
    missing += [f"rubric {r}" for r in config["rubrics"] if r not in rubrics]
    return missing


def _starts(setup: dict[str, Any]) -> str:
    then = (
        "then judging and reflection follow without stopping, which call the judge models"
        if setup["auto_run"]
        else "then it stops after each stage until you continue it"
    )
    return f"A new run: scoring starts now and writes Scores to CES; {then}."


def _request(action: str, setup: dict[str, Any], server: Server, **extra: Any) -> dict[str, Any]:
    return {
        "action": action,
        **extra,
        "sessions": list(setup["selection"]["session_ids"]),
        "judges": list(setup["config"]["judge_models"]),
        "continue_automatically": setup["auto_run"],
        "starts": _starts(setup),
        "server": server.base,
        "setup": setup,
    }


def rerun_setup(path: Path, run_id: str, found: dict[str, Any]) -> dict[str, Any]:
    """The old run's selection, configuration under the current catalog versions in `found`
    (what `catalog` returns), and auto-run. Refuses when a model or rubric it used is no longer
    in the catalog."""

    row = _old_run(path, run_id)
    config = json.loads(row["run_config"])
    missing = _missing(config, found, _labels(row["models"]))
    if missing:
        raise Refused(
            f"Can't rerun {run_id}: {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} no longer in the catalog. "
            "Use Edit and rerun to choose another."
        )
    config["model_catalog_version"] = found["models"]["catalog_version"]
    config["rubric_catalog_version"] = found["rubrics"]["catalog_version"]
    return {
        "selection": json.loads(row["data_selection"]),
        "config": config,
        "auto_run": bool(row["auto_run"]),
    }


def rerun_request(
    path: Path, run_id: str, server: Server, found: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The rerun request the user confirms, and the setup it starts, under the catalogs in
    `found`."""

    setup = rerun_setup(path, run_id, found)
    return _request("rerun", setup, server, from_run=run_id), setup


def parse_setup(text: str) -> dict[str, Any]:
    """New run's `{"selection", "config", "auto_run"}`. The server checks the values."""

    try:
        setup = json.loads(text)
    except ValueError as exc:
        raise Refused(f"The new run's settings are not JSON: {exc}") from exc
    if not isinstance(setup, dict) or set(setup) != _SETUP_KEYS:
        raise Refused("The new run's settings need exactly selection, config, and auto_run.")
    if not isinstance(setup["selection"], dict) or not isinstance(setup["config"], dict):
        raise Refused("selection and config must be objects.")
    if type(setup["auto_run"]) is not bool:
        raise Refused("auto_run must be true or false.")
    if not setup["selection"].get("session_ids") or not setup["config"].get("judge_models"):
        raise Refused("Choose at least one session and one judge.")
    return setup


def create_request(setup: dict[str, Any], server: Server) -> str:
    return request_text("create-run", [_request("create-run", setup, server)])


def start_run(server: Server, setup: dict[str, Any]) -> dict[str, Any]:
    """Create the run, save its setup, and start scoring. A setup the server refuses deletes
    the draft it created, so a refused New run leaves nothing behind."""

    run_id = server.call("POST", "/api/runs")["run_id"]
    try:
        server.call("PUT", f"/api/runs/{run_id}/selection", body=setup["selection"])
        server.call("PUT", f"/api/runs/{run_id}/config", body=setup["config"])
        server.call("PUT", f"/api/runs/{run_id}/auto_run", body={"auto_run": setup["auto_run"]})
        started = server.call("POST", f"/api/runs/{run_id}/advance")
    except Refused:
        try:
            server.call("DELETE", f"/api/runs/{run_id}")
        except Refused:
            pass
        raise
    return {"format": 1, "run_id": run_id, "status": started["status"], "route": f"/runs/{run_id}"}
