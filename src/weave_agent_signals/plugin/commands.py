"""`weave-agent-signals plugin <action>`: the commands `prompting-time-plugin.json` declares.

Each prints JSON, except where an action marked `confirm: true` previews its request: those follow
Prompting Time's confirmation convention (`confirm.py`), so `--print` changes nothing and
`--expect <sha256>` runs only the request the user confirmed.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from weave_agent_signals.models import ConfigurationError
from weave_agent_signals.plugin import keys, newrun, providers, server, signin, tools
from weave_agent_signals.plugin.confirm import Refused, add_confirm_flags, confirmed
from weave_agent_signals.plugin.runs import (
    DETAIL_LIMIT,
    WAITING_REASONS,
    get_card,
    list_cards,
    one_line,
    run_detail,
)
from weave_agent_signals.plugin.settings import (
    apply_settings,
    load_settings,
    run_db_path,
    save_settings,
    state_dir,
    write_atomic,
)
from weave_agent_signals.runs.store import RunStore

STATUS_ITEM_LIMIT = 50
RUNS_CHANGED_FILE = "runs-changed"
# How often the server looks for committed run changes. While a run writes progress, the page
# and Prompting Time read again at most this often; a run-store commit callback would be
# needed for anything faster.
RUN_CHANGE_INTERVAL_SECONDS = 2.0


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False))


def notify_runs_changed() -> None:
    """Rewrite a file in the state folder, so Prompting Time's watcher tells the page and reads
    status again."""

    stamp = datetime.now(timezone.utc).isoformat()
    write_atomic(state_dir() / RUNS_CHANGED_FILE, stamp + "\n")


def watch_run_database(path: Path, stop: threading.Event, interval: float) -> None:
    """Calls `notify_runs_changed` after any connection commits to the run database. SQLite's
    `data_version` changes when another connection commits, so one cheap read per interval
    sees the server's writes and the deletes other processes make."""

    connection = None
    seen = None
    try:
        while not stop.is_set():
            if connection is None and path.exists():
                connection = sqlite3.connect(path, check_same_thread=False)
            if connection is not None:
                version = connection.execute("PRAGMA data_version").fetchone()[0]
                if seen is not None and version != seen:
                    notify_runs_changed()
                seen = version
            stop.wait(interval)
    finally:
        if connection is not None:
            connection.close()


def _item_detail(card: dict[str, Any]) -> str | None:
    if card["failure"]:
        return one_line(card["failure"]["cause"], DETAIL_LIMIT)
    if card["group"] == "waiting":
        return WAITING_REASONS[card["next_stage"]]
    if card["group"] == "running":
        return card["progress"] or card["label"]
    parts = []
    proposal = card["proposal"]
    if card["group"] == "review":
        parts.append("A proposal waits for your review")
        if proposal and proposal["proposed"] is not None and proposal["current"] is not None:
            parts.append(f"predicted {proposal['proposed']:.2f} vs {proposal['current']:.2f} now")
    else:
        parts.append(card["label"])
    if card["mean_score"] is not None:
        parts.append(f"{card['rated']} ratings · mean {card['mean_score']:.2f}")
    return " · ".join(parts)


def _local_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone().strftime("%b %-d, %-I:%M %p")


def status(path: Path, now: datetime | None = None) -> dict[str, Any]:
    """Prompting Time's status format 2: the agent row, blockers, and the newest 50 runs that
    are not drafts. Drafts are runs still being set up in Monopole's web UI."""

    cards, truncated = list_cards(path, STATUS_ITEM_LIMIT, drafts=False)
    found = providers.blockers(now)
    running = sum(card["group"] == "running" for card in cards)
    if running:
        words, dot = f"Running {running} run{'s' if running > 1 else ''}", "running"
    elif cards:
        words, dot = f"Idle · last run {_local_time(cards[0]['created_at'])}", "idle"
    else:
        words, dot = "Idle · no runs yet", "idle"
    if found:
        words = f"{words} · {found[0]['title'] if len(found) == 1 else f'{len(found)} blockers'}"
        dot = "error"
    items = [
        {
            "id": card["run_id"],
            "group": card["group"],
            "title": card["title"],
            "detail": _item_detail(card),
            "dot": card["dot"],
            "project": None,
            "updated_at": card["updated_at"],
            "route": f"/runs/{card['run_id']}",
            "actions": [
                *(["continue"] if card["next_stage"] else []),
                "rerun",
                *(["delete"] if card["deletable"] else []),
            ],
        }
        for card in cards
    ]
    return {
        "format": 2,
        "agent": {"words": words, "dot": dot},
        "blockers": found,
        "items": items,
        "truncated": truncated,
    }


def delete_request(path: Path, run_id: str) -> str:
    """The text of the delete request the user confirms. It holds only what decides the
    request, so its fingerprint changes when the run does."""

    card = get_card(path, run_id)
    if card is None:
        raise Refused(f"There is no run {run_id} in {path}.")
    if not card["deletable"]:
        raise Refused(f"{run_id} is {card['status']}; cancel it in Monopole before deleting it.")
    request = {
        "action": "delete",
        "run_id": run_id,
        "title": card["title"],
        "status": card["label"],
        "created_at": card["created_at"],
        "run_database": str(path),
        "deletes": "The run, its saved judge calls, and its events. Its Scores in CES stay.",
    }
    return json.dumps(request, indent=2, ensure_ascii=False) + "\n"


def delete(path: Path, run_id: str, *, print_only: bool, expect: str | None) -> None:
    text = delete_request(path, run_id)
    if not confirmed("delete", text, print_only=print_only, expect=expect, what=f"delete {run_id}"):
        return
    store = RunStore(path)
    try:
        store.delete(run_id)
    finally:
        store.close()
    notify_runs_changed()
    print(f"Deleted {run_id}.")


def rerun(path: Path, run_id: str, settings: dict[str, Any], args: argparse.Namespace) -> None:
    with server.open_server(settings["port"]) as monopole:
        text, setup = newrun.rerun_request(path, run_id, monopole)
        what = f"rerun {run_id}"
        if confirmed("rerun", text, print_only=args.print_only, expect=args.expect, what=what):
            _print_json(newrun.start_run(monopole, setup))


def continue_request(path: Path, run_id: str, server_base: str) -> str:
    """The text of the Continue request the user confirms: the run, its status, and the stage
    it starts with what that stage calls or writes. A change to the run changes it."""

    detail = run_detail(path, run_id)
    if detail is None:
        raise Refused(f"There is no run {run_id} in {path}.")
    stage = detail["next_stage"]
    if stage is None:
        raise Refused(f"{run_id} is {detail['label']}, so it isn't waiting for Continue.")
    if stage == "scoring":
        then = (
            "then judging and reflection follow without stopping, which call the judge models"
            if detail["auto_run"]
            else "then it waits for Continue again"
        )
        starts = f"Scoring, which writes Scores to CES; {then}."
    elif stage == "judging":
        judges = ", ".join(judge["label"] for judge in detail["judges"]) or "the run's judges"
        starts = f"Judging, which calls the judge models: {judges}."
    else:
        models = detail["models"]
        starts = (
            f"Reflection, which calls the proposal writer ({models['proposal_writer'] or '?'}) "
            f"and the evaluator ({models['proposal_evaluator'] or '?'})."
        )
    request = {
        "action": "continue",
        "run_id": run_id,
        "title": detail["title"],
        "status": detail["label"],
        "starts": starts,
        "server": server_base,
    }
    return json.dumps(request, indent=2, ensure_ascii=False) + "\n"


def continue_run(
    path: Path, run_id: str, settings: dict[str, Any], args: argparse.Namespace
) -> None:
    """Starts the stage a waiting run waits for, through the server's advance route, as
    Continue in Monopole's web UI does."""

    with server.open_server(settings["port"]) as monopole:
        text = continue_request(path, run_id, monopole.base)
        if args.print_only:
            monopole.check()
        what = f"continue {run_id}"
        if confirmed("continue", text, print_only=args.print_only, expect=args.expect, what=what):
            status = monopole.call("POST", f"/api/runs/{run_id}/advance")["status"]
            _print_json(
                {"format": 1, "run_id": run_id, "status": status, "route": f"/runs/{run_id}"}
            )


def create_run(settings: dict[str, Any], args: argparse.Namespace) -> None:
    setup = newrun.parse_setup(args.setup)
    with server.open_server(settings["port"]) as monopole:
        text = newrun.create_request(setup, monopole)
        what = "start a new run"
        if confirmed("create-run", text, print_only=args.print_only, expect=args.expect, what=what):
            _print_json(newrun.start_run(monopole, setup))


def run_tool(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    """score, backfill, and monitor confirm first, since they write Scores or post alerts;
    insights only reads."""

    action = args.plugin_action
    if action == "insights":
        command, namespace = tools.insights_command(args.kind)
        _print_json(tools.run(action, {"kind": args.kind}, command, namespace))
        return
    if action == "score":
        command, namespace = tools.score_command(args.hours)
        recorded = {"hours": args.hours}
        writes = (
            f"Deterministic Scores for turns from the last {args.hours} hours that lack them, "
            "at most 100 turns"
        )
        extra: dict[str, Any] = {}
    elif action == "backfill":
        command, namespace = tools.backfill_command(args.start, args.end)
        recorded = {"start": args.start, "end": args.end}
        writes = "Every Scorer a turn, session, or subagent in the range lacks"
        extra = {}
    else:
        webhook = settings["alert_webhook"]
        command, namespace = tools.monitor_command(webhook)
        recorded = {}
        writes = "Nothing in CES; new alerts go to the log" + (
            " and are posted to the webhook" if webhook else ""
        )
        extra = {"webhook_host": urlparse(webhook).netloc if webhook else None}
    text = tools.request(action, recorded, writes, **extra)
    if confirmed(action, text, print_only=args.print_only, expect=args.expect, what=action):
        _print_json(tools.run(action, recorded, command, namespace))


def serve(args: argparse.Namespace, settings: dict[str, Any]) -> int:
    """Monopole's server, which runs the run stages and recovers interrupted runs at start,
    with a thread that reports run changes to the state folder."""

    from weave_agent_signals.cli import cmd_serve

    registry = os.environ.get("TARGET_REGISTRY")
    runtime = os.environ.get("SMOLMACHINES_RUNTIME")
    if not registry or not runtime:
        raise ConfigurationError(
            "Monopole's server needs a target registry and a sandbox runtime; set them in "
            "Monopole's settings in Prompting Time"
        )
    stop = threading.Event()
    watcher = threading.Thread(
        target=watch_run_database,
        args=(run_db_path(), stop, RUN_CHANGE_INTERVAL_SECONDS),
        name="run-changes",
        daemon=True,
    )
    watcher.start()
    try:
        return cmd_serve(
            argparse.Namespace(
                host="127.0.0.1",
                port=settings["port"],
                target_registry=registry,
                sandbox_runtime=runtime,
                entity=args.entity,
                project=args.project,
            )
        )
    finally:
        stop.set()


def add_parser(subs: argparse._SubParsersAction) -> None:
    parser = subs.add_parser(
        "plugin", help="Actions Prompting Time runs for the Monopole plugin (JSON output)"
    )
    actions = parser.add_subparsers(dest="plugin_action", required=True)
    actions.add_parser("status", help="Prompting Time status format 2")
    runs = actions.add_parser("runs", help="The newest runs, as the plugin's page lists them")
    runs.add_argument("--limit", type=int, default=100)
    run = actions.add_parser("run", help="One run's detail")
    run.add_argument("--item", required=True, metavar="RUN_ID")
    remove = actions.add_parser("delete", help="Delete a run after Prompting Time's confirmation")
    remove.add_argument("--item", required=True, metavar="RUN_ID")
    add_confirm_flags(remove)
    again = actions.add_parser("rerun", help="Start a new run with an old run's settings")
    again.add_argument("--item", required=True, metavar="RUN_ID")
    add_confirm_flags(again)
    advance = actions.add_parser("continue", help="Start the stage a waiting run waits for")
    advance.add_argument("--item", required=True, metavar="RUN_ID")
    add_confirm_flags(advance)
    create = actions.add_parser("create-run", help="Start a new run from New run's settings")
    create.add_argument("setup", help='{"selection": ..., "config": ..., "auto_run": ...} as JSON')
    add_confirm_flags(create)
    actions.add_parser("catalog", help="The models and rubrics a new run can choose")
    listed = actions.add_parser("sessions", help="Sessions a new run can choose")
    listed.add_argument("--since")
    listed.add_argument("--until")
    listed.add_argument("--timezone")
    listed.add_argument("--limit", type=int, default=50)
    actions.add_parser("providers", help="The last provider sign-in check and sign-in")
    actions.add_parser("check-providers", help="Check every provider's sign-in now")
    login = actions.add_parser("sign-in", help="Run a provider CLI's login command")
    login.add_argument("--item", required=True, choices=sorted(signin.COMMANDS))
    add_confirm_flags(login)
    code = actions.add_parser("sign-in-code", help="Type a pasted code into the waiting sign-in")
    code.add_argument("--item", required=True, choices=sorted(signin.COMMANDS))
    code.add_argument("--code", required=True)
    stop = actions.add_parser("sign-in-cancel", help="Stop the waiting sign-in")
    stop.add_argument("--item", required=True, choices=sorted(signin.COMMANDS))
    stage = actions.add_parser("stage-key", help="Hold an API key for the set-key that follows")
    stage.add_argument("--item", required=True, choices=keys.KEY_PROVIDERS)
    stage.add_argument("--key", required=True)
    key = actions.add_parser(
        "set-key", help="Check and save the held W&B or OpenAI API key, else one read from stdin"
    )
    key.add_argument("--item", required=True, choices=keys.KEY_PROVIDERS)
    score = actions.add_parser("score", help="Score recent turns, as `score` does")
    score.add_argument("--hours", type=int, choices=(24, 168), default=24)
    add_confirm_flags(score)
    backfill = actions.add_parser("backfill", help="Score a date range, as `backfill` does")
    backfill.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    backfill.add_argument("--end", metavar="YYYY-MM-DD", help="The last day included")
    add_confirm_flags(backfill)
    monitor = actions.add_parser("monitor", help="Check for regressions now, as `monitor` does")
    add_confirm_flags(monitor)
    insights = actions.add_parser("insights", help="One of analyze's insight tables")
    insights.add_argument("--kind", required=True, choices=tools.INSIGHT_KINDS)
    actions.add_parser("tools", help="Each tool's last run")
    actions.add_parser("get-settings", help="The plugin's settings")
    save = actions.add_parser("save-settings", help="Check and save changed settings")
    save.add_argument("changes", help="The changed settings as one JSON object")
    actions.add_parser("serve", help="Monopole's server, configured by the plugin's settings")


def run_plugin_action(args: argparse.Namespace) -> int:
    """Runs one plugin action. A refusal prints its reason on stderr, which Prompting Time
    shows, and exits 1."""

    try:
        return _run(args)
    except Refused as refusal:
        print(refusal, file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    settings = load_settings()
    apply_settings(settings)
    keys.apply_saved_keys()
    action = args.plugin_action
    if action == "status":
        _print_json(status(run_db_path()))
    elif action == "runs":
        if args.limit < 1:
            raise ConfigurationError("--limit must be at least 1")
        cards, truncated = list_cards(run_db_path(), args.limit)
        _print_json({"format": 1, "runs": cards, "truncated": truncated})
    elif action == "run":
        detail = run_detail(run_db_path(), args.item)
        if detail is None:
            raise Refused(f"There is no run {args.item}.")
        _print_json({"format": 1, "run": detail})
    elif action == "delete":
        delete(run_db_path(), args.item, print_only=args.print_only, expect=args.expect)
    elif action == "rerun":
        rerun(run_db_path(), args.item, settings, args)
    elif action == "continue":
        continue_run(run_db_path(), args.item, settings, args)
    elif action == "create-run":
        create_run(settings, args)
    elif action == "catalog":
        with server.open_server(settings["port"]) as monopole:
            _print_json(newrun.catalog(monopole))
    elif action == "sessions":
        with server.open_server(settings["port"]) as monopole:
            found = newrun.sessions(monopole, args.since, args.until, args.timezone, args.limit)
            _print_json(found)
    elif action == "providers":
        _print_json({**providers.last_check(), "sign_in": signin.current()})
    elif action == "check-providers":
        _print_json(providers.check_providers())
    elif action == "sign-in":
        text = signin.sign_in_request(args.item)
        what = f"sign in to {providers.PROVIDER_TITLES[args.item]}"
        if confirmed("sign-in", text, print_only=args.print_only, expect=args.expect, what=what):
            _print_json(signin.sign_in(args.item))
    elif action == "sign-in-code":
        _print_json(signin.send_code(args.item, args.code))
    elif action == "sign-in-cancel":
        _print_json(signin.cancel(args.item))
    elif action == "stage-key":
        _print_json(keys.stage_key(args.item, args.key))
    elif action == "set-key":
        _print_json(keys.set_key(args.item, keys.take_key(args.item)))
    elif action in ("score", "backfill", "monitor", "insights"):
        run_tool(args, settings)
    elif action == "tools":
        _print_json(tools.last_runs())
    elif action == "get-settings":
        _print_json(settings)
    elif action == "save-settings":
        _print_json(save_settings(args.changes))
    elif action == "serve":
        return serve(args, settings)
    return 0
