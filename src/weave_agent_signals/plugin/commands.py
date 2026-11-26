"""`weave-agent-signals plugin <action>`: the commands `prompting-time-plugin.json` declares.

Each prints JSON, except `delete`, which follows Prompting Time's confirmation convention:
`--print` saves the request in the state folder and prints `fingerprint: <sha256>` of that file
on its last line; `--expect <sha256>` deletes only when the same request still has that
fingerprint, so a run that changed after the user confirmed is not deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from weave_agent_signals.models import ConfigurationError
from weave_agent_signals.plugin import providers
from weave_agent_signals.plugin.runs import DETAIL_LIMIT, get_card, list_cards, one_line, run_detail
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
DELETE_REQUEST_FILE = "delete-request.json"
# How often the server looks for committed run changes. While a run writes progress, the page
# and Prompting Time read again at most this often; a run-store commit callback would be
# needed for anything faster.
RUN_CHANGE_INTERVAL_SECONDS = 2.0


class Refused(Exception):
    """The action did nothing, for the reason in its message."""


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
            "actions": ["delete"] if card["deletable"] else [],
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


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def delete(path: Path, run_id: str, *, print_only: bool, expect: str | None) -> None:
    text = delete_request(path, run_id)
    request_file = state_dir() / DELETE_REQUEST_FILE
    if print_only:
        write_atomic(request_file, text)
        print(f"Saved the request to delete {run_id} in {request_file}; nothing deleted yet.")
        print(f"fingerprint: {_fingerprint(text)}")
        return
    if expect != _fingerprint(text):
        raise Refused(f"{run_id} changed after you confirmed; nothing deleted. Try again.")
    store = RunStore(path)
    try:
        store.delete(run_id)
    finally:
        store.close()
    request_file.unlink(missing_ok=True)
    notify_runs_changed()
    print(f"Deleted {run_id}.")


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
    mode = remove.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print", action="store_true", dest="print_only")
    mode.add_argument("--expect", metavar="FINGERPRINT")
    actions.add_parser("providers", help="The last provider sign-in check")
    actions.add_parser("check-providers", help="Check every provider's sign-in now")
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
    elif action == "providers":
        _print_json(providers.last_check())
    elif action == "check-providers":
        _print_json(providers.check_providers())
    elif action == "get-settings":
        _print_json(settings)
    elif action == "save-settings":
        _print_json(save_settings(args.changes))
    elif action == "serve":
        return serve(args, settings)
    return 0
