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
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from weave_agent_signals.models import ConfigurationError
from weave_agent_signals.plugin import keys, newrun, providers, server, signin, tools
from weave_agent_signals.plugin.confirm import (
    Refused,
    add_confirm_flags,
    confirmed,
    request_text,
)
from weave_agent_signals.plugin.runs import (
    DETAIL_LIMIT,
    WAITING_REASONS,
    fit_run_view,
    get_card,
    list_cards,
    one_line,
    run_detail,
    run_page,
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
# How long `trends` waits for the server's analysis, which reads every Score's statistics and
# the newest turns' Scores from CES; on a large Dataset that takes minutes.
TRENDS_TIMEOUT_SECONDS = 600
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


def _what(action: str, run_ids: list[str]) -> str:
    """The request's name in messages: "delete run-1", or "delete 3 runs"."""

    return f"{action} {run_ids[0]}" if len(run_ids) == 1 else f"{action} {len(run_ids)} runs"


def distinct_items(items: list[str]) -> list[str]:
    """The run IDs given as `--item`, each at most once, in the order given."""

    for item in items:
        if items.count(item) > 1:
            raise Refused(f"--item {item} is given more than once; nothing ran.")
    return items


def each_run(run_ids: list[str], act: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """Runs `act(run_id)` for each run the user confirmed together, going on past a run the
    server refuses, and prints what happened to each: `{"item", "result"}` or `{"item",
    "error"}`. Refuses, with every reason, when every run was refused."""

    results: list[dict[str, Any]] = []
    for run_id in run_ids:
        try:
            results.append({"item": run_id, "result": act(run_id)})
        except Refused as refusal:
            results.append({"item": run_id, "error": str(refusal)})
    if all("error" in result for result in results):
        raise Refused("; ".join(f"{result['item']}: {result['error']}" for result in results))
    return {"format": 1, "results": results}


def delete_request(path: Path, run_id: str) -> dict[str, Any]:
    """The delete request the user confirms. It holds only what decides the request, so its
    fingerprint changes when the run does."""

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
    return request


def delete(path: Path, run_ids: list[str], *, print_only: bool, expect: str | None) -> None:
    """Deletes the runs after one confirmation covering all of them; any run that can't be
    deleted refuses the whole request before anything is saved."""

    text = request_text("delete", [delete_request(path, run_id) for run_id in run_ids])
    what = _what("delete", run_ids)
    if not confirmed("delete", text, print_only=print_only, expect=expect, what=what):
        return
    store = RunStore(path)
    try:
        for run_id in run_ids:
            store.delete(run_id)
    finally:
        store.close()
    notify_runs_changed()
    print(f"Deleted {', '.join(run_ids)}.")


def rerun(
    path: Path, run_ids: list[str], settings: dict[str, Any], args: argparse.Namespace
) -> None:
    """Starts a new run per old run after one confirmation covering all of them. One run
    prints what `start_run` does; several print each one's result or refusal."""

    with server.open_server(settings["port"]) as monopole:
        found = newrun.catalog(monopole)
        built = {run_id: newrun.rerun_request(path, run_id, monopole, found) for run_id in run_ids}
        text = request_text("rerun", [request for request, _ in built.values()])
        what = _what("rerun", run_ids)
        if not confirmed("rerun", text, print_only=args.print_only, expect=args.expect, what=what):
            return

        def start(run_id: str) -> dict[str, Any]:
            return newrun.start_run(monopole, built[run_id][1])

        _print_json(start(run_ids[0]) if len(run_ids) == 1 else each_run(run_ids, start))


def continue_request(path: Path, run_id: str, server_base: str) -> dict[str, Any]:
    """The Continue request the user confirms: the run, its status, and the stage it starts
    with what that stage calls or writes. A change to the run changes it."""

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
    return request


def continue_run(
    path: Path, run_ids: list[str], settings: dict[str, Any], args: argparse.Namespace
) -> None:
    """Starts the stage each waiting run waits for, through the server's advance route, as
    Continue in Monopole's web UI does, after one confirmation covering all of them. One run
    prints its new status; several print each one's result or refusal."""

    with server.open_server(settings["port"]) as monopole:
        requests = [continue_request(path, run_id, monopole.base) for run_id in run_ids]
        text = request_text("continue", requests)
        if args.print_only:
            monopole.check()
        what = _what("continue", run_ids)
        if not confirmed(
            "continue", text, print_only=args.print_only, expect=args.expect, what=what
        ):
            return

        def advance(run_id: str) -> dict[str, Any]:
            status = monopole.call("POST", f"/api/runs/{run_id}/advance")["status"]
            return {"format": 1, "run_id": run_id, "status": status, "route": f"/runs/{run_id}"}

        _print_json(advance(run_ids[0]) if len(run_ids) == 1 else each_run(run_ids, advance))


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
        command, namespace, tables = tools.insights_command(args.kind)
        _print_json(tools.run(action, {"kind": args.kind}, command, namespace, tables))
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
    # Delete, Rerun, and Continue take one or more --item; several confirm once, together.
    remove = actions.add_parser("delete", help="Delete runs after Prompting Time's confirmation")
    again = actions.add_parser("rerun", help="Start new runs with old runs' settings")
    advance = actions.add_parser("continue", help="Start the stage each waiting run waits for")
    for batch in (remove, again, advance):
        batch.add_argument("--item", required=True, action="append", metavar="RUN_ID")
        add_confirm_flags(batch)
    create = actions.add_parser("create-run", help="Start a new run from New run's settings")
    create.add_argument("setup", help='{"selection": ..., "config": ..., "auto_run": ...} as JSON')
    add_confirm_flags(create)
    actions.add_parser("catalog", help="The models and rubrics a new run can choose")
    listed = actions.add_parser("sessions", help="Sessions a new run can choose")
    listed.add_argument("--since")
    listed.add_argument("--until")
    listed.add_argument("--timezone")
    listed.add_argument("--limit", type=int, default=50)
    actions.add_parser(
        "trends", help="Score summaries, trends, A/B leaderboard, and coaching, as Analysis shows"
    )
    sizes = actions.add_parser(
        "session-estimates", help="Judging token estimates for the sessions New run selects"
    )
    sizes.add_argument("session_ids", nargs="+", metavar="SESSION_ID")
    actions.add_parser("providers", help="The last provider sign-in check and sign-in")
    actions.add_parser("check-providers", help="Check every provider's sign-in now")
    login = actions.add_parser("sign-in", help="Run a provider CLI's login command")
    login.add_argument("--item", required=True, choices=sorted(signin.COMMANDS))
    login.add_argument(
        "--device-auth",
        action="store_true",
        help="Codex only: sign in with a one-time code entered in any browser",
    )
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
        keys.discard_stale_staged_keys()
        _print_json(status(run_db_path()))
    elif action == "runs":
        if args.limit < 1:
            raise ConfigurationError("--limit must be at least 1")
        cards, truncated = list_cards(run_db_path(), args.limit)
        _print_json({"format": 1, "runs": cards, "truncated": truncated})
    elif action == "run":
        detail = run_page(run_db_path(), args.item)
        if detail is None:
            raise Refused(f"There is no run {args.item}.")
        view = fit_run_view(detail)
        if view is None:
            raise Refused(f"{args.item} is too large to show here; open it in Monopole's web UI.")
        _print_json({"format": 1, "run": view})
    elif action == "delete":
        run_ids = distinct_items(args.item)
        delete(run_db_path(), run_ids, print_only=args.print_only, expect=args.expect)
    elif action == "rerun":
        rerun(run_db_path(), distinct_items(args.item), settings, args)
    elif action == "continue":
        continue_run(run_db_path(), distinct_items(args.item), settings, args)
    elif action == "create-run":
        create_run(settings, args)
    elif action == "catalog":
        with server.open_server(settings["port"]) as monopole:
            _print_json(newrun.catalog(monopole))
    elif action == "sessions":
        with server.open_server(settings["port"]) as monopole:
            found = newrun.sessions(monopole, args.since, args.until, args.timezone, args.limit)
            _print_json(found)
    elif action == "trends":
        # The web UI's analysis, from the same route: summaries, the A/B leaderboard, trends,
        # and coaching. The route only reads CES.
        with server.open_server(settings["port"]) as monopole:
            _print_json(monopole.call("GET", "/api/analyze", timeout=TRENDS_TIMEOUT_SECONDS))
    elif action == "session-estimates":
        with server.open_server(settings["port"]) as monopole:
            _print_json(newrun.session_estimates(monopole, args.session_ids))
    elif action == "providers":
        _print_json({**providers.last_check(), "sign_in": signin.current()})
    elif action == "check-providers":
        _print_json(providers.check_providers())
    elif action == "sign-in":
        text = signin.sign_in_request(args.item, args.device_auth)
        what = f"sign in to {providers.PROVIDER_TITLES[args.item]}"
        if confirmed("sign-in", text, print_only=args.print_only, expect=args.expect, what=what):
            _print_json(signin.sign_in(args.item, device_auth=args.device_auth))
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
