"""The Tools screen's actions: `score`, `backfill`, `monitor`, and `insights`, the commands that
were terminal-only, run with the same code as `weave-agent-signals score` and the rest.

Prompting Time runs an action to its end with no time limit and runs other actions alongside it,
so a long backfill runs in its action's own process without holding anything else up. Each tool
records itself in `<state folder>/tools.json` when it starts and when it ends, with what it
printed, so the page shows a tool that is still running, and its last result, after it reloads;
Prompting Time's watcher tells the page when that file changes. The commands report no progress
while they run, so the page shows only when the tool started.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from typing import Any

from weave_agent_signals import ces_commands
from weave_agent_signals.plugin.confirm import Refused
from weave_agent_signals.plugin.settings import state_dir, write_atomic

TOOLS_FILE = "tools.json"
MONITOR_STATE_FILE = "monitor-state.json"
INSIGHT_KINDS = ("subagents", "judges", "segments")
# Newest turns whose Scores feed monitor's config order, as the command's default.
FEED_LIMIT = 1000
OUTPUT_LIMIT = 64 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def last_runs() -> dict[str, Any]:
    """Each tool's last run. A run whose process is gone without finishing reads as stopped."""

    try:
        tools = json.loads((state_dir() / TOOLS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tools = {}
    for entry in tools.values():
        if entry.get("state") == "running" and not _alive(entry.get("pid")):
            entry.update(state="stopped", output="The tool stopped without finishing.")
    return {"format": 1, "tools": tools}


def _record(tool: str, entry: dict[str, Any]) -> None:
    # Two tools finishing at once can each write the file they read; the later write wins and
    # the other tool's entry shows its previous run. Lock the file if that ever matters.
    tools = last_runs()["tools"]
    tools[tool] = entry
    write_atomic(state_dir() / TOOLS_FILE, json.dumps(tools, indent=2) + "\n")


def _ces_args(**values: Any) -> argparse.Namespace:
    return argparse.Namespace(ces_url=None, ces_scope_ref=None, ces_dataset_id=None, **values)


def ces_scope() -> dict[str, str]:
    """Where the tools read and write, for confirmations."""

    url, scope, dataset = ces_commands.ces_settings()
    return {"ces_url": url, "scope": scope, "dataset": dataset}


def request(tool: str, args: dict[str, Any], writes: str, **extra: Any) -> str:
    text = {"action": tool, **args, **ces_scope(), "writes": writes, **extra}
    return json.dumps(text, indent=2, ensure_ascii=False) + "\n"


def run(tool: str, args: dict[str, Any], command, namespace: argparse.Namespace) -> dict[str, Any]:
    """Run one of the CLI's commands, capturing what it prints and logs, and record it."""

    if last_runs()["tools"].get(tool, {}).get("state") == "running":
        raise Refused(f"{tool} is already running; wait for it to finish.")
    entry: dict[str, Any] = {
        "state": "running",
        "args": args,
        "started_at": _now(),
        "finished_at": None,
        "pid": os.getpid(),
        "exit_code": None,
        "output": None,
    }
    _record(tool, entry)
    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("weave_agent_signals")
    logger.addHandler(handler)
    try:
        with redirect_stdout(captured):
            code = command(namespace)
        state = "done" if code == 0 else "failed"
    except Exception as exc:
        captured.write(f"\n{type(exc).__name__}: {exc}\n")
        code, state = 1, "failed"
    finally:
        logger.removeHandler(handler)
    output = captured.getvalue().strip()
    if len(output) > OUTPUT_LIMIT:
        output = output[:OUTPUT_LIMIT] + "\n…"
    entry.update(state=state, finished_at=_now(), exit_code=code, output=output)
    _record(tool, entry)
    return {"format": 1, "tool": tool, **entry}


def score_command(hours: int):
    def command(namespace: argparse.Namespace) -> int:
        namespace.since = datetime.now(timezone.utc) - timedelta(hours=hours)
        return ces_commands.score(namespace)

    return command, _ces_args(since=None, limit=100, force=False)


def _day(value: str, name: str) -> datetime:
    try:
        return datetime.combine(date.fromisoformat(value), datetime.min.time(), timezone.utc)
    except ValueError as exc:
        raise Refused(f"{name} must be a date such as 2026-11-01, not {value!r}.") from exc


def backfill_command(start: str, end: str | None):
    first = _day(start, "--start")
    # --end names the last day included, so the range runs to the start of the day after.
    last = _day(end, "--end") + timedelta(days=1) if end else None
    if last is not None and last <= first:
        raise Refused("--end must be on or after --start.")
    return ces_commands.backfill, _ces_args(start=first, end=last, force=False)


def monitor_command(webhook: str | None):
    from weave_agent_signals.cli import cmd_monitor

    namespace = _ces_args(
        limit=FEED_LIMIT,
        alert_webhook=webhook or None,
        state_file=str(state_dir() / MONITOR_STATE_FILE),
    )
    return cmd_monitor, namespace


def insights_command(kind: str):
    from weave_agent_signals.cli import cmd_analyze

    if kind not in INSIGHT_KINDS:
        raise Refused(f"--kind is one of {', '.join(INSIGHT_KINDS)}.")
    namespace = _ces_args(
        limit=FEED_LIMIT,
        ab=False,
        trends=False,
        coaching=False,
        subagents=kind == "subagents",
        judges=kind == "judges",
        segments="repo" if kind == "segments" else None,
    )
    return cmd_analyze, namespace
