"""The Tools screen's actions: `score`, `backfill`, `monitor`, and `insights`, the commands that
were terminal-only, run with the same code as `weave-agent-signals score` and the rest.

Prompting Time runs an action to its end with no time limit and runs other actions alongside it,
so a long backfill runs in its action's own process without holding anything else up. Each tool
records itself in `<state folder>/tools.json` when it starts and when it ends, with what it
printed, so the page shows a tool that is still running, and its last result, after it reloads;
Prompting Time's watcher tells the page when that file changes. The commands report no progress
while they run, so the page shows only when the tool started.

The insights also record their results as tables, built from the analysis functions' values, which
the page shows instead of the printed text; score, backfill, and monitor print only counts.
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
from weave_agent_signals.patterns import agreement, segment_gaps, subagent_gaps
from weave_agent_signals.plugin.confirm import Refused
from weave_agent_signals.plugin.settings import state_dir, write_atomic
from weave_agent_signals.stats import excludes_zero

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


def run(
    tool: str,
    args: dict[str, Any],
    command,
    namespace: argparse.Namespace,
    tables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one of the CLI's commands, capturing what it prints and logs, and record it, with
    the tables the command added to `tables`, when it takes them."""

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
    if tables is not None:
        entry["tables"] = tables
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


def _interval(ci: tuple[float, float] | None) -> str:
    return "—" if ci is None else f"{ci[0]:+.2f} to {ci[1]:+.2f}"


def _rate(mean: float, binary: bool) -> str:
    return f"{mean:.0%}" if binary else f"{mean:.2f}"


def insight_tables(found: ces_commands.Insights) -> list[dict[str, Any]]:
    """The insights as tables of formatted cells, from the same analysis functions `analyze`
    prints from. An insight with nothing to show has no table; the page shows the text then."""

    tables = []
    if found.subagents:
        counts, gaps = [], []
        for config, depths in sorted(found.subagents.items()):
            sessions = depths[0].targets if 0 in depths else 0
            subagents = sum(group.targets for depth, group in depths.items() if depth > 0)
            per_session = f"{subagents / sessions:.2f}" if sessions else "—"
            counts.append([config, str(sessions), str(subagents), per_session])
            for g in subagent_gaps({depth: group.stats for depth, group in depths.items()}):
                gaps.append(
                    [
                        config,
                        g["scorer"],
                        f"{_rate(g['session_mean'], g['binary'])} (n={g['session_count']})",
                        f"{_rate(g['subagent_mean'], g['binary'])} (n={g['subagent_count']})",
                        f"{g['delta']:+.2f}",
                        _interval(g["difference_ci"]),
                        "yes" if g["significant"] else "",
                    ]
                )
        tables.append(
            {
                "title": "Sessions and subagents by config",
                "columns": ["Config", "Sessions", "Subagents", "Per session"],
                "rows": counts,
            }
        )
        if gaps:
            tables.append(
                {
                    "title": "Outcomes in sessions vs subagents",
                    "columns": [
                        "Config",
                        "Scorer",
                        "Sessions",
                        "Subagents",
                        "Difference",
                        "95% interval",
                        "Significant",
                    ],
                    "rows": gaps,
                }
            )
    if found.judge_tables:
        rows = []
        for item in (agreement(*table) for table in found.judge_tables):
            agree = when_passes = when_fails = gap = "—"
            significant = ""
            if item.count:
                lo, hi = item.ci
                agree = f"{item.agree / item.count:.0%} ({lo:.0%}–{hi:.0%})"
            if item.passing is not None and item.failing is not None:
                when_passes, when_fails = f"{item.passing.mean:.2f}", f"{item.failing.mean:.2f}"
                gap = f"{item.passing.mean - item.failing.mean:+.2f} ({_interval(item.gap_ci)})"
                significant = "yes" if excludes_zero(item.gap_ci) else ""
            rows.append(
                [
                    item.judge,
                    item.reference,
                    str(item.count),
                    agree,
                    when_passes,
                    when_fails,
                    gap,
                    significant,
                ]
            )
        tables.append(
            {
                "title": "Judge agreement",
                "columns": [
                    "Judge",
                    "Reference",
                    "Sessions",
                    "Agree (95% interval)",
                    "Judge mean, reference passes",
                    "Judge mean, reference fails",
                    "Gap (95% interval)",
                    "Significant",
                ],
                "rows": rows,
            }
        )
    if found.segments:
        rows = [
            [
                g["config"],
                "unknown" if g["segment"] is None else str(g["segment"]),
                g["scorer"],
                f"{_rate(g['mean'], g['binary'])} (n={g['sample_count']})",
                f"{_rate(g['rest_mean'], g['binary'])} (n={g['rest_sample_count']})",
                f"{g['delta']:+.2f}",
                _interval(g["difference_ci"]),
                "yes" if g["significant"] else "",
            ]
            for g in segment_gaps(found.segments)
        ]
        if rows:
            tables.append(
                {
                    "title": "Segments by repo, each against the rest of its config",
                    "columns": [
                        "Config",
                        "Repo",
                        "Scorer",
                        "Segment",
                        "Rest",
                        "Difference",
                        "95% interval",
                        "Significant",
                    ],
                    "rows": rows,
                }
            )
    return tables


def insights_command(kind: str):
    """One insight, as `analyze` prints it, and a list the command fills with its tables."""

    from weave_agent_signals.cli import print_insights

    if kind not in INSIGHT_KINDS:
        raise Refused(f"--kind is one of {', '.join(INSIGHT_KINDS)}.")
    segments = "repo" if kind == "segments" else None
    tables: list[dict[str, Any]] = []

    def command(namespace: argparse.Namespace) -> int:
        with ces_commands.analysis_reads(namespace) as reads:
            found = reads.insights(
                subagents=kind == "subagents", judges=kind == "judges", segments=segments
            )
        tables.extend(insight_tables(found))
        print_insights(found, segments)
        return 0

    return command, _ces_args(limit=FEED_LIMIT), tables
