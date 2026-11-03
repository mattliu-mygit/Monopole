"""`score`, `backfill`, and `inspect` over CES sessions (`--backend ces`).

Sessions are the adapter's session Trajectories. Each pass writes under its own pass
ID, so `--force` writes newer Scores and reads return the newest Score per scorer.
Without `--force`, a scorer that already has a Score on a target is skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

from weave_agent_signals.ces_catalog import Catalog
from weave_agent_signals.ces_client import CesClient, CesSession, plan_judge_writes
from weave_agent_signals.models import Score, SessionView, session_is_evaluable
from weave_agent_signals.scorers import score_session, score_turn
from weave_agent_signals.scorers.outcome import classify_command

log = logging.getLogger("weave_agent_signals")

DEFAULT_CES_URL = "https://evaluations.wandb.ai"
ADAPTER_STATE = "~/.weave-agent-adapter/evaluations.json"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ces-url", default=None, help=f"Default: {DEFAULT_CES_URL}")
    parser.add_argument(
        "--ces-scope-ref",
        default=None,
        help="Default: the adapter's evaluations.json",
    )
    parser.add_argument(
        "--ces-dataset-id",
        default=None,
        help="Default: the adapter's evaluations.json",
    )


def ces_settings(
    url: str | None = None, scope: str | None = None, dataset: str | None = None
) -> tuple[str, str, str]:
    """The CES URL, scope, and Dataset: explicit values, then env vars, then the adapter's state."""
    state: dict = {}
    path = os.path.expanduser(
        os.environ.get("WEAVE_AGENT_ADAPTER_EVALUATIONS_STATE", ADAPTER_STATE)
    )
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        pass
    url = url or os.environ.get("WEAVE_AGENT_SIGNALS_CES_URL") or DEFAULT_CES_URL
    scope = scope or os.environ.get("WEAVE_AGENT_SIGNALS_CES_SCOPE_REF") or state.get("scope_ref")
    dataset = (
        dataset or os.environ.get("WEAVE_AGENT_SIGNALS_CES_DATASET_ID") or state.get("dataset_id")
    )
    if not scope or not dataset:
        raise RuntimeError(
            "CES needs --ces-scope-ref and --ces-dataset-id, "
            "or `weave-agent-adapter evaluations setup`"
        )
    return url, scope, dataset


def open_session(
    url: str | None = None, scope: str | None = None, dataset: str | None = None
) -> CesSession:
    url, scope, dataset = ces_settings(url, scope, dataset)
    client = CesClient(url, scope)
    try:
        return CesSession(client, client.ensure_catalog(dataset))
    except Exception:
        client.close()
        raise


def connect(args: argparse.Namespace) -> CesSession:
    try:
        return open_session(args.ces_url, args.ces_scope_ref, args.ces_dataset_id)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


class _Stats:
    def __init__(self) -> None:
        self.turns = self.sessions = self.scored = self.written = self.skipped = self.errors = 0
        self.tags: list[str] = []


def _write(
    client, catalog, pass_id, kind, target, scores: list[Score], have: set[str], force, stats
) -> None:
    stats.scored += len(scores)
    todo = [s for s in scores if force or s.scorer not in have]
    stats.skipped += len(scores) - len(todo)
    client.write_scores(catalog, pass_id, kind, target, todo)
    stats.written += len(todo)
    stats.tags += [tag for s in todo for tag in s.tags]


def _score_sessions(client, catalog, session_ids, *, since, until, force, limit=None):
    """Score each session's turns in the window, then the session. Returns (stats, sessions)."""
    pass_id, stats, sessions = str(uuid.uuid4()), _Stats(), []
    session_ids = list(session_ids)
    # One read for every session: the session-row read pages all session rows each time.
    have_sessions = {} if force else client.session_scores(catalog, session_ids)
    for sid in session_ids:
        if limit is not None and stats.turns >= limit:
            break
        try:
            session = client.read_session(sid)
        except Exception as error:
            stats.errors += 1
            log.warning("Error reading session %s: %s", sid[:12], error)
            continue
        if not session_is_evaluable(session.turns):
            continue
        sessions.append(session)
        have_turns = {} if force else client.turn_scores(catalog, sid)
        for turn in session.turns:
            if turn.started_at < since or (until is not None and turn.started_at > until):
                continue
            if limit is not None and stats.turns >= limit:
                break
            stats.turns += 1
            try:
                scores = score_turn(turn)
                for s in scores:
                    s.stamp(
                        config_version=turn.config_version,
                        git_branch=turn.git_branch,
                        run_time=turn.started_at,
                    )
                have = catalog.scored(have_turns.get(turn.trace_id, []))
                _write(client, catalog, pass_id, "case", turn.trace_id, scores, have, force, stats)
            except Exception as error:
                stats.errors += 1
                log.warning("Error scoring turn %s: %s", turn.trace_id[:12], error)
        try:
            scores = score_session(session)
            run_time = session.turns[0].started_at if session.turns else None
            for s in scores:
                s.stamp(
                    config_version=session.config_version,
                    git_branch=session.git_branch,
                    run_time=run_time,
                )
            have = catalog.scored(have_sessions.get(sid, []))
            _write(client, catalog, pass_id, "trajectory", sid, scores, have, force, stats)
            stats.sessions += 1
        except Exception as error:
            stats.errors += 1
            log.warning("Error scoring session %s: %s", sid[:12], error)
    return stats, sessions


def score(args: argparse.Namespace) -> int:
    since = args.since or (datetime.now(timezone.utc) - timedelta(hours=24))
    with connect(args) as ces:
        client, catalog = ces.client, ces.catalog
        session_ids = client.find_sessions(catalog, since=since)
        if not session_ids:
            print("No turns found.")
            return 0
        stats, _ = _score_sessions(
            client,
            catalog,
            session_ids,
            since=since,
            until=None,
            force=args.force,
            limit=args.limit,
        )
    print(
        f"Scored {stats.turns} turns, {stats.sessions} sessions"
        f" ({stats.scored} scores). Wrote {stats.written}."
    )
    if stats.errors:
        print(f"  Errors: {stats.errors} (use -v for details)")
    if stats.tags:
        print("  Tags: " + ", ".join(f"{k}={v}" for k, v in Counter(stats.tags).most_common(10)))
    return 1 if stats.errors else 0


def backfill(args: argparse.Namespace) -> int:
    with connect(args) as ces:
        client, catalog = ces.client, ces.catalog
        session_ids = client.find_sessions(catalog, since=args.start, until=args.end)
        if not session_ids:
            print("No turns found in date range.")
            return 0
        stats, sessions = _score_sessions(
            client, catalog, session_ids, since=args.start, until=args.end, force=args.force
        )
    commands = []
    for session in sessions:
        for turn in session.turns:
            in_range = turn.started_at >= args.start and (
                args.end is None or turn.started_at <= args.end
            )
            for call in turn.tool_calls if in_range else ():
                if call.tool_name == "Bash":
                    try:
                        commands.append(json.loads(call.arguments).get("command", ""))
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        commands.append("")
    classified = sum(1 for command in commands if command and classify_command(command) is not None)
    pct = classified / len(commands) * 100 if commands else 0
    print(
        f"Backfilled {stats.written} scores across {stats.turns} turns, {stats.sessions} sessions."
    )
    if stats.skipped:
        print(f"  Skipped: {stats.skipped} (existing, use --force to write newer scores)")
    print(f"  Bash coverage: {classified}/{len(commands)} classified ({pct:.0f}%)")
    if stats.errors:
        print(f"  Errors: {stats.errors} (use -v for details)")
    return 1 if stats.errors else 0


def _print_scores(catalog: Catalog, rows: list[dict]) -> None:
    names = catalog.by_scorer_id()
    values, details = {}, {}
    for row in rows:
        name, is_details = names.get(row["scorer_id"], (row["scorer_id"], False))
        (details if is_details else values)[name] = row["value"]
    if not values:
        print("    (no scores)")
    for name, value in sorted(values.items()):
        extra = details.get(name) or {}
        reason = f'  "{extra["reason"]}"' if extra.get("reason") else ""
        print(f"    {name:30s}  rating={value}  tags={extra.get('tags', [])}{reason}")


def _print_session(
    catalog: Catalog, session: SessionView, turn_rows, session_rows, show_scores: bool
) -> None:
    print(f"Session {session.conversation_id}")
    print(f"  Turns: {len(session.turns)}")
    print(f"  Tokens: {session.total_tokens}")
    if show_scores:
        print("  Session scores:")
        _print_scores(catalog, session_rows)
    for i, t in enumerate(session.turns):
        print(
            f"  [{i}] {t.trace_id[:12]}  status={t.status_code} steer={t.steering_count} "
            f"deny={t.denial_count} err={t.tool_error_count} tools={len(t.tool_calls)} "
            f"subagents={len(t.subagents)} role={t.trace_role.value}"
        )
        if show_scores:
            _print_scores(catalog, turn_rows.get(t.trace_id, []))


def inspect(args: argparse.Namespace) -> int:
    if not args.session and not args.recent:
        print("Specify --recent N or --session SESSION_TRAJECTORY_ID")
        return 1
    with connect(args) as ces:
        client, catalog = ces.client, ces.catalog
        if args.session:
            session_ids = [args.session]
        else:
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            session_ids = client.find_sessions(catalog, since=epoch, limit=args.recent)
        sessions = [client.read_session(sid) for sid in session_ids]
        session_rows = client.session_scores(catalog, session_ids) if args.feedback else {}
        for session in sessions:
            turn_rows = (
                client.turn_scores(catalog, session.conversation_id) if args.feedback else {}
            )
            _print_session(
                catalog,
                session,
                turn_rows,
                session_rows.get(session.conversation_id, []),
                args.feedback,
            )
    return 0


def sessions_to_judge(
    client, catalog: Catalog, *, since: datetime, limit: int
) -> list[SessionView]:
    """The newest `limit` sessions with a turn since `since` that are agent sessions throughout."""
    sessions = [
        client.read_session(sid) for sid in client.find_sessions(catalog, since=since, limit=limit)
    ]
    return [
        session for session in sessions if session.turns and session_is_evaluable(session.turns)
    ]


def write_judge_scores(client, catalog: Catalog, pending: list, *, force: bool) -> int:
    """Write each finished review's Score under its panel's Scorer. Returns the count written."""
    catalog, writes = plan_judge_writes(client, catalog, pending, force=force)
    pass_id = str(uuid.uuid4())
    for sid, scores in writes.items():
        client.write_scores(catalog, pass_id, "trajectory", sid, scores)
    return sum(map(len, writes.values()))


def read_feedback(args: argparse.Namespace) -> list[dict]:
    """The newest `args.limit` Scores as feedback records, for analyze, monitor, and reflect."""
    with connect(args) as ces:
        client, catalog = ces.client, ces.catalog
        return client.read_all_scores(catalog)[: args.limit]
