"""`score`, `backfill`, and `inspect` over CES sessions.

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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import cached_property
from typing import Any, NamedTuple

from weave_agent_signals import ces_queries
from weave_agent_signals.ces_catalog import Catalog
from weave_agent_signals.ces_client import (
    CesClient,
    CesSession,
    SubagentCounts,
    plan_judge_writes,
    session_or_none,
)
from weave_agent_signals.judges.inference import DEFAULT_ENTITY, DEFAULT_PROJECT
from weave_agent_signals.models import (
    ConfigurationError,
    Score,
    SessionView,
    TurnSpan,
    session_is_evaluable,
)
from weave_agent_signals.patterns import ConfigStats
from weave_agent_signals.scorers import score_session, score_trajectory, score_turn
from weave_agent_signals.scorers.outcome import classify_command
from weave_agent_signals.scorers.session import scored_turns, session_dimensions
from weave_agent_signals.stats import Stat

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


def _adapter_state() -> dict:
    path = os.path.expanduser(
        os.environ.get("WEAVE_AGENT_ADAPTER_EVALUATIONS_STATE", ADAPTER_STATE)
    )
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def inference_project(entity: str | None, project: str | None) -> tuple[str, str]:
    """The W&B entity and project judge inference runs under: explicit values, then the
    adapter's project, which the key can always use, then weave-team/agent-sessions."""
    adapter_entity, _, adapter_project = str(_adapter_state().get("project") or "").partition("/")
    return (
        entity or adapter_entity or DEFAULT_ENTITY,
        project or adapter_project or DEFAULT_PROJECT,
    )


def ces_settings(
    url: str | None = None, scope: str | None = None, dataset: str | None = None
) -> tuple[str, str, str]:
    """The CES URL, scope, and Dataset: explicit values, then env vars, then the adapter's state."""
    state = _adapter_state()
    url = url or os.environ.get("WEAVE_AGENT_SIGNALS_CES_URL") or DEFAULT_CES_URL
    scope = scope or os.environ.get("WEAVE_AGENT_SIGNALS_CES_SCOPE_REF") or state.get("scope_ref")
    dataset = (
        dataset or os.environ.get("WEAVE_AGENT_SIGNALS_CES_DATASET_ID") or state.get("dataset_id")
    )
    if not scope or not dataset:
        raise ConfigurationError(
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
    return open_session(args.ces_url, args.ces_scope_ref, args.ces_dataset_id)


class _Stats:
    def __init__(self) -> None:
        self.turns = self.sessions = self.subagents = 0
        self.scored = self.written = self.skipped = self.errors = 0
        self.bash_commands = self.bash_classified = 0
        self.tags: list[str] = []

    def count_bash(self, turn: TurnSpan) -> None:
        """Count the turn's Bash commands, and those the outcome scorers classify."""
        for call in turn.tool_calls:
            if call.tool_name != "Bash":
                continue
            self.bash_commands += 1
            try:
                command = json.loads(call.arguments).get("command", "")
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
            if command and classify_command(command) is not None:
                self.bash_classified += 1


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
    """Score each session's turns in the window, then the session, its whole-Trajectory Scores,
    and each subagent Trajectory below it.

    Without `force`, a Scorer that already has a Score on its target is skipped, except a
    whole-Trajectory Scorer whose Score recorded other turns than the Trajectory has now
    (`scored_turns`): a conversation can stay live, so its Scores follow its new turns."""
    pass_id, stats = str(uuid.uuid4()), _Stats()
    session_ids = list(session_ids)
    # Two reads per session: its own row, and one filtered read for every subagent below it.
    have_sessions = {} if force else client.session_scores(catalog, session_ids, subagents=True)
    for sid in session_ids:
        if limit is not None and stats.turns >= limit:
            break
        try:
            session, subagent_views = client.read_session_tree(sid)
        except Exception as error:
            stats.errors += 1
            log.warning("Error reading session %s: %s", sid[:12], error)
            continue
        if not session_is_evaluable(session.turns):
            continue
        try:
            have_turns = {} if force else client.turn_scores(catalog, sid, details=False)
        except Exception as error:
            log.warning("Could not read existing Scores for %s: %s", sid, error)
            stats.errors += 1
            continue
        for turn in session.turns:
            if turn.started_at < since or (until is not None and turn.started_at > until):
                continue
            if limit is not None and stats.turns >= limit:
                break
            stats.turns += 1
            stats.count_bash(turn)
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
            dimensions = session_dimensions(session, client.root_case_input(sid))
        except Exception as error:
            stats.errors += 1
            log.warning("Could not read the root Case of %s: %s", sid[:12], error)
            continue
        # The session's Scores, then the whole-Trajectory Scores of it and each subagent, all
        # stamped with the session's config and branch and sliced by the session's dimensions.
        for view in [session, *subagent_views]:
            try:
                whole, turns = score_trajectory(view, dimensions), scored_turns(view)
                for s in whole:
                    s.metadata["scored_turns"] = turns
                scores = score_session(view) + whole if view is session else whole
                run_time = view.turns[0].started_at if view.turns else None
                for s in scores:
                    s.stamp(
                        config_version=session.config_version,
                        git_branch=session.git_branch,
                        run_time=run_time,
                    )
                tid = view.conversation_id
                rows = have_sessions.get(tid, [])
                recorded = catalog.recorded(rows)
                changed = {
                    s.scorer
                    for s in whole
                    if recorded.get(s.scorer, {}).get("scored_turns") != turns
                }
                have = catalog.scored(rows) - changed
                _write(client, catalog, pass_id, "trajectory", tid, scores, have, force, stats)
                if view is session:
                    stats.sessions += 1
                elif scores:
                    stats.subagents += 1
            except Exception as error:
                stats.errors += 1
                log.warning("Error scoring Trajectory %s: %s", view.conversation_id[:12], error)
    return stats


def score(args: argparse.Namespace) -> int:
    since = args.since or (datetime.now(timezone.utc) - timedelta(hours=24))
    with connect(args) as ces:
        client, catalog = ces.client, ces.catalog
        session_ids = client.find_sessions(catalog, since=since)
        if not session_ids:
            print("No turns found.")
            return 0
        stats = _score_sessions(
            client,
            catalog,
            session_ids,
            since=since,
            until=None,
            force=args.force,
            limit=args.limit,
        )
    print(
        f"Scored {stats.turns} turns, {stats.sessions} sessions, {stats.subagents} subagents"
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
        stats = _score_sessions(
            client, catalog, session_ids, since=args.start, until=args.end, force=args.force
        )
    commands, classified = stats.bash_commands, stats.bash_classified
    pct = classified / commands * 100 if commands else 0
    print(
        f"Backfilled {stats.written} scores across {stats.turns} turns, {stats.sessions} sessions,"
        f" {stats.subagents} subagents."
    )
    if stats.skipped:
        print(f"  Skipped: {stats.skipped} (existing, use --force to write newer scores)")
    print(f"  Bash coverage: {classified}/{commands} classified ({pct:.0f}%)")
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
            session_ids = client.find_sessions(catalog, since=None, limit=args.recent)
        # The listing shows each turn's subagent count, so counts stand in for their Cases.
        counts = SubagentCounts(client, catalog)
        sessions = [
            s
            for sid in session_ids
            if (s := session_or_none(client, sid, counts=counts)) is not None
        ]
        session_rows = client.session_scores(catalog, session_ids) if args.scores else {}
        for session in sessions:
            turn_rows = client.turn_scores(catalog, session.conversation_id) if args.scores else {}
            _print_session(
                catalog,
                session,
                turn_rows,
                session_rows.get(session.conversation_id, []),
                args.scores,
            )
    return 0


def sessions_to_judge(
    client, catalog: Catalog, *, since: datetime, limit: int
) -> list[SessionView]:
    """The newest `limit` sessions with a turn since `since` that are agent sessions throughout.
    The judge renders each subagent's type and tool-call count only, so counts stand in for
    their Cases."""
    counts = SubagentCounts(client, catalog)
    sessions = [
        client.read_session(sid, counts=counts)
        for sid in client.find_sessions(catalog, since=since, limit=limit)
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


def read_scores(args: argparse.Namespace) -> list[dict]:
    """The Scores on the newest `args.limit` turns and their sessions as feedback records."""
    with connect(args) as ces:
        client, catalog = ces.client, ces.catalog
        return client.read_all_scores(catalog, limit=args.limit)


class Insights(NamedTuple):
    """Query results for `analyze`'s opt-in insights; None where not asked for."""

    subagents: dict[str, dict[int, ces_queries.Group]] | None
    judge_tables: list[tuple[str, str, dict[tuple, int]]] | None
    segments: dict[tuple[str, Any], dict[str, Stat]] | None


class AnalysisReads:
    """One `analyze` or `monitor` run's CES reads. They share one connection, so the catalog is
    ensured once, one QueryState, and one read of the Scorers the sessions Evaluation selects,
    which is a PATCH."""

    def __init__(self, ces: CesSession, limit: int | None) -> None:
        self.client, self.catalog, self.limit = ces.client, ces.catalog, limit
        self.state = ces_queries.QueryState()

    @cached_property
    def session_scorers(self) -> list[ces_queries.QueryScorer]:
        return ces_queries.session_scorers(self.client, self.catalog)

    def stats(
        self, *, feedback: bool, success: bool = False, trends: bool = False
    ) -> tuple[ConfigStats, list[dict]]:
        """Every config's Score stats from CES queries, with each config's session success and
        spend when `success`, each session and turn Scorer's stats per day when `trends`, and,
        with `feedback`, the Scores on the newest `limit` turns and their sessions as feedback
        records."""
        client, catalog, state = self.client, self.catalog, self.state
        stats = ces_queries.config_stats(client, catalog, state, self.session_scorers)
        if success:
            found = ces_queries.session_success_by_config(client, catalog, state)
            stats = replace(stats, success=found)
        if trends:
            days = ces_queries.session_day_stats(client, catalog, state, self.session_scorers)
            turn_days = ces_queries.turn_day_stats(client, catalog, state)
            stats = replace(stats, days=days, turn_days=turn_days)
        records = (
            client.read_all_scores(catalog, limit=self.limit, outcome_details=False)
            if feedback
            else []
        )
        return stats, records

    def insights(self, *, subagents: bool, judges: bool, segments: str | None) -> Insights:
        """The asked-for insights, from CES queries over every Trajectory and session Score."""
        client, catalog, state = self.client, self.catalog, self.state
        return Insights(
            ces_queries.subagent_outcomes(client, catalog, state) if subagents else None,
            ces_queries.judge_tables(client, catalog, state, self.session_scorers)
            if judges
            else None,
            ces_queries.segment_stats(client, catalog, state, self.session_scorers, segments)
            if segments
            else None,
        )


@contextmanager
def analysis_reads(args: argparse.Namespace) -> Iterator[AnalysisReads]:
    with connect(args) as ces:
        yield AnalysisReads(ces, args.limit)
