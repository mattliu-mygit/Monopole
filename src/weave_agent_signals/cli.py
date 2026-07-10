from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Sequence

import httpx

from weave_agent_signals.client import WeaveClient
from weave_agent_signals.models import Score, SessionView
from weave_agent_signals.scorers import score_session, score_turn
from weave_agent_signals.scorers.outcome import classify_command

log = logging.getLogger("weave_agent_signals")


def _parse_datetime(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Cannot parse datetime: {s}")


class _WriteStats:
    """Accumulates scoring stats across turns and sessions."""

    def __init__(self) -> None:
        self.total_scored = 0
        self.total_written = 0
        self.skipped = 0
        self.errors = 0
        self.all_tags: list[str] = []

    def write_score(
        self,
        client: WeaveClient,
        score: Score,
        ref: str,
        *,
        dry_run: bool,
        force: bool,
        dry_run_label: str = "",
    ) -> None:
        self.total_scored += 1
        if dry_run:
            if dry_run_label:
                print(f"  [{score.scorer}] {score.value} {dry_run_label}")
            return
        feedback_type = f"weave_agent_signals.{score.scorer}"
        existing = client.query_existing_feedback(ref, feedback_type)
        if existing and not force:
            self.skipped += 1
            return
        if existing and force:
            client.delete_feedback_ids(existing)
        client.write_score(score, ref)
        self.total_written += 1
        self.all_tags.extend(score.tags)


def _score_sessions(
    client: WeaveClient,
    turns: list,
    stats: _WriteStats,
    args: argparse.Namespace,
) -> int:
    sessions: dict[str, list] = {}
    for turn in turns:
        sessions.setdefault(turn.conversation_id, []).append(turn)

    sessions_scored = 0
    for conv_id, sess_turns in sessions.items():
        session = SessionView(
            conversation_id=conv_id,
            turns=sorted(sess_turns, key=lambda t: t.started_at),
            config_version=sess_turns[0].config_version if sess_turns else None,
            git_branch=sess_turns[0].git_branch if sess_turns else None,
        )
        sess_ref = session.ref_for(args.entity, args.project)
        try:
            for s in score_session(session):
                s.metadata.setdefault("config_version", session.config_version)
                s.metadata.setdefault("git_branch", session.git_branch)
                stats.write_score(
                    client, s, sess_ref,
                    dry_run=args.dry_run, force=args.force,
                    dry_run_label=f"→ {conv_id[:12]}",
                )
        except Exception as e:
            stats.errors += 1
            log.warning("Error scoring session %s: %s", conv_id[:12], e)
        sessions_scored += 1
    return sessions_scored


def cmd_score(args: argparse.Namespace) -> int:
    since = args.since or (datetime.now(timezone.utc) - timedelta(hours=24))

    with WeaveClient(entity=args.entity, project=args.project) as client:
        turns = client.query_turns(limit=args.limit, since=since)

        if not turns:
            print("No turns found.")
            return 0

        stats = _WriteStats()

        for turn in turns:
            try:
                client.hydrate_turn_children(turn)
                for s in score_turn(turn):
                    s.metadata.setdefault("config_version", turn.config_version)
                    s.metadata.setdefault("git_branch", turn.git_branch)
                    turn_ref = turn.ref_for(args.entity, args.project)
                    stats.write_score(
                        client, s, turn_ref,
                        dry_run=args.dry_run, force=args.force,
                        dry_run_label=str(s.tags),
                    )
            except Exception as e:
                stats.errors += 1
                log.warning("Error scoring turn %s: %s", turn.trace_id[:12], e)

        sessions_scored = _score_sessions(client, turns, stats, args)

    print(f"Scored {len(turns)} turns, {sessions_scored} sessions ({stats.total_scored} scores). Wrote {stats.total_written}.")
    if stats.errors:
        print(f"  Errors: {stats.errors} (use -v for details)")
    if stats.all_tags:
        tag_counts = Counter(stats.all_tags)
        print("  Tags: " + ", ".join(f"{k}={v}" for k, v in tag_counts.most_common(10)))

    return 1 if stats.errors else 0


def cmd_backfill(args: argparse.Namespace) -> int:
    with WeaveClient(entity=args.entity, project=args.project) as client:
        turns = client.query_turns_paginated(page_size=args.page_size, since=args.start)

        if args.end:
            turns = [t for t in turns if t.started_at <= args.end]

        if not turns:
            print("No turns found in date range.")
            return 0

        total_bash = 0
        classified_bash = 0
        stats = _WriteStats()

        for turn in turns:
            try:
                client.hydrate_turn_children(turn)
            except Exception as e:
                stats.errors += 1
                log.warning("Error hydrating turn %s: %s", turn.trace_id[:12], e)
                continue

            for tc in turn.tool_calls:
                if tc.tool_name == "Bash":
                    total_bash += 1
                    try:
                        cmd = json.loads(tc.arguments).get("command", "")
                    except (json.JSONDecodeError, TypeError):
                        cmd = ""
                    if cmd and classify_command(cmd) is not None:
                        classified_bash += 1

            try:
                turn_ref = turn.ref_for(args.entity, args.project)
                for s in score_turn(turn):
                    s.metadata.setdefault("config_version", turn.config_version)
                    s.metadata.setdefault("git_branch", turn.git_branch)
                    stats.write_score(
                        client, s, turn_ref,
                        dry_run=args.dry_run, force=args.force,
                    )
            except Exception as e:
                stats.errors += 1
                log.warning("Error scoring turn %s: %s", turn.trace_id[:12], e)

        sessions_scored = _score_sessions(client, turns, stats, args)

    if args.dry_run:
        print(f"\nDry run: {stats.total_scored} scores computed, 0 written.")
        pct = (classified_bash / total_bash * 100) if total_bash else 0
        print(f"  Bash coverage: {classified_bash}/{total_bash} classified ({pct:.0f}%)")
        return 0

    pct = (classified_bash / total_bash * 100) if total_bash else 0
    print(f"Backfilled {stats.total_written} scores across {len(turns)} turns, {sessions_scored} sessions.")
    if stats.skipped:
        print(f"  Skipped: {stats.skipped} (existing, use --force to overwrite)")
    print(f"  Bash coverage: {classified_bash}/{total_bash} classified ({pct:.0f}%)")
    if stats.errors:
        print(f"  Errors: {stats.errors} (use -v for details)")
    return 1 if stats.errors else 0


def _print_feedback(feedback: list[dict]) -> None:
    if not feedback:
        print("    (no feedback)")
        return
    for fb in feedback:
        ftype = fb.get("feedback_type", "?")
        short = ftype.replace("weave_agent_signals.", "")
        payload = fb.get("payload", {})
        rating = payload.get("rating", "?")
        tags = payload.get("tags", [])
        reason = payload.get("reason", "")
        extra = f'  "{reason}"' if reason else ""
        print(f"    {short:30s}  rating={rating}  tags={tags}{extra}")


def cmd_inspect(args: argparse.Namespace) -> int:
    with WeaveClient(entity=args.entity, project=args.project) as client:
        if args.recent:
            turns = client.query_turns(limit=args.recent)
            for turn in turns:
                client.hydrate_turn_children(turn)
                print(f"\n{'='*60}")
                print(f"Turn {turn.trace_id[:12]}  [{turn.started_at}]")
                print(f"  config={turn.config_version}  steering={turn.steering_count}  "
                      f"denials={turn.denial_count}  errors={turn.tool_error_count}")
                print(f"  tokens: in={turn.input_tokens} out={turn.output_tokens} "
                      f"cache={turn.cache_read_tokens}")
                print(f"  tools={len(turn.tool_calls)}  subagents={len(turn.subagents)}")
                if args.feedback:
                    feedback = client.query_all_feedback(
                        turn.ref_for(args.entity, args.project))
                    print("  Feedback:")
                    _print_feedback(feedback)
        elif args.session:
            session = client.query_session(args.session)
            for t in session.turns:
                client.hydrate_turn_children(t)
            print(f"Session {session.conversation_id}")
            print(f"  Turns: {len(session.turns)}")
            print(f"  Tokens: {session.total_tokens}")
            if args.feedback:
                feedback = client.query_all_feedback(
                    session.ref_for(args.entity, args.project))
                print("  Session feedback:")
                _print_feedback(feedback)
            for i, t in enumerate(session.turns):
                print(f"  [{i}] {t.trace_id[:12]}  steer={t.steering_count} "
                      f"deny={t.denial_count} err={t.tool_error_count} "
                      f"tools={len(t.tool_calls)} subagents={len(t.subagents)}")
                if args.feedback:
                    feedback = client.query_all_feedback(
                        t.ref_for(args.entity, args.project))
                    _print_feedback(feedback)
        else:
            print("Specify --recent N or --session CONVERSATION_ID")
            return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weave-agent-signals",
        description="Score agent traces from Weave",
    )
    parser.add_argument("--entity", default="mliu-wandb-weights-biases")
    parser.add_argument("--project", default="agent-sessions")
    parser.add_argument("-v", "--verbose", action="store_true")

    subs = parser.add_subparsers(dest="command")

    # score
    p_score = subs.add_parser("score", help="Score recent unscored turns")
    p_score.add_argument("--since", type=_parse_datetime, default=None)
    p_score.add_argument("--limit", type=int, default=100)
    p_score.add_argument("--dry-run", action="store_true")
    p_score.add_argument("--force", action="store_true")

    # backfill
    p_back = subs.add_parser("backfill", help="Backfill scores for a date range")
    p_back.add_argument("--start", type=_parse_datetime, required=True)
    p_back.add_argument("--end", type=_parse_datetime, default=None)
    p_back.add_argument("--page-size", type=int, default=500)
    p_back.add_argument("--dry-run", action="store_true")
    p_back.add_argument("--force", action="store_true")

    # inspect
    p_insp = subs.add_parser("inspect", help="Inspect turns or sessions")
    p_insp.add_argument("--recent", type=int, default=None)
    p_insp.add_argument("--session", type=str, default=None)
    p_insp.add_argument("--feedback", action="store_true", help="Show feedback/scores")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "score": cmd_score,
        "backfill": cmd_backfill,
        "inspect": cmd_inspect,
    }
    handler = handlers.get(args.command)
    if handler:
        try:
            return handler(args)
        except RuntimeError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            return 2
        except httpx.HTTPStatusError as e:
            log.error("API error: %s", e)
            if args.verbose:
                raise
            print(f"API error: {e.response.status_code} {e.request.url}", file=sys.stderr)
            return 3
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            log.error("API error: %s", e)
            print(f"API error: {e}", file=sys.stderr)
            return 3
        except Exception as e:
            log.error("Error: %s", e)
            if args.verbose:
                raise
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
