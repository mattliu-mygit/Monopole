from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Sequence

from weave_agent_signals.client import WeaveClient
from weave_agent_signals.models import SessionView
from weave_agent_signals.scorers.efficiency import score_turn_efficiency
from weave_agent_signals.scorers.implicit import score_session_implicit
from weave_agent_signals.scorers.outcome import score_turn_outcomes

log = logging.getLogger("weave_agent_signals")


def _parse_datetime(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Cannot parse datetime: {s}")


def cmd_score(args: argparse.Namespace) -> int:
    since = args.since or (datetime.now(timezone.utc) - timedelta(hours=24))

    client = WeaveClient(entity=args.entity, project=args.project)
    turns = client.query_turns(limit=args.limit, since=since)

    if not turns:
        print("No turns found.")
        return 0

    total_scored = 0
    total_written = 0
    all_tags: list[str] = []

    for turn in turns:
        client.hydrate_turn_children(turn)
        scores = []
        scores.extend(score_turn_outcomes(turn))
        scores.append(score_turn_efficiency(turn))

        for score in scores:
            total_scored += 1
            if args.dry_run:
                print(f"  [{score.scorer}] {score.value} {score.tags}")
                continue

            existing = client.query_existing_feedback(
                turn.ref, f"weave_agent_signals.{score.scorer}"
            )
            if existing and not args.force:
                continue

            client.write_score(score, turn.ref)
            total_written += 1
            all_tags.extend(score.tags)

    print(f"Scored {len(turns)} turns ({total_scored} scores). Wrote {total_written}.")
    if all_tags:
        tag_counts = Counter(all_tags)
        print("  Tags: " + ", ".join(f"{k}={v}" for k, v in tag_counts.most_common(10)))

    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    client = WeaveClient(entity=args.entity, project=args.project)
    turns = client.query_turns(limit=1000, since=args.start)

    if args.end:
        turns = [t for t in turns if t.started_at <= args.end]

    if not turns:
        print("No turns found in date range.")
        return 0

    batch: list[tuple] = []

    for turn in turns:
        client.hydrate_turn_children(turn)
        scores = []
        scores.extend(score_turn_outcomes(turn))
        scores.append(score_turn_efficiency(turn))

        for score in scores:
            batch.append((score, turn.ref))

    # group turns by session for session-level scores
    sessions: dict[str, list] = {}
    for turn in turns:
        sessions.setdefault(turn.conversation_id, []).append(turn)

    for conv_id, sess_turns in sessions.items():
        session = SessionView(
            conversation_id=conv_id,
            turns=sorted(sess_turns, key=lambda t: t.started_at),
            config_version=sess_turns[0].config_version if sess_turns else None,
            git_branch=sess_turns[0].git_branch if sess_turns else None,
        )
        ref = session.ref
        for score in score_session_implicit(session):
            batch.append((score, ref))

    if args.dry_run:
        for score, ref in batch:
            print(f"  [{score.scorer}] {score.value} → {ref.split('/')[-1][:12]}")
        print(f"\nDry run: {len(batch)} scores computed, 0 written.")
        return 0

    batch_size = args.batch_size
    written = 0
    for i in range(0, len(batch), batch_size):
        chunk = batch[i : i + batch_size]
        client.write_scores_batch(chunk)
        written += len(chunk)

    print(f"Backfilled {written} scores across {len(turns)} turns, {len(sessions)} sessions.")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    client = WeaveClient(entity=args.entity, project=args.project)

    if args.recent:
        turns = client.query_turns(limit=args.recent)
        for turn in turns:
            print(f"\n{'='*60}")
            print(f"Turn {turn.trace_id[:12]}  [{turn.started_at}]")
            print(f"  config={turn.config_version}  steering={turn.steering_count}  "
                  f"denials={turn.denial_count}  errors={turn.tool_error_count}")
            print(f"  tokens: in={turn.input_tokens} out={turn.output_tokens} "
                  f"cache={turn.cache_read_tokens}")
    elif args.session:
        session = client.query_session(args.session)
        print(f"Session {session.conversation_id}")
        print(f"  Turns: {len(session.turns)}")
        print(f"  Tokens: {session.total_tokens}")
        for i, t in enumerate(session.turns):
            print(f"  [{i}] {t.trace_id[:12]}  steer={t.steering_count} "
                  f"deny={t.denial_count} err={t.tool_error_count}")
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
    p_back.add_argument("--batch-size", type=int, default=50)
    p_back.add_argument("--dry-run", action="store_true")

    # inspect
    p_insp = subs.add_parser("inspect", help="Inspect turns or sessions")
    p_insp.add_argument("--recent", type=int, default=None)
    p_insp.add_argument("--session", type=str, default=None)

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
        except Exception as e:
            log.error("Error: %s", e)
            if args.verbose:
                raise
            print(f"Error: {e}", file=sys.stderr)
            return 3
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
