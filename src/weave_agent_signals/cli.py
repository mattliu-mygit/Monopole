from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Sequence

import httpx
from dotenv import load_dotenv

from weave_agent_signals import alerts
from weave_agent_signals.client import WeaveClient
from weave_agent_signals.judges.cli_backend import CliJudgeClient
from weave_agent_signals.judges.inference import InferenceClient
from weave_agent_signals.judges.rubrics import RUBRICS, SESSION_RUBRICS
from weave_agent_signals.judges.runner import (
    judge_default_model,
    judge_session,
    judge_turn,
)
from weave_agent_signals.models import Score, SessionView
from weave_agent_signals.patterns import (
    ab_leaderboard,
    coaching_digest,
    detect_config_regressions,
    detect_regressions,
)
from weave_agent_signals.reflector import (
    extract_artifacts,
    render_proposal_diff,
    run_reflection,
)
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


def _make_judge_client(args: argparse.Namespace):
    """Build the judge inference client for the requested backend.

    backend="cli" uses the local coding-agent CLIs (TEMPORARY, no credits);
    anything else is the HTTP InferenceClient (openai / wandb / custom URL).
    """
    backend = getattr(args, "judge_backend", None)
    if backend == "cli":
        return CliJudgeClient()
    return InferenceClient(entity=args.entity, project=args.project, backend=backend)


def _stamp_metadata(score: Score, *, config_version, git_branch, run_time) -> None:
    """Attach the fields analysis needs but scorers don't set themselves.

    ``run_time`` is the agent's run timestamp (turn/session start) — trend
    detection orders by this, not by when the score was written.
    """
    score.metadata.setdefault("config_version", config_version)
    score.metadata.setdefault("git_branch", git_branch)
    if run_time is not None:
        score.metadata.setdefault("turn_started_at", run_time.isoformat())


def _group_sessions(turns: list) -> list[SessionView]:
    """Group turns into per-conversation SessionViews (turns sorted by start)."""
    by_conv: dict[str, list] = {}
    for turn in turns:
        by_conv.setdefault(turn.conversation_id, []).append(turn)

    sessions = []
    for conv_id, sess_turns in by_conv.items():
        ordered = sorted(sess_turns, key=lambda t: t.started_at)
        sessions.append(
            SessionView(
                conversation_id=conv_id,
                turns=ordered,
                config_version=ordered[0].config_version if ordered else None,
                git_branch=ordered[0].git_branch if ordered else None,
            )
        )
    return sessions


def _score_sessions(
    client: WeaveClient,
    turns: list,
    stats: _WriteStats,
    args: argparse.Namespace,
) -> int:
    sessions_scored = 0
    for session in _group_sessions(turns):
        conv_id = session.conversation_id
        sess_ref = session.ref_for(args.entity, args.project)
        try:
            run_time = session.turns[0].started_at if session.turns else None
            for s in score_session(session):
                _stamp_metadata(
                    s,
                    config_version=session.config_version,
                    git_branch=session.git_branch,
                    run_time=run_time,
                )
                stats.write_score(
                    client,
                    s,
                    sess_ref,
                    dry_run=args.dry_run,
                    force=args.force,
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
                turn_ref = turn.ref_for(args.entity, args.project)
                for s in score_turn(turn):
                    _stamp_metadata(
                        s,
                        config_version=turn.config_version,
                        git_branch=turn.git_branch,
                        run_time=turn.started_at,
                    )
                    stats.write_score(
                        client,
                        s,
                        turn_ref,
                        dry_run=args.dry_run,
                        force=args.force,
                        dry_run_label=str(s.tags),
                    )
            except Exception as e:
                stats.errors += 1
                log.warning("Error scoring turn %s: %s", turn.trace_id[:12], e)

        sessions_scored = _score_sessions(client, turns, stats, args)

    print(
        f"Scored {len(turns)} turns, {sessions_scored} sessions"
        f" ({stats.total_scored} scores). Wrote {stats.total_written}."
    )
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
                    _stamp_metadata(
                        s,
                        config_version=turn.config_version,
                        git_branch=turn.git_branch,
                        run_time=turn.started_at,
                    )
                    stats.write_score(
                        client,
                        s,
                        turn_ref,
                        dry_run=args.dry_run,
                        force=args.force,
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
    print(
        f"Backfilled {stats.total_written} scores across"
        f" {len(turns)} turns, {sessions_scored} sessions."
    )
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
                print(f"\n{'=' * 60}")
                print(f"Turn {turn.trace_id[:12]}  [{turn.started_at}]")
                print(
                    f"  config={turn.config_version}  steering={turn.steering_count}  "
                    f"denials={turn.denial_count}  errors={turn.tool_error_count}"
                )
                print(
                    f"  tokens: in={turn.input_tokens} out={turn.output_tokens} "
                    f"cache={turn.cache_read_tokens}"
                )
                print(f"  tools={len(turn.tool_calls)}  subagents={len(turn.subagents)}")
                if args.feedback:
                    feedback = client.query_all_feedback(turn.ref_for(args.entity, args.project))
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
                feedback = client.query_all_feedback(session.ref_for(args.entity, args.project))
                print("  Session feedback:")
                _print_feedback(feedback)
            for i, t in enumerate(session.turns):
                print(
                    f"  [{i}] {t.trace_id[:12]}  steer={t.steering_count} "
                    f"deny={t.denial_count} err={t.tool_error_count} "
                    f"tools={len(t.tool_calls)} subagents={len(t.subagents)}"
                )
                if args.feedback:
                    feedback = client.query_all_feedback(t.ref_for(args.entity, args.project))
                    _print_feedback(feedback)
        else:
            print("Specify --recent N or --session CONVERSATION_ID")
            return 1

    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    since = args.since or (datetime.now(timezone.utc) - timedelta(hours=24))
    all_rubrics = {**RUBRICS, **SESSION_RUBRICS}

    with WeaveClient(entity=args.entity, project=args.project) as client:
        turns = client.query_turns(limit=args.limit, since=since)

        if not turns:
            print("No turns found.")
            return 0

        turn_rubrics = None
        session_rubrics = None
        if args.rubric:
            rubric_names = [r.strip() for r in args.rubric.split(",")]
            turn_rubrics = [RUBRICS[n] for n in rubric_names if n in RUBRICS]
            session_rubrics = [SESSION_RUBRICS[n] for n in rubric_names if n in SESSION_RUBRICS]
            if not turn_rubrics and not session_rubrics:
                print(
                    f"Unknown rubric(s): {args.rubric}. Available: {', '.join(all_rubrics.keys())}"
                )
                return 2

        run_turns = turn_rubrics is None or bool(turn_rubrics)
        run_sessions = session_rubrics is None or bool(session_rubrics)

        stats = _WriteStats()

        # Session digests are built from the turns' tool/chat children, so the
        # turns must be hydrated whenever we judge sessions — not only when we
        # judge turns. Hydrate once up front for every turn we'll use.
        if run_turns or run_sessions:
            for turn in turns:
                try:
                    client.hydrate_turn_children(turn)
                except Exception as e:
                    stats.errors += 1
                    log.warning("Error hydrating turn %s: %s", turn.trace_id[:12], e)

        with _make_judge_client(args) as inference:
            for turn in turns if run_turns else []:
                try:
                    turn_ref = turn.ref_for(args.entity, args.project)
                    scores = judge_turn(turn, inference, rubrics=turn_rubrics or None)
                    for s in scores:
                        _stamp_metadata(
                            s,
                            config_version=turn.config_version,
                            git_branch=turn.git_branch,
                            run_time=turn.started_at,
                        )
                        stats.write_score(
                            client,
                            s,
                            turn_ref,
                            dry_run=args.dry_run,
                            force=args.force,
                            dry_run_label=s.reason,
                        )
                except Exception as e:
                    stats.errors += 1
                    log.warning("Error judging turn %s: %s", turn.trace_id[:12], e)

            if run_sessions:
                for session in _group_sessions(turns):
                    conv_id = session.conversation_id
                    run_time = session.turns[0].started_at if session.turns else None
                    try:
                        scores = judge_session(
                            session,
                            inference,
                            rubrics=session_rubrics or None,
                            panel_size=args.panel_size,
                        )
                        sess_ref = session.ref_for(args.entity, args.project)
                        for s in scores:
                            _stamp_metadata(
                                s,
                                config_version=session.config_version,
                                git_branch=session.git_branch,
                                run_time=run_time,
                            )
                            stats.write_score(
                                client,
                                s,
                                sess_ref,
                                dry_run=args.dry_run,
                                force=args.force,
                                dry_run_label=f"→ {conv_id[:12]}  {s.reason}",
                            )
                    except Exception as e:
                        stats.errors += 1
                        log.warning("Error judging session %s: %s", conv_id[:12], e)

    print(f"Judged {len(turns)} turns ({stats.total_scored} scores). Wrote {stats.total_written}.")
    if stats.errors:
        print(f"  Errors: {stats.errors} (use -v for details)")
    return 1 if stats.errors else 0


def cmd_analyze(args: argparse.Namespace) -> int:
    with WeaveClient(entity=args.entity, project=args.project) as client:
        feedback = client.query_project_feedback(limit=args.limit)

    if not feedback:
        print("No scored feedback found.")
        return 0

    if args.ab:
        results = ab_leaderboard(feedback)
        if not results:
            print("No A/B data (config_version not set on scores).")
        else:
            print(f"\nA/B Leaderboard ({len(results)} config versions):\n")
            for r in results:
                print(f"  Config: {r.config_version}  ({r.turn_count} turns)")
                for scorer, s in sorted(r.scores.items()):
                    lo, hi = s.ci
                    flag = "" if s.confident else "  ⚠ low n"
                    if s.pass_rate is not None:
                        print(
                            f"    {scorer:35s} {s.pass_rate:.0%} pass  "
                            f"(n={s.count}, CI {lo:.0%}–{hi:.0%}){flag}"
                        )
                    else:
                        print(
                            f"    {scorer:35s} mean={s.mean:.2f}  "
                            f"(n={s.count}, CI {lo:.2f}–{hi:.2f}){flag}"
                        )
                print()

    if args.trends:
        regressions = detect_regressions(feedback)
        if not regressions:
            print("No significant trends detected.")
        else:
            print(f"\nTrends ({len(regressions)} detected):\n")
            for r in regressions:
                arrow = "↓" if r["direction"] == "regression" else "↑"
                mark = "" if r["significant"] else "  (tentative)"
                print(
                    f"  {arrow} {r['scorer']:35s} "
                    f"{r['older_mean']:.2f} → {r['recent_mean']:.2f} "
                    f"({r['delta']:+.2f}, n={r['sample_count']}){mark}"
                )

    if args.coaching:
        print(coaching_digest(feedback))

    if not (args.ab or args.trends or args.coaching):
        print(coaching_digest(feedback))

    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    with WeaveClient(entity=args.entity, project=args.project) as client:
        feedback = client.query_project_feedback(limit=args.limit)

    if not feedback:
        print("No scored feedback found.")
        return 0

    trend = detect_regressions(feedback)
    config = detect_config_regressions(feedback)
    # Only downward, statistically-significant moves are alerts — an improvement
    # is significant too but is not a regression. detect_config_regressions
    # already returns drops only.
    sig_trend = [r for r in trend if r["significant"] and r["direction"] == "regression"]
    sig_config = [r for r in config if r["significant"]]

    found = [alerts.trend_alert(r) for r in sig_trend] + [
        alerts.config_alert(r) for r in sig_config
    ]
    seen = alerts.load_seen(args.state_file)
    new = [a for a in found if a.key not in seen]

    tentative = len([r for r in trend if not r["significant"]]) + len(
        [r for r in config if not r["significant"]]
    )
    if not new:
        print(
            f"No new significant regressions "
            f"({len(found)} already alerted, {tentative} tentative/low-confidence)."
        )
        return 0

    if args.dry_run:
        for a in new:
            print(f"  would alert: {a.text}")
        print(f"{len(new)} new significant regression(s) (dry run — not sent, state unchanged).")
        return 0

    delivered = alerts.send(new, webhook=args.alert_webhook)
    if args.state_file and delivered:
        # dedup only what actually went out, so a failed webhook retries next run
        alerts.save_seen(args.state_file, seen | {a.key for a in delivered})
    print(f"{len(delivered)} new significant regression(s) alerted.")
    return 0


def cmd_reflect(args: argparse.Namespace) -> int:
    with WeaveClient(entity=args.entity, project=args.project) as client:
        feedback = client.query_project_feedback(limit=args.limit)

    if not feedback:
        print("No scored feedback found. Run 'score' or 'judge' first.")
        return 0

    coaching = coaching_digest(feedback)
    print("Current state:")
    print(coaching)
    print()

    project_root = args.project_root or os.getcwd()
    originals = extract_artifacts(project_root)
    if not originals:
        print(f"No CLAUDE.md or agent artifacts found in {project_root}")
        return 2

    print(f"Found {len(originals)} artifact(s): {', '.join(a.name for a in originals)}")

    # --dry-run is a cheap preview: show the coaching state and artifacts without
    # invoking the (expensive, credit-consuming) GEPA reflection.
    if args.dry_run:
        print("\n(dry run — skipping GEPA reflection; re-run without --dry-run to propose edits)")
        return 0

    print(f"Running GEPA reflector (model={args.model}, iterations={args.iterations})...")
    print()

    try:
        with _make_judge_client(args) as judge:
            proposal = run_reflection(
                project_root=project_root,
                feedback=feedback,
                coaching_text=coaching,
                judge_client=judge,
                judge_model=judge_default_model(judge),
                model=args.model,
                max_iterations=args.iterations,
            )
    except ModuleNotFoundError as e:
        print(
            f"Error: reflector dependencies missing ({e.name}). "
            f"Install with: pip install 'weave-agent-signals[rsi]'"
        )
        return 2

    if proposal is None:
        print("No improvements proposed.")
        return 0

    diff = render_proposal_diff(originals, proposal)
    print(diff)

    if not args.apply:
        print("\nReview the diff above. Re-run with --apply to write changes.")
        return 0

    for artifact in proposal.artifacts:
        path = os.path.join(project_root, artifact.path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(artifact.content)
        print(f"  Wrote {artifact.path}")
    print("Changes applied. Run 'score' to measure the impact.")

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

    # judge
    p_judge = subs.add_parser("judge", help="Run LLM judges on recent turns")
    p_judge.add_argument("--since", type=_parse_datetime, default=None)
    p_judge.add_argument("--limit", type=int, default=10)
    p_judge.add_argument(
        "--rubric",
        type=str,
        default=None,
        help="Comma-separated rubric names (default: all)",
    )
    p_judge.add_argument(
        "--judge-backend",
        type=str,
        default=None,
        help="Judge backend: openai, wandb, custom URL, or cli "
        "(local claude/codex CLIs, no credits — temporary) "
        "(default: JUDGE_BACKEND env or openai)",
    )
    p_judge.add_argument(
        "--panel-size",
        type=int,
        default=1,
        help="Number of judges for session PoLL panel (default: 1)",
    )
    p_judge.add_argument("--dry-run", action="store_true")
    p_judge.add_argument("--force", action="store_true")

    # inspect
    p_insp = subs.add_parser("inspect", help="Inspect turns or sessions")
    p_insp.add_argument("--recent", type=int, default=None)
    p_insp.add_argument("--session", type=str, default=None)
    p_insp.add_argument("--feedback", action="store_true", help="Show feedback/scores")

    # analyze
    p_analyze = subs.add_parser("analyze", help="Analyze scored feedback patterns")
    p_analyze.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max feedback records to query (default: 1000)",
    )
    p_analyze.add_argument(
        "--ab", action="store_true", help="Show A/B leaderboard by config_version"
    )
    p_analyze.add_argument(
        "--trends", action="store_true", help="Detect score regressions/improvements"
    )
    p_analyze.add_argument("--coaching", action="store_true", help="Generate coaching digest")

    # monitor
    p_mon = subs.add_parser("monitor", help="Alert on significant score regressions")
    p_mon.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max feedback records to query (default: 1000)",
    )
    p_mon.add_argument(
        "--alert-webhook",
        type=str,
        default=None,
        help="POST alerts to this URL (e.g. a Slack incoming webhook)",
    )
    p_mon.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="JSON file to dedup alerts across scheduled runs",
    )
    p_mon.add_argument(
        "--dry-run",
        action="store_true",
        help="Show alerts without sending to the webhook or updating state",
    )

    # reflect
    p_reflect = subs.add_parser(
        "reflect", help="GEPA reflector: propose CLAUDE.md edits from scores"
    )
    p_reflect.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max feedback records to analyze (default: 500)",
    )
    p_reflect.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model for GEPA reflection (default: gpt-4o)",
    )
    p_reflect.add_argument(
        "--judge-backend",
        type=str,
        default=None,
        help="Backend for the artifact-quality judge: openai, wandb, "
        "custom URL, or cli (local, temporary)",
    )
    p_reflect.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Max GEPA optimization iterations (default: 3)",
    )
    p_reflect.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root to find CLAUDE.md (default: cwd)",
    )
    p_reflect.add_argument(
        "--dry-run", action="store_true", help="Show proposed changes without writing"
    )
    p_reflect.add_argument("--apply", action="store_true", help="Write proposed changes to disk")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()  # load .env (OPENAI_API_KEY, JUDGE_BACKEND, …) before parsing

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
        "judge": cmd_judge,
        "inspect": cmd_inspect,
        "analyze": cmd_analyze,
        "monitor": cmd_monitor,
        "reflect": cmd_reflect,
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
