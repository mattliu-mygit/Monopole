from __future__ import annotations

import argparse
import difflib
import ipaddress
import logging
import os
import sys
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from typing import Sequence

import httpx
from dotenv import load_dotenv

from weave_agent_signals import alerts, ces_commands
from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.judges.cli_backend import CliJudgeClient
from weave_agent_signals.judges.inference import InferenceClient
from weave_agent_signals.judges.plan import build_canonical_judging_plan, judging_plan_totals
from weave_agent_signals.judges.records import JudgeCallRecord
from weave_agent_signals.judges.runner import JudgeExecutionError, judge_session
from weave_agent_signals.models import (
    Score,
    SessionView,
)
from weave_agent_signals.patterns import (
    coaching_digest,
    config_latest,
    config_regressions,
    detect_regressions,
    is_evaluation_feedback_eligible,
    stat_leaderboard,
)
from weave_agent_signals.run_config import (
    DEFAULT_JUDGING_CONTEXT_POLICY,
    ModelDescriptor,
    PositionedJudge,
)
from weave_agent_signals.runs.bundles import BundleSnapshot, compare_bundles
from weave_agent_signals.runs.promotion import TargetPromoter
from weave_agent_signals.runs.reflection import run_reflection
from weave_agent_signals.runs.targets import load_target_registry

log = logging.getLogger("weave_agent_signals")


def _parse_datetime(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Cannot parse datetime: {s}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _make_model_client(args: argparse.Namespace, model: ModelDescriptor):
    """Build a chat client for one catalog-resolved model role."""

    if model.provider in {"claude", "codex", "agy"}:
        return CliJudgeClient(provider=model.provider)
    return InferenceClient(
        entity=args.entity,
        project=args.project,
        backend=model.provider,
    )


def _positioned_judge(model: ModelDescriptor, position: int) -> PositionedJudge:
    return PositionedJudge.model_validate({**model.model_dump(mode="python"), "position": position})


def _resolve_judge_panel(args: argparse.Namespace) -> tuple[PositionedJudge, ...]:
    """Resolve the standalone ordered judge panel through the current catalog."""

    catalog = build_model_catalog()
    model_ids = tuple(args.judge_models or catalog.recommended_judges)
    if not 1 <= len(model_ids) <= 3:
        raise RuntimeError("judge_models must contain one through three models")
    judges: list[PositionedJudge] = []
    for position, model_id in enumerate(model_ids, start=1):
        try:
            model = catalog.model(model_id)
        except KeyError as exc:
            raise RuntimeError(f"Unknown judge model: {model_id}") from exc
        if "judge" not in model.supported_roles:
            raise RuntimeError(f"Model {model_id} does not support judging")
        judges.append(_positioned_judge(model, position))

    if len({judge.id for judge in judges}) != len(judges):
        raise RuntimeError("judge model IDs must be unique")
    return tuple(judges)


def _resolve_reflection_models(
    args: argparse.Namespace,
) -> tuple[ModelDescriptor, ModelDescriptor]:
    """Resolve standalone reflection's writer and evaluator independently."""

    catalog = build_model_catalog()
    writer_id = args.model or catalog.recommended_proposal_model
    try:
        writer = catalog.model(writer_id) if writer_id is not None else None
    except KeyError:
        writer = None
    if writer is None or "proposal_writer" not in writer.supported_roles:
        raise RuntimeError(f"Unknown or unavailable proposal model: {writer_id}")

    evaluator_id = args.proposal_evaluator_model
    if evaluator_id is None:
        evaluator_id = next(iter(catalog.proposal_evaluator_preferences), None)
    if evaluator_id is None:
        raise RuntimeError("Model catalog has no proposal evaluator")
    try:
        evaluator = catalog.model(evaluator_id)
    except KeyError as exc:
        raise RuntimeError(f"Unknown proposal evaluator: {evaluator_id}") from exc
    if "proposal_evaluator" not in evaluator.supported_roles:
        raise RuntimeError(f"Model {evaluator_id} does not support proposal evaluation")
    return writer, evaluator


def cmd_judge(args: argparse.Namespace) -> int:
    since = args.since or (datetime.now(timezone.utc) - timedelta(hours=24))
    judges = _resolve_judge_panel(args)
    rubric_catalog = build_rubric_catalog()
    descriptors = {rubric.id: rubric for rubric in rubric_catalog.rubrics}
    if args.rubric:
        requested_ids = tuple(dict.fromkeys(part.strip() for part in args.rubric.split(",")))
        unknown = [name for name in requested_ids if name not in descriptors]
        if not requested_ids or any(not name for name in requested_ids) or unknown:
            rendered = ", ".join(unknown) if unknown else args.rubric
            print(f"Unknown rubric(s): {rendered}. Available: {', '.join(descriptors)}")
            return 2
        selected_rubrics = tuple(descriptors[name] for name in requested_ids)
    else:
        selected_rubrics = rubric_catalog.rubrics

    with ces_commands.connect(args) as ces:
        client, catalog = ces.client, ces.catalog
        sessions = ces_commands.sessions_to_judge(client, catalog, since=since, limit=args.limit)
        if not sessions:
            print("No eligible agent sessions found.")
            return 0
        judging_plan, pending, errors = _judge_sessions(args, sessions, judges, selected_rubrics)
        # Finished reviews are written even when another session failed, so a rerun doesn't pay
        # to judge them again; a failed session writes nothing and the failure is reported.
        written = ces_commands.write_judge_scores(client, catalog, pending, force=args.force)
    return _report_judging(judging_plan, pending, written, errors)


def _judge_sessions(
    args: argparse.Namespace,
    sessions: list[SessionView],
    judges: tuple[PositionedJudge, ...],
    selected_rubrics,
) -> tuple:
    """Judge each session with the panel. Returns (plan, [(score, session)], errors).

    A session whose reviewers failed counts as an error and adds no scores. A session
    with only not-evaluable rubrics keeps the scores it has."""
    trace_ids = [turn.trace_id for session in sessions for turn in session.turns]
    judging_plan = build_canonical_judging_plan(
        sessions,
        cohort_id="cli-direct:" + ",".join(sorted(trace_ids)),
        rubrics=selected_rubrics,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    pending: list[tuple[Score, SessionView]] = []
    errors = 0
    with ExitStack() as stack:
        inference = {
            judge.id: stack.enter_context(_make_model_client(args, judge.model)) for judge in judges
        }
        calls: dict[str, JudgeCallRecord] = {}

        def record_call(call: JudgeCallRecord) -> None:
            existing = calls.setdefault(call.request_id, call)
            if existing != call:
                raise ValueError(f"judge call conflict: {call.request_id}")

        for session in sessions:
            conv_id = session.conversation_id
            run_time = session.turns[0].started_at if session.turns else None
            try:
                scores = judge_session(
                    session,
                    inference,
                    rubrics=selected_rubrics,
                    judges=judges,
                    judging_plan=judging_plan,
                    context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
                    call_loader=calls.get,
                    call_recorder=record_call,
                )
            except JudgeExecutionError as error:
                if error.failures:
                    errors += 1
                    log.warning("Error judging session %s: %s", conv_id[:12], error)
                    continue
                scores = list(error.scores)
                log.info(
                    "Session %s had %d not-evaluable rubric(s)",
                    conv_id[:12],
                    len(error.not_evaluable),
                )
            except Exception as error:
                errors += 1
                log.warning("Error judging session %s: %s", conv_id[:12], error)
                continue
            for score in scores:
                score.stamp(
                    config_version=session.config_version,
                    git_branch=session.git_branch,
                    run_time=run_time,
                )
                pending.append((score, session))
    return judging_plan, pending, errors


def _report_judging(judging_plan, pending: list, written: int, errors: int) -> int:
    plan_totals = judging_plan_totals(judging_plan)
    print(
        f"Judged {plan_totals['sessions_planned']} sessions across "
        f"{plan_totals['windows_planned']} reviewer windows "
        f"({len(pending)} rubric scores). Wrote {written}."
    )
    if errors:
        print(f"  Errors: {errors} (use -v for details)")
    return 1 if errors else 0


def cmd_analyze(args: argparse.Namespace) -> int:
    insights = args.subagents
    digest = args.coaching or not (args.ab or args.trends or insights)
    if args.ab or args.trends or digest:
        _print_score_analysis(args, digest)
    if insights:
        found = ces_commands.read_insights(args)
        if found.subagents is not None:
            _print_subagents(found.subagents)
    return 0


def _print_score_analysis(args: argparse.Namespace, digest: bool) -> None:
    # Summaries and the A/B leaderboard come from CES queries over every Score; trends and
    # coaching examples from the Scores on the newest `--limit` turns.
    stats, feedback = ces_commands.read_analysis(args, feedback=args.trends or digest)

    if not stats.by_config and not feedback:
        print("No scored feedback found.")
        return

    if args.ab:
        results = stat_leaderboard(stats)
        if not results:
            print("No A/B data (no Scores found).")
        else:
            print(f"\nA/B Leaderboard ({len(results)} config versions):\n")
            for r in results:
                print(
                    f"  Config: {r.config_version}  ({r.evaluated_target_count} evaluated targets)"
                )
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

    if digest:
        print(coaching_digest(feedback, stats))


def _print_subagents(counts: dict[str, Counter]) -> None:
    if not counts:
        print("No Trajectories found.")
        return
    print("\nSessions and subagents by config (every Trajectory; no Scorer scores a subagent):\n")
    for config, depths in sorted(counts.items()):
        sessions = depths.get(0, 0)
        subagents = sum(n for depth, n in depths.items() if depth > 0)
        per_session = f"{subagents / sessions:.2f} per session" if sessions else "no sessions"
        by_depth = ", ".join(
            f"depth {depth}: {n}" for depth, n in sorted(depths.items()) if depth > 0
        )
        detail = f"; {by_depth}" if by_depth else ""
        print(
            f"  Config: {config}  sessions={sessions}  "
            f"subagents={subagents} ({per_session}{detail})"
        )


def cmd_monitor(args: argparse.Namespace) -> int:
    stats, feedback = ces_commands.read_analysis(args, feedback=True)

    if not stats.by_config and not feedback:
        print("No scored feedback found.")
        return 0

    trend = detect_regressions(feedback)
    config = config_regressions(stats, config_latest(feedback))
    # Only downward, statistically-significant moves are alerts — an improvement
    # is significant too but is not a regression. config_regressions
    # already returns drops only.
    sig_trend = [r for r in trend if r["significant"] and r["direction"] == "regression"]
    sig_config = [r for r in config if r["significant"]]

    found = [alerts.trend_alert(r) for r in sig_trend] + [
        alerts.config_alert(r) for r in sig_config
    ]
    previously_active = alerts.load_active(args.state_file)
    current_keys = {alert.key for alert in found}
    new = [alert for alert in found if alert.key not in previously_active]

    tentative = len([r for r in trend if not r["significant"]]) + len(
        [r for r in config if not r["significant"]]
    )
    if not new:
        if args.state_file:
            alerts.save_active(args.state_file, previously_active & current_keys)
        print(
            f"No new significant regressions "
            f"({len(found)} already alerted, {tentative} tentative/low-confidence)."
        )
        return 0

    delivered = alerts.send(new, webhook=args.alert_webhook)
    if args.state_file:
        # Retain ongoing delivered alerts, clear recovered alerts, and do not
        # mark failed deliveries so they retry on the next run.
        active = (previously_active & current_keys) | {alert.key for alert in delivered}
        alerts.save_active(args.state_file, active)
    print(f"{len(delivered)} new significant regression(s) alerted.")
    return 0


def _render_bundle_diff(past: BundleSnapshot, proposed: BundleSnapshot) -> str:
    """Render an auditable unified diff for every changed bundle target."""

    chunks: list[str] = []
    for action in compare_bundles(past, proposed).actions:
        before = (
            (action.before.content or "").splitlines(keepends=True) if action.before.exists else []
        )
        after = (
            (action.after.content or "").splitlines(keepends=True) if action.after.exists else []
        )
        chunks.extend(
            line if line.endswith("\n") else f"{line}\n"
            for line in difflib.unified_diff(
                before,
                after,
                fromfile=f"A/{action.locator}",
                tofile=f"B/{action.locator}",
            )
        )
    return "".join(chunks)


def cmd_reflect(args: argparse.Namespace) -> int:
    feedback = ces_commands.read_scores(args)

    feedback = [item for item in feedback if is_evaluation_feedback_eligible(item)]
    if not feedback:
        print("No eligible scored feedback found. Run 'score' or 'judge' first.")
        return 0

    coaching = coaching_digest(feedback)
    print("Current state:")
    print(coaching)
    print()

    adapter = TargetPromoter(load_target_registry(args.target_registry))
    baseline = adapter.capture()
    existing_targets = [target for target in baseline.targets if target.exists]
    if not existing_targets:
        print("No managed instruction targets found in the configured registry")
        return 2

    print(
        f"Found {len(existing_targets)} managed target(s): "
        f"{', '.join(target.locator for target in existing_targets)}"
    )
    writer, evaluator = _resolve_reflection_models(args)

    print(
        "Running reflection "
        f"(proposal_writer={writer.id}, proposal_evaluator={evaluator.id}, "
        f"candidate_budget={args.candidate_budget})..."
    )
    print()

    try:
        with ExitStack() as stack:
            writer_client = stack.enter_context(_make_model_client(args, writer))
            evaluator_client = stack.enter_context(_make_model_client(args, evaluator))
            proposal = run_reflection(
                baseline=baseline,
                feedback=feedback,
                coaching_text=coaching,
                scope_policy=adapter.contract_manifest(),
                requested_writer=writer,
                requested_evaluator=evaluator,
                writer_client=writer_client,
                evaluator_client=evaluator_client,
                resolve_locator=adapter.resolve_locator,
                candidate_budget=args.candidate_budget,
            )
    except ModuleNotFoundError as e:
        print(
            f"Error: reflection dependencies missing ({e.name}). "
            f"Install with: pip install 'weave-agent-signals[reflection]'"
        )
        return 2

    generated = proposal.candidates
    recommended_id = proposal.recommended_candidate_id
    if recommended_id is None:
        if generated:
            noun = "candidate" if len(generated) == 1 else "candidates"
            print(f"Generated {len(generated)} {noun}.")
        print("Kept the current baseline because none of the generated candidates scored higher.")
    else:
        selected_generated_index = next(
            (
                index
                for index, candidate in enumerate(generated)
                if candidate.candidate_id == recommended_id
            ),
            None,
        )
        if selected_generated_index is None:
            raise RuntimeError("Reflection recommendation is missing from its candidates")
        best = generated[selected_generated_index]
        if len(generated) > 1:
            print(
                f"Generated {len(generated)} candidates, best is #{selected_generated_index + 1}:"
            )
            for i, candidate in enumerate(generated):
                marker = " *" if i == selected_generated_index else ""
                print(
                    f"  #{i + 1} model={candidate.resolved_writer_model} "
                    f"score_delta={candidate.score_delta:+.3f}{marker}"
                )
            print()

        print(_render_bundle_diff(proposal.baseline, best.bundle))

    print(
        "\nStandalone reflect is preview-only; this proposal is not saved. "
        "Start a Run to generate its own proposal for review and promotion."
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weave-agent-signals",
        description="Score coding-agent sessions in CES",
    )
    parser.add_argument("--entity", default="weave-team")
    parser.add_argument("--project", default="agent-sessions")
    parser.add_argument("-v", "--verbose", action="store_true")

    subs = parser.add_subparsers(dest="command")

    # score
    p_score = subs.add_parser("score", help="Score recent unscored turns")
    p_score.add_argument("--since", type=_parse_datetime, default=None)
    p_score.add_argument("--limit", type=int, default=100)
    p_score.add_argument("--force", action="store_true")
    ces_commands.add_arguments(p_score)

    # backfill
    p_back = subs.add_parser("backfill", help="Backfill scores for a date range")
    p_back.add_argument("--start", type=_parse_datetime, required=True)
    p_back.add_argument("--end", type=_parse_datetime, default=None)
    p_back.add_argument("--force", action="store_true")
    ces_commands.add_arguments(p_back)

    # judge
    p_judge = subs.add_parser("judge", help="Run sliding-window LLM judges on sessions")
    p_judge.add_argument("--since", type=_parse_datetime, default=None)
    p_judge.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Most recent sessions to judge (default: 10)",
    )
    p_judge.add_argument(
        "--rubric",
        type=str,
        default=None,
        help="Comma-separated rubric names (default: all)",
    )
    p_judge.add_argument(
        "--judge-model",
        dest="judge_models",
        action="append",
        default=None,
        help="Ordered judge model ID; repeat for each reviewer (default: catalog recommendation)",
    )
    p_judge.add_argument("--force", action="store_true")
    ces_commands.add_arguments(p_judge)

    # inspect
    p_insp = subs.add_parser("inspect", help="Inspect turns or sessions")
    p_insp.add_argument("--recent", type=int, default=None)
    p_insp.add_argument("--session", type=str, default=None)
    p_insp.add_argument(
        "--scores", "--feedback", action="store_true", help="Show each session's and turn's Scores"
    )
    ces_commands.add_arguments(p_insp)

    # analyze
    p_analyze = subs.add_parser(
        "analyze",
        help="Analyze scored feedback patterns",
        description=(
            "Summaries and the A/B leaderboard cover every Score, by config version, from CES "
            "queries. Trends, coaching examples, and summary tag counts cover the Scores on the "
            "newest --limit turns and their sessions."
        ),
    )
    ces_commands.add_arguments(p_analyze)
    p_analyze.add_argument(
        "--limit",
        type=int,
        default=1000,
        help=(
            "Newest turns whose Scores feed trends, coaching examples, summary tag counts, and, "
            "until CES has a stddev aggregation, continuous-Scorer summaries (default: 1000)"
        ),
    )
    p_analyze.add_argument(
        "--ab", action="store_true", help="Show A/B leaderboard by config_version (all Scores)"
    )
    p_analyze.add_argument(
        "--trends",
        action="store_true",
        help="Detect score regressions/improvements over the newest --limit turns",
    )
    p_analyze.add_argument("--coaching", action="store_true", help="Generate coaching digest")
    p_analyze.add_argument(
        "--subagents",
        action="store_true",
        help="Count each config's sessions and subagent Trajectories (every Trajectory)",
    )

    # monitor
    p_mon = subs.add_parser(
        "monitor",
        help="Alert on significant score regressions",
        description=(
            "Config regressions compare the newest two config versions over every Score, from "
            "CES queries; a drop is significant when the 95% interval on the difference lies "
            "below 0. Configs are ordered, and trends computed, from the Scores on the newest "
            "--limit turns."
        ),
    )
    ces_commands.add_arguments(p_mon)
    p_mon.add_argument(
        "--limit",
        type=int,
        default=1000,
        help=(
            "Newest turns whose Scores feed trends, config order, and, until CES has a stddev "
            "aggregation, continuous-Scorer stats (default: 1000)"
        ),
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
    # reflect
    p_reflect = subs.add_parser(
        "reflect", help="Propose managed instruction edits from evaluation scores"
    )
    ces_commands.add_arguments(p_reflect)
    p_reflect.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max feedback records to analyze (default: 500)",
    )
    p_reflect.add_argument(
        "--model",
        type=str,
        default=None,
        help="Proposal-writer model ID (default: catalog recommendation)",
    )
    p_reflect.add_argument(
        "--proposal-evaluator-model",
        type=str,
        default=None,
        help="Proposal-evaluator model ID (default: catalog recommendation)",
    )
    p_reflect.add_argument(
        "--candidate-budget",
        type=_positive_int,
        default=3,
        help="Maximum generated reflection candidates (default: 3)",
    )
    p_reflect.add_argument(
        "--target-registry",
        type=str,
        required=True,
        help="Closed JSON registry of exact files and bounded skill collections",
    )
    # serve
    p_serve = subs.add_parser("serve", help="Start the API server and frontend")
    p_serve.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Bind address (default: 127.0.0.1; non-loopback addresses expose the "
            "unauthenticated API)"
        ),
    )
    p_serve.add_argument("--port", type=int, default=8787)
    p_serve.add_argument(
        "--target-registry",
        type=str,
        required=True,
        help="Closed JSON registry of exact files and bounded skill collections",
    )
    p_serve.add_argument(
        "--sandbox-runtime",
        type=str,
        required=True,
        help="Closed JSON runtime for identical local Smol Machines comparison arms",
    )

    return parser


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    os.environ["TARGET_REGISTRY"] = args.target_registry
    os.environ["SMOLMACHINES_RUNTIME"] = args.sandbox_runtime
    os.environ.setdefault("WANDB_ENTITY", args.entity)
    os.environ.setdefault("WANDB_PROJECT", args.project)

    try:
        is_loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        is_loopback = args.host.casefold() == "localhost"
    if not is_loopback:
        print(
            "WARNING: this non-loopback bind exposes an unauthenticated API and its "
            "configured project access to the reachable network.",
            file=sys.stderr,
        )
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "weave_agent_signals.api:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()  # load optional provider credentials before parsing

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.command:
        parser.print_help()
        return 0

    handlers = {
        "score": ces_commands.score,
        "backfill": ces_commands.backfill,
        "judge": cmd_judge,
        "inspect": ces_commands.inspect,
        "analyze": cmd_analyze,
        "monitor": cmd_monitor,
        "reflect": cmd_reflect,
        "serve": cmd_serve,
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
