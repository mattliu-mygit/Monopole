from __future__ import annotations

import argparse
import difflib
import ipaddress
import json
import logging
import os
import sys
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from typing import Sequence

import httpx
from dotenv import load_dotenv

from weave_agent_signals import alerts
from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.client import WeaveClient
from weave_agent_signals.judges.cli_backend import CliJudgeClient
from weave_agent_signals.judges.inference import InferenceClient
from weave_agent_signals.judges.plan import build_judging_plan
from weave_agent_signals.judges.review import ReviewPolicy
from weave_agent_signals.judges.runner import judge_session
from weave_agent_signals.models import Score, SessionView
from weave_agent_signals.patterns import (
    ab_leaderboard,
    coaching_digest,
    detect_config_regressions,
    detect_regressions,
)
from weave_agent_signals.run_config import (
    DEFAULT_JUDGING_CONTEXT_POLICY,
    ModelDescriptor,
    PositionedJudge,
)
from weave_agent_signals.runs.bundles import BundleSnapshot, compare_bundles
from weave_agent_signals.runs.promotion import ProjectFileAdapter
from weave_agent_signals.runs.reflection import run_reflection
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


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
        force: bool,
    ) -> None:
        self.total_scored += 1
        feedback_type = f"weave_agent_signals.{score.scorer}"
        existing = client.query_existing_feedback(ref, feedback_type)
        if existing and not force:
            self.skipped += 1
            return
        client.write_score(score, ref)
        self.total_written += 1
        self.all_tags.extend(score.tags)
        if existing:
            client.delete_feedback_ids(existing)


def _make_model_client(args: argparse.Namespace, model: ModelDescriptor):
    """Build a chat client for one catalog-resolved model role."""

    if model.backend == "cli":
        return CliJudgeClient()
    return InferenceClient(
        entity=args.entity,
        project=args.project,
        backend=model.backend,
    )


def _positioned_judge(model: ModelDescriptor, position: int) -> PositionedJudge:
    return PositionedJudge.model_validate({**model.model_dump(mode="python"), "position": position})


def _resolve_judge_policy(args: argparse.Namespace) -> ReviewPolicy:
    """Resolve standalone judge options through the current guided catalog."""

    catalog = build_model_catalog()
    backend_name = args.judge_backend or catalog.recommended_judge_backend
    try:
        backend = catalog.backend(backend_name)
    except KeyError as exc:
        raise RuntimeError(str(exc)) from exc

    depth = args.review_depth or backend.recommended_review_depth
    if depth is None:
        raise RuntimeError(f"Judge backend {backend_name} has no recommended review depth")
    model_ids = tuple(args.judge_models or backend.recommended_judges)
    judges: list[PositionedJudge] = []
    for position, model_id in enumerate(model_ids, start=1):
        try:
            model = backend.model(model_id)
        except KeyError as exc:
            raise RuntimeError(f"Unknown judge model for {backend_name}: {model_id}") from exc
        if "judge" not in model.supported_roles:
            raise RuntimeError(f"Model {model_id} does not support judging")
        judges.append(_positioned_judge(model, position))

    margin = args.second_opinion_margin
    if depth == "selective" and margin is None:
        margin = catalog.review_defaults["second_opinion_margin"]
    try:
        return ReviewPolicy(
            depth=depth,
            judges=tuple(judges),
            second_opinion_margin=margin,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _resolve_reflection_models(
    args: argparse.Namespace,
) -> tuple[ModelDescriptor, ModelDescriptor]:
    """Resolve standalone reflection's writer and evaluator independently."""

    catalog = build_model_catalog()
    writer_id = args.model or catalog.proposal.recommended_model
    writers = {model.id: model for model in catalog.proposal.available_models}
    writer = writers.get(writer_id) if writer_id is not None else None
    if writer is None or "proposal_writer" not in writer.supported_roles:
        raise RuntimeError(f"Unknown or unavailable proposal model: {writer_id}")

    backend_name = args.judge_backend or catalog.recommended_judge_backend
    try:
        backend = catalog.backend(backend_name)
    except KeyError as exc:
        raise RuntimeError(str(exc)) from exc
    evaluator_id = args.proposal_evaluator_model
    if evaluator_id is None:
        evaluator_id = next(iter(backend.proposal_evaluator_preferences), None)
    if evaluator_id is None:
        raise RuntimeError(f"Judge backend {backend_name} has no proposal evaluator")
    try:
        evaluator = backend.model(evaluator_id)
    except KeyError as exc:
        raise RuntimeError(
            f"Unknown proposal evaluator for {backend_name}: {evaluator_id}"
        ) from exc
    if "proposal_evaluator" not in evaluator.supported_roles:
        raise RuntimeError(f"Model {evaluator_id} does not support proposal evaluation")
    return writer, evaluator


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
                s.stamp(
                    config_version=session.config_version,
                    git_branch=session.git_branch,
                    run_time=run_time,
                )
                stats.write_score(
                    client,
                    s,
                    sess_ref,
                    force=args.force,
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
                    s.stamp(
                        config_version=turn.config_version,
                        git_branch=turn.git_branch,
                        run_time=turn.started_at,
                    )
                    stats.write_score(
                        client,
                        s,
                        turn_ref,
                        force=args.force,
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
                    s.stamp(
                        config_version=turn.config_version,
                        git_branch=turn.git_branch,
                        run_time=turn.started_at,
                    )
                    stats.write_score(
                        client,
                        s,
                        turn_ref,
                        force=args.force,
                    )
            except Exception as e:
                stats.errors += 1
                log.warning("Error scoring turn %s: %s", turn.trace_id[:12], e)

        sessions_scored = _score_sessions(client, turns, stats, args)

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
            client.hydrate_turns_batch(turns)
            feedback_by_ref = (
                client.query_all_feedback_batch(
                    [turn.ref_for(args.entity, args.project) for turn in turns]
                )
                if args.feedback
                else {}
            )
            for turn in turns:
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
                    feedback = feedback_by_ref[turn.ref_for(args.entity, args.project)]
                    print("  Feedback:")
                    _print_feedback(feedback)
        elif args.session:
            session = client.query_session(args.session)
            client.hydrate_turns_batch(session.turns)
            session_ref = session.ref_for(args.entity, args.project)
            turn_refs = {
                turn.trace_id: turn.ref_for(args.entity, args.project) for turn in session.turns
            }
            feedback_by_ref = (
                client.query_all_feedback_batch([session_ref, *turn_refs.values()])
                if args.feedback
                else {}
            )
            print(f"Session {session.conversation_id}")
            print(f"  Turns: {len(session.turns)}")
            print(f"  Tokens: {session.total_tokens}")
            if args.feedback:
                print("  Session feedback:")
                _print_feedback(feedback_by_ref[session_ref])
            for i, t in enumerate(session.turns):
                print(
                    f"  [{i}] {t.trace_id[:12]}  steer={t.steering_count} "
                    f"deny={t.denial_count} err={t.tool_error_count} "
                    f"tools={len(t.tool_calls)} subagents={len(t.subagents)}"
                )
                if args.feedback:
                    _print_feedback(feedback_by_ref[turn_refs[t.trace_id]])
        else:
            print("Specify --recent N or --session CONVERSATION_ID")
            return 1

    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    since = args.since or (datetime.now(timezone.utc) - timedelta(hours=24))
    policy = _resolve_judge_policy(args)
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

    with WeaveClient(entity=args.entity, project=args.project) as client:
        turns = client.query_turns(limit=args.limit, since=since)

        if not turns:
            print("No turns found.")
            return 0

        stats = _WriteStats()

        # Session digests are built from the turns' tool/chat children, so the
        # turns must be hydrated whenever we judge sessions — not only when we
        # judge turns. Batch hydration validates completeness before any judge
        # call, so a partial trace cannot produce durable feedback.
        client.hydrate_turns_batch(turns)

        sessions = _group_sessions(turns)
        cohort_id = "cli-direct:" + ",".join(sorted(turn.trace_id for turn in turns))
        judging_plan = build_judging_plan(
            sessions,
            cohort_id=cohort_id,
            rubrics=selected_rubrics,
            review_depth=policy.depth,
            second_opinion_margin=policy.second_opinion_margin,
            judge_models=policy.judges,
            context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
        )
        with _make_model_client(args, policy.judges[0].model) as inference:
            artifacts: dict[str, dict] = {}

            def record_artifact(key: str, value: dict) -> None:
                normalized = dict(value)
                existing = artifacts.setdefault(key, normalized)
                if existing != normalized:
                    raise ValueError(f"judging artifact conflict: {key}")

            pending_scores: list[tuple[Score, str]] = []
            for session in sessions:
                conv_id = session.conversation_id
                run_time = session.turns[0].started_at if session.turns else None
                try:
                    scores = judge_session(
                        session,
                        inference,
                        rubrics=selected_rubrics,
                        judges=policy.judges,
                        review_depth=policy.depth,
                        second_opinion_margin=policy.second_opinion_margin,
                        judging_plan=judging_plan,
                        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
                        artifact_loader=artifacts.get,
                        artifact_recorder=record_artifact,
                    )
                    sess_ref = session.ref_for(args.entity, args.project)
                    for score in scores:
                        score.stamp(
                            config_version=session.config_version,
                            git_branch=session.git_branch,
                            run_time=run_time,
                        )
                        pending_scores.append((score, sess_ref))
                except Exception as error:
                    stats.errors += 1
                    log.warning("Error judging session %s: %s", conv_id[:12], error)
            if not stats.errors:
                for score, sess_ref in pending_scores:
                    stats.write_score(client, score, sess_ref, force=args.force)

    print(
        f"Judged {judging_plan['totals']['sessions_planned']} sessions across "
        f"{judging_plan['totals']['windows_planned']} reviewer windows "
        f"({len(pending_scores)} rubric scores). Wrote {stats.total_written}."
    )
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
                fromfile=f"B/{action.locator}",
                tofile=f"C/{action.locator}",
            )
        )
    return "".join(chunks)


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
    adapter = ProjectFileAdapter(project_root)
    baseline = adapter.capture()
    existing_targets = [target for target in baseline.targets if target.exists]
    if not existing_targets:
        print(f"No managed instruction targets found in {project_root}")
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
                build_candidate=adapter.bundle_from_content_map,
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
        description="Score agent traces from Weave",
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

    # backfill
    p_back = subs.add_parser("backfill", help="Backfill scores for a date range")
    p_back.add_argument("--start", type=_parse_datetime, required=True)
    p_back.add_argument("--end", type=_parse_datetime, default=None)
    p_back.add_argument("--page-size", type=int, default=500)
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
        help="Judge catalog backend ID: cli, wandb, or openai (default: catalog recommendation)",
    )
    p_judge.add_argument(
        "--judge-model",
        dest="judge_models",
        action="append",
        default=None,
        help="Ordered judge model ID; repeat for each reviewer (default: catalog recommendation)",
    )
    p_judge.add_argument(
        "--review-depth",
        choices=("primary", "selective", "full_panel"),
        default=None,
        help="Reviewer escalation depth (default: catalog recommendation)",
    )
    p_judge.add_argument(
        "--second-opinion-margin",
        type=float,
        default=None,
        help="Selective-review threshold margin (default: catalog recommendation)",
    )
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
    # reflect
    p_reflect = subs.add_parser(
        "reflect", help="Propose managed instruction edits from evaluation scores"
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
        default=None,
        help="Proposal-writer model ID (default: catalog recommendation)",
    )
    p_reflect.add_argument(
        "--judge-backend",
        type=str,
        default=None,
        help="Proposal-evaluator catalog backend ID: cli, wandb, or openai "
        "(default: catalog recommendation)",
    )
    p_reflect.add_argument(
        "--proposal-evaluator-model",
        type=str,
        default=None,
        help="Proposal-evaluator model ID (default: selected backend recommendation)",
    )
    p_reflect.add_argument(
        "--candidate-budget",
        type=_positive_int,
        default=3,
        help="Maximum generated reflection candidates (default: 3)",
    )
    p_reflect.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root containing managed instruction artifacts (default: cwd)",
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
        "--project-root",
        type=str,
        default=None,
        help="Project root for artifact lookups (default: cwd)",
    )

    return parser


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    if args.project_root:
        os.environ["PROJECT_ROOT"] = args.project_root
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
