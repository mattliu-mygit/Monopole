"""Deterministic session-only plans for sliding-window model judging."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.review import ReviewPolicy
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.judges.sliding import sliding_protocol_contract_manifest
from weave_agent_signals.judges.windowing import build_window_plan
from weave_agent_signals.models import SessionView
from weave_agent_signals.run_config import (
    JudgingContextPolicy,
    PositionedJudge,
    ReviewDepth,
    RubricDescriptor,
)

PLAN_SCHEMA_VERSION = "2"


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _attempt_bounds(depth: ReviewDepth, judge_count: int) -> tuple[int, int]:
    if depth == "primary":
        return 1, 1
    if depth == "selective":
        return 1, judge_count
    return judge_count, judge_count


def session_evidence_trace_ids(session_plan: Mapping[str, Any]) -> list[str]:
    """Return the plan's complete raw session coverage."""

    values = session_plan.get("raw_coverage_trace_ids")
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("session plan has invalid raw coverage")
    return list(values)


def build_judging_plan(
    sessions: Sequence[SessionView],
    *,
    cohort_id: str,
    rubrics: Sequence[RubricDescriptor],
    review_depth: ReviewDepth,
    second_opinion_margin: float | None,
    judge_models: Sequence[PositionedJudge],
    context_policy: JudgingContextPolicy,
) -> dict[str, Any]:
    """Pin every session, reviewer window manifest, rubric, and protocol input."""

    if not cohort_id:
        raise ValueError("cohort_id is required")
    if not isinstance(context_policy, JudgingContextPolicy):
        raise TypeError("context_policy must be a JudgingContextPolicy")
    judges = tuple(judge_models)
    ReviewPolicy(
        depth=review_depth,
        judges=judges,
        second_opinion_margin=second_opinion_margin,
    )
    requested = tuple(rubrics)
    if not requested:
        raise ValueError("rubrics must not be empty")
    current = build_rubric_catalog()
    seen: set[str] = set()
    for descriptor in requested:
        if not isinstance(descriptor, RubricDescriptor):
            raise TypeError("rubrics must contain RubricDescriptor values")
        if descriptor.id in seen:
            raise ValueError("rubric IDs must be unique")
        seen.add(descriptor.id)
        if descriptor.id not in SESSION_RUBRICS:
            raise ValueError(f"Unknown session rubric: {descriptor.id}")
        if descriptor != current.rubric(descriptor.id):
            raise ValueError(f"Pinned rubric {descriptor.id} does not match current content")

    minimum_attempts, maximum_attempts = _attempt_bounds(review_depth, len(judges))
    session_plans: list[dict[str, Any]] = []
    turns_considered = 0
    windows_planned = 0
    digest_calls_planned = 0
    maximum_window_calls = 0
    maximum_merge_calls = 0
    for session in sorted(sessions, key=lambda value: value.conversation_id):
        if not isinstance(session, SessionView):
            raise TypeError("sessions must contain SessionView values")
        turns_considered += len(session.turns)
        reviewers = []
        for judge in judges:
            window_plan = build_window_plan(session, context_policy, judge.max_input_tokens)
            count = int(window_plan["chunk_count"])
            reviewers.append(
                {
                    "ordinal": judge.position,
                    "judge": judge.model_dump(mode="json"),
                    "window_plan": window_plan,
                    "work_bounds": {
                        "digest_calls": count,
                        "window_calls_per_rubric": count,
                        "merge_calls_per_rubric": 1,
                    },
                }
            )
            windows_planned += count
            digest_calls_planned += count
            maximum_window_calls += count * len(requested)
            maximum_merge_calls += len(requested)
        coverage = [turn.trace_id for turn in session.turns]
        session_plans.append(
            {
                "conversation_id": session.conversation_id,
                "turn_count": len(session.turns),
                "raw_coverage_trace_ids": coverage,
                "rubrics": [
                    {
                        **descriptor.model_dump(mode="json"),
                        "minimum_reviewer_attempts": minimum_attempts,
                        "maximum_reviewer_attempts": maximum_attempts,
                    }
                    for descriptor in requested
                ],
                "reviewers": reviewers,
            }
        )

    planned_rubrics = len(session_plans) * len(requested)
    body: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "cohort_id": cohort_id,
        "review_depth": review_depth,
        "second_opinion_margin": second_opinion_margin,
        "requested_rubrics": [value.model_dump(mode="json") for value in requested],
        "input_policy": context_policy.model_dump(mode="json"),
        "protocol": sliding_protocol_contract_manifest(),
        "sessions": session_plans,
        "totals": {
            "sessions_planned": len(session_plans),
            "turns_considered": turns_considered,
            "windows_planned": windows_planned,
            "planned_rubrics": planned_rubrics,
            "minimum_reviewer_attempts": planned_rubrics * minimum_attempts,
            "maximum_reviewer_attempts": planned_rubrics * maximum_attempts,
            "maximum_digest_calls": digest_calls_planned,
            "maximum_window_calls": maximum_window_calls,
            "maximum_merge_calls": maximum_merge_calls,
        },
    }
    return {"plan_id": _digest(body), **body}
