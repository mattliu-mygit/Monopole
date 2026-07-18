"""Pure product views derived from canonical run domain records."""

from __future__ import annotations

import hashlib
import json

from weave_agent_signals.judges.plan import JudgingPlan, judging_plan_totals
from weave_agent_signals.judges.sliding import sliding_protocol_contract_manifest
from weave_agent_signals.runs.events import RunEvent
from weave_agent_signals.runs.reflection_records import ReflectionResultRecord


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def judging_plan_view(plan: JudgingPlan | None) -> dict[str, object] | None:
    """Reconstruct the useful detailed plan view without persisted projections."""

    if plan is None:
        return None
    sessions = []
    for session in plan.sessions:
        coverage = [turn.trace_id for turn in session.turns]
        turn_digests = {turn.trace_id: turn.raw_digest for turn in session.turns}
        applicable = sum(reviewer.status == "planned" for reviewer in session.reviewers)
        reviewers = []
        for descriptor, reviewer in zip(plan.reviewers, session.reviewers, strict=True):
            window_plan = None
            window_count = 0
            if reviewer.window_plan is not None:
                value = reviewer.window_plan
                window_count = len(value.windows)
                window_plan_id = _digest(
                    {
                        "plan_id": plan.plan_id,
                        "conversation_id": session.conversation_id,
                        "reviewer_position": reviewer.position,
                    }
                )
                window_plan = {
                    "plan_id": window_plan_id,
                    "contract_version": "3",
                    "conversation_id": session.conversation_id,
                    "input_cap_tokens": value.input_cap_tokens,
                    "raw_budget_tokens": value.raw_budget_tokens,
                    "target_raw_tokens": value.target_raw_tokens,
                    "chunk_count": window_count,
                    "overlap_turns": plan.context_policy.overlap_turns,
                    "token_counter": value.token_counter,
                    "capacity_reserve_tokens": value.capacity_reserve_tokens,
                    "merge_input_tokens": value.merge_input_tokens,
                    "raw_turns": [turn.model_dump(mode="json") for turn in session.turns],
                    "raw_coverage_trace_ids": coverage,
                    "windows": [
                        {
                            "window_id": _digest(
                                {
                                    "window_plan_id": window_plan_id,
                                    **window.model_dump(mode="json"),
                                }
                            ),
                            **window.model_dump(mode="json"),
                            "raw_turn_digests": [
                                turn_digests[trace_id] for trace_id in window.raw_trace_ids
                            ],
                        }
                        for window in value.windows
                    ],
                }
            reviewers.append(
                {
                    "ordinal": reviewer.position,
                    "judge": descriptor.model_dump(mode="json"),
                    "status": reviewer.status,
                    "skip_reason": reviewer.skip_reason,
                    "window_plan": window_plan,
                    "work_bounds": {
                        "digest_calls": window_count,
                        "window_calls_per_rubric": window_count,
                        "merge_calls_per_rubric": 1 if window_count else 0,
                    },
                }
            )
        sessions.append(
            {
                "conversation_id": session.conversation_id,
                "turn_count": len(session.turns),
                "raw_coverage_trace_ids": coverage,
                "rubrics": [
                    {
                        **rubric.model_dump(mode="json"),
                        "minimum_reviewer_attempts": applicable,
                        "maximum_reviewer_attempts": applicable,
                    }
                    for rubric in plan.rubrics
                ],
                "reviewers": reviewers,
            }
        )
    return {
        "plan_id": plan.plan_id,
        "schema_version": plan.protocol_version,
        "cohort_id": plan.cohort_id,
        "panel_size": len(plan.reviewers),
        "requested_rubrics": [rubric.model_dump(mode="json") for rubric in plan.rubrics],
        "input_policy": plan.context_policy.model_dump(mode="json"),
        "protocol": sliding_protocol_contract_manifest(),
        "totals": judging_plan_totals(plan),
        "sessions": sessions,
    }


def reflection_result_view(result: ReflectionResultRecord | None) -> dict[str, object] | None:
    """Join normalized attempts and evaluations into the existing candidate view."""

    if result is None:
        return None
    if not result.evaluations:
        return {"candidates": [], "reason": result.reason or "Reflection produced no result"}
    baseline = next(value for value in result.evaluations if value.target_id == "baseline")

    def evaluation_view(evaluation, target_revision: str) -> dict[str, object]:
        return {
            **evaluation.model_dump(mode="json", exclude={"target_id"}),
            "target_revision": target_revision,
        }

    candidates = []
    for attempt in result.attempts:
        if attempt.status != "succeeded" or attempt.candidate_id is None:
            continue
        candidate = result.candidate(attempt.candidate_id)
        candidate_view = candidate.model_dump(mode="json", exclude={"evaluation"})
        candidate_view["evaluation"] = evaluation_view(
            candidate.evaluation,
            candidate.bundle.revision,
        )
        candidates.append(candidate_view)
    attempts = [
        {
            "attempt_id": attempt.attempt_id,
            "number": attempt.number,
            "status": attempt.status,
            "requested_writer": attempt.requested_writer.model_dump(mode="json"),
            "resolved_model": attempt.resolved_model,
            "resolved_family": attempt.resolved_family,
            "resolved_backend": attempt.resolved_backend,
            "usage": attempt.usage,
            "candidate_revision": attempt.bundle.revision if attempt.bundle is not None else None,
            "changed_paths": list(attempt.changed_paths),
            "response_digest": attempt.response_digest,
            "response_excerpt": attempt.response_excerpt,
            "error_type": attempt.error_type,
            "error": attempt.error,
        }
        for attempt in result.attempts
    ]
    return {
        "baseline": result.baseline.to_dict(),
        "baseline_score": baseline.score,
        "baseline_evaluation": evaluation_view(baseline, result.baseline.revision),
        "candidates": candidates,
        "generation_attempts": attempts,
        "recommended_candidate_id": result.recommended_candidate_id,
        "provisional_candidate_id": result.provisional_candidate_id,
        "baseline_won": result.baseline_won,
        "reason": result.reason,
        "score_basis": result.score_basis,
        "challenge": (
            result.challenge.model_dump(mode="json")
            if hasattr(result.challenge, "model_dump")
            else result.challenge
        ),
    }


def progress_with_events(
    progress: dict[str, object] | None,
    events: list[RunEvent],
) -> dict[str, object] | None:
    if progress is None:
        return None
    return {
        **progress,
        "events": [
            {
                "id": event.sequence,
                "at": event.at.isoformat(),
                "phase": event.phase,
                "message": event.message,
                **event.details,
            }
            for event in events
        ],
    }


__all__ = ["judging_plan_view", "progress_with_events", "reflection_result_view"]
