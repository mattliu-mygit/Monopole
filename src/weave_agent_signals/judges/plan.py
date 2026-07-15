"""Deterministic, auditable planning for selective LLM judging.

Every turn remains available to deterministic scorers. This module chooses a
bounded set of high-information episodes for expensive process judgments and
records why each rubric is, or is not, applicable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from weave_agent_signals.judges.rubrics import RUBRICS, SESSION_RUBRICS
from weave_agent_signals.models import SessionView, ToolSpan, TurnSpan
from weave_agent_signals.run_config import ReviewDepth, RubricDescriptor

PLAN_SCHEMA_VERSION = "1"
DEFAULT_MAX_EPISODES_PER_SESSION = 8

_MODIFICATION_TOOLS = {
    "edit",
    "write",
    "apply_patch",
    "multiedit",
    "notebookedit",
    "patch",
}
_VERIFICATION_RE = re.compile(
    r"(?:^|[\s\"'/])(?:pytest|test|tests|ruff|mypy|pyright|tsc|lint|check|build|"
    r"cargo test|go test|npm test|pnpm test|vitest|jest)(?:$|[\s\"'/:-])",
    re.IGNORECASE,
)
_EPISODE_PRIORITY = {
    "terminal": 100,
    "user_correction": 90,
    "tool_error": 80,
    "recovery_opportunity": 70,
    "verification": 60,
    "file_modification": 50,
}


def _tool_text(tool: ToolSpan) -> str:
    return f"{tool.tool_name}\n{tool.arguments}\n{tool.result}"


def _has_error(turn: TurnSpan) -> bool:
    if turn.tool_error_count > 0 or turn.status_code.upper() in {"ERROR", "FAILED"}:
        return True
    return any(tool.status_code.upper() in {"ERROR", "FAILED"} for tool in turn.tool_calls)


def _has_in_turn_recovery_opportunity(turn: TurnSpan) -> bool:
    if len(turn.tool_calls) < 2:
        return False
    return any(tool.status_code.upper() in {"ERROR", "FAILED"} for tool in turn.tool_calls[:-1])


def _has_modification(turn: TurnSpan) -> bool:
    for tool in turn.tool_calls:
        name = tool.tool_name.lower().replace("-", "_")
        if name in _MODIFICATION_TOOLS or "apply_patch" in _tool_text(tool).lower():
            return True
    return False


def _has_verification(turn: TurnSpan) -> bool:
    return any(_VERIFICATION_RE.search(_tool_text(tool)) for tool in turn.tool_calls)


def _is_substantive(turn: TurnSpan) -> bool:
    return bool(
        turn.tool_calls
        or turn.user_input
        or getattr(turn, "assistant_output", None)
        or turn.input_tokens
        or turn.output_tokens
    )


def _episode_reasons(turns: list[TurnSpan]) -> dict[int, set[str]]:
    reasons: dict[int, set[str]] = {}
    for index, turn in enumerate(turns):
        current: set[str] = set()
        if _has_error(turn):
            current.add("tool_error")
        if turn.steering_count or turn.denial_count:
            current.add("user_correction")
        if _has_modification(turn):
            current.add("file_modification")
        if _has_verification(turn):
            current.add("verification")
        if index > 0 and _has_error(turns[index - 1]):
            current.add("recovery_opportunity")
        if current:
            reasons[index] = current

    terminal = next(
        (index for index in range(len(turns) - 1, -1, -1) if _is_substantive(turns[index])),
        None,
    )
    if terminal is not None:
        reasons.setdefault(terminal, set()).add("terminal")
    return reasons


def _bounded_indices(
    reasons: dict[int, set[str]],
    max_episodes: int,
) -> list[int]:
    ranked = sorted(
        reasons,
        key=lambda index: (
            -max(_EPISODE_PRIORITY[reason] for reason in reasons[index]),
            index,
        ),
    )
    return sorted(ranked[:max_episodes])


def _applicability(
    turns: list[TurnSpan],
    index: int,
    requested_turn_rubrics: Sequence[str],
) -> dict[str, str]:
    turn = turns[index]
    modification_seen = any(_has_modification(candidate) for candidate in turns[: index + 1])
    prior_error = index > 0 and _has_error(turns[index - 1])
    terminal = not any(_is_substantive(candidate) for candidate in turns[index + 1 :])

    applicable = {
        "judge.tool_choice": bool(turn.tool_calls),
        "judge.error_recovery": prior_error or _has_in_turn_recovery_opportunity(turn),
        "judge.verification": modification_seen and (_has_verification(turn) or terminal),
        "judge.state_consistency": index > 0,
    }
    return {
        rubric: "applicable" if applicable.get(rubric, False) else "not_applicable"
        for rubric in requested_turn_rubrics
    }


def _evidence_trace_ids(
    turns: list[TurnSpan],
    index: int,
    applicability: dict[str, str],
) -> list[str]:
    start = index
    needs_context = index > 0 and any(
        applicability.get(rubric) == "applicable"
        for rubric in ("judge.error_recovery", "judge.state_consistency")
    )
    if needs_context:
        start = index - 1
    if applicability.get("judge.verification") == "applicable":
        latest_modification = next(
            (
                candidate_index
                for candidate_index in range(index, -1, -1)
                if _has_modification(turns[candidate_index])
            ),
            index,
        )
        start = min(start, latest_modification)
    return [turn.trace_id for turn in turns[start : index + 1]]


def session_evidence_trace_ids(session_plan: Mapping[str, Any]) -> list[str]:
    """Return each selected episode's complete evidence in stable first-seen order."""

    return list(
        dict.fromkeys(
            trace_id
            for episode in session_plan["selected_episodes"]
            for trace_id in episode["evidence_trace_ids"]
        )
    )


def _reviewer_attempt_bounds(
    review_depth: ReviewDepth,
    judge_count: int,
) -> tuple[int, int]:
    valid_counts = {
        "primary": {1},
        "selective": {2, 3},
        "full_panel": {3},
    }
    if review_depth not in valid_counts:
        raise ValueError("review_depth must be primary, selective, or full_panel")
    if isinstance(judge_count, bool) or judge_count not in valid_counts[review_depth]:
        raise ValueError(f"judge_count is invalid for {review_depth} review")
    if review_depth == "primary":
        return 1, 1
    if review_depth == "selective":
        return 1, judge_count
    return 3, 3


def _rubric_record(
    descriptor: RubricDescriptor,
    *,
    applicability: str,
    minimum_reviewer_attempts: int,
    maximum_reviewer_attempts: int,
) -> dict[str, Any]:
    return {
        **descriptor.model_dump(mode="json"),
        "applicability": applicability,
        "minimum_reviewer_attempts": minimum_reviewer_attempts,
        "maximum_reviewer_attempts": maximum_reviewer_attempts,
    }


def build_judging_plan(
    sessions: Sequence[SessionView],
    *,
    cohort_id: str,
    rubrics: Sequence[RubricDescriptor],
    review_depth: ReviewDepth,
    judge_count: int,
    max_episodes_per_session: int = DEFAULT_MAX_EPISODES_PER_SESSION,
) -> dict[str, Any]:
    """Return a stable plan for the exact pinned cohort.

    The returned object is JSON serializable. Its ID is content-derived, so
    retries over the same hydrated cohort and configuration produce the same
    plan.
    """
    if not cohort_id:
        raise ValueError("cohort_id is required")
    if max_episodes_per_session < 1:
        raise ValueError("max_episodes_per_session must be at least 1")
    minimum_per_rubric, maximum_per_rubric = _reviewer_attempt_bounds(
        review_depth,
        judge_count,
    )
    requested = tuple(rubrics)
    if not requested:
        raise ValueError("rubrics must not be empty")
    if any(not isinstance(descriptor, RubricDescriptor) for descriptor in requested):
        raise TypeError("rubrics must contain RubricDescriptor values")
    requested_ids = tuple(descriptor.id for descriptor in requested)
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("rubric IDs must be unique")
    known_units = {
        **{rubric_id: "episode" for rubric_id in RUBRICS},
        **{rubric_id: "session" for rubric_id in SESSION_RUBRICS},
    }
    for descriptor in requested:
        if descriptor.id not in known_units:
            raise ValueError(f"Unknown rubric: {descriptor.id}")
        if descriptor.evaluation_unit != known_units[descriptor.id]:
            raise ValueError(f"Rubric {descriptor.id} has the wrong evaluation unit")
    requested_turn = tuple(
        descriptor for descriptor in requested if descriptor.evaluation_unit == "episode"
    )
    requested_session = tuple(
        descriptor for descriptor in requested if descriptor.evaluation_unit == "session"
    )

    session_plans: list[dict[str, Any]] = []
    turns_considered = 0
    episodes_selected = 0
    planned_episode_rubrics = 0
    minimum_episode_reviewer_attempts = 0
    maximum_episode_reviewer_attempts = 0
    for session in sorted(sessions, key=lambda item: item.conversation_id):
        turns = sorted(session.turns, key=lambda turn: (turn.started_at, turn.trace_id))
        turns_considered += len(turns)
        reasons = _episode_reasons(turns)
        selected_indices = _bounded_indices(reasons, max_episodes_per_session)
        episodes: list[dict[str, Any]] = []
        for index in selected_indices:
            applicability = _applicability(
                turns,
                index,
                tuple(descriptor.id for descriptor in requested_turn),
            )
            rubric_records: list[dict[str, Any]] = []
            for descriptor in requested_turn:
                state = applicability[descriptor.id]
                is_applicable = state == "applicable"
                minimum_attempts = minimum_per_rubric if is_applicable else 0
                maximum_attempts = maximum_per_rubric if is_applicable else 0
                rubric_records.append(
                    _rubric_record(
                        descriptor,
                        applicability=state,
                        minimum_reviewer_attempts=minimum_attempts,
                        maximum_reviewer_attempts=maximum_attempts,
                    )
                )
                if is_applicable:
                    planned_episode_rubrics += 1
                    minimum_episode_reviewer_attempts += minimum_attempts
                    maximum_episode_reviewer_attempts += maximum_attempts
            episodes.append(
                {
                    "trace_id": turns[index].trace_id,
                    "turn_index": index,
                    "selection_kind": "deterministic_trigger",
                    "selection_reasons": sorted(reasons[index]),
                    "evidence_trace_ids": _evidence_trace_ids(turns, index, applicability),
                    "rubrics": rubric_records,
                }
            )
        episodes_selected += len(episodes)
        session_rubric_records = [
            _rubric_record(
                descriptor,
                applicability="applicable",
                minimum_reviewer_attempts=minimum_per_rubric,
                maximum_reviewer_attempts=maximum_per_rubric,
            )
            for descriptor in requested_session
        ]
        session_plans.append(
            {
                "conversation_id": session.conversation_id,
                "turn_count": len(turns),
                "selected_episodes": episodes,
                "omitted_turn_count": len(turns) - len(episodes),
                "session_rubrics": session_rubric_records,
            }
        )

    planned_session_rubrics = len(session_plans) * len(requested_session)
    minimum_session_reviewer_attempts = planned_session_rubrics * minimum_per_rubric
    maximum_session_reviewer_attempts = planned_session_rubrics * maximum_per_rubric
    planned_rubrics = planned_episode_rubrics + planned_session_rubrics
    minimum_reviewer_attempts = (
        minimum_episode_reviewer_attempts + minimum_session_reviewer_attempts
    )
    maximum_reviewer_attempts = (
        maximum_episode_reviewer_attempts + maximum_session_reviewer_attempts
    )
    body: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "cohort_id": cohort_id,
        "requested_rubrics": [descriptor.model_dump(mode="json") for descriptor in requested],
        "review_depth": review_depth,
        "judge_count": judge_count,
        "max_episodes_per_session": max_episodes_per_session,
        "sessions": session_plans,
        "totals": {
            "turns_considered": turns_considered,
            "episodes_selected": episodes_selected,
            "planned_episode_rubrics": planned_episode_rubrics,
            "planned_session_rubrics": planned_session_rubrics,
            "planned_rubrics": planned_rubrics,
            "minimum_episode_reviewer_attempts": minimum_episode_reviewer_attempts,
            "maximum_episode_reviewer_attempts": maximum_episode_reviewer_attempts,
            "minimum_session_reviewer_attempts": minimum_session_reviewer_attempts,
            "maximum_session_reviewer_attempts": maximum_session_reviewer_attempts,
            "minimum_reviewer_attempts": minimum_reviewer_attempts,
            "maximum_reviewer_attempts": maximum_reviewer_attempts,
        },
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {"plan_id": f"sha256:{hashlib.sha256(encoded).hexdigest()}", **body}
