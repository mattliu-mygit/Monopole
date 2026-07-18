"""Deterministic session-only plans for sliding-window model judging."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

from pydantic import model_validator

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.judges.tokens import TokenCounterName, count_tokens
from weave_agent_signals.judges.windowing import (
    WindowPlanInapplicable,
    build_window_plan,
    render_raw_turn,
)
from weave_agent_signals.models import SessionView
from weave_agent_signals.run_config import (
    JudgingContextPolicy,
    PositionedJudge,
    RubricDescriptor,
    StrictFrozenModel,
)

PLAN_SCHEMA_VERSION = "3"


class WindowBoundary(StrictFrozenModel):
    index: int
    core_trace_ids: tuple[str, ...]
    raw_trace_ids: tuple[str, ...]
    raw_tokens: int


class WindowPlan(StrictFrozenModel):
    input_cap_tokens: int
    raw_budget_tokens: int
    target_raw_tokens: int
    token_counter: TokenCounterName
    capacity_reserve_tokens: int
    merge_input_tokens: int
    windows: tuple[WindowBoundary, ...]


class ReviewerPlan(StrictFrozenModel):
    position: int
    status: Literal["planned", "skipped"]
    skip_reason: str | None = None
    window_plan: WindowPlan | None = None

    @model_validator(mode="after")
    def _valid_disposition(self) -> Self:
        if self.status == "planned" and (self.skip_reason is not None or self.window_plan is None):
            raise ValueError("planned reviewer requires a window plan and no skip reason")
        if self.status == "skipped" and (
            self.skip_reason != "insufficient_context_capacity" or self.window_plan is not None
        ):
            raise ValueError("skipped reviewer requires the capacity reason and no window plan")
        return self


class PlannedTurn(StrictFrozenModel):
    trace_id: str
    position: int
    estimated_tokens: int
    raw_digest: str


class SessionPlan(StrictFrozenModel):
    conversation_id: str
    turns: tuple[PlannedTurn, ...]
    reviewers: tuple[ReviewerPlan, ...]

    def reviewer(self, position: int) -> ReviewerPlan:
        for reviewer in self.reviewers:
            if reviewer.position == position:
                return reviewer
        raise KeyError(position)


class JudgingPlan(StrictFrozenModel):
    plan_id: str
    protocol_version: str
    cohort_id: str
    context_policy: JudgingContextPolicy
    rubrics: tuple[RubricDescriptor, ...]
    reviewers: tuple[PositionedJudge, ...]
    sessions: tuple[SessionPlan, ...]

    @model_validator(mode="after")
    def _valid_identity(self) -> Self:
        body = {
            "protocol_version": self.protocol_version,
            "cohort_id": self.cohort_id,
            "context_policy": self.context_policy.model_dump(mode="json"),
            "rubrics": [rubric.model_dump(mode="json") for rubric in self.rubrics],
            "reviewers": [reviewer.model_dump(mode="json") for reviewer in self.reviewers],
            "sessions": [session.model_dump(mode="json") for session in self.sessions],
        }
        if self.plan_id != _digest(body):
            raise ValueError("judging plan ID does not match its content")
        return self

    def session(self, conversation_id: str) -> SessionPlan:
        for session in self.sessions:
            if session.conversation_id == conversation_id:
                return session
        raise KeyError(conversation_id)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def build_canonical_judging_plan(
    sessions: Sequence[SessionView],
    *,
    cohort_id: str,
    rubrics: Sequence[RubricDescriptor],
    judge_models: Sequence[PositionedJudge],
    context_policy: JudgingContextPolicy,
) -> JudgingPlan:
    """Build the one authoritative typed judging plan."""

    if not cohort_id:
        raise ValueError("cohort_id is required")
    if not isinstance(context_policy, JudgingContextPolicy):
        raise TypeError("context_policy must be a JudgingContextPolicy")
    judges = tuple(judge_models)
    if not 1 <= len(judges) <= 3:
        raise ValueError("judge_models must contain one through three judges")
    if any(not isinstance(judge, PositionedJudge) for judge in judges):
        raise TypeError("judge_models must contain PositionedJudge values")
    if tuple(judge.position for judge in judges) != tuple(range(1, len(judges) + 1)):
        raise ValueError("judge positions must be contiguous and ordered from 1")
    if len({judge.id for judge in judges}) != len(judges):
        raise ValueError("judge model IDs must be unique")

    requested = tuple(rubrics)
    if not requested:
        raise ValueError("rubrics must not be empty")
    current = build_rubric_catalog()
    if len({descriptor.id for descriptor in requested}) != len(requested):
        raise ValueError("rubric IDs must be unique")
    for descriptor in requested:
        if not isinstance(descriptor, RubricDescriptor):
            raise TypeError("rubrics must contain RubricDescriptor values")
        if descriptor.id not in SESSION_RUBRICS:
            raise ValueError(f"Unknown session rubric: {descriptor.id}")
        if descriptor != current.rubric(descriptor.id):
            raise ValueError(f"Pinned rubric {descriptor.id} does not match current content")

    typed_sessions: list[SessionPlan] = []
    for session in sorted(sessions, key=lambda value: value.conversation_id):
        if not isinstance(session, SessionView):
            raise TypeError("sessions must contain SessionView values")
        turns = tuple(
            PlannedTurn(
                trace_id=turn.trace_id,
                position=position,
                estimated_tokens=count_tokens(rendered, judges[0].token_counter),
                raw_digest=_digest(rendered),
            )
            for position, turn in enumerate(session.turns, start=1)
            for rendered in (render_raw_turn(turn, position),)
        )
        reviewer_plans: list[ReviewerPlan] = []
        for judge in judges:
            try:
                value = build_window_plan(
                    session,
                    context_policy,
                    judge.max_input_tokens,
                    judge.token_counter,
                )
            except WindowPlanInapplicable:
                reviewer_plans.append(
                    ReviewerPlan(
                        position=judge.position,
                        status="skipped",
                        skip_reason="insufficient_context_capacity",
                    )
                )
                continue
            reviewer_plans.append(
                ReviewerPlan(
                    position=judge.position,
                    status="planned",
                    window_plan=WindowPlan(
                        input_cap_tokens=value["input_cap_tokens"],
                        raw_budget_tokens=value["raw_budget_tokens"],
                        target_raw_tokens=value["target_raw_tokens"],
                        token_counter=value["token_counter"],
                        capacity_reserve_tokens=value["capacity_reserve_tokens"],
                        merge_input_tokens=value["merge_input_tokens"],
                        windows=tuple(
                            WindowBoundary(
                                index=window["index"],
                                core_trace_ids=tuple(window["core_trace_ids"]),
                                raw_trace_ids=tuple(window["raw_trace_ids"]),
                                raw_tokens=window["raw_tokens"],
                            )
                            for window in value["windows"]
                        ),
                    ),
                )
            )
        typed_sessions.append(
            SessionPlan(
                conversation_id=session.conversation_id,
                turns=turns,
                reviewers=tuple(reviewer_plans),
            )
        )

    body = {
        "protocol_version": PLAN_SCHEMA_VERSION,
        "cohort_id": cohort_id,
        "context_policy": context_policy.model_dump(mode="json"),
        "rubrics": [rubric.model_dump(mode="json") for rubric in requested],
        "reviewers": [judge.model_dump(mode="json") for judge in judges],
        "sessions": [session.model_dump(mode="json") for session in typed_sessions],
    }
    return JudgingPlan(
        plan_id=_digest(body),
        protocol_version=PLAN_SCHEMA_VERSION,
        cohort_id=cohort_id,
        context_policy=context_policy,
        rubrics=requested,
        reviewers=judges,
        sessions=tuple(typed_sessions),
    )


build_judging_plan = build_canonical_judging_plan


def judging_plan_totals(plan: JudgingPlan) -> dict[str, int]:
    """Derive progress bounds without persisting duplicate counters."""

    windows = sum(
        len(reviewer.window_plan.windows)
        for session in plan.sessions
        for reviewer in session.reviewers
        if reviewer.window_plan is not None
    )
    attempts = sum(
        reviewer.status == "planned" for session in plan.sessions for reviewer in session.reviewers
    ) * len(plan.rubrics)
    return {
        "sessions_planned": len(plan.sessions),
        "turns_considered": sum(len(session.turns) for session in plan.sessions),
        "windows_planned": windows,
        "planned_rubrics": len(plan.sessions) * len(plan.rubrics),
        "minimum_reviewer_attempts": attempts,
        "maximum_reviewer_attempts": attempts,
        "maximum_digest_calls": windows,
        "maximum_window_calls": windows * len(plan.rubrics),
        "maximum_merge_calls": attempts,
    }


def canonical_plan_from_epoch9(value: Mapping[str, Any]) -> JudgingPlan:
    """Convert an authenticated epoch-9 plan without rehydrating old traces."""

    _encode = _encode_judging_plan_for_migration(value)
    sessions_value = _encode["sessions"]
    first_reviewers = sessions_value[0]["reviewers"] if sessions_value else []
    judge_values = []
    for item in first_reviewers:
        judge = dict(item["judge"])
        if "provider" not in judge and "backend" in judge:
            judge["provider"] = judge.pop("backend")
            judge["provider_model"] = judge["id"]
        judge_values.append(judge)
    judges = tuple(PositionedJudge.model_validate(judge) for judge in judge_values)
    rubrics = tuple(RubricDescriptor.model_validate(item) for item in _encode["requested_rubrics"])
    context_policy = JudgingContextPolicy.model_validate(_encode["input_policy"])
    sessions: list[SessionPlan] = []
    for session_value in sessions_value:
        raw_turns = next(
            (
                reviewer["window_plan"]["raw_turns"]
                for reviewer in session_value["reviewers"]
                if reviewer["window_plan"] is not None
            ),
            None,
        )
        if raw_turns is None:
            raw_turns = [
                {
                    "trace_id": trace_id,
                    "position": position,
                    "estimated_tokens": 0,
                    "raw_digest": _digest({"epoch9_trace_id": trace_id}),
                }
                for position, trace_id in enumerate(
                    session_value["raw_coverage_trace_ids"], start=1
                )
            ]
        reviewers: list[ReviewerPlan] = []
        for reviewer in session_value["reviewers"]:
            window_value = reviewer["window_plan"]
            window_plan = None
            if window_value is not None:
                window_plan = WindowPlan(
                    input_cap_tokens=window_value["input_cap_tokens"],
                    raw_budget_tokens=window_value["raw_budget_tokens"],
                    target_raw_tokens=window_value["target_raw_tokens"],
                    token_counter=window_value["token_counter"],
                    capacity_reserve_tokens=window_value["capacity_reserve_tokens"],
                    merge_input_tokens=window_value["merge_input_tokens"],
                    windows=tuple(
                        WindowBoundary(
                            index=window["index"],
                            core_trace_ids=tuple(window["core_trace_ids"]),
                            raw_trace_ids=tuple(window["raw_trace_ids"]),
                            raw_tokens=window["raw_tokens"],
                        )
                        for window in window_value["windows"]
                    ),
                )
            reviewers.append(
                ReviewerPlan(
                    position=reviewer["ordinal"],
                    status=reviewer["status"],
                    skip_reason=reviewer["skip_reason"],
                    window_plan=window_plan,
                )
            )
        sessions.append(
            SessionPlan(
                conversation_id=session_value["conversation_id"],
                turns=tuple(PlannedTurn.model_validate(turn) for turn in raw_turns),
                reviewers=tuple(reviewers),
            )
        )
    protocol_version = str(_encode["protocol"]["protocol_version"])
    body = {
        "protocol_version": protocol_version,
        "cohort_id": _encode["cohort_id"],
        "context_policy": context_policy.model_dump(mode="json"),
        "rubrics": [rubric.model_dump(mode="json") for rubric in rubrics],
        "reviewers": [judge.model_dump(mode="json") for judge in judges],
        "sessions": [session.model_dump(mode="json") for session in sessions],
    }
    return JudgingPlan(
        plan_id=_digest(body),
        protocol_version=protocol_version,
        cohort_id=_encode["cohort_id"],
        context_policy=context_policy,
        rubrics=rubrics,
        reviewers=judges,
        sessions=tuple(sessions),
    )


def _encode_judging_plan_for_migration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact epoch-9 envelope before its fields are discarded."""

    required = {
        "plan_id",
        "schema_version",
        "cohort_id",
        "panel_size",
        "requested_rubrics",
        "input_policy",
        "protocol",
        "sessions",
        "totals",
    }
    if set(value) != required or value.get("schema_version") != "3":
        raise ValueError("invalid epoch-9 judging plan")
    body = {key: item for key, item in value.items() if key != "plan_id"}
    if value.get("plan_id") != _digest(body):
        raise ValueError("epoch-9 judging plan ID does not match its content")
    if not isinstance(value.get("sessions"), list):
        raise ValueError("epoch-9 judging plan sessions must be a list")
    return dict(value)
