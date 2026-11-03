"""Deterministic session-only plans for sliding-window model judging."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal, Self

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
        if self.plan_id == _digest(body):
            return self
        legacy_body = dict(body)
        changed = False
        policy_field = "large_model_output_reserve_tokens"
        if policy_field not in self.context_policy.model_fields_set:
            legacy_policy = dict(body["context_policy"])
            legacy_policy.pop(policy_field)
            legacy_body["context_policy"] = legacy_policy
            changed = True
        legacy_reviewers = [dict(reviewer) for reviewer in body["reviewers"]]
        for reviewer, legacy_reviewer in zip(self.reviewers, legacy_reviewers, strict=True):
            if "raw_window_target_tokens" not in reviewer.model_fields_set:
                legacy_reviewer.pop("raw_window_target_tokens")
                changed = True
        if changed:
            legacy_body["reviewers"] = legacy_reviewers
            if self.plan_id == _digest(legacy_body):
                return self
        raise ValueError("judging plan ID does not match its content")

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
                    judge.raw_window_target_tokens,
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
