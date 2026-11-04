from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges import plan as plan_module
from weave_agent_signals.judges.plan import (
    JudgingPlan,
    build_canonical_judging_plan,
    judging_plan_totals,
)
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.models import SessionView, TurnSpan
from weave_agent_signals.run_config import DEFAULT_JUDGING_CONTEXT_POLICY, PositionedJudge


def _turn(trace_id: str, minute: int) -> TurnSpan:
    started = datetime(2026, 7, 15, 12, minute, tzinfo=timezone.utc)
    return TurnSpan(
        trace_id=trace_id,
        conversation_id="session-1",
        started_at=started,
        ended_at=started + timedelta(seconds=1),
        model="agent-model",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        status_code="OK",
        config_version="cfg",
        git_branch="main",
        effort_level=None,
        session_id="session-1",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
        user_input=f"request {trace_id}",
        assistant_output=f"response {trace_id}",
    )


def _session(count: int = 4) -> SessionView:
    return SessionView(
        "session-1", [_turn(f"turn-{i}", i) for i in range(1, count + 1)], "cfg", "main"
    )


def _judge(model_id: str, position: int, limit: int = 128_000) -> PositionedJudge:
    return PositionedJudge(
        id=model_id,
        label=model_id,
        provider="openai",
        provider_model=model_id,
        family=f"family-{position}",
        supported_roles=("judge",),
        max_input_tokens=limit,
        question_budget_tokens=4_000,
        window_budget_tokens=32_000,
        whole_budget_tokens=32_000,
        merge_budget_tokens=32_000,
        position=position,
    )


def test_plan_pins_incapable_reviewer_as_skipped_and_excludes_its_work(
    monkeypatch,
) -> None:
    original = plan_module.build_window_plan

    def build(session, policy, limit, counter, window_budget_tokens):
        if limit == 64_000:
            from weave_agent_signals.judges.windowing import WindowPlanInapplicable

            raise WindowPlanInapplicable()
        return original(session, policy, limit, counter, window_budget_tokens)

    monkeypatch.setattr(plan_module, "build_window_plan", build)
    plan = _canonical_plan(judges=(_judge("small", 1, 64_000), _judge("large", 2)))

    skipped, planned = plan.sessions[0].reviewers
    assert skipped.status == "skipped"
    assert skipped.skip_reason == "insufficient_context_capacity"
    assert skipped.window_plan is None
    assert planned.status == "planned"
    assert planned.skip_reason is None
    assert planned.window_plan is not None
    assert judging_plan_totals(plan)["maximum_reviewer_attempts"] == len(plan.rubrics)


def test_plan_does_not_convert_unexpected_window_planning_errors(monkeypatch) -> None:
    def fail(*_args):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(plan_module, "build_window_plan", fail)
    with pytest.raises(RuntimeError, match="unexpected"):
        _canonical_plan()


def _canonical_plan(
    *,
    sessions=(_session(),),
    judges=(_judge("judge-1", 1),),
) -> JudgingPlan:
    return build_canonical_judging_plan(
        sessions,
        cohort_id="cohort-1",
        rubrics=build_rubric_catalog().rubrics,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )


def test_canonical_plan_stores_shared_descriptors_and_turn_digests_once() -> None:
    plan = _canonical_plan(judges=(_judge("judge-1", 1), _judge("judge-2", 2)))

    assert isinstance(plan, JudgingPlan)
    assert tuple(rubric.id for rubric in plan.rubrics) == tuple(SESSION_RUBRICS)
    assert tuple(judge.position for judge in plan.reviewers) == (1, 2)
    assert tuple(turn.trace_id for turn in plan.sessions[0].turns) == tuple(
        f"turn-{index}" for index in range(1, 5)
    )
    assert all(turn.raw_digest.startswith("sha256:") for turn in plan.sessions[0].turns)
    assert plan.sessions[0].reviewers[0].window_plan is not None


def test_canonical_plan_id_authenticates_evidence_model_and_policy() -> None:
    base = _canonical_plan()
    changed_evidence = _canonical_plan(sessions=(_session(3),))
    changed_model = _canonical_plan(judges=(_judge("judge-2", 1),))
    changed_policy = build_canonical_judging_plan(
        (_session(),),
        cohort_id="cohort-1",
        rubrics=build_rubric_catalog().rubrics,
        judge_models=(_judge("judge-1", 1),),
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY.model_copy(update={"max_chunks": 39}),
    )

    assert (
        len(
            {
                base.plan_id,
                changed_evidence.plan_id,
                changed_model.plan_id,
                changed_policy.plan_id,
            }
        )
        == 4
    )


def test_canonical_plan_rejects_tampered_content() -> None:
    payload = _canonical_plan().model_dump(mode="json")
    payload["cohort_id"] = "tampered"

    with pytest.raises(ValueError, match="plan ID does not match"):
        JudgingPlan.model_validate(payload)


def test_canonical_plan_derives_totals_instead_of_storing_them() -> None:
    plan = _canonical_plan(judges=(_judge("judge-1", 1), _judge("judge-2", 2)))

    totals = judging_plan_totals(plan)

    assert totals["sessions_planned"] == 1
    assert totals["turns_considered"] == 4
    assert totals["planned_rubrics"] == len(SESSION_RUBRICS)
    assert totals["maximum_reviewer_attempts"] == 2 * len(SESSION_RUBRICS)


def test_canonical_plan_locates_exact_session_and_reviewer() -> None:
    plan = _canonical_plan(judges=(_judge("judge-1", 1), _judge("judge-2", 2)))

    session = plan.session("session-1")

    assert session.reviewer(2).position == 2
    with pytest.raises(KeyError, match="missing-session"):
        plan.session("missing-session")
    with pytest.raises(KeyError, match="3"):
        session.reviewer(3)
