from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.plan import build_judging_plan
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
        family=f"family-{position}",
        backend="openai",
        supported_roles=("judge",),
        max_input_tokens=limit,
        position=position,
    )


def _plan(
    *,
    sessions=(_session(),),
    judges=(_judge("judge-1", 1),),
    second_opinion_margin=None,
):
    depth = "primary" if len(judges) == 1 else "selective"
    return build_judging_plan(
        sessions,
        cohort_id="cohort-1",
        rubrics=build_rubric_catalog().rubrics,
        review_depth=depth,
        second_opinion_margin=(
            0.1 if depth == "selective" and second_opinion_margin is None else second_opinion_margin
        ),
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )


def test_plan_is_session_only_and_covers_every_turn() -> None:
    plan = _plan(judges=(_judge("judge-1", 1), _judge("judge-2", 2)))
    assert plan["schema_version"] == "2"
    assert plan["totals"]["planned_rubrics"] == 6
    assert plan["totals"]["sessions_planned"] == 1
    session = plan["sessions"][0]
    assert session["raw_coverage_trace_ids"] == [f"turn-{i}" for i in range(1, 5)]
    assert [row["id"] for row in session["rubrics"]] == list(SESSION_RUBRICS)
    assert [reviewer["judge"]["position"] for reviewer in session["reviewers"]] == [1, 2]
    assert all(reviewer["window_plan"]["windows"] for reviewer in session["reviewers"])
    assert plan["input_policy"] == DEFAULT_JUDGING_CONTEXT_POLICY.model_dump(mode="json")
    assert plan["second_opinion_margin"] == 0.1


def test_plan_id_authenticates_reviewer_ordinal_and_model_limit() -> None:
    base = _plan(judges=(_judge("judge-1", 1), _judge("judge-2", 2)))
    reordered = _plan(judges=(_judge("judge-2", 1), _judge("judge-1", 2)))
    limited = _plan(judges=(_judge("judge-1", 1, 120_000), _judge("judge-2", 2)))
    assert len({base["plan_id"], reordered["plan_id"], limited["plan_id"]}) == 3
    changed_margin = _plan(
        judges=(_judge("judge-1", 1), _judge("judge-2", 2)),
        second_opinion_margin=0.2,
    )
    assert changed_margin["plan_id"] != base["plan_id"]


def test_plan_rejects_non_session_or_stale_rubrics() -> None:
    descriptor = build_rubric_catalog().rubrics[0]
    stale = descriptor.model_copy(update={"version": "stale"})
    with pytest.raises(ValueError, match="does not match"):
        build_judging_plan(
            [_session()],
            cohort_id="cohort-1",
            rubrics=(stale,),
            review_depth="primary",
            second_opinion_margin=None,
            judge_models=(_judge("judge-1", 1),),
            context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
        )
