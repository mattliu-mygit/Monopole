from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges import plan as plan_module
from weave_agent_signals.judges import runner
from weave_agent_signals.judges.plan import build_judging_plan
from weave_agent_signals.judges.review import AttemptObservation
from weave_agent_signals.judges.runner import JudgeExecutionError, judge_session
from weave_agent_signals.models import SessionView, TurnSpan
from weave_agent_signals.run_config import DEFAULT_JUDGING_CONTEXT_POLICY, PositionedJudge


def _session() -> SessionView:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    turn = TurnSpan(
        "turn-1",
        "session-1",
        now,
        now,
        "agent",
        1,
        1,
        0,
        "OK",
        "cfg",
        "main",
        None,
        "sid",
        0,
        0,
        0,
        [],
        [],
        [],
        [],
        "request",
        "response",
    )
    return SessionView("session-1", [turn], "cfg", "main")


def _judge(model_id: str, position: int) -> PositionedJudge:
    return PositionedJudge(
        id=model_id,
        label=model_id,
        family=f"family-{position}",
        backend="openai",
        supported_roles=("judge",),
        max_input_tokens=128_000,
        position=position,
    )


def _observation(score: float | None, *, failed: bool = False) -> AttemptObservation:
    if failed:
        return AttemptObservation(
            status="failed",
            resolved_model=None,
            score=None,
            rationale=None,
            usage={},
            error_type="RuntimeError",
            message="offline",
        )
    return AttemptObservation(
        status="succeeded",
        resolved_model="resolved",
        score=score,
        rationale="merged",
        usage={},
        error_type=None,
        message=None,
        evidence_ids=("turn-1",),
        behavioral_feedback={
            "success": "Kept state consistent.",
            "problem": "One check was late.",
            "desired_behavior": "Check immediately.",
        },
    )


def _abstention() -> AttemptObservation:
    return AttemptObservation(
        status="abstained",
        resolved_model="resolved",
        score=None,
        rationale="The rubric has no applicable evidence.",
        usage={},
        error_type=None,
        message=None,
    )


def _run(monkeypatch, outcomes, *, judges=None, rubrics=None):
    session = _session()
    judges = judges or (_judge("judge-1", 1),)
    rubrics = rubrics or build_rubric_catalog().rubrics
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    created = []

    class FakeReviewer:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.judge = kwargs["judge"]

        def review(self, _rubric):
            return outcomes[self.judge.id].pop(0)

    monkeypatch.setattr(runner, "SlidingReviewer", FakeReviewer)
    scores = judge_session(
        session,
        object(),
        rubrics=rubrics,
        judges=judges,
        judging_plan=plan,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
        artifact_loader=lambda _key: None,
        artifact_recorder=lambda *_args: None,
    )
    return scores, created


def test_runner_reuses_one_lazy_reviewer_per_judge_and_preserves_feedback(monkeypatch) -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:2]
    outcomes = {
        "judge-1": [_observation(0.75), _observation(0.75)],
        "judge-2": [_observation(1.0), _observation(1.0)],
    }
    scores, created = _run(monkeypatch, outcomes, judges=judges, rubrics=rubrics)
    assert [item["judge"].id for item in created] == ["judge-1", "judge-2"]
    assert len(scores) == 2
    assert all(score.granularity == "session" for score in scores)
    assert scores[0].metadata["behavioral_feedback"][0]["problem"] == "One check was late."
    assert "Problem: One check was late." in scores[0].reason
    assert scores[0].metadata["panel_contract_version"] == "1"
    assert scores[0].metadata["attempt_count"] == 2


def test_runner_invokes_every_selected_judge(monkeypatch) -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:1]
    scores, created = _run(
        monkeypatch,
        {"judge-1": [_observation(0.5)], "judge-2": [_observation(0.75)]},
        judges=judges,
        rubrics=rubrics,
    )
    assert [item["judge"].id for item in created] == ["judge-1", "judge-2"]
    assert scores[0].value == pytest.approx(0.625)


def test_zero_success_raises_with_attempt_audit(monkeypatch) -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:1]
    with pytest.raises(JudgeExecutionError) as caught:
        _run(
            monkeypatch,
            {
                "judge-1": [_observation(None, failed=True)],
                "judge-2": [_observation(None, failed=True)],
            },
            judges=judges,
            rubrics=rubrics,
        )
    assert len(caught.value.failures[0].attempts) == 2


def test_unanimous_abstention_is_audited_without_becoming_a_failure(monkeypatch) -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:1]
    with pytest.raises(JudgeExecutionError) as caught:
        _run(
            monkeypatch,
            {"judge-1": [_abstention()], "judge-2": [_abstention()]},
            judges=judges,
            rubrics=rubrics,
        )

    assert caught.value.failures == ()
    assert caught.value.not_evaluable[0].rubric == rubrics[0].id
    assert len(caught.value.not_evaluable[0].attempts) == 2


def test_runner_rejects_detached_rubric_selection(monkeypatch) -> None:
    session = _session()
    judges = (_judge("judge-1", 1),)
    rubrics = build_rubric_catalog().rubrics
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    with pytest.raises(ValueError, match="rubrics or attempt bounds"):
        judge_session(
            session,
            object(),
            rubrics=rubrics[:1],
            judges=judges,
            judging_plan=plan,
            context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
            artifact_loader=lambda _: None,
            artifact_recorder=lambda *_: None,
        )


def test_runner_rejects_alternate_later_judge_before_reviewer_instantiation(monkeypatch) -> None:
    session = _session()
    pinned = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:1]
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        judge_models=pinned,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    created = []
    monkeypatch.setattr(runner, "SlidingReviewer", lambda **kwargs: created.append(kwargs))
    alternate = (_judge("judge-1", 1), _judge("alternate", 2))
    with pytest.raises(ValueError, match="ordered judges"):
        judge_session(
            session,
            object(),
            rubrics=rubrics,
            judges=alternate,
            judging_plan=plan,
            context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
            artifact_loader=lambda _: None,
            artifact_recorder=lambda *_: None,
        )
    assert created == []


def test_runner_synthesizes_authenticated_skip_without_creating_reviewer(monkeypatch) -> None:
    session = _session()
    judges = (_judge("small", 1), _judge("large", 2))
    judges = (judges[0].model_copy(update={"max_input_tokens": 64_000}), judges[1])
    rubrics = build_rubric_catalog().rubrics[:1]
    original = plan_module.build_window_plan

    def build(value, policy, limit, counter):
        if limit == 64_000:
            from weave_agent_signals.judges.windowing import WindowPlanInapplicable

            raise WindowPlanInapplicable()
        return original(value, policy, limit, counter)

    monkeypatch.setattr(plan_module, "build_window_plan", build)
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    created = []

    class FakeReviewer:
        def __init__(self, **kwargs):
            created.append(kwargs["judge"].id)

        def review(self, _rubric):
            return _observation(0.75)

    monkeypatch.setattr(runner, "SlidingReviewer", FakeReviewer)
    scores = judge_session(
        session,
        object(),
        rubrics=rubrics,
        judges=judges,
        judging_plan=plan,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
        artifact_loader=lambda _: None,
        artifact_recorder=lambda *_: None,
    )

    assert created == ["large"]
    assert scores[0].metadata["review_status"] == "degraded"
    assert [attempt["status"] for attempt in scores[0].metadata["attempts"]] == [
        "skipped",
        "succeeded",
    ]
    assert scores[0].metadata["attempts"][0]["skip_reason"] == ("insufficient_context_capacity")
