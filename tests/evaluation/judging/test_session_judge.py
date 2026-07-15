from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weave_agent_signals.catalogs import build_rubric_catalog
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


def _run(monkeypatch, outcomes, *, judges=None, rubrics=None, depth="primary", margin=None):
    session = _session()
    judges = judges or (_judge("judge-1", 1),)
    rubrics = rubrics or build_rubric_catalog().rubrics
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        review_depth=depth,
        second_opinion_margin=margin,
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
        review_depth=depth,
        second_opinion_margin=margin,
        judging_plan=plan,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
        artifact_loader=lambda _key: None,
        artifact_recorder=lambda *_args: None,
    )
    return scores, created


def test_runner_reuses_one_lazy_reviewer_per_judge_and_preserves_feedback(monkeypatch) -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:2]
    outcomes = {"judge-1": [_observation(0.75), _observation(0.75)], "judge-2": []}
    scores, created = _run(
        monkeypatch, outcomes, judges=judges, rubrics=rubrics, depth="selective", margin=0.1
    )
    assert len(created) == 1
    assert len(scores) == 2
    assert all(score.granularity == "session" for score in scores)
    assert scores[0].metadata["behavioral_feedback"][0]["problem"] == "One check was late."
    assert "Problem: One check was late." in scores[0].reason
    assert scores[0].metadata["review_policy_version"] == "3"


def test_selective_runner_lazily_invokes_second_reviewer(monkeypatch) -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:1]
    scores, created = _run(
        monkeypatch,
        {"judge-1": [_observation(0.5)], "judge-2": [_observation(0.75)]},
        judges=judges,
        rubrics=rubrics,
        depth="selective",
        margin=0.1,
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
            depth="selective",
            margin=0.1,
        )
    assert len(caught.value.failures[0].attempts) == 2


def test_runner_rejects_detached_rubric_selection(monkeypatch) -> None:
    session = _session()
    judges = (_judge("judge-1", 1),)
    rubrics = build_rubric_catalog().rubrics
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        review_depth="primary",
        second_opinion_margin=None,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    with pytest.raises(ValueError, match="rubrics or attempt bounds"):
        judge_session(
            session,
            object(),
            rubrics=rubrics[:1],
            judges=judges,
            review_depth="primary",
            second_opinion_margin=None,
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
        review_depth="selective",
        second_opinion_margin=0.1,
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
            review_depth="selective",
            second_opinion_margin=0.1,
            judging_plan=plan,
            context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
            artifact_loader=lambda _: None,
            artifact_recorder=lambda *_: None,
        )
    assert created == []


def test_runner_rejects_unpinned_second_opinion_margin_before_inference(monkeypatch) -> None:
    session = _session()
    judges = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:1]
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        review_depth="selective",
        second_opinion_margin=0.1,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    created = []
    monkeypatch.setattr(runner, "SlidingReviewer", lambda **kwargs: created.append(kwargs))
    with pytest.raises(ValueError, match="second opinion margin"):
        judge_session(
            session,
            object(),
            rubrics=rubrics,
            judges=judges,
            review_depth="selective",
            second_opinion_margin=0.2,
            judging_plan=plan,
            context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
            artifact_loader=lambda _: None,
            artifact_recorder=lambda *_: None,
        )
    assert created == []
