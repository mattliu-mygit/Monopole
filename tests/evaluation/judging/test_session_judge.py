from __future__ import annotations

import copy
import threading
from datetime import datetime, timezone

import pytest

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges import runner
from weave_agent_signals.judges.plan import build_judging_plan
from weave_agent_signals.judges.review import AttemptObservation
from weave_agent_signals.judges.runner import JudgeExecutionError, judge_session
from weave_agent_signals.judges.sliding import sliding_protocol_contract_manifest
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
        provider="openai",
        provider_model=model_id,
        family=f"family-{position}",
        supported_roles=("judge",),
        max_input_tokens=128_000,
        position=position,
    )


def _observation(
    score: float | None,
    *,
    failed: bool = False,
    evidence_ids: tuple[str, ...] = ("turn-1",),
) -> AttemptObservation:
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
        evidence_ids=evidence_ids,
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


def _run(
    monkeypatch,
    outcomes,
    *,
    judges=None,
    rubrics=None,
    client=None,
    before_review=None,
):
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
            if before_review is not None:
                before_review()
            return outcomes[self.judge.id].pop(0)

    monkeypatch.setattr(runner, "SlidingReviewer", FakeReviewer)
    scores = judge_session(
        session,
        {judge.id: client or object() for judge in judges},
        rubrics=rubrics,
        judges=judges,
        judging_plan=plan,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
        artifact_loader=lambda _key: None,
        artifact_recorder=lambda *_args: None,
    )
    return scores, created


def _rehash(plan: dict) -> dict:
    body = {key: value for key, value in plan.items() if key != "plan_id"}
    plan["plan_id"] = runner._plan_digest(body)
    return plan


def test_runner_reuses_one_lazy_reviewer_per_judge_and_preserves_feedback(monkeypatch) -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:2]
    outcomes = {
        "judge-1": [_observation(0.75), _observation(0.75)],
        "judge-2": [_observation(1.0), _observation(1.0)],
    }
    scores, created = _run(monkeypatch, outcomes, judges=judges, rubrics=rubrics)
    assert sorted(item["judge"].id for item in created) == ["judge-1", "judge-2"]
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
    assert sorted(item["judge"].id for item in created) == ["judge-1", "judge-2"]
    assert scores[0].value == pytest.approx(0.625)


def test_runner_retains_duplicate_evidence_ids(monkeypatch) -> None:
    scores, _ = _run(
        monkeypatch,
        {"judge-1": [_observation(0.75, evidence_ids=("turn-1", "turn-1"))]},
        rubrics=build_rubric_catalog().rubrics[:1],
    )

    assert scores[0].metadata["evidence_trace_ids"] == ["turn-1", "turn-1"]


def test_runner_wires_transport_abort_to_panel_cancellation(monkeypatch) -> None:
    callbacks = []
    original_execute_panel = runner.execute_panel

    class Client:
        aborted = 0

        def abort(self):
            self.aborted += 1

    client = Client()

    def capture(*args, cancel_pending, **kwargs):
        callbacks.append(cancel_pending)
        return original_execute_panel(*args, cancel_pending=cancel_pending, **kwargs)

    monkeypatch.setattr(runner, "execute_panel", capture)
    _run(
        monkeypatch,
        {"judge-1": [_observation(0.75)]},
        rubrics=build_rubric_catalog().rubrics[:1],
        client=client,
    )

    assert len(callbacks) == 1
    callbacks[0]()
    assert client.aborted == 1


def test_zero_success_raises_with_attempt_audit(monkeypatch) -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:1]
    started = threading.Barrier(2, timeout=1)
    outcomes = {
        "judge-1": [_observation(None, failed=True)],
        "judge-2": [_observation(None, failed=True)],
    }
    with pytest.raises(JudgeExecutionError) as caught:
        _run(
            monkeypatch,
            outcomes,
            judges=judges,
            rubrics=rubrics,
            before_review=started.wait,
        )
    assert len(caught.value.failures[0].attempts) == 2
    assert outcomes["judge-2"] == []


def test_failed_panel_reports_actual_reviewer_error_after_another_reviewer_succeeds(
    monkeypatch,
) -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2))
    rubrics = build_rubric_catalog().rubrics[:1]
    started = threading.Barrier(2, timeout=1)

    with pytest.raises(JudgeExecutionError) as caught:
        _run(
            monkeypatch,
            {
                "judge-1": [_observation(0.5)],
                "judge-2": [_observation(None, failed=True)],
            },
            judges=judges,
            rubrics=rubrics,
            before_review=started.wait,
        )

    failure = caught.value.failures[0]
    assert failure.message == (
        "Review failed for judge.verification after 1 reviewer succeeded: "
        "judge-2 (RuntimeError): offline"
    )


def test_failed_rubric_stops_remaining_rubrics(monkeypatch) -> None:
    rubrics = build_rubric_catalog().rubrics[:2]
    outcomes = {"judge-1": [_observation(None, failed=True), _observation(0.75)]}

    with pytest.raises(JudgeExecutionError) as caught:
        _run(monkeypatch, outcomes, rubrics=rubrics)

    assert caught.value.failures[0].rubric == rubrics[0].id
    assert len(outcomes["judge-1"]) == 1


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
            {judge.id: object() for judge in judges},
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
            {judge.id: object() for judge in alternate},
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
    judges = (judges[0].model_copy(update={"max_input_tokens": 16_000}), judges[1])
    rubrics = build_rubric_catalog().rubrics[:1]
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
        {judge.id: object() for judge in judges},
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


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("protocol", "protocol"),
        ("turn_count", "turn count"),
        ("raw_coverage_trace_ids", "raw coverage"),
    ],
)
def test_runner_rejects_rehashed_all_skipped_session_tampering_before_outcomes(
    monkeypatch, tamper, message
) -> None:
    session = _session()
    judges = (_judge("incapable", 1).model_copy(update={"max_input_tokens": 16_000}),)
    rubrics = build_rubric_catalog().rubrics[:1]
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    assert plan["sessions"][0]["reviewers"][0]["status"] == "skipped"
    forged = copy.deepcopy(plan)
    if tamper == "protocol":
        forged["protocol"] = {
            **sliding_protocol_contract_manifest(),
            "protocol_version": "forged",
        }
    elif tamper == "turn_count":
        forged["sessions"][0]["turn_count"] += 1
    else:
        forged["sessions"][0]["raw_coverage_trace_ids"] = ["forged-trace"]
    _rehash(forged)
    monkeypatch.setattr(
        runner,
        "execute_panel",
        lambda *_args, **_kwargs: pytest.fail("tampering must fail before outcome recording"),
    )
    monkeypatch.setattr(
        runner,
        "SlidingReviewer",
        lambda **_kwargs: pytest.fail("tampering must fail before model-client work"),
    )

    with pytest.raises(ValueError, match=message):
        judge_session(
            session,
            {judge.id: object() for judge in judges},
            rubrics=rubrics,
            judges=judges,
            judging_plan=forged,
            context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
            artifact_loader=lambda _: pytest.fail("tampering must fail before artifact loading"),
            artifact_recorder=lambda *_: pytest.fail(
                "tampering must fail before outcome recording"
            ),
        )


def test_runner_rejects_forged_skip_for_capable_reviewer(monkeypatch) -> None:
    session = _session()
    judges = (_judge("capable", 1),)
    rubrics = build_rubric_catalog().rubrics[:1]
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    forged = copy.deepcopy(plan)
    reviewer = forged["sessions"][0]["reviewers"][0]
    reviewer.update(
        status="skipped",
        skip_reason="insufficient_context_capacity",
        window_plan=None,
        work_bounds={
            "digest_calls": 0,
            "window_calls_per_rubric": 0,
            "merge_calls_per_rubric": 0,
        },
    )
    forged["sessions"][0]["rubrics"][0].update(
        minimum_reviewer_attempts=0,
        maximum_reviewer_attempts=0,
    )
    _rehash(forged)
    monkeypatch.setattr(
        runner,
        "SlidingReviewer",
        lambda **_kwargs: pytest.fail("forged disposition must fail before reviewer creation"),
    )

    with pytest.raises(ValueError, match="reviewer disposition"):
        judge_session(
            session,
            {judge.id: object() for judge in judges},
            rubrics=rubrics,
            judges=judges,
            judging_plan=forged,
            context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
            artifact_loader=lambda _: None,
            artifact_recorder=lambda *_: None,
        )


def test_runner_rejects_forged_plan_for_incapable_reviewer(monkeypatch) -> None:
    session = _session()
    judges = (_judge("incapable", 1).model_copy(update={"max_input_tokens": 64_000}),)
    rubrics = build_rubric_catalog().rubrics[:1]
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    forged = copy.deepcopy(plan)
    reviewer = forged["sessions"][0]["reviewers"][0]
    reviewer.update(
        status="planned",
        skip_reason=None,
        window_plan={},
        work_bounds={
            "digest_calls": 1,
            "window_calls_per_rubric": 1,
            "merge_calls_per_rubric": 1,
        },
    )
    forged["sessions"][0]["rubrics"][0].update(
        minimum_reviewer_attempts=1,
        maximum_reviewer_attempts=1,
    )
    _rehash(forged)
    monkeypatch.setattr(
        runner,
        "SlidingReviewer",
        lambda **_kwargs: pytest.fail("forged disposition must fail before reviewer creation"),
    )

    with pytest.raises(ValueError, match="reviewer disposition"):
        judge_session(
            session,
            {judge.id: object() for judge in judges},
            rubrics=rubrics,
            judges=judges,
            judging_plan=forged,
            context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
            artifact_loader=lambda _: None,
            artifact_recorder=lambda *_: None,
        )


def test_runner_rejects_tampered_planned_work_bounds() -> None:
    session = _session()
    judges = (_judge("capable", 1),)
    rubrics = build_rubric_catalog().rubrics[:1]
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=rubrics,
        judge_models=judges,
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    forged = copy.deepcopy(plan)
    forged["sessions"][0]["reviewers"][0]["work_bounds"]["merge_calls_per_rubric"] = 0
    _rehash(forged)

    with pytest.raises(ValueError, match="reviewer disposition"):
        judge_session(
            session,
            {judge.id: object() for judge in judges},
            rubrics=rubrics,
            judges=judges,
            judging_plan=forged,
            context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
            artifact_loader=lambda _: None,
            artifact_recorder=lambda *_: None,
        )
