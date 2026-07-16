from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.judges import runner
from weave_agent_signals.judges.inference import InferenceCancelled
from weave_agent_signals.judges.plan import build_judging_plan
from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.run_config import RunConfig, resolve_run_config
from weave_agent_signals.runs.stages import StageCancelled
from weave_agent_signals.runs.stages.judging import JudgingDependencies, run_judging_stage
from weave_agent_signals.runs.store import (
    DataSelection,
    RunStatus,
    RunStore,
    judging_artifact_payload_digest,
)


@pytest.fixture
def store(tmp_path):
    value = RunStore(tmp_path / "runs.db")
    yield value
    value.close()


def _turn() -> TurnSpan:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    return TurnSpan(
        trace_id="turn-1",
        conversation_id="session-1",
        started_at=now,
        ended_at=now,
        model="agent",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        status_code="OK",
        config_version="cfg",
        git_branch="main",
        effort_level=None,
        session_id="sid",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
        user_input="request",
        assistant_output="response",
    )


def _setup(
    store: RunStore,
    *,
    force: bool = False,
    rubric_ids: tuple[str, ...] = ("judge.session_outcome",),
):
    turn = _turn()
    session = SessionView("session-1", [turn], "cfg", "main")
    models = build_model_catalog(which=lambda name: f"/bin/{name}")
    rubrics = build_rubric_catalog()
    request = RunConfig(
        model_catalog_version=models.catalog_version,
        rubric_catalog_version=rubrics.catalog_version,
        judge_backend="cli",
        judge_models=("claude-sonnet-5",),
        proposal_model="gpt-5.6-sol",
        proposal_evaluator_model="claude-sonnet-5",
        rubrics=rubric_ids,
        candidate_budget=3,
        force=force,
    )
    effective = resolve_run_config(request, model_catalog=models, rubric_catalog=rubrics)
    cohort = {
        "schema_version": 1,
        "pinned_at": now_iso(),
        "cohort_id": "cohort",
        "turn_count": 1,
        "session_count": 1,
        "turns": [
            {
                "trace_id": "turn-1",
                "weave_ref": turn.ref_for(),
                "conversation_id": "session-1",
                "started_at": turn.started_at.isoformat(),
                "model": "agent",
                "model_family": "unknown",
            }
        ],
        "sessions": [
            {"conversation_id": "session-1", "weave_ref": session.ref_for(), "turn_count": 1}
        ],
    }
    created = store.create()
    selection = DataSelection(session_ids=("session-1",))
    store.save_selection(created.run_id, selection)
    store.save_config(created.run_id, request)
    run = store.start(
        created.run_id,
        expected_selection=selection,
        expected_config=request,
        turn_cohort=cohort,
        effective_config=effective,
    )
    store.record_stage_result(run.run_id, stage=RunStatus.SCORING, result={"scores_written": 0})
    run = store.finalize_stage_success(run.run_id, stage=RunStatus.SCORING, advance=True)
    return run, effective, turn, session


def now_iso() -> str:
    return datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()


class _Weave:
    entity = "weave-team"
    project = "agent-sessions"

    def __init__(self, *, existing=None, fail_write_at=None, fail_delete=False):
        self.writes = []
        self.events = []
        self.existing = existing or {}
        self.fail_write_at = fail_write_at
        self.fail_delete = fail_delete

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def query_existing_feedback_batch(self, _refs):
        return self.existing

    def write_score(self, score, ref):
        self.events.append(("write", score.scorer))
        if self.fail_write_at == len(self.writes) + 1:
            raise RuntimeError("write failed")
        self.writes.append((score, ref))

    def delete_feedback_ids(self, rows):
        self.events.append(("delete", rows))
        if self.fail_delete:
            raise RuntimeError("delete failed")


def test_stage_persists_artifacts_buffers_scores_and_writes_only_session_refs(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    weave = _Weave()

    def fake_judge(_session, _client, **kwargs):
        payload = {"ok": True}
        kwargs["artifact_recorder"](
            "digest/test",
            {
                "schema_version": "1",
                "kind": "chunk_digest",
                "content_digest": judging_artifact_payload_digest(payload),
                "payload": payload,
            },
        )
        assert kwargs["artifact_loader"]("digest/test") is not None
        return [
            Score(
                "judge.session_outcome",
                0.75,
                ["good_outcome"],
                {
                    "attempts": [],
                    "behavioral_feedback": [
                        {"success": "done", "problem": None, "desired_behavior": None}
                    ],
                },
                "session",
                reason="Success: done",
            )
        ]

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", fake_judge)
    dependencies = JudgingDependencies(
        store,
        lambda: weave,
        lambda: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )
    run_judging_stage(run, effective, threading.Event(), dependencies=dependencies)
    current = store.get(run.run_id)
    assert current is not None and current.judging_artifacts
    assert len(weave.writes) == 1
    assert "/agent_conversation/" in weave.writes[0][1]
    assert "/agent_turn/" not in weave.writes[0][1]
    assert current.judging_result["coverage_complete"] is True


@contextmanager
def context(value):
    yield value


def test_stage_does_not_write_partial_scores_when_any_session_rubric_fails(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    weave = _Weave()
    failure = runner.JudgeFailure("judge.session_outcome", "offline", "ReviewFailed", ())

    def fail(*_args, **_kwargs):
        raise runner.JudgeExecutionError([], [failure])

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", fail)
    deps = JudgingDependencies(
        store,
        lambda: weave,
        lambda: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )
    with pytest.raises(runner.JudgeExecutionError):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    assert weave.writes == []


def test_stage_completes_unanimous_abstention_without_writing_feedback(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    weave = _Weave()
    outcome = runner.JudgeNotEvaluable("judge.session_outcome", ())

    def abstain(*_args, **_kwargs):
        raise runner.JudgeExecutionError([], [], [outcome])

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", abstain)
    deps = JudgingDependencies(
        store,
        lambda: weave,
        lambda: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )

    run_judging_stage(run, effective, threading.Event(), dependencies=deps)

    result = store.get(run.run_id).judging_result
    assert result["not_evaluable_rubrics"] == 1
    assert result["failure_count"] == 0
    assert result["coverage_complete"] is True
    assert weave.writes == []


def test_stage_with_no_applicable_reviewers_avoids_chat_and_judge_feedback(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    from weave_agent_signals.judges.windowing import WindowPlanInapplicable

    monkeypatch.setattr(
        "weave_agent_signals.judges.plan.build_window_plan",
        lambda *_args: (_ for _ in ()).throw(WindowPlanInapplicable()),
    )
    chat_factory = Mock(side_effect=AssertionError("chat backend must not be created"))
    weave = _Weave()
    deps = JudgingDependencies(
        store,
        lambda: weave,
        chat_factory,
        lambda _cohort: ([turn], {"session-1": session}),
    )

    run_judging_stage(run, effective, threading.Event(), dependencies=deps)

    result = store.get(run.run_id).judging_result
    assert result["not_evaluable_rubrics"] == 1
    assert result["reviewer_attempts_completed"] == 0
    assert result["attempt_summaries"][0]["attempts"][0]["status"] == "skipped"
    assert weave.writes == []


def test_stage_passes_exact_pinned_plan_and_context_to_runner(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    captured = Mock(
        return_value=[Score("judge.session_outcome", 0.75, [], {"attempts": []}, "session")]
    )
    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", captured)
    deps = JudgingDependencies(
        store, _Weave, lambda: context(None), lambda _cohort: ([turn], {"session-1": session})
    )
    run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    kwargs = captured.call_args.kwargs
    assert kwargs["judging_plan"]["schema_version"] == "3"
    assert kwargs["context_policy"] == effective.judging_context
    assert kwargs["rubrics"] == list(effective.rubrics)


def test_stage_persists_unique_artifact_progress_before_mid_session_cancellation(
    store, monkeypatch
):
    run, effective, turn, session = _setup(store)

    def cancel_after_artifact(_session, _client, **kwargs):
        payload = {"ok": True}
        artifact = {
            "schema_version": "1",
            "kind": "chunk_digest",
            "content_digest": judging_artifact_payload_digest(payload),
            "payload": payload,
        }
        kwargs["artifact_recorder"]("digest/test", artifact)
        kwargs["artifact_recorder"]("digest/test", artifact)
        raise InferenceCancelled("cancelled")

    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session", cancel_after_artifact
    )
    deps = JudgingDependencies(
        store,
        _Weave,
        lambda: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )
    with pytest.raises(InferenceCancelled):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    current = store.get(run.run_id)
    assert current is not None
    assert current.judging_progress["digest_steps_completed"] == 1


def test_stage_honors_cancellation_before_inference(store):
    run, effective, turn, session = _setup(store)
    cancel = threading.Event()
    cancel.set()
    deps = JudgingDependencies(
        store,
        _Weave,
        lambda: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )
    with pytest.raises(StageCancelled):
        run_judging_stage(run, effective, cancel, dependencies=deps)


def test_stage_cancellation_after_inference_blocks_external_writes(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    cancel = threading.Event()
    weave = _Weave()

    def finish_then_cancel(*_args, **_kwargs):
        cancel.set()
        return _scores_for(("judge.session_outcome",))

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", finish_then_cancel)
    deps = JudgingDependencies(
        store, lambda: weave, lambda: context(None), lambda _: ([turn], {"session-1": session})
    )
    with pytest.raises(StageCancelled):
        run_judging_stage(run, effective, cancel, dependencies=deps)
    assert weave.writes == []


def test_durable_cancellation_race_at_write_barrier_becomes_stage_cancelled(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    weave = _Weave()
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: _scores_for(("judge.session_outcome",)),
    )
    original = store.external_write_barrier

    def cancel_then_barrier(run_id: str, status: RunStatus):
        store.cancel_if_safe(run_id)
        return original(run_id, status)

    monkeypatch.setattr(store, "external_write_barrier", cancel_then_barrier)
    deps = JudgingDependencies(
        store, lambda: weave, lambda: context(None), lambda _: ([turn], {"session-1": session})
    )
    with pytest.raises(StageCancelled):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    current = store.get(run.run_id)
    assert current.status is RunStatus.CANCELLED
    assert current.judging_result is None
    assert weave.writes == []


def _scores_for(rubric_ids: tuple[str, ...]) -> list[Score]:
    return [Score(rubric_id, 0.75, [], {"attempts": []}, "session") for rubric_id in rubric_ids]


def test_force_writes_new_score_before_deleting_prior_feedback(store, monkeypatch):
    run, effective, turn, session = _setup(store, force=True)
    ref = session.ref_for("weave-team", "agent-sessions")
    feedback_type = "weave_agent_signals.judge.session_outcome"
    weave = _Weave(existing={(ref, feedback_type): [{"id": "old"}]})
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: _scores_for(("judge.session_outcome",)),
    )
    deps = JudgingDependencies(
        store, lambda: weave, lambda: context(None), lambda _: ([turn], {"session-1": session})
    )
    run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    assert weave.events == [
        ("write", "judge.session_outcome"),
        ("delete", [{"id": "old"}]),
    ]


def test_force_create_failure_preserves_prior_feedback(store, monkeypatch):
    run, effective, turn, session = _setup(store, force=True)
    ref = session.ref_for("weave-team", "agent-sessions")
    feedback_type = "weave_agent_signals.judge.session_outcome"
    weave = _Weave(existing={(ref, feedback_type): [{"id": "old"}]}, fail_write_at=1)
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: _scores_for(("judge.session_outcome",)),
    )
    deps = JudgingDependencies(
        store, lambda: weave, lambda: context(None), lambda _: ([turn], {"session-1": session})
    )
    with pytest.raises(RuntimeError, match="write incomplete"):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    assert weave.events == [("write", "judge.session_outcome")]


def test_force_cleanup_failure_reports_incomplete_after_new_score_exists(store, monkeypatch):
    run, effective, turn, session = _setup(store, force=True)
    ref = session.ref_for("weave-team", "agent-sessions")
    feedback_type = "weave_agent_signals.judge.session_outcome"
    weave = _Weave(existing={(ref, feedback_type): [{"id": "old"}]}, fail_delete=True)
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: _scores_for(("judge.session_outcome",)),
    )
    deps = JudgingDependencies(
        store, lambda: weave, lambda: context(None), lambda _: ([turn], {"session-1": session})
    )
    with pytest.raises(RuntimeError, match="write incomplete"):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    assert len(weave.writes) == 1
    assert [event[0] for event in weave.events] == ["write", "delete"]


def test_partial_write_failure_reports_incomplete_after_attempting_all_scores(store, monkeypatch):
    rubric_ids = ("judge.session_outcome", "judge.session_autonomy")
    run, effective, turn, session = _setup(store, rubric_ids=rubric_ids)
    weave = _Weave(fail_write_at=2)
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: _scores_for(rubric_ids),
    )
    deps = JudgingDependencies(
        store, lambda: weave, lambda: context(None), lambda _: ([turn], {"session-1": session})
    )
    with pytest.raises(RuntimeError, match="write incomplete"):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    current = store.get(run.run_id)
    assert current.judging_result["write_failure_count"] == 1
    assert len(weave.writes) == 1


def test_resumed_artifact_progress_is_reconstructed_and_not_double_counted(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=effective.rubrics,
        judge_models=effective.models.judges,
        context_policy=effective.judging_context,
    )
    store.pin_judging_plan(run.run_id, plan)
    payload = {"ok": True}
    artifact = {
        "schema_version": "1",
        "kind": "chunk_digest",
        "content_digest": judging_artifact_payload_digest(payload),
        "payload": payload,
    }
    store.record_judging_artifact(run.run_id, "digest/test", artifact)
    score = Score(
        "judge.session_outcome",
        0.75,
        [],
        {
            "attempts": [
                {"steps": [{"phase": "digest", "artifact_id": "digest/test", "reused": True}]}
            ]
        },
        "session",
    )
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: [score],
    )
    deps = JudgingDependencies(
        store, _Weave, lambda: context(None), lambda _: ([turn], {"session-1": session})
    )
    run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    current = store.get(run.run_id)
    assert current.judging_result["digest_steps_completed"] == 1


def test_failure_and_attempt_summaries_are_bounded(store, monkeypatch):
    rubric_ids = ("judge.session_outcome", "judge.session_autonomy")
    run, effective, turn, session = _setup(store, rubric_ids=rubric_ids)
    monkeypatch.setattr("weave_agent_signals.runs.stages.judging._MAX_PERSISTED_OUTCOMES", 1)
    failures = [
        runner.JudgeFailure(rubric_id, "offline", "ReviewFailed", ()) for rubric_id in rubric_ids
    ]

    def fail(*_args, **_kwargs):
        raise runner.JudgeExecutionError([], failures)

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", fail)
    deps = JudgingDependencies(
        store, _Weave, lambda: context(None), lambda _: ([turn], {"session-1": session})
    )
    with pytest.raises(runner.JudgeExecutionError):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    current = store.get(run.run_id)
    result = current.judging_result
    assert result["failure_detail_count"] == 2
    assert len(result["failure_details"]) == 1
    assert result["failure_details_truncated"] is True
    assert result["attempt_summary_count"] == 2
    assert len(result["attempt_summaries"]) == 1
