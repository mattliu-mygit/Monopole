from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.ces_catalog import Catalog, ScorerRef, judge_spec
from weave_agent_signals.judges import runner
from weave_agent_signals.judges.inference import InferenceCancelled
from weave_agent_signals.judges.plan import build_canonical_judging_plan
from weave_agent_signals.judges.records import JudgeCallAudit, JudgeCallRecord
from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.run_config import RunConfig, resolve_run_config
from weave_agent_signals.runs.stages import StageCancelled
from weave_agent_signals.runs.stages.judging import JudgingDependencies, run_judging_stage
from weave_agent_signals.runs.store import (
    DataSelection,
    RunStatus,
    RunStore,
)


@pytest.fixture
def store(tmp_path):
    value = RunStore(tmp_path / "runs.db")
    yield value
    value.close()


def _judge_call() -> JudgeCallRecord:
    return JudgeCallRecord(
        request_id="sha256:" + "0" * 64,
        phase="digest",
        conversation_id="session-1",
        reviewer_position=1,
        requested_model_id="claude:claude-sonnet-5",
        rubric_id=None,
        status="succeeded",
        reusable=True,
        result={"ok": True},
        audit=JudgeCallAudit(
            resolved_model="claude-sonnet-5",
            output_mode="json_schema",
            schema_name="chunk_digest",
            transport_request_count=1,
            raw_output_digest="0" * 64,
        ),
        created_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )


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
    extra_session_ids: tuple[str, ...] = (),
):
    turn = _turn()
    session = SessionView("session-1", [turn], "cfg", "main")
    extra_turns = [
        replace(
            turn,
            trace_id=f"turn-{index}",
            conversation_id=conversation_id,
            session_id=f"sid-{index}",
        )
        for index, conversation_id in enumerate(extra_session_ids, start=2)
    ]
    models = build_model_catalog(which=lambda name: f"/bin/{name}")
    rubrics = build_rubric_catalog()
    request = RunConfig(
        model_catalog_version=models.catalog_version,
        rubric_catalog_version=rubrics.catalog_version,
        judge_models=("claude:claude-sonnet-5",),
        challenge_judge_models=("codex:gpt-5.6-sol",),
        proposal_model="codex:gpt-5.6-sol",
        proposal_evaluator_model="claude:claude-sonnet-5",
        rubrics=rubric_ids,
        candidate_budget=3,
        force=force,
    )
    effective = resolve_run_config(request, model_catalog=models, rubric_catalog=rubrics)
    cohort = {
        "schema_version": 3,
        "pinned_at": now_iso(),
        "cohort_id": "cohort",
        "turn_count": 1 + len(extra_turns),
        "session_count": 1 + len(extra_turns),
        "turns": [
            {
                "trace_id": "turn-1",
                "case_ids": ["turn-1"],
                "conversation_id": "session-1",
                "started_at": turn.started_at.isoformat(),
                "model": "agent",
                "model_family": "unknown",
            },
            *[
                {
                    "trace_id": extra.trace_id,
                    "case_ids": [extra.trace_id],
                    "conversation_id": extra.conversation_id,
                    "started_at": extra.started_at.isoformat(),
                    "model": "agent",
                    "model_family": "unknown",
                }
                for extra in extra_turns
            ],
        ],
        "sessions": [
            {"conversation_id": "session-1", "turn_count": 1},
            *[{"conversation_id": extra.conversation_id, "turn_count": 1} for extra in extra_turns],
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


CATALOG = Catalog("ev-turns", "ev-sessions", {})


class _Ces:
    """A CesSession stand-in that records each session's write request."""

    def __init__(self, *, existing=None, fail_write_at=None):
        self.writes = []
        self.events = []
        self.existing = existing or {}
        self.fail_write_at = fail_write_at
        self.client = self
        self.catalog = CATALOG

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def add_scorers(self, catalog, specs):
        refs = {
            spec.name: ScorerRef(spec, f"sc-{spec.name}", "v", f"sc-{spec.name}.details", "v")
            for spec in specs
        }
        return Catalog(
            catalog.turns_evaluation_id, catalog.sessions_evaluation_id, {**catalog.scorers, **refs}
        )

    def session_scores(self, catalog, trajectory_ids, *, subagents=False):
        return {
            sid: [{"scorer_id": f"sc-{name}"} for name in self.existing.get(sid, [])]
            for sid in trajectory_ids
        }

    def write_scores(self, catalog, pass_id, kind, target_id, scores):
        names = [score.scorer for score in scores]
        self.events.append(("write", names))
        if self.fail_write_at == len(self.events):
            raise RuntimeError("write failed")
        self.writes.append((pass_id, kind, target_id, names))
        return []


def _panel(rubric_id: str) -> str:
    return judge_spec(rubric_id, _scores_for((rubric_id,))[0].metadata).name


class _ActivityChat:
    def __init__(self):
        self.activity = None

    def set_activity(self, callback):
        self.activity = callback

    def emit(self, event):
        assert self.activity is not None
        self.activity(event)


def test_stage_persists_calls_buffers_scores_and_writes_only_session_refs(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    ces = _Ces()

    def fake_judge(_session, _client, **kwargs):
        call = _judge_call()
        kwargs["call_recorder"](call)
        assert kwargs["call_loader"](call.request_id) == call
        return [
            Score(
                "judge.session_outcome",
                0.75,
                ["good_outcome"],
                {
                    "review_status": "complete",
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
        lambda: ces,
        lambda _judge: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )
    run_judging_stage(run, effective, threading.Event(), dependencies=dependencies)
    current = store.get(run.run_id)
    assert current is not None and store.list_judge_calls(run.run_id) == [_judge_call()]
    assert [(pass_id, kind, target) for pass_id, kind, target, _ in ces.writes] == [
        (run.run_id, "trajectory", "session-1")
    ]
    assert ces.writes[0][3][0].startswith("judge.session_outcome.panel-")
    assert current.judging_result["coverage_complete"] is True


def test_stage_persists_safe_semantic_and_transport_activity(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    chat = _ActivityChat()
    secret = "do-not-persist"

    def fake_judge(_session, clients, **kwargs):
        client = next(iter(clients.values()))
        kwargs["activity"](
            {
                "phase": "digest_started",
                "message": "Claude Sonnet is digesting chunk 1 of 1",
                "model": "claude-sonnet-5",
                "conversation_id": "session-1",
                "artifact_id": "digest/test",
                "item_index": 1,
                "item_total": 1,
            }
        )
        client.emit(
            {
                "phase": "transport_retry",
                "message": f"Sonnet failed; token={secret}; retrying attempt 2 of 3",
                "model": "claude-sonnet-5",
                "request_attempt": 2,
                "max_attempts": 3,
                "elapsed_seconds": 191.8,
                "error_category": "retryable_process_error",
                "provider_status": 429,
                "provider_error_code": "rate_limit_exceeded",
                "provider_error_message": f"Too many requests; token={secret}",
                "finish_reason": "length",
                "completion_tokens": 10_000,
                "reasoning_tokens": 9_500,
                "output_sha256": "a" * 64,
                "raw_output": secret,
            }
        )
        client.emit(
            {
                "phase": "transport_recovered",
                "message": "Sonnet recovered on request attempt 2 of 3",
                "model": "claude-sonnet-5",
                "request_attempt": 2,
                "max_attempts": 3,
                "elapsed_seconds": 202.0,
                "output_sha256": "b" * 64,
            }
        )
        return _scores_for(("judge.session_outcome",))

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", fake_judge)
    deps = JudgingDependencies(
        store,
        _Ces,
        lambda _judge: context(chat),
        lambda _: ([turn], {"session-1": session}),
    )

    run_judging_stage(run, effective, threading.Event(), dependencies=deps)

    result = store.get(run.run_id).judging_result
    assert result["phase"] == "judging_complete"
    assert result["started_at"]
    events = store.list_run_events(run.run_id, stage="judging")
    assert [event.phase for event in events] == [
        "judging_started",
        "session_started",
        "digest_started",
        "transport_retry",
        "transport_recovered",
        "rubrics_completed",
        "score_writes_started",
        "judging_complete",
    ]
    retry = next(event for event in events if event.phase == "transport_retry")
    assert retry.details["elapsed_seconds"] == 191.8
    assert retry.details["error_category"] == "retryable_process_error"
    assert retry.details["provider_status"] == 429
    assert retry.details["provider_error_code"] == "rate_limit_exceeded"
    assert retry.details["provider_error_message"] == "Too many requests; token=[REDACTED]"
    assert retry.details["finish_reason"] == "length"
    assert retry.details["completion_tokens"] == 10_000
    assert retry.details["reasoning_tokens"] == 9_500
    assert retry.details["output_sha256"] == "a" * 64
    assert "raw_output" not in retry.details
    assert secret not in str(events)


@contextmanager
def context(value):
    yield value


def test_stage_does_not_write_partial_scores_when_any_session_rubric_fails(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    ces = _Ces()
    failure = runner.JudgeFailure("judge.session_outcome", "offline", "ReviewFailed", ())

    def fail(*_args, **_kwargs):
        raise runner.JudgeExecutionError([], [failure])

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", fail)
    deps = JudgingDependencies(
        store,
        lambda: ces,
        lambda _judge: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )
    with pytest.raises(runner.JudgeExecutionError):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    assert ces.writes == []


def test_stage_preserves_successful_reviewer_count_for_failed_panel(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    attempts = (
        {"status": "succeeded", "requested_model": "judge-1", "score": 0.75},
        {"status": "abstained", "requested_model": "judge-2"},
        {
            "status": "failed",
            "requested_model": "judge-3",
            "error_type": "ReadTimeout",
            "message": "judge invocation failed",
            "transport_request_count": 6,
        },
    )
    failure = runner.JudgeFailure(
        "judge.session_outcome",
        "judge-2 (ValueError): unknown evidence ID",
        "ReviewFailed",
        attempts,
    )

    def fail(*_args, **_kwargs):
        raise runner.JudgeExecutionError([], [failure])

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", fail)
    deps = JudgingDependencies(
        store,
        _Ces,
        lambda _judge: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )

    with pytest.raises(runner.JudgeExecutionError):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)

    summary = store.get(run.run_id).judging_result["attempt_summaries"][0]
    assert summary["successful_reviewer_count"] == 1
    failed_event = store.list_run_events(run.run_id, stage="judging")[-1]
    assert failed_event.phase == "judging_failed"
    assert failed_event.message == (
        "Session Outcome Quality failed: judge-1 scored 0.75; "
        "judge-2 abstained; judge-3 failed ReadTimeout after 6 provider requests: "
        "judge invocation failed"
    )


def test_stage_stops_remaining_sessions_after_first_review_failure(store, monkeypatch):
    run, effective, turn, session = _setup(store, extra_session_ids=("session-2",))
    second_turn = replace(
        turn,
        trace_id="turn-2",
        conversation_id="session-2",
        session_id="sid-2",
    )
    second_session = SessionView("session-2", [second_turn], "cfg", "main")
    calls = []
    failure = runner.JudgeFailure("judge.session_outcome", "offline", "ReviewFailed", ())

    def fail(current_session, *_args, **_kwargs):
        calls.append(current_session.conversation_id)
        raise runner.JudgeExecutionError([], [failure])

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", fail)
    deps = JudgingDependencies(
        store,
        _Ces,
        lambda _judge: context(None),
        lambda _cohort: (
            [turn, second_turn],
            {"session-1": session, "session-2": second_session},
        ),
    )

    with pytest.raises(runner.JudgeExecutionError):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)

    assert calls == ["session-1"]


def test_stage_completes_unanimous_abstention_without_writing_feedback(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    ces = _Ces()
    outcome = runner.JudgeNotEvaluable("judge.session_outcome", ())

    def abstain(*_args, **_kwargs):
        raise runner.JudgeExecutionError([], [], [outcome])

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", abstain)
    deps = JudgingDependencies(
        store,
        lambda: ces,
        lambda _judge: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )

    run_judging_stage(run, effective, threading.Event(), dependencies=deps)

    result = store.get(run.run_id).judging_result
    assert result["not_evaluable_rubrics"] == 1
    assert result["failure_count"] == 0
    assert result["coverage_complete"] is True
    assert ces.writes == []


def test_stage_with_no_applicable_reviewers_avoids_chat_and_judge_feedback(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    from weave_agent_signals.judges.windowing import WindowPlanInapplicable

    def inapplicable(*_args):
        raise WindowPlanInapplicable()

    monkeypatch.setattr("weave_agent_signals.judges.plan.build_window_plan", inapplicable)
    chat_factory = Mock(side_effect=AssertionError("chat backend must not be created"))
    ces = _Ces()
    deps = JudgingDependencies(
        store,
        lambda: ces,
        chat_factory,
        lambda _cohort: ([turn], {"session-1": session}),
    )

    run_judging_stage(run, effective, threading.Event(), dependencies=deps)

    result = store.get(run.run_id).judging_result
    assert result["not_evaluable_rubrics"] == 1
    assert result["reviewer_attempts_completed"] == 0
    assert result["attempt_summaries"][0]["attempts"][0]["status"] == "skipped"
    assert ces.writes == []


def test_stage_passes_exact_pinned_plan_and_context_to_runner(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    captured = Mock(
        return_value=[
            Score(
                "judge.session_outcome",
                0.75,
                [],
                {"review_status": "complete", "attempts": []},
                "session",
            )
        ]
    )
    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", captured)
    deps = JudgingDependencies(
        store,
        _Ces,
        lambda _judge: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )
    run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    kwargs = captured.call_args.kwargs
    assert kwargs["judging_plan"].protocol_version == "3"
    assert kwargs["context_policy"] == effective.judging_context
    assert kwargs["rubrics"] == list(effective.rubrics)


def test_stage_persists_unique_call_progress_before_mid_session_cancellation(store, monkeypatch):
    run, effective, turn, session = _setup(store)

    def cancel_after_call(_session, _client, **kwargs):
        call = _judge_call()
        kwargs["call_recorder"](call)
        kwargs["call_recorder"](call)
        raise InferenceCancelled("cancelled")

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", cancel_after_call)
    deps = JudgingDependencies(
        store,
        _Ces,
        lambda _judge: context(None),
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
        _Ces,
        lambda _judge: context(None),
        lambda _cohort: ([turn], {"session-1": session}),
    )
    with pytest.raises(StageCancelled):
        run_judging_stage(run, effective, cancel, dependencies=deps)


def test_stage_cancellation_after_inference_blocks_external_writes(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    cancel = threading.Event()
    ces = _Ces()

    def finish_then_cancel(*_args, **_kwargs):
        cancel.set()
        return _scores_for(("judge.session_outcome",))

    monkeypatch.setattr("weave_agent_signals.runs.stages.judging.judge_session", finish_then_cancel)
    deps = JudgingDependencies(
        store,
        lambda: ces,
        lambda _judge: context(None),
        lambda _: ([turn], {"session-1": session}),
    )
    with pytest.raises(StageCancelled):
        run_judging_stage(run, effective, cancel, dependencies=deps)
    assert ces.writes == []


def test_durable_cancellation_race_at_write_barrier_becomes_stage_cancelled(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    ces = _Ces()
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
        store,
        lambda: ces,
        lambda _judge: context(None),
        lambda _: ([turn], {"session-1": session}),
    )
    with pytest.raises(StageCancelled):
        run_judging_stage(run, effective, threading.Event(), dependencies=deps)
    current = store.get(run.run_id)
    assert current.status is RunStatus.CANCELLED
    assert current.judging_result is None
    assert ces.writes == []


def _scores_for(rubric_ids: tuple[str, ...]) -> list[Score]:
    return [
        Score(rubric_id, 0.75, [], {"review_status": "complete", "attempts": []}, "session")
        for rubric_id in rubric_ids
    ]


def _deps(store, ces, turns, sessions):
    return JudgingDependencies(
        store,
        lambda: ces,
        lambda _judge: context(None),
        lambda _: (turns, sessions),
    )


def test_an_existing_panel_score_is_skipped_unless_forced(store, monkeypatch):
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: _scores_for(("judge.session_outcome",)),
    )
    existing = {"session-1": [_panel("judge.session_outcome")]}

    run, effective, turn, session = _setup(store)
    ces = _Ces(existing=existing)
    run_judging_stage(
        run,
        effective,
        threading.Event(),
        dependencies=_deps(store, ces, [turn], {"session-1": session}),
    )
    assert ces.events == []

    run, effective, turn, session = _setup(store, force=True)
    ces = _Ces(existing=existing)
    run_judging_stage(
        run,
        effective,
        threading.Event(),
        dependencies=_deps(store, ces, [turn], {"session-1": session}),
    )
    assert ces.events == [("write", [_panel("judge.session_outcome")])]


def test_an_unfinished_review_writes_no_score(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    degraded = _scores_for(("judge.session_outcome",))
    degraded[0].metadata["review_status"] = "degraded"
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: degraded,
    )
    ces = _Ces()

    run_judging_stage(
        run,
        effective,
        threading.Event(),
        dependencies=_deps(store, ces, [turn], {"session-1": session}),
    )

    assert ces.events == []


def test_force_write_failure_reports_incomplete(store, monkeypatch):
    run, effective, turn, session = _setup(store, force=True)
    ces = _Ces(fail_write_at=1)
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: _scores_for(("judge.session_outcome",)),
    )

    with pytest.raises(RuntimeError, match="write incomplete"):
        run_judging_stage(
            run,
            effective,
            threading.Event(),
            dependencies=_deps(store, ces, [turn], {"session-1": session}),
        )

    assert ces.events == [("write", [_panel("judge.session_outcome")])]
    assert ces.writes == []


def test_partial_write_failure_reports_incomplete_after_attempting_every_session(
    store, monkeypatch
):
    run, effective, turn, session = _setup(store, extra_session_ids=("session-2",))
    extra = replace(turn, trace_id="turn-2", conversation_id="session-2", session_id="sid-2")
    sessions = {"session-1": session, "session-2": SessionView("session-2", [extra], "cfg", "main")}
    ces = _Ces(fail_write_at=1)
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: _scores_for(("judge.session_outcome",)),
    )

    with pytest.raises(RuntimeError, match="write incomplete"):
        run_judging_stage(
            run,
            effective,
            threading.Event(),
            dependencies=_deps(store, ces, [turn, extra], sessions),
        )

    current = store.get(run.run_id)
    assert current.judging_result["write_failure_count"] == 1
    assert [target for _, _, target, _ in ces.writes] == ["session-2"]


def test_resumed_call_progress_is_reconstructed_and_not_double_counted(store, monkeypatch):
    run, effective, turn, session = _setup(store)
    plan = build_canonical_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=effective.rubrics,
        judge_models=effective.models.judges,
        context_policy=effective.judging_context,
    )
    store.pin_judging_plan(run.run_id, plan)
    call = _judge_call()
    store.record_judge_call(run.run_id, call)
    score = Score(
        "judge.session_outcome",
        0.75,
        [],
        {
            "review_status": "complete",
            "attempts": [
                {"steps": [{"phase": "digest", "artifact_id": call.request_id, "reused": True}]}
            ],
        },
        "session",
    )
    monkeypatch.setattr(
        "weave_agent_signals.runs.stages.judging.judge_session",
        lambda *_args, **_kwargs: [score],
    )
    deps = JudgingDependencies(
        store,
        _Ces,
        lambda _judge: context(None),
        lambda _: ([turn], {"session-1": session}),
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
        store,
        _Ces,
        lambda _judge: context(None),
        lambda _: ([turn], {"session-1": session}),
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
