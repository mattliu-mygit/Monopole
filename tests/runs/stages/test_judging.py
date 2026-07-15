from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.judges import runner
from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.run_config import RunConfig, resolve_run_config
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


def _setup(store: RunStore):
    turn = _turn()
    session = SessionView("session-1", [turn], "cfg", "main")
    models = build_model_catalog(which=lambda name: f"/bin/{name}")
    rubrics = build_rubric_catalog()
    request = RunConfig(
        model_catalog_version=models.catalog_version,
        rubric_catalog_version=rubrics.catalog_version,
        judge_backend="cli",
        review_depth="primary",
        judge_models=("claude-sonnet-5",),
        second_opinion_margin=None,
        proposal_model="gpt-5.6-sol",
        proposal_evaluator_model="claude-sonnet-5",
        rubrics=("judge.session_outcome",),
        candidate_budget=3,
        force=False,
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

    def __init__(self):
        self.writes = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def query_existing_feedback_batch(self, _refs):
        return {}

    def write_score(self, score, ref):
        self.writes.append((score, ref))

    def delete_feedback_ids(self, _rows):
        raise AssertionError("no old rows")


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
    assert kwargs["judging_plan"]["schema_version"] == "2"
    assert kwargs["context_policy"] == effective.judging_context
    assert kwargs["rubrics"] == list(effective.rubrics)
