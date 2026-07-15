from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.judges.families import model_family
from weave_agent_signals.judges.inference import InferenceCancelled
from weave_agent_signals.judges.plan import build_judging_plan
from weave_agent_signals.judges.runner import JudgeExecutionError, JudgeFailure
from weave_agent_signals.models import Score, SessionView, ToolSpan, TurnSpan
from weave_agent_signals.run_config import RunConfig, resolve_run_config
from weave_agent_signals.runs.stages import StageCancelled
from weave_agent_signals.runs.stages import judging as judging_stage
from weave_agent_signals.runs.stages.judging import (
    JudgingDependencies,
    run_judging_stage,
)
from weave_agent_signals.runs.store import DataSelection, RunStatus, RunStore


@pytest.fixture
def store(tmp_path):
    instance = RunStore(tmp_path / "runs.db")
    yield instance
    instance.close()


def _turn(
    trace_id: str,
    *,
    minute: int,
    tools: bool = False,
) -> TurnSpan:
    started_at = datetime(2026, 7, 14, 12, minute, tzinfo=timezone.utc)
    tool_calls = []
    if tools:
        tool_calls.append(
            ToolSpan(
                span_id=f"tool-{trace_id}",
                tool_name="Read",
                arguments='{"path":"app.py"}',
                result="ok",
                status_code="SUCCESS",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
            )
        )
    return TurnSpan(
        trace_id=trace_id,
        conversation_id="conversation-1",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=2),
        model="gpt-5.6-sol",
        input_tokens=11,
        output_tokens=7,
        cache_read_tokens=0,
        status_code="SUCCESS",
        config_version="config-v1",
        git_branch="main",
        effort_level="medium",
        session_id="session-1",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=tool_calls,
        chat_spans=[],
        subagents=[],
        user_input=f"request {trace_id}",
        assistant_output=f"response {trace_id}",
    )


def _sessions(turns: list[TurnSpan]) -> dict[str, SessionView]:
    return {
        "conversation-1": SessionView(
            conversation_id="conversation-1",
            turns=turns,
            config_version="config-v1",
            git_branch="main",
        )
    }


def _cohort(turns: list[TurnSpan]) -> dict:
    session = _sessions(turns)["conversation-1"]
    return {
        "schema_version": 1,
        "pinned_at": "2026-07-14T12:30:00+00:00",
        "cohort_id": "sha256:test-cohort",
        "turn_count": len(turns),
        "session_count": 1,
        "turns": [
            {
                "trace_id": turn.trace_id,
                "weave_ref": turn.ref_for(),
                "conversation_id": turn.conversation_id,
                "started_at": turn.started_at.isoformat(),
                "model": turn.model,
                "model_family": model_family(turn.model or ""),
            }
            for turn in turns
        ],
        "sessions": [
            {
                "conversation_id": session.conversation_id,
                "weave_ref": session.ref_for(),
                "turn_count": len(turns),
            }
        ],
    }


def _start(
    store: RunStore,
    turns: list[TurnSpan],
    rubric_ids: tuple[str, ...],
    *,
    force: bool = False,
    pin_plan: bool = True,
):
    models = build_model_catalog(which=lambda name: f"/bin/{name}")
    rubrics = build_rubric_catalog()
    requested = RunConfig(
        model_catalog_version=models.catalog_version,
        rubric_catalog_version=rubrics.catalog_version,
        judge_backend="cli",
        review_depth="selective",
        judge_models=("claude-sonnet-5", "gpt-5.6-sol"),
        second_opinion_margin=0.1,
        proposal_model="gpt-5.6-sol",
        proposal_evaluator_model="claude-sonnet-5",
        rubrics=rubric_ids,
        candidate_budget=3,
        force=force,
    )
    effective = resolve_run_config(
        requested,
        model_catalog=models,
        rubric_catalog=rubrics,
    )
    selection = DataSelection(session_ids=("conversation-1",))
    created = store.create()
    store.save_selection(created.run_id, selection)
    store.save_config(created.run_id, requested)
    started = store.start(
        created.run_id,
        expected_selection=selection,
        expected_config=requested,
        turn_cohort=_cohort(turns),
        effective_config=effective,
    )
    store.record_stage_result(
        started.run_id,
        stage=RunStatus.SCORING,
        result={"scores_written": 1},
    )
    judging = store.finalize_stage_success(
        started.run_id,
        stage=RunStatus.SCORING,
        advance=True,
    )
    plan = build_judging_plan(
        list(_sessions(turns).values()),
        cohort_id=judging.turn_cohort["cohort_id"],
        rubrics=effective.rubrics,
        review_depth=effective.review_depth,
        judge_count=len(effective.models.judges),
    )
    if pin_plan:
        judging = store.pin_judging_plan(judging.run_id, plan)
    return judging, effective


def _attempt(
    status: str,
    *,
    position: int,
    rationale: str | None = None,
    message: str | None = None,
) -> dict[str, object]:
    resolved = status in {"succeeded", "abstained"}
    return {
        "position": position,
        "role": f"judge_{position}",
        "trigger": "initial" if position == 1 else "near_boundary",
        "requested_model": f"judge-{position}",
        "requested_family": f"family-{position}",
        "requested_backend": "cli",
        "status": status,
        "resolved_model": f"resolved-{position}" if resolved else None,
        "resolved_family": f"family-{position}" if resolved else None,
        "score": 0.6 if status == "succeeded" else None,
        "rationale": rationale,
        "evidence_ids": ["turn-1"] if status == "succeeded" else [],
        "usage": {"input_tokens": 5, "output_tokens": 2},
        "output_mode": "json_schema",
        "schema_name": "judge_verdict",
        "schema_fallback_reason": None,
        "transport_request_count": 1,
        "verdict_schema_version": 3,
        "raw_output_digest": f"{position:064x}",
        "error_type": "RuntimeError" if status == "failed" else None,
        "message": message,
    }


def _score(
    scorer: str,
    *,
    status: str,
    attempts: list[dict[str, object]],
    granularity: str,
) -> Score:
    return Score(
        scorer=scorer,
        value=0.6,
        tags=[],
        metadata={
            "review_status": status,
            "attempt_count": len(attempts),
            "successful_reviewer_count": sum(
                attempt["status"] == "succeeded" for attempt in attempts
            ),
            "attempts": attempts,
        },
        granularity=granularity,
        reason="auditable rationale",
    )


class TrackingWeaveClient:
    entity = "weave-team"
    project = "agent-sessions"

    def __init__(
        self,
        timeline: list[str],
        existing: dict[tuple[str, str], list[dict]] | None = None,
        *,
        write_error: Exception | None = None,
    ) -> None:
        self.timeline = timeline
        self.existing = existing or {}
        self.write_error = write_error
        self.events: list[tuple[str, object]] = []
        self.inside_barrier = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def query_existing_feedback_batch(self, refs: list[str]):
        self.timeline.append("query")
        self.events.append(("query", tuple(refs)))
        return self.existing

    def delete_feedback_ids(self, feedback: list[dict]) -> None:
        assert self.inside_barrier
        self.timeline.append("delete")
        self.events.append(("delete", tuple(item["id"] for item in feedback)))

    def write_score(self, score: Score, ref: str) -> dict:
        assert self.inside_barrier
        self.timeline.append(f"write:{score.scorer}")
        self.events.append(("write", (score, ref)))
        if self.write_error is not None:
            raise self.write_error
        return {"id": f"written-{score.scorer}"}


class TrackingChatClient:
    backend = "cli"

    def __init__(self) -> None:
        self.cancel: threading.Event | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def set_cancel(self, cancel: threading.Event) -> None:
        self.cancel = cancel


def _dependencies(
    store: RunStore,
    weave_client: TrackingWeaveClient,
    chat_client: TrackingChatClient,
    turns: list[TurnSpan],
) -> tuple[JudgingDependencies, Mock]:
    hydrate = Mock(return_value=(turns, _sessions(turns)))
    return (
        JudgingDependencies(
            store=store,
            client_factory=lambda: weave_client,
            chat_client_factory=lambda: chat_client,
            hydrate_cohort=hydrate,
        ),
        hydrate,
    )


def _track_barrier(
    store: RunStore,
    client: TrackingWeaveClient,
    timeline: list[str],
    monkeypatch,
) -> None:
    original = store.external_write_barrier

    @contextmanager
    def tracked(run_id: str, status: RunStatus) -> Iterator[None]:
        timeline.append("barrier")
        with original(run_id, status):
            client.inside_barrier = True
            try:
                yield
            finally:
                client.inside_barrier = False

    monkeypatch.setattr(store, "external_write_barrier", tracked)


def test_effective_review_policy_and_pinned_evidence_reach_both_wrappers(
    store,
    monkeypatch,
) -> None:
    turns = [_turn("turn-1", minute=0), _turn("turn-2", minute=2)]
    run, effective = _start(
        store,
        turns,
        ("judge.state_consistency", "judge.session_outcome"),
    )
    timeline: list[str] = []
    weave_client = TrackingWeaveClient(timeline)
    chat_client = TrackingChatClient()
    dependencies, hydrate = _dependencies(store, weave_client, chat_client, turns)
    _track_barrier(store, weave_client, timeline, monkeypatch)
    calls: list[tuple[str, dict]] = []

    def judge_turn(turn, client, **kwargs):
        timeline.append("turn")
        calls.append(("turn", {"turn": turn, "client": client, **kwargs}))
        return [
            _score(
                "judge.state_consistency",
                status="degraded",
                attempts=[
                    _attempt("failed", position=1, message="temporary failure"),
                    _attempt("succeeded", position=2, rationale="recovered"),
                ],
                granularity="turn",
            )
        ]

    def judge_session(session, client, **kwargs):
        timeline.append("session")
        calls.append(("session", {"session": session, "client": client, **kwargs}))
        return [
            _score(
                "judge.session_outcome",
                status="unresolved",
                attempts=[
                    _attempt("succeeded", position=1, rationale="pass"),
                    _attempt("succeeded", position=2, rationale="fail"),
                ],
                granularity="session",
            )
        ]

    monkeypatch.setattr(judging_stage, "judge_turn", judge_turn)
    monkeypatch.setattr(judging_stage, "judge_session", judge_session)

    run_judging_stage(
        run,
        effective,
        threading.Event(),
        dependencies=dependencies,
    )

    assert [name for name, _ in calls] == ["turn", "session"]
    for _, call in calls:
        assert tuple(call["judges"]) == effective.models.judges
        assert call["review_depth"] == effective.review_depth
        assert call["second_opinion_margin"] == effective.second_opinion_margin
        assert call["client"] is chat_client
    assert [turn.trace_id for turn in calls[0][1]["prior_turns"]] == ["turn-1"]
    assert calls[1][1]["evidence_trace_ids"] == ["turn-1", "turn-2"]
    assert timeline.index("query") > timeline.index("session")
    assert timeline.index("barrier") > timeline.index("query")
    assert timeline.index("write:judge.state_consistency") > timeline.index("barrier")
    assert chat_client.cancel is not None
    hydrate.assert_called_once_with(run.turn_cohort)
    queried_refs = next(value for event, value in weave_client.events if event == "query")
    assert queried_refs == (turns[1].ref_for(), _sessions(turns)["conversation-1"].ref_for())

    updated = store.get(run.run_id)
    assert updated is not None
    result = updated.judging_result
    assert result["planned_rubrics"] == 2
    assert result["rubrics_completed"] == 2
    assert result["rated_rubrics"] == 2
    assert result["reviewer_attempts_completed"] == 4
    assert result["scores_written"] == 2
    assert result["failure_count"] == 0
    assert result["coverage_complete"] is True
    assert {attempt["review_status"] for attempt in result["attempt_summaries"]} == {
        "degraded",
        "unresolved",
    }
    assert updated.judging_progress["reviewer_attempts_completed"] == 4
    assert updated.judging_progress["scores_written"] == 2
    writes = [value for event, value in weave_client.events if event == "write"]
    assert {score.metadata["plan_id"] for score, _ in writes} == {run.judging_plan["plan_id"]}
    assert {tuple(score.metadata["evidence_trace_ids"]) for score, _ in writes} == {
        ("turn-1", "turn-2")
    }


def test_zero_success_retains_partial_audit_but_blocks_every_external_mutation(
    store,
    monkeypatch,
) -> None:
    turns = [
        _turn("turn-1", minute=0),
        _turn("turn-2", minute=2, tools=True),
    ]
    run, effective = _start(
        store,
        turns,
        (
            "judge.tool_choice",
            "judge.state_consistency",
            "judge.session_outcome",
        ),
        force=True,
    )
    timeline: list[str] = []
    session = _sessions(turns)["conversation-1"]
    existing = {
        (
            turns[1].ref_for(),
            "weave_agent_signals.judge.tool_choice",
        ): [{"id": "old-turn"}],
        (
            session.ref_for(),
            "weave_agent_signals.judge.session_outcome",
        ): [{"id": "old-session"}],
    }
    weave_client = TrackingWeaveClient(timeline, existing)
    dependencies, _ = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    _track_barrier(store, weave_client, timeline, monkeypatch)
    wrapper_calls: list[str] = []
    partial = _score(
        "judge.tool_choice",
        status="complete",
        attempts=[_attempt("succeeded", position=1, rationale="r" * 2_000)],
        granularity="turn",
    )
    failed_attempts = (
        _attempt("failed", position=1, message="m" * 2_000),
        _attempt("failed", position=2, message="m" * 2_000),
    )

    def judge_turn(*_args, **_kwargs):
        wrapper_calls.append("turn")
        raise JudgeExecutionError(
            [partial],
            [
                JudgeFailure(
                    rubric="judge.state_consistency",
                    message="Every reviewer failed",
                    error_type="ReviewFailed",
                    attempts=failed_attempts,
                )
            ],
        )

    def judge_session(*_args, **_kwargs):
        wrapper_calls.append("session")
        return [
            _score(
                "judge.session_outcome",
                status="complete",
                attempts=[_attempt("succeeded", position=1, rationale="ok")],
                granularity="session",
            )
        ]

    monkeypatch.setattr(judging_stage, "judge_turn", judge_turn)
    monkeypatch.setattr(judging_stage, "judge_session", judge_session)

    with pytest.raises(JudgeExecutionError) as caught:
        run_judging_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    assert wrapper_calls == ["turn", "session"]
    assert {score.scorer for score in caught.value.scores} == {
        "judge.tool_choice",
        "judge.session_outcome",
    }
    assert [failure.rubric for failure in caught.value.failures] == ["judge.state_consistency"]
    assert timeline == []
    assert weave_client.events == []

    updated = store.get(run.run_id)
    assert updated is not None
    result = updated.judging_result
    assert result["planned_rubrics"] == 3
    assert result["rubrics_completed"] == 3
    assert result["rated_rubrics"] == 2
    assert result["reviewer_attempts_completed"] == 4
    assert result["scores_written"] == 0
    assert result["failure_count"] == 1
    assert result["coverage_complete"] is False
    assert len(result["failure_details"]) == 1
    assert all(
        len(attempt.get("rationale") or "") <= 500 and len(attempt.get("message") or "") <= 500
        for summary in result["attempt_summaries"]
        for attempt in summary["attempts"]
    )


def test_stage_persists_failed_coverage_without_feedback_write(store, monkeypatch) -> None:
    turns = [_turn("turn-1", minute=0)]
    run, effective = _start(store, turns, ("judge.session_outcome",))
    timeline: list[str] = []
    weave_client = TrackingWeaveClient(timeline)
    dependencies, _ = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    attempts = (
        _attempt("failed", position=1, message="invalid verdict"),
        _attempt("abstained", position=2, rationale="not enough evidence"),
    )
    monkeypatch.setattr(
        judging_stage,
        "judge_session",
        Mock(
            side_effect=JudgeExecutionError(
                [],
                [
                    JudgeFailure(
                        rubric="judge.session_outcome",
                        message="No selected judge produced a valid score",
                        error_type="ReviewFailed",
                        attempts=attempts,
                    )
                ],
            )
        ),
    )

    with pytest.raises(JudgeExecutionError):
        run_judging_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    assert timeline == []
    assert weave_client.events == []
    result = store.get(run.run_id).judging_result
    assert result["coverage_complete"] is False
    assert result["rated_rubrics"] == 0
    assert result["scores_written"] == 0
    assert result["reviewer_attempts_completed"] == 2
    summary = result["attempt_summaries"][0]
    assert summary["rating"] is None
    assert [attempt["status"] for attempt in summary["attempts"]] == [
        "failed",
        "abstained",
    ]
    assert summary["attempts"][1]["verdict_schema_version"] == 3
    assert summary["attempts"][1]["raw_output_digest"] == f"{2:064x}"
    assert "content" not in summary["attempts"][1]


def test_failed_attempt_audit_survives_summary_cap(store, monkeypatch) -> None:
    turns = [_turn("turn-1", minute=0)]
    run, effective = _start(
        store,
        turns,
        ("judge.session_outcome", "judge.session_autonomy"),
    )
    weave_client = TrackingWeaveClient([])
    dependencies, _ = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    monkeypatch.setattr(judging_stage, "_MAX_PERSISTED_OUTCOMES", 1)
    failed_attempts = (
        _attempt("failed", position=1, message="invalid verdict"),
        _attempt("abstained", position=2, rationale="not enough evidence"),
    )
    monkeypatch.setattr(
        judging_stage,
        "judge_session",
        Mock(
            side_effect=JudgeExecutionError(
                [
                    _score(
                        "judge.session_outcome",
                        status="complete",
                        attempts=[_attempt("succeeded", position=1, rationale="ok")],
                        granularity="session",
                    )
                ],
                [
                    JudgeFailure(
                        rubric="judge.session_autonomy",
                        message="No selected judge produced a valid score",
                        error_type="ReviewFailed",
                        attempts=failed_attempts,
                    )
                ],
            )
        ),
    )

    with pytest.raises(JudgeExecutionError):
        run_judging_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    result = store.get(run.run_id).judging_result
    assert result["attempt_summary_count"] == 2
    assert result["attempt_summaries_truncated"] is True
    assert len(result["attempt_summaries"]) == 1
    assert result["failure_detail_count"] == 1
    assert result["failure_details_truncated"] is False
    failure = result["failure_details"][0]
    assert failure["rubric"] == "judge.session_autonomy"
    assert [attempt["status"] for attempt in failure["attempts"]] == [
        "failed",
        "abstained",
    ]
    assert failure["attempts"][0]["requested_model"] == "judge-1"
    assert failure["attempts"][1]["raw_output_digest"] == f"{2:064x}"


@pytest.mark.parametrize("force", [False, True])
def test_existing_feedback_is_queried_then_skipped_or_rewritten_inside_barrier(
    store,
    monkeypatch,
    force,
) -> None:
    turns = [_turn("turn-1", minute=0)]
    run, effective = _start(
        store,
        turns,
        ("judge.session_outcome",),
        force=force,
    )
    timeline: list[str] = []
    session = _sessions(turns)["conversation-1"]
    weave_client = TrackingWeaveClient(
        timeline,
        {
            (
                session.ref_for(),
                "weave_agent_signals.judge.session_outcome",
            ): [{"id": "old-session"}, {"id": "duplicate-session"}]
        },
    )
    dependencies, _ = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    _track_barrier(store, weave_client, timeline, monkeypatch)
    monkeypatch.setattr(judging_stage, "judge_turn", Mock(return_value=[]))
    monkeypatch.setattr(
        judging_stage,
        "judge_session",
        Mock(
            return_value=[
                _score(
                    "judge.session_outcome",
                    status="complete",
                    attempts=[_attempt("succeeded", position=1, rationale="ok")],
                    granularity="session",
                )
            ]
        ),
    )

    run_judging_stage(
        run,
        effective,
        threading.Event(),
        dependencies=dependencies,
    )

    if force:
        assert timeline == [
            "query",
            "barrier",
            "write:judge.session_outcome",
            "delete",
        ]
        assert weave_client.events[-1] == (
            "delete",
            ("old-session", "duplicate-session"),
        )
        assert store.get(run.run_id).judging_result["scores_written"] == 1
    else:
        assert timeline == ["query"]
        assert store.get(run.run_id).judging_result["scores_written"] == 0


def test_force_create_failure_preserves_prior_judge_feedback(store, monkeypatch) -> None:
    turns = [_turn("turn-1", minute=0)]
    run, effective = _start(
        store,
        turns,
        ("judge.session_outcome",),
        force=True,
    )
    timeline: list[str] = []
    session = _sessions(turns)["conversation-1"]
    weave_client = TrackingWeaveClient(
        timeline,
        {
            (
                session.ref_for(),
                "weave_agent_signals.judge.session_outcome",
            ): [{"id": "old-session"}, {"id": "duplicate-session"}]
        },
        write_error=RuntimeError("create failed"),
    )
    dependencies, _ = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    _track_barrier(store, weave_client, timeline, monkeypatch)
    monkeypatch.setattr(judging_stage, "judge_turn", Mock(return_value=[]))
    monkeypatch.setattr(
        judging_stage,
        "judge_session",
        Mock(
            return_value=[
                _score(
                    "judge.session_outcome",
                    status="complete",
                    attempts=[_attempt("succeeded", position=1, rationale="ok")],
                    granularity="session",
                )
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="Judge feedback write incomplete: 1 failure"):
        run_judging_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    assert timeline == ["query", "barrier", "write:judge.session_outcome"]
    result = store.get(run.run_id).judging_result
    assert result["scores_written"] == 0
    assert result["write_failure_count"] == 1


def test_cancellation_after_a_rating_preserves_progress_and_writes_nothing(
    store,
    monkeypatch,
) -> None:
    turns = [_turn("turn-1", minute=0), _turn("turn-2", minute=2)]
    run, effective = _start(
        store,
        turns,
        ("judge.state_consistency", "judge.session_outcome"),
    )
    timeline: list[str] = []
    weave_client = TrackingWeaveClient(timeline)
    dependencies, _ = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    cancel = threading.Event()

    def judge_then_cancel(*_args, **_kwargs):
        cancel.set()
        return [
            _score(
                "judge.state_consistency",
                status="complete",
                attempts=[_attempt("succeeded", position=1, rationale="ok")],
                granularity="turn",
            )
        ]

    monkeypatch.setattr(judging_stage, "judge_turn", judge_then_cancel)
    session_judge = Mock(return_value=[])
    monkeypatch.setattr(judging_stage, "judge_session", session_judge)

    with pytest.raises(StageCancelled):
        run_judging_stage(run, effective, cancel, dependencies=dependencies)

    updated = store.get(run.run_id)
    assert updated.judging_result is None
    assert updated.judging_progress["rubrics_completed"] == 1
    assert updated.judging_progress["reviewer_attempts_completed"] == 1
    assert session_judge.call_count == 0
    assert timeline == []


def test_durable_cancellation_race_at_write_barrier_becomes_stage_cancelled(
    store,
    monkeypatch,
) -> None:
    turns = [_turn("turn-1", minute=0)]
    run, effective = _start(store, turns, ("judge.session_outcome",))
    timeline: list[str] = []
    weave_client = TrackingWeaveClient(timeline)
    dependencies, _ = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    original = store.external_write_barrier

    def cancel_then_barrier(run_id: str, status: RunStatus):
        store.cancel_if_safe(run_id)
        return original(run_id, status)

    monkeypatch.setattr(store, "external_write_barrier", cancel_then_barrier)
    monkeypatch.setattr(
        judging_stage,
        "judge_session",
        Mock(
            return_value=[
                _score(
                    "judge.session_outcome",
                    status="complete",
                    attempts=[_attempt("succeeded", position=1, rationale="ok")],
                    granularity="session",
                )
            ]
        ),
    )

    with pytest.raises(StageCancelled):
        run_judging_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    updated = store.get(run.run_id)
    assert updated.status is RunStatus.CANCELLED
    assert updated.judging_result is None
    assert [event for event, _ in weave_client.events] == ["query"]


def test_inference_cancellation_propagates_without_becoming_coverage_failure(
    store,
    monkeypatch,
) -> None:
    turns = [_turn("turn-1", minute=0), _turn("turn-2", minute=2)]
    run, effective = _start(store, turns, ("judge.state_consistency",))
    timeline: list[str] = []
    weave_client = TrackingWeaveClient(timeline)
    dependencies, _ = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    cancelled = InferenceCancelled("cancelled")

    def stop_inference(*_args, **_kwargs):
        raise cancelled

    monkeypatch.setattr(judging_stage, "judge_turn", stop_inference)

    with pytest.raises(InferenceCancelled) as caught:
        run_judging_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    assert caught.value is cancelled
    updated = store.get(run.run_id)
    assert updated.judging_result is None
    assert updated.judging_progress["rubrics_completed"] == 0
    assert timeline == []


def test_programming_errors_propagate_instead_of_becoming_coverage_failures(
    store,
    monkeypatch,
) -> None:
    turns = [_turn("turn-1", minute=0), _turn("turn-2", minute=2)]
    run, effective = _start(store, turns, ("judge.state_consistency",))
    timeline: list[str] = []
    weave_client = TrackingWeaveClient(timeline)
    dependencies, _ = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )

    def programming_defect(*_args, **_kwargs):
        raise AssertionError("programming defect")

    monkeypatch.setattr(judging_stage, "judge_turn", programming_defect)

    with pytest.raises(AssertionError, match="programming defect"):
        run_judging_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    updated = store.get(run.run_id)
    assert updated.judging_result is None
    assert updated.judging_progress["rubrics_completed"] == 0
    assert timeline == []


def test_durable_run_snapshot_is_authoritative(store, monkeypatch) -> None:
    turns = [_turn("turn-1", minute=0)]
    run, effective = _start(store, turns, ("judge.session_outcome",))
    timeline: list[str] = []
    weave_client = TrackingWeaveClient(timeline)
    dependencies, hydrate = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    _track_barrier(store, weave_client, timeline, monkeypatch)
    monkeypatch.setattr(
        judging_stage,
        "judge_session",
        Mock(
            return_value=[
                _score(
                    "judge.session_outcome",
                    status="complete",
                    attempts=[_attempt("succeeded", position=1, rationale="ok")],
                    granularity="session",
                )
            ]
        ),
    )
    stale_caller_snapshot = replace(run, judging_plan=None, turn_cohort=None)

    run_judging_stage(
        stale_caller_snapshot,
        effective,
        threading.Event(),
        dependencies=dependencies,
    )

    hydrate.assert_called_once_with(run.turn_cohort)
    assert store.get(run.run_id).judging_result["coverage_complete"] is True


def test_persisted_attempt_summaries_are_bounded(store, monkeypatch) -> None:
    turns = [
        _turn("turn-1", minute=0),
        _turn("turn-2", minute=2, tools=True),
    ]
    run, effective = _start(
        store,
        turns,
        ("judge.tool_choice", "judge.state_consistency"),
    )
    timeline: list[str] = []
    weave_client = TrackingWeaveClient(timeline)
    dependencies, _ = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    _track_barrier(store, weave_client, timeline, monkeypatch)
    monkeypatch.setattr(judging_stage, "_MAX_PERSISTED_OUTCOMES", 1)
    noisy_usage = {f"token-{index}": "v" * 1_000 for index in range(50)}
    attempts = [_attempt("succeeded", position=1, rationale="ok")]
    attempts[0]["usage"] = noisy_usage
    monkeypatch.setattr(
        judging_stage,
        "judge_turn",
        Mock(
            return_value=[
                _score(
                    rubric,
                    status="complete",
                    attempts=attempts,
                    granularity="turn",
                )
                for rubric in ("judge.tool_choice", "judge.state_consistency")
            ]
        ),
    )

    run_judging_stage(
        run,
        effective,
        threading.Event(),
        dependencies=dependencies,
    )

    result = store.get(run.run_id).judging_result
    assert result["attempt_summary_count"] == 2
    assert result["attempt_summaries_truncated"] is True
    assert len(result["attempt_summaries"]) == 1
    usage = result["attempt_summaries"][0]["attempts"][0]["usage"]
    assert len(usage) <= 20
    assert all(len(str(value)) <= 500 for value in usage.values())


def test_stage_builds_and_pins_missing_plan_from_exact_hydration(
    store,
    monkeypatch,
) -> None:
    turns = [_turn("turn-1", minute=0)]
    run, effective = _start(
        store,
        turns,
        ("judge.session_outcome",),
        pin_plan=False,
    )
    assert run.judging_plan is None
    timeline: list[str] = []
    weave_client = TrackingWeaveClient(timeline)
    dependencies, hydrate = _dependencies(
        store,
        weave_client,
        TrackingChatClient(),
        turns,
    )
    _track_barrier(store, weave_client, timeline, monkeypatch)
    monkeypatch.setattr(
        judging_stage,
        "judge_session",
        Mock(
            return_value=[
                _score(
                    "judge.session_outcome",
                    status="complete",
                    attempts=[_attempt("succeeded", position=1, rationale="ok")],
                    granularity="session",
                )
            ]
        ),
    )

    run_judging_stage(
        run,
        effective,
        threading.Event(),
        dependencies=dependencies,
    )

    updated = store.get(run.run_id)
    assert updated.judging_plan is not None
    assert updated.judging_result["plan_id"] == updated.judging_plan["plan_id"]
    hydrate.assert_called_once_with(run.turn_cohort)
