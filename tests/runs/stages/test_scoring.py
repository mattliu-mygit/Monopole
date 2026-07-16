from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.judges.families import model_family
from weave_agent_signals.models import Score, SessionView, ToolSpan, TurnSpan
from weave_agent_signals.run_config import RunConfig, resolve_run_config
from weave_agent_signals.runs.stages import StageCancelled
from weave_agent_signals.runs.stages import scoring as scoring_stage
from weave_agent_signals.runs.stages.scoring import (
    ScoringDependencies,
    run_scoring_stage,
)
from weave_agent_signals.runs.store import DataSelection, RunStatus, RunStore


@pytest.fixture
def store(tmp_path):
    instance = RunStore(tmp_path / "runs.db")
    yield instance
    instance.close()


def _turn(
    trace_id: str,
    conversation_id: str = "conversation-with-a-long-id",
    *,
    minute: int = 0,
    user_input: str | None = None,
    tools: int = 0,
) -> TurnSpan:
    started_at = datetime(2026, 7, 14, 12, minute, tzinfo=timezone.utc)
    tool_calls = [
        ToolSpan(
            span_id=f"tool-{index}",
            tool_name=f"tool-{index:02d}",
            arguments="{}",
            result="ok",
            status_code="SUCCESS",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=1),
        )
        for index in range(tools)
    ]
    return TurnSpan(
        trace_id=trace_id,
        conversation_id=conversation_id,
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
        steering_count=1,
        denial_count=2,
        tool_error_count=3,
        events=[],
        tool_calls=tool_calls,
        chat_spans=[],
        subagents=[],
        user_input=user_input,
        assistant_output="done",
    )


def _score(name: str, value: float = 0.75, *, granularity: str = "turn") -> Score:
    return Score(
        scorer=name,
        value=value,
        tags=[],
        metadata={},
        granularity=granularity,
    )


def _sessions(turns: list[TurnSpan]) -> dict[str, SessionView]:
    grouped: dict[str, list[TurnSpan]] = {}
    for turn in turns:
        grouped.setdefault(turn.conversation_id, []).append(turn)
    return {
        conversation_id: SessionView(
            conversation_id=conversation_id,
            turns=session_turns,
            config_version=session_turns[0].config_version,
            git_branch=session_turns[0].git_branch,
        )
        for conversation_id, session_turns in grouped.items()
    }


def _cohort(turns: list[TurnSpan]) -> dict:
    sessions = _sessions(turns)
    return {
        "schema_version": 1,
        "pinned_at": "2026-07-14T12:30:00+00:00",
        "cohort_id": "sha256:test-cohort",
        "turn_count": len(turns),
        "session_count": len(sessions),
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
                "conversation_id": conversation_id,
                "weave_ref": session.ref_for(),
                "turn_count": len(session.turns),
            }
            for conversation_id, session in sessions.items()
        ],
    }


def _start(store: RunStore, turns: list[TurnSpan], *, force: bool = False):
    models = build_model_catalog(which=lambda name: f"/bin/{name}")
    rubrics = build_rubric_catalog()
    requested = RunConfig(
        model_catalog_version=models.catalog_version,
        rubric_catalog_version=rubrics.catalog_version,
        judge_backend="cli",
        judge_models=("claude-sonnet-5", "gpt-5.6-sol"),
        proposal_model="gpt-5.6-sol",
        proposal_evaluator_model="claude-sonnet-5",
        rubrics=(rubrics.rubrics[0].id,),
        candidate_budget=3,
        force=force,
    )
    effective = resolve_run_config(
        requested,
        model_catalog=models,
        rubric_catalog=rubrics,
    )
    selection = DataSelection(
        session_ids=tuple(dict.fromkeys(turn.conversation_id for turn in turns))
    )
    created = store.create()
    store.save_selection(created.run_id, selection)
    store.save_config(created.run_id, requested)
    run = store.start(
        created.run_id,
        expected_selection=selection,
        expected_config=requested,
        turn_cohort=_cohort(turns),
        effective_config=effective,
    )
    return run, effective


class TrackingClient:
    entity = "weave-team"
    project = "agent-sessions"

    def __init__(
        self,
        existing: dict[tuple[str, str], list[dict]] | None = None,
        *,
        write_error: Exception | None = None,
        delete_error: Exception | None = None,
    ):
        self.existing = existing or {}
        self.write_error = write_error
        self.delete_error = delete_error
        self.events: list[tuple[str, object]] = []
        self.inside_barrier = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def query_existing_feedback_batch(self, refs: list[str]):
        self.events.append(("query", tuple(refs)))
        return self.existing

    def delete_feedback_ids(self, feedback: list[dict]) -> None:
        assert self.inside_barrier
        self.events.append(("delete", tuple(item["id"] for item in feedback)))
        if self.delete_error is not None:
            raise self.delete_error

    def write_score(self, score: Score, ref: str) -> dict:
        assert self.inside_barrier
        self.events.append(("write", (score, ref)))
        if self.write_error is not None:
            raise self.write_error
        return {"id": f"written-{score.scorer}"}


def _dependencies(
    store: RunStore,
    client: TrackingClient,
    turns: list[TurnSpan],
) -> tuple[ScoringDependencies, Mock]:
    hydrate = Mock(return_value=(turns, _sessions(turns)))
    dependencies = ScoringDependencies(
        store=store,
        client_factory=lambda: client,
        hydrate_cohort=hydrate,
    )
    return dependencies, hydrate


def _track_barrier(store: RunStore, client: TrackingClient, monkeypatch) -> None:
    original = store.external_write_barrier

    @contextmanager
    def tracked(run_id: str, status: RunStatus) -> Iterator[None]:
        with original(run_id, status):
            client.inside_barrier = True
            try:
                yield
            finally:
                client.inside_barrier = False

    monkeypatch.setattr(store, "external_write_barrier", tracked)


def test_missing_feedback_is_written_and_persists_bounded_progress(store, monkeypatch):
    turn = _turn(
        "turn-with-a-long-trace-id",
        user_input="x" * 140,
        tools=10,
    )
    run, effective = _start(store, [turn])
    client = TrackingClient()
    dependencies, hydrate = _dependencies(store, client, [turn])
    _track_barrier(store, client, monkeypatch)
    monkeypatch.setattr(
        scoring_stage,
        "score_turn",
        lambda _turn: [_score("outcome.test")],
    )
    monkeypatch.setattr(
        scoring_stage,
        "score_session",
        lambda _session: [_score("efficiency.session", granularity="session")],
    )

    run_scoring_stage(
        run,
        effective,
        threading.Event(),
        dependencies=dependencies,
    )

    updated = store.get(run.run_id)
    assert updated is not None
    result = updated.scoring_result
    assert result == {
        "turns_scored": 1,
        "sessions_scored": 1,
        "scores_written": 2,
        "errors": 0,
        "turn_details": result["turn_details"],
    }
    detail = result["turn_details"][0]
    assert detail == {
        "trace_id": "turn-with-a-",
        "conversation_id": "conversation",
        "model": "gpt-5.6-sol",
        "user_input": "x" * 120 + "...",
        "tokens": 18,
        "tool_count": 10,
        "tools_used": [f"tool-{index:02d}" for index in range(8)],
        "errors": 3,
        "steering": 1,
        "denials": 2,
        "scores": {"outcome.test": 0.75},
    }
    assert updated.scoring_progress["scored"] == 1
    assert updated.scoring_progress["written"] == 2
    assert updated.scoring_progress["turn_details"] == result["turn_details"]
    hydrate.assert_called_once_with(run.turn_cohort)
    queried_refs = next(value for event, value in client.events if event == "query")
    assert queried_refs == (turn.ref_for(), _sessions([turn])[turn.conversation_id].ref_for())
    writes = [value for event, value in client.events if event == "write"]
    assert len(writes) == 2
    assert all(score.metadata["config_version"] == "config-v1" for score, _ in writes)


def test_force_creates_before_purging_all_prior_feedback(store, monkeypatch):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn], force=True)
    turn_type = "weave_agent_signals.outcome.test"
    client = TrackingClient(
        {
            (turn.ref_for(), turn_type): [
                {"id": "old-turn"},
                {"id": "duplicate-turn"},
            ],
        }
    )
    dependencies, _ = _dependencies(store, client, [turn])
    _track_barrier(store, client, monkeypatch)
    monkeypatch.setattr(
        scoring_stage,
        "score_turn",
        lambda _turn: [_score("outcome.test")],
    )
    monkeypatch.setattr(
        scoring_stage,
        "score_session",
        lambda _session: [],
    )

    run_scoring_stage(
        run,
        effective,
        threading.Event(),
        dependencies=dependencies,
    )

    mutations = [(event, value) for event, value in client.events if event != "query"]
    assert [event for event, _ in mutations] == ["write", "delete"]
    _, ref = mutations[0][1]
    assert ref == turn.ref_for()
    assert mutations[1] == ("delete", ("old-turn", "duplicate-turn"))
    assert store.get(run.run_id).scoring_result["scores_written"] == 1


def test_force_create_failure_preserves_all_prior_feedback(store, monkeypatch):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn], force=True)
    client = TrackingClient(
        {
            (turn.ref_for(), "weave_agent_signals.outcome.test"): [
                {"id": "old-turn"},
                {"id": "duplicate-turn"},
            ]
        },
        write_error=RuntimeError("create failed"),
    )
    dependencies, _ = _dependencies(store, client, [turn])
    _track_barrier(store, client, monkeypatch)
    monkeypatch.setattr(scoring_stage, "score_turn", lambda _turn: [_score("outcome.test")])
    monkeypatch.setattr(scoring_stage, "score_session", lambda _session: [])

    with pytest.raises(RuntimeError, match="Scoring completed with 1 error"):
        run_scoring_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    assert [event for event, _ in client.events] == ["query", "write"]
    result = store.get(run.run_id).scoring_result
    assert result["scores_written"] == 0
    assert result["errors"] == 1


def test_prior_cleanup_failure_reports_partial_success(store, monkeypatch):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn], force=True)
    client = TrackingClient(
        {
            (turn.ref_for(), "weave_agent_signals.outcome.test"): [
                {"id": "old-turn"},
                {"id": "duplicate-turn"},
            ]
        },
        delete_error=RuntimeError("cleanup failed"),
    )
    dependencies, _ = _dependencies(store, client, [turn])
    _track_barrier(store, client, monkeypatch)
    monkeypatch.setattr(scoring_stage, "score_turn", lambda _turn: [_score("outcome.test")])
    monkeypatch.setattr(scoring_stage, "score_session", lambda _session: [])

    with pytest.raises(RuntimeError, match="Scoring completed with 1 error"):
        run_scoring_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    assert [event for event, _ in client.events] == ["query", "write", "delete"]
    result = store.get(run.run_id).scoring_result
    assert result["scores_written"] == 1
    assert result["errors"] == 1


def test_existing_feedback_is_always_queried_and_skipped_without_force(store, monkeypatch):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn])
    session = _sessions([turn])[turn.conversation_id]
    client = TrackingClient(
        {
            (turn.ref_for(), "weave_agent_signals.outcome.test"): [{"id": "old-turn"}],
            (session.ref_for(), "weave_agent_signals.efficiency.session"): [{"id": "old-session"}],
        }
    )
    dependencies, _ = _dependencies(store, client, [turn])
    monkeypatch.setattr(
        scoring_stage,
        "score_turn",
        lambda _turn: [_score("outcome.test")],
    )
    monkeypatch.setattr(
        scoring_stage,
        "score_session",
        lambda _session: [_score("efficiency.session", granularity="session")],
    )

    run_scoring_stage(
        run,
        effective,
        threading.Event(),
        dependencies=dependencies,
    )

    assert [event for event, _ in client.events] == ["query"]
    assert store.get(run.run_id).scoring_result["scores_written"] == 0


def test_scoring_errors_persist_partial_result_then_raise(store, monkeypatch):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn])
    client = TrackingClient()
    dependencies, _ = _dependencies(store, client, [turn])
    monkeypatch.setattr(
        scoring_stage,
        "score_turn",
        Mock(side_effect=RuntimeError("scorer failed")),
    )
    monkeypatch.setattr(scoring_stage, "score_session", lambda _session: [])

    with pytest.raises(RuntimeError, match="Scoring completed with 1 error"):
        run_scoring_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    updated = store.get(run.run_id)
    assert updated.scoring_result == {
        "turns_scored": 1,
        "sessions_scored": 1,
        "scores_written": 0,
        "errors": 1,
        "turn_details": [],
    }
    assert updated.scoring_progress["status_message"] == (
        "Scoring failed: 1 error; 0 scores written."
    )
    assert updated.scoring_succeeded is False


def test_cancellation_race_before_write_barrier_preserves_feedback_and_progress(store, monkeypatch):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn], force=True)
    client = TrackingClient(
        {(turn.ref_for(), "weave_agent_signals.outcome.test"): [{"id": "existing"}]}
    )
    dependencies, _ = _dependencies(store, client, [turn])
    original = store.external_write_barrier

    def cancel_then_barrier(run_id: str, status: RunStatus):
        store.cancel_if_safe(run_id)
        return original(run_id, status)

    monkeypatch.setattr(store, "external_write_barrier", cancel_then_barrier)
    monkeypatch.setattr(
        scoring_stage,
        "score_turn",
        lambda _turn: [_score("outcome.test")],
    )
    monkeypatch.setattr(scoring_stage, "score_session", lambda _session: [])

    with pytest.raises(StageCancelled):
        run_scoring_stage(
            run,
            effective,
            threading.Event(),
            dependencies=dependencies,
        )

    updated = store.get(run.run_id)
    assert updated.status is RunStatus.CANCELLED
    assert updated.scoring_result is None
    assert updated.scoring_progress["scored"] == 1
    assert updated.scoring_progress["written"] == 0
    assert [event for event, _ in client.events] == ["query"]


def test_cancellation_retains_progress_for_turns_already_scored(store, monkeypatch):
    turns = [
        _turn("turn-1", conversation_id="conversation-1"),
        _turn("turn-2", conversation_id="conversation-1", minute=1),
    ]
    run, effective = _start(store, turns)
    client = TrackingClient()
    dependencies, _ = _dependencies(store, client, turns)
    cancel = threading.Event()
    calls = 0

    def score_then_cancel(_turn):
        nonlocal calls
        calls += 1
        cancel.set()
        return [_score("outcome.test")]

    monkeypatch.setattr(scoring_stage, "score_turn", score_then_cancel)
    monkeypatch.setattr(scoring_stage, "score_session", lambda _session: [])

    with pytest.raises(StageCancelled):
        run_scoring_stage(run, effective, cancel, dependencies=dependencies)

    updated = store.get(run.run_id)
    assert calls == 1
    assert updated.status is RunStatus.SCORING
    assert updated.scoring_result is None
    assert updated.scoring_progress["scored"] == 1
    assert len(updated.scoring_progress["turn_details"]) == 1
    assert [event for event, _ in client.events] == ["query"]
