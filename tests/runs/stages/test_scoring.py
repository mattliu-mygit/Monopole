from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.ces_catalog import DETAILS_SUFFIX, DETERMINISTIC, Catalog, ScorerRef
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
        "schema_version": 3,
        "pinned_at": "2026-07-14T12:30:00+00:00",
        "cohort_id": "sha256:test-cohort",
        "turn_count": len(turns),
        "session_count": len(sessions),
        "turns": [
            {
                "trace_id": turn.trace_id,
                "case_ids": [turn.trace_id],
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
        judge_models=("claude:claude-sonnet-5", "codex:gpt-5.6-sol"),
        challenge_judge_models=("codex:gpt-5.6-sol",),
        proposal_model="codex:gpt-5.6-sol",
        proposal_evaluator_model="claude:claude-sonnet-5",
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


CATALOG = Catalog(
    "ev-turns",
    "ev-sessions",
    {
        s.name: ScorerRef(s, f"sc-{s.name}", "v", f"sc-{s.name}{DETAILS_SUFFIX}", "v")
        for s in DETERMINISTIC
    },
)


class TrackingCes:
    """A CesSession stand-in that records reads and writes in order."""

    def __init__(
        self,
        existing: dict[tuple[str, str], list[str]] | None = None,
        *,
        write_error: Exception | None = None,
    ):
        self.existing = existing or {}
        self.write_error = write_error
        self.events: list[tuple[str, object]] = []
        self.inside_barrier = False
        self.client = self
        self.catalog = CATALOG

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def _rows(self, kind: str, target: str) -> list[dict]:
        return [{"scorer_id": f"sc-{name}"} for name in self.existing.get((kind, target), [])]

    def turn_scores(self, catalog, trajectory_id):
        self.events.append(("query", ("case", trajectory_id)))
        return {
            target: self._rows(kind, target) for kind, target in self.existing if kind == "case"
        }

    def session_scores(self, catalog, trajectory_ids, *, subagents=False):
        self.events.append(("query", ("trajectory", tuple(trajectory_ids))))
        return {sid: self._rows("trajectory", sid) for sid in trajectory_ids}

    def write_scores(self, catalog, pass_id, kind, target_id, scores):
        assert self.inside_barrier
        self.events.append(("write", (pass_id, kind, target_id, list(scores))))
        if self.write_error is not None:
            raise self.write_error
        return [f"id-{score.scorer}" for score in scores]


def _dependencies(
    store: RunStore,
    ces: TrackingCes,
    turns: list[TurnSpan],
) -> tuple[ScoringDependencies, Mock]:
    hydrate = Mock(return_value=(turns, _sessions(turns)))
    dependencies = ScoringDependencies(
        store=store,
        ces_factory=lambda: ces,
        hydrate_cohort=hydrate,
    )
    return dependencies, hydrate


def _track_barrier(store: RunStore, ces: TrackingCes, monkeypatch) -> None:
    original = store.external_write_barrier

    @contextmanager
    def tracked(run_id: str, status: RunStatus) -> Iterator[None]:
        with original(run_id, status):
            ces.inside_barrier = True
            try:
                yield
            finally:
                ces.inside_barrier = False

    monkeypatch.setattr(store, "external_write_barrier", tracked)


def _events(ces: TrackingCes) -> list[str]:
    return [event for event, _ in ces.events]


def test_missing_scores_are_written_and_persist_bounded_progress(store, monkeypatch):
    turn = _turn(
        "turn-with-a-long-trace-id",
        user_input="x" * 140,
        tools=10,
    )
    run, effective = _start(store, [turn])
    ces = TrackingCes()
    dependencies, hydrate = _dependencies(store, ces, [turn])
    _track_barrier(store, ces, monkeypatch)
    monkeypatch.setattr(scoring_stage, "score_turn", lambda _turn: [_score("outcome.test")])
    monkeypatch.setattr(
        scoring_stage,
        "score_session",
        lambda _session: [_score("efficiency.session", granularity="session")],
    )

    run_scoring_stage(run, effective, threading.Event(), dependencies=dependencies)

    updated = store.get(run.run_id)
    assert updated is not None
    result = updated.scoring_result
    assert result == {
        "turns_scored": 1,
        "sessions_scored": 1,
        "scores_written": 2,
        "errors": 0,
        "turn_details": result["turn_details"],
        "status_message": "Scoring complete: 2 scores written.",
    }
    assert result["turn_details"][0] == {
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
    assert [value for event, value in ces.events if event == "query"] == [
        ("case", turn.conversation_id),
        ("trajectory", (turn.conversation_id,)),
    ]
    writes = sorted((value for event, value in ces.events if event == "write"), key=lambda w: w[1])
    assert [(pass_id, kind, target) for pass_id, kind, target, _ in writes] == [
        (run.run_id, "case", turn.trace_id),
        (run.run_id, "trajectory", turn.conversation_id),
    ]
    assert all(s.metadata["config_version"] == "config-v1" for *_, scores in writes for s in scores)


def test_force_writes_newer_scores_without_reading_existing_ones(store, monkeypatch):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn], force=True)
    ces = TrackingCes({("case", turn.trace_id): ["outcome.test"]})
    dependencies, _ = _dependencies(store, ces, [turn])
    _track_barrier(store, ces, monkeypatch)
    monkeypatch.setattr(scoring_stage, "score_turn", lambda _turn: [_score("outcome.test")])
    monkeypatch.setattr(scoring_stage, "score_session", lambda _session: [])

    run_scoring_stage(run, effective, threading.Event(), dependencies=dependencies)

    assert _events(ces) == ["write"]
    pass_id, kind, target, scores = ces.events[0][1]
    assert (pass_id, kind, target, [s.scorer for s in scores]) == (
        run.run_id,
        "case",
        turn.trace_id,
        ["outcome.test"],
    )
    assert store.get(run.run_id).scoring_result["scores_written"] == 1


def test_a_failed_write_counts_each_of_its_scores_as_an_error(store, monkeypatch):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn], force=True)
    ces = TrackingCes(write_error=RuntimeError("create failed"))
    dependencies, _ = _dependencies(store, ces, [turn])
    _track_barrier(store, ces, monkeypatch)
    monkeypatch.setattr(scoring_stage, "score_turn", lambda _turn: [_score("outcome.test")])
    monkeypatch.setattr(scoring_stage, "score_session", lambda _session: [])

    with pytest.raises(RuntimeError, match="Scoring completed with 1 error"):
        run_scoring_stage(run, effective, threading.Event(), dependencies=dependencies)

    assert _events(ces) == ["write"]
    result = store.get(run.run_id).scoring_result
    assert (result["scores_written"], result["errors"]) == (0, 1)


def test_existing_scores_are_read_and_skipped_without_force(store, monkeypatch):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn])
    ces = TrackingCes(
        {
            ("case", turn.trace_id): ["outcome.test"],
            ("trajectory", turn.conversation_id): ["efficiency.session"],
        }
    )
    dependencies, _ = _dependencies(store, ces, [turn])
    monkeypatch.setattr(scoring_stage, "score_turn", lambda _turn: [_score("outcome.test")])
    monkeypatch.setattr(
        scoring_stage,
        "score_session",
        lambda _session: [_score("efficiency.session", granularity="session")],
    )

    run_scoring_stage(run, effective, threading.Event(), dependencies=dependencies)

    assert _events(ces) == ["query", "query"]
    assert store.get(run.run_id).scoring_result["scores_written"] == 0


def test_scoring_errors_persist_partial_result_then_raise(store, monkeypatch):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn])
    ces = TrackingCes()
    dependencies, _ = _dependencies(store, ces, [turn])
    monkeypatch.setattr(
        scoring_stage,
        "score_turn",
        Mock(side_effect=RuntimeError("scorer failed")),
    )
    monkeypatch.setattr(scoring_stage, "score_session", lambda _session: [])

    with pytest.raises(RuntimeError, match="Scoring completed with 1 error"):
        run_scoring_stage(run, effective, threading.Event(), dependencies=dependencies)

    updated = store.get(run.run_id)
    assert updated.scoring_result == {
        "turns_scored": 1,
        "sessions_scored": 1,
        "scores_written": 0,
        "errors": 1,
        "turn_details": [],
        "status_message": "Scoring failed: 1 error; 0 scores written.",
    }
    assert updated.scoring_progress["status_message"] == (
        "Scoring failed: 1 error; 0 scores written."
    )
    assert updated.scoring_succeeded is False


def test_cancellation_race_before_write_barrier_writes_nothing_and_keeps_progress(
    store, monkeypatch
):
    turn = _turn("turn-1", conversation_id="conversation-1")
    run, effective = _start(store, [turn], force=True)
    ces = TrackingCes()
    dependencies, _ = _dependencies(store, ces, [turn])
    original = store.external_write_barrier

    def cancel_then_barrier(run_id: str, status: RunStatus):
        store.cancel_if_safe(run_id)
        return original(run_id, status)

    monkeypatch.setattr(store, "external_write_barrier", cancel_then_barrier)
    monkeypatch.setattr(scoring_stage, "score_turn", lambda _turn: [_score("outcome.test")])
    monkeypatch.setattr(scoring_stage, "score_session", lambda _session: [])

    with pytest.raises(StageCancelled):
        run_scoring_stage(run, effective, threading.Event(), dependencies=dependencies)

    updated = store.get(run.run_id)
    assert updated.status is RunStatus.CANCELLED
    assert updated.scoring_result is None
    assert updated.scoring_progress["scored"] == 1
    assert updated.scoring_progress["written"] == 0
    assert _events(ces) == []


def test_cancellation_retains_progress_for_turns_already_scored(store, monkeypatch):
    turns = [
        _turn("turn-1", conversation_id="conversation-1"),
        _turn("turn-2", conversation_id="conversation-1", minute=1),
    ]
    run, effective = _start(store, turns)
    ces = TrackingCes()
    dependencies, _ = _dependencies(store, ces, turns)
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
    assert "write" not in _events(ces)
