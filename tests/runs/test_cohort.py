from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from weave_agent_signals.judges.families import model_family
from weave_agent_signals.models import SessionView, TraceRole, TurnSpan
from weave_agent_signals.runs.cohort import discover_turn_cohort, hydrate_turn_cohort
from weave_agent_signals.runs.store import DataSelection

STARTED = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)


def _turn(
    trace_id: str,
    conversation_id: str,
    started_at: datetime = STARTED,
    *,
    model: str | None = "gpt-5.6-sol",
    trace_role: TraceRole = TraceRole.AGENT_SESSION,
) -> TurnSpan:
    return TurnSpan(
        trace_id=trace_id,
        conversation_id=conversation_id,
        started_at=started_at,
        ended_at=started_at + timedelta(minutes=1),
        model=model,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        status_code="OK",
        config_version="config-v1",
        git_branch="main",
        effort_level="medium",
        session_id=conversation_id,
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
        trace_role=trace_role,
        case_ids=[trace_id, f"{trace_id}/reply"],
    )


class FakeCes:
    """A CesSession stand-in whose client serves sessions by session Trajectory ID."""

    def __init__(self, *turns: TurnSpan):
        self.turns = list(turns)
        self.reads: list[str] = []
        self.closed = False
        self.client = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def read_session(self, trajectory_id: str) -> SessionView:
        self.reads.append(trajectory_id)
        turns = [turn for turn in self.turns if turn.conversation_id == trajectory_id]
        if not turns:
            request = httpx.Request("GET", "http://ces.test")
            raise httpx.HTTPStatusError("missing", request=request, response=httpx.Response(404))
        return SessionView(trajectory_id, turns, "config-v1", "main")


def _selection(*session_ids: str, since: str | None = None, until: str | None = None):
    return DataSelection(
        since=since,
        until=until,
        timezone="America/Los_Angeles",
        session_ids=session_ids,
    )


def _discover(ces: FakeCes, *session_ids: str, **bounds) -> dict:
    return discover_turn_cohort(_selection(*session_ids, **bounds), ces_factory=lambda: ces)


def test_discover_turn_cohort_is_stable_and_pins_case_ids():
    first = _turn("turn-b", "session-2", model="claude-sonnet-5")
    second = _turn("turn-a", "session-1")
    ces = FakeCes(first, second)

    cohort = _discover(ces, "session-1", "session-2")

    assert (cohort["schema_version"], cohort["turn_count"], cohort["session_count"]) == (3, 2, 2)
    assert [entry["trace_id"] for entry in cohort["turns"]] == ["turn-a", "turn-b"]
    assert cohort["turns"][0] == {
        "trace_id": "turn-a",
        "case_ids": ["turn-a", "turn-a/reply"],
        "conversation_id": "session-1",
        "started_at": STARTED.isoformat(),
        "model": "gpt-5.6-sol",
        "model_family": model_family("gpt-5.6-sol"),
        "effort_level": "medium",
    }
    assert cohort["sessions"] == [
        {"conversation_id": "session-1", "turn_count": 1},
        {"conversation_id": "session-2", "turn_count": 1},
    ]
    assert cohort["cohort_id"].startswith("sha256:")
    assert ces.closed is True

    repeat = _discover(FakeCes(second, first), "session-1", "session-2")
    assert (repeat["cohort_id"], repeat["turns"]) == (cohort["cohort_id"], cohort["turns"])


def test_discover_turn_cohort_reads_each_selected_session_in_full_and_filters_dates():
    early = _turn("early", "session-1", STARTED - timedelta(days=2))
    inside = _turn("inside", "session-1")
    late = _turn("late", "session-1", STARTED + timedelta(days=2))
    ces = FakeCes(early, inside, late)

    cohort = _discover(
        ces,
        "session-1",
        since=(STARTED - timedelta(hours=1)).isoformat(),
        until=(STARTED + timedelta(hours=1)).isoformat(),
    )

    assert [entry["trace_id"] for entry in cohort["turns"]] == ["inside"]
    assert ces.reads == ["session-1"]


@pytest.mark.parametrize("selected", [("session-1", "session-gone"), ("session-1", "session-late")])
def test_discover_turn_cohort_requires_every_selected_session(selected):
    ces = FakeCes(
        _turn("turn-1", "session-1"), _turn("late", "session-late", STARTED + timedelta(days=3))
    )

    with pytest.raises(ValueError, match=f"missing selected sessions: {selected[1]}"):
        _discover(ces, *selected, until=(STARTED + timedelta(hours=1)).isoformat())


@pytest.mark.parametrize(
    "turns",
    [
        [_turn("eval-1", "session-1", trace_role=TraceRole.SIGNAL_EVALUATION)],
        [
            _turn("agent-1", "session-1"),
            _turn("eval-1", "session-1", trace_role=TraceRole.SIGNAL_EVALUATION),
        ],
    ],
)
def test_discover_turn_cohort_rejects_non_agent_or_mixed_sessions(turns):
    with pytest.raises(ValueError, match="selected sessions are not evaluable: session-1"):
        _discover(FakeCes(*turns), "session-1")


def test_discover_turn_cohort_checks_roles_outside_the_selected_dates():
    agent = _turn("agent-1", "session-1")
    evaluator = _turn(
        "eval-1",
        "session-1",
        STARTED - timedelta(days=5),
        trace_role=TraceRole.SIGNAL_EVALUATION,
    )

    with pytest.raises(ValueError, match="not evaluable"):
        _discover(FakeCes(agent, evaluator), "session-1", since=STARTED.isoformat())


def test_hydrate_turn_cohort_restores_pinned_order_and_ignores_later_turns():
    first = _turn("turn-1", "session-1")
    second = _turn("turn-2", "session-2", STARTED + timedelta(minutes=5))
    cohort = _discover(FakeCes(first, second), "session-1", "session-2")
    added = _turn("turn-3", "session-1", STARTED + timedelta(hours=1))

    turns, sessions = hydrate_turn_cohort(cohort, ces_factory=lambda: FakeCes(added, second, first))

    assert [turn.trace_id for turn in turns] == ["turn-1", "turn-2"]
    assert {sid: [t.trace_id for t in s.turns] for sid, s in sessions.items()} == {
        "session-1": ["turn-1"],
        "session-2": ["turn-2"],
    }


def test_hydrate_turn_cohort_fails_closed_for_a_missing_session_or_turn():
    first = _turn("turn-1", "session-1")
    second = _turn("turn-2", "session-1", STARTED + timedelta(minutes=5))
    other = _turn("turn-3", "session-2")
    cohort = _discover(FakeCes(first, second, other), "session-1", "session-2")

    with pytest.raises(RuntimeError, match="sessions are missing: session-2"):
        hydrate_turn_cohort(cohort, ces_factory=lambda: FakeCes(first, second))
    with pytest.raises(RuntimeError, match="missing trace IDs: turn-2"):
        hydrate_turn_cohort(cohort, ces_factory=lambda: FakeCes(first, other))


def test_hydrate_turn_cohort_rejects_sessions_that_stopped_being_evaluable():
    pinned = _turn("turn-1", "session-1")
    cohort = _discover(FakeCes(pinned), "session-1")
    drifted = replace(pinned, trace_role=TraceRole.SIGNAL_EVALUATION)

    with pytest.raises(RuntimeError, match="not evaluable: session-1"):
        hydrate_turn_cohort(cohort, ces_factory=lambda: FakeCes(drifted))


@pytest.mark.parametrize(
    "change",
    [
        {"model": "claude-sonnet-5"},
        {"effort_level": "high"},
        {"started_at": STARTED + timedelta(seconds=1)},
        {"case_ids": ["turn-1", "turn-1/other"]},
    ],
)
def test_hydrate_turn_cohort_fails_when_pinned_metadata_or_case_ids_change(change):
    pinned = _turn("turn-1", "session-1")
    cohort = _discover(FakeCes(pinned), "session-1")

    with pytest.raises(RuntimeError, match="metadata changed for trace IDs: turn-1"):
        hydrate_turn_cohort(cohort, ces_factory=lambda: FakeCes(replace(pinned, **change)))


@pytest.mark.parametrize("field", ["model_family", "case_ids"])
def test_hydrate_turn_cohort_rejects_missing_or_tampered_pinned_metadata_before_io(field):
    pinned = _turn("turn-1", "session-1")
    tampered = deepcopy(_discover(FakeCes(pinned), "session-1"))
    tampered["turns"][0].pop(field)
    ces = FakeCes(pinned)

    with pytest.raises(RuntimeError, match="invalid pinned turn cohort"):
        hydrate_turn_cohort(tampered, ces_factory=lambda: ces)
    assert ces.reads == []


def test_discover_turn_cohort_uses_plain_value_errors_for_invalid_selection():
    ces = FakeCes()

    with pytest.raises(ValueError, match="timezone offset"):
        _discover(ces, "session-1", since="2026-07-13T00:00:00")
    assert ces.reads == []
