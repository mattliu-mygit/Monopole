from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from weave_agent_signals.judges.families import model_family
from weave_agent_signals.models import SessionView, TraceRole, TurnSpan
from weave_agent_signals.runs.cohort import discover_turn_cohort, hydrate_turn_cohort
from weave_agent_signals.runs.store import DataSelection


def _turn(
    trace_id: str,
    conversation_id: str,
    started_at: datetime,
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
        status_code="SUCCESS",
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
    )


class FakeClient:
    def __init__(
        self,
        *,
        discovered: list[TurnSpan] | None = None,
        hydrated: list[TurnSpan] | None = None,
        entity: str = "weave-team",
        project: str = "agent-sessions",
        hydrate_model: str | None = None,
    ):
        self.discovered = list(discovered or [])
        self.hydrated = list(hydrated if hydrated is not None else self.discovered)
        self.entity = entity
        self.project = project
        self.hydrate_model = hydrate_model
        self.paginated_calls: list[dict] = []
        self.trace_id_calls: list[list[str]] = []
        self.hydration_calls: list[list[str]] = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def query_turns_paginated(self, **kwargs):
        self.paginated_calls.append(kwargs)
        return list(self.discovered)

    def query_turns_by_trace_ids(self, trace_ids):
        self.trace_id_calls.append(list(trace_ids))
        return list(self.hydrated)

    def hydrate_turns_batch(self, turns):
        self.hydration_calls.append([turn.trace_id for turn in turns])
        if self.hydrate_model is not None:
            for turn in turns:
                turn.model = self.hydrate_model


def _selection(*session_ids: str, since: str | None = None, until: str | None = None):
    return DataSelection(
        since=since,
        until=until,
        timezone="America/Los_Angeles",
        session_ids=session_ids,
    )


def test_discover_turn_cohort_is_stable_and_pins_exact_hydrated_identity():
    started = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    first = _turn("turn-b", "session/2", started, model="claude-sonnet-5")
    second = _turn("turn-a", "session/1", started, model=None)
    client = FakeClient(
        discovered=[first, second],
        entity="custom-entity",
        project="custom-project",
    )

    def hydrate_models(turns):
        client.hydration_calls.append([turn.trace_id for turn in turns])
        next(turn for turn in turns if turn.trace_id == "turn-a").model = "gpt-5.6-sol"

    client.hydrate_turns_batch = hydrate_models
    cohort = discover_turn_cohort(
        _selection("session/1", "session/2"),
        client_factory=lambda: client,
        entity="custom-entity",
        project="custom-project",
    )

    assert cohort["schema_version"] == 1
    assert cohort["turn_count"] == 2
    assert cohort["session_count"] == 2
    assert [entry["trace_id"] for entry in cohort["turns"]] == ["turn-a", "turn-b"]
    assert cohort["turns"][0] == {
        "trace_id": "turn-a",
        "weave_ref": "weave:///custom-entity/custom-project/agent_turn/turn-a",
        "conversation_id": "session/1",
        "started_at": started.isoformat(),
        "model": "gpt-5.6-sol",
        "model_family": model_family("gpt-5.6-sol"),
    }
    assert cohort["sessions"] == [
        {
            "conversation_id": "session/1",
            "weave_ref": ("weave:///custom-entity/custom-project/agent_conversation/session%2F1"),
            "turn_count": 1,
        },
        {
            "conversation_id": "session/2",
            "weave_ref": ("weave:///custom-entity/custom-project/agent_conversation/session%2F2"),
            "turn_count": 1,
        },
    ]
    assert cohort["cohort_id"].startswith("sha256:")
    assert client.hydration_calls == [["turn-b", "turn-a"]]
    assert client.closed is True

    repeat_client = FakeClient(discovered=[second, first])
    repeat = discover_turn_cohort(
        _selection("session/1", "session/2"),
        client_factory=lambda: repeat_client,
        entity="custom-entity",
        project="custom-project",
    )
    assert repeat["cohort_id"] == cohort["cohort_id"]
    assert repeat["turns"] == cohort["turns"]
    assert repeat["sessions"] == cohort["sessions"]


def test_discover_turn_cohort_filters_exact_aware_dates_and_sessions():
    lower = datetime(2026, 7, 13, 7, tzinfo=timezone.utc)
    upper = datetime(2026, 7, 14, 6, 59, 59, 999999, tzinfo=timezone.utc)
    client = FakeClient(
        discovered=[
            _turn("before", "allowed", lower - timedelta(microseconds=1)),
            _turn("lower", "allowed", lower),
            _turn("other-session", "other", lower + timedelta(hours=1)),
            _turn("upper", "allowed", upper),
            _turn("after", "allowed", upper + timedelta(microseconds=1)),
        ]
    )

    cohort = discover_turn_cohort(
        _selection(
            "allowed",
            since="2026-07-13T00:00:00-07:00",
            until="2026-07-13T23:59:59.999999-07:00",
        ),
        client_factory=lambda: client,
        entity="weave-team",
        project="agent-sessions",
    )

    assert [entry["trace_id"] for entry in cohort["turns"]] == ["lower", "upper"]
    assert client.paginated_calls == [{"page_size": 500, "since": lower}]


def test_discover_turn_cohort_requires_every_selected_session():
    started = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    client = FakeClient(discovered=[_turn("turn-1", "present", started)])

    with pytest.raises(ValueError, match="missing selected sessions: absent"):
        discover_turn_cohort(
            _selection("present", "absent"),
            client_factory=lambda: client,
            entity="weave-team",
            project="agent-sessions",
        )

    assert client.hydration_calls == []


@pytest.mark.parametrize(
    "turns",
    [
        [
            _turn(
                "signal",
                "selected",
                datetime(2026, 7, 13, 12, tzinfo=timezone.utc),
                trace_role=TraceRole.SIGNAL_EVALUATION,
            )
        ],
        [
            _turn("agent", "selected", datetime(2026, 7, 13, 12, tzinfo=timezone.utc)),
            _turn(
                "judge",
                "selected",
                datetime(2026, 7, 13, 13, tzinfo=timezone.utc),
                trace_role=TraceRole.JUDGE_EVALUATION,
            ),
        ],
    ],
)
def test_discover_turn_cohort_rejects_non_agent_or_mixed_sessions_before_hydration(turns):
    client = FakeClient(discovered=turns)

    with pytest.raises(ValueError, match="not evaluable"):
        discover_turn_cohort(
            _selection("selected"),
            client_factory=lambda: client,
            entity="weave-team",
            project="agent-sessions",
        )

    assert client.hydration_calls == []


def test_discover_turn_cohort_uses_unbounded_paginated_query():
    started = datetime(2026, 7, 1, tzinfo=timezone.utc)
    turns = [
        _turn(f"turn-{index:04d}", "large-session", started + timedelta(seconds=index))
        for index in range(1_201)
    ]
    client = FakeClient(discovered=list(reversed(turns)))

    cohort = discover_turn_cohort(
        _selection("large-session"),
        client_factory=lambda: client,
        entity="weave-team",
        project="agent-sessions",
    )

    assert cohort["turn_count"] == 1_201
    assert client.paginated_calls == [{"page_size": 500, "since": None}]
    assert client.hydration_calls == [[turn.trace_id for turn in reversed(turns)]]


def test_hydrate_turn_cohort_restores_pinned_order_and_session_views():
    started = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    first = _turn("turn-1", "session-1", started)
    second = _turn("turn-2", "session-1", started + timedelta(minutes=1))
    discovery = FakeClient(discovered=[first, second])
    cohort = discover_turn_cohort(
        _selection("session-1"),
        client_factory=lambda: discovery,
        entity="weave-team",
        project="agent-sessions",
    )
    hydration = FakeClient(hydrated=[second, first])

    turns, sessions = hydrate_turn_cohort(
        cohort,
        client_factory=lambda: hydration,
    )

    assert [turn.trace_id for turn in turns] == ["turn-1", "turn-2"]
    assert hydration.trace_id_calls == [["turn-1", "turn-2"]]
    assert hydration.hydration_calls == [["turn-1", "turn-2"]]
    assert list(sessions) == ["session-1"]
    assert isinstance(sessions["session-1"], SessionView)
    assert sessions["session-1"].turns == turns
    assert sessions["session-1"].config_version == "config-v1"
    assert sessions["session-1"].git_branch == "main"


def test_hydrate_turn_cohort_fails_closed_for_missing_or_duplicate_trace():
    started = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    pinned = _turn("turn-1", "session-1", started)
    discovery = FakeClient(discovered=[pinned])
    cohort = discover_turn_cohort(
        _selection("session-1"),
        client_factory=lambda: discovery,
        entity="weave-team",
        project="agent-sessions",
    )

    missing = FakeClient(hydrated=[])
    with pytest.raises(RuntimeError, match="missing trace IDs: turn-1"):
        hydrate_turn_cohort(cohort, client_factory=lambda: missing)

    duplicate = FakeClient(hydrated=[pinned, replace(pinned)])
    with pytest.raises(RuntimeError, match="duplicate trace IDs: turn-1"):
        hydrate_turn_cohort(cohort, client_factory=lambda: duplicate)


def test_hydrate_turn_cohort_rejects_role_drift_before_child_hydration():
    started = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    original = _turn("turn-1", "session-1", started)
    cohort = discover_turn_cohort(
        _selection("session-1"),
        client_factory=lambda: FakeClient(discovered=[original]),
        entity="weave-team",
        project="agent-sessions",
    )
    changed = _turn(
        "turn-1",
        "session-1",
        started,
        trace_role=TraceRole.REFLECTION_EVALUATION,
    )
    client = FakeClient(hydrated=[changed])

    with pytest.raises(RuntimeError, match="not evaluable"):
        hydrate_turn_cohort(cohort, client_factory=lambda: client)

    assert client.hydration_calls == []


@pytest.mark.parametrize("changed_field", ["conversation", "started_at", "model"])
def test_hydrate_turn_cohort_fails_when_live_metadata_or_model_identity_changes(
    changed_field,
):
    started = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    pinned = _turn("turn-1", "session-1", started)
    discovery = FakeClient(discovered=[pinned])
    cohort = discover_turn_cohort(
        _selection("session-1"),
        client_factory=lambda: discovery,
        entity="weave-team",
        project="agent-sessions",
    )
    changed = replace(pinned)
    if changed_field == "conversation":
        changed.conversation_id = "different-session"
    elif changed_field == "started_at":
        changed.started_at += timedelta(seconds=1)
    else:
        changed.model = "claude-sonnet-5"

    client = FakeClient(hydrated=[changed])
    with pytest.raises(RuntimeError, match="metadata changed for trace IDs: turn-1"):
        hydrate_turn_cohort(cohort, client_factory=lambda: client)


def test_hydrate_turn_cohort_checks_model_identity_after_child_hydration():
    started = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    pinned = _turn("turn-1", "session-1", started, model="gpt-5.6-sol")
    discovery = FakeClient(discovered=[pinned])
    cohort = discover_turn_cohort(
        _selection("session-1"),
        client_factory=lambda: discovery,
        entity="weave-team",
        project="agent-sessions",
    )
    fetched = replace(pinned)
    client = FakeClient(hydrated=[fetched], hydrate_model="claude-sonnet-5")

    with pytest.raises(RuntimeError, match="metadata changed for trace IDs: turn-1"):
        hydrate_turn_cohort(cohort, client_factory=lambda: client)


def test_hydrate_turn_cohort_rejects_missing_or_tampered_pinned_metadata_before_io():
    started = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)
    pinned = _turn("turn-1", "session-1", started)
    discovery = FakeClient(discovered=[pinned])
    cohort = discover_turn_cohort(
        _selection("session-1"),
        client_factory=lambda: discovery,
        entity="weave-team",
        project="agent-sessions",
    )
    tampered = deepcopy(cohort)
    tampered["turns"][0].pop("model_family")
    client = FakeClient(hydrated=[pinned])

    with pytest.raises(RuntimeError, match="invalid pinned turn cohort"):
        hydrate_turn_cohort(tampered, client_factory=lambda: client)
    assert client.trace_id_calls == []


def test_discover_turn_cohort_uses_plain_value_errors_for_invalid_selection():
    client = FakeClient(discovered=[])
    with pytest.raises(ValueError, match="timezone offset"):
        discover_turn_cohort(
            _selection("session-1", since="2026-07-13T00:00:00"),
            client_factory=lambda: client,
            entity="weave-team",
            project="agent-sessions",
        )
