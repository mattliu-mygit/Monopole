from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from weave_signal_monitoring.hydration import HydrationError, hydrate_conversations
from weave_signal_monitoring.models import FeedbackRecord, SignalIdentity, TurnRecord
from weave_signal_monitoring.weave_gateway import WeaveGateway, WeaveReadError

NOW = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)
IDENTITY = SignalIdentity(
    slug="user-frustration",
    version="v1",
    monitor_name="agent-signal-user-frustration-v1",
    scorer_ref=(
        "weave:///weave-team/agent-sessions/object/agent-signal-user-frustration-v1-scorer:digest"
    ),
)
QUALITY_IDENTITY = SignalIdentity(
    slug="low-quality-response",
    version="v1",
    monitor_name="agent-signal-low-quality-response-v1",
    scorer_ref=(
        "weave:///weave-team/agent-sessions/object/"
        "agent-signal-low-quality-response-v1-scorer:digest"
    ),
)


def feedback(
    id,
    turn,
    rating,
    created_minute,
    reason="visible evidence",
    identity=IDENTITY,
):
    return FeedbackRecord(
        id=id,
        weave_ref=f"weave:///weave-team/agent-sessions/call/{turn}",
        runnable_ref=identity.scorer_ref,
        created_at=NOW + timedelta(minutes=created_minute),
        output={"rating": rating, "reason": reason},
    )


def turn(id, conversation, minute, request=None, name=None, duration_seconds=30):
    return TurnRecord(
        turn_id=id,
        conversation_id=conversation,
        started_at=NOW + timedelta(minutes=minute),
        ended_at=NOW + timedelta(minutes=minute, seconds=duration_seconds),
        display_name=name,
        user_request=request,
    )


def hydrate(feedback_rows, triggering, all_turns, identities=(IDENTITY,)):
    return hydrate_conversations(
        identities=identities,
        feedback=feedback_rows,
        triggering_turns=triggering,
        conversation_turns=all_turns,
        entity="weave-team",
        project="agent-sessions",
    )


def test_hydration_filters_deduplicates_groups_orders_and_ranks():
    feedback_rows = [
        feedback("old", "a2", 0.5, 3, "old score"),
        feedback("new", "a2", 0.25, 4, "new score"),
        feedback("healthy", "a1", 0.75, 2),
        feedback("other", "b1", 0.5, 5),
    ]
    triggering = [turn("a2", "a", 2), turn("b1", "b", 4, name="Named chat")]
    all_turns = [
        turn("a1", "a", 0, request="Please fix the deployment failure in production"),
        *triggering,
    ]

    result = hydrate(feedback_rows, triggering, all_turns)

    assert [item.conversation_id for item in result] == ["a", "b"]
    assert result[0].lowest_rating == 0.25
    assert result[0].signals[0].reason == "new score"
    assert result[0].display_name == "Please fix the deployment failure in production"
    assert result[0].triggering_turn_ids == ["a2"]
    assert result[1].display_name == "Named chat"


def test_evidence_is_ordered_by_turn_time_not_feedback_time():
    rows = [
        feedback("late-feedback", "a1", 0.5, 8),
        feedback("early-feedback", "a2", 0.25, 3, identity=QUALITY_IDENTITY),
    ]
    turns = [turn("a1", "a", 1), turn("a2", "a", 2)]

    result = hydrate(
        rows,
        triggering=list(reversed(turns)),
        all_turns=turns,
        identities=(IDENTITY, QUALITY_IDENTITY),
    )

    assert [item.turn_id for item in result[0].signals] == ["a1", "a2"]
    assert result[0].triggering_turn_ids == ["a1", "a2"]


def test_equal_scores_rank_most_recent_conversation_first():
    rows = [feedback("a", "a1", 0.5, 2), feedback("b", "b1", 0.5, 3)]
    turns = [turn("a1", "a", 1), turn("b1", "b", 2)]

    result = hydrate(rows, turns, turns)

    assert [item.conversation_id for item in result] == ["b", "a"]


def test_display_fallback_normalizes_and_bounds_first_request():
    request = "  Please   investigate\n" + "a" * 100
    only_turn = turn("a1", "a", 0, request=request)

    result = hydrate([feedback("a", "a1", 0.5, 1)], [only_turn], [only_turn])

    assert len(result[0].display_name) == 80
    assert result[0].display_name.startswith("Please investigate ")
    assert result[0].display_name.endswith("...")


def test_conversation_link_targets_encoded_wandb_agents_session():
    only_turn = turn("turn-1", "session/with spaces", 0)

    result = hydrate(
        [feedback("one", "turn-1", 0.5, 1)],
        [only_turn],
        [only_turn],
    )

    assert result[0].wandb_url == (
        "https://wandb.ai/weave-team/agent-sessions/weave/agents/conversations/"
        "session%2Fwith%20spaces"
    )


def test_latest_recorded_display_name_wins():
    turns = [
        turn("a1", "a", 0, name="Old name"),
        turn("a2", "a", 1, name="New name"),
    ]

    result = hydrate([feedback("a", "a1", 0.5, 2)], [turns[0]], turns)

    assert result[0].display_name == "New name"


def test_hydration_fails_on_missing_triggering_turn():
    with pytest.raises(HydrationError, match="missing triggering turn"):
        hydrate([feedback("one", "a1", 0.25, 1)], [], [])


def test_hydration_fails_on_missing_conversation_identity():
    missing = turn("a1", "", 0)
    with pytest.raises(HydrationError, match="conversation identity"):
        hydrate([feedback("one", "a1", 0.25, 1)], [missing], [missing])


def test_unknown_scorer_is_ignored_before_turn_resolution():
    unknown = feedback("unknown", "missing", 0.0, 1).model_copy(
        update={"runnable_ref": "weave:///other"}
    )

    assert hydrate([unknown], [], []) == []


def test_inconsistent_duplicate_feedback_identity_fails_closed():
    first = feedback("same-id", "a1", 0.5, 1)
    second = feedback("same-id", "a1", 0.25, 2)
    a1 = turn("a1", "a", 0)

    with pytest.raises(HydrationError, match="inconsistent duplicate feedback"):
        hydrate([first, second], [a1], [a1])


def test_duplicate_turn_identity_fails_closed():
    first = turn("a1", "a", 0)
    conflicting = turn("a1", "b", 0)

    with pytest.raises(HydrationError, match="inconsistent duplicate turn"):
        hydrate([feedback("one", "a1", 0.5, 1)], [first, conflicting], [first])


def test_empty_complete_input_is_successful():
    assert hydrate([], [], []) == []


def raw_feedback(index, *, runnable_ref=IDENTITY.scorer_ref, output=None):
    return {
        "id": f"feedback-{index}",
        "weave_ref": f"weave:///weave-team/agent-sessions/call/turn-{index}",
        "runnable_ref": runnable_ref,
        "feedback_type": "wandb.agent_monitor",
        "created_at": NOW + timedelta(seconds=index),
        "payload": {"output": output or {"value": 0.5, "reason": "visible"}},
        "scorer_ratings": {"_rating_": (output or {}).get("value", 0.5)},
        "scorer_rating_reasons": {"_rating_": (output or {}).get("reason", "visible")},
    }


class FeedbackServer:
    def __init__(self, rows):
        self.rows = rows
        self.requests = []

    def feedback_query(self, request):
        self.requests.append(request)
        start = request.offset or 0
        end = start + (request.limit or len(self.rows))
        return SimpleNamespace(result=self.rows[start:end])

    def agent_spans_query(self, request):
        return SimpleNamespace(spans=[])


def test_gateway_reads_complete_feedback_pages_for_exact_identities():
    rows = [raw_feedback(index) for index in range(201)]
    server = FeedbackServer(rows)
    gateway = WeaveGateway("weave-team", "agent-sessions", client=SimpleNamespace(server=server))

    result = gateway.read_feedback((IDENTITY,), NOW, NOW + timedelta(hours=1))

    assert len(result) == 201
    assert [request.offset for request in server.requests] == [0, 100, 200]
    assert all(request.limit == 100 for request in server.requests)
    query_text = str(server.requests[0].query)
    assert IDENTITY.scorer_ref in query_text
    assert "created_at" in query_text


def test_gateway_ignores_unknown_and_incomplete_feedback_before_parsing():
    rows = [
        raw_feedback(1, runnable_ref="weave:///unknown", output={"bad": "shape"}),
        {**raw_feedback(2), "payload": {}},
    ]
    gateway = WeaveGateway(
        "weave-team",
        "agent-sessions",
        client=SimpleNamespace(server=FeedbackServer(rows)),
    )

    assert gateway.read_feedback((IDENTITY,), NOW, NOW + timedelta(hours=1)) == ()


def test_gateway_fails_on_malformed_eligible_feedback():
    gateway = WeaveGateway(
        "weave-team",
        "agent-sessions",
        client=SimpleNamespace(server=FeedbackServer([raw_feedback(1, output={"rating": 0.5})])),
    )

    with pytest.raises(WeaveReadError, match="malformed eligible feedback"):
        gateway.read_feedback((IDENTITY,), NOW, NOW + timedelta(hours=1))


def test_gateway_fails_when_typed_rating_disagrees_with_output():
    row = raw_feedback(1, output={"value": 0.25, "reason": "visible"})
    row["scorer_ratings"] = {"_rating_": 0.5}
    gateway = WeaveGateway(
        "weave-team",
        "agent-sessions",
        client=SimpleNamespace(server=FeedbackServer([row])),
    )

    with pytest.raises(WeaveReadError, match="malformed eligible feedback"):
        gateway.read_feedback((IDENTITY,), NOW, NOW + timedelta(hours=1))


def test_gateway_fails_on_repeated_feedback_id_across_pages():
    rows = [raw_feedback(index) for index in range(100)]
    rows.append(raw_feedback(0))
    gateway = WeaveGateway(
        "weave-team",
        "agent-sessions",
        client=SimpleNamespace(server=FeedbackServer(rows)),
    )

    with pytest.raises(WeaveReadError, match="repeated feedback id"):
        gateway.read_feedback((IDENTITY,), NOW, NOW + timedelta(hours=1))


def raw_call(
    id,
    *,
    trace_id=None,
    thread_id=None,
    conversation_id=None,
    minute=0,
    parent_id=None,
    turn_id=None,
    display_name=None,
    user_request="Please investigate",
):
    attributes = {}
    if conversation_id is not None:
        attributes["conversation_id"] = conversation_id
    return SimpleNamespace(
        id=id,
        trace_id=trace_id or f"trace-{id}",
        thread_id=thread_id,
        turn_id=turn_id,
        parent_id=parent_id,
        started_at=NOW + timedelta(minutes=minute),
        ended_at=NOW + timedelta(minutes=minute, seconds=30),
        display_name=display_name,
        attributes=attributes,
        inputs={"input_messages": [{"role": "user", "content": user_request}]},
    )


class CallsClient:
    def __init__(self, calls):
        self.calls = calls
        self.requests = []
        self.server = FeedbackServer([])

    def get_calls(self, **kwargs):
        self.requests.append(kwargs)
        filter_value = kwargs.get("filter") or {}
        call_ids = set(filter_value.get("call_ids") or [])
        trace_ids = set(filter_value.get("trace_ids") or [])
        thread_ids = set(filter_value.get("thread_ids") or [])
        if call_ids:
            return [call for call in self.calls if call.id in call_ids]
        if trace_ids:
            return [call for call in self.calls if call.trace_id in trace_ids]
        if thread_ids:
            return [call for call in self.calls if call.thread_id in thread_ids]
        if kwargs.get("query") is not None:
            return [
                call for call in self.calls if call.attributes.get("conversation_id") is not None
            ]
        return []


def raw_agent_span(
    trace_id,
    *,
    conversation_id,
    minute=0,
    parent_span_id="",
    conversation_name="",
    user_request="Please investigate",
):
    return SimpleNamespace(
        trace_id=trace_id,
        span_id=f"span-{trace_id}",
        parent_span_id=parent_span_id,
        conversation_id=conversation_id,
        conversation_name=conversation_name,
        started_at=NOW + timedelta(minutes=minute),
        ended_at=NOW + timedelta(minutes=minute, seconds=30),
        input_messages=[{"role": "user", "content": user_request}],
    )


class AgentSpanServer(FeedbackServer):
    def __init__(self, spans):
        super().__init__([])
        self.spans = spans
        self.agent_requests = []

    def agent_spans_query(self, request):
        self.agent_requests.append(request)
        return SimpleNamespace(spans=self.spans)


class AgentSpanClient(CallsClient):
    def __init__(self, calls, spans):
        super().__init__(calls)
        self.server = AgentSpanServer(spans)


def test_gateway_reads_agent_turn_from_agent_spans_endpoint():
    span = raw_agent_span(
        "otel-trace",
        conversation_id="otel-conversation",
        conversation_name="OTel chat",
    )
    client = AgentSpanClient([], [span])
    gateway = WeaveGateway("weave-team", "agent-sessions", client=client)

    result = gateway.read_turns(("weave:///weave-team/agent-sessions/agent_turn/otel-trace",))

    assert result == (
        TurnRecord(
            turn_id="otel-trace",
            conversation_id="otel-conversation",
            started_at=NOW,
            ended_at=NOW + timedelta(seconds=30),
            display_name="OTel chat",
            user_request="Please investigate",
        ),
    )
    assert len(client.server.agent_requests) == 1
    assert "otel-trace" in str(client.server.agent_requests[0].query)


def test_gateway_interprets_naive_agent_span_timestamps_as_utc():
    span = raw_agent_span("otel-trace", conversation_id="otel-conversation")
    span.started_at = span.started_at.replace(tzinfo=None)
    span.ended_at = span.ended_at.replace(tzinfo=None)
    gateway = WeaveGateway(
        "weave-team",
        "agent-sessions",
        client=AgentSpanClient([], [span]),
    )

    result = gateway.read_turns(("weave:///weave-team/agent-sessions/agent_turn/otel-trace",))

    assert result[0].started_at.tzinfo == timezone.utc
    assert result[0].ended_at is not None
    assert result[0].ended_at.tzinfo == timezone.utc


def test_gateway_reads_native_and_adapter_triggering_turn_refs():
    native = raw_call(
        "native-call", thread_id="native-conversation", turn_id="native-call", minute=1
    )
    adapter = raw_agent_span(
        "adapter-trace",
        conversation_id="adapter-conversation",
        minute=2,
        conversation_name="Adapter chat",
    )
    client = AgentSpanClient([native], [adapter])
    gateway = WeaveGateway("weave-team", "agent-sessions", client=client)

    result = gateway.read_turns(
        (
            "weave:///weave-team/agent-sessions/call/native-call",
            "weave:///weave-team/agent-sessions/agent_turn/adapter-trace",
        )
    )

    assert [item.turn_id for item in result] == ["native-call", "adapter-trace"]
    assert [item.conversation_id for item in result] == [
        "native-conversation",
        "adapter-conversation",
    ]
    assert result[1].display_name == "Adapter chat"
    assert result[0].user_request == "Please investigate"


def test_gateway_fails_when_triggering_turn_is_missing_or_ambiguous():
    gateway = WeaveGateway("e", "p", client=CallsClient([]))
    with pytest.raises(WeaveReadError, match="missing requested turn"):
        gateway.read_turns(("weave:///e/p/call/missing",))

    duplicate_roots = [
        raw_agent_span("same", conversation_id="a"),
        raw_agent_span("same", conversation_id="a"),
    ]
    gateway = WeaveGateway("e", "p", client=AgentSpanClient([], duplicate_roots))
    with pytest.raises(WeaveReadError, match="ambiguous requested turn"):
        gateway.read_turns(("weave:///e/p/agent_turn/same",))


def test_gateway_reads_complete_native_and_adapter_conversations():
    calls = [
        raw_call("n1", thread_id="native", turn_id="n1", minute=0),
        raw_call("n2", thread_id="native", turn_id="n2", minute=1),
        raw_call("a1", trace_id="a1", conversation_id="adapter", minute=2),
        raw_call("nested", trace_id="nested", conversation_id="adapter", parent_id="a1"),
    ]
    gateway = WeaveGateway("e", "p", client=CallsClient(calls))

    result = gateway.read_conversation_turns(("native", "adapter"))

    assert [item.turn_id for item in result] == ["n1", "n2", "a1"]


def test_gateway_reads_agent_conversation_from_agent_spans_endpoint():
    spans = [
        raw_agent_span("turn-1", conversation_id="otel", minute=0),
        raw_agent_span("turn-2", conversation_id="otel", minute=1),
    ]
    client = AgentSpanClient([], spans)
    gateway = WeaveGateway("weave-team", "agent-sessions", client=client)

    result = gateway.read_conversation_turns(("otel",))

    assert [item.turn_id for item in result] == ["turn-1", "turn-2"]
    assert all(item.conversation_id == "otel" for item in result)
    assert len(client.server.agent_requests) == 1
    assert "otel" in str(client.server.agent_requests[0].query)


def test_gateway_fails_when_conversation_cannot_be_hydrated():
    gateway = WeaveGateway("e", "p", client=CallsClient([]))

    with pytest.raises(WeaveReadError, match="missing conversation"):
        gateway.read_conversation_turns(("missing",))
