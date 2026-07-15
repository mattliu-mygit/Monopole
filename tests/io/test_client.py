from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
import respx

from weave_agent_signals.client import TRACE_BASE, WeaveClient
from weave_agent_signals.models import Score


@pytest.fixture
def client():
    return WeaveClient(
        api_key="test-key",
        entity="test-entity",
        project="test-project",
    )


def _fake_span(**kw):
    # Real invoke_agent root shape: parent_span_id is "" (not null), token
    # columns are all-zero on the root (tokens live on chat children), and
    # custom attrs arrive in typed maps — never a merged `custom_attrs` dict.
    defaults = {
        "trace_id": "tr-1",
        "span_id": "sp-1",
        "conversation_id": "conv-1",
        "operation_name": "invoke_agent",
        "parent_span_id": "",
        "started_at": "2026-07-09T12:00:00Z",
        "ended_at": "2026-07-09T12:05:00Z",
        "agent_name": "claude-code",
        "response_model": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "status_code": "OK",
        "events_dump": "[]",
        "custom_attrs_string": {
            "weave_agent_adapter.config_version": "abc123",
            "weave_agent_adapter.git_branch": "main",
            "weave_agent_adapter.effort_level": "high",
            "weave_agent_adapter.session_id": "sess-1",
        },
        "custom_attrs_int": {
            "weave_agent_adapter.steering_count": 0,
            "weave_agent_adapter.denial_count": 0,
            "weave_agent_adapter.tool_error_count": 0,
        },
    }
    defaults.update(kw)
    return defaults


def _fake_chat_span(**kw):
    # Chat spans carry the token quartet as top-level columns and finish_reasons
    # as a list; model is response_model/request_model.
    defaults = {
        "trace_id": "tr-1",
        "span_id": "sp-chat",
        "conversation_id": "conv-1",
        "operation_name": "chat",
        "parent_span_id": "sp-1",
        "started_at": "2026-07-09T12:01:00Z",
        "ended_at": "2026-07-09T12:01:10Z",
        "status_code": "OK",
        "response_model": "claude-opus-4-6",
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 30,
        "finish_reasons": ["end_turn"],
    }
    defaults.update(kw)
    return defaults


def _fake_child_span(op="execute_tool", tool_name="Bash", **kw):
    # Mirrors the real API shape with include_details=True: tool_name and the
    # tool_call_* detail columns arrive as top-level fields, not custom_attrs.
    defaults = {
        "trace_id": "tr-1",
        "span_id": "sp-child",
        "conversation_id": "conv-1",
        "operation_name": op,
        "parent_span_id": "sp-1",
        "started_at": "2026-07-09T12:01:00Z",
        "ended_at": "2026-07-09T12:01:30Z",
        "status_code": "OK",
        "tool_name": tool_name,
        "tool_call_arguments": '{"command": "pytest"}',
        "tool_call_result": "5 passed in 1.0s",
        "custom_attrs": {},
    }
    defaults.update(kw)
    return defaults


# --- Hydration ---


def test_hydrate_turn(client):
    raw = _fake_span()
    turn = client._hydrate_turn(raw)
    assert turn.trace_id == "tr-1"
    assert turn.conversation_id == "conv-1"
    assert turn.config_version == "abc123"
    assert turn.git_branch == "main"
    assert turn.effort_level == "high"
    assert turn.steering_count == 0


def test_hydrate_turn_reads_typed_custom_attr_maps(client):
    """config_version and counters come from custom_attrs_string/_int, the real
    API layout — not a merged custom_attrs dict."""
    raw = _fake_span(
        custom_attrs_string={"weave_agent_adapter.config_version": "v9"},
        custom_attrs_int={
            "weave_agent_adapter.steering_count": 3,
            "weave_agent_adapter.denial_count": 2,
        },
    )
    turn = client._hydrate_turn(raw)
    assert turn.config_version == "v9"
    assert turn.steering_count == 3
    assert turn.denial_count == 2


def test_hydrate_turn_missing_attrs(client):
    raw = _fake_span(custom_attrs_string={}, custom_attrs_int={})
    turn = client._hydrate_turn(raw)
    assert turn.config_version is None
    assert turn.steering_count == 0


@pytest.mark.parametrize(
    ("output_messages", "expected"),
    [
        (
            json.dumps(
                [
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": "Implemented the fix."}],
                    }
                ]
            ),
            "Implemented the fix.",
        ),
        ([{"role": "assistant", "content": "Tests now pass."}], "Tests now pass."),
        (
            [
                {"role": "assistant", "content": "Draft response"},
                {"role": "tool", "content": "ignored"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Final"},
                        {"type": "image", "url": "ignored"},
                        {"type": "text", "text": "response"},
                    ],
                },
            ],
            "Final response",
        ),
    ],
)
def test_hydrate_turn_extracts_assistant_output_from_root(client, output_messages, expected):
    turn = client._hydrate_turn(_fake_span(output_messages=output_messages))

    assert turn.assistant_output == expected


@pytest.mark.parametrize("output_messages", [None, "{not-json", {"role": "assistant"}, []])
def test_hydrate_turn_ignores_invalid_assistant_output(client, output_messages):
    turn = client._hydrate_turn(_fake_span(output_messages=output_messages))

    assert turn.assistant_output is None


def test_hydrate_turn_retains_event_identity_without_unused_attributes(client):
    raw = _fake_span(
        events_dump=json.dumps(
            [
                {
                    "name": "agent_event",
                    "timestamp": "2026-07-09T12:02:03Z",
                    "attributes": {"source-only": "discarded"},
                }
            ]
        )
    )

    turn = client._hydrate_turn(raw)

    assert len(turn.events) == 1
    assert turn.events[0].name == "agent_event"
    assert turn.events[0].timestamp == datetime(2026, 7, 9, 12, 2, 3, tzinfo=timezone.utc)
    assert not hasattr(turn.events[0], "attributes")


def test_hydrate_tool_span(client):
    raw = _fake_child_span()
    tool = client._hydrate_tool_span(raw)
    assert tool.tool_name == "Bash"
    assert "pytest" in tool.arguments


def test_hydrate_tool_span_detail_columns(client):
    """arguments/result come from the top-level detail-only columns."""
    raw = {
        "span_id": "sp-child",
        "tool_name": "Bash",
        "tool_call_arguments": '{"command": "pytest tests/"}',
        "tool_call_result": "3 passed in 0.5s",
        "status_code": "OK",
        "started_at": "2026-07-09T12:01:00Z",
        "ended_at": "2026-07-09T12:01:30Z",
    }
    tool = client._hydrate_tool_span(raw)
    assert tool.tool_name == "Bash"
    assert "pytest tests/" in tool.arguments
    assert "3 passed" in tool.result


def test_hydrate_tool_span_null_details(client):
    """Missing detail columns degrade to empty strings, not None."""
    raw = {
        "span_id": "sp-child",
        "tool_name": "Bash",
        "tool_call_arguments": None,
        "tool_call_result": None,
        "status_code": "OK",
        "started_at": "2026-07-09T12:01:00Z",
    }
    tool = client._hydrate_tool_span(raw)
    assert tool.arguments == ""
    assert tool.result == ""


# --- Request building ---


def test_build_turn_query(client):
    q = client._build_turn_query(limit=50)
    assert q["project_id"] == "test-entity/test-project"
    assert q["limit"] == 50
    assert len(q["custom_attr_columns"]) > 0


def test_build_feedback_body(client):
    score = Score(
        scorer="outcome.test",
        value=1.0,
        tags=["test_pass"],
        confidence=1.0,
        metadata={"passed": 5},
        granularity="turn",
    )
    ref = "weave:///test-entity/test-project/agent_turn/tr-1"
    body = client._build_feedback_body(score, ref)
    assert body["project_id"] == "test-entity/test-project"
    assert body["weave_ref"] == ref
    assert body["feedback_type"] == "weave_agent_signals.outcome.test"
    assert body["payload"]["rating"] == 1.0


# --- HTTP integration (respx) ---


@pytest.fixture
def mock_client():
    c = WeaveClient(api_key="test-key", entity="test-entity", project="test-project")
    return c


@respx.mock
def test_query_turns_http(mock_client):
    route = respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={"spans": [_fake_span(), _fake_span(trace_id="tr-2")]},
    )
    turns = mock_client.query_turns(limit=10)
    assert len(turns) == 2
    assert turns[0].trace_id == "tr-1"
    assert turns[1].trace_id == "tr-2"
    assert route.called


@respx.mock
def test_query_turns_empty(mock_client):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(json={"spans": []})
    turns = mock_client.query_turns(limit=10)
    assert turns == []


@respx.mock
def test_hydrate_children_http(mock_client):
    route = respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={
            "spans": [
                _fake_span(),
                _fake_child_span(span_id="sp-tool1"),
            ]
        },
    )
    turn = mock_client._hydrate_turn(_fake_span())
    assert len(turn.tool_calls) == 0
    mock_client.hydrate_turn_children(turn)
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].tool_name == "Bash"
    body = json.loads(route.calls[0].request.content)
    assert body["include_details"] is True
    assert body["limit"] == 10_000


@respx.mock
def test_hydrate_turn_children_rejects_saturated_detail_limit(mock_client):
    spans = [_fake_span()]
    spans.extend(
        _fake_child_span(span_id=f"child-{index}", operation_name="noop") for index in range(9_999)
    )
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(json={"spans": spans})
    turn = mock_client._hydrate_turn(_fake_span())

    with pytest.raises(RuntimeError, match=r"reached the 10000-span limit"):
        mock_client.hydrate_turn_children(turn)

    assert turn.tool_calls == []


@respx.mock
def test_hydrate_children_backfills_root_assistant_output(mock_client):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={
            "spans": [
                _fake_span(output_messages=[{"role": "assistant", "content": "Completed."}]),
                _fake_child_span(),
            ]
        }
    )
    turn = mock_client._hydrate_turn(_fake_span())

    mock_client.hydrate_turn_children(turn)

    assert turn.assistant_output == "Completed."


@respx.mock
def test_hydrate_turns_batch_backfills_root_assistant_output(mock_client):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={
            "spans": [
                _fake_span(
                    trace_id="tr-1",
                    output_messages=[{"role": "assistant", "content": "First result"}],
                ),
                _fake_span(
                    trace_id="tr-2",
                    output_messages=[{"role": "assistant", "content": "Second result"}],
                ),
            ]
        }
    )
    turns = [
        mock_client._hydrate_turn(_fake_span(trace_id="tr-1")),
        mock_client._hydrate_turn(_fake_span(trace_id="tr-2")),
    ]

    mock_client.hydrate_turns_batch(turns)

    assert [turn.assistant_output for turn in turns] == ["First result", "Second result"]


@respx.mock
def test_hydrate_turns_batch_propagates_http_failure(mock_client):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(status_code=503)
    turn = mock_client._hydrate_turn(_fake_span())

    with pytest.raises(httpx.HTTPStatusError):
        mock_client.hydrate_turns_batch([turn])


@respx.mock
def test_hydrate_turns_batch_rejects_missing_requested_trace(mock_client):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={"spans": [_fake_span(trace_id="tr-1")]}
    )
    turns = [
        mock_client._hydrate_turn(_fake_span(trace_id="tr-1")),
        mock_client._hydrate_turn(_fake_span(trace_id="tr-2")),
    ]

    with pytest.raises(RuntimeError, match=r"missing requested traces?: tr-2"):
        mock_client.hydrate_turns_batch(turns)

    assert all(turn.tool_calls == [] for turn in turns)


@respx.mock
def test_hydrate_turns_batch_rejects_trace_without_detail_root(mock_client):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={"spans": [_fake_child_span(trace_id="tr-1")]}
    )
    turn = mock_client._hydrate_turn(_fake_span(trace_id="tr-1"))

    with pytest.raises(RuntimeError, match=r"missing detailed root trace: tr-1"):
        mock_client.hydrate_turns_batch([turn])

    assert turn.tool_calls == []


@respx.mock
def test_hydrate_turns_batch_rejects_duplicate_detail_roots(mock_client):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={
            "spans": [
                _fake_span(trace_id="tr-1", span_id="root-1"),
                _fake_span(trace_id="tr-1", span_id="root-2"),
            ]
        }
    )
    turn = mock_client._hydrate_turn(_fake_span(trace_id="tr-1"))

    with pytest.raises(RuntimeError, match=r"duplicate detailed root trace: tr-1"):
        mock_client.hydrate_turns_batch([turn])

    assert turn.tool_calls == []


@respx.mock
def test_hydrate_turns_batch_rejects_unexpected_trace(mock_client):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={
            "spans": [
                _fake_span(trace_id="tr-1"),
                _fake_span(trace_id="tr-unexpected"),
            ]
        }
    )
    turn = mock_client._hydrate_turn(_fake_span(trace_id="tr-1"))

    with pytest.raises(RuntimeError, match=r"unexpected trace: tr-unexpected"):
        mock_client.hydrate_turns_batch([turn])

    assert turn.tool_calls == []


@respx.mock
def test_hydrate_turns_batch_rejects_saturated_detail_limit(mock_client):
    spans = [_fake_span()]
    spans.extend(
        _fake_child_span(span_id=f"child-{index}", operation_name="noop") for index in range(9_999)
    )
    route = respx.post(f"{TRACE_BASE}/agents/spans/query").respond(json={"spans": spans})
    turn = mock_client._hydrate_turn(_fake_span())

    with pytest.raises(RuntimeError, match=r"reached the 10000-span limit"):
        mock_client.hydrate_turns_batch([turn])

    body = json.loads(route.calls[0].request.content)
    assert body["limit"] == 10_000
    assert turn.tool_calls == []


@respx.mock
def test_write_score_http(mock_client):
    route = respx.post(f"{TRACE_BASE}/feedback/create").respond(
        json={"id": "fb-1"},
    )
    score = Score(
        scorer="outcome.test",
        value=1.0,
        tags=["test_pass"],
        confidence=1.0,
        metadata={"passed": 5},
        granularity="turn",
    )
    ref = "weave:///test-entity/test-project/agent_turn/tr-1"
    result = mock_client.write_score(score, ref)
    assert result["id"] == "fb-1"
    body = json.loads(route.calls[0].request.content)
    assert body["feedback_type"] == "weave_agent_signals.outcome.test"
    assert body["weave_ref"] == ref


@respx.mock
def test_query_existing_feedback_http(mock_client):
    respx.post(f"{TRACE_BASE}/feedback/query").respond(
        json={"feedback": [{"id": "fb-1", "feedback_type": "weave_agent_signals.outcome.test"}]},
    )
    ref = "weave:///test-entity/test-project/agent_turn/tr-1"
    results = mock_client.query_existing_feedback(ref, "weave_agent_signals.outcome.test")
    assert len(results) == 1


@respx.mock
def test_query_all_feedback_http(mock_client):
    respx.post(f"{TRACE_BASE}/feedback/query").respond(
        json={
            "feedback": [
                {"id": "fb-1", "feedback_type": "weave_agent_signals.outcome.test"},
                {"id": "fb-2", "feedback_type": "weave_agent_signals.efficiency"},
            ]
        },
    )
    ref = "weave:///test-entity/test-project/agent_turn/tr-1"
    results = mock_client.query_all_feedback(ref)
    assert len(results) == 2


@respx.mock
def test_query_all_feedback_batch_groups_every_type_by_ref(mock_client):
    route = respx.post(f"{TRACE_BASE}/feedback/query").respond(
        json={
            "result": [
                {
                    "id": "fb-1",
                    "weave_ref": "weave:///test/turn-1",
                    "feedback_type": "weave_agent_signals.outcome.test",
                },
                {
                    "id": "fb-2",
                    "weave_ref": "weave:///test/turn-1",
                    "feedback_type": "weave_agent_signals.efficiency",
                },
            ]
        }
    )

    results = mock_client.query_all_feedback_batch(["weave:///test/turn-1", "weave:///test/turn-2"])

    assert [item["id"] for item in results["weave:///test/turn-1"]] == ["fb-1", "fb-2"]
    assert results["weave:///test/turn-2"] == []
    assert len(route.calls) == 1
    assert json.loads(route.calls[0].request.content)["limit"] == 10_000


@respx.mock
def test_query_all_feedback_batch_rejects_saturated_response(mock_client):
    respx.post(f"{TRACE_BASE}/feedback/query").respond(
        json={
            "result": [
                {
                    "id": f"fb-{index}",
                    "weave_ref": "weave:///test/turn-1",
                    "feedback_type": "weave_agent_signals.outcome.test",
                }
                for index in range(10_000)
            ]
        }
    )

    with pytest.raises(RuntimeError, match="feedback hydration reached the 10000-row limit"):
        mock_client.query_all_feedback_batch(["weave:///test/turn-1"])


@respx.mock
def test_query_feedback_for_refs_fetches_each_ref_without_project_limit(mock_client):
    route = respx.post(f"{TRACE_BASE}/feedback/query").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "id": "fb-turn",
                            "weave_ref": "weave:///test/turn-1",
                            "feedback_type": "weave_agent_signals.outcome.test",
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "id": "fb-session",
                            "weave_ref": "weave:///test/session-1",
                            "feedback_type": "weave_agent_signals.judge.session_outcome",
                        }
                    ]
                },
            ),
        ]
    )

    feedback = mock_client.query_feedback_for_refs(
        ["weave:///test/turn-1", "weave:///test/session-1"]
    )

    assert [item["id"] for item in feedback] == ["fb-turn", "fb-session"]
    assert route.call_count == 2
    bodies = [json.loads(call.request.content) for call in route.calls]
    assert [body["query"]["$expr"]["$eq"][1]["$literal"] for body in bodies] == [
        "weave:///test/turn-1",
        "weave:///test/session-1",
    ]
    assert all("limit" not in body for body in bodies)


@respx.mock
def test_query_turns_by_trace_ids_is_exact_and_preserves_cohort_order(mock_client):
    route = respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={
            "spans": [
                _fake_span(trace_id="turn-2", started_at="2026-07-09T12:10:00Z"),
                _fake_span(trace_id="turn-1", started_at="2026-07-09T12:00:00Z"),
            ]
        }
    )

    turns = mock_client.query_turns_by_trace_ids(["turn-1", "turn-2"])

    assert [turn.trace_id for turn in turns] == ["turn-1", "turn-2"]
    body = json.loads(route.calls[0].request.content)
    conditions = body["query"]["$expr"]["$and"]
    trace_condition = next(condition for condition in conditions if "$in" in condition)
    assert [literal["$literal"] for literal in trace_condition["$in"][1]] == [
        "turn-1",
        "turn-2",
    ]
    assert body["limit"] == 3


@respx.mock
def test_query_turns_by_trace_ids_rejects_duplicate_roots_before_collapse(mock_client):
    route = respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={
            "spans": [
                _fake_span(trace_id="turn-1", span_id="root-1"),
                _fake_span(trace_id="turn-1", span_id="root-2"),
            ]
        }
    )

    with pytest.raises(RuntimeError, match=r"duplicate root trace: turn-1"):
        mock_client.query_turns_by_trace_ids(["turn-1", "turn-2"])

    body = json.loads(route.calls[0].request.content)
    assert body["limit"] == 3


@respx.mock
def test_query_turns_by_trace_ids_rejects_saturated_root_query(mock_client):
    route = respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={
            "spans": [
                _fake_span(trace_id="turn-1", span_id="root-1"),
                _fake_span(trace_id="turn-2", span_id="root-2"),
                _fake_span(trace_id="turn-1", span_id="root-3"),
            ]
        }
    )

    with pytest.raises(RuntimeError, match=r"reached the 3-row limit"):
        mock_client.query_turns_by_trace_ids(["turn-1", "turn-2"])

    body = json.loads(route.calls[0].request.content)
    assert body["limit"] == 3


@respx.mock
def test_query_turns_by_trace_ids_rejects_unexpected_root(mock_client):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={
            "spans": [
                _fake_span(trace_id="turn-1"),
                _fake_span(trace_id="turn-unexpected"),
            ]
        }
    )

    with pytest.raises(RuntimeError, match=r"unexpected root trace: turn-unexpected"):
        mock_client.query_turns_by_trace_ids(["turn-1", "turn-2"])


@respx.mock
def test_query_session_http(mock_client):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(
        json={
            "spans": [
                _fake_span(trace_id="tr-1", started_at="2026-07-09T12:00:00Z"),
                _fake_span(trace_id="tr-2", started_at="2026-07-09T12:10:00Z"),
            ]
        },
    )
    session = mock_client.query_session("conv-1")
    assert session.conversation_id == "conv-1"
    assert len(session.turns) == 2
    assert session.turns[0].trace_id == "tr-1"


def test_query_session_uses_paginated_root_query(mock_client, monkeypatch):
    turns = [
        mock_client._hydrate_turn(_fake_span(trace_id="tr-2", started_at="2026-07-09T12:10:00Z")),
        mock_client._hydrate_turn(_fake_span(trace_id="tr-1", started_at="2026-07-09T12:00:00Z")),
    ]
    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        return turns

    monkeypatch.setattr(mock_client, "query_turns_paginated", query)

    session = mock_client.query_session("conv-1")

    assert calls == [
        {
            "page_size": 500,
            "conversation_id": "conv-1",
            "include_details": False,
        }
    ]
    assert [turn.trace_id for turn in session.turns] == ["tr-1", "tr-2"]


def test_query_session_breaks_equal_timestamp_ties_by_trace_id(mock_client, monkeypatch):
    timestamp = "2026-07-09T12:00:00Z"
    turns = [
        mock_client._hydrate_turn(_fake_span(trace_id="tr-b", started_at=timestamp)),
        mock_client._hydrate_turn(_fake_span(trace_id="tr-a", started_at=timestamp)),
    ]
    monkeypatch.setattr(mock_client, "query_turns_paginated", lambda **_kwargs: turns)

    session = mock_client.query_session("conv-1")

    assert [turn.trace_id for turn in session.turns] == ["tr-a", "tr-b"]


@respx.mock
def test_delete_feedback_http(mock_client):
    route = respx.post(f"{TRACE_BASE}/feedback/purge").respond(json={"deleted": 1})
    mock_client.delete_feedback("fb-123")
    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["query"]["$expr"]["$eq"][1]["$literal"] == "fb-123"


@respx.mock
def test_delete_feedback_ids(mock_client):
    purge_route = respx.post(f"{TRACE_BASE}/feedback/purge").respond(json={})
    mock_client.delete_feedback_ids([{"id": "fb-1"}, {"id": "fb-2"}])
    assert purge_route.call_count == 2


@respx.mock
def test_query_turns_paginated_multiple_pages(mock_client):
    """Pagination uses ascending sort and $gte cursor with dedup."""
    page1 = [
        _fake_span(trace_id="tr-1", started_at="2026-07-01T12:00:00Z"),
        _fake_span(trace_id="tr-2", started_at="2026-07-01T13:00:00Z"),
    ]
    page2 = [
        _fake_span(trace_id="tr-2", started_at="2026-07-01T13:00:00Z"),
        _fake_span(trace_id="tr-3", started_at="2026-07-01T14:00:00Z"),
    ]
    route = respx.post(f"{TRACE_BASE}/agents/spans/query").mock(
        side_effect=[
            respx.MockResponse(json={"spans": page1}),
            respx.MockResponse(json={"spans": page2}),
            respx.MockResponse(json={"spans": []}),
        ]
    )
    since = datetime(2026, 7, 1, tzinfo=timezone.utc)
    turns = mock_client.query_turns_paginated(page_size=2, since=since)
    # tr-2 appears in both pages but dedup keeps it once
    assert len(turns) == 3
    assert turns[0].trace_id == "tr-1"
    assert turns[2].trace_id == "tr-3"
    assert route.call_count == 3
    # Second query uses $gte cursor (not $gt) to avoid boundary loss
    # Two $gte: one from original `since`, one from the pagination cursor
    second_body = json.loads(route.calls[1].request.content)
    conditions = second_body["query"]["$expr"]["$and"]
    gte_conds = [c for c in conditions if "$gte" in c]
    assert len(gte_conds) == 2
    cursor_literals = [c["$gte"][1]["$literal"] for c in gte_conds]
    assert "2026-07-01T13:00:00.000000" in cursor_literals


@respx.mock
def test_query_turns_paginated_rejects_saturated_timestamp(mock_client):
    timestamp = "2026-07-01T12:00:00Z"
    route = respx.post(f"{TRACE_BASE}/agents/spans/query").mock(
        side_effect=[
            respx.MockResponse(
                json={
                    "spans": [
                        _fake_span(trace_id="tr-1", started_at=timestamp),
                        _fake_span(trace_id="tr-2", started_at=timestamp),
                    ]
                }
            ),
            respx.MockResponse(
                json={
                    "spans": [
                        _fake_span(trace_id="tr-3", started_at=timestamp),
                        _fake_span(trace_id="tr-4", started_at=timestamp),
                    ]
                }
            ),
        ]
    )

    with pytest.raises(RuntimeError, match=r"cannot advance past saturated timestamp"):
        mock_client.query_turns_paginated(page_size=2)

    assert route.call_count == 2


@respx.mock
def test_query_turns_paginated_can_request_root_details(mock_client):
    route = respx.post(f"{TRACE_BASE}/agents/spans/query").respond(json={"spans": []})

    mock_client.query_turns_paginated(include_details=True)

    body = json.loads(route.calls[0].request.content)
    assert body["include_details"] is True


def test_attach_children_filters_by_trace(client):
    """Children from other turns in the same conversation must not leak."""
    turn = client._hydrate_turn(_fake_span(trace_id="tr-1"))

    children = [
        _fake_child_span(span_id="sp-mine", tool_name="Bash"),
        # This child belongs to a different turn (different trace_id)
        {
            **_fake_child_span(span_id="sp-other", tool_name="Read"),
            "trace_id": "tr-2",
        },
    ]
    client._attach_children(turn, children)
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].tool_name == "Bash"


def test_attach_children_detects_subagents(client):
    """Nested invoke_agent spans with parent_span_id should become subagents."""
    turn = client._hydrate_turn(_fake_span(trace_id="tr-1", span_id="sp-root"))

    subagent_span = {
        "trace_id": "tr-1",
        "span_id": "sp-sub-1",
        "conversation_id": "conv-1",
        "operation_name": "invoke_agent",
        "parent_span_id": "sp-root",
        "started_at": "2026-07-09T12:01:00Z",
        "ended_at": "2026-07-09T12:02:00Z",
        "status_code": "OK",
        # subagent type is the agent identity, in the top-level agent_name column
        "agent_name": "general-purpose",
    }
    tool_in_subagent = {
        **_fake_child_span(span_id="sp-sub-tool", tool_name="Read"),
        "parent_span_id": "sp-sub-1",
    }

    client._attach_children(turn, [subagent_span, tool_in_subagent])
    assert len(turn.subagents) == 1
    assert turn.subagents[0].span_id == "sp-sub-1"
    assert turn.subagents[0].agent_type == "general-purpose"
    assert len(turn.subagents[0].tool_calls) == 1
    assert turn.subagents[0].tool_calls[0].tool_name == "Read"
    # Tool calls parented to subagent should NOT be on the main turn
    assert all(tc.span_id != "sp-sub-tool" for tc in turn.tool_calls)


def test_attach_children_aggregates_tokens_from_chat_spans(client):
    """Turn tokens are summed from chat spans (main + subagent), not the
    all-zero invoke_agent root."""
    turn = client._hydrate_turn(_fake_span(trace_id="tr-1", span_id="sp-root"))
    assert turn.input_tokens == 0  # root carries no tokens

    children = [
        _fake_chat_span(
            span_id="c1",
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=200,
        ),
        _fake_chat_span(
            span_id="c2", input_tokens=30, output_tokens=20, cache_read_input_tokens=10
        ),
    ]
    client._attach_children(turn, children)
    assert turn.input_tokens == 130
    assert turn.output_tokens == 70
    assert turn.cache_read_tokens == 210
    assert turn.model == "claude-opus-4-6"


def test_attach_children_hydrates_chat_span_fields(client):
    turn = client._hydrate_turn(_fake_span(trace_id="tr-1", span_id="sp-root"))
    client._attach_children(turn, [_fake_chat_span(span_id="c1")])
    assert len(turn.chat_spans) == 1
    chat = turn.chat_spans[0]
    assert chat.model == "claude-opus-4-6"
    assert chat.cache_creation_tokens == 30
    assert chat.finish_reason == "end_turn"


def test_attach_children_selects_earliest_main_agent_model_deterministically(client):
    turn = client._hydrate_turn(_fake_span(trace_id="tr-1", span_id="sp-root"))
    subagent = {
        **_fake_span(
            span_id="sp-subagent",
            parent_span_id="sp-root",
            agent_name="general-purpose",
        ),
        "operation_name": "invoke_agent",
    }
    subagent_chat = _fake_chat_span(
        span_id="chat-subagent",
        parent_span_id="sp-subagent",
        started_at="2026-07-09T12:00:00Z",
        response_model="gemini-2.5-pro",
    )
    later_main_chat = _fake_chat_span(
        span_id="chat-main-later",
        parent_span_id="sp-root",
        started_at="2026-07-09T12:03:00Z",
        response_model="claude-sonnet-5",
    )
    earliest_main_chat = _fake_chat_span(
        span_id="chat-main-earliest",
        parent_span_id="sp-root",
        started_at="2026-07-09T12:01:00Z",
        response_model="claude-opus-4-8",
    )

    client._attach_children(
        turn,
        [subagent_chat, later_main_chat, subagent, earliest_main_chat],
    )

    assert turn.model == "claude-opus-4-8"
