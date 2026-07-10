from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from weave_agent_signals.client import WeaveClient
from weave_agent_signals.models import Score


@pytest.fixture
def client():
    return WeaveClient(
        api_key="test-key",
        entity="test-entity",
        project="test-project",
    )


def _fake_span(**kw):
    defaults = {
        "trace_id": "tr-1",
        "span_id": "sp-1",
        "conversation_id": "conv-1",
        "operation_name": "invoke_agent",
        "parent_span_id": None,
        "started_at": "2026-07-09T12:00:00Z",
        "ended_at": "2026-07-09T12:05:00Z",
        "model": "claude-opus-4",
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_tokens": 200,
        "status_code": "OK",
        "events_dump": "[]",
        "custom_attrs": {
            "weave_agent_adapter.config_version": "abc123",
            "weave_agent_adapter.git_branch": "main",
            "weave_agent_adapter.effort_level": "high",
            "weave_agent_adapter.session_id": "sess-1",
            "weave_agent_adapter.steering_count": 0,
            "weave_agent_adapter.denial_count": 0,
            "weave_agent_adapter.tool_error_count": 0,
        },
    }
    defaults.update(kw)
    return defaults


def _fake_child_span(op="execute_tool", tool_name="Bash", **kw):
    defaults = {
        "trace_id": "tr-1",
        "span_id": "sp-child",
        "conversation_id": "conv-1",
        "operation_name": op,
        "parent_span_id": "sp-1",
        "started_at": "2026-07-09T12:01:00Z",
        "ended_at": "2026-07-09T12:01:30Z",
        "status_code": "OK",
        "custom_attrs": {
            "gen_ai.tool.call.name": tool_name,
            "gen_ai.tool.call.arguments": '{"command": "pytest"}',
            "gen_ai.tool.call.result": "5 passed in 1.0s",
        },
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
    assert turn.steering_count == 0
    assert turn.input_tokens == 1000


def test_hydrate_turn_missing_attrs(client):
    raw = _fake_span(custom_attrs={})
    turn = client._hydrate_turn(raw)
    assert turn.config_version is None
    assert turn.steering_count == 0


def test_hydrate_tool_span(client):
    raw = _fake_child_span()
    tool = client._hydrate_tool_span(raw)
    assert tool.tool_name == "Bash"
    assert "pytest" in tool.arguments


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
    assert body["scorer_ratings"]["_rating_"] == 1.0
