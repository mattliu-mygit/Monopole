"""End-to-end pipeline test: mock Weave API → score → verify output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from unittest.mock import patch

import respx

from weave_agent_signals.cli import cmd_backfill, cmd_score, main
from weave_agent_signals.client import TRACE_BASE


def _fake_turn_span(trace_id="tr-1", conv_id="conv-1", tool_children=None):
    # Real invoke_agent root shape: typed custom-attr maps, all-zero root tokens.
    return {
        "trace_id": trace_id,
        "span_id": trace_id,
        "conversation_id": conv_id,
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
            "weave_agent_adapter.config_version": "cfg-abc",
            "weave_agent_adapter.git_branch": "main",
            "weave_agent_adapter.effort_level": "high",
            "weave_agent_adapter.session_id": "sess-1",
        },
        "custom_attrs_int": {
            "weave_agent_adapter.steering_count": 1,
            "weave_agent_adapter.denial_count": 0,
            "weave_agent_adapter.tool_error_count": 0,
        },
    }


def _fake_tool_span(tool_name="Bash", cmd="pytest tests/", result="===== 5 passed in 1.0s ====="):
    # Matches the real include_details=True response: tool detail columns are
    # top-level, not nested under custom_attrs.
    return {
        "trace_id": "tr-1",
        "span_id": "sp-tool-1",
        "conversation_id": "conv-1",
        "operation_name": "execute_tool",
        "parent_span_id": "tr-1",
        "started_at": "2026-07-09T12:01:00Z",
        "ended_at": "2026-07-09T12:01:30Z",
        "status_code": "OK",
        "tool_name": tool_name,
        "tool_call_arguments": json.dumps({"command": cmd}),
        "tool_call_result": result,
        "custom_attrs": {},
    }


def _make_args(**kw):
    defaults = dict(
        entity="test-entity",
        project="test-project",
        since=None,
        limit=100,
        dry_run=True,
        force=False,
        verbose=False,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


@respx.mock
@patch("weave_agent_signals.client._get_api_key", return_value="test-key")
def test_pipeline_dry_run_produces_scores(_mock_key, capsys):
    # Turn query returns one turn with a pytest run
    respx.post(f"{TRACE_BASE}/agents/spans/query").mock(
        side_effect=[
            # first call: query_turns
            respx.MockResponse(json={"spans": [_fake_turn_span()]}),
            # second call: hydrate_turn_children
            respx.MockResponse(json={"spans": [_fake_tool_span()]}),
        ]
    )

    result = cmd_score(_make_args())
    assert result == 0

    output = capsys.readouterr().out
    assert "outcome.test" in output
    assert "efficiency" in output
    assert "implicit.correction_density" in output
    assert "implicit.abandonment" in output
    assert "efficiency.session" in output


@respx.mock
@patch("weave_agent_signals.client._get_api_key", return_value="test-key")
def test_pipeline_no_turns(_mock_key, capsys):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(json={"spans": []})

    result = cmd_score(_make_args())
    assert result == 0
    assert "No turns found" in capsys.readouterr().out


@respx.mock
@patch("weave_agent_signals.client._get_api_key", return_value="test-key")
def test_force_deletes_before_write(_mock_key):
    """--force should delete existing feedback before writing new."""
    respx.post(f"{TRACE_BASE}/agents/spans/query").mock(
        side_effect=[
            respx.MockResponse(json={"spans": [_fake_turn_span()]}),
            respx.MockResponse(json={"spans": [_fake_tool_span()]}),
        ]
    )
    # query_existing_feedback returns existing feedback for every scorer
    respx.post(f"{TRACE_BASE}/feedback/query").respond(
        json={"feedback": [{"id": "fb-old", "feedback_type": "weave_agent_signals.outcome.test"}]}
    )
    purge_route = respx.post(f"{TRACE_BASE}/feedback/purge").respond(json={"deleted": 1})
    respx.post(f"{TRACE_BASE}/feedback/create").respond(json={"id": "fb-new"})

    result = cmd_score(_make_args(dry_run=False, force=True))
    assert result == 0
    assert purge_route.called


@patch(
    "weave_agent_signals.client._get_api_key",
    side_effect=RuntimeError("No W&B API key found"),
)
def test_pipeline_config_error_returns_2(_mock_key):
    result = main(["score", "--dry-run"])
    assert result == 2


@respx.mock
@patch("weave_agent_signals.client._get_api_key", return_value="test-key")
def test_pipeline_api_error_returns_3(_mock_key):
    respx.post(f"{TRACE_BASE}/agents/spans/query").respond(status_code=401)
    result = main(["score", "--dry-run"])
    assert result == 3


@respx.mock
@patch("weave_agent_signals.client._get_api_key", return_value="test-key")
def test_pipeline_multiple_turns_same_session(_mock_key, capsys):
    turns = [
        _fake_turn_span(trace_id="tr-1", conv_id="conv-1"),
        _fake_turn_span(trace_id="tr-2", conv_id="conv-1"),
    ]
    tool1 = _fake_tool_span(cmd="pytest", result="===== 3 passed in 0.5s =====")
    tool1["parent_span_id"] = "tr-1"
    tool2 = _fake_tool_span(cmd="git commit -m 'fix'", result=" 1 file changed, 5 insertions(+)")
    tool2["span_id"] = "sp-tool-2"
    tool2["parent_span_id"] = "tr-2"

    respx.post(f"{TRACE_BASE}/agents/spans/query").mock(
        side_effect=[
            respx.MockResponse(json={"spans": turns}),
            respx.MockResponse(json={"spans": [tool1]}),
            respx.MockResponse(json={"spans": [tool2]}),
        ]
    )

    result = cmd_score(_make_args())
    assert result == 0

    output = capsys.readouterr().out
    # session-level scores should appear once (1 session)
    assert output.count("efficiency.session") == 1
    assert output.count("implicit.abandonment") == 1


def _make_backfill_args(**kw):
    defaults = dict(
        entity="test-entity",
        project="test-project",
        start=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end=None,
        page_size=500,
        dry_run=True,
        force=False,
        verbose=False,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


@respx.mock
@patch("weave_agent_signals.client._get_api_key", return_value="test-key")
def test_backfill_paginates(_mock_key, capsys):
    """Backfill fetches multiple pages when a page is full."""
    page1 = [
        _fake_turn_span(
            trace_id=f"tr-{i}",
            conv_id=f"conv-{i}",
        )
        for i in range(3)
    ]
    # Override started_at so page2 cursor works
    for i, t in enumerate(page1):
        t["started_at"] = f"2026-07-09T12:{i:02d}:00Z"
    page2 = [_fake_turn_span(trace_id="tr-3", conv_id="conv-3")]
    page2[0]["started_at"] = "2026-07-09T12:03:00Z"

    respx.post(f"{TRACE_BASE}/agents/spans/query").mock(
        side_effect=[
            # page 1 (full — triggers next page)
            respx.MockResponse(json={"spans": page1}),
            # page 2 (partial — stops pagination)
            respx.MockResponse(json={"spans": page2}),
            # children hydration for 4 turns
            *[respx.MockResponse(json={"spans": []}) for _ in range(4)],
        ]
    )
    result = cmd_backfill(_make_backfill_args(page_size=3))
    assert result == 0
    output = capsys.readouterr().out
    # 4 turns × 1 efficiency score each + 4 sessions × 3 session scores = 16 scores
    assert "16 scores computed" in output


@respx.mock
@patch("weave_agent_signals.client._get_api_key", return_value="test-key")
def test_backfill_single_page(_mock_key, capsys):
    """When results fit in one page, no extra queries."""
    turns = [_fake_turn_span(trace_id="tr-1", conv_id="conv-1")]
    route = respx.post(f"{TRACE_BASE}/agents/spans/query").mock(
        side_effect=[
            respx.MockResponse(json={"spans": turns}),
            respx.MockResponse(json={"spans": [_fake_tool_span()]}),
        ]
    )
    result = cmd_backfill(_make_backfill_args(page_size=100))
    assert result == 0
    # Only 2 queries: 1 for turns + 1 for children (no second page)
    assert route.call_count == 2
