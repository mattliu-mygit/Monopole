from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from weave_agent_signals.judges.windowing import (
    build_window_plan,
    estimate_tokens,
    render_raw_turn,
    render_raw_window,
)
from weave_agent_signals.models import (
    ChatSpan,
    SessionView,
    SpanEvent,
    SubagentSpan,
    ToolSpan,
    TurnSpan,
)
from weave_agent_signals.run_config import JudgingContextPolicy


def _ts(position: int = 0) -> datetime:
    return datetime(2026, 7, 9, 12, tzinfo=timezone.utc) + timedelta(minutes=position)


def _turn(trace_id: str, *, position: int, text: str = "") -> TurnSpan:
    started_at = _ts(position)
    return TurnSpan(
        trace_id=trace_id,
        conversation_id="session-1",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=30),
        model="secret-generating-model",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=10,
        status_code="OK",
        config_version="config-1",
        git_branch="main",
        effort_level="high",
        session_id="session-id-1",
        steering_count=1,
        denial_count=0,
        tool_error_count=0,
        events=[SpanEvent(name="user_steering", timestamp=started_at + timedelta(seconds=1))],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
        user_input=text,
        assistant_output=f"assistant-{trace_id}",
    )


def _session_with_rendered_turn_sizes(sizes: list[int]) -> SessionView:
    turns: list[TurnSpan] = []
    for index, size in enumerate(sizes, start=1):
        turn = _turn(f"t{index}", position=index)
        base_size = len(render_raw_turn(turn, index).encode("utf-8"))
        if size < base_size:
            raise ValueError(f"requested size {size} is smaller than renderer overhead {base_size}")
        turn = replace(turn, user_input="x" * (size - base_size))
        assert len(render_raw_turn(turn, index).encode("utf-8")) == size
        turns.append(turn)
    return SessionView("session-1", turns, "config-1", "main")


def _session_with_text(text: str) -> SessionView:
    return SessionView("session-1", [_turn("t1", position=1, text=text)], "config-1", "main")


def _small_policy(**overrides: int) -> JudgingContextPolicy:
    values = {
        "target_input_tokens": 34_000,
        "prompt_reserve_tokens": 3_000,
        "output_reserve_tokens": 3_000,
        "safety_reserve_tokens": 3_000,
        "digest_max_tokens": 2_000,
        "finding_max_tokens": 1_000,
        "max_chunks": 10,
    }
    values.update(overrides)
    return JudgingContextPolicy(**values)


def test_estimate_tokens_uses_versioned_conservative_utf8_bytes_estimator():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 2
    assert estimate_tokens("é") == 1


def test_raw_turn_renders_complete_evidence_without_model_identity():
    turn = _turn("trace-1", position=1, text="complete user request")
    tool = ToolSpan(
        span_id="tool-1",
        tool_name="Bash",
        arguments='{"command":"printf complete-arguments"}',
        result="complete-tool-result",
        status_code="ERROR",
        started_at=_ts(1),
        ended_at=_ts(1) + timedelta(seconds=1),
    )
    subtool = replace(tool, span_id="tool-2", tool_name="Read", result="subagent-result")
    turn = replace(
        turn,
        assistant_output="complete assistant response",
        tool_calls=[tool],
        chat_spans=[ChatSpan("chat-1", "secret-chat-model", 10, 20, 3, 4, "stop")],
        subagents=[SubagentSpan("subagent-1", "reviewer", [subtool])],
    )

    rendered = render_raw_turn(turn, 1)

    for value in (
        "trace-1",
        "tool-1",
        "tool-2",
        "chat-1",
        "subagent-1",
        "complete user request",
        "complete assistant response",
        "complete-arguments",
        "complete-tool-result",
        "subagent-result",
        "user_steering",
        "ERROR",
    ):
        assert value in rendered
    assert "secret-generating-model" not in rendered
    assert "secret-chat-model" not in rendered


def test_window_plan_covers_every_turn_raw_and_overlaps_one_turn():
    session = _session_with_rendered_turn_sizes([18_000, 18_000, 18_000, 18_000])
    plan = build_window_plan(session, _small_policy(), model_limit=60_000)
    cores = [window["core_trace_ids"] for window in plan["windows"]]
    assert [trace for core in cores for trace in core] == ["t1", "t2", "t3", "t4"]
    assert plan["windows"][0]["raw_trace_ids"][-1] == "t3"
    assert plan["windows"][1]["raw_trace_ids"][0] == "t2"
    assert plan["raw_coverage_trace_ids"] == ["t1", "t2", "t3", "t4"]
    assert set(plan["windows"][0]["raw_trace_ids"]) & set(plan["windows"][1]["raw_trace_ids"]) == {
        "t2",
        "t3",
    }


def test_window_plan_preserves_session_order_and_is_deterministic():
    session = _session_with_rendered_turn_sizes([18_000, 18_000, 18_000, 18_000])

    first = build_window_plan(session, _small_policy(), model_limit=60_000)
    second = build_window_plan(session, _small_policy(), model_limit=60_000)

    assert first == second
    assert first["plan_id"].startswith("sha256:")
    assert [trace_id for window in first["windows"] for trace_id in window["core_trace_ids"]] == [
        "t1",
        "t2",
        "t3",
        "t4",
    ]


def test_window_plan_id_changes_when_same_size_raw_evidence_changes():
    session = _session_with_rendered_turn_sizes([18_000])
    original_turn = session.turns[0]
    changed_turn = replace(
        original_turn,
        user_input="y" + (original_turn.user_input or "")[1:],
    )
    changed_session = replace(session, turns=[changed_turn])

    original = build_window_plan(session, _small_policy(), model_limit=60_000)
    changed = build_window_plan(changed_session, _small_policy(), model_limit=60_000)

    assert original["plan_id"] != changed["plan_id"]
    assert original["windows"][0]["window_id"] != changed["windows"][0]["window_id"]


def test_window_plan_rejects_a_turn_that_cannot_fit_with_reserves():
    session = _session_with_text("x" * 400_000)
    with pytest.raises(ValueError, match="single turn exceeds the raw window budget"):
        build_window_plan(session, _small_policy(), model_limit=60_000)


def test_window_plan_budgets_the_final_raw_window_separators():
    session = _session_with_rendered_turn_sizes([18_000, 18_000])
    policy = _small_policy(target_input_tokens=23_000)

    with pytest.raises(ValueError, match="required one-turn overlap"):
        build_window_plan(session, policy, model_limit=60_000)


def test_window_plan_rejects_empty_raw_capacity():
    session = SessionView("session-1", [], "config-1", "main")
    with pytest.raises(ValueError, match="leave no raw window capacity"):
        build_window_plan(session, _small_policy(), model_limit=9_000)


def test_window_plan_rejects_more_than_max_chunks():
    session = _session_with_rendered_turn_sizes([18_000, 18_000, 18_000, 18_000])
    with pytest.raises(ValueError, match="maximum chunk count"):
        build_window_plan(session, _small_policy(max_chunks=1), model_limit=60_000)


def test_window_plan_rejects_worst_case_merge_input_over_model_cap():
    session = _session_with_rendered_turn_sizes([18_000, 18_000, 18_000, 18_000])
    policy = _small_policy(
        target_input_tokens=30_000,
        prompt_reserve_tokens=1_000,
        output_reserve_tokens=1_000,
        safety_reserve_tokens=1_000,
        digest_max_tokens=4_000,
        finding_max_tokens=10_000,
    )

    with pytest.raises(ValueError, match="worst-case merge input exceeds the input cap"):
        build_window_plan(session, policy, model_limit=30_000)


def test_render_raw_window_uses_planned_order_and_visible_evidence_ids():
    session = _session_with_rendered_turn_sizes([18_000, 18_000, 18_000, 18_000])
    plan = build_window_plan(session, _small_policy(), model_limit=60_000)

    digest = render_raw_window(session, plan["windows"][0])

    assert digest.evidence_ids == tuple(plan["windows"][0]["raw_trace_ids"])
    assert digest.text.index("evidence_id=t1") < digest.text.index("evidence_id=t2")
    assert digest.text.index("evidence_id=t2") < digest.text.index("evidence_id=t3")
