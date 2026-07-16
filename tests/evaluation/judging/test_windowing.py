from __future__ import annotations

import hashlib
import json
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


def _tool(span_id: str) -> ToolSpan:
    return ToolSpan(
        span_id=span_id,
        tool_name="Bash",
        arguments="{}",
        result="result",
        status_code="OK",
        started_at=_ts(1),
        ended_at=_ts(1),
    )


def _self_consistent_window(
    session: SessionView,
    *,
    core_trace_ids: list[str],
    raw_trace_ids: list[str],
) -> dict[str, object]:
    positions = {turn.trace_id: position for position, turn in enumerate(session.turns, start=1)}
    turns = {turn.trace_id: turn for turn in session.turns}
    rendered = [render_raw_turn(turns[trace_id], positions[trace_id]) for trace_id in raw_trace_ids]
    body: dict[str, object] = {
        "index": 1,
        "core_trace_ids": core_trace_ids,
        "raw_trace_ids": raw_trace_ids,
        "raw_turn_digests": [
            "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest() for text in rendered
        ],
        "raw_tokens": estimate_tokens("\n\n".join(rendered)),
    }
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {"window_id": "sha256:" + hashlib.sha256(encoded).hexdigest(), **body}


def _rehash_window(window: dict[str, object], **updates: object) -> dict[str, object]:
    body = {key: value for key, value in window.items() if key != "window_id"}
    body.update(updates)
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {"window_id": "sha256:" + hashlib.sha256(encoded).hexdigest(), **body}


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


def test_window_plan_accepts_exact_raw_budget_equality():
    session = _session_with_rendered_turn_sizes([18_000, 18_000])
    policy = _small_policy(target_input_tokens=23_001)

    plan = build_window_plan(session, policy, model_limit=60_000)

    assert plan["raw_budget_tokens"] == 12_001
    assert plan["windows"][0]["raw_tokens"] == 12_001


@pytest.mark.parametrize(
    "turn",
    [
        _turn(" ", position=1),
        replace(
            _turn("t1", position=1),
            chat_spans=[ChatSpan("", "model", 1, 1, 0, 0, "stop")],
        ),
        replace(_turn("t1", position=1), tool_calls=[_tool(" ")]),
        replace(_turn("t1", position=1), subagents=[SubagentSpan("", "reviewer", [])]),
        replace(
            _turn("t1", position=1),
            subagents=[SubagentSpan("subagent-1", "reviewer", [_tool("")])],
        ),
    ],
)
def test_window_plan_rejects_blank_evidence_ids(turn: TurnSpan):
    session = SessionView("session-1", [turn], "config-1", "main")

    with pytest.raises(ValueError, match="session evidence IDs must be nonblank"):
        build_window_plan(session, _small_policy(), model_limit=60_000)


def test_window_plan_rejects_duplicate_evidence_ids_across_categories_and_turns():
    first = replace(_turn("duplicate", position=1), tool_calls=[_tool("tool-1")])
    second = replace(_turn("t2", position=2), tool_calls=[_tool("duplicate")])
    session = SessionView("session-1", [first, second], "config-1", "main")

    with pytest.raises(ValueError, match="session evidence IDs must be globally unique: duplicate"):
        build_window_plan(session, _small_policy(), model_limit=60_000)


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


def test_render_raw_window_rejects_stale_same_trace_evidence():
    session = _session_with_rendered_turn_sizes([18_000])
    plan = build_window_plan(session, _small_policy(), model_limit=60_000)
    turn = session.turns[0]
    changed = replace(turn, user_input="y" + (turn.user_input or "")[1:])
    stale_session = replace(session, turns=[changed])

    with pytest.raises(
        ValueError,
        match="window raw_turn_digests do not match current session evidence",
    ):
        render_raw_window(stale_session, plan["windows"][0])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"window_id": "sha256:" + "0" * 64}, "window_id does not match canonical window body"),
        (
            {"raw_turn_digests": ["sha256:" + "0" * 64]},
            "window raw_turn_digests do not match current session evidence",
        ),
        ({"raw_tokens": 99}, "window raw_tokens do not match current session rendering"),
    ],
)
def test_render_raw_window_rejects_tampered_planned_fields(
    mutation: dict[str, object],
    message: str,
):
    session = _session_with_rendered_turn_sizes([18_000])
    plan = build_window_plan(session, _small_policy(), model_limit=60_000)
    window = {**plan["windows"][0], **mutation}

    with pytest.raises(ValueError, match=message):
        render_raw_window(session, window)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"window_id": None}, "window fields must exactly match the planned schema"),
        ({"window_id": "not-a-digest"}, "window_id must be a SHA-256 digest"),
        ({"index": True}, "window index must be a positive integer"),
        ({"core_trace_ids": "t1"}, "window core_trace_ids must be a nonempty list"),
        ({"raw_trace_ids": [1]}, "window raw_trace_ids must be a nonempty list"),
        ({"raw_turn_digests": "not-a-list"}, "window raw_turn_digests must be a list"),
        (
            {"raw_turn_digests": ["not-a-digest"]},
            "window raw_turn_digests must contain SHA-256 digests",
        ),
        ({"raw_tokens": True}, "window raw_tokens must be a nonnegative integer"),
        ({"extra": "field"}, "window fields must exactly match the planned schema"),
    ],
)
def test_render_raw_window_rejects_missing_or_malformed_planned_fields(
    mutation: dict[str, object],
    message: str,
):
    session = _session_with_rendered_turn_sizes([18_000])
    plan = build_window_plan(session, _small_policy(), model_limit=60_000)
    window = {**plan["windows"][0], **mutation}
    if mutation == {"window_id": None}:
        del window["window_id"]

    with pytest.raises(ValueError, match=message):
        render_raw_window(session, window)


def test_render_raw_window_rejects_duplicate_raw_trace_ids():
    session = _session_with_rendered_turn_sizes([18_000])
    plan = build_window_plan(session, _small_policy(), model_limit=60_000)
    window = {**plan["windows"][0], "raw_trace_ids": ["t1", "t1"]}

    with pytest.raises(ValueError, match="window raw_trace_ids must be unique"):
        render_raw_window(session, window)


@pytest.mark.parametrize(
    ("core_trace_ids", "message"),
    [
        (["t1", "t1"], "window core_trace_ids must be unique"),
        (["missing"], "window core_trace_ids reference missing trace IDs: missing"),
        (["t2", "t1"], "window core_trace_ids must follow session order"),
        (["t1", "t3"], "window core_trace_ids must form a contiguous session range"),
    ],
)
def test_render_raw_window_rejects_self_consistent_invalid_core_geometry(
    core_trace_ids: list[str],
    message: str,
):
    session = _session_with_rendered_turn_sizes([3_000, 3_000, 3_000, 3_000])
    window = _self_consistent_window(
        session,
        core_trace_ids=core_trace_ids,
        raw_trace_ids=["t1", "t2", "t3", "t4"],
    )

    with pytest.raises(ValueError, match=message):
        render_raw_window(session, window)


@pytest.mark.parametrize(
    ("raw_trace_ids", "message"),
    [
        (["t2", "t1", "t3"], "window raw_trace_ids must follow session order"),
        (["t1", "t3"], "window raw_trace_ids must form a contiguous session range"),
        (
            ["t2"],
            "window raw_trace_ids must equal the core range plus one available neighboring turn",
        ),
    ],
)
def test_render_raw_window_rejects_self_consistent_invalid_raw_geometry(
    raw_trace_ids: list[str],
    message: str,
):
    session = _session_with_rendered_turn_sizes([3_000, 3_000, 3_000, 3_000])
    window = _self_consistent_window(
        session,
        core_trace_ids=["t2"],
        raw_trace_ids=raw_trace_ids,
    )

    with pytest.raises(ValueError, match=message):
        render_raw_window(session, window)


def test_render_raw_window_rejects_rehashed_bogus_raw_geometry():
    session = _session_with_rendered_turn_sizes([3_000, 3_000, 3_000, 3_000])
    valid = _self_consistent_window(
        session,
        core_trace_ids=["t2"],
        raw_trace_ids=["t1", "t2", "t3"],
    )
    window = _rehash_window(valid, raw_trace_ids=["missing"])

    with pytest.raises(ValueError, match="window references missing trace IDs: missing"):
        render_raw_window(session, window)
