from __future__ import annotations

from datetime import datetime, timezone

from weave_agent_signals.models import (
    SessionView,
    SpanEvent,
    SubagentSpan,
    ToolSpan,
    TurnSpan,
)
from weave_agent_signals.scorers.efficiency import (
    detect_error_loops,
    detect_repeated_reads,
    score_session_efficiency,
    score_turn_efficiency,
    token_efficiency,
)


def _ts(h=12, m=0, s=0):
    return datetime(2026, 7, 9, h, m, s, tzinfo=timezone.utc)


def _tool(name="Bash", args="{}", result="ok", status="OK", t=None):
    return ToolSpan(
        span_id=f"sp-{id(args)}",
        tool_name=name,
        arguments=args,
        result=result,
        status_code=status,
        started_at=t or _ts(),
        ended_at=_ts(12, 1),
    )


def _turn(tool_calls, events=None):
    return TurnSpan(
        trace_id="t1",
        conversation_id="c1",
        started_at=_ts(),
        ended_at=_ts(12, 5),
        model="claude-opus-4",
        input_tokens=5000,
        output_tokens=1000,
        cache_read_tokens=2000,
        status_code="OK",
        config_version="abc",
        git_branch="main",
        effort_level="high",
        session_id="s1",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=events or [],
        tool_calls=tool_calls,
        chat_spans=[],
        subagents=[],
    )


# --- Error loop detection ---


def test_error_loop_detected():
    calls = [
        _tool("Bash", '{"command": "npm install"}', "error: ENOENT", "ERROR"),
        _tool("Bash", '{"command": "npm install"}', "error: ENOENT", "ERROR"),
        _tool("Bash", '{"command": "npm install"}', "error: ENOENT", "ERROR"),
    ]
    loops = detect_error_loops(calls)
    assert len(loops) == 1
    assert loops[0].tool_name == "Bash"
    assert loops[0].attempt_count == 3
    assert loops[0].all_failed is True


def test_no_loop_with_different_tools():
    calls = [
        _tool("Bash", '{"command": "npm install"}', "err", "ERROR"),
        _tool("Read", '{"path": "foo"}', "contents", "OK"),
        _tool("Bash", '{"command": "npm install"}', "err", "ERROR"),
    ]
    loops = detect_error_loops(calls)
    assert len(loops) == 0


def test_no_loop_when_mostly_passing():
    calls = [
        _tool("Bash", '{"command": "pytest"}', "passed", "OK"),
        _tool("Bash", '{"command": "pytest"}', "passed", "OK"),
        _tool("Bash", '{"command": "pytest"}', "failed", "ERROR"),
    ]
    loops = detect_error_loops(calls)
    assert len(loops) == 0


def test_loop_with_similar_args():
    calls = [
        _tool("Edit", '{"file": "foo.py", "old": "a", "new": "b"}', "err", "ERROR"),
        _tool("Edit", '{"file": "foo.py", "old": "a", "new": "c"}', "err", "ERROR"),
        _tool("Edit", '{"file": "foo.py", "old": "a", "new": "d"}', "err", "ERROR"),
    ]
    loops = detect_error_loops(calls)
    assert len(loops) == 1


# --- Repeated reads ---


def test_repeated_reads_detected():
    calls = [
        _tool("Read", '{"file_path": "/app/foo.py"}', "contents"),
        _tool("Read", '{"file_path": "/app/foo.py"}', "contents"),
    ]
    repeats = detect_repeated_reads(calls)
    assert len(repeats) == 1
    assert repeats[0].count == 2


def test_read_after_write_is_ok():
    calls = [
        _tool("Read", '{"file_path": "/app/foo.py"}', "v1"),
        _tool("Edit", '{"file_path": "/app/foo.py", "old": "a", "new": "b"}', "ok"),
        _tool("Read", '{"file_path": "/app/foo.py"}', "v2"),
    ]
    repeats = detect_repeated_reads(calls)
    assert len(repeats) == 0


def test_read_after_compaction_is_ok():
    calls = [
        _tool("Read", '{"file_path": "/app/foo.py"}', "contents", t=_ts(12, 0, 0)),
        _tool("Read", '{"file_path": "/app/foo.py"}', "contents", t=_ts(12, 1, 0)),
    ]
    events = [SpanEvent(name="compaction", timestamp=_ts(12, 0, 30))]
    repeats = detect_repeated_reads(calls, compaction_events=events)
    assert len(repeats) == 0


def test_different_files_not_repeated():
    calls = [
        _tool("Read", '{"file_path": "/app/foo.py"}', "a"),
        _tool("Read", '{"file_path": "/app/bar.py"}', "b"),
    ]
    repeats = detect_repeated_reads(calls)
    assert len(repeats) == 0


# --- Turn-level scoring ---


def test_efficient_turn_scores_high():
    calls = [
        _tool("Read", '{"file_path": "foo.py"}', "contents"),
        _tool("Edit", '{"file_path": "foo.py"}', "ok"),
        _tool("Bash", '{"command": "pytest"}', "5 passed"),
    ]
    score = score_turn_efficiency(_turn(calls))
    assert score.value > 0.9
    assert "efficient" in score.tags


def test_wasteful_turn_scores_low():
    calls = [
        _tool("Bash", '{"command": "npm install"}', "err", "ERROR"),
        _tool("Bash", '{"command": "npm install"}', "err", "ERROR"),
        _tool("Bash", '{"command": "npm install"}', "err", "ERROR"),
        _tool("Bash", '{"command": "npm install"}', "err", "ERROR"),
    ]
    score = score_turn_efficiency(_turn(calls))
    assert score.value < 0.5
    assert "error_loop" in score.tags


# --- Session-level efficiency ---


def _session(turns):
    return SessionView(
        conversation_id="c1",
        turns=turns,
        config_version="abc",
        git_branch="main",
    )


def test_session_efficiency_all_efficient():
    t1 = _turn(
        [
            _tool("Read", '{"file_path": "a.py"}', "contents"),
            _tool("Edit", '{"file_path": "a.py"}', "ok"),
        ]
    )
    t2 = _turn(
        [
            _tool("Bash", '{"command": "pytest"}', "5 passed"),
        ]
    )
    score = score_session_efficiency(_session([t1, t2]))
    assert score.scorer == "efficiency.session"
    assert score.value > 0.9
    assert score.granularity == "session"
    assert "efficient_session" in score.tags


def test_session_efficiency_with_waste():
    good = _turn([_tool("Bash", '{"command": "pytest"}', "ok")])
    bad = _turn(
        [
            _tool("Bash", '{"command": "npm install"}', "err", "ERROR"),
            _tool("Bash", '{"command": "npm install"}', "err", "ERROR"),
            _tool("Bash", '{"command": "npm install"}', "err", "ERROR"),
        ]
    )
    score = score_session_efficiency(_session([good, bad]))
    assert score.value < 1.0
    assert "has_error_loops" in score.tags
    assert score.metadata["total_error_loops"] == 1


def test_session_efficiency_empty():
    score = score_session_efficiency(_session([]))
    assert score.value == 1.0
    assert score.granularity == "session"


# --- Token efficiency ---


def test_token_efficiency_normal():
    t = _turn([], events=[])
    ratio = token_efficiency(t)
    # input=5000, output=1000 → ratio = 1000/5000 = 0.2
    assert ratio == 0.2


def test_token_efficiency_zero_input():
    t = _turn([], events=[])
    t.input_tokens = 0
    ratio = token_efficiency(t)
    assert ratio == 0.0


def test_token_efficiency_in_score_metadata():
    t = _turn([_tool("Bash", '{"command": "pytest"}', "ok")])
    score = score_turn_efficiency(t)
    assert "token_efficiency" in score.metadata


# --- Subagent scope tracking ---


def test_repeated_reads_across_scopes_not_flagged():
    """Main turn and subagent both reading the same file is NOT waste."""
    main_read = _tool("Read", '{"file_path": "/app/foo.py"}', "v1")
    sub_read = _tool("Read", '{"file_path": "/app/foo.py"}', "v1")
    subagent = SubagentSpan(span_id="sub-1", agent_type="Explore", tool_calls=[sub_read])

    t = _turn([main_read])
    t.subagents = [subagent]
    score = score_turn_efficiency(t)
    # Should NOT flag as repeated — different scopes
    assert "repeated_reads" not in score.tags


def test_repeated_reads_within_subagent_flagged():
    """Same subagent reading the same file twice IS waste."""
    r1 = _tool("Read", '{"file_path": "/app/foo.py"}', "v1")
    r2 = _tool("Read", '{"file_path": "/app/foo.py"}', "v1")
    subagent = SubagentSpan(span_id="sub-1", agent_type="Explore", tool_calls=[r1, r2])

    t = _turn([])
    t.subagents = [subagent]
    score = score_turn_efficiency(t)
    assert "repeated_reads" in score.tags
