"""CES reads: session reconstruction from the adapter's Cases, paging, and retries."""

from __future__ import annotations

import httpx
import respx

from weave_agent_signals.ces_client import CesClient, session_view

BASE = "http://ces.test/v1/namespaces/wandb/scopes/scope-1"


def _case(case_id, kind, turn_id, step, children=(), output=None, **fields):
    return {
        "case_id": case_id,
        "input": {
            "kind": kind,
            "turn_id": turn_id,
            "turn_index": int(turn_id[-1]),
            "step_index": step,
            "started_at": f"2026-11-02T10:0{int(turn_id[-1])}:0{step}.000Z",
            "model": "gpt-6",
            "effort": "high",
            "git_branch": "main",
            "config_version": "cfg1",
            "session_id": "sess-1",
            "trace_role": "agent_session",
            **fields,
        },
        "output": output or {},
        "child_trajectory_ids": list(children),
    }


SESSION = [
    {
        "case_id": "c-session",
        "input": {"kind": "session", "session_id": "sess-1"},
        "output": {},
        "child_trajectory_ids": [],
    },
    _case(
        "c1",
        "user_message",
        "t1",
        0,
        text="count files",
        output={
            "turn_status": "completed",
            "turn_input_tokens": 300,
            "turn_output_tokens": 30,
            "turn_cache_read_tokens": 50,
            "turn_steering_count": 1,
            "turn_denial_count": 0,
            "turn_tool_error_count": 1,
        },
    ),
    _case(
        "c2",
        "assistant_message",
        "t1",
        1,
        output={
            "text": "Asking a helper.",
            "finish_reason": "tool_calls",
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_read_tokens": 50,
            "cache_write_tokens": 5,
        },
    ),
    _case(
        "c3",
        "tool_call",
        "t1",
        2,
        children=["traj-a"],
        tool_name="Task",
        arguments={"prompt": "count"},
        output={"status": "completed", "output": "3 files", "error": None},
    ),
    _case(
        "c4",
        "assistant_message",
        "t1",
        3,
        output={"text": "There are 3 files.", "finish_reason": "stop"},
    ),
    _case("c5", "user_message", "t2", 0, text="list them", output={"turn_status": "aborted"}),
    _case("c6", "compaction", "t2", 1, trigger="auto"),
]
AGENT_A = [
    _case("a1", "user_message", "k1", 0, text="count"),
    _case(
        "a2",
        "tool_call",
        "k1",
        1,
        children=["traj-b"],
        tool_name="Task",
        arguments={"prompt": "really"},
        output={"status": "completed", "output": "3"},
    ),
]
AGENT_B = [
    _case(
        "b1",
        "tool_call",
        "n1",
        0,
        tool_name="Bash",
        arguments={"command": "ls"},
        output={"status": "failed", "output": "denied", "error": "denied"},
    )
]


def test_a_session_becomes_turns_with_tools_chats_events_and_nested_subagents():
    view = session_view("traj-s", SESSION, {"traj-a": AGENT_A, "traj-b": AGENT_B})

    first, second = view.turns
    assert (view.conversation_id, view.config_version, view.git_branch) == (
        "traj-s",
        "cfg1",
        "main",
    )
    assert (
        first.trace_id,
        first.conversation_id,
        first.status_code,
        first.model,
        first.effort_level,
    ) == ("c1", "traj-s", "OK", "gpt-6", "high")
    assert (first.input_tokens, first.output_tokens, first.cache_read_tokens) == (300, 30, 50)
    assert (first.steering_count, first.tool_error_count, first.session_id) == (1, 1, "sess-1")
    assert (first.user_input, first.assistant_output) == ("count files", "There are 3 files.")
    assert [
        (t.span_id, t.tool_name, t.arguments, t.result, t.status_code) for t in first.tool_calls
    ] == [("c3", "Task", '{"prompt": "count"}', "3 files", "OK")]
    assert [(c.span_id, c.finish_reason, c.cache_creation_tokens) for c in first.chat_spans] == [
        ("c2", "tool_calls", 5),
        ("c4", "stop", 0),
    ]
    assert [(s.span_id, [t.status_code for t in s.tool_calls]) for s in first.subagents] == [
        ("traj-a", ["OK"]),
        ("traj-b", ["ERROR"]),
    ]
    assert (second.trace_id, second.status_code, [e.name for e in second.events]) == (
        "c5",
        "ERROR",
        ["compaction"],
    )
    assert second.subagents == []


@respx.mock
def test_read_session_pages_and_follows_subagents_with_retries():
    respx.get(f"{BASE}/trajectories/traj-s/cases").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, json={"cases": SESSION[:3], "next_cursor": "p2"}),
            httpx.Response(200, json={"cases": SESSION[3:], "next_cursor": None}),
        ]
    )
    respx.get(f"{BASE}/trajectories/traj-a/cases").mock(
        return_value=httpx.Response(200, json={"cases": AGENT_A})
    )
    respx.get(f"{BASE}/trajectories/traj-b/cases").mock(
        return_value=httpx.Response(200, json={"cases": AGENT_B})
    )
    slept = []
    client = CesClient("http://ces.test", "scope-1", api_key="k", sleep=slept.append)

    view = client.read_session("traj-s")

    assert [t.trace_id for t in view.turns] == ["c1", "c5"]
    assert [s.span_id for s in view.turns[0].subagents] == ["traj-a", "traj-b"]
    assert slept == [1.0]
