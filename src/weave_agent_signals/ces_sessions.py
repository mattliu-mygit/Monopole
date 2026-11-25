"""Monopole's sessions and turns, rebuilt from the Cases weave-agent-adapter writes to CES.

A session Case roots the session Trajectory S, whose member Cases each hold one user message,
assistant message, tool call, or compaction, and each subagent run is a child Trajectory. The
adapter's `specs/EVALUATIONS_CASE_CONTRACT.md` defines the fields. Nothing here reads CES;
`ces_client` reads the Cases and calls these functions.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from weave_agent_signals.models import (
    ChatSpan,
    SessionView,
    SpanEvent,
    SubagentSpan,
    ToolSpan,
    TurnSpan,
    parse_utc,
    resolve_trace_role,
)

_STATUS = {"completed": "OK", "aborted": "ERROR"}


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _span_id(case_id: str, turn_id: str | None) -> str:
    """A turn with no user prompt starts with a reply or tool call, whose Case ID is also the
    turn's ID. Judges need every evidence ID to be unique, so that span gets a suffix."""
    return f"{case_id}/span" if case_id == turn_id else case_id


def _tool_span(case: dict[str, Any], turn_id: str | None = None) -> ToolSpan:
    case_in, out = case["input"], case.get("output") or {}
    arguments = case_in.get("arguments")
    return ToolSpan(
        span_id=_span_id(case["case_id"], turn_id),
        tool_name=case_in.get("tool_name") or "",
        arguments="" if arguments is None else json.dumps(arguments, ensure_ascii=False),
        result=out.get("output") or out.get("error") or "",
        status_code="ERROR"
        if out.get("error") is not None or out.get("status") == "failed"
        else "OK"
        if out.get("status") == "completed"
        else "UNSET",
        started_at=parse_utc(case_in.get("started_at")) or datetime.now(timezone.utc),
        ended_at=parse_utc(case_in.get("ended_at")),
        exit_code=exit_code if type(exit_code := out.get("exit_code")) is int else None,
        permission_decision=decision
        if isinstance(decision := out.get("permission_decision"), str)
        else None,
    )


def _is_turn_case(case: dict) -> bool:
    """Whether a Case is a step of a turn: the session Case isn't, and a Case without a turn
    ID, which the adapter never writes, has no turn to join."""
    case_input = case.get("input") or {}
    return bool(case_input.get("turn_id")) and case_input.get("kind") != "session"


def _turn_fields(trajectory_id: str, first_case: dict, user_input: str | None) -> dict[str, Any]:
    """The TurnSpan fields a turn's first Case holds: its identity, settings, and totals."""
    first, totals = first_case["input"], first_case.get("output") or {}
    return {
        "trace_id": first_case["case_id"],
        "conversation_id": trajectory_id,
        "model": first.get("model"),
        "input_tokens": _int(totals.get("turn_input_tokens")),
        "output_tokens": _int(totals.get("turn_output_tokens")),
        "cache_read_tokens": _int(totals.get("turn_cache_read_tokens")),
        "status_code": _STATUS.get(totals.get("turn_status"), "UNSET"),
        "config_version": first.get("config_version"),
        "git_branch": first.get("git_branch"),
        "effort_level": first.get("effort"),
        "session_id": first.get("session_id"),
        "steering_count": _int(totals.get("turn_steering_count")),
        "denial_count": _int(totals.get("turn_denial_count")),
        "tool_error_count": _int(totals.get("turn_tool_error_count")),
        "user_input": user_input,
        "trace_role": resolve_trace_role(first.get("trace_role"), user_input),
    }


def _subagent_span(child: str, cases: list[dict]) -> SubagentSpan:
    return SubagentSpan(
        span_id=child,
        agent_type=next((c["input"].get("agent_id") for c in cases), None),
        tool_calls=[_tool_span(c) for c in cases if c["input"].get("kind") == "tool_call"],
        tokens=sum(
            _int((c.get("output") or {}).get("turn_input_tokens"))
            + _int((c.get("output") or {}).get("turn_output_tokens"))
            for c in cases
        ),
    )


class _SubagentTree:
    """Which subagent Trajectories each Case launched, indexed once per session read, so each
    Trajectory's view only walks the subagents below it."""

    def __init__(self, session_id: str, cases: list[dict], children: dict[str, list[dict]]):
        self.children = children
        self.position = {child: index for index, child in enumerate(children)}
        owner = {c["case_id"]: child for child, cs in children.items() for c in cs}
        launched = {
            child: case["case_id"]
            for case in [*cases, *(c for cs in children.values() for c in cs)]
            for child in case.get("child_trajectory_ids") or []
            if child in children
        }
        # (launching Case ID, subagent) for each Trajectory's direct subagents.
        self.launches: dict[str, list[tuple[str, str]]] = {}
        for child, case_id in launched.items():
            self.launches.setdefault(owner.get(case_id, session_id), []).append((case_id, child))

    def below(self, trajectory_id: str) -> dict[str, list[str]]:
        """Each Case of this Trajectory that launched subagents, with every subagent below it,
        nested ones included, in read order."""
        attached: dict[str, list[str]] = {}
        seen = {trajectory_id}
        for case_id, child in self.launches.get(trajectory_id, []):
            stack = [child]
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                attached.setdefault(case_id, []).append(current)
                stack += [kid for _, kid in self.launches.get(current, [])]
        for subagents in attached.values():
            subagents.sort(key=self.position.__getitem__)
        return attached


def session_view(
    trajectory_id: str, cases: list[dict], children: dict[str, list[dict]]
) -> SessionView:
    """Rebuild Monopole's turns from S's Cases and its subagents' Cases.

    A turn is the Cases sharing `turn_id`, ordered by `step_index`. Its first Case
    carries the turn totals, and its ID is the turn's identity, where Weave used a
    trace ID. A subagent's Cases attach to the turn whose tool call launched it.
    """
    return _view(trajectory_id, cases, _SubagentTree(trajectory_id, cases, children))


def session_tree(
    trajectory_id: str, cases: list[dict], children: dict[str, list[dict]]
) -> tuple[SessionView, list[SessionView]]:
    """S's view, and each subagent Trajectory below S as its own view with its own subagents."""
    tree = _SubagentTree(trajectory_id, cases, children)
    return _view(trajectory_id, cases, tree), [
        _view(child, child_cases, tree) for child, child_cases in children.items()
    ]


def _view(trajectory_id: str, cases: list[dict], tree: _SubagentTree) -> SessionView:
    members = [c for c in cases if _is_turn_case(c)]
    turns: dict[str, list[dict]] = {}
    for case in sorted(
        members,
        key=lambda c: (_int(c["input"].get("turn_index")), _int(c["input"].get("step_index"))),
    ):
        turns.setdefault(case["input"]["turn_id"], []).append(case)
    attached = tree.below(trajectory_id)

    built = []
    for steps in turns.values():
        first = steps[0]["input"]
        texts = [c["input"].get("text") for c in steps if c["input"]["kind"] == "user_message"]
        answers = [
            (c.get("output") or {}).get("text")
            for c in steps
            if c["input"]["kind"] == "assistant_message"
        ]
        stamps = [
            s
            for c in steps
            for s in (
                parse_utc(c["input"].get("started_at")),
                parse_utc(c["input"].get("ended_at")),
            )
            if s
        ]
        launched = sorted(
            {child for c in steps for child in attached.get(c["case_id"], [])},
            key=tree.position.__getitem__,
        )
        subagents = [_subagent_span(child, tree.children[child]) for child in launched]
        user_input = next((t for t in texts if t), None)
        built.append(
            TurnSpan(
                **_turn_fields(trajectory_id, steps[0], user_input),
                started_at=min(stamps) if stamps else datetime.now(timezone.utc),
                ended_at=max(stamps) if stamps else None,
                events=[
                    SpanEvent("compaction", parse_utc(c["input"].get("started_at")))
                    for c in steps
                    if c["input"]["kind"] == "compaction"
                ],
                tool_calls=[
                    _tool_span(c, steps[0]["case_id"])
                    for c in steps
                    if c["input"]["kind"] == "tool_call"
                ],
                chat_spans=[
                    ChatSpan(
                        span_id=_span_id(c["case_id"], steps[0]["case_id"]),
                        model=first.get("model") or "",
                        input_tokens=_int((c.get("output") or {}).get("input_tokens")),
                        output_tokens=_int((c.get("output") or {}).get("output_tokens")),
                        cache_read_tokens=_int((c.get("output") or {}).get("cache_read_tokens")),
                        cache_creation_tokens=_int(
                            (c.get("output") or {}).get("cache_write_tokens")
                        ),
                        finish_reason=(c.get("output") or {}).get("finish_reason"),
                    )
                    for c in steps
                    if c["input"]["kind"] == "assistant_message"
                ],
                subagents=subagents,
                assistant_output=next((a for a in reversed(answers) if a), None),
                case_ids=[c["case_id"] for c in steps],
            )
        )
    first_turn = built[0] if built else None
    return SessionView(
        conversation_id=trajectory_id,
        turns=built,
        config_version=first_turn.config_version if first_turn else None,
        git_branch=first_turn.git_branch if first_turn else None,
    )


def child_ids(cases: list[dict]) -> list[str]:
    """The subagent Trajectories these Cases launched, in launch order."""
    return [child for case in cases for child in case.get("child_trajectory_ids") or []]


def _turn_from_row(trajectory_id: str, row: dict) -> TurnSpan:
    """A turn from its row, the turn's first Case, which carries the turn totals. The turn's
    other steps aren't read, so it has no tool calls, model calls, or subagents, and it ends
    its recorded duration after it starts."""
    first, totals = row["input"], row.get("output") or {}
    started = parse_utc(first.get("started_at")) or datetime.now(timezone.utc)
    duration = totals.get("turn_duration_ms")
    user_input = first.get("text") if first.get("kind") == "user_message" else None
    return TurnSpan(
        **_turn_fields(trajectory_id, row, user_input),
        started_at=started,
        ended_at=started + timedelta(milliseconds=duration)
        if isinstance(duration, (int, float)) and not isinstance(duration, bool)
        else None,
        events=[],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
        assistant_output=None,
        case_ids=[row["case_id"]],
    )


def turns_from_rows(trajectory_id: str, rows: list[dict]) -> list[TurnSpan]:
    """A session's own turns, in order, from their rows. Subagent turns carry an agent ID."""
    own = [r for r in rows if _is_turn_case(r) and r["input"].get("agent_id") is None]
    own.sort(key=lambda r: _int(r["input"].get("turn_index")))
    return [_turn_from_row(trajectory_id, row) for row in own]
