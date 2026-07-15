from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from weave_agent_signals.models import Score, SessionView, SpanEvent, ToolSpan, TurnSpan


@dataclass
class ErrorLoop:
    tool_name: str
    attempt_count: int
    similarity: float
    all_failed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "attempt_count": self.attempt_count,
            "similarity": round(self.similarity, 2),
            "all_failed": self.all_failed,
        }


@dataclass
class RepeatedRead:
    path: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "count": self.count}


def _arg_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _group_consecutive_by_tool(calls: list[ToolSpan]) -> list[list[ToolSpan]]:
    if not calls:
        return []
    groups = [[calls[0]]]
    for tc in calls[1:]:
        if tc.tool_name == groups[-1][0].tool_name:
            groups[-1].append(tc)
        else:
            groups.append([tc])
    return groups


def detect_error_loops(calls: list[ToolSpan], threshold: int = 3) -> list[ErrorLoop]:
    groups = _group_consecutive_by_tool(calls)
    loops = []
    for group in groups:
        if len(group) < threshold:
            continue

        sims = []
        for i in range(1, len(group)):
            sims.append(_arg_similarity(group[0].arguments, group[i].arguments))
        avg_sim = sum(sims) / len(sims) if sims else 0

        if avg_sim < 0.5:
            continue

        failed = [t for t in group if t.status_code != "OK"]
        if len(failed) / len(group) < 0.6:
            continue

        loops.append(
            ErrorLoop(
                tool_name=group[0].tool_name,
                attempt_count=len(group),
                similarity=avg_sim,
                all_failed=len(failed) == len(group),
            )
        )
    return loops


def _extract_path(args_json: str) -> str | None:
    try:
        args = json.loads(args_json)
        return args.get("file_path") or args.get("path")
    except (json.JSONDecodeError, TypeError):
        return None


def detect_repeated_reads(
    calls: list[ToolSpan],
    compaction_events: list[SpanEvent] | None = None,
) -> list[RepeatedRead]:
    compaction_times = []
    if compaction_events:
        compaction_times = [e.timestamp for e in compaction_events if e.timestamp]

    reads: dict[str, list[ToolSpan]] = {}
    writes: set[str] = set()

    for tc in calls:
        # reset tracking at compaction boundaries
        if compaction_times:
            for ct in compaction_times:
                if tc.started_at >= ct:
                    reads.clear()
                    writes.clear()
                    compaction_times = [t for t in compaction_times if t > ct]
                    break

        if tc.tool_name == "Read":
            path = _extract_path(tc.arguments)
            if path and path not in writes:
                reads.setdefault(path, []).append(tc)
            if path:
                writes.discard(path)
        elif tc.tool_name in ("Edit", "Write"):
            path = _extract_path(tc.arguments)
            if path:
                writes.add(path)
                reads.pop(path, None)

    return [
        RepeatedRead(path=path, count=len(spans))
        for path, spans in reads.items()
        if len(spans) >= 2
    ]


def token_efficiency(turn: TurnSpan) -> float:
    if turn.input_tokens == 0:
        return 0.0
    return round(turn.output_tokens / turn.input_tokens, 4)


def score_turn_efficiency(turn: TurnSpan) -> Score:
    compaction_events = [e for e in turn.events if e.name == "compaction"]
    # Per-scope detection: main turn + each subagent independently
    loops = detect_error_loops(turn.tool_calls)
    repeats = detect_repeated_reads(turn.tool_calls, compaction_events)
    for sub in turn.subagents:
        loops.extend(detect_error_loops(sub.tool_calls))
        repeats.extend(detect_repeated_reads(sub.tool_calls))

    all_tool_calls = turn.tool_calls + [tc for sub in turn.subagents for tc in sub.tool_calls]
    loop_waste = sum(lp.attempt_count - 1 for lp in loops)
    repeat_waste = sum(r.count - 1 for r in repeats)
    total_calls = len(all_tool_calls) or 1
    waste_ratio = min((loop_waste + repeat_waste) / total_calls, 1.0)

    tags = []
    if loops:
        tags.append("error_loop")
    if repeats:
        tags.append("repeated_reads")
    if not tags:
        tags.append("efficient")

    parts = []
    if loops:
        parts.append(f"{len(loops)} error loop(s)")
    if repeats:
        parts.append(f"{len(repeats)} repeated read(s)")
    reason = "; ".join(parts) if parts else "no waste detected"

    return Score(
        scorer="efficiency",
        value=round(1.0 - waste_ratio, 3),
        tags=tags,
        confidence=0.9,
        metadata={
            "error_loops": [lp.to_dict() for lp in loops],
            "repeated_reads": [r.to_dict() for r in repeats],
            "waste_ratio": round(waste_ratio, 3),
            "tool_call_count": len(all_tool_calls),
            "token_efficiency": token_efficiency(turn),
        },
        granularity="turn",
        reason=reason,
    )


def score_session_efficiency(session: SessionView) -> Score:
    turn_scores = [score_turn_efficiency(t) for t in session.turns]
    avg_value = sum(s.value for s in turn_scores) / len(turn_scores) if turn_scores else 1.0

    total_loops = sum(len(s.metadata["error_loops"]) for s in turn_scores)
    total_repeats = sum(len(s.metadata["repeated_reads"]) for s in turn_scores)

    tags = []
    if total_loops > 0:
        tags.append("has_error_loops")
    if total_repeats > 0:
        tags.append("has_repeated_reads")
    if not tags:
        tags.append("efficient_session")

    parts = []
    if total_loops > 0:
        parts.append(f"{total_loops} error loop(s)")
    if total_repeats > 0:
        parts.append(f"{total_repeats} repeated read(s)")
    reason = f"{len(session.turns)} turns, " + ("; ".join(parts) if parts else "no waste detected")

    return Score(
        scorer="efficiency.session",
        value=round(avg_value, 3),
        tags=tags,
        confidence=0.85,
        metadata={
            "total_error_loops": total_loops,
            "total_repeated_reads": total_repeats,
            "turn_count": len(session.turns),
            "avg_turn_efficiency": round(avg_value, 3),
        },
        granularity="session",
        reason=reason,
    )
