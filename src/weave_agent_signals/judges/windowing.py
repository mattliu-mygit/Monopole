"""Deterministic raw evidence rendering and bounded judging-window planning."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime

from weave_agent_signals.judges.digest import JudgeDigest
from weave_agent_signals.judges.tokens import TokenCounterName, count_tokens
from weave_agent_signals.models import SessionView, ToolSpan, TurnSpan
from weave_agent_signals.run_config import JudgingContextPolicy

WINDOW_PLAN_CONTRACT_VERSION = "3"
TOOL_TEXT_LIMIT = 12_000
_TOOL_TEXT_HEAD_CHARACTERS = 3_500
_TOOL_TEXT_TAIL_CHARACTERS = 3_500
_TOOL_DIAGNOSTIC_CHARACTERS = 3_500
_DIAGNOSTIC_PATTERN = re.compile(
    r"traceback|exception|error|failed|failure|warning|assert|exit(?:ed)?\s*(?:code|status)?",
    re.IGNORECASE,
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_WINDOW_FIELDS = frozenset(
    {
        "window_id",
        "index",
        "core_trace_ids",
        "raw_trace_ids",
        "raw_turn_digests",
        "raw_tokens",
    }
)


class WindowPlanInapplicable(RuntimeError):
    """Signal that valid raw evidence cannot fit the judge's context capacity."""

    reason = "insufficient_context_capacity"

    def __init__(self) -> None:
        super().__init__(self.reason)


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _diagnostic_excerpts(text: str, max_characters: int) -> str:
    excerpts: list[str] = []
    seen: set[str] = set()
    used = 0
    for match in _DIAGNOSTIC_PATTERN.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end < 0:
            line_end = len(text)
        line = text[line_start:line_end].strip()
        if len(line) > 500:
            relative = match.start() - line_start
            start = max(0, relative - 200)
            line = line[start : start + 500]
        if not line or line in seen:
            continue
        addition = len(line) + (1 if excerpts else 0)
        if used + addition > max_characters:
            break
        excerpts.append(line)
        seen.add(line)
        used += addition
    return "\n".join(excerpts)


def compact_tool_text(value: str) -> str:
    """Return a deterministic bounded judge view of one tool input or result."""

    if len(value) <= TOOL_TEXT_LIMIT:
        return value
    head = value[:_TOOL_TEXT_HEAD_CHARACTERS]
    tail = value[-_TOOL_TEXT_TAIL_CHARACTERS:]
    middle = value[_TOOL_TEXT_HEAD_CHARACTERS:-_TOOL_TEXT_TAIL_CHARACTERS]
    diagnostics = _diagnostic_excerpts(middle, _TOOL_DIAGNOSTIC_CHARACTERS)
    marker = (
        "[... compacted tool text; "
        f"original_chars={len(value)}; middle_chars={len(middle)}; "
        f"sha256={hashlib.sha256(value.encode('utf-8')).hexdigest()} ...]"
    )
    sections = [head, marker]
    if diagnostics:
        sections.extend(("[diagnostic excerpts]", diagnostics, "[/diagnostic excerpts]"))
    sections.append(tail)
    compacted = "\n".join(sections)
    if len(compacted) > TOOL_TEXT_LIMIT:  # Defensive bound for future marker changes.
        compacted = compacted[: TOOL_TEXT_LIMIT - _TOOL_TEXT_TAIL_CHARACTERS] + tail
    return compacted


def _render_tool(tool: ToolSpan, *, indent: str) -> list[str]:
    return [
        f"{indent}Tool [evidence_id={tool.span_id}]",
        f"{indent}  name: {_value(tool.tool_name)}",
        f"{indent}  status: {_value(tool.status_code)}",
        f"{indent}  started_at: {_value(_timestamp(tool.started_at))}",
        f"{indent}  ended_at: {_value(_timestamp(tool.ended_at))}",
        f"{indent}  arguments: {_value(compact_tool_text(tool.arguments))}",
        f"{indent}  result: {_value(compact_tool_text(tool.result))}",
    ]


def render_raw_turn(turn: TurnSpan, position: int) -> str:
    """Render all captured turn evidence deterministically without model identity."""

    lines = [
        f"## Turn {position} [evidence_id={turn.trace_id}]",
        f"trace_id: {_value(turn.trace_id)}",
        f"conversation_id: {_value(turn.conversation_id)}",
        f"status: {_value(turn.status_code)}",
        f"started_at: {_value(_timestamp(turn.started_at))}",
        f"ended_at: {_value(_timestamp(turn.ended_at))}",
        f"input_tokens: {turn.input_tokens}",
        f"output_tokens: {turn.output_tokens}",
        f"cache_read_tokens: {turn.cache_read_tokens}",
        f"config_version: {_value(turn.config_version)}",
        f"git_branch: {_value(turn.git_branch)}",
        f"effort_level: {_value(turn.effort_level)}",
        f"session_id: {_value(turn.session_id)}",
        f"steering_count: {turn.steering_count}",
        f"denial_count: {turn.denial_count}",
        f"tool_error_count: {turn.tool_error_count}",
        f"user_input: {_value(turn.user_input)}",
        f"assistant_output: {_value(turn.assistant_output)}",
        f"events ({len(turn.events)}):",
    ]
    for index, event in enumerate(turn.events, start=1):
        lines.extend(
            [
                f"  Event {index}",
                f"    name: {_value(event.name)}",
                f"    timestamp: {_value(_timestamp(event.timestamp))}",
            ]
        )

    lines.append(f"chat_spans ({len(turn.chat_spans)}):")
    for chat in turn.chat_spans:
        lines.extend(
            [
                f"  Chat [evidence_id={chat.span_id}]",
                f"    input_tokens: {chat.input_tokens}",
                f"    output_tokens: {chat.output_tokens}",
                f"    cache_read_tokens: {chat.cache_read_tokens}",
                f"    cache_creation_tokens: {chat.cache_creation_tokens}",
                f"    finish_reason: {_value(chat.finish_reason)}",
            ]
        )

    lines.append(f"tool_calls ({len(turn.tool_calls)}):")
    for tool in turn.tool_calls:
        lines.extend(_render_tool(tool, indent="  "))

    lines.append(f"subagents ({len(turn.subagents)}):")
    for subagent in turn.subagents:
        lines.extend(
            [
                f"  Subagent [evidence_id={subagent.span_id}]",
                f"    agent_type: {_value(subagent.agent_type)}",
                f"    internal_tool_calls_omitted: {len(subagent.tool_calls)}",
            ]
        )

    return "\n".join(lines)


def _turn_evidence_ids(turn: TurnSpan) -> tuple[str, ...]:
    ids = [turn.trace_id]
    ids.extend(chat.span_id for chat in turn.chat_spans)
    ids.extend(tool.span_id for tool in turn.tool_calls)
    for subagent in turn.subagents:
        ids.append(subagent.span_id)
    return tuple(ids)


def _validate_evidence_ids(evidence_ids: list[str], *, scope: str = "evidence") -> None:
    if any(
        not isinstance(evidence_id, str) or not evidence_id.strip() for evidence_id in evidence_ids
    ):
        raise ValueError(f"{scope} IDs must be nonblank")

    seen: set[str] = set()
    duplicates: list[str] = []
    for evidence_id in evidence_ids:
        if evidence_id in seen and evidence_id not in duplicates:
            duplicates.append(evidence_id)
        seen.add(evidence_id)
    if duplicates:
        raise ValueError(f"{scope} IDs must be globally unique: " + ", ".join(duplicates))


def _validate_session_evidence_ids(session: SessionView) -> None:
    evidence_ids = [
        evidence_id for turn in session.turns for evidence_id in _turn_evidence_ids(turn)
    ]
    _validate_evidence_ids(evidence_ids, scope="session evidence")


def _partition_cores(
    turn_count: int,
    range_tokens: Callable[[int, int], int],
    raw_budget: int,
    target_raw_tokens: int,
) -> list[tuple[int, int]]:
    cores: list[tuple[int, int]] = []
    start = 0
    while start < turn_count:
        if range_tokens(start, start) > raw_budget:
            raise ValueError("core turn exceeds the raw window budget")
        accepted_end = start
        while accepted_end + 1 < turn_count:
            current_tokens = range_tokens(start, accepted_end)
            candidate_tokens = range_tokens(start, accepted_end + 1)
            if candidate_tokens > raw_budget or current_tokens >= target_raw_tokens:
                break
            if abs(target_raw_tokens - candidate_tokens) > abs(target_raw_tokens - current_tokens):
                break
            accepted_end += 1
        cores.append((start, accepted_end))
        start = accepted_end + 1
    return cores


def _expand_raw_bounds(
    turn_count: int,
    range_tokens: Callable[[int, int], int],
    core_start: int,
    core_end: int,
    raw_budget: int,
    overlap_turns: int,
) -> tuple[int, int]:
    """Add neighboring turns when each expanded raw range remains within budget."""

    raw_start = core_start
    raw_end = core_end
    for _ in range(overlap_turns):
        if raw_start > 0 and range_tokens(raw_start - 1, raw_end) <= raw_budget:
            raw_start -= 1
        if raw_end < turn_count - 1 and range_tokens(raw_start, raw_end + 1) <= raw_budget:
            raw_end += 1
    return raw_start, raw_end


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _text_digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def build_window_plan(
    session: SessionView,
    policy: JudgingContextPolicy,
    model_limit: int,
    token_counter: TokenCounterName,
) -> dict[str, object]:
    """Build a deterministic overlap-aware fixed-point raw-window plan."""

    if isinstance(model_limit, bool) or not isinstance(model_limit, int) or model_limit <= 0:
        raise ValueError("model_limit must be a positive integer")
    _validate_session_evidence_ids(session)
    trace_ids = [turn.trace_id for turn in session.turns]

    input_cap = model_limit
    base_reserve = (
        policy.prompt_reserve_tokens + policy.output_reserve_tokens + policy.safety_reserve_tokens
    )
    rendered_turns = [
        render_raw_turn(turn, position) for position, turn in enumerate(session.turns, start=1)
    ]
    range_cache: dict[tuple[int, int], int] = {}

    def range_tokens(start: int, end: int) -> int:
        key = (start, end)
        if key not in range_cache:
            rendered_range = "\n\n".join(rendered_turns[start : end + 1])
            range_cache[key] = count_tokens(rendered_range, token_counter)
        return range_cache[key]

    turn_tokens = [count_tokens(rendered, token_counter) for rendered in rendered_turns]
    raw_turns = [
        {
            "trace_id": turn.trace_id,
            "position": position,
            "estimated_tokens": turn_tokens[position - 1],
            "raw_digest": _text_digest(rendered_turns[position - 1]),
        }
        for position, turn in enumerate(session.turns, start=1)
    ]

    tier_reserve = policy.capacity_reserve(model_limit)
    tier_target = policy.raw_window_target(model_limit)
    capacity_reserve = max(tier_reserve, base_reserve)
    if input_cap - capacity_reserve <= 0:
        raise WindowPlanInapplicable()

    if not session.turns:
        body: dict[str, object] = {
            "contract_version": WINDOW_PLAN_CONTRACT_VERSION,
            "conversation_id": session.conversation_id,
            "input_cap_tokens": input_cap,
            "raw_budget_tokens": input_cap - capacity_reserve,
            "target_raw_tokens": min(tier_target, input_cap - capacity_reserve),
            "chunk_count": 0,
            "overlap_turns": policy.overlap_turns,
            "token_counter": token_counter,
            "capacity_reserve_tokens": capacity_reserve,
            "merge_input_tokens": base_reserve,
            "raw_turns": raw_turns,
            "raw_coverage_trace_ids": [],
            "windows": [],
        }
        return {"plan_id": _canonical_digest(body), **body}

    chunk_count = 1
    seen_counts: set[int] = set()
    while True:
        if chunk_count in seen_counts:
            raise ValueError("window planning did not converge")
        seen_counts.add(chunk_count)
        protocol_overhead = base_reserve + max(0, chunk_count - 1) * policy.digest_max_tokens
        capacity_reserve = max(tier_reserve, protocol_overhead)
        raw_budget = input_cap - capacity_reserve
        if raw_budget <= 0:
            raise WindowPlanInapplicable()
        if any(tokens > raw_budget for tokens in turn_tokens):
            raise WindowPlanInapplicable()
        target_raw_tokens = min(tier_target, raw_budget)

        cores = _partition_cores(
            len(rendered_turns),
            range_tokens,
            raw_budget,
            target_raw_tokens,
        )
        planned_count = len(cores)
        if planned_count > policy.max_chunks:
            raise WindowPlanInapplicable()
        if planned_count == chunk_count:
            break
        chunk_count = planned_count

    merge_input_tokens = base_reserve + chunk_count * (
        policy.digest_max_tokens + policy.finding_max_tokens
    )
    if merge_input_tokens > input_cap:
        raise WindowPlanInapplicable()

    windows: list[dict[str, object]] = []
    covered_trace_ids: list[str] = []
    for index, (core_start, core_end) in enumerate(cores, start=1):
        core_tokens = range_tokens(core_start, core_end)
        raw_start, raw_end = _expand_raw_bounds(
            len(rendered_turns),
            range_tokens,
            core_start,
            core_end,
            min(raw_budget, max(target_raw_tokens, core_tokens)),
            policy.overlap_turns,
        )
        raw_tokens = range_tokens(raw_start, raw_end)
        core_trace_ids = trace_ids[core_start : core_end + 1]
        raw_trace_ids = trace_ids[raw_start : raw_end + 1]
        covered_trace_ids.extend(core_trace_ids)
        window_body: dict[str, object] = {
            "index": index,
            "core_trace_ids": core_trace_ids,
            "raw_trace_ids": raw_trace_ids,
            "raw_turn_digests": [
                raw_turns[position]["raw_digest"] for position in range(raw_start, raw_end + 1)
            ],
            "raw_tokens": raw_tokens,
        }
        windows.append({"window_id": _canonical_digest(window_body), **window_body})

    body = {
        "contract_version": WINDOW_PLAN_CONTRACT_VERSION,
        "conversation_id": session.conversation_id,
        "input_cap_tokens": input_cap,
        "raw_budget_tokens": raw_budget,
        "target_raw_tokens": target_raw_tokens,
        "chunk_count": chunk_count,
        "overlap_turns": policy.overlap_turns,
        "token_counter": token_counter,
        "capacity_reserve_tokens": capacity_reserve,
        "merge_input_tokens": merge_input_tokens,
        "raw_turns": raw_turns,
        "raw_coverage_trace_ids": covered_trace_ids,
        "windows": windows,
    }
    return {"plan_id": _canonical_digest(body), **body}


def render_raw_window(
    session: SessionView,
    window: Mapping[str, object],
    token_counter: TokenCounterName,
) -> JudgeDigest:
    """Render one planned window in session order with all visible evidence IDs."""

    if set(window) != _WINDOW_FIELDS:
        raise ValueError("window fields must exactly match the planned schema")

    window_id = window["window_id"]
    if not isinstance(window_id, str) or _SHA256_PATTERN.fullmatch(window_id) is None:
        raise ValueError("window_id must be a SHA-256 digest")
    index = window["index"]
    if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
        raise ValueError("window index must be a positive integer")
    core_trace_ids = window["core_trace_ids"]
    if (
        not isinstance(core_trace_ids, list)
        or not core_trace_ids
        or not all(isinstance(trace_id, str) and trace_id.strip() for trace_id in core_trace_ids)
    ):
        raise ValueError("window core_trace_ids must be a nonempty list of nonblank strings")
    raw_trace_ids = window.get("raw_trace_ids")
    if (
        not isinstance(raw_trace_ids, list)
        or not raw_trace_ids
        or not all(isinstance(trace_id, str) and trace_id.strip() for trace_id in raw_trace_ids)
    ):
        raise ValueError("window raw_trace_ids must be a nonempty list of nonblank strings")
    if len(raw_trace_ids) != len(set(raw_trace_ids)):
        raise ValueError("window raw_trace_ids must be unique")
    raw_turn_digests = window["raw_turn_digests"]
    if not isinstance(raw_turn_digests, list):
        raise ValueError("window raw_turn_digests must be a list")
    if not all(
        isinstance(digest, str) and _SHA256_PATTERN.fullmatch(digest) is not None
        for digest in raw_turn_digests
    ):
        raise ValueError("window raw_turn_digests must contain SHA-256 digests")
    raw_tokens = window["raw_tokens"]
    if isinstance(raw_tokens, bool) or not isinstance(raw_tokens, int) or raw_tokens < 0:
        raise ValueError("window raw_tokens must be a nonnegative integer")

    _validate_session_evidence_ids(session)
    positions = {turn.trace_id: index for index, turn in enumerate(session.turns, start=1)}
    turns = {turn.trace_id: turn for turn in session.turns}

    if len(core_trace_ids) != len(set(core_trace_ids)):
        raise ValueError("window core_trace_ids must be unique")
    missing_core_ids = [trace_id for trace_id in core_trace_ids if trace_id not in turns]
    if missing_core_ids:
        raise ValueError(
            "window core_trace_ids reference missing trace IDs: " + ", ".join(missing_core_ids)
        )
    core_positions = [positions[trace_id] for trace_id in core_trace_ids]
    if core_positions != sorted(core_positions):
        raise ValueError("window core_trace_ids must follow session order")
    if any(
        current != previous + 1 for previous, current in zip(core_positions, core_positions[1:])
    ):
        raise ValueError("window core_trace_ids must form a contiguous session range")

    missing = [trace_id for trace_id in raw_trace_ids if trace_id not in turns]
    if missing:
        raise ValueError("window references missing trace IDs: " + ", ".join(missing))
    expected_order = sorted(raw_trace_ids, key=positions.__getitem__)
    if raw_trace_ids != expected_order:
        raise ValueError("window raw_trace_ids must follow session order")
    raw_positions = [positions[trace_id] for trace_id in raw_trace_ids]
    if any(current != previous + 1 for previous, current in zip(raw_positions, raw_positions[1:])):
        raise ValueError("window raw_trace_ids must form a contiguous session range")

    allowed_raw_starts = {core_positions[0], max(1, core_positions[0] - 1)}
    allowed_raw_ends = {core_positions[-1], min(len(session.turns), core_positions[-1] + 1)}
    if raw_positions[0] not in allowed_raw_starts or raw_positions[-1] not in allowed_raw_ends:
        raise ValueError(
            "window raw_trace_ids must equal the core range plus at most one neighboring turn "
            "on each side"
        )

    selected_turns = [turns[trace_id] for trace_id in raw_trace_ids]
    rendered = [render_raw_turn(turn, positions[turn.trace_id]) for turn in selected_turns]
    expected_digests = [_text_digest(text) for text in rendered]
    if raw_turn_digests != expected_digests:
        raise ValueError("window raw_turn_digests do not match current session evidence")
    text = "\n\n".join(rendered)
    if raw_tokens != count_tokens(text, token_counter):
        raise ValueError("window raw_tokens do not match current session rendering")

    window_body = {
        "index": index,
        "core_trace_ids": core_trace_ids,
        "raw_trace_ids": raw_trace_ids,
        "raw_turn_digests": raw_turn_digests,
        "raw_tokens": raw_tokens,
    }
    if window_id != _canonical_digest(window_body):
        raise ValueError("window_id does not match canonical window body")

    evidence_ids = tuple(
        evidence_id for turn in selected_turns for evidence_id in _turn_evidence_ids(turn)
    )
    return JudgeDigest(text=text, evidence_ids=evidence_ids)
