"""Turn and session digest builder — formats data into structured judge prompts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from weave_agent_signals.models import SessionView, ToolSpan, TurnSpan

JUDGE_DIGEST_CONTRACT_VERSION = "3.0.0"
_TURN_USER_LIMIT = 1_200
_TURN_ASSISTANT_LIMIT = 2_000
_SESSION_MESSAGE_LIMIT = 400
_JUDGE_USER_PROMPT_TEMPLATE = (
    "## Scoring criteria\n\n{rubric_criteria}\n\n"
    "## {unit_heading} data\n\n{digest}\n\n"
    "Allowed evidence IDs: {allowed_evidence_ids}\n\n"
    "## Instructions\n\n"
    "Evaluate this {unit} against the criteria above. Respond with exactly one JSON object "
    "and no prose or Markdown fences. A scored verdict has this exact layout:\n"
    "{{\n"
    '  "schema_version": 3,\n'
    '  "status": "scored",\n'
    '  "score": 0.75,\n'
    '  "rationale": "Concise explanation grounded only in the supplied evidence.",\n'
    '  "evidence": [\n'
    "    {{\n"
    '      "id": {example_evidence_id},\n'
    '      "observations": ["Specific behavior supporting the verdict."]\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Every scored verdict must cite at least one allowed evidence ID. Cite each ID exactly "
    "once, group its distinct nonblank observations in the observations array, and choose "
    "only 0, 0.25, 0.5, 0.75, or 1 as the score. If the supplied evidence is "
    "insufficient, return:\n"
    "{{\n"
    '  "schema_version": 3,\n'
    '  "status": "insufficient_evidence",\n'
    '  "score": null,\n'
    '  "rationale": "Why the supplied evidence is insufficient.",\n'
    '  "evidence": []\n'
    "}}"
)


@dataclass(frozen=True)
class JudgeDigest:
    text: str
    evidence_ids: tuple[str, ...]


def _prompt_digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def judge_digest_contract_manifest() -> dict[str, object]:
    """Return deterministic digest and judge-message formatting semantics."""

    return {
        "limits": {
            "turn_user_chars": _TURN_USER_LIMIT,
            "turn_assistant_chars": _TURN_ASSISTANT_LIMIT,
            "session_message_chars": _SESSION_MESSAGE_LIMIT,
            "tool_argument_chars": 300,
            "tool_result_chars": 500,
            "session_tool_excerpt_chars": 200,
            "session_default_max_turns": 100,
            "prior_turn_default_window": 3,
        },
        "evidence_selection": "requested_then_terminal_head_context",
        "prompt_digests": {
            "judge_user": _prompt_digest(_JUDGE_USER_PROMPT_TEMPLATE),
        },
    }


def _bounded_text(text: str | None, max_chars: int) -> str:
    if not text or not text.strip():
        return "[not captured]"
    value = text.strip()
    if len(value) <= max_chars:
        return value
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}... [{omitted} chars omitted]"


def _summarize_tool(tool: ToolSpan, max_result: int = 500) -> str:
    result = tool.result or ""
    if len(result) > max_result:
        result = result[:max_result] + f"... ({len(result)} chars total)"
    args_str = tool.arguments or "{}"
    if len(args_str) > 300:
        args_str = args_str[:300] + "..."
    return f"  - {tool.tool_name} [{tool.status_code}]\n    args: {args_str}\n    result: {result}"


def build_turn_digest(turn: TurnSpan) -> JudgeDigest:
    """Build a structured text digest of a turn for judge consumption."""
    parts = []

    parts.append(f"Turn ID: {turn.trace_id} [evidence_id={turn.trace_id}]")
    parts.append(f"Status: {turn.status_code}")
    parts.append(f"Duration: {turn.started_at} → {turn.ended_at}")
    parts.append(
        f"Tokens: in={turn.input_tokens} out={turn.output_tokens} "
        f"cache_read={turn.cache_read_tokens}"
    )
    parts.append(f"User request: {_bounded_text(turn.user_input, _TURN_USER_LIMIT)}")
    parts.append(f"Assistant output: {_bounded_text(turn.assistant_output, _TURN_ASSISTANT_LIMIT)}")

    if turn.steering_count or turn.denial_count or turn.tool_error_count:
        parts.append(
            f"Events: steering={turn.steering_count} "
            f"denials={turn.denial_count} "
            f"tool_errors={turn.tool_error_count}"
        )

    all_tools = turn.tool_calls + [tc for sub in turn.subagents for tc in sub.tool_calls]

    if all_tools:
        parts.append(f"\nTool calls ({len(all_tools)}):")
        for tool in all_tools:
            parts.append(_summarize_tool(tool))

    if turn.subagents:
        parts.append(f"\nSubagents ({len(turn.subagents)}):")
        for sub in turn.subagents:
            parts.append(f"  - {sub.agent_type or 'unknown'} ({len(sub.tool_calls)} tool calls)")

    return JudgeDigest(text="\n".join(parts), evidence_ids=(turn.trace_id,))


def _turn_summary(turn: TurnSpan, idx: int, max_tools: int = 5) -> str:
    """Compact single-turn summary for session digest."""
    tools = turn.tool_calls + [tc for sub in turn.subagents for tc in sub.tool_calls]
    parts = [
        f"Turn {idx + 1} [evidence_id={turn.trace_id}]: "
        f"tokens={turn.input_tokens + turn.output_tokens} "
        f"tools={len(tools)}"
    ]
    parts.append(f"  user request: {_bounded_text(turn.user_input, _SESSION_MESSAGE_LIMIT)}")
    parts.append(
        f"  assistant output: {_bounded_text(turn.assistant_output, _SESSION_MESSAGE_LIMIT)}"
    )
    if turn.steering_count or turn.denial_count or turn.tool_error_count:
        parts[0] += (
            f" steering={turn.steering_count}"
            f" denials={turn.denial_count}"
            f" errors={turn.tool_error_count}"
        )

    if len(tools) <= max_tools:
        selected_tools = tools
    else:
        head_count = (max_tools + 1) // 2
        tail_count = max_tools - head_count
        selected_tools = tools[:head_count]
        if tail_count:
            selected_tools += tools[-tail_count:]

    for tool in selected_tools:
        cmd = tool.arguments or ""
        if len(cmd) > 200:
            cmd = cmd[:200] + "..."
        result = tool.result or ""
        if len(result) > 200:
            result = result[:200] + "..."
        parts.append(f"  {tool.tool_name} [{tool.status_code}]: {cmd}")
        if result:
            parts.append(f"    → {result}")

    omitted_tools = len(tools) - len(selected_tools)
    if omitted_tools:
        parts.append(f"  ... {omitted_tools} tool calls omitted")
    return "\n".join(parts)


def _context_turn_indexes(total_turns: int, max_turns: int) -> list[int]:
    limit = max(0, max_turns)
    if total_turns <= limit:
        return list(range(total_turns))
    if limit == 0:
        return []
    if limit == 1:
        return [total_turns - 1]

    head_count = limit // 2
    tail_count = limit - head_count
    return list(range(head_count)) + list(range(total_turns - tail_count, total_turns))


def _selected_turn_indexes(
    turns: list[TurnSpan],
    max_turns: int,
    evidence_trace_ids: list[str] | None,
) -> list[int]:
    requested = set(evidence_trace_ids or [])
    prioritized = [index for index, turn in enumerate(turns) if turn.trace_id in requested]
    target = min(len(turns), max(max(0, max_turns), len(prioritized)))
    if target == len(turns):
        return list(range(len(turns)))

    selected = set(prioritized)
    candidates = [len(turns) - 1, 0, *_context_turn_indexes(len(turns), target)]
    for index in candidates:
        if len(selected) >= target:
            break
        selected.add(index)
    return sorted(selected)


def _position_ranges(indexes: list[int]) -> str:
    ranges: list[str] = []
    start = previous = indexes[0]
    for index in indexes[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(str(start + 1) if start == previous else f"{start + 1}-{previous + 1}")
        start = previous = index
    ranges.append(str(start + 1) if start == previous else f"{start + 1}-{previous + 1}")
    return ", ".join(ranges)


def build_session_digest(
    session: SessionView,
    max_turns: int = 100,
    evidence_trace_ids: list[str] | None = None,
) -> JudgeDigest:
    """Build a structured session digest for judge consumption."""
    if evidence_trace_ids is not None:
        available_ids = {turn.trace_id for turn in session.turns}
        missing_ids = [
            trace_id
            for trace_id in dict.fromkeys(evidence_trace_ids)
            if trace_id not in available_ids
        ]
        if missing_ids:
            raise ValueError("missing requested evidence IDs: " + ", ".join(missing_ids))

    total_tokens = sum(t.input_tokens + t.output_tokens for t in session.turns)
    total_steering = sum(t.steering_count for t in session.turns)
    total_denials = sum(t.denial_count for t in session.turns)
    total_errors = sum(t.tool_error_count for t in session.turns)
    total_tools = sum(
        len(t.tool_calls) + sum(len(s.tool_calls) for s in t.subagents) for t in session.turns
    )

    parts = [
        "## Session overview",
        f"Session: {session.conversation_id[:16]}",
        f"Turns: {len(session.turns)}",
        f"Tokens: {total_tokens} total",
        f"Tools: {total_tools} total",
    ]

    if total_steering or total_denials or total_errors:
        parts.append(
            f"Events: steering={total_steering} denials={total_denials} tool_errors={total_errors}"
        )

    if session.config_version:
        parts.append(f"Config: {session.config_version}")

    selected_indexes = _selected_turn_indexes(
        session.turns,
        max_turns,
        evidence_trace_ids,
    )
    selected_ids = [session.turns[index].trace_id for index in selected_indexes]
    initial_turn = next((turn for turn in session.turns if turn.user_input), None)
    if initial_turn is None:
        parts.append("Initial request: [not captured]")
        visible_ids = tuple(selected_ids)
    else:
        parts.append(
            f"Initial request [evidence_id={initial_turn.trace_id}]: "
            f"{_bounded_text(initial_turn.user_input, _TURN_USER_LIMIT)}"
        )
        visible_ids = tuple(dict.fromkeys((initial_turn.trace_id, *selected_ids)))

    parts.extend(
        [
            "",
            "## Evidence selection",
            f"Selected {len(selected_indexes)} of {len(session.turns)} turns.",
            "Selected evidence IDs: " + (", ".join(selected_ids) if selected_ids else "[none]"),
        ]
    )

    selected_set = set(selected_indexes)
    omitted_indexes = [index for index in range(len(session.turns)) if index not in selected_set]
    if omitted_indexes:
        parts.append(
            f"{len(omitted_indexes)} turns omitted (positions {_position_ranges(omitted_indexes)})."
        )
    else:
        parts.append("No turns omitted.")

    parts.extend(["", "## Selected turn evidence", ""])
    for index in selected_indexes:
        parts.append(_turn_summary(session.turns[index], index))
        parts.append("")

    return JudgeDigest(text="\n".join(parts), evidence_ids=visible_ids)


def build_turn_digest_with_context(
    turn: TurnSpan,
    prior_turns: list[TurnSpan],
    window: int = 3,
) -> JudgeDigest:
    """Turn digest augmented with compact summaries of recent prior turns.

    Used by cross-turn rubrics (state_consistency, error_recovery) that need
    local context from the surrounding conversation.
    """
    parts = []
    recent = prior_turns[-window:] if len(prior_turns) > window else prior_turns
    if recent:
        parts.append(f"--- Prior turns ({len(recent)} of {len(prior_turns)} total) ---")
        offset = len(prior_turns) - len(recent)
        for i, t in enumerate(recent):
            parts.append(_turn_summary(t, offset + i, max_tools=2))
            parts.append("")
        parts.append("--- Current turn ---")

    current_digest = build_turn_digest(turn)
    parts.append(current_digest.text)
    return JudgeDigest(
        text="\n".join(parts),
        evidence_ids=tuple(item.trace_id for item in recent) + current_digest.evidence_ids,
    )


def build_judge_messages(
    rubric_system: str,
    rubric_criteria: str,
    digest: JudgeDigest,
    granularity: str = "turn",
) -> list[dict[str, str]]:
    """Build the messages array for a judge LLM call.

    ``granularity`` ("turn" or "session") frames the data section and the
    instruction so a session judge is not told to "evaluate this turn".
    """
    unit = "session" if granularity == "session" else "turn"
    return [
        {"role": "system", "content": rubric_system},
        {
            "role": "user",
            "content": _JUDGE_USER_PROMPT_TEMPLATE.format(
                rubric_criteria=rubric_criteria,
                unit_heading=unit.capitalize(),
                digest=digest.text,
                unit=unit,
                allowed_evidence_ids=json.dumps(list(digest.evidence_ids)),
                example_evidence_id=json.dumps(
                    digest.evidence_ids[0] if digest.evidence_ids else "<allowed evidence ID>"
                ),
            ),
        },
    ]
