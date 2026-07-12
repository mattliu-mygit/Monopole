"""Turn and session digest builder — formats data into structured judge prompts."""
from __future__ import annotations

from weave_agent_signals.models import SessionView, TurnSpan, ToolSpan


def _summarize_tool(tool: ToolSpan, max_result: int = 500) -> str:
    result = tool.result or ""
    if len(result) > max_result:
        result = result[:max_result] + f"... ({len(result)} chars total)"
    args_str = tool.arguments or "{}"
    if len(args_str) > 300:
        args_str = args_str[:300] + "..."
    return (
        f"  - {tool.tool_name} [{tool.status_code}]\n"
        f"    args: {args_str}\n"
        f"    result: {result}"
    )


def build_turn_digest(turn: TurnSpan) -> str:
    """Build a structured text digest of a turn for judge consumption."""
    parts = []

    parts.append(f"Turn ID: {turn.trace_id[:16]}")
    parts.append(f"Model: {turn.model or 'unknown'}")
    parts.append(f"Status: {turn.status_code}")
    parts.append(f"Duration: {turn.started_at} → {turn.ended_at}")
    parts.append(f"Tokens: in={turn.input_tokens} out={turn.output_tokens} "
                 f"cache_read={turn.cache_read_tokens}")

    if turn.steering_count or turn.denial_count or turn.tool_error_count:
        parts.append(
            f"Events: steering={turn.steering_count} "
            f"denials={turn.denial_count} "
            f"tool_errors={turn.tool_error_count}"
        )

    all_tools = turn.tool_calls + [
        tc for sub in turn.subagents for tc in sub.tool_calls
    ]

    if all_tools:
        parts.append(f"\nTool calls ({len(all_tools)}):")
        for tool in all_tools:
            parts.append(_summarize_tool(tool))

    if turn.subagents:
        parts.append(f"\nSubagents ({len(turn.subagents)}):")
        for sub in turn.subagents:
            parts.append(f"  - {sub.agent_type or 'unknown'} "
                         f"({len(sub.tool_calls)} tool calls)")

    return "\n".join(parts)


def _turn_summary(turn: TurnSpan, idx: int, max_tools: int = 5) -> str:
    """Compact single-turn summary for session digest."""
    tools = turn.tool_calls + [
        tc for sub in turn.subagents for tc in sub.tool_calls
    ]
    parts = [
        f"Turn {idx + 1}: {turn.model or '?'} "
        f"tokens={turn.input_tokens + turn.output_tokens} "
        f"tools={len(tools)}"
    ]
    if turn.steering_count or turn.denial_count or turn.tool_error_count:
        parts[0] += (
            f" steering={turn.steering_count}"
            f" denials={turn.denial_count}"
            f" errors={turn.tool_error_count}"
        )

    for tool in tools[:max_tools]:
        cmd = tool.arguments or ""
        if len(cmd) > 200:
            cmd = cmd[:200] + "..."
        result = tool.result or ""
        if len(result) > 200:
            result = result[:200] + "..."
        parts.append(f"  {tool.tool_name} [{tool.status_code}]: {cmd}")
        if result:
            parts.append(f"    → {result}")

    if len(tools) > max_tools:
        parts.append(f"  ... and {len(tools) - max_tools} more tool calls")
    return "\n".join(parts)


def build_session_digest(session: SessionView, max_turns: int = 30) -> str:
    """Build a structured session digest for judge consumption."""
    total_tokens = sum(
        t.input_tokens + t.output_tokens for t in session.turns
    )
    total_steering = sum(t.steering_count for t in session.turns)
    total_denials = sum(t.denial_count for t in session.turns)
    total_errors = sum(t.tool_error_count for t in session.turns)
    total_tools = sum(
        len(t.tool_calls) + sum(len(s.tool_calls) for s in t.subagents)
        for t in session.turns
    )

    parts = [
        f"Session: {session.conversation_id[:16]}",
        f"Turns: {len(session.turns)}",
        f"Tokens: {total_tokens} total",
        f"Tools: {total_tools} total",
    ]

    if total_steering or total_denials or total_errors:
        parts.append(
            f"Events: steering={total_steering} "
            f"denials={total_denials} "
            f"tool_errors={total_errors}"
        )

    if session.config_version:
        parts.append(f"Config: {session.config_version}")

    parts.append("")

    display_turns = session.turns[:max_turns]
    for i, turn in enumerate(display_turns):
        parts.append(_turn_summary(turn, i))
        parts.append("")

    if len(session.turns) > max_turns:
        parts.append(f"... {len(session.turns) - max_turns} more turns omitted")

    return "\n".join(parts)


def build_judge_messages(
    rubric_system: str,
    rubric_criteria: str,
    digest: str,
    granularity: str = "turn",
) -> list[dict[str, str]]:
    """Build the messages array for a judge LLM call.

    ``granularity`` ("turn" or "session") frames the data section and the
    instruction so a session judge is not told to "evaluate this turn".
    """
    unit = "session" if granularity == "session" else "turn"
    return [
        {"role": "system", "content": rubric_system},
        {"role": "user", "content": (
            f"## Scoring criteria\n\n{rubric_criteria}\n\n"
            f"## {unit.capitalize()} data\n\n{digest}\n\n"
            "## Instructions\n\n"
            f"Evaluate this {unit} against the criteria above. Respond with JSON:\n"
            '{"score": <float 0.0-1.0>, "rationale": "<1-2 sentences>"}'
        )},
    ]
