from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Union
from urllib.parse import quote

ENTITY = "weave-team"
PROJECT = "agent-sessions"

# Prefix for every feedback type this system writes: weave_agent_signals.<scorer>
FEEDBACK_PREFIX = "weave_agent_signals."
TRACE_ROLE_ATTRIBUTE = "weave_agent_signals.trace_role"


class TraceRole(StrEnum):
    AGENT_SESSION = "agent_session"
    SIGNAL_EVALUATION = "signal_evaluation"
    JUDGE_EVALUATION = "judge_evaluation"
    REFLECTION_EVALUATION = "reflection_evaluation"
    OTHER_SYSTEM = "other_system"


_LEGACY_TRACE_ROLE_PREFIXES = (
    ("## Scoring criteria", TraceRole.JUDGE_EVALUATION),
    ("You are an expert optimization assistant.", TraceRole.REFLECTION_EVALUATION),
    ("You are a high recall evaluation rater for an AI agent.", TraceRole.SIGNAL_EVALUATION),
)


def resolve_trace_role(explicit_role: object, user_input: str | None) -> TraceRole:
    """Resolve an immutable root role, failing safe for unknown explicit values."""
    if explicit_role is not None:
        try:
            return TraceRole(explicit_role)
        except (TypeError, ValueError):
            return TraceRole.OTHER_SYSTEM
    if user_input is not None:
        for prefix, role in _LEGACY_TRACE_ROLE_PREFIXES:
            if user_input.startswith(prefix):
                return role
    return TraceRole.AGENT_SESSION


@dataclass
class ToolSpan:
    span_id: str
    tool_name: str
    arguments: str
    result: str
    status_code: str
    started_at: datetime
    ended_at: datetime | None


@dataclass
class ChatSpan:
    span_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    finish_reason: str | None


@dataclass
class SubagentSpan:
    span_id: str
    agent_type: str | None
    tool_calls: list[ToolSpan] = field(default_factory=list)


@dataclass
class SpanEvent:
    name: str
    timestamp: datetime | None = None


@dataclass
class TurnSpan:
    trace_id: str
    conversation_id: str
    started_at: datetime
    ended_at: datetime | None
    model: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    status_code: str

    config_version: str | None
    git_branch: str | None
    effort_level: str | None
    session_id: str | None
    steering_count: int
    denial_count: int
    tool_error_count: int

    events: list[SpanEvent]
    tool_calls: list[ToolSpan]
    chat_spans: list[ChatSpan]
    subagents: list[SubagentSpan]

    user_input: str | None = None
    assistant_output: str | None = None
    trace_role: TraceRole = TraceRole.AGENT_SESSION

    def ref_for(self, entity: str = ENTITY, project: str = PROJECT) -> str:
        return f"weave:///{entity}/{project}/agent_turn/{self.trace_id}"


def session_is_evaluable(turns: list[TurnSpan]) -> bool:
    return bool(turns) and all(turn.trace_role is TraceRole.AGENT_SESSION for turn in turns)


def filter_evaluable_turns(turns: list[TurnSpan]) -> list[TurnSpan]:
    """Keep complete agent-session groups while preserving input turn order."""
    grouped: dict[str, list[TurnSpan]] = {}
    for turn in turns:
        grouped.setdefault(turn.conversation_id, []).append(turn)
    eligible = {
        conversation_id
        for conversation_id, session_turns in grouped.items()
        if session_is_evaluable(session_turns)
    }
    return [turn for turn in turns if turn.conversation_id in eligible]


@dataclass
class SessionView:
    conversation_id: str
    turns: list[TurnSpan]
    config_version: str | None
    git_branch: str | None

    def ref_for(self, entity: str = ENTITY, project: str = PROJECT) -> str:
        encoded_id = quote(self.conversation_id, safe="")
        return f"weave:///{entity}/{project}/agent_conversation/{encoded_id}"

    @property
    def total_tokens(self) -> int:
        return sum(t.input_tokens + t.output_tokens for t in self.turns)


@dataclass
class Score:
    scorer: str
    value: Union[float, bool]
    tags: list[str]
    metadata: dict[str, Any]
    granularity: str
    confidence: float | None = None
    reason: str = ""

    def stamp(self, *, config_version, git_branch, run_time) -> None:
        self.metadata.setdefault("config_version", config_version)
        self.metadata.setdefault("git_branch", git_branch)
        if run_time is not None:
            self.metadata.setdefault("turn_started_at", run_time.isoformat())

    def to_feedback_payload(self, ref: str, project_id: str) -> dict[str, Any]:
        rating = float(self.value) if isinstance(self.value, bool) else self.value
        payload = {
            "scorer_version": "v1",
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "rating": rating,
            "tags": self.tags,
            "reason": self.reason,
            "granularity": self.granularity,
            "details": self.metadata,
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return {
            "project_id": project_id,
            "weave_ref": ref,
            "feedback_type": f"{FEEDBACK_PREFIX}{self.scorer}",
            "payload": payload,
        }
