from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union


ENTITY = "mliu-wandb-weights-biases"
PROJECT = "agent-sessions"


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
    attributes: dict[str, Any] = field(default_factory=dict)


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

    @property
    def ref(self) -> str:
        return f"weave:///{ENTITY}/{PROJECT}/agent_turn/{self.trace_id}"


@dataclass
class SessionView:
    conversation_id: str
    turns: list[TurnSpan]
    config_version: str | None
    git_branch: str | None

    @property
    def ref(self) -> str:
        return f"weave:///{ENTITY}/{PROJECT}/agent_conversation/{self.conversation_id}"

    @property
    def total_tokens(self) -> int:
        return sum(t.input_tokens + t.output_tokens for t in self.turns)


@dataclass
class Score:
    scorer: str
    value: Union[float, bool]
    tags: list[str]
    confidence: float
    metadata: dict[str, Any]
    granularity: str

    def to_feedback_payload(self, ref: str, project_id: str) -> dict[str, Any]:
        rating = float(self.value) if isinstance(self.value, bool) else self.value
        return {
            "project_id": project_id,
            "weave_ref": ref,
            "feedback_type": f"weave_agent_signals.{self.scorer}",
            "payload": self.metadata,
            "scorer_ratings": {"_rating_": rating},
            "scorer_rating_reasons": {},
            "scorer_rating_confidences": {"_rating_": self.confidence},
            "scorer_tags": self.tags,
            "scorer_tag_reasons": {},
            "scorer_tag_confidences": {t: self.confidence for t in self.tags},
        }
