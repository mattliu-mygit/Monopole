from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Union
from urllib.parse import quote


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

    def ref_for(self, entity: str = ENTITY, project: str = PROJECT) -> str:
        return f"weave:///{entity}/{project}/agent_turn/{self.trace_id}"

    @property
    def ref(self) -> str:
        return self.ref_for()


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
    def ref(self) -> str:
        return self.ref_for()

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
    reason: str = ""

    def to_feedback_payload(self, ref: str, project_id: str) -> dict[str, Any]:
        rating = float(self.value) if isinstance(self.value, bool) else self.value
        return {
            "project_id": project_id,
            "weave_ref": ref,
            "feedback_type": f"weave_agent_signals.{self.scorer}",
            "payload": {
                "scorer_version": "v1",
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "rating": rating,
                "confidence": self.confidence,
                "tags": self.tags,
                "reason": self.reason,
                "granularity": self.granularity,
                "details": self.metadata,
            },
        }
