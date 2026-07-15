from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Rating = Literal[0.0, 0.25, 0.5, 0.75, 1.0]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScoreOutput(StrictModel):
    rating: Rating
    reason: str = Field(min_length=1, max_length=240)

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reason must contain visible evidence")
        return value


class SignalIdentity(StrictModel):
    slug: str
    version: str
    monitor_name: str
    scorer_ref: str


class FeedbackRecord(StrictModel):
    id: str
    weave_ref: str
    runnable_ref: str
    created_at: datetime
    output: ScoreOutput


class TurnRecord(StrictModel):
    turn_id: str
    conversation_id: str
    started_at: datetime
    ended_at: datetime | None = None
    display_name: str | None = None
    user_request: str | None = None


class SignalEvidence(StrictModel):
    signal: str
    version: str
    rating: Rating
    reason: str
    turn_id: str
    turn_started_at: datetime


class FlaggedConversation(StrictModel):
    conversation_id: str
    display_name: str
    started_at: datetime
    last_activity_at: datetime
    lowest_rating: Rating
    signals: list[SignalEvidence]
    triggering_turn_ids: list[str]
    wandb_url: str


class HydrationWindow(StrictModel):
    since: datetime
    until: datetime


class HydrationEnvelope(StrictModel):
    schema_version: Literal[1] = 1
    generated_at: datetime
    entity: str
    project: str
    window: HydrationWindow
    conversations: list[FlaggedConversation]
