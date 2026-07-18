from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from weave_agent_signals.runs.events import RunEvent, sanitize_event

NOW = datetime(2026, 7, 17, 12, tzinfo=UTC)


def test_judging_event_sanitizes_text_and_typed_details() -> None:
    event = sanitize_event(
        "judging",
        "judge_failed",
        "Bearer secret-token failed at https://user:pass@example.test/path",
        {
            "model": "wandb:model",
            "provider_status": 429,
            "elapsed_seconds": 1.25,
            "output_sha256": "a" * 64,
        },
        NOW,
    )

    assert event.message == ("Bearer [REDACTED] failed at https://[REDACTED]@example.test/path")
    assert event.details["provider_status"] == 429
    assert event.details["elapsed_seconds"] == 1.25


def test_reflection_and_scoring_use_distinct_detail_contracts() -> None:
    reflection = sanitize_event(
        "reflecting",
        "candidate_evaluated",
        "Candidate evaluated",
        {"acting_role": "proposal_evaluator", "score": 0.8},
        NOW,
    )
    assert reflection.details["score"] == 0.8

    with pytest.raises(ValueError, match="unexpected scoring event fields"):
        sanitize_event(
            "scoring",
            "turn_scored",
            "Turn scored",
            {"acting_role": "proposal_evaluator"},
            NOW,
        )


def test_event_rejects_unknown_or_nonfinite_details() -> None:
    with pytest.raises(ValueError, match="unexpected judging event fields"):
        sanitize_event("judging", "working", "Working", {"raw_output": "secret"}, NOW)
    with pytest.raises(ValueError, match="elapsed_seconds"):
        sanitize_event(
            "judging",
            "working",
            "Working",
            {"elapsed_seconds": float("nan")},
            NOW,
        )


def test_persisted_event_requires_store_assigned_sequence() -> None:
    draft = sanitize_event("scoring", "turn_scored", "Scored", {"scored": 1}, NOW)

    event = RunEvent(sequence=1, **draft.model_dump())

    assert event.sequence == 1
    with pytest.raises(ValidationError):
        RunEvent(sequence=0, **draft.model_dump())
