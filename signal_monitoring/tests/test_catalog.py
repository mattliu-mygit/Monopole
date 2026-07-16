from importlib.metadata import version

import pytest
from pydantic import ValidationError

from weave_signal_monitoring.catalog import (
    ANCHORS,
    CATALOG,
    CATALOG_VERSION,
    RECOMMENDATION_THRESHOLD,
    SAMPLING_RATE,
    TURN_OP_NAME,
)
from weave_signal_monitoring.models import ScoreOutput


def test_v1_catalog_is_exact_and_high_recall():
    assert CATALOG_VERSION == "v1"
    assert ANCHORS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert RECOMMENDATION_THRESHOLD == 0.5
    assert SAMPLING_RATE == 1.0
    assert TURN_OP_NAME == "weave.genai.turn_ended"
    assert [signal.slug for signal in CATALOG] == [
        "user-frustration",
        "user-correction-or-rejection",
        "explicit-repeat-or-rephrase-cue",
        "stalled-or-deferred-response",
        "low-quality-response",
    ]
    assert len({signal.monitor_name for signal in CATALOG}) == 5
    assert all(signal.version == "v1" for signal in CATALOG)
    assert all(signal.model_name == f"{signal.monitor_name}-model" for signal in CATALOG)
    assert all("high recall" in signal.scoring_prompt.lower() for signal in CATALOG)
    assert all("{output_messages}" in signal.scoring_prompt for signal in CATALOG)
    assert all('"value": <float' in signal.scoring_prompt for signal in CATALOG)
    assert all("JSON number" in signal.scoring_prompt for signal in CATALOG)


def test_repeat_signal_explicitly_targets_user_cues():
    repeat = next(signal for signal in CATALOG if signal.slug == "explicit-repeat-or-rephrase-cue")

    assert "Evaluate only the user messages" in repeat.rubric
    assert "I already asked" in repeat.rubric


@pytest.mark.parametrize("rating", (0.0, 0.25, 0.5, 0.75, 1.0))
def test_score_output_accepts_only_closed_anchors(rating):
    assert ScoreOutput(rating=rating, reason="Visible evidence.").rating == rating


@pytest.mark.parametrize("rating", (-0.1, 0.1, 0.6, 1.1))
def test_score_output_rejects_non_anchor_values(rating):
    with pytest.raises(ValidationError):
        ScoreOutput(rating=rating, reason="Visible evidence.")


def test_score_output_requires_short_nonempty_reason():
    with pytest.raises(ValidationError):
        ScoreOutput(rating=0.5, reason="")
    with pytest.raises(ValidationError):
        ScoreOutput(rating=0.5, reason="x" * 241)


def test_score_output_strips_reason_whitespace():
    output = ScoreOutput(rating=0.5, reason="  Visible evidence.  ")
    assert output.reason == "Visible evidence."


def test_score_output_accepts_agent_signal_value_key():
    output = ScoreOutput.model_validate(
        {"value": 0.25, "confidence": 0.9, "reason": "Visible evidence."}
    )
    assert output.rating == 0.25
    assert output.confidence == 0.9


def test_runtime_weave_meets_complete_mode_floor():
    runtime = tuple(int(part) for part in version("weave").split(".")[:3])
    assert runtime >= (0, 52, 26)
