from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weave_agent_signals.models import SessionView, TurnSpan
from weave_agent_signals.scorers.implicit import (
    correction_density,
    is_abandoned,
    score_session_implicit,
)


def _ts(h=12, m=0, s=0):
    return datetime(2026, 7, 9, h, m, s, tzinfo=timezone.utc)


def _turn(
    trace_id="t1",
    steering=0,
    denials=0,
    errors=0,
    status="OK",
):
    return TurnSpan(
        trace_id=trace_id,
        conversation_id="c1",
        started_at=_ts(),
        ended_at=_ts(12, 5),
        model="claude-opus-4",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        status_code=status,
        config_version="abc",
        git_branch="main",
        effort_level="high",
        session_id="s1",
        steering_count=steering,
        denial_count=denials,
        tool_error_count=errors,
        events=[],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
    )


def _session(turns, conv_id="c1"):
    return SessionView(
        conversation_id=conv_id,
        turns=turns,
        config_version="abc",
        git_branch="main",
    )


# --- Abandonment ---


def test_abandoned_with_error_end():
    turns = [_turn("t1"), _turn("t2"), _turn("t3", status="ERROR")]
    assert is_abandoned(_session(turns)) is True


def test_not_abandoned_short_session():
    assert is_abandoned(_session([_turn("t1")])) is False


def test_not_abandoned_clean_end():
    turns = [_turn("t1"), _turn("t2"), _turn("t3")]
    assert is_abandoned(_session(turns)) is False


def test_abandoned_empty_session():
    assert is_abandoned(_session([])) is True


# --- Correction density ---


def test_correction_density_zero():
    turns = [_turn("t1"), _turn("t2"), _turn("t3")]
    assert correction_density(_session(turns)) == 0.0


def test_correction_density_half():
    turns = [_turn("t1", steering=1), _turn("t2"), _turn("t3", denials=1), _turn("t4")]
    assert correction_density(_session(turns)) == 0.5


def test_correction_density_single_turn():
    assert correction_density(_session([_turn("t1")])) == 0.0


# --- Session scores ---


def test_score_session_implicit_returns_scores():
    turns = [_turn("t1", steering=1), _turn("t2"), _turn("t3")]
    scores = score_session_implicit(_session(turns))
    scores_by_name = {score.scorer: score for score in scores}

    assert set(scores_by_name) == {
        "implicit.correction_free_rate",
        "implicit.completion",
    }
    assert scores_by_name["implicit.correction_free_rate"].value == pytest.approx(2 / 3, abs=1e-4)
    assert scores_by_name["implicit.correction_free_rate"].metadata["correction_density"] == 1 / 3
    assert scores_by_name["implicit.completion"].value is True
    assert scores_by_name["implicit.completion"].metadata["is_abandoned"] is False


def test_score_session_implicit_persists_higher_is_better_failure_values():
    turns = [
        _turn("t1", steering=1),
        _turn("t2"),
        _turn("t3", denials=1, status="ERROR"),
    ]

    scores = {score.scorer: score for score in score_session_implicit(_session(turns))}

    assert scores["implicit.correction_free_rate"].value == pytest.approx(1 / 3, abs=1e-4)
    assert scores["implicit.correction_free_rate"].metadata["correction_density"] == 2 / 3
    assert scores["implicit.completion"].value is False
    assert scores["implicit.completion"].metadata["is_abandoned"] is True


def test_score_session_implicit_no_frustration():
    turns = [_turn("t1"), _turn("t2"), _turn("t3")]
    scores = score_session_implicit(_session(turns))
    scorer_names = {s.scorer for s in scores}
    assert "implicit.frustration" not in scorer_names
    assert "implicit.session_frustration" not in scorer_names
