from __future__ import annotations

from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.scorers.efficiency import (
    score_session_efficiency,
    score_turn_efficiency,
)
from weave_agent_signals.scorers.implicit import score_session_implicit
from weave_agent_signals.scorers.outcome import score_turn_outcomes


def score_turn(turn: TurnSpan) -> list[Score]:
    scores = score_turn_outcomes(turn)
    scores.append(score_turn_efficiency(turn))
    return scores


def score_session(session: SessionView) -> list[Score]:
    scores = score_session_implicit(session)
    scores.append(score_session_efficiency(session))
    return scores
