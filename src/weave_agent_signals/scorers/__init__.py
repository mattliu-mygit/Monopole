from __future__ import annotations

from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.scorers.efficiency import (
    score_session_efficiency,
    score_turn_efficiency,
)
from weave_agent_signals.scorers.implicit import score_session_implicit
from weave_agent_signals.scorers.outcome import score_turn_outcomes
from weave_agent_signals.scorers.session import score_session_measures, score_session_outcomes


def score_turn(turn: TurnSpan) -> list[Score]:
    scores = score_turn_outcomes(turn)
    scores.append(score_turn_efficiency(turn))
    return scores


def score_session(session: SessionView) -> list[Score]:
    scores = score_session_implicit(session)
    scores.append(score_session_efficiency(session))
    return scores


def score_trajectory(trajectory: SessionView) -> list[Score]:
    """Scores that need every turn of a session or subagent Trajectory. `score` and `backfill`
    write them; a run pins only some of a session's turns, so it doesn't."""
    if not trajectory.turns:
        return []
    return score_session_outcomes(trajectory) + score_session_measures(trajectory)
