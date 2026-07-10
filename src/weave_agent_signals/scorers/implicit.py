from __future__ import annotations

from weave_agent_signals.models import Score, SessionView


def is_abandoned(session: SessionView) -> bool:
    if not session.turns:
        return True
    if len(session.turns) <= 2:
        return False
    last = session.turns[-1]
    if last.status_code == "ERROR":
        return True
    return False


def correction_density(session: SessionView) -> float:
    if len(session.turns) <= 1:
        return 0.0
    corrections = sum(
        1 for t in session.turns
        if t.steering_count > 0 or t.denial_count > 0
    )
    return corrections / len(session.turns)


def score_session_implicit(session: SessionView) -> list[Score]:
    abandoned = is_abandoned(session)
    density = correction_density(session)

    total_steerings = sum(t.steering_count for t in session.turns)
    total_denials = sum(t.denial_count for t in session.turns)
    corrections = total_steerings + total_denials
    density_reason = (f"{corrections} corrections across {len(session.turns)} turns"
                      if corrections else f"no corrections in {len(session.turns)} turns")

    abandon_reason = ("session ended with error status" if abandoned
                      else f"session completed normally ({len(session.turns)} turns)")

    return [
        Score(
            scorer="implicit.correction_density",
            value=round(density, 4),
            tags=[],
            confidence=0.9,
            metadata={
                "turn_count": len(session.turns),
                "total_steerings": total_steerings,
                "total_denials": total_denials,
            },
            granularity="session",
            reason=density_reason,
        ),
        Score(
            scorer="implicit.abandonment",
            value=abandoned,
            tags=["abandoned"] if abandoned else ["completed"],
            confidence=0.8,
            metadata={},
            granularity="session",
            reason=abandon_reason,
        ),
    ]
