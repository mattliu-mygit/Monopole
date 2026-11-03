"""Monopole's scorers as CES Scorers.

Each Monopole scorer is two CES Scorers: `<name>` holds the number or boolean
value, and `<name>.details` holds the tags, reason, confidence, and scorer details
as JSON. Both go in one create-scores request, so a reader gets them together.
Turn scorers score the turn's first Case, and session scorers score the session
Trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_KINDS = {"turn": "case", "session": "trajectory"}
DETAILS_SUFFIX = ".details"


@dataclass(frozen=True)
class ScorerSpec:
    name: str
    value_type: str  # "number" or "boolean"
    granularity: str  # "turn" or "session"

    @property
    def target_kind(self) -> str:
        return TARGET_KINDS[self.granularity]


DETERMINISTIC = (
    ScorerSpec("efficiency", "number", "turn"),
    ScorerSpec("outcome.test", "number", "turn"),
    ScorerSpec("outcome.build", "number", "turn"),
    ScorerSpec("outcome.lint", "number", "turn"),
    ScorerSpec("outcome.install", "number", "turn"),
    ScorerSpec("outcome.git", "number", "turn"),
    ScorerSpec("outcome.command", "number", "turn"),
    ScorerSpec("efficiency.session", "number", "session"),
    ScorerSpec("implicit.correction_free_rate", "number", "session"),
    ScorerSpec("implicit.completion", "boolean", "session"),
)


@dataclass(frozen=True)
class ScorerRef:
    """The CES identities for one Monopole scorer: its value Scorer and its details Scorer."""

    spec: ScorerSpec
    scorer_id: str
    version_id: str
    details_scorer_id: str
    details_version_id: str


@dataclass(frozen=True)
class Catalog:
    turns_evaluation_id: str
    sessions_evaluation_id: str
    scorers: dict[str, ScorerRef]

    def evaluation_for(self, target_kind: str) -> str:
        return self.turns_evaluation_id if target_kind == "case" else self.sessions_evaluation_id

    def by_scorer_id(self) -> dict[str, tuple[str, bool]]:
        """CES scorer ID → (Monopole scorer name, whether it is the details Scorer)."""
        found = {}
        for name, ref in self.scorers.items():
            found[ref.scorer_id] = (name, False)
            found[ref.details_scorer_id] = (name, True)
        return found
