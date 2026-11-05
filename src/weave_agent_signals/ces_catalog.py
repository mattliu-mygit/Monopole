"""Monopole's scorers as CES Scorers.

Each Monopole scorer is two CES Scorers: `<name>` holds the number or boolean
value, and `<name>.details` holds the tags, reason, confidence, and scorer details
as JSON. Both go in one create-scores request, so a reader gets them together.
Turn scorers score the turn's first Case, and session scorers score the session
Trajectory; the whole-Trajectory ones (`scorers/session.py`) also score each subagent
Trajectory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

TARGET_KINDS = {"turn": "case", "session": "trajectory", "comparison": "comparison"}
DETAILS_SUFFIX = ".details"
# What a Scorer's value means. A rating is a boolean or a [0, 1] number where higher is better;
# summaries, the A/B leaderboard, trends, alerts, and coaching read only ratings. A measurement
# is a number with no better direction, such as a token count, that queries sum and count. A
# dimension is a string, such as the repository, that queries group by.
RATING, MEASUREMENT, DIMENSION = "rating", "measurement", "dimension"


@dataclass(frozen=True)
class ScorerSpec:
    name: str
    value_type: str  # "number", "boolean", or "string"
    granularity: str  # "turn", "session", or "comparison"
    description: str | None = None
    # Every value is 0 or 1, so a summary is a pass rate with a Wilson interval. It isn't part of
    # the CES Scorer, whose create key names only the name, value type, and target kind.
    binary: bool = False
    kind: str = RATING

    @property
    def target_kind(self) -> str:
        return TARGET_KINDS[self.granularity]


DETERMINISTIC = (
    ScorerSpec("efficiency", "number", "turn"),
    ScorerSpec("outcome.test", "number", "turn", binary=True),
    ScorerSpec("outcome.build", "number", "turn", binary=True),
    ScorerSpec("outcome.lint", "number", "turn", binary=True),
    ScorerSpec("outcome.install", "number", "turn", binary=True),
    ScorerSpec("outcome.git", "number", "turn", binary=True),
    ScorerSpec("outcome.command", "number", "turn", binary=True),
    ScorerSpec("efficiency.session", "number", "session"),
    ScorerSpec("implicit.correction_free_rate", "number", "session"),
    ScorerSpec("implicit.completion", "boolean", "session", binary=True),
    # Session outcomes also score subagent Trajectories (`scorers/session.py`).
    ScorerSpec("outcome.session_tests_passed", "boolean", "session", binary=True),
    ScorerSpec("outcome.session_no_failed_builds", "boolean", "session", binary=True),
    ScorerSpec("measure.turns", "number", "session", kind=MEASUREMENT),
    ScorerSpec("measure.tokens", "number", "session", kind=MEASUREMENT),
    ScorerSpec("measure.duration_s", "number", "session", kind=MEASUREMENT),
    ScorerSpec("dim.repo", "string", "session", kind=DIMENSION),
    ScorerSpec("dim.git_branch", "string", "session", kind=DIMENSION),
    ScorerSpec("dim.day", "string", "session", kind=DIMENSION),
)
# Scorers whose Scores never join a rating path.
NOT_RATINGS = frozenset(spec.name for spec in DETERMINISTIC if spec.kind != RATING)

VERDICT = ScorerSpec(
    "ab.verdict",
    "number",
    "comparison",
    description=(
        "Paired A/B verification: 1 when B (the candidate) wins, 0.5 for a tie, 0 when A wins"
    ),
)


def judge_spec(scorer: str, metadata: dict[str, Any]) -> ScorerSpec:
    """The Scorer for one rubric judged by one panel, so a saved query's average covers one panel.

    A different panel, panel contract, or rubric version gets a different Scorer."""
    models = list(metadata.get("requested_judge_models") or [])
    panel = [metadata.get("panel_contract_version"), metadata.get("rubric_version"), models]
    digest = hashlib.sha256(json.dumps(panel).encode()).hexdigest()[:10]
    return ScorerSpec(
        f"{scorer}.panel-{digest}",
        "number",
        "session",
        description=f"{scorer} v{panel[1]} judged by {', '.join(models) or 'no models'}",
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

    def scored(self, rows: list[dict]) -> set[str]:
        """Names of the scorers whose value Score appears among a target's Score rows."""
        names = self.by_scorer_id()
        return {
            names[r["scorer_id"]][0]
            for r in rows
            if r["scorer_id"] in names and not names[r["scorer_id"]][1]
        }

    def by_scorer_id(self) -> dict[str, tuple[str, bool]]:
        """CES scorer ID → (Monopole scorer name, whether it is the details Scorer)."""
        found = {}
        for name, ref in self.scorers.items():
            found[ref.scorer_id] = (name, False)
            found[ref.details_scorer_id] = (name, True)
        return found
