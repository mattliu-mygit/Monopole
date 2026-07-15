"""LLM judge implementations and rubric definitions."""

from weave_agent_signals.judges.digest import JudgeDigest
from weave_agent_signals.judges.rubrics import RUBRICS, SESSION_RUBRICS, Rubric
from weave_agent_signals.judges.verdicts import (
    JUDGE_SCORE_ANCHORS,
    JUDGE_VERDICT_SCHEMA,
    JUDGE_VERDICT_SCHEMA_VERSION,
    JudgeEvidence,
    JudgeVerdict,
    parse_judge_verdict,
)

__all__ = [
    "JUDGE_SCORE_ANCHORS",
    "JUDGE_VERDICT_SCHEMA",
    "JUDGE_VERDICT_SCHEMA_VERSION",
    "JudgeDigest",
    "JudgeEvidence",
    "JudgeVerdict",
    "RUBRICS",
    "SESSION_RUBRICS",
    "Rubric",
    "judge_session",
    "judge_turn",
    "parse_judge_verdict",
]


def __getattr__(name: str) -> object:
    if name in {"judge_session", "judge_turn"}:
        from weave_agent_signals.judges import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
