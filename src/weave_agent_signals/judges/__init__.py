"""LLM judge implementations and rubric definitions."""

from weave_agent_signals.judges.digest import JudgeDigest
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS, Rubric

__all__ = [
    "JudgeDigest",
    "SESSION_RUBRICS",
    "Rubric",
    "judge_session",
]


def __getattr__(name: str) -> object:
    if name == "judge_session":
        from weave_agent_signals.judges import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
