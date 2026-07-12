"""Turn-level process quality rubrics for LLM judges.

Each rubric defines a scoring dimension with a system prompt, scoring criteria,
and output schema. Judges score 0.0–1.0 with a rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rubric:
    name: str
    scorer_name: str
    description: str
    system_prompt: str
    criteria: dict[str, str]
    tags_on_low: list[str] = field(default_factory=list)
    tags_on_high: list[str] = field(default_factory=list)
    threshold: float = 0.5

    @property
    def criteria_text(self) -> str:
        lines = []
        for level, desc in sorted(self.criteria.items()):
            lines.append(f"- **{level}**: {desc}")
        return "\n".join(lines)


VERIFICATION_DISCIPLINE = Rubric(
    name="Verification Discipline",
    scorer_name="judge.verification",
    description="Did the agent verify its work before claiming completion?",
    system_prompt="""\
You are evaluating whether a coding agent verified its work before finishing.
Verification means running tests, type-checking, linting, building, or otherwise
confirming the change works — not just reading code or claiming it looks correct.

You will receive a structured summary of one agent turn: the tool calls made,
their results, and whether the turn ended the session.

Score the turn on a 0.0–1.0 scale based on verification discipline.""",
    criteria={
        "1.0": "Agent ran relevant verification (tests, build, lint, type-check) after making changes, and the verification passed.",
        "0.75": "Agent ran verification but it failed, or ran partial verification (e.g. only lint, not tests).",
        "0.5": "Agent made changes but ran no verification. Turn did not end the session, so verification may come in a later turn.",
        "0.25": "Agent ended the session or claimed completion without running any verification after making changes.",
        "0.0": "Agent explicitly skipped verification despite evidence it was needed (e.g. test failures in previous turns, user asked for tests).",
    },
    tags_on_low=["no_verification"],
    tags_on_high=["verified"],
    threshold=0.5,
)

ERROR_RECOVERY = Rubric(
    name="Error Recovery",
    scorer_name="judge.error_recovery",
    description="When errors occurred, did the agent recover effectively?",
    system_prompt="""\
You are evaluating how well a coding agent recovered from errors during a turn.
Good recovery means: diagnosing the root cause, changing approach when retries fail,
and not repeating the same failed action. Poor recovery means: blind retries,
ignoring error messages, or giving up without trying alternatives.

If no errors occurred in this turn, score 1.0 (nothing to recover from).

You will receive a structured summary of one agent turn with tool calls and results.

Score the turn on a 0.0–1.0 scale based on error recovery quality.""",
    criteria={
        "1.0": "No errors occurred, OR agent diagnosed the error, changed approach, and resolved it.",
        "0.75": "Agent recovered from errors but took unnecessary retries or a roundabout path.",
        "0.5": "Agent partially recovered — fixed some errors but left others or made new ones.",
        "0.25": "Agent retried the same failing approach multiple times without meaningful changes.",
        "0.0": "Agent gave up after errors without attempting recovery, or made the situation worse.",
    },
    tags_on_low=["poor_recovery"],
    tags_on_high=["clean_recovery"],
    threshold=0.5,
)

TOOL_CHOICE = Rubric(
    name="Tool Choice Quality",
    scorer_name="judge.tool_choice",
    description="Did the agent use appropriate tools for the task?",
    system_prompt="""\
You are evaluating whether a coding agent chose appropriate tools for its task.
Good tool choice means: using Read to understand before Edit, using grep/find to
locate code instead of guessing paths, using dedicated tools (Edit, Write) instead
of shell commands for file operations, running tests after changes.

You will receive a structured summary of one agent turn with tool calls and results.

Score the turn on a 0.0–1.0 scale based on tool choice quality.""",
    criteria={
        "1.0": "All tool choices were appropriate and efficient for the task.",
        "0.75": "Mostly good choices with minor inefficiencies (e.g. reading an entire file when grep would suffice).",
        "0.5": "Mixed — some good choices, some questionable (e.g. editing without reading first).",
        "0.25": "Mostly poor choices — using wrong tools, excessive exploration, missing obvious approaches.",
        "0.0": "Actively counterproductive tool usage — destructive commands, editing wrong files, etc.",
    },
    tags_on_low=["poor_tool_choice"],
    tags_on_high=["good_tool_choice"],
    threshold=0.5,
)

TASK_COMPLETION = Rubric(
    name="Task Completion Quality",
    scorer_name="judge.completion",
    description="How completely and correctly did the agent address the user's request?",
    system_prompt="""\
You are evaluating how well a coding agent completed the user's request in this turn.
Consider: did it address all parts of the request? Did it introduce regressions?
Did it leave loose ends? Was the implementation appropriate for the request scope?

You will receive a structured summary of one agent turn including the user's message
(if available), tool calls, results, and any test outcomes.

Score the turn on a 0.0–1.0 scale based on task completion quality.""",
    criteria={
        "1.0": "Fully addressed the request with correct, clean implementation. Tests pass.",
        "0.75": "Addressed the main request but missed minor aspects or left small issues.",
        "0.5": "Partially addressed the request — significant parts incomplete or incorrect.",
        "0.25": "Attempted the task but result is mostly wrong or incomplete.",
        "0.0": "Did not meaningfully address the request, or made things worse.",
    },
    tags_on_low=["incomplete"],
    tags_on_high=["complete"],
    threshold=0.5,
)

SESSION_OUTCOME = Rubric(
    name="Session Outcome Quality",
    scorer_name="judge.session_outcome",
    description="Overall: did the agent accomplish what the user wanted across the full session?",
    system_prompt="""\
You are evaluating the overall outcome of a coding agent session. A session is a
complete conversation where the user gave one or more requests and the agent worked
to fulfill them. You will see a summary of all turns: tool calls, results, errors,
and user steering corrections.

Consider the full arc: did the agent converge on a correct solution? Did it leave
unresolved issues? Did the user have to steer it heavily? Did it regress work it
had already done? Score the session holistically.""",
    criteria={
        "1.0": "Agent fully accomplished all user requests. Clean execution, tests passing, no regressions.",
        "0.75": "Agent accomplished the main request with minor loose ends or inefficiencies.",
        "0.5": "Agent partially accomplished the request — significant parts incomplete or user heavily steered.",
        "0.25": "Agent struggled substantially. Many errors, heavy user intervention, incomplete result.",
        "0.0": "Agent failed to accomplish the request or made things worse overall.",
    },
    tags_on_low=["poor_outcome"],
    tags_on_high=["good_outcome"],
    threshold=0.5,
)

SESSION_AUTONOMY = Rubric(
    name="Session Autonomy",
    scorer_name="judge.session_autonomy",
    description="How independently did the agent work without needing user corrections?",
    system_prompt="""\
You are evaluating how autonomously a coding agent worked during a session.
High autonomy means: the agent understood the task, made good decisions, and
required minimal steering or corrections. Low autonomy means: the user had to
repeatedly redirect, correct mistakes, or provide information the agent should
have found itself.

Steering events and denial events indicate user intervention.""",
    criteria={
        "1.0": "Agent worked fully independently. Zero or minimal steering, no denials.",
        "0.75": "Agent worked mostly independently with occasional, minor steering.",
        "0.5": "Agent needed moderate user guidance — several steering events or a key correction.",
        "0.25": "Agent required heavy user intervention — frequent steering, denials, or re-explanation.",
        "0.0": "Agent could not work without constant user direction.",
    },
    tags_on_low=["low_autonomy"],
    tags_on_high=["high_autonomy"],
    threshold=0.5,
)

RUBRICS: dict[str, Rubric] = {
    r.scorer_name: r
    for r in [VERIFICATION_DISCIPLINE, ERROR_RECOVERY, TOOL_CHOICE, TASK_COMPLETION]
}

SESSION_RUBRICS: dict[str, Rubric] = {
    r.scorer_name: r
    for r in [SESSION_OUTCOME, SESSION_AUTONOMY]
}
