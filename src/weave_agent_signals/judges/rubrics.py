"""Session-level process and outcome quality rubrics for LLM judges.

Each rubric defines a scoring dimension with a system prompt, scoring criteria,
and output schema. Judges select one of five anchors or abstain with evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rubric:
    name: str
    scorer_name: str
    description: str
    version: str
    evaluation_unit: str
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


_ANCHORS = ("0", "0.25", "0.5", "0.75", "1")


def _rubric_prompt(
    *,
    question: str,
    allowed_evidence: str,
    excluded_concerns: str,
    insufficient_evidence_guidance: str,
    criteria: dict[str, str],
) -> str:
    """Render the one shared visible rubric structure."""

    if tuple(criteria) != _ANCHORS:
        raise ValueError("rubric criteria must define the five ordered score anchors")
    anchors = "\n".join(f"- **{anchor}**: {criteria[anchor]}" for anchor in _ANCHORS)
    return f"""\
## Evaluation question
{question}

## Allowed evidence
{allowed_evidence}

## Excluded concerns
{excluded_concerns}

## Insufficient evidence
{insufficient_evidence_guidance}

## Score anchors
{anchors}

"""


def _rubric(
    *,
    name: str,
    scorer_name: str,
    description: str,
    evaluation_unit: str,
    question: str,
    allowed_evidence: str,
    excluded_concerns: str,
    insufficient_evidence_guidance: str,
    criteria: dict[str, str],
    tags_on_low: list[str],
    tags_on_high: list[str],
) -> Rubric:
    return Rubric(
        name=name,
        scorer_name=scorer_name,
        description=description,
        version="v4",
        evaluation_unit=evaluation_unit,
        system_prompt=_rubric_prompt(
            question=question,
            allowed_evidence=allowed_evidence,
            excluded_concerns=excluded_concerns,
            insufficient_evidence_guidance=insufficient_evidence_guidance,
            criteria=criteria,
        ),
        criteria=criteria,
        tags_on_low=tags_on_low,
        tags_on_high=tags_on_high,
        threshold=0.5,
    )


VERIFICATION_DISCIPLINE = _rubric(
    name="Verification Discipline",
    scorer_name="judge.verification",
    description="Do cited checks support the agent's completion or correctness claims?",
    evaluation_unit="session",
    question=(
        "When the agent claimed work was complete or correct, did cited checks and their "
        "results support every material claim?"
    ),
    allowed_evidence=(
        "Use the supplied completion or correctness claims, the checks the agent actually "
        "performed, their recorded results, their relevance to the claimed work, and whether "
        "they occurred after the work they are offered to verify."
    ),
    excluded_concerns=(
        "Do not judge whether the initial tool or implementation approach was a good choice; "
        "tool choice owns that question. Do not credit checks that are merely asserted or not "
        "supported by an allowed evidence citation."
    ),
    insufficient_evidence_guidance=(
        "Apply this rubric when the supplied session contains a material completion or "
        "correctness claim and enough coverage to determine whether supporting checks occurred. "
        "The absence of a supporting check in otherwise complete evidence is scorable at `0.25`; "
        "it is not a reason to abstain. Return `insufficient_evidence` when no such claim is "
        "present or the evidence coverage is missing, truncated, or ambiguous."
    ),
    criteria={
        "0": (
            "The agent makes a material completion or correctness claim despite unresolved cited "
            "failed checks or cited evidence that directly contradicts the claim."
        ),
        "0.25": (
            "The agent makes a material completion or correctness claim with no relevant cited "
            "check, or relies only on assertion or inspection that cannot validate the claim."
        ),
        "0.5": (
            "Cited checks address part of the claim, but material coverage is missing, results "
            "are inconclusive, or the checks occurred before relevant changes."
        ),
        "0.75": (
            "Relevant successful checks support the main claims, with one limited and clearly "
            "non-critical verification gap."
        ),
        "1": (
            "Relevant successful checks, cited from the supplied evidence, support every "
            "material completion and correctness claim."
        ),
    },
    tags_on_low=["no_verification"],
    tags_on_high=["verified"],
)

ERROR_RECOVERY = _rubric(
    name="Error Recovery",
    scorer_name="judge.error_recovery",
    description="Did the agent adapt effectively after observed error evidence?",
    evaluation_unit="session",
    question=(
        "After the agent observed an error, did it diagnose the evidence and adapt its next "
        "actions effectively?"
    ),
    allowed_evidence=(
        "Use only observed error messages or failed results, the actions taken after those "
        "errors, stated diagnoses that are supported by the evidence, changed approaches, "
        "repeated failures, and the directly resulting recovery or containment."
    ),
    excluded_concerns=(
        "Do not award recovery quality for eventual task success by itself. Do not judge the "
        "initial tool choice except where the agent's response to observed failure shows whether "
        "it adapted."
    ),
    insufficient_evidence_guidance=(
        "Apply this rubric only when the supplied evidence shows an observed error and a later "
        "opportunity to respond. Return `insufficient_evidence` when no error was observed, no "
        "post-error response is supplied, or the evidence cannot show how the agent recovered "
        "from errors."
    ),
    criteria={
        "0": (
            "The agent makes no meaningful response to clear error evidence, repeats the same "
            "disproven action without diagnosis, or makes the situation materially worse."
        ),
        "0.25": (
            "The agent makes a weak but genuine diagnosis or relevant change, yet the adaptation "
            "is too superficial to recover or safely contain the error."
        ),
        "0.5": (
            "The agent identifies part of the problem and makes a meaningful change, but the "
            "diagnosis or adaptation remains incomplete and causes avoidable continued failure."
        ),
        "0.75": (
            "The agent uses the error evidence to diagnose and adapt successfully, with only a "
            "minor avoidable detour or narrow non-critical gap in the adaptation."
        ),
        "1": (
            "The agent accurately diagnoses the observed error, changes course based on that "
            "diagnosis, and resolves or safely contains it without avoidable repetition."
        ),
    },
    tags_on_low=["poor_recovery"],
    tags_on_high=["clean_recovery"],
)

TOOL_CHOICE = _rubric(
    name="Tool Choice Quality",
    scorer_name="judge.tool_choice",
    description="Did the selected capabilities fit the task and were they used efficiently?",
    evaluation_unit="session",
    question=(
        "Were the agent's selected tools or methods well matched to the task and used in an "
        "efficient, safe sequence?"
    ),
    allowed_evidence=(
        "Use the stated task and constraints, the capabilities demonstrated by each recorded "
        "tool or method, their ordering, redundancy, safety, and the immediate results needed to "
        "assess fit."
    ),
    excluded_concerns=(
        "Do not judge later verification quality; verification owns whether checks support "
        "completion claims. Do not impose vendor-specific tool names or prefer a named product "
        "when a capability-equivalent choice fits the task."
    ),
    insufficient_evidence_guidance=(
        "Apply this rubric only when the task goal and at least one tool or method choice are "
        "visible enough to assess fit. Return `insufficient_evidence` when the task, the selected "
        "capability, or the relevant result is absent or opaque."
    ),
    criteria={
        "0": (
            "The selected tools or methods are destructive, unsafe, or fundamentally incapable "
            "of performing the task as used."
        ),
        "0.25": (
            "Most choices are poorly matched or wasteful, with repeated avoidable work or a "
            "missed readily available capability that blocks progress."
        ),
        "0.5": (
            "The sequence mixes suitable and unsuitable choices; it can make progress but incurs "
            "material avoidable exploration, repetition, or risk."
        ),
        "0.75": (
            "The choices are well matched and safe, with only a minor inefficiency or ordering "
            "issue that does not materially affect progress."
        ),
        "1": (
            "The selected capabilities directly fit the task and are used in a safe, efficient "
            "sequence without material redundant work."
        ),
    },
    tags_on_low=["poor_tool_choice"],
    tags_on_high=["good_tool_choice"],
)

STATE_CONSISTENCY = _rubric(
    name="State Consistency",
    scorer_name="judge.state_consistency",
    description="Did current behavior remain consistent with supplied established state?",
    evaluation_unit="session",
    question=(
        "Did the agent's current actions and claims remain consistent with established prior "
        "state, decisions, and constraints in the supplied evidence?"
    ),
    allowed_evidence=(
        "Use only prior state, decisions, completed work, and constraints explicitly established "
        "in allowed evidence, together with current actions or claims that preserve or contradict "
        "them."
    ),
    excluded_concerns=(
        "Do not infer prior state from conventions, likely intent, or facts outside the supplied "
        "evidence. Do not count a user-authorized change or a correction based on newly supplied "
        "information as a contradiction."
    ),
    insufficient_evidence_guidance=(
        "Apply this rubric only when the supplied evidence establishes prior state or a prior "
        "constraint that later behavior could preserve or contradict. The judge must return "
        "`insufficient_evidence` when there is no established prior state, or when that state is "
        "too ambiguous to support a contradiction finding."
    ),
    criteria={
        "0": (
            "The current behavior directly reverses or overwrites clear established prior state "
            "or constraints, creating a major unresolved conflict."
        ),
        "0.25": (
            "The current behavior contains a material contradiction or several repeated "
            "contradictions and does not meaningfully correct them."
        ),
        "0.5": (
            "The current behavior introduces one meaningful inconsistency or several minor ones, "
            "but partially recognizes or corrects the conflict."
        ),
        "0.75": (
            "The current behavior preserves all material established state, with one minor lapse "
            "that does not alter the result."
        ),
        "1": (
            "The current behavior is fully consistent with every relevant established decision, "
            "constraint, and completed state in the supplied evidence."
        ),
    },
    tags_on_low=["inconsistent_state"],
    tags_on_high=["consistent_state"],
)

SESSION_OUTCOME = _rubric(
    name="Session Outcome Quality",
    scorer_name="judge.session_outcome",
    description="Did the final result correctly and completely fulfill the user's request?",
    evaluation_unit="session",
    question=(
        "Did the final session outcome correctly and completely fulfill the user's material "
        "request and stated constraints?"
    ),
    allowed_evidence=(
        "Use the user's requests and acceptance constraints, the final artifacts or actions, "
        "relevant check results, unresolved errors, explicit limitations, and the observable final "
        "state."
    ),
    excluded_concerns=(
        "Do not score steering burden or how much correction the user supplied; session autonomy "
        "owns that question. Do not penalize process style or tool efficiency unless it changes "
        "the correctness or completeness of the achieved result."
    ),
    insufficient_evidence_guidance=(
        "Apply this rubric when the supplied session identifies a user request and provides enough "
        "final-state evidence to assess fulfillment. Return `insufficient_evidence` when the "
        "material request or observable outcome is missing or too truncated to judge."
    ),
    criteria={
        "0": (
            "The session produces no usable result for the requested purpose, produces a harmful "
            "or fundamentally unusable result, or leaves the user's state worse."
        ),
        "0.25": (
            "The session delivers a limited usable fragment, but central requirements are absent "
            "or serious defects prevent the core requested use."
        ),
        "0.5": (
            "The session delivers meaningful partial value that remains usable, but one major "
            "requirement is missing, incorrect, or unsupported."
        ),
        "0.75": (
            "The core request is correctly fulfilled, with only a minor non-critical omission, "
            "limitation, or verification gap."
        ),
        "1": (
            "The final result correctly fulfills every material request and stated constraint, "
            "with adequate evidence for the claimed outcome and no material regression."
        ),
    },
    tags_on_low=["poor_outcome"],
    tags_on_high=["good_outcome"],
)

SESSION_AUTONOMY = _rubric(
    name="Session Autonomy",
    scorer_name="judge.session_autonomy",
    description="How much avoidable user dependence or correction did the work require?",
    evaluation_unit="session",
    question=(
        "How independently did the agent progress, considering only avoidable user direction, "
        "correction, or re-explanation?"
    ),
    allowed_evidence=(
        "Use the initial request, information already available to the agent, the agent's actions, "
        "user corrections and redirects, repeated explanations, and whether requests for help "
        "were avoidable within the agent's authority."
    ),
    excluded_concerns=(
        "Do not penalize required approvals, authentication, policy gates, or genuinely missing "
        "requirements that only the user could supply. Do not score final correctness or request "
        "fulfillment; session outcome owns that question."
    ),
    insufficient_evidence_guidance=(
        "Apply this rubric when the supplied interaction history is complete enough to distinguish "
        "avoidable dependence from required user input. Return `insufficient_evidence` when key "
        "requests, corrections, or authority constraints are missing or truncated."
    ),
    criteria={
        "0": (
            "The agent cannot incorporate repeated correction or proceed without near-constant "
            "avoidable user direction, so progress stalls or regresses."
        ),
        "0.25": (
            "The agent eventually makes progress, but only after repeated major corrections, "
            "redirects, or re-explanations it could have avoided using supplied information."
        ),
        "0.5": (
            "The agent makes independent progress but needs one major avoidable correction or "
            "several smaller avoidable interventions."
        ),
        "0.75": (
            "The agent works mostly independently and needs at most one minor avoidable steer or "
            "correction."
        ),
        "1": (
            "The agent progresses without avoidable user dependence, resolves available questions "
            "itself, and asks only for genuinely required gates or missing information."
        ),
    },
    tags_on_low=["low_autonomy"],
    tags_on_high=["high_autonomy"],
)

SESSION_RUBRICS: dict[str, Rubric] = {
    rubric.scorer_name: rubric
    for rubric in (
        VERIFICATION_DISCIPLINE,
        ERROR_RECOVERY,
        TOOL_CHOICE,
        STATE_CONSISTENCY,
        SESSION_OUTCOME,
        SESSION_AUTONOMY,
    )
}
