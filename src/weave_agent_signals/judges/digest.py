"""Judge evidence container and prompt-message formatting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from weave_agent_signals.models import SessionView, TurnSpan

JUDGE_DIGEST_CONTRACT_VERSION = "3.0.0"
_JUDGE_USER_PROMPT_TEMPLATE = (
    "## Scoring criteria\n\n{rubric_criteria}\n\n"
    "## {unit_heading} data\n\n{digest}\n\n"
    "Allowed evidence IDs: {allowed_evidence_ids}\n\n"
    "## Instructions\n\n"
    "Evaluate this {unit} against the criteria above. Respond with exactly one JSON object "
    "and no prose or Markdown fences. A scored verdict has this exact layout:\n"
    "{{\n"
    '  "schema_version": 3,\n'
    '  "status": "scored",\n'
    '  "score": 0.75,\n'
    '  "rationale": "Concise explanation grounded only in the supplied evidence.",\n'
    '  "evidence": [\n'
    "    {{\n"
    '      "id": {example_evidence_id},\n'
    '      "observations": ["Specific behavior supporting the verdict."]\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Every scored verdict must cite at least one allowed evidence ID. Cite each ID exactly "
    "once, group its distinct nonblank observations in the observations array, and choose "
    "only 0, 0.25, 0.5, 0.75, or 1 as the score. If the supplied evidence is "
    "insufficient, return:\n"
    "{{\n"
    '  "schema_version": 3,\n'
    '  "status": "insufficient_evidence",\n'
    '  "score": null,\n'
    '  "rationale": "Why the supplied evidence is insufficient.",\n'
    '  "evidence": []\n'
    "}}"
)


@dataclass(frozen=True)
class JudgeDigest:
    text: str
    evidence_ids: tuple[str, ...]


def _prompt_digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def judge_digest_contract_manifest() -> dict[str, object]:
    """Return deterministic raw-evidence and judge-message formatting semantics."""

    return {
        "raw_evidence": "complete_captured_turns_without_model_identity",
        "prompt_digests": {
            "judge_user": _prompt_digest(_JUDGE_USER_PROMPT_TEMPLATE),
        },
    }


def build_turn_digest(turn: TurnSpan) -> JudgeDigest:
    """Compatibility entry point for complete raw single-turn evidence."""

    from weave_agent_signals.judges.windowing import _turn_evidence_ids, render_raw_turn

    return JudgeDigest(text=render_raw_turn(turn, 1), evidence_ids=_turn_evidence_ids(turn))


def build_turn_digest_with_context(
    turn: TurnSpan,
    prior_turns: list[TurnSpan],
    window: int = 3,
) -> JudgeDigest:
    """Compatibility entry point for complete raw prior and current turns."""

    del window
    from weave_agent_signals.judges.windowing import _turn_evidence_ids, render_raw_turn

    selected = [*prior_turns, turn]
    rendered = [render_raw_turn(item, index) for index, item in enumerate(selected, start=1)]
    return JudgeDigest(
        text="\n\n".join(rendered),
        evidence_ids=tuple(
            evidence_id for item in selected for evidence_id in _turn_evidence_ids(item)
        ),
    )


def build_session_digest(
    session: SessionView,
    max_turns: int = 100,
    evidence_trace_ids: list[str] | None = None,
) -> JudgeDigest:
    """Temporary runner compatibility wrapper for complete raw session evidence.

    ``max_turns`` and ``evidence_trace_ids`` remain in the signature only until
    the runner adopts planned windows; raw evidence is never selected or truncated.
    """

    del max_turns
    from weave_agent_signals.judges.windowing import render_raw_window

    available_ids = {turn.trace_id for turn in session.turns}
    missing_ids = [
        trace_id
        for trace_id in dict.fromkeys(evidence_trace_ids or [])
        if trace_id not in available_ids
    ]
    if missing_ids:
        raise ValueError("missing requested evidence IDs: " + ", ".join(missing_ids))

    return render_raw_window(
        session,
        {"raw_trace_ids": [turn.trace_id for turn in session.turns]},
    )


def build_judge_messages(
    rubric_system: str,
    rubric_criteria: str,
    digest: JudgeDigest,
    granularity: str = "turn",
) -> list[dict[str, str]]:
    """Build the messages array for a judge LLM call."""

    unit = "session" if granularity == "session" else "turn"
    return [
        {"role": "system", "content": rubric_system},
        {
            "role": "user",
            "content": _JUDGE_USER_PROMPT_TEMPLATE.format(
                rubric_criteria=rubric_criteria,
                unit_heading=unit.capitalize(),
                digest=digest.text,
                unit=unit,
                allowed_evidence_ids=json.dumps(list(digest.evidence_ids)),
                example_evidence_id=json.dumps(
                    digest.evidence_ids[0] if digest.evidence_ids else "<allowed evidence ID>"
                ),
            ),
        },
    ]
