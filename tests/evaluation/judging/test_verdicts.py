from __future__ import annotations

import math
import subprocess
import sys

import pytest

import weave_agent_signals.judges as judges_package
from weave_agent_signals.judges.rubrics import RUBRICS, SESSION_RUBRICS, Rubric
from weave_agent_signals.judges.runner import judge_session, judge_turn
from weave_agent_signals.judges.verdicts import (
    JUDGE_SCORE_ANCHORS,
    JUDGE_VERDICT_SCHEMA,
    JUDGE_VERDICT_SCHEMA_VERSION,
    JudgeEvidence,
    JudgeVerdict,
    parse_judge_verdict,
)


def _scored(**changes: object) -> dict[str, object]:
    verdict: dict[str, object] = {
        "schema_version": 3,
        "status": "scored",
        "score": 0.75,
        "rationale": "The assistant verified the requested behavior.",
        "evidence": [
            {
                "id": "trace-1",
                "observations": ["The trace shows a focused verification command passing."],
            }
        ],
    }
    verdict.update(changes)
    return verdict


def test_judge_verdict_schema_is_closed_and_versioned():
    assert JUDGE_VERDICT_SCHEMA_VERSION == 3
    assert JUDGE_SCORE_ANCHORS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert JUDGE_VERDICT_SCHEMA.name == "judge_verdict"

    schema = JUDGE_VERDICT_SCHEMA.schema
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == 3
    assert schema["$defs"]["JudgeEvidence"]["additionalProperties"] is False
    assert set(schema["required"]) == {
        "schema_version",
        "status",
        "score",
        "rationale",
        "evidence",
    }

    verdict = parse_judge_verdict(_scored(), evidence_ids=("trace-1",))
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.schema_version == JUDGE_VERDICT_SCHEMA_VERSION


def test_scored_verdict_requires_an_anchored_finite_score():
    for score in JUDGE_SCORE_ANCHORS:
        verdict = parse_judge_verdict(_scored(score=score), evidence_ids=("trace-1",))
        assert verdict.score == score

    for score in (-0.25, 0.1, 1.25, math.nan, math.inf, -math.inf, None):
        with pytest.raises(ValueError):
            parse_judge_verdict(_scored(score=score), evidence_ids=("trace-1",))


def test_scored_verdict_requires_known_grouped_evidence():
    with pytest.raises(ValueError, match="at least one evidence"):
        parse_judge_verdict(_scored(evidence=[]), evidence_ids=("trace-1",))

    with pytest.raises(ValueError, match="unknown evidence ID"):
        parse_judge_verdict(
            _scored(evidence=[{"id": "trace-2", "observations": ["Not in the digest."]}]),
            evidence_ids=("trace-1",),
        )

    with pytest.raises(ValueError, match="must not be empty"):
        parse_judge_verdict(
            _scored(evidence=[{"id": "trace-1", "observations": []}]),
            evidence_ids=("trace-1",),
        )

    with pytest.raises(ValueError, match="observation must be nonblank"):
        parse_judge_verdict(
            _scored(evidence=[{"id": "trace-1", "observations": ["   "]}]),
            evidence_ids=("trace-1",),
        )


def test_verdict_groups_repeated_evidence_and_deduplicates_observations():
    verdict = parse_judge_verdict(
        {
            "schema_version": 3,
            "status": "scored",
            "score": 0.75,
            "rationale": "The trace contains both useful and avoidable behavior.",
            "evidence": [
                {
                    "id": "trace-1",
                    "observations": [
                        "Repository inspection was thorough.",
                        "Several environment checks were redundant.",
                    ],
                },
                {
                    "id": "trace-1",
                    "observations": [
                        "Several environment checks were redundant.",
                        "Verification eventually covered the changed behavior.",
                    ],
                },
            ],
        },
        evidence_ids=("trace-1",),
    )

    assert tuple(item.id for item in verdict.evidence) == ("trace-1",)
    assert verdict.evidence[0].observations == (
        "Repository inspection was thorough.",
        "Several environment checks were redundant.",
        "Verification eventually covered the changed behavior.",
    )


def test_insufficient_evidence_requires_null_score():
    verdict = parse_judge_verdict(
        {
            "schema_version": 3,
            "status": "insufficient_evidence",
            "score": None,
            "rationale": "The supplied trace does not establish the required prior state.",
            "evidence": [],
        },
        evidence_ids=("trace-1",),
    )
    assert verdict.status == "insufficient_evidence"
    assert verdict.score is None
    assert verdict.evidence == ()

    with pytest.raises(ValueError, match="null score"):
        parse_judge_verdict(
            {
                "schema_version": 3,
                "status": "insufficient_evidence",
                "score": 0.0,
                "rationale": "There is not enough evidence.",
                "evidence": [],
            },
            evidence_ids=("trace-1",),
        )

    with pytest.raises(ValueError, match="empty evidence"):
        parse_judge_verdict(
            {
                "schema_version": 3,
                "status": "insufficient_evidence",
                "score": None,
                "rationale": "The evidence does not establish the required state.",
                "evidence": [{"id": "trace-1", "observations": ["One trace was supplied."]}],
            },
            evidence_ids=("trace-1",),
        )


def test_verdict_rejects_extra_fields_booleans_and_numeric_strings():
    invalid_verdicts = [
        _scored(extra="not allowed"),
        _scored(
            evidence=[
                {
                    "id": "trace-1",
                    "observations": ["Observed behavior."],
                    "extra": "not allowed",
                }
            ]
        ),
        _scored(schema_version="3"),
        _scored(score=True),
        _scored(score="0.75"),
        _scored(rationale="   "),
    ]

    for raw_verdict in invalid_verdicts:
        with pytest.raises(ValueError):
            parse_judge_verdict(raw_verdict, evidence_ids=("trace-1",))

    mutated_verdict = JudgeVerdict.model_validate(_scored())
    mutated_verdict.score = True
    with pytest.raises(ValueError):
        parse_judge_verdict(mutated_verdict, evidence_ids=("trace-1",))


def test_verdict_rejects_forbidden_extra_on_direct_model_instances():
    verdict = JudgeVerdict.model_validate(_scored())
    verdict_with_extra = verdict.model_copy(update={"extra": "not allowed"})
    with pytest.raises(ValueError):
        parse_judge_verdict(verdict_with_extra, evidence_ids=("trace-1",))

    evidence_with_extra = verdict.evidence[0].model_copy(update={"extra": "not allowed"})
    verdict_with_nested_extra = verdict.model_copy(update={"evidence": (evidence_with_extra,)})
    with pytest.raises(ValueError):
        parse_judge_verdict(verdict_with_nested_extra, evidence_ids=("trace-1",))


def test_verdict_rejects_forbidden_extra_on_nested_model_in_mapping():
    evidence = JudgeEvidence.model_validate(
        {
            "id": "trace-1",
            "observations": ["The trace shows a focused verification command passing."],
        }
    )
    evidence_with_extra = evidence.model_copy(update={"extra": "not allowed"})
    with pytest.raises(ValueError):
        parse_judge_verdict(
            _scored(evidence=[evidence_with_extra]),
            evidence_ids=("trace-1",),
        )


def test_judges_package_preserves_public_exports():
    assert judges_package.Rubric is Rubric
    assert judges_package.RUBRICS is RUBRICS
    assert judges_package.SESSION_RUBRICS is SESSION_RUBRICS
    assert judges_package.judge_turn is judge_turn
    assert judges_package.judge_session is judge_session

    fresh_import = subprocess.run(
        [sys.executable, "-c", "from weave_agent_signals.catalogs import build_rubric_catalog"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fresh_import.returncode == 0, fresh_import.stderr
