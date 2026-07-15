from __future__ import annotations

import math

import pytest

from weave_agent_signals.judges.sliding_contracts import (
    BEHAVIORAL_FEEDBACK_SCHEMA,
    CHUNK_DIGEST_SCHEMA,
    MERGED_VERDICT_SCHEMA,
    WINDOW_FINDING_SCHEMA,
    WINDOW_FINDINGS_SCHEMA,
    BehavioralFeedback,
    ChunkDigest,
    MergedVerdict,
    WindowFinding,
    WindowFindings,
    parse_behavioral_feedback,
    parse_chunk_digest,
    parse_merged_verdict,
    parse_window_finding,
    parse_window_findings,
)


def _valid_merged_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "scored",
        "score": 0.5,
        "rationale": "The agent completed the task with one verification gap.",
        "evidence_ids": ["trace-1"],
        "feedback": {
            "success": "It completed the requested change.",
            "problem": "It did not rerun the relevant check.",
            "desired_behavior": "Rerun the relevant check after the final change.",
        },
    }


def _valid_finding(**changes: object) -> dict[str, object]:
    finding: dict[str, object] = {
        "finding_id": "finding-1",
        "polarity": "negative",
        "observation": "The agent did not rerun the failed check.",
        "evidence_ids": ["trace-1"],
    }
    finding.update(changes)
    return finding


def test_sliding_contract_schemas_are_closed_and_versioned():
    roots = (
        (CHUNK_DIGEST_SCHEMA, "chunk_digest"),
        (WINDOW_FINDING_SCHEMA, "window_finding"),
        (WINDOW_FINDINGS_SCHEMA, "window_findings"),
        (BEHAVIORAL_FEEDBACK_SCHEMA, "behavioral_feedback"),
        (MERGED_VERDICT_SCHEMA, "merged_verdict"),
    )
    for schema_spec, name in roots:
        assert schema_spec.name == name
        assert schema_spec.schema["additionalProperties"] is False

    assert CHUNK_DIGEST_SCHEMA.schema["properties"]["schema_version"]["const"] == 1
    assert WINDOW_FINDINGS_SCHEMA.schema["properties"]["schema_version"]["const"] == 1
    assert MERGED_VERDICT_SCHEMA.schema["properties"]["schema_version"]["const"] == 1


def test_chunk_digest_parses_normalized_bounded_cited_text():
    parsed = parse_chunk_digest(
        {
            "schema_version": 1,
            "chunk_id": "chunk-1",
            "text": "Observed   recovery\n after a failed check.",
            "evidence_ids": ["trace-1", "span-1"],
        },
        allowed_evidence_ids=("trace-1", "span-1"),
        max_tokens=20,
    )
    assert isinstance(parsed, ChunkDigest)
    assert parsed.text == "Observed recovery after a failed check."


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"evidence_ids": ["unknown"]}, "unknown evidence ID"),
        ({"evidence_ids": [" "]}, "nonblank"),
        ({"evidence_ids": ["trace-1", "trace-1"]}, "unique"),
        ({"text": "  \n "}, "nonblank"),
        ({"extra": "forbidden"}, "Extra inputs"),
    ],
)
def test_chunk_digest_rejects_open_or_uncited_content(changes: dict[str, object], match: str):
    payload: dict[str, object] = {
        "schema_version": 1,
        "chunk_id": "chunk-1",
        "text": "Observed behavior.",
        "evidence_ids": ["trace-1"],
    }
    payload.update(changes)
    with pytest.raises(ValueError, match=match):
        parse_chunk_digest(
            payload,
            allowed_evidence_ids=("trace-1",),
            max_tokens=50,
        )


def test_chunk_digest_rejects_text_over_token_limit():
    with pytest.raises(ValueError, match="token limit"):
        parse_chunk_digest(
            {
                "schema_version": 1,
                "chunk_id": "chunk-1",
                "text": "A digest that cannot fit in one token.",
                "evidence_ids": ["trace-1"],
            },
            allowed_evidence_ids=("trace-1",),
            max_tokens=1,
        )


def test_window_finding_parses_normalized_bounded_observation():
    parsed = parse_window_finding(
        _valid_finding(observation="The agent  did not\n rerun the check."),
        allowed_evidence_ids=("trace-1",),
    )
    assert isinstance(parsed, WindowFinding)
    assert parsed.observation == "The agent did not rerun the check."


@pytest.mark.parametrize("polarity", ["neutral", "Positive", "", 1])
def test_window_finding_rejects_invalid_polarity(polarity: object):
    with pytest.raises(ValueError):
        parse_window_finding(
            _valid_finding(polarity=polarity),
            allowed_evidence_ids=("trace-1",),
        )


def test_window_finding_rejects_unknown_blank_duplicate_and_over_limit_content():
    invalid = (
        (_valid_finding(evidence_ids=["trace-2"]), "unknown evidence ID"),
        (_valid_finding(evidence_ids=[" trace-1 "]), "unknown evidence ID"),
        (_valid_finding(evidence_ids=[" "]), "nonblank"),
        (_valid_finding(evidence_ids=["trace-1", "trace-1"]), "unique"),
        (_valid_finding(observation=" "), "nonblank"),
        (_valid_finding(observation="x" * 351), "at most 350"),
        (_valid_finding(extra="forbidden"), "Extra inputs"),
    )
    for payload, match in invalid:
        with pytest.raises(ValueError, match=match):
            parse_window_finding(payload, allowed_evidence_ids=("trace-1",))


def test_window_findings_rejects_duplicate_or_excess_findings():
    payload = {
        "schema_version": 1,
        "window_id": "window-1",
        "findings": [_valid_finding(), _valid_finding()],
    }
    with pytest.raises(ValueError, match="finding IDs must be unique"):
        parse_window_findings(payload, allowed_evidence_ids=("trace-1",))

    payload["findings"] = [_valid_finding(finding_id=f"finding-{index}") for index in range(5)]
    with pytest.raises(ValueError, match="at most 4"):
        parse_window_findings(payload, allowed_evidence_ids=("trace-1",))


def test_window_findings_requires_expected_window_and_rejects_extra_fields():
    payload = {"schema_version": 1, "window_id": "window-1", "findings": []}
    parsed = parse_window_findings(
        payload,
        allowed_evidence_ids=("trace-1",),
        expected_window_id="window-1",
    )
    assert isinstance(parsed, WindowFindings)

    with pytest.raises(ValueError, match="unexpected window ID"):
        parse_window_findings(
            payload,
            allowed_evidence_ids=("trace-1",),
            expected_window_id="window-2",
        )
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_window_findings(
            {**payload, "extra": True},
            allowed_evidence_ids=("trace-1",),
        )


def test_behavioral_feedback_requires_bounded_nonblank_content():
    parsed = parse_behavioral_feedback(
        {"success": "  Good   recovery. ", "problem": None, "desired_behavior": None}
    )
    assert isinstance(parsed, BehavioralFeedback)
    assert parsed.success == "Good recovery."

    invalid = (
        ({"success": None, "problem": None, "desired_behavior": None}, "at least one"),
        ({"success": " ", "problem": None, "desired_behavior": None}, "nonblank"),
        ({"success": "x" * 501, "problem": None, "desired_behavior": None}, "at most 500"),
        (
            {"success": "Good.", "problem": None, "desired_behavior": None, "extra": True},
            "Extra inputs",
        ),
    )
    for payload, match in invalid:
        with pytest.raises(ValueError, match=match):
            parse_behavioral_feedback(payload)


def test_merged_verdict_keeps_behavioral_feedback_grounded():
    parsed = parse_merged_verdict(
        {
            "schema_version": 1,
            "status": "scored",
            "score": 0.5,
            "rationale": "The agent recovered but did not rerun the final check.",
            "evidence_ids": ["trace-2", "trace-3"],
            "feedback": {
                "success": "It changed approach after the failure.",
                "problem": "It claimed completion without rerunning the failed check.",
                "desired_behavior": "Rerun relevant checks after the final change.",
            },
        },
        allowed_evidence_ids=("trace-1", "trace-2", "trace-3"),
    )
    assert isinstance(parsed, MergedVerdict)
    assert parsed.feedback is not None
    assert parsed.feedback.desired_behavior == "Rerun relevant checks after the final change."


def test_feedback_must_not_prescribe_managed_file_edits():
    payload = _valid_merged_payload()
    feedback = payload["feedback"]
    assert isinstance(feedback, dict)
    feedback["desired_behavior"] = "Add a rule to AGENTS.md."
    with pytest.raises(ValueError, match="behavior rather than instruction edits"):
        parse_merged_verdict(payload, allowed_evidence_ids=("trace-1",))


@pytest.mark.parametrize(
    "desired_behavior",
    [
        "Update CLAUDE.md with the workflow.",
        "Delete the old SKILL.md rule.",
        "Create an instruction file for verification.",
        "Rewrite the prompt file to require checks.",
        "Add a new policy.py file.",
    ],
)
def test_feedback_rejects_managed_or_imperative_file_edit_language(desired_behavior: str):
    payload = _valid_merged_payload()
    feedback = payload["feedback"]
    assert isinstance(feedback, dict)
    feedback["desired_behavior"] = desired_behavior
    with pytest.raises(ValueError, match="behavior rather than instruction edits"):
        parse_merged_verdict(payload, allowed_evidence_ids=("trace-1",))


def test_merged_scored_verdict_requires_anchor_citations_and_feedback():
    for score in (0.0, 0.25, 0.5, 0.75, 1.0):
        parsed = parse_merged_verdict(
            {**_valid_merged_payload(), "score": score},
            allowed_evidence_ids=("trace-1",),
        )
        assert parsed.score == score

    invalid = (
        ({"score": 0.1}, "score must be one of"),
        ({"score": math.inf}, "score must be one of"),
        ({"score": None}, "requires a score"),
        ({"evidence_ids": []}, "at least one evidence"),
        ({"evidence_ids": ["trace-2"]}, "unknown evidence ID"),
        ({"evidence_ids": ["trace-1", "trace-1"]}, "unique"),
        ({"feedback": None}, "requires behavioral feedback"),
        ({"extra": True}, "Extra inputs"),
    )
    for changes, match in invalid:
        with pytest.raises(ValueError, match=match):
            parse_merged_verdict(
                {**_valid_merged_payload(), **changes},
                allowed_evidence_ids=("trace-1",),
            )


def test_merged_insufficient_evidence_combination_is_fail_closed():
    payload = {
        "schema_version": 1,
        "status": "insufficient_evidence",
        "score": None,
        "rationale": "The covered session does not contain assessable behavior.",
        "evidence_ids": [],
        "feedback": None,
    }
    parsed = parse_merged_verdict(payload, allowed_evidence_ids=("trace-1",))
    assert parsed.status == "insufficient_evidence"

    invalid = (
        ({"score": 0.0}, "null score"),
        ({"evidence_ids": ["trace-1"]}, "empty evidence"),
        ({"feedback": _valid_merged_payload()["feedback"]}, "null feedback"),
    )
    for changes, match in invalid:
        with pytest.raises(ValueError, match=match):
            parse_merged_verdict(
                {**payload, **changes},
                allowed_evidence_ids=("trace-1",),
            )


def test_contract_parsers_reject_invalid_allowed_evidence_ids():
    for allowed in (("trace-1", "trace-1"), (" ",)):
        with pytest.raises(ValueError):
            parse_merged_verdict(_valid_merged_payload(), allowed_evidence_ids=allowed)
