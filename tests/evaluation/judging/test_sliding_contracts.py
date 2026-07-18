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
    bind_chunk_digest_schema,
    bind_merged_verdict_schema,
    bind_window_findings_schema,
    parse_behavioral_feedback,
    parse_chunk_digest,
    parse_merged_verdict,
    parse_window_finding,
    parse_window_findings,
    render_window_findings,
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


def test_structured_schemas_advertise_parser_bounds_to_models():
    digest = CHUNK_DIGEST_SCHEMA.schema["properties"]
    finding = WINDOW_FINDING_SCHEMA.schema["properties"]
    feedback = BEHAVIORAL_FEEDBACK_SCHEMA.schema["properties"]
    verdict = MERGED_VERDICT_SCHEMA.schema["properties"]

    assert digest["text"]["maxLength"] == 2_400
    assert digest["evidence_ids"]["minItems"] == 1
    assert "uniqueItems" not in digest["evidence_ids"]
    assert finding["observation"]["maxLength"] == 350
    assert finding["evidence_ids"]["minItems"] == 1
    assert "uniqueItems" not in finding["evidence_ids"]
    assert "minItems" not in WINDOW_FINDINGS_SCHEMA.schema["properties"]["findings"]
    assert feedback["success"]["anyOf"][0]["maxLength"] == 10_000
    assert feedback["problem"]["anyOf"][0]["maxLength"] == 10_000
    assert feedback["desired_behavior"]["anyOf"][0]["maxLength"] == 10_000
    assert verdict["score"]["enum"] == [0.0, 0.25, 0.5, 0.75, 1.0, None]
    assert verdict["score"]["anyOf"] == [{"type": "number"}, {"type": "null"}]


def test_digest_schema_binds_exact_chunk_and_evidence_ids():
    schema = bind_chunk_digest_schema("chunk-7", ("trace-1", "span-2"))

    assert schema.name == "chunk_digest"
    assert schema.schema["properties"]["chunk_id"]["const"] == "chunk-7"
    assert schema.schema["properties"]["evidence_ids"]["items"]["enum"] == [
        "trace-1",
        "span-2",
    ]
    assert "const" not in CHUNK_DIGEST_SCHEMA.schema["properties"]["chunk_id"]


def test_window_schema_binds_exact_window_and_evidence_ids():
    schema = bind_window_findings_schema("window-3", ("trace-2", "tool-4"))

    assert schema.name == "window_findings"
    assert schema.schema["properties"]["window_id"]["const"] == "window-3"
    finding = schema.schema["$defs"]["WindowFinding"]["properties"]
    assert finding["evidence_ids"]["items"]["enum"] == ["trace-2", "tool-4"]
    assert (
        "enum"
        not in WINDOW_FINDINGS_SCHEMA.schema["$defs"]["WindowFinding"]["properties"][
            "evidence_ids"
        ]["items"]
    )


def test_merge_schema_binds_exact_session_evidence_ids():
    schema = bind_merged_verdict_schema(("trace-1", "trace-2"))

    assert schema.name == "merged_verdict"
    assert schema.schema["properties"]["evidence_ids"]["items"]["enum"] == [
        "trace-1",
        "trace-2",
    ]


@pytest.mark.parametrize(
    "factory,args",
    [
        (bind_chunk_digest_schema, ("chunk-1", ())),
        (bind_window_findings_schema, ("window-1", ("trace-1", "trace-1"))),
        (bind_merged_verdict_schema, ((" ",),)),
    ],
)
def test_bound_schemas_reject_empty_blank_or_duplicate_evidence(factory, args):
    with pytest.raises(ValueError):
        factory(*args)


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
        (_valid_finding(observation=" "), "nonblank"),
        (_valid_finding(observation="x" * 351), "at most 350"),
        (_valid_finding(extra="forbidden"), "Extra inputs"),
    )
    for payload, match in invalid:
        with pytest.raises(ValueError, match=match):
            parse_window_finding(payload, allowed_evidence_ids=("trace-1",))


def test_window_findings_canonicalizes_local_ids_and_rejects_excess_findings():
    payload = {
        "schema_version": 1,
        "window_id": "window-1",
        "findings": [
            _valid_finding(finding_id="reused", observation="First observation."),
            _valid_finding(finding_id="reused", observation="Second observation."),
        ],
    }
    parsed = parse_window_findings(
        payload,
        allowed_evidence_ids=("trace-1",),
        max_tokens=750,
    )
    assert [finding.finding_id for finding in parsed.findings] == ["finding-1", "finding-2"]

    payload["findings"] = [_valid_finding(finding_id=f"finding-{index}") for index in range(5)]
    with pytest.raises(ValueError, match="at most 4"):
        parse_window_findings(
            payload,
            allowed_evidence_ids=("trace-1",),
            max_tokens=750,
        )


def test_duplicate_evidence_count_does_not_hide_duplicate_window_findings():
    first = _valid_finding(finding_id="finding-1", evidence_ids=["trace-1"])
    second = _valid_finding(
        finding_id="finding-2",
        evidence_ids=["trace-1", "trace-1"],
    )

    with pytest.raises(ValueError, match="duplicate findings"):
        parse_window_findings(
            {
                "schema_version": 1,
                "window_id": "window-1",
                "findings": [first, second],
            },
            allowed_evidence_ids=("trace-1",),
            max_tokens=100,
        )


def test_window_findings_rejects_duplicates_with_reversed_citation_order():
    payload = {
        "schema_version": 1,
        "window_id": "window-1",
        "findings": [
            _valid_finding(
                finding_id="finding-1",
                evidence_ids=["trace-1", "trace-2"],
            ),
            _valid_finding(
                finding_id="finding-2",
                evidence_ids=["trace-2", "trace-1"],
            ),
        ],
    }

    with pytest.raises(ValueError, match="duplicate findings"):
        parse_window_findings(
            payload,
            allowed_evidence_ids=("trace-1", "trace-2"),
            max_tokens=750,
        )


def test_window_findings_rejects_complete_canonical_artifact_over_token_limit():
    evidence_ids = tuple(f"evidence-{index}-{'x' * 48}" for index in range(20))
    payload = {
        "schema_version": 1,
        "window_id": f"window-{'w' * 120}",
        "findings": [
            _valid_finding(
                finding_id=f"finding-{index}-{'f' * 80}",
                observation=f"{index} {'o' * 340}",
                evidence_ids=list(evidence_ids),
            )
            for index in range(4)
        ],
    }

    with pytest.raises(ValueError, match="complete window findings artifact.*token limit"):
        parse_window_findings(
            payload,
            allowed_evidence_ids=evidence_ids,
            max_tokens=750,
        )


@pytest.mark.parametrize("max_tokens", [True, 0, -1, 1.5])
def test_window_findings_requires_strict_positive_token_limit(max_tokens: object):
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        parse_window_findings(
            {"schema_version": 1, "window_id": "window-1", "findings": []},
            allowed_evidence_ids=("trace-1",),
            max_tokens=max_tokens,  # type: ignore[arg-type]
        )


def test_window_findings_canonical_render_is_normalized_and_deterministic():
    parsed = parse_window_findings(
        {
            "schema_version": 1,
            "window_id": "window-1",
            "findings": [_valid_finding(observation="Observed   behavior.")],
        },
        allowed_evidence_ids=("trace-1",),
        max_tokens=750,
    )

    assert render_window_findings(parsed) == (
        '{"findings":[{"evidence_ids":["trace-1"],"finding_id":"finding-1",'
        '"observation":"Observed behavior.","polarity":"negative"}],'
        '"schema_version":1,"window_id":"window-1"}'
    )


def test_window_findings_requires_expected_window_and_rejects_extra_fields():
    payload = {"schema_version": 1, "window_id": "window-1", "findings": []}
    parsed = parse_window_findings(
        payload,
        allowed_evidence_ids=("trace-1",),
        max_tokens=750,
        expected_window_id="window-1",
    )
    assert isinstance(parsed, WindowFindings)

    with pytest.raises(ValueError, match="unexpected window ID"):
        parse_window_findings(
            payload,
            allowed_evidence_ids=("trace-1",),
            max_tokens=750,
            expected_window_id="window-2",
        )
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_window_findings(
            {**payload, "extra": True},
            allowed_evidence_ids=("trace-1",),
            max_tokens=750,
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
        (
            {"success": "x" * 10_001, "problem": None, "desired_behavior": None},
            "at most 10000",
        ),
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


@pytest.mark.parametrize(
    "desired_behavior",
    [
        (
            "When an explicit repo-level rule (like AGENTS.md's uv run requirement) "
            "cannot be honored, explain the blocker."
        ),
        "Add a rule to AGENTS.md.",
        "Update CLAUDE.md with the workflow.",
        "Delete the old SKILL.md rule.",
        "Create an instruction file for verification.",
        "Rewrite the prompt file to require checks.",
        "Add a new policy.py file.",
    ],
)
def test_feedback_semantic_wording_does_not_invalidate_structured_verdict(
    desired_behavior: str,
):
    payload = _valid_merged_payload()
    feedback = payload["feedback"]
    assert isinstance(feedback, dict)
    feedback["desired_behavior"] = desired_behavior
    parsed = parse_merged_verdict(payload, allowed_evidence_ids=("trace-1",))

    assert parsed.feedback is not None
    assert parsed.feedback.desired_behavior == desired_behavior


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
        ({"feedback": None}, "requires behavioral feedback"),
        ({"extra": True}, "Extra inputs"),
    )
    for changes, match in invalid:
        with pytest.raises(ValueError, match=match):
            parse_merged_verdict(
                {**_valid_merged_payload(), **changes},
                allowed_evidence_ids=("trace-1",),
            )


def test_contract_parsers_allow_and_retain_duplicate_evidence_ids():
    duplicate_ids = ("trace-1", "trace-1")
    digest = parse_chunk_digest(
        {
            "schema_version": 1,
            "chunk_id": "chunk-1",
            "text": "Observed behavior.",
            "evidence_ids": list(duplicate_ids),
        },
        allowed_evidence_ids=("trace-1",),
        max_tokens=50,
    )
    finding = parse_window_finding(
        _valid_finding(evidence_ids=list(duplicate_ids)),
        allowed_evidence_ids=("trace-1",),
    )
    verdict = parse_merged_verdict(
        {**_valid_merged_payload(), "evidence_ids": list(duplicate_ids)},
        allowed_evidence_ids=("trace-1",),
    )

    assert digest.evidence_ids == duplicate_ids
    assert finding.evidence_ids == duplicate_ids
    assert verdict.evidence_ids == duplicate_ids


def test_merged_insufficient_evidence_status_is_canonicalized():
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

    contradictory = parse_merged_verdict(
        {
            **payload,
            "score": 0.75,
            "evidence_ids": ["unknown-trace"],
            "feedback": _valid_merged_payload()["feedback"],
        },
        allowed_evidence_ids=("trace-1",),
    )
    assert contradictory.score is None
    assert contradictory.evidence_ids == ()
    assert contradictory.feedback is None


def test_contract_parsers_reject_invalid_allowed_evidence_ids():
    for allowed in (("trace-1", "trace-1"), (" ",)):
        with pytest.raises(ValueError):
            parse_merged_verdict(_valid_merged_payload(), allowed_evidence_ids=allowed)
