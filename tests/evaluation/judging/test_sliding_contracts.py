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
        "finding_ids": ["window-1:finding-1"],
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


def test_sliding_contract_schemas_are_closed():
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


def test_model_output_schemas_omit_host_owned_envelope_fields():
    assert "schema_version" not in CHUNK_DIGEST_SCHEMA.schema["properties"]
    assert "chunk_id" not in CHUNK_DIGEST_SCHEMA.schema["properties"]
    assert "evidence_ids" not in CHUNK_DIGEST_SCHEMA.schema["properties"]
    assert "schema_version" not in WINDOW_FINDINGS_SCHEMA.schema["properties"]
    assert "window_id" not in WINDOW_FINDINGS_SCHEMA.schema["properties"]
    assert "schema_version" not in MERGED_VERDICT_SCHEMA.schema["properties"]


def test_parsers_attach_host_owned_envelope_fields():
    digest = parse_chunk_digest(
        {"text": "Observed behavior."},
        max_tokens=100,
        expected_chunk_id="chunk-1",
    )
    findings = parse_window_findings(
        {"findings": []},
        evidence_aliases={"e1": "trace-1"},
        raw_text="",
        max_tokens=100,
        expected_window_id="window-1",
    )
    verdict = parse_merged_verdict(
        {
            "status": "insufficient_evidence",
            "score": None,
            "rationale": "Insufficient evidence.",
            "finding_ids": [],
            "feedback": None,
        },
        finding_evidence={},
    )

    assert (digest.schema_version, digest.chunk_id) == (1, "chunk-1")
    assert (findings.schema_version, findings.window_id) == (1, "window-1")
    assert verdict.schema_version == 1


def test_structured_schemas_advertise_parser_bounds_to_models():
    digest = CHUNK_DIGEST_SCHEMA.schema["properties"]
    finding = WINDOW_FINDING_SCHEMA.schema["properties"]
    feedback = BEHAVIORAL_FEEDBACK_SCHEMA.schema["properties"]
    verdict = MERGED_VERDICT_SCHEMA.schema["properties"]

    assert digest["text"]["maxLength"] == 2_400
    assert "maxLength" not in finding["observation"]
    assert finding["evidence_ids"]["minItems"] == 1
    assert "uniqueItems" not in finding["evidence_ids"]
    assert "minItems" not in WINDOW_FINDINGS_SCHEMA.schema["properties"]["findings"]
    assert feedback["success"]["anyOf"][0]["maxLength"] == 10_000
    assert feedback["problem"]["anyOf"][0]["maxLength"] == 10_000
    assert feedback["desired_behavior"]["anyOf"][0]["maxLength"] == 10_000
    assert verdict["score"]["enum"] == [0.0, 0.25, 0.5, 0.75, 1.0, None]
    assert verdict["score"]["anyOf"] == [{"type": "number"}, {"type": "null"}]


def test_window_schema_binds_exact_evidence_ids():
    schema = bind_window_findings_schema(("trace-2", "tool-4"))

    assert schema.name == "window_findings"
    finding = schema.schema["$defs"]["WindowFinding"]["properties"]
    assert finding["evidence_ids"]["items"]["enum"] == ["trace-2", "tool-4"]
    assert (
        "enum"
        not in WINDOW_FINDINGS_SCHEMA.schema["$defs"]["WindowFinding"]["properties"][
            "evidence_ids"
        ]["items"]
    )


def test_merge_schema_binds_exact_finding_ids():
    schema = bind_merged_verdict_schema(("window-1:finding-1", "window-2:finding-1"))

    assert schema.name == "merged_verdict"
    assert "evidence_ids" not in schema.schema["properties"]
    assert schema.schema["properties"]["finding_ids"]["items"]["enum"] == [
        "window-1:finding-1",
        "window-2:finding-1",
    ]


def test_merge_schema_allows_empty_finding_list_for_abstention():
    schema = bind_merged_verdict_schema(())

    assert "enum" not in schema.schema["properties"]["finding_ids"]["items"]


@pytest.mark.parametrize(
    "factory,args",
    [
        (bind_window_findings_schema, (("trace-1", "trace-1"),)),
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
        },
        max_tokens=20,
    )
    assert isinstance(parsed, ChunkDigest)
    assert parsed.text == "Observed recovery after a failed check."


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"text": "  \n "}, "nonblank"),
        ({"extra": "forbidden"}, "Extra inputs"),
    ],
)
def test_chunk_digest_rejects_open_or_uncited_content(changes: dict[str, object], match: str):
    payload: dict[str, object] = {
        "schema_version": 1,
        "chunk_id": "chunk-1",
        "text": "Observed behavior.",
    }
    payload.update(changes)
    with pytest.raises(ValueError, match=match):
        parse_chunk_digest(
            payload,
            max_tokens=50,
        )


def test_window_findings_resolve_short_aliases_and_drop_unverified_quote():
    parsed = parse_window_findings(
        {
            "findings": [
                _valid_finding(
                    evidence_ids=["e2"],
                    quote="a phrase that is not in the source",
                )
            ]
        },
        evidence_aliases={"e1": "trace-1", "e2": "tool-2"},
        raw_text="The tool reported success.",
        max_tokens=100,
        expected_window_id="window-1",
    )

    assert parsed.findings[0].evidence_ids == ("tool-2",)
    assert parsed.findings[0].quote is None


def test_window_findings_keep_exact_source_quote():
    parsed = parse_window_findings(
        {
            "findings": [
                _valid_finding(
                    evidence_ids=["e1"],
                    quote="tool reported success",
                )
            ]
        },
        evidence_aliases={"e1": "trace-1"},
        raw_text="The tool reported success.",
        max_tokens=100,
        expected_window_id="window-1",
    )

    assert parsed.findings[0].quote == "tool reported success"


def test_chunk_digest_rejects_text_over_token_limit():
    with pytest.raises(ValueError, match="token limit"):
        parse_chunk_digest(
            {
                "schema_version": 1,
                "chunk_id": "chunk-1",
                "text": "A digest that cannot fit in one token.",
            },
            max_tokens=1,
        )


def test_window_finding_parses_normalized_bounded_observation():
    parsed = parse_window_finding(
        _valid_finding(observation="The agent  did not\n rerun the check."),
        evidence_aliases={"trace-1": "trace-1"},
        raw_text="",
    )
    assert isinstance(parsed, WindowFinding)
    assert parsed.observation == "The agent did not rerun the check."


@pytest.mark.parametrize("polarity", ["neutral", "Positive", "", 1])
def test_window_finding_rejects_invalid_polarity(polarity: object):
    with pytest.raises(ValueError):
        parse_window_finding(
            _valid_finding(polarity=polarity),
            evidence_aliases={"trace-1": "trace-1"},
            raw_text="",
        )


def test_window_finding_rejects_unknown_blank_and_invalid_content():
    invalid = (
        (_valid_finding(evidence_ids=["trace-2"]), "unknown evidence ID"),
        (_valid_finding(evidence_ids=[" trace-1 "]), "unknown evidence ID"),
        (_valid_finding(evidence_ids=[" "]), "nonblank"),
        (_valid_finding(observation=" "), "nonblank"),
        (_valid_finding(extra="forbidden"), "Extra inputs"),
    )
    for payload, match in invalid:
        with pytest.raises(ValueError, match=match):
            parse_window_finding(
                payload,
                evidence_aliases={"trace-1": "trace-1"},
                raw_text="",
            )


def test_window_finding_allows_long_observation_within_artifact_token_budget():
    parsed = parse_window_findings(
        {
            "schema_version": 1,
            "window_id": "window-1",
            "findings": [_valid_finding(observation="x" * 2_000)],
        },
        evidence_aliases={"trace-1": "trace-1"},
        raw_text="",
        max_tokens=4_000,
    )

    assert parsed.findings[0].observation == "x" * 2_000


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
        evidence_aliases={"trace-1": "trace-1"},
        raw_text="",
        max_tokens=750,
    )
    assert [finding.finding_id for finding in parsed.findings] == ["finding-1", "finding-2"]

    payload["findings"] = [_valid_finding(finding_id=f"finding-{index}") for index in range(5)]
    with pytest.raises(ValueError, match="at most 4"):
        parse_window_findings(
            payload,
            evidence_aliases={"trace-1": "trace-1"},
            raw_text="",
            max_tokens=750,
        )


def test_window_findings_deduplicates_repeated_evidence_semantics():
    first = _valid_finding(finding_id="finding-1", evidence_ids=["trace-1"])
    second = _valid_finding(
        finding_id="finding-2",
        evidence_ids=["trace-1", "trace-1"],
    )

    parsed = parse_window_findings(
        {
            "schema_version": 1,
            "window_id": "window-1",
            "findings": [first, second],
        },
        evidence_aliases={"trace-1": "trace-1"},
        raw_text="",
        max_tokens=100,
    )

    assert [finding.finding_id for finding in parsed.findings] == ["finding-1"]


def test_window_findings_deduplicates_reversed_citation_order():
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

    parsed = parse_window_findings(
        payload,
        evidence_aliases={"trace-1": "trace-1", "trace-2": "trace-2"},
        raw_text="",
        max_tokens=750,
    )

    assert [finding.finding_id for finding in parsed.findings] == ["finding-1"]


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
            evidence_aliases={evidence_id: evidence_id for evidence_id in evidence_ids},
            raw_text="",
            max_tokens=750,
        )


@pytest.mark.parametrize("max_tokens", [True, 0, -1, 1.5])
def test_window_findings_requires_strict_positive_token_limit(max_tokens: object):
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        parse_window_findings(
            {"schema_version": 1, "window_id": "window-1", "findings": []},
            evidence_aliases={"trace-1": "trace-1"},
            raw_text="",
            max_tokens=max_tokens,  # type: ignore[arg-type]
        )


def test_window_findings_canonical_render_is_normalized_and_deterministic():
    parsed = parse_window_findings(
        {
            "schema_version": 1,
            "window_id": "window-1",
            "findings": [_valid_finding(observation="Observed   behavior.")],
        },
        evidence_aliases={"trace-1": "trace-1"},
        raw_text="",
        max_tokens=750,
    )

    assert render_window_findings(parsed) == (
        '{"findings":[{"evidence_ids":["trace-1"],"finding_id":"finding-1",'
        '"observation":"Observed behavior.","polarity":"negative"}],'
        '"schema_version":1,"window_id":"window-1"}'
    )


def test_window_findings_uses_expected_window_and_rejects_extra_fields():
    payload = {"schema_version": 1, "window_id": "window-1", "findings": []}
    parsed = parse_window_findings(
        payload,
        evidence_aliases={"trace-1": "trace-1"},
        raw_text="",
        max_tokens=750,
        expected_window_id="window-1",
    )
    assert isinstance(parsed, WindowFindings)

    replaced = parse_window_findings(
        payload,
        evidence_aliases={"trace-1": "trace-1"},
        raw_text="",
        max_tokens=750,
        expected_window_id="window-2",
    )
    assert replaced.window_id == "window-2"
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_window_findings(
            {**payload, "extra": True},
            evidence_aliases={"trace-1": "trace-1"},
            raw_text="",
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
            "finding_ids": ["window-1:finding-1"],
            "feedback": {
                "success": "It changed approach after the failure.",
                "problem": "It claimed completion without rerunning the failed check.",
                "desired_behavior": "Rerun relevant checks after the final change.",
            },
        },
        finding_evidence={"window-1:finding-1": ("trace-2", "trace-3")},
    )
    assert isinstance(parsed, MergedVerdict)
    assert parsed.feedback is not None
    assert parsed.feedback.desired_behavior == "Rerun relevant checks after the final change."
    assert parsed.evidence_ids == ("trace-2", "trace-3")


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
    parsed = parse_merged_verdict(
        payload,
        finding_evidence={"window-1:finding-1": ("trace-1",)},
    )

    assert parsed.feedback is not None
    assert parsed.feedback.desired_behavior == desired_behavior


def test_merged_scored_verdict_requires_anchor_citations_and_feedback():
    for score in (0.0, 0.25, 0.5, 0.75, 1.0):
        parsed = parse_merged_verdict(
            {**_valid_merged_payload(), "score": score},
            finding_evidence={"window-1:finding-1": ("trace-1",)},
        )
        assert parsed.score == score

    invalid = (
        ({"score": 0.1}, "score must be one of"),
        ({"score": math.inf}, "score must be one of"),
        ({"score": None}, "requires a score"),
        ({"finding_ids": []}, "at least one finding"),
        ({"finding_ids": ["unknown-finding"]}, "unknown finding ID"),
        ({"feedback": None}, "requires behavioral feedback"),
        ({"extra": True}, "Extra inputs"),
    )
    for changes, match in invalid:
        with pytest.raises(ValueError, match=match):
            parse_merged_verdict(
                {**_valid_merged_payload(), **changes},
                finding_evidence={"window-1:finding-1": ("trace-1",)},
            )


def test_contract_parsers_allow_duplicate_alias_and_finding_citations():
    finding = parse_window_finding(
        _valid_finding(evidence_ids=["e1", "e1"]),
        evidence_aliases={"e1": "trace-1"},
        raw_text="",
    )
    verdict = parse_merged_verdict(
        {
            **_valid_merged_payload(),
            "finding_ids": ["window-1:finding-1", "window-1:finding-1"],
        },
        finding_evidence={"window-1:finding-1": ("trace-1",)},
    )

    assert finding.evidence_ids == ("trace-1", "trace-1")
    assert verdict.evidence_ids == ("trace-1", "trace-1")


def test_merged_insufficient_evidence_status_is_canonicalized():
    payload = {
        "schema_version": 1,
        "status": "insufficient_evidence",
        "score": None,
        "rationale": "The covered session does not contain assessable behavior.",
        "finding_ids": [],
        "feedback": None,
    }
    parsed = parse_merged_verdict(payload, finding_evidence={})
    assert parsed.status == "insufficient_evidence"

    contradictory = parse_merged_verdict(
        {
            **payload,
            "score": 0.75,
            "finding_ids": ["unknown-finding"],
            "feedback": _valid_merged_payload()["feedback"],
        },
        finding_evidence={},
    )
    assert contradictory.score is None
    assert contradictory.evidence_ids == ()
    assert contradictory.feedback is None


def test_contract_parsers_reject_invalid_finding_evidence():
    for finding_evidence in ({" ": ("trace-1",)}, {"finding-1": (" ",)}):
        with pytest.raises(ValueError):
            parse_merged_verdict(_valid_merged_payload(), finding_evidence=finding_evidence)
