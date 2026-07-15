"""Tests for explicit, reviewed turn judging."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import respx

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.digest import (
    JUDGE_DIGEST_CONTRACT_VERSION,
    judge_digest_contract_manifest,
)
from weave_agent_signals.judges.inference import (
    INFERENCE_BASE,
    InferenceCancelled,
    InferenceClient,
    JudgeResponse,
)
from weave_agent_signals.judges.rubrics import RUBRICS, SESSION_RUBRICS
from weave_agent_signals.judges.runner import (
    REVIEW_POLICY_VERSION,
    JudgeExecutionError,
    judge_turn,
)
from weave_agent_signals.judges.verdicts import JUDGE_VERDICT_SCHEMA
from weave_agent_signals.models import ToolSpan, TurnSpan
from weave_agent_signals.run_config import PositionedJudge, RubricDescriptor

_RUBRIC_SECTION_HEADINGS = (
    "## Evaluation question",
    "## Allowed evidence",
    "## Excluded concerns",
    "## Applicability and insufficient evidence",
    "## Score anchors",
    "## Verdict layout",
)
_ANCHORS = ("0", "0.25", "0.5", "0.75", "1")


def _all_rubrics():
    return (*RUBRICS.values(), *SESSION_RUBRICS.values())


def test_rubric_v3_prompts_share_six_ordered_sections_and_exact_verdict_layout() -> None:
    verdict_sections = []

    for rubric in _all_rubrics():
        assert rubric.version == "v3"
        for heading in _RUBRIC_SECTION_HEADINGS:
            assert heading in rubric.system_prompt
        positions = [rubric.system_prompt.index(heading) for heading in _RUBRIC_SECTION_HEADINGS]
        assert positions == sorted(positions)
        assert tuple(rubric.criteria) == _ANCHORS
        assert len(set(rubric.criteria.values())) == len(_ANCHORS)
        for anchor in _ANCHORS:
            assert f"- **{anchor}**:" in rubric.system_prompt
        verdict_sections.append(rubric.system_prompt.split("## Verdict layout\n", 1)[1])

    assert len(set(verdict_sections)) == 1
    verdict_layout = verdict_sections[0]
    assert '"schema_version": 3' in verdict_layout
    assert '"status": "scored"' in verdict_layout
    assert '"status": "insufficient_evidence"' in verdict_layout
    assert '"score": 0.75' in verdict_layout
    assert '"score": null' in verdict_layout
    assert '"rationale"' in verdict_layout
    assert '"evidence"' in verdict_layout
    assert '"id"' in verdict_layout
    assert '"observations"' in verdict_layout
    decoder = json.JSONDecoder()
    scored_text = verdict_layout[verdict_layout.index("{") :]
    scored, _ = decoder.raw_decode(scored_text)
    insufficient_text = verdict_layout.split(
        "For insufficient evidence, use exactly this layout:\n",
        1,
    )[1]
    insufficient, _ = decoder.raw_decode(insufficient_text)
    assert tuple(scored) == ("schema_version", "status", "score", "rationale", "evidence")
    assert tuple(scored["evidence"][0]) == ("id", "observations")
    assert tuple(insufficient) == ("schema_version", "status", "score", "rationale", "evidence")


@pytest.mark.parametrize(
    ("rubric_id", "required_exclusions"),
    [
        ("judge.verification", ("initial tool", "tool choice")),
        ("judge.error_recovery", ("eventual task success",)),
        ("judge.tool_choice", ("later verification", "vendor-specific")),
        ("judge.state_consistency", ("infer prior state",)),
        ("judge.session_outcome", ("steering burden", "session autonomy")),
        (
            "judge.session_autonomy",
            (
                "required approvals",
                "authentication",
                "policy gates",
                "genuinely missing requirements",
                "session outcome",
            ),
        ),
    ],
)
def test_rubric_v3_exclusions_preserve_dimension_boundaries(
    rubric_id: str,
    required_exclusions: tuple[str, ...],
) -> None:
    prompt = {**RUBRICS, **SESSION_RUBRICS}[rubric_id].system_prompt
    excluded_section = prompt.split("## Excluded concerns\n", 1)[1].split(
        "\n## Applicability and insufficient evidence",
        1,
    )[0]

    for exclusion in required_exclusions:
        assert exclusion in excluded_section.lower()


def test_state_consistency_abstains_without_established_prior_state() -> None:
    prompt = RUBRICS["judge.state_consistency"].system_prompt.lower()

    assert "established prior state" in prompt
    assert "must return `insufficient_evidence`" in prompt
    assert "no established prior state" in prompt


def test_verification_scores_affirmative_no_check_evidence_instead_of_abstaining() -> None:
    prompt = RUBRICS["judge.verification"].system_prompt.lower()

    assert "absence of a supporting check in otherwise complete evidence" in prompt
    assert "scorable at `0.25`" in prompt
    assert "missing, truncated, or ambiguous" in prompt


def test_session_outcome_low_anchors_separate_harm_from_partial_value() -> None:
    criteria = SESSION_RUBRICS["judge.session_outcome"].criteria

    assert "unusable" in criteria["0"]
    assert "limited usable fragment" in criteria["0.25"]
    assert "meaningful partial value" in criteria["0.5"]


def test_error_recovery_low_anchors_separate_no_response_from_weak_adaptation() -> None:
    criteria = RUBRICS["judge.error_recovery"].criteria

    assert "no meaningful response" in criteria["0"]
    assert "weak but genuine" in criteria["0.25"]


def test_error_recovery_does_not_require_a_narrated_explanation() -> None:
    criteria = RUBRICS["judge.error_recovery"].criteria

    assert "incomplete explanation" not in criteria["0.75"]


def test_session_autonomy_low_anchors_separate_stall_from_eventual_progress() -> None:
    criteria = SESSION_RUBRICS["judge.session_autonomy"].criteria

    assert "cannot incorporate" in criteria["0"]
    assert "eventually makes progress" in criteria["0.25"]


def test_rubric_v3_prompts_are_platform_neutral() -> None:
    combined = "\n".join(rubric.system_prompt for rubric in _all_rubrics())

    for platform_term in ("Read tool", "Edit tool", "grep", "pytest", "Claude", "Codex"):
        assert platform_term not in combined


def test_review_policy_version_is_unchanged_by_the_v3_verdict_shape() -> None:
    assert REVIEW_POLICY_VERSION == "2"


def test_digest_contract_version_covers_v3_prompt_boundary_deterministically() -> None:
    assert JUDGE_DIGEST_CONTRACT_VERSION == "3.0.0"
    assert judge_digest_contract_manifest() == judge_digest_contract_manifest()


def _ts(h: int = 12, m: int = 0) -> datetime:
    return datetime(2026, 7, 9, h, m, tzinfo=timezone.utc)


def _turn(
    tool_calls: list[ToolSpan] | None = None,
    *,
    model: str | None = "claude-opus-4",
) -> TurnSpan:
    return TurnSpan(
        trace_id="t1",
        conversation_id="c1",
        started_at=_ts(),
        ended_at=_ts(12, 5),
        model=model,
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        status_code="OK",
        config_version="abc",
        git_branch="main",
        effort_level="high",
        session_id="s1",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=tool_calls or [],
        chat_spans=[],
        subagents=[],
    )


def _bash(command: str, result: str = "", status: str = "OK") -> ToolSpan:
    return ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments=f'{{"command": "{command}"}}',
        result=result,
        status_code=status,
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )


def _rubric(rubric_id: str) -> RubricDescriptor:
    return build_rubric_catalog().rubric(rubric_id)


def _judge(
    model_id: str,
    position: int,
    *,
    family: str = "openai",
    backend: str = "cli",
) -> PositionedJudge:
    return PositionedJudge(
        id=model_id,
        label=model_id,
        family=family,
        backend=backend,
        supported_roles=("judge",),
        position=position,
    )


def _scored_verdict(
    score: object,
    rationale: str = "reviewed",
    *,
    evidence_id: str = "t1",
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "status": "scored",
        "score": score,
        "rationale": rationale,
        "evidence": [
            {
                "id": evidence_id,
                "observations": ["The trace contains the behavior used for this verdict."],
            }
        ],
    }


def _insufficient_verdict(rationale: str = "The supplied evidence is insufficient.") -> dict:
    return {
        "schema_version": 3,
        "status": "insufficient_evidence",
        "score": None,
        "rationale": rationale,
        "evidence": [],
    }


def _response(model: str, verdict: dict[str, object]) -> JudgeResponse:
    content = json.dumps(verdict, separators=(",", ":"))
    return JudgeResponse(
        content=content,
        model=f"{model}-resolved",
        usage={"total_tokens": 10},
        output_mode="json_schema",
        schema_name="judge_verdict",
        transport_request_count=1,
        raw_output_digest=hashlib.sha256(content.encode()).hexdigest(),
    )


class _Client:
    backend = "cli"

    def __init__(self, outcomes: dict[str, Sequence[object]]) -> None:
        self.outcomes = {model: list(values) for model, values in outcomes.items()}
        self.calls: list[dict[str, object]] = []

    def chat_json(self, *, model: str, messages: list[dict[str, str]], **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        outcome = self.outcomes[model].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        verdict = outcome if isinstance(outcome, dict) else _scored_verdict(*outcome)
        return verdict, _response(model, verdict)


class _NestedUsageClient:
    backend = "cli"

    def __init__(self, score: object) -> None:
        self.score = score

    def chat_json(self, *, model: str, messages: list[dict[str, str]], **_kwargs):
        verdict = _scored_verdict(self.score)
        return (
            verdict,
            JudgeResponse(
                content=json.dumps(verdict),
                model=f"{model}-resolved",
                usage={
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            ),
        )


class _ProviderMetadataUsageClient:
    backend = "cli"

    def chat_json(self, *, model: str, messages: list[dict[str, str]], **_kwargs):
        verdict = _scored_verdict(0.75)
        return (
            verdict,
            JudgeResponse(
                content=json.dumps(verdict),
                model=f"{model}-resolved",
                usage={
                    "total_tokens": 15,
                    "service_tier": "default",
                    "cached": None,
                    "regions": ["us-west"],
                },
            ),
        )


def _judge_turn(
    client: object,
    *,
    rubrics: Sequence[RubricDescriptor] = (_rubric("judge.verification"),),
    judges: Sequence[PositionedJudge] = (_judge("judge-1", 1),),
    review_depth: str = "primary",
    second_opinion_margin: float | None = None,
    turn: TurnSpan | None = None,
    prior_turns: Sequence[TurnSpan] = (),
):
    return judge_turn(
        turn or _turn(),
        client,
        rubrics=rubrics,
        judges=judges,
        review_depth=review_depth,
        second_opinion_margin=second_opinion_margin,
        prior_turns=prior_turns,
    )


def test_judge_turn_uses_exact_pinned_order_without_family_filtering() -> None:
    judges = (
        _judge("claude-first", 1, family="anthropic"),
        _judge("claude-second", 2, family="anthropic"),
    )
    client = _Client(
        {
            "claude-first": [(0.5, "near")],
            "claude-second": [(0.75, "pass")],
        }
    )

    score = _judge_turn(
        client,
        judges=judges,
        review_depth="selective",
        second_opinion_margin=0.1,
    )[0]

    assert [call["model"] for call in client.calls] == ["claude-first", "claude-second"]
    assert score.value == pytest.approx(0.625)
    assert score.tags == ["verified"]
    assert score.metadata["requested_judge_models"] == [
        "claude-first",
        "claude-second",
    ]
    assert score.metadata["review_status"] == "complete"
    assert [attempt["trigger"] for attempt in score.metadata["attempts"]] == [
        "initial",
        "near_boundary",
    ]


def test_judge_turn_emits_canonical_review_metadata() -> None:
    descriptor = _rubric("judge.verification")
    judge = _judge("judge-1", 1)
    client = _Client({"judge-1": [(0.5, "threshold passes")]})

    score = _judge_turn(client, rubrics=(descriptor,), judges=(judge,))[0]

    assert score.scorer == descriptor.id
    assert score.value == 0.5
    assert score.tags == ["verified"]
    assert score.granularity == "turn"
    assert score.confidence is None
    assert score.reason == "threshold passes"
    assert score.metadata == {
        "rubric_id": descriptor.id,
        "rubric_version": descriptor.version,
        "rubric_threshold": descriptor.pass_threshold,
        "evaluation_unit": "episode",
        "review_depth": "primary",
        "review_policy_version": "2",
        "second_opinion_margin": None,
        "requested_judge_models": ["judge-1"],
        "aggregate": "mean",
        "review_status": "complete",
        "attempt_count": 1,
        "successful_reviewer_count": 1,
        "attempts": [
            {
                "position": 1,
                "role": "judge",
                "trigger": "initial",
                "requested_model": "judge-1",
                "requested_family": "openai",
                "requested_backend": "cli",
                "status": "succeeded",
                "resolved_model": "judge-1-resolved",
                "resolved_family": "unknown",
                "score": 0.5,
                "rationale": "threshold passes",
                "evidence_ids": ["t1"],
                "usage": {"total_tokens": 10},
                "output_mode": "json_schema",
                "schema_name": "judge_verdict",
                "schema_fallback_reason": None,
                "transport_request_count": 1,
                "verdict_schema_version": 3,
                "raw_output_digest": score.metadata["attempts"][0]["raw_output_digest"],
                "error_type": None,
                "message": None,
            }
        ],
        "evaluated_models": ["claude-opus-4"],
        "evaluated_families": ["anthropic"],
        "scored_at": score.metadata["scored_at"],
    }


def test_judge_turn_keeps_flat_token_counts_from_nested_provider_usage() -> None:
    score = _judge_turn(_NestedUsageClient(0.75))[0]

    assert score.metadata["attempts"][0]["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }


def test_judge_turn_ignores_non_counter_provider_usage_metadata() -> None:
    score = _judge_turn(_ProviderMetadataUsageClient())[0]

    assert score.metadata["attempts"][0]["usage"] == {"total_tokens": 15}


def test_judge_turn_normalizes_nested_usage_on_failed_attempt() -> None:
    with pytest.raises(JudgeExecutionError) as caught:
        _judge_turn(_NestedUsageClient("invalid"))

    attempt = caught.value.failures[0].attempts[0]
    assert attempt["status"] == "failed"
    assert attempt["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }
    assert attempt["error_type"] == "ValidationError"


def test_judge_turn_keeps_unresolved_rating_with_only_unresolved_tag() -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2, family="meta"))
    client = _Client(
        {
            "judge-1": [(0.5, "equal is passing")],
            "judge-2": [(0.25, "failing")],
        }
    )

    score = _judge_turn(
        client,
        judges=judges,
        review_depth="selective",
        second_opinion_margin=0.0,
    )[0]

    assert score.value == pytest.approx(0.375)
    assert score.tags == ["unresolved"]
    assert score.metadata["review_status"] == "unresolved"


def test_judge_turn_returns_degraded_rating_and_retains_failed_attempt() -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2, family="meta"))
    client = _Client(
        {
            "judge-1": [RuntimeError("first offline")],
            "judge-2": [(0.75, "recovered")],
        }
    )

    score = _judge_turn(
        client,
        judges=judges,
        review_depth="selective",
        second_opinion_margin=0.1,
    )[0]

    assert score.value == 0.75
    assert score.metadata["review_status"] == "degraded"
    assert score.metadata["attempts"][0] == {
        "position": 1,
        "role": "judge",
        "trigger": "initial",
        "requested_model": "judge-1",
        "requested_family": "openai",
        "requested_backend": "cli",
        "status": "failed",
        "resolved_model": None,
        "resolved_family": None,
        "score": None,
        "rationale": None,
        "evidence_ids": [],
        "usage": {},
        "output_mode": None,
        "schema_name": "judge_verdict",
        "schema_fallback_reason": None,
        "transport_request_count": 1,
        "verdict_schema_version": 3,
        "raw_output_digest": None,
        "error_type": "RuntimeError",
        "message": "judge invocation failed",
    }


def test_judge_turn_raises_only_when_every_reviewer_fails() -> None:
    judges = (
        _judge("judge-1", 1),
        _judge("judge-2", 2, family="meta"),
        _judge("judge-3", 3, family="ibm"),
    )
    client = _Client(
        {
            "judge-1": [RuntimeError("first offline")],
            "judge-2": [RuntimeError("second offline")],
            "judge-3": [RuntimeError("third offline")],
        }
    )

    with pytest.raises(JudgeExecutionError) as caught:
        _judge_turn(
            client,
            judges=judges,
            review_depth="selective",
            second_opinion_margin=0.1,
        )

    assert caught.value.scores == ()
    failure = caught.value.failures[0]
    assert failure.rubric == "judge.verification"
    assert [attempt["requested_model"] for attempt in failure.attempts] == [
        "judge-1",
        "judge-2",
        "judge-3",
    ]
    assert [attempt["trigger"] for attempt in failure.attempts] == [
        "initial",
        "judge_1_failed",
        "both_prior_failed",
    ]


def test_judge_turn_preserves_successful_other_rubrics_on_zero_success() -> None:
    class _MixedClient:
        backend = "cli"

        def chat_json(self, *, model: str, messages: list[dict[str, str]], **_kwargs):
            if "recovered from errors" in messages[0]["content"]:
                raise RuntimeError("error recovery offline")
            return (
                _scored_verdict(0.75, "verified"),
                JudgeResponse(content="", model=model, usage={}),
            )

    with pytest.raises(JudgeExecutionError) as caught:
        _judge_turn(
            _MixedClient(),
            rubrics=(
                _rubric("judge.verification"),
                _rubric("judge.error_recovery"),
            ),
        )

    assert [score.scorer for score in caught.value.scores] == ["judge.verification"]
    assert [failure.rubric for failure in caught.value.failures] == ["judge.error_recovery"]


def test_judge_turn_propagates_cancellation() -> None:
    cancelled = InferenceCancelled("cancelled")
    client = _Client({"judge-1": [cancelled]})

    with pytest.raises(InferenceCancelled) as caught:
        _judge_turn(client)

    assert caught.value is cancelled


def test_judge_turn_rejects_stale_rubric_before_inference() -> None:
    descriptor = _rubric("judge.verification").model_copy(update={"content_digest": "sha256:stale"})
    client = _Client({})

    with pytest.raises(ValueError, match="does not match current prompt content"):
        _judge_turn(client, rubrics=(descriptor,))

    assert client.calls == []


def test_verification_judge_consumes_all_pinned_prior_evidence() -> None:
    prior_turns = [_turn() for _ in range(4)]
    for index, prior in enumerate(prior_turns):
        prior.trace_id = f"prior-{index}"
    prior_turns[0].trace_id = "modification-evidence"
    prior_turns[0].tool_calls = [_bash("apply_patch", "updated")]
    current = _turn([_bash("pytest", "passed")])
    client = _Client({"judge-1": [(0.75, "verified")]})

    _judge_turn(client, turn=current, prior_turns=prior_turns)

    prompt = client.calls[0]["messages"][1]["content"]
    assert "evidence_id=modification-evidence" in prompt


def test_judge_turn_empty_rubrics_is_an_explicit_noop() -> None:
    assert _judge_turn(object(), rubrics=()) == []


def _mock_http_response(score: object, rationale: str = "test rationale") -> dict:
    verdict = _scored_verdict(score, rationale)
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(verdict),
                }
            }
        ],
        "model": "judge-1-resolved",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


@respx.mock
@patch.dict("os.environ", {"WANDB_API_KEY": "test-key"})
def test_judge_rejects_out_of_range_score_instead_of_clamping() -> None:
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json=_mock_http_response(1.5, "over-scored"),
    )
    client = InferenceClient(backend="wandb")
    judge = _judge("judge-1", 1, backend="wandb")

    with pytest.raises(JudgeExecutionError) as caught:
        _judge_turn(client, judges=(judge,))

    assert caught.value.scores == ()
    attempt = caught.value.failures[0].attempts[0]
    assert attempt["status"] == "failed"
    assert attempt["score"] is None
    assert attempt["error_type"] == "ValueError"
    assert "one of" in attempt["message"]


def test_judge_rejects_unknown_evidence_id_and_records_digest() -> None:
    verdict = _scored_verdict(0.75, evidence_id="invented-trace")
    client = _Client({"judge-1": [verdict]})

    with pytest.raises(JudgeExecutionError) as caught:
        _judge_turn(client)

    attempt = caught.value.failures[0].attempts[0]
    assert attempt["status"] == "failed"
    assert attempt["error_type"] == "ValueError"
    assert attempt["raw_output_digest"] == _response("judge-1", verdict).raw_output_digest
    assert "invented-trace" in attempt["message"]
    assert "content" not in attempt


def test_selective_review_advances_after_invalid_verdict() -> None:
    judges = (_judge("judge-1", 1), _judge("judge-2", 2, family="meta"))
    client = _Client(
        {
            "judge-1": [_scored_verdict(1.5, "invalid")],
            "judge-2": [_scored_verdict(0.75, "recovered")],
        }
    )

    score = _judge_turn(
        client,
        judges=judges,
        review_depth="selective",
        second_opinion_margin=0.1,
    )[0]

    assert [call["model"] for call in client.calls] == ["judge-1", "judge-2"]
    assert [attempt["status"] for attempt in score.metadata["attempts"]] == [
        "failed",
        "succeeded",
    ]
    assert score.value == 0.75
    assert score.metadata["review_status"] == "degraded"


def test_selective_review_accepts_repeated_groups_for_one_allowed_evidence_id() -> None:
    grouped_verdict = {
        "schema_version": 3,
        "status": "scored",
        "score": 0.75,
        "rationale": "The same trace supports two distinct observations.",
        "evidence": [
            {"id": "t1", "observations": ["The tool choice was appropriate."]},
            {"id": "t1", "observations": ["The final verification was focused."]},
        ],
    }
    client = _Client(
        {
            "judge-1": [grouped_verdict],
            "judge-2": [(0.25, "must not be requested")],
        }
    )

    score = _judge_turn(
        client,
        judges=(_judge("judge-1", 1), _judge("judge-2", 2, family="meta")),
        review_depth="selective",
        second_opinion_margin=0.1,
    )[0]

    assert [call["model"] for call in client.calls] == ["judge-1"]
    assert score.metadata["attempts"][0]["evidence_ids"] == ["t1"]
    assert score.value == 0.75
    assert score.metadata["review_status"] == "complete"


def test_all_selected_judges_invalid_or_abstained_write_no_score() -> None:
    judges = (
        _judge("judge-1", 1),
        _judge("judge-2", 2, family="meta"),
        _judge("judge-3", 3, family="ibm"),
    )
    client = _Client(
        {
            "judge-1": [_scored_verdict(1.5, "invalid")],
            "judge-2": [_insufficient_verdict()],
            "judge-3": [_scored_verdict(0.75, evidence_id="invented-trace")],
        }
    )

    with pytest.raises(JudgeExecutionError) as caught:
        _judge_turn(
            client,
            judges=judges,
            review_depth="selective",
            second_opinion_margin=0.1,
        )

    assert caught.value.scores == ()
    assert [attempt["status"] for attempt in caught.value.failures[0].attempts] == [
        "failed",
        "abstained",
        "failed",
    ]


def test_attempt_persists_schema_and_transport_audit_without_raw_output() -> None:
    raw_output = "sensitive raw judge output that must never be persisted"
    raw_output_digest = hashlib.sha256(raw_output.encode()).hexdigest()

    class _AuditClient:
        backend = "cli"

        def __init__(self) -> None:
            self.response_schema = None

        def chat_json(self, *, model, messages, response_schema=None, **_kwargs):
            self.response_schema = response_schema
            return (
                _scored_verdict(0.75),
                JudgeResponse(
                    content=raw_output,
                    model=f"{model}-resolved",
                    usage={"input_tokens": 4, "output_tokens": 2},
                    output_mode="json_object_fallback",
                    schema_name="judge_verdict",
                    schema_fallback_reason="schema mode unsupported by pinned model",
                    transport_request_count=2,
                    raw_output_digest=raw_output_digest,
                ),
            )

    client = _AuditClient()
    score = _judge_turn(client)[0]
    attempt = score.metadata["attempts"][0]

    assert client.response_schema is JUDGE_VERDICT_SCHEMA
    assert attempt["resolved_model"] == "judge-1-resolved"
    assert attempt["usage"] == {"input_tokens": 4, "output_tokens": 2}
    assert attempt["output_mode"] == "json_object_fallback"
    assert attempt["schema_name"] == "judge_verdict"
    assert attempt["schema_fallback_reason"] == "schema mode unsupported by pinned model"
    assert attempt["transport_request_count"] == 2
    assert attempt["verdict_schema_version"] == 3
    assert attempt["evidence_ids"] == ["t1"]
    assert attempt["raw_output_digest"] == raw_output_digest
    assert "content" not in attempt
    assert raw_output not in repr(score.metadata)


def test_failed_attempt_sanitizes_transport_error_without_raw_output() -> None:
    raw_output = "sensitive raw CLI output"
    error = RuntimeError(f"codex judge exited 1: {raw_output}")
    error._transport_request_count = 3  # type: ignore[attr-defined]

    with pytest.raises(JudgeExecutionError) as caught:
        _judge_turn(_Client({"judge-1": [error]}))

    attempt = caught.value.failures[0].attempts[0]
    assert attempt["error_type"] == "RuntimeError"
    assert attempt["message"] == "judge invocation failed"
    assert attempt["transport_request_count"] == 3
    assert raw_output not in repr(attempt)
