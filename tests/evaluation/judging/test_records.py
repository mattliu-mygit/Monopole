from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from weave_agent_signals.judges.inference import InferenceResponseDiagnostic, JsonSchemaSpec
from weave_agent_signals.judges.records import (
    JudgeCallAudit,
    JudgeCallRecord,
    PanelResult,
    ReviewerOutcome,
    judge_request_id,
)


def _audit(**overrides: object) -> JudgeCallAudit:
    values = {
        "schema_name": "window_findings",
        "transport_request_count": 1,
    }
    values.update(overrides)
    return JudgeCallAudit.model_validate(values)


def _call(**overrides: object) -> JudgeCallRecord:
    values = {
        "request_id": "sha256:" + "a" * 64,
        "phase": "window",
        "conversation_id": "conversation-1",
        "reviewer_position": 1,
        "requested_model_id": "codex:gpt-5.6-sol",
        "rubric_id": "correctness",
        "status": "succeeded",
        "reusable": True,
        "result": {"score": 0.75},
        "audit": _audit(),
        "created_at": datetime(2026, 7, 17, tzinfo=UTC),
    }
    values.update(overrides)
    return JudgeCallRecord.model_validate(values)


def test_request_id_binds_messages_schema_model_and_options() -> None:
    base = {
        "requested_model_id": "codex:gpt-5.6-sol",
        "provider_model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "evidence A"}],
        "response_schema": JsonSchemaSpec("digest", {"type": "object"}),
        "temperature": 0.0,
        "max_tokens": 1000,
        "reasoning": "default",
        "protocol_version": "3",
    }

    identity = judge_request_id(**base)

    assert identity == judge_request_id(**base)
    assert identity.startswith("sha256:")
    assert identity != judge_request_id(**{**base, "max_tokens": 1001})
    assert identity != judge_request_id(
        **{**base, "messages": [{"role": "user", "content": "evidence B"}]}
    )
    assert identity != judge_request_id(**{**base, "reasoning": "disabled"})


def test_audit_only_success_is_not_reusable() -> None:
    record = _call(reusable=False, result=None)

    assert record.result is None
    with pytest.raises(ValidationError, match="reusable success"):
        _call(reusable=True, result=None)


def test_audit_response_diagnostics_are_optional_and_round_trip() -> None:
    assert _audit().response_diagnostics == ()

    diagnostic = InferenceResponseDiagnostic(
        finish_reason="length",
        usage={"completion_tokens": 10_000},
        completion_details={"reasoning_tokens": 9_500},
        content_characters=0,
    )
    audit = JudgeCallAudit.model_validate_json(
        _audit(response_diagnostics=(diagnostic,)).model_dump_json()
    )

    assert audit.response_diagnostics == (diagnostic,)


def test_failed_call_cannot_be_reused_or_contain_a_result() -> None:
    failed = _call(
        status="failed",
        reusable=False,
        result=None,
        audit=_audit(error_type="QuotaError", message="quota reached"),
    )

    assert failed.status == "failed"
    with pytest.raises(ValidationError, match="failed call"):
        _call(status="failed", reusable=True, result=None)


def test_digest_call_has_no_rubric_scope() -> None:
    assert _call(phase="digest", rubric_id=None).rubric_id is None

    with pytest.raises(ValidationError, match="digest call"):
        _call(phase="digest", rubric_id="correctness")
    with pytest.raises(ValidationError, match="rubric_id"):
        _call(phase="merge", rubric_id=None)


def test_panel_result_references_ordered_canonical_calls() -> None:
    panel = PanelResult(
        conversation_id="conversation-1",
        rubric_id="correctness",
        status="complete",
        rating=0.75,
        successful_count=1,
        minimum=0.75,
        maximum=0.75,
        spread=0.0,
        attempts=(
            ReviewerOutcome(
                position=1,
                requested_model_id="codex:gpt-5.6-sol",
                status="succeeded",
                score=0.75,
                rationale="Good evidence use.",
                evidence_ids=("trace-1",),
                call_ids=("sha256:" + "a" * 64,),
            ),
        ),
    )

    assert panel.attempts[0].call_ids == ("sha256:" + "a" * 64,)
