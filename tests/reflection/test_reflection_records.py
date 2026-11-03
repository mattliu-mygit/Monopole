from __future__ import annotations

import pytest
from pydantic import ValidationError

from weave_agent_signals.routes.models import SuccessfulReflectionResultResponse
from weave_agent_signals.routes.run_views import reflection_result_view
from weave_agent_signals.run_config import ModelDescriptor
from weave_agent_signals.runs.bundles import ScopeDescriptor, bundle_from_content_map
from weave_agent_signals.runs.reflection import ReflectionResult
from weave_agent_signals.runs.reflection_records import (
    EvaluatorRecord,
    ReflectionResultRecord,
)

WRITER = ModelDescriptor(
    id="writer",
    label="Writer",
    provider="codex",
    provider_model="writer",
    family="openai",
    supported_roles=("proposal_writer",),
)
SCOPE = ScopeDescriptor("file", "/project", ("AGENTS.md",))


def _result_dict() -> dict[str, object]:
    baseline = bundle_from_content_map({"AGENTS.md": "old"}, scope=SCOPE)
    candidate = bundle_from_content_map({"AGENTS.md": "new"}, scope=SCOPE)
    baseline_evaluation = {
        "evaluation_id": "evaluation-1",
        "target_revision": baseline.revision,
        "requested_model": "evaluator",
        "requested_family": "meta",
        "requested_backend": "wandb",
        "resolved_model": "resolved-evaluator",
        "resolved_family": "meta",
        "resolved_backend": "wandb",
        "score": 0.3,
        "rationale": "Baseline misses verification.",
        "usage": {"total_tokens": 10},
    }
    candidate_evaluation = {
        **baseline_evaluation,
        "evaluation_id": "evaluation-2",
        "target_revision": candidate.revision,
        "score": 0.8,
        "rationale": "Candidate adds verification.",
    }
    return {
        "baseline": baseline.to_dict(),
        "baseline_score": 0.3,
        "baseline_evaluation": baseline_evaluation,
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "bundle": candidate.to_dict(),
                "score": 0.8,
                "score_delta": 0.5,
                "rationale": "Candidate adds verification.",
                "generation_attempt_id": "attempt-1",
                "requested_writer": WRITER.model_dump(mode="json"),
                "resolved_writer_model": "resolved-writer",
                "resolved_writer_family": "openai",
                "resolved_writer_backend": "codex",
                "evaluation": candidate_evaluation,
            }
        ],
        "generation_attempts": [
            {
                "attempt_id": "attempt-1",
                "number": 1,
                "status": "succeeded",
                "requested_writer": WRITER.model_dump(mode="json"),
                "resolved_model": "resolved-writer",
                "resolved_family": "openai",
                "resolved_backend": "codex",
                "usage": {"total_tokens": 20},
                "candidate_revision": candidate.revision,
                "changed_paths": ["AGENTS.md"],
                "response_digest": None,
                "response_excerpt": None,
                "error_type": None,
                "error": None,
            }
        ],
        "recommended_candidate_id": "candidate-1",
        "baseline_won": False,
        "reason": None,
        "score_basis": "predicted_evaluator",
        "provisional_candidate_id": "candidate-1",
        "challenge": None,
    }


def test_result_normalizes_candidates_into_attempts_and_evaluations() -> None:
    result = ReflectionResultRecord.from_result(ReflectionResult.from_dict(_result_dict()))

    assert len(result.attempts) == 1
    assert result.attempts[0].candidate_id == "candidate-1"
    assert result.attempts[0].bundle is not None
    assert tuple(item.target_id for item in result.evaluations) == (
        "baseline",
        "candidate-1",
    )
    candidate = result.candidate("candidate-1")
    assert candidate.score == 0.8
    assert candidate.score_delta == pytest.approx(0.5)
    assert candidate.rationale == "Candidate adds verification."


def test_normalized_result_projects_evaluator_target_revisions_for_api() -> None:
    result = ReflectionResultRecord.from_result(ReflectionResult.from_dict(_result_dict()))

    view = reflection_result_view(result)
    response = SuccessfulReflectionResultResponse.model_validate(view)

    assert response.baseline_evaluation.target_revision == result.baseline.revision
    assert response.candidates[0].evaluation.target_revision == result.attempts[0].bundle.revision


def test_result_rejects_inconsistent_duplicated_candidate_score() -> None:
    value = _result_dict()
    value["candidates"][0]["score"] = 0.7  # type: ignore[index]

    with pytest.raises(ValueError, match="matching evaluator record"):
        ReflectionResultRecord.from_result(ReflectionResult.from_dict(value))


def test_result_rejects_evaluation_without_a_successful_target() -> None:
    result = ReflectionResultRecord.from_result(ReflectionResult.from_dict(_result_dict()))
    extra = EvaluatorRecord(
        evaluation_id="evaluation-extra",
        target_id="candidate-missing",
        requested_model="evaluator",
        requested_family="meta",
        requested_backend="wandb",
        resolved_model="resolved-evaluator",
        resolved_family="meta",
        resolved_backend="wandb",
        score=0.5,
        rationale="No matching candidate.",
    )

    with pytest.raises(ValidationError, match="evaluation targets"):
        result.model_copy(update={"evaluations": (*result.evaluations, extra)}).model_validate(
            {**result.model_dump(), "evaluations": (*result.evaluations, extra)}
        )
