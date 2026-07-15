"""Thin HTTP boundary for reflection review decisions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from weave_agent_signals.routes import serialize_run
from weave_agent_signals.runs.review import (
    ReviewConflictError,
    ReviewNotFoundError,
    ReviewOperationError,
    ReviewRequestError,
    ReviewService,
)

T = TypeVar("T")
Revision = Annotated[StrictInt, Field(ge=0)]
Identifier = Annotated[StrictStr, Field(min_length=1, max_length=200)]


class SelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: Identifier
    expected_revision: Revision
    discard_draft: StrictBool = False


class DraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: Revision
    expected_draft_revision: StrictStr | None = None
    contents: dict[StrictStr, StrictStr | None]


class DraftResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: Revision
    expected_draft_revision: StrictStr | None = None


class PromoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: Revision
    expected_draft_revision: StrictStr | None = None
    idempotency_key: Identifier
    acknowledge_unevaluated: StrictBool = False
    git_metadata: dict[str, Any] | None = None


class DismissRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: Revision


def call_review(operation: Callable[[], T]) -> T:
    try:
        return operation()
    except ReviewNotFoundError as error:
        raise HTTPException(status_code=404, detail=error.to_detail()) from error
    except ReviewRequestError as error:
        raise HTTPException(status_code=400, detail=error.to_detail()) from error
    except ReviewConflictError as error:
        raise HTTPException(status_code=409, detail=error.to_detail()) from error
    except ReviewOperationError as error:
        raise HTTPException(status_code=500, detail=error.to_detail()) from error


def create_reviews_router(service: ReviewService) -> APIRouter:
    router = APIRouter(prefix="/api/runs", tags=["reviews"])

    @router.put("/{run_id}/reflection_selection")
    def select_candidate(run_id: str, request: SelectionRequest) -> dict:
        return serialize_run(
            call_review(
                lambda: service.select_candidate(
                    run_id,
                    candidate_id=request.candidate_id,
                    expected_revision=request.expected_revision,
                    discard_draft=request.discard_draft,
                )
            )
        )

    @router.put("/{run_id}/reflection_draft")
    def save_draft(run_id: str, request: DraftRequest) -> dict:
        return serialize_run(
            call_review(
                lambda: service.save_draft(
                    run_id,
                    contents=request.contents,
                    expected_revision=request.expected_revision,
                    expected_draft_revision=request.expected_draft_revision,
                )
            )
        )

    @router.delete("/{run_id}/reflection_draft")
    def reset_draft(run_id: str, request: DraftResetRequest) -> dict:
        return serialize_run(
            call_review(
                lambda: service.reset_draft(
                    run_id,
                    expected_revision=request.expected_revision,
                    expected_draft_revision=request.expected_draft_revision,
                )
            )
        )

    @router.post("/{run_id}/promote")
    def promote(run_id: str, request: PromoteRequest) -> dict:
        return serialize_run(
            call_review(
                lambda: service.promote(
                    run_id,
                    promotion_id=request.idempotency_key,
                    expected_revision=request.expected_revision,
                    expected_draft_revision=request.expected_draft_revision,
                    acknowledge_unevaluated=request.acknowledge_unevaluated,
                    git_metadata=request.git_metadata,
                )
            )
        )

    @router.post("/{run_id}/dismiss")
    def dismiss(run_id: str, request: DismissRequest) -> dict:
        return serialize_run(
            call_review(
                lambda: service.dismiss(
                    run_id,
                    expected_revision=request.expected_revision,
                )
            )
        )

    return router
