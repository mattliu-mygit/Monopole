"""Evaluation-run setup and lifecycle routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, model_validator

from weave_agent_signals.routes import serialize_run, serialize_run_summary
from weave_agent_signals.routes._time import DateFilterError, parse_selection_bounds
from weave_agent_signals.routes.models import RunListResponse, RunResponse
from weave_agent_signals.routes.reviews import call_review
from weave_agent_signals.run_config import RunConfig
from weave_agent_signals.runs.review import ReviewService
from weave_agent_signals.runs.service import (
    RunAdvanceError,
    RunNotFoundError,
    RunService,
)
from weave_agent_signals.runs.store import (
    DataSelection,
    RunCancellationConflictError,
    RunLifecycleConflictError,
    RunStoreConflictError,
)

T = TypeVar("T")


class SelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    since: StrictStr | None = None
    until: StrictStr | None = None
    timezone: StrictStr | None = None
    session_ids: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def validate_selection(self) -> SelectionRequest:
        if not self.session_ids:
            raise ValueError("Select at least one session")
        if any(not session_id.strip() for session_id in self.session_ids):
            raise ValueError("Session IDs must be nonblank")
        if len(self.session_ids) != len(set(self.session_ids)):
            raise ValueError("Session IDs must be unique")
        try:
            parse_selection_bounds(self.since, self.until, self.timezone)
        except DateFilterError as error:
            raise ValueError(str(error)) from error
        return self


class AutoRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_run: StrictBool


def _detail(code: str, error: Exception, **values: object) -> dict:
    return {"code": code, "message": str(error), **values}


def _call(operation: Callable[[], T]) -> T:
    try:
        return operation()
    except RunNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=_detail("run_not_found", error, run_id=error.run_id),
        ) from error
    except RunCancellationConflictError as error:
        raise HTTPException(
            status_code=409,
            detail=_detail(
                "run_cancellation_conflict",
                error,
                run_id=error.run_id,
                current_status=error.current_status.value,
                reflection_finalizing=error.reflection_finalizing,
            ),
        ) from error
    except RunLifecycleConflictError as error:
        raise HTTPException(
            status_code=409,
            detail=_detail(
                "run_lifecycle_conflict",
                error,
                run_id=error.run_id,
                current_status=error.current_status.value,
            ),
        ) from error
    except RunStoreConflictError as error:
        raise HTTPException(
            status_code=409,
            detail=_detail("run_conflict", error),
        ) from error
    except RunAdvanceError as error:
        raise HTTPException(
            status_code=400,
            detail=_detail("run_not_ready", error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=_detail("invalid_run_request", error),
        ) from error


def create_runs_router(
    service: RunService,
    review_service: ReviewService | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/runs", tags=["runs"])

    def response(run):
        return RunResponse.model_validate(serialize_run(run, getattr(service, "store", None)))

    @router.post("")
    def create_run() -> RunResponse:
        return response(_call(service.create))

    @router.get("")
    def list_runs(limit: int = Query(default=50, ge=1, le=200)) -> RunListResponse:
        summaries = _call(lambda: service.list_summaries(limit))
        return RunListResponse.model_validate(
            {"runs": [serialize_run_summary(summary) for summary in summaries]}
        )

    @router.get("/{run_id}")
    def get_run(run_id: str) -> RunResponse:
        run = (
            call_review(lambda: review_service.read(run_id))
            if review_service is not None
            else _call(lambda: service.get(run_id))
        )
        return response(run)

    @router.put("/{run_id}/selection")
    def save_selection(run_id: str, request: SelectionRequest) -> RunResponse:
        selection = DataSelection(
            since=request.since,
            until=request.until,
            timezone=request.timezone,
            session_ids=request.session_ids,
        )
        return response(_call(lambda: service.save_selection(run_id, selection)))

    @router.put("/{run_id}/config")
    def save_config(run_id: str, config: RunConfig) -> RunResponse:
        return response(_call(lambda: service.save_config(run_id, config)))

    @router.put("/{run_id}/auto_run")
    def set_auto_run(run_id: str, request: AutoRunRequest) -> RunResponse:
        return response(_call(lambda: service.set_auto_run(run_id, request.auto_run)))

    @router.post("/{run_id}/advance")
    def advance_run(run_id: str) -> RunResponse:
        return response(_call(lambda: service.advance(run_id)))

    @router.post("/{run_id}/cancel")
    def cancel_run(run_id: str) -> RunResponse:
        return response(_call(lambda: service.cancel(run_id)))

    return router
