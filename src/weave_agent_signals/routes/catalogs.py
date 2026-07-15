"""Versioned model and rubric catalog routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from weave_agent_signals.run_config import ModelCatalog, RubricCatalog


def create_catalogs_router(
    *,
    build_model_catalog: Callable[[], ModelCatalog],
    build_rubric_catalog: Callable[[], RubricCatalog],
) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["catalogs"])

    @router.get("/models")
    def get_models() -> JSONResponse:
        return JSONResponse(
            build_model_catalog().model_dump(mode="json"),
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/rubrics")
    def get_rubrics() -> JSONResponse:
        return JSONResponse(
            build_rubric_catalog().model_dump(mode="json"),
            headers={"Cache-Control": "no-store"},
        )

    return router
