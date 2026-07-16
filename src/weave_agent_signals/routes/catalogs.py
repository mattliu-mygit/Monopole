"""Versioned model and rubric catalog routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Response

from weave_agent_signals.routes.models import ModelCatalogResponse
from weave_agent_signals.run_config import ModelCatalog, RubricCatalog


def create_catalogs_router(
    *,
    build_model_catalog: Callable[[], ModelCatalog],
    build_rubric_catalog: Callable[[], RubricCatalog],
) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["catalogs"])

    @router.get("/models")
    def get_models(response: Response) -> ModelCatalogResponse:
        response.headers["Cache-Control"] = "no-store"
        return ModelCatalogResponse.model_validate(build_model_catalog().model_dump(mode="json"))

    @router.get("/rubrics")
    def get_rubrics(response: Response) -> RubricCatalog:
        response.headers["Cache-Control"] = "no-store"
        return build_rubric_catalog()

    return router
