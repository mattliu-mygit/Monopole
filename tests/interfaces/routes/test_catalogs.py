from fastapi import FastAPI
from fastapi.testclient import TestClient

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.routes.catalogs import create_catalogs_router


def test_catalog_routes_return_exact_versioned_snapshots_without_caching():
    models = build_model_catalog(which=lambda name: f"/bin/{name}")
    rubrics = build_rubric_catalog()
    app = FastAPI()
    app.include_router(
        create_catalogs_router(
            build_model_catalog=lambda: models,
            build_rubric_catalog=lambda: rubrics,
        )
    )
    client = TestClient(app)

    model_response = client.get("/api/models")
    rubric_response = client.get("/api/rubrics")

    assert model_response.status_code == 200
    assert model_response.headers["cache-control"] == "no-store"
    assert model_response.json() == models.model_dump(mode="json")
    assert rubric_response.status_code == 200
    assert rubric_response.headers["cache-control"] == "no-store"
    assert rubric_response.json() == rubrics.model_dump(mode="json")
