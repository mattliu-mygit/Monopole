from __future__ import annotations

from fastapi.testclient import TestClient

from weave_agent_signals import api as api_module
from weave_agent_signals.api import ApiDependencies, create_app


class _Runs:
    def __init__(self):
        self.recovery_calls = 0

    def list_summaries(self, _limit: int):
        return []

    def recover_interrupted_runs(self):
        self.recovery_calls += 1


class _Reviews:
    def overlay(self, run):
        return run


class _Client:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def _dependencies(*, closed: list[bool] | None = None) -> ApiDependencies:
    return ApiDependencies(
        run_service=_Runs(),
        review_service=_Reviews(),
        client_factory=lambda: _Client(),
        close=(lambda: closed.append(True)) if closed is not None else (lambda: None),
    )


def test_create_app_wires_catalog_run_review_and_inspection_routes(tmp_path):
    dependencies = _dependencies()
    app = create_app(
        dependencies=dependencies,
        frontend_dist=tmp_path / "missing-dist",
    )

    with TestClient(app) as client:
        assert client.get("/api/models").status_code == 200
        assert client.get("/api/rubrics").status_code == 200
        assert client.get("/api/runs").json() == {"runs": []}
        paths = client.get("/openapi.json").json()["paths"]

    assert "/api/sessions" in paths
    assert "/api/runs/{run_id}/reflection_draft" in paths
    assert "/api/runs/{run_id}/promote" in paths
    assert dependencies.run_service.recovery_calls == 1


def test_default_dependencies_are_built_at_startup_not_app_construction(
    monkeypatch,
    tmp_path,
):
    built: list[bool] = []
    closed: list[bool] = []

    def build(**_kwargs):
        built.append(True)
        return _dependencies(closed=closed)

    monkeypatch.setattr(api_module, "_build_default_dependencies", build)
    app = create_app(frontend_dist=tmp_path / "missing-dist")
    assert built == []

    with TestClient(app) as client:
        assert built == [True]
        assert client.get("/api/runs").json() == {"runs": []}

    assert closed == [True]


def test_static_frontend_uses_spa_fallback_without_shadowing_api(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<main>evaluation app</main>")
    (dist / "asset.txt").write_text("asset")
    app = create_app(dependencies=_dependencies(), frontend_dist=dist)

    with TestClient(app) as client:
        assert client.get("/asset.txt").text == "asset"
        assert client.get("/runs/run-1").text == "<main>evaluation app</main>"
        assert client.get("/api/runs").json() == {"runs": []}
