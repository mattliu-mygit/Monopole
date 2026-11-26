from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from weave_agent_signals import api as api_module
from weave_agent_signals.api import ApiDependencies, create_app
from weave_agent_signals.models import ConfigurationError
from weave_agent_signals.runs.store import RunStatus


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
        ces_factory=lambda: _Client(),
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


def test_an_unknown_api_path_is_a_404_not_the_app_shell(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<main>evaluation app</main>")
    app = create_app(dependencies=_dependencies(), frontend_dist=dist)

    with TestClient(app) as client:
        assert client.get("/api/no-such-route").status_code == 404


def test_scoring_hydrates_whole_subagents_and_judging_only_their_counts(monkeypatch, tmp_path):
    hydrated: list[bool] = []
    monkeypatch.setattr(
        api_module,
        "hydrate_turn_cohort",
        lambda _cohort, *, ces_factory, subagent_calls=True: hydrated.append(subagent_calls),
    )
    stages = {}
    for name in ("run_scoring_stage", "run_judging_stage"):
        monkeypatch.setattr(
            api_module,
            name,
            lambda run, config, cancel, *, dependencies, name=name: stages.update(
                {name: dependencies}
            ),
        )
    for name in ("load_target_registry", "load_sandbox_runtime"):
        monkeypatch.setattr(api_module, name, lambda _path: None)
    monkeypatch.setattr(api_module, "create_challenge_runner", lambda *_a, **_k: None)
    deps = api_module._build_default_dependencies(
        entity="e",
        project="p",
        target_registry=tmp_path / "targets.json",
        sandbox_runtime=tmp_path / "runtime.json",
        db_path=tmp_path / "runs.db",
    )
    try:
        service = deps.run_service
        service._stages[RunStatus.SCORING](None, None, None)
        service._stages[RunStatus.JUDGING](None, None, None)
        stages["run_scoring_stage"].hydrate_cohort({})
        stages["run_judging_stage"].hydrate_cohort({})
    finally:
        deps.close()

    # Scoring reads each subagent's tool calls; judging renders only their type and count.
    assert hydrated == [True, False]


def test_a_second_server_on_the_same_run_database_stops_before_recovery(monkeypatch, tmp_path):
    for name in ("load_target_registry", "load_sandbox_runtime"):
        monkeypatch.setattr(api_module, name, lambda _path: None)
    monkeypatch.setattr(api_module, "create_challenge_runner", lambda *_a, **_k: None)

    def build():
        return api_module._build_default_dependencies(
            entity="e",
            project="p",
            target_registry=tmp_path / "targets.json",
            sandbox_runtime=tmp_path / "runtime.json",
            db_path=tmp_path / "runs.db",
        )

    first = build()
    try:
        with pytest.raises(ConfigurationError, match="another Monopole server"):
            build()
    finally:
        first.close()
    # Closing the first server releases the database for the next one.
    build().close()
