"""FastAPI composition root for evaluation runs and read-only inspection."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.client import WeaveClient
from weave_agent_signals.judges.cli_backend import CliJudgeClient
from weave_agent_signals.judges.inference import ChatClient, InferenceClient
from weave_agent_signals.patterns import coaching_digest
from weave_agent_signals.routes.catalogs import create_catalogs_router
from weave_agent_signals.routes.inspection import create_inspection_router
from weave_agent_signals.routes.reviews import create_reviews_router
from weave_agent_signals.routes.runs import create_runs_router
from weave_agent_signals.run_config import EffectiveRunConfig, ModelDescriptor
from weave_agent_signals.runs.cohort import discover_turn_cohort, hydrate_turn_cohort
from weave_agent_signals.runs.promotion import TargetPromoter
from weave_agent_signals.runs.review import ReviewService
from weave_agent_signals.runs.service import RunService
from weave_agent_signals.runs.stages.judging import JudgingDependencies, run_judging_stage
from weave_agent_signals.runs.stages.reflection import (
    ReflectionDependencies,
    run_reflection_stage,
)
from weave_agent_signals.runs.stages.scoring import ScoringDependencies, run_scoring_stage
from weave_agent_signals.runs.store import Run, RunStore
from weave_agent_signals.runs.targets import load_target_registry

load_dotenv()

_DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@dataclass(frozen=True)
class ApiDependencies:
    """Application-owned services and their shared external client factory."""

    run_service: RunService
    review_service: ReviewService
    client_factory: Callable[[], WeaveClient]
    close: Callable[[], None]


class _LazyReference:
    """Bind default services at startup so module import does not touch SQLite."""

    def __init__(self) -> None:
        self._value: object | None = None

    def bind(self, value: object) -> None:
        self._value = value

    def clear(self) -> None:
        self._value = None

    def __getattr__(self, name: str) -> Any:
        if self._value is None:
            raise RuntimeError("API dependencies are not initialized")
        return getattr(self._value, name)


def _chat_client(
    backend: str,
    *,
    entity: str,
    project: str,
) -> AbstractContextManager[ChatClient]:
    if backend == "cli":
        return CliJudgeClient()
    return InferenceClient(entity=entity, project=project, backend=backend)


def _build_default_dependencies(
    *,
    entity: str,
    project: str,
    target_registry: Path | None,
    db_path: str | Path | None,
) -> ApiDependencies:
    """Build production services once during application startup."""

    store = RunStore(db_path)
    executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="evaluation-run")

    def client_factory() -> WeaveClient:
        return WeaveClient(entity=entity, project=project)

    def hydrate(cohort):
        return hydrate_turn_cohort(cohort, client_factory=client_factory)

    scoring_dependencies = ScoringDependencies(
        store=store,
        client_factory=client_factory,
        hydrate_cohort=hydrate,
    )

    def scoring_stage(
        run: Run,
        config: EffectiveRunConfig,
        cancel: threading.Event,
    ) -> None:
        run_scoring_stage(
            run,
            config,
            cancel,
            dependencies=scoring_dependencies,
        )

    def judging_stage(
        run: Run,
        config: EffectiveRunConfig,
        cancel: threading.Event,
    ) -> None:
        run_judging_stage(
            run,
            config,
            cancel,
            dependencies=JudgingDependencies(
                store=store,
                client_factory=client_factory,
                chat_client_factory=lambda: _chat_client(
                    config.judge_backend,
                    entity=entity,
                    project=project,
                ),
                hydrate_cohort=hydrate,
            ),
        )

    def model_client(
        descriptor: ModelDescriptor,
    ) -> AbstractContextManager[ChatClient]:
        return _chat_client(
            descriptor.backend,
            entity=entity,
            project=project,
        )

    if target_registry is None:
        raise RuntimeError("TARGET_REGISTRY is required to start the server")
    registry = load_target_registry(target_registry)
    reflection_dependencies = ReflectionDependencies(
        store=store,
        client_factory=client_factory,
        adapter_factory=lambda: TargetPromoter(registry),
        writer_client_factory=model_client,
        evaluator_client_factory=model_client,
        coaching_digest=coaching_digest,
    )

    def reflection_stage(
        run: Run,
        config: EffectiveRunConfig,
        cancel: threading.Event,
    ) -> None:
        run_reflection_stage(
            run,
            config,
            cancel,
            dependencies=reflection_dependencies,
        )

    run_service = RunService(
        store=store,
        build_model_catalog=build_model_catalog,
        build_rubric_catalog=build_rubric_catalog,
        discover_cohort=lambda selection: discover_turn_cohort(
            selection,
            client_factory=client_factory,
            entity=entity,
            project=project,
        ),
        scoring_stage=scoring_stage,
        judging_stage=judging_stage,
        reflection_stage=reflection_stage,
        executor=executor,
    )
    review_service = ReviewService(
        store=store,
        adapter_factory=lambda: TargetPromoter(registry),
    )

    def close() -> None:
        executor.shutdown(wait=True, cancel_futures=True)
        store.close()

    return ApiDependencies(
        run_service=run_service,
        review_service=review_service,
        client_factory=client_factory,
        close=close,
    )


def _register_frontend(application: FastAPI, frontend_dist: Path) -> None:
    if not frontend_dist.is_dir():
        return
    resolved_dist = frontend_dist.resolve()
    assets = resolved_dist / "assets"
    if assets.is_dir():
        application.mount("/assets", StaticFiles(directory=assets), name="assets")

    @application.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> FileResponse:
        requested = (resolved_dist / full_path).resolve()
        try:
            requested.relative_to(resolved_dist)
        except ValueError:
            requested = resolved_dist / "index.html"
        if requested.is_file():
            return FileResponse(requested)
        return FileResponse(resolved_dist / "index.html")


def create_app(
    *,
    dependencies: ApiDependencies | None = None,
    entity: str | None = None,
    project: str | None = None,
    target_registry: str | Path | None = None,
    db_path: str | Path | None = None,
    frontend_dist: str | Path | None = None,
) -> FastAPI:
    """Create an app with injected services or lazily built production services."""

    resolved_entity = entity or os.environ.get("WANDB_ENTITY", "weave-team")
    resolved_project = project or os.environ.get("WANDB_PROJECT", "agent-sessions")
    registry_value = target_registry or os.environ.get("TARGET_REGISTRY")
    resolved_registry = Path(registry_value).expanduser() if registry_value else None
    resolved_db = db_path or os.environ.get("WEAVE_AGENT_SIGNALS_RUN_DB")

    run_reference = _LazyReference()
    review_reference = _LazyReference()
    client_reference = _LazyReference()
    if dependencies is not None:
        run_reference.bind(dependencies.run_service)
        review_reference.bind(dependencies.review_service)
        client_reference.bind(dependencies.client_factory)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        active = dependencies
        if active is None:
            active = _build_default_dependencies(
                entity=resolved_entity,
                project=resolved_project,
                target_registry=resolved_registry,
                db_path=resolved_db,
            )
            run_reference.bind(active.run_service)
            review_reference.bind(active.review_service)
            client_reference.bind(active.client_factory)
        application.state.dependencies = active
        try:
            active.run_service.recover_interrupted_runs()
            yield
        finally:
            active.close()
            run_reference.clear()
            review_reference.clear()
            client_reference.clear()

    application = FastAPI(
        title="weave-agent-signals",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(
        create_catalogs_router(
            build_model_catalog=build_model_catalog,
            build_rubric_catalog=build_rubric_catalog,
        )
    )
    application.include_router(
        create_runs_router(
            cast(RunService, run_reference),
            cast(ReviewService, review_reference),
        )
    )
    application.include_router(create_reviews_router(cast(ReviewService, review_reference)))

    def client_factory() -> WeaveClient:
        return cast(Callable[[], WeaveClient], client_reference._value)()

    application.include_router(create_inspection_router(client_factory))
    _register_frontend(
        application,
        Path(frontend_dist) if frontend_dist is not None else _DEFAULT_FRONTEND_DIST,
    )
    return application


# Default resources are intentionally created by the lifespan, not by import.
app = create_app()
