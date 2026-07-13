# Spec 07: REST API

FastAPI server exposing weave-agent-signals over HTTP. Lives alongside the CLI (spec 06) as a second entry point (`weave-agent-signals serve`) — the frontend (spec 08) is its only consumer.

## Design principles

- **Thin wrapper, not a second implementation.** Every endpoint delegates to the same functions the CLI calls (scorers, judge runner, reflector, pattern analysis). Business logic lives once, in the library; the API and CLI are both callers.
- **Background jobs for anything slow.** Scoring, judging, reflecting, and monitor checks don't block the request — they run in a background task/thread and the client polls for status.
- **One `WeaveClient` for the process lifetime** — a single connection pool, not one per request.
- **No auth in v1.** This is a local, single-user developer tool; the server inherits `WANDB_API_KEY` from the environment exactly like the CLI.
- **State persists in SQLite** (`~/.weave-agent-signals/`), for both jobs and runs — surviving a server restart matters once a run's pipeline can span multiple sessions of work.

## Endpoints

**Read-only**: turns (list + single with full children), sessions (list + detail), feedback query, pattern analysis (`/api/analyze`), current on-disk artifacts, the rubric catalog, per-backend model rosters, monitor state, and job list/detail. The rubric and model endpoints exist so the Runs pipeline UI can populate its judging-step forms from the same source of truth the judge runner itself uses, rather than duplicating choices in the frontend.

**`POST /api/jobs/monitor`**: the sole remaining ad hoc job-submission endpoint. Monitoring is a continuous background concern (see DESIGN.md's Monitoring & alerting section), not something you'd curate a data selection for — it doesn't fit the run pipeline model and isn't meant to.

**Run endpoints** (spec 09): create a run, list/get runs, set a run's data selection, advance a run to its next pipeline step, and apply a completed reflection proposal. Advancing a run is what actually executes scoring/judging/reflecting now — it dispatches to the same underlying logic the old `POST /api/jobs/score|backfill|judge|reflect` endpoints used to call directly, but scoped to the run's pinned data selection instead of per-request `since`/`limit` params. Those old endpoints, and `/api/jobs/reflect/apply`, are removed — do not reintroduce them.

## CORS and static serving

CORS allows the Vite dev server origin in development only. In production, `serve` mounts the built frontend as static files and serves the API from the same origin/process, avoiding CORS and a second deployable entirely.
