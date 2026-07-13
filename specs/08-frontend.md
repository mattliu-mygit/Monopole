# Spec 08: Frontend

React single-page application for viewing and triggering weave-agent-signals operations, talking to the FastAPI backend (spec 07) over REST.

## Stack

Vite + React 19 + TypeScript, Tailwind CSS v4, React Router v7, Recharts, TanStack Query — and deliberately no component library. This is a single-developer tool where the priority is being able to change the UI quickly, not a polished design system.

## Pages

- **Dashboard** (`/`) — read-only system health overview: per-scorer summary, trend direction, active config version, recent job activity.
- **Sessions** (`/sessions`, `/sessions/:id`) — browse and inspect agent sessions: filterable list, and a detail view with turn timeline, expandable tool calls, and per-turn/session feedback.
- **Runs** (`/runs`, `/runs/:runId`) — see below.
- **Analyze** (`/analyze`) — pattern analysis: coaching digest, score summary, A/B leaderboard by `config_version`, trend charts, regression alerts.
- **Monitor** (`/monitor`) — trigger a monitor check; view alert history and dedup state.
- **Jobs** (`/jobs`) — history of background job runs, with status and expandable results/errors.

Dashboard, Sessions, Analyze, Monitor, and Jobs are unchanged from the original design. Only the scoring/judging/reflection workflow moved — see below.

## Runs page (replaces Scoring, Judging, Reflect)

The former `/scoring`, `/judging`, and `/reflect` pages each triggered an independent job against its own `since`/`limit` params, with no shared data selection and no link between a reflection proposal and the scores/judgments that produced it (see spec 09's Problem section). They're replaced by a single **Runs** page:

- **List view** (`/runs`): every run with status, date range/session count, and current pipeline step; create a new run here.
- **Detail view** (`/runs/:runId`): the pipeline as step cards — data selection, scoring, judging, reflecting, in fixed order — each expandable to its own config form and results, with an action to run/advance that step. This makes it possible to inspect what data produced a given reflection proposal, which the old three-pages design couldn't do.

The job-trigger forms that used to live on the Scoring/Judging/Reflect pages now live inside the matching step card, scoped to the run's data selection instead of ad hoc request params.

## Data fetching

TanStack Query throughout: queries for reads, mutations for job/run submission (invalidating the relevant query on success), and `refetchInterval` polling for anything in a "running" state. Runs, jobs, and monitor checks all follow this same poll-based pattern rather than a WebSocket — unnecessary complexity for a single-user local tool.

## Non-goals (v1)

Auth/multi-user, real-time WebSocket updates, mobile responsiveness, theming, frontend E2E tests (the API is unit-tested; the frontend is tested manually).
