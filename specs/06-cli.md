# Spec 06: CLI

Command-line interface for running scorers, judges, analysis, reflection, and monitoring, plus backfilling history and serving the API. Entry point: `weave-agent-signals` (installed via `pyproject.toml`). The API (spec 07) is a thin wrapper over the same underlying functions these commands call.

## Commands

`score`, `backfill`, `judge`, `analyze`, `reflect`, `monitor`, `inspect`, `serve` — see `--help` on each for current flags.

**`score` vs `backfill`**: `score` is the frequent, cheap, incremental path — recent unscored turns only (see spec 01's incremental-scoring rationale). `backfill` is the expensive, full-history pass over a date range, meant to run occasionally (or once, to seed history) rather than on a schedule alongside `score`.

**`inspect`** is a debugging tool — dump scores for a turn/session/recent window without triggering any scoring.

## Configuration

Entity/project are CLI flags with hardcoded defaults (single user, single project) rather than a config file — not worth the complexity until there's a second project to point at.

## Scheduling

Deployed via launchd (`deploy/`): `score` then `monitor` run hourly, logging to `~/Library/Logs/weave-agent-signals.log`. This is the only scheduled cadence today — `judge`, `analyze`, `reflect`, and `backfill` are run manually or triggered through the API/run pipeline (spec 09).

## Exit codes

Distinguishes total failure from partial failure so a scheduler (launchd, or any future CI-style caller) can react appropriately: `0` success, `1` partial failure (some scores written, some scoring errors), `2` configuration error (bad API key/project — retrying won't help), `3` API error (Weave unreachable/auth failure — transient, retrying might help).
