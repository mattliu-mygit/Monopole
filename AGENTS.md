# weave-agent-signals

Monopole provides layered evaluation and human-reviewed instruction improvement
for agent traces recorded by
`weave-agent-adapter`. It reads and scores Weave traces, writes custom feedback,
analyzes regressions, and manages human-reviewed instruction promotion. The
package and CLI remain `weave-agent-signals`.

## Communication

For direct questions, answer concisely—usually one or two sentences plus only
the minimum useful elaboration. Let follow-up questions drive deeper
explanation instead of preemptively expanding into a full walkthrough.

## Setup

Python 3.11 or newer is required. Use the lockfiles rather than ad hoc installs:

```bash
uv sync --frozen --all-extras
cd frontend && npm ci
```

The default Weave scope is entity `weave-team`, project `agent-sessions`.
Credentials come from `WANDB_API_KEY` or the `api.wandb.ai` entry in `~/.netrc`.

## Verification

Run backend gates from the repository root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
```

Run frontend gates from `frontend/`:

```bash
npm test
npm run lint
npm run build
```

Tests use synthetic spans and mocked HTTP. Prefer focused tests while iterating,
then run all gates before declaring completion.

## Architecture map

- [`specs/DESIGN.md`](specs/DESIGN.md) — product purpose and system overview
- [`specs/01-weave-io.md`](specs/01-weave-io.md) — reads, hydration, refs, writes
- [`specs/02-evaluation.md`](specs/02-evaluation.md) — deterministic and model
  evaluation, applicability, review policy, and inference trust
- [`specs/03-analysis-monitoring.md`](specs/03-analysis-monitoring.md) —
  comparable evidence, trends, coaching, and alerts
- [`specs/04-evaluation-runs.md`](specs/04-evaluation-runs.md) — reproducible runs,
  review, and promotion
- `src/weave_agent_signals/` — Weave I/O, evaluation, analysis, CLI, and API
- `src/weave_agent_signals/runs/` — durable run, reflection, review, and
  promotion lifecycle
- `frontend/src/features/` — run and reflection product behavior
- `tests/` and `frontend/tests/` — domain-organized backend and frontend tests

Specifications describe product intent, observable behavior, boundaries,
invariants, tradeoffs, and current architecture. Do not duplicate class/function
inventories, schemas, route tables, algorithms, or framework details that an
agent can read from source, CLI help, OpenAPI, or tests.

## Current commands

The installed entry point is `weave-agent-signals`. Use command `--help` for
arguments and defaults.

- `score` — deterministic recent scoring
- `backfill` — paginated historical scoring
- `judge` — turn/session rubric inference
- `inspect` — read-only trace and feedback inspection
- `analyze` — summary, cohort, trend, and coaching analysis
- `monitor` — deduplicated regression alerts
- `reflect` — proposal preview only
- `serve` — local API and built-SPA serving

Evaluation-run persistence, proposal editing, promotion, and receipts are web
product behavior; standalone reflection does not mutate managed files.

## Code conventions

- Scorer names normally use `<category>.<subcategory>`; the aggregate turn
  efficiency scorer remains `efficiency`.
- Feedback types use `weave_agent_signals.<scorer_name>`.
- Ratings are booleans or finite floats in `[0, 1]`, always use higher-is-better
  direction, and declare turn/session granularity.
- Payloads include scorer version and score time; scorer-specific data belongs
  under `payload.details`.
- Trend ordering uses `details.turn_started_at`, not write time.
- `--force` queries matching scorer-plus-ref feedback, creates the new row, then
  purges every prior match only after creation succeeds.
- Evaluation runs pin exact trace identities and configuration before work.
- Judging pins a selective episode/applicability plan before inference;
  trigger-selected scores are diagnostics, not population estimates.
- Runs pin model and rubric catalog versions, rubric thresholds, proposal
  writer, ordered judge choices, review depth, and proposal evaluator;
  recommendations never remain runtime defaults.
- Runs pin one pipeline version in their effective configuration; each stage
  fails closed before external work when that version is incompatible.
- Explicit judge overrides are honored in order. Family overlap and low
  diversity warn for bias risk but do not silently replace selected models.
- Review scores belong to immutable whole-bundle revisions, not individual files.
- Reflection pins separate proposal-writer and evaluator models plus a candidate
  budget (default three), and stops after two consecutive scored, distinct,
  non-baseline candidate iterations without a better predicted evaluator score.
- Reflection score basis is `predicted_evaluator`; it is not a verification run.
- Local judge and reflection CLIs receive only platform essentials and stored
  credential locations, never arbitrary parent secrets or token/proxy variables.
  Claude is tool-disabled. Codex uses a deny-by-default filesystem profile with
  an isolated `HOME`; only its empty workspace and minimal runtime paths are
  readable, and network, web search, apps, login shells, customizations, and
  persistence are disabled. Google-family local models remain disabled until
  Gemini has a verified confined mode.

## Weave correctness checklist

These constraints are operationally dangerous to get wrong:

1. Root turn spans use `parent_span_id == ""`, never null or `$not`.
2. Request adapter attributes through `custom_attr_columns` and flatten the
   typed custom-attribute maps.
3. Format ClickHouse `DateTime64(6)` literals as bare UTC ISO strings without
   `Z` or `+00:00`.
4. Detail columns require `include_details: true`; it cannot accompany
   `group_by`.
5. Hydrate children by exact `trace_id`, not conversation alone. Each detail
   batch must also contain the requested `invoke_agent` roots.
6. Unwrap Bash JSON `stdout`/`stderr`. Status is often `UNSET`, and an explicit
   null exit code means unknown.
7. Derive token totals/effective model from child chat spans when roots are zero.
8. Feedback query returns records under `result`.
9. Forced feedback replacement creates the new row before purging any prior
   match. Purge supports `$eq` on feedback `id` and runs only after creation
   succeeds.
10. Custom scores use flat `payload`. `scorer_*` fields require monitor-specific
    runnable, call, and trigger refs that this project does not create.
11. W&B Inference is `https://api.inference.wandb.ai/v1` and requires bearer W&B
    auth plus `OpenAI-Project: {entity}/{project}`. HTTP 402 means credits are
    disabled.
12. URL-encode `conversation_id` when constructing session refs.

## Run and promotion safety

Do not simplify away cohort/config pinning, reflection-input provenance, review
compare-and-swap revisions, cancellation coordination, whole-scope drift checks,
promotion journaling, rollback/recovery, or monotonic progress accounting. They
protect current correctness and auditability.

The local run database is disposable pre-release state. Do not add schema
migrations or retired payload adapters unless the product support policy changes.
