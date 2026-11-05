# weave-agent-signals

Monopole provides layered evaluation and human-reviewed instruction improvement
for agent sessions that `weave-agent-adapter` writes to the CoreWeave
Evaluations service (CES). It reads and scores those sessions, writes CES Scores,
analyzes regressions, and manages human-reviewed instruction promotion. The
package and CLI remain `weave-agent-signals`.

## Communication

For direct questions, answer concisely—usually one or two sentences plus only
the minimum useful elaboration. Let follow-up questions drive deeper
explanation instead of preemptively expanding into a full walkthrough.

## Engineering proportionality

Optimize guarantees and architecture for the service's actual product goals and
risk level. A service does not need to be perfect when the implementation,
operational, or maintenance cost of stronger guarantees is substantially larger
than their practical benefit. For meaningful design choices, compare the
benefits, costs, failure modes, and reversibility explicitly, then choose the
simplest design whose tradeoffs are acceptable. Do not preserve complexity only
because it provides a theoretically stronger guarantee.

Match planning and documentation effort to the size of the change. Small,
localized changes normally need only a concise implementation outline in the
task; do not create standalone design or plan documents unless the behavior,
architecture, or tradeoffs are genuinely complex or the user requests one.

## Setup

Python 3.11 or newer is required. Use the lockfiles rather than ad hoc installs:

```bash
uv sync --frozen --all-extras
cd frontend && npm ci
```

The CES scope and Dataset come from the adapter's
`~/.weave-agent-adapter/evaluations.json`, or the `--ces-*` flags and
`WEAVE_AGENT_SIGNALS_CES_*` env vars. Credentials come from `WANDB_API_KEY` or the
`api.wandb.ai` entry in `~/.netrc`.

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
npm run check:api
npm run lint
npm run build
```

Tests use synthetic spans and mocked HTTP. Prefer focused tests while iterating,
then run all gates before declaring completion.

## Architecture map

- [`specs/DESIGN.md`](specs/DESIGN.md) — product purpose and system overview
- [`specs/01-ces-io.md`](specs/01-ces-io.md) — session reads, Scores, pinning, writes
- [`specs/02-evaluation.md`](specs/02-evaluation.md) — deterministic and model
  evaluation, sliding context policy, fixed-panel aggregation, and inference trust
- [`specs/03-analysis-monitoring.md`](specs/03-analysis-monitoring.md) —
  comparable evidence, trends, coaching, and alerts
- [`specs/04-evaluation-runs.md`](specs/04-evaluation-runs.md) — reproducible runs,
  review, and promotion
- `src/weave_agent_signals/` — CES I/O, evaluation, analysis, CLI, and API
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
- `judge` — sliding-window session rubric inference
- `inspect` — read-only session and Score inspection
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
- `--force` writes newer Scores under a new pass; nothing is purged, and reads take
  the newest Score per Scorer.
- Evaluation runs pin exact trace identities and configuration before work.
- Judging pins its context policy, complete raw-turn coverage, reviewer-specific
  window plans, and digest/window/merge protocol before inference.
- Judge feedback is session-only. Ordinary analysis requires a complete review
  and a fully identified comparison context.
- Runs pin model and rubric catalog versions, rubric thresholds, proposal
  writer, one through three ordered judge choices, and proposal evaluator;
  recommendations never remain runtime defaults.
- Runs pin one pipeline version in their effective configuration; each stage
  fails closed before external work when that version is incompatible.
- Persisted judging artifacts are content-authenticated and reusable only when
  their exact plan, reviewer, protocol, model, phase, and schema still match.
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
  persistence are disabled. Antigravity runs from an empty workspace in
  confined `accept-edits` mode with permission auto-approval, a minimal child
  environment, an isolated temporary home, and a one-use FIFO carrying only
  its stored OAuth token. Persistent Gemini histories and state are not exposed.

## CES correctness checklist

These constraints are operationally dangerous to get wrong. The details live in
[`specs/01-ces-io.md`](specs/01-ces-io.md).

1. A turn's identity is its first Case ID. Turn Scores target that Case, and run
   cohorts pin every member Case ID; never rebuild a turn's identity another way.
2. CES filters cannot range-compare string fields. Scan newest first and stop at
   the bound instead of filtering on `started_at`.
3. Every create carries an `Idempotency-Key` built from its content. Score keys
   include the pass ID and a body digest, so a retry replays and a changed body
   writes a newer Score. Never reuse a key for different content.
4. Reads return the newest Score per Scorer. Rescoring writes newer Scores; there
   is no purge.
5. Only finished judge reviews write Scores, and each rubric and panel has its
   own Scorer.
6. CoreWeave Serverless Inference is `https://api.inference.wandb.ai/v1` and
   requires bearer W&B auth plus `OpenAI-Project: {entity}/{project}`. HTTP 402
   means credits are disabled.

## Run and promotion safety

Do not simplify away cohort/config pinning, reflection-input provenance, review
compare-and-swap revisions, cancellation coordination, whole-scope drift checks,
per-file atomic writes and outcomes, or monotonic progress accounting. They
protect current correctness and auditability. Promotion intentionally does not
journal, roll back, or recover a multi-file transaction; completed files may
remain applied when a later file fails.

The local run database is disposable pre-release state. Do not add schema
migrations or retired payload adapters unless the product support policy changes.
