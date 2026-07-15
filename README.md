# Monopole

Monopole evaluates Claude Code and Codex sessions recorded by
[`weave-agent-adapter`](https://github.com/wandb/weave-agent-adapter). It reads
traces from W&B Weave, writes scores back beside the same turns and sessions,
detects regressions, and proposes auditable instruction changes for human
review. The Python package and CLI are named `weave-agent-signals`.

## What it does

- Extracts deterministic test, build, lint, install, Git, and command outcomes.
- Measures correction-free rate, completion, and repeated-work efficiency while
  retaining the raw negative signals as context.
- Uses a pinned applicability plan for selected process episodes and guided,
  user-overridable review depth for every LLM-judged episode/session rubric.
- Compares configuration cohorts, calculates representative trends, labels
  trigger-selected judgments as diagnostics, produces coaching input, and
  alerts only on new significant regressions.
- Runs a reproducible scoring-to-reflection pipeline over pinned Weave traces.
- Uses a proposal evaluator that predicts a whole-bundle score for the exact
  current instruction bundle and generated candidates, then lets a user review
  Past B, scored candidate C, and optional unevaluated edit D before promotion.
- Applies approved multi-file changes through drift-checked, journaled promotion
  with an immutable receipt.

Scores are custom Weave feedback attached to the turn or conversation they
describe. Local SQLite state tracks evaluation runs and review audit evidence.

## Requirements and setup

Python 3.11 or newer, Node.js/npm for the UI, and a W&B API key are required.
`wandb login` can store the key in `~/.netrc`; a repo-root `.env` is also loaded.

For the complete local web product, including proposal generation:

```bash
uv sync --frozen --extra server --extra reflection
cd frontend && npm ci
```

For contributor tooling, tests, and the web server (`dev` already includes the
proposal-generation dependency):

```bash
uv sync --frozen --all-extras
cd frontend && npm ci
```

The default Weave scope is `weave-team/agent-sessions`; global CLI arguments can
override the entity and project.

## CLI

Run `weave-agent-signals COMMAND --help` for current arguments and defaults.

| Command | Purpose |
|---|---|
| `score` | Score recent turns and sessions deterministically |
| `backfill` | Paginate and score a historical date range |
| `judge` | Run selected-episode and session model rubrics |
| `inspect` | Inspect recent trace/session detail and feedback |
| `analyze` | Summarize scores, cohorts, trends, and coaching |
| `monitor` | Alert on new significant regressions |
| `reflect` | Preview candidate instruction bundles and diffs |
| `serve` | Run the API and serve a built frontend |

Standalone `reflect` only previews output. Create an evaluation run in the web
UI to persist candidates, edit a proposal, promote it, or retain a receipt.
For each invocation it resolves a catalog-selected proposal writer and proposal
evaluator, makes up to three proposal attempts by default (configurable from one
through ten), and stops after two consecutive scored attempts without
improvement. Evaluation runs persist and pin those roles; the standalone preview
does not. Displayed scores are predicted evaluator scores, not verification
runs.

## Web UI

For development, run the API and Vite in separate terminals:

```bash
weave-agent-signals serve
```

```bash
cd frontend
npm run dev
```

Vite serves `http://localhost:5173`. For one-process serving, build first:

```bash
cd frontend
npm run build
cd ..
weave-agent-signals serve
```

The UI includes dashboard, session list/detail, run list/detail, and analysis
pages. Run detail shows persisted scoring/judging results, live semantic
reflection activity, evaluated bundle snapshots and scores, editable D state,
drift, promotion receipts, and dismissal decisions.

The API is unauthenticated and intended for a trusted local, single-user
environment. It binds to `127.0.0.1` by default and inherits credentials from
the server process. Passing a non-loopback `--host` explicitly exposes that
unauthenticated API and its configured credentials to the reachable network.

## Scheduling

[`deploy/com.weave-agent-signals.monitor.plist`](deploy/com.weave-agent-signals.monitor.plist)
runs `score`, then `monitor`, hourly and logs under `~/Library/Logs/`. To load it:

```bash
cp deploy/com.weave-agent-signals.monitor.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.weave-agent-signals.monitor.plist
```

Configure a webhook in the plist when external alerts are desired.

## Development and design

Contributor setup and verification live in [AGENTS.md](AGENTS.md). Start product
architecture in [specs/DESIGN.md](specs/DESIGN.md), then use the component specs:

- [Weave I/O](specs/01-weave-io.md)
- [Evaluation](specs/02-evaluation.md)
- [Analysis and monitoring](specs/03-analysis-monitoring.md)
- [Evaluation runs](specs/04-evaluation-runs.md)

Implementation is under `src/weave_agent_signals/`; the durable evaluation-run
lifecycle is grouped under its `runs/` feature package. The React SPA keeps run
and reflection behavior under `frontend/src/features/`. Synthetic backend and
frontend tests are centralized under `tests/` and `frontend/tests/`; the
schedule is under `deploy/`.
