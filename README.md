# Monopole

Monopole evaluates your Claude Code and Codex sessions and turns that into a
feedback loop. It reads each session from [W&B Weave](https://wandb.ai/site/weave)
(recorded by the `weave-agent-adapter`), scores what the agent did, writes the
scores back onto the same traces, watches for drops in performance, and can
propose changes to your agent config. The Python package is `weave-agent-signals`.

## What it does

Five layers, from plain facts up to self-improvement:

- **M1 — facts.** Deterministic per-turn scores read straight from the trace:
  did tests pass, did git / install / lint succeed, how efficient the turn was
  (error loops, repeated reads), and implicit feedback (how often you corrected
  the agent, whether a session was abandoned). No model calls.
- **M2 — turn judgment.** LLM judges score each turn on process quality:
  verification, error recovery, tool choice, task completion.
- **M3 — session judgment.** A panel of judges — one same-family plus at least
  two cross-family, mean-pooled — scores a compact digest of a whole conversation.
- **M4 — patterns.** A/B comparisons keyed by config version, trend detection,
  and a coaching summary; a GEPA-based reflector proposes edits to CLAUDE.md,
  skills, and commands, gated by human review.
- **M5 — monitoring.** A scheduled check that alerts when a score drops in a
  statistically real way — a downward trend, or a config change that made things
  worse.

Scores are stored in Weave as feedback attached to the turn or conversation they
describe, so they sit next to the traces they evaluate and stay queryable.

## Status

- **M1** is implemented and checked against real data — trust it.
- **M2–M5** are implemented and unit-tested. The judge scores (M2/M3) are **not
  yet validated** against hand-read sessions, so treat them as provisional until
  that check is done.
- Judging currently runs through a **temporary local CLI backend**: it shells out
  to the installed `claude` / `codex` / `gemini` CLIs (subscription auth) as
  one-shot judges, so no metered-inference credits are needed. This is a
  developer-machine stopgap, not the long-term path (W&B Inference or Weave
  Signals, both pending billing).

## Setup

```bash
pip install -e ".[dev]"     # optional extras: [rsi] for the reflector
wandb login                 # auth for reading traces and writing scores (stored in ~/.netrc)
```

The Weave entity/project default to `mliu-wandb-weights-biases` / `agent-sessions`
(override with `--entity` / `--project`). Put any API keys in a `.env` at the repo
root — the CLI loads it on startup.

## Usage

```bash
weave-agent-signals score    [--since DATE] [--limit N] [--dry-run] [--force]
weave-agent-signals backfill --start DATE [--end DATE] [--dry-run] [--force]
weave-agent-signals judge    [--rubric NAME,...] [--panel-size N] [--judge-backend cli]
weave-agent-signals analyze  [--ab] [--trends] [--coaching]
weave-agent-signals reflect  [--dry-run] [--apply]
weave-agent-signals monitor  [--alert-webhook URL] [--state-file PATH] [--dry-run]
weave-agent-signals inspect  [--recent N] [--session ID] [--feedback]
```

- `score` / `backfill` — write M1 scores (recent turns, or a date range).
- `judge` — run the turn and session LLM judges. `--judge-backend cli` uses the
  local `claude` / `codex` / `gemini` CLIs, which must be installed for it to work.
- `analyze` — A/B leaderboard, trends, and coaching summary from the scores.
- `reflect` — propose config edits from the scores; review the diff, then `--apply`.
- `monitor` — alert on significant regressions (see Scheduling).

## Scheduling

`monitor` and `score` are meant to run on a timer. A ready-to-load launchd job is
in [`deploy/`](deploy/): every hour it scores recent turns, then checks for drops,
logging to `~/Library/Logs/weave-agent-signals.log`. Turn it on:

```bash
cp deploy/com.weave-agent-signals.monitor.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.weave-agent-signals.monitor.plist
```

Add `--alert-webhook <url>` to the monitor line in that file to send alerts to Slack.

## Testing

```bash
pytest                      # full suite; synthetic span data, no live Weave calls
ruff check src/ tests/      # lint
ruff format --check src/ tests/  # format
```

## Layout

- `src/weave_agent_signals/` — implementation
- `specs/DESIGN.md` — architecture and layer design (start here)
- `specs/` — per-component specs
- `deploy/` — the scheduled-job file
