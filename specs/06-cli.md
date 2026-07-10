# Spec 06: CLI

Command-line interface for running scorers and backfilling history. Entry point: `weave-agent-signals` (installed via `pyproject.toml`). Future commands (`reflect`, `propose`, coach-agent query tools) are sketched in [FUTURE.md](FUTURE.md) and get specced with their milestones.

---

## Commands

### `score`

Score recent unscored turns and sessions.

```
weave-agent-signals score [--since DATETIME] [--limit N] [--scorers NAMES] [--dry-run]
```

| Flag | Default | Description |
|---|---|---|
| `--since` | 24h ago | Score turns started after this time |
| `--limit` | 100 | Max turns to score per run |
| `--scorers` | all L1 | Comma-separated scorer names (e.g. `outcome.test,implicit.frustration`) |
| `--dry-run` | false | Compute scores, print them, don't write feedback |
| `--force` | false | Re-score even if feedback already exists |

Flow:
1. Query recent turns via `agents/spans/query` (spec 01)
2. Check existing feedback for dedup (spec 05)
3. Run applicable scorers (specs 02–04)
4. Write scores as feedback (spec 05)
5. Print summary: `Scored 42 turns, 3 sessions. 5 test_pass, 2 test_failure, 1 error_loop.`

### `backfill`

Score historical sessions in a date range.

```
weave-agent-signals backfill --from DATE --to DATE [--scorers NAMES] [--batch-size N]
```

| Flag | Default | Description |
|---|---|---|
| `--from` | required | Start date (inclusive) |
| `--to` | today | End date (inclusive) |
| `--scorers` | all L1 | Which scorers to run |
| `--batch-size` | 50 | Feedback batch write size |

Polite backfill: page through sessions, sleep between batches. Uses `/feedback/batch/create` for efficiency.

### `inspect`

Show scores for a specific turn or session (debugging).

```
weave-agent-signals inspect <ref>
weave-agent-signals inspect --session <conversation_id>
weave-agent-signals inspect --recent [N]
```

Prints: all feedback on the ref, formatted as a table.

### `setup`

One-time configuration.

```
weave-agent-signals setup
```

Interactive: prompts for Weave entity/project, verifies API key, checks adapter is installed, runs a test query.

---

## Configuration

Config file: `~/.config/weave-agent-signals/config.toml` (XDG) or `~/.weave-agent-signals.toml`.

```toml
[weave]
entity = "mliu-wandb-weights-biases"
project = "agent-sessions"

[scoring]
default_scorers = ["outcome", "implicit", "efficiency"]
default_since_hours = 24
```

Gate and judge config sections land with M2 (see FUTURE.md).

---

## Scheduling (launchd)

Example plist for periodic scoring:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "...">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.weave-agent-signals.score</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/weave-agent-signals</string>
        <string>score</string>
    </array>
    <key>StartInterval</key>
    <integer>1800</integer>
    <key>StandardOutPath</key>
    <string>/tmp/weave-agent-signals.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/weave-agent-signals.err</string>
</dict>
</plist>
```

Typical schedule: `score` every 30 minutes. Future commands add their own cadences.

---

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Scoring errors (partial failure, some scores written) |
| 2 | Configuration error (missing API key, bad project) |
| 3 | API error (Weave unreachable, auth failure) |
