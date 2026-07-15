# weave-signal-monitoring

`weave-signal-monitoring` is a small, standalone hydration system for agent-quality
signals in W&B Weave. It installs five high-recall turn monitors and converts their
low numeric ratings into a ranked list of conversations worth reviewing in
Monopole.

A low rating is a recommendation to inspect a conversation, not a final failure
judgment. Monopole remains responsible for complete, evidence-pinned session
evaluation.

## Boundary

This folder has its own package, lockfile, command, tests, and documentation. It
does not import Monopole, write Monopole state, start evaluation runs, or modify
the Monopole API or frontend.

There is no daemon, scheduler, database, cursor, webhook, or Slack integration.
Each `hydrate` invocation reads one bounded window directly from Weave.

## Setup

Python 3.11 or newer is required. From this folder:

```bash
uv sync --frozen --all-extras
```

Authentication uses `WANDB_API_KEY` or the `api.wandb.ai` entry in `~/.netrc`.
The default scope is entity `weave-team`, project `agent-sessions`. Override it
before the command name:

```bash
uv run weave-signal-monitor --entity my-team --project my-project hydrate
```

## Install monitors

```bash
uv run weave-signal-monitor install
```

Installation creates and activates the exact versioned catalog. Re-running the
command reuses identical definitions. If a monitor name already exists with a
different definition, installation fails instead of overwriting it or creating a
duplicate. A partial external failure is resumable: monitors created before the
failure remain installed and are reused on the next invocation.

The implementation uses Weave's generic `Monitor` with a small local LLM judge
scorer. The local scorer avoids the unrelated model and NLP packages in the full
`weave[scorers]` extra.

## Hydrate conversations

The default window is the preceding 24 hours:

```bash
uv run weave-signal-monitor hydrate
```

Use an explicit timezone-aware UTC window when needed:

```bash
uv run weave-signal-monitor hydrate \
  --since 2026-07-15T00:00:00Z \
  --until 2026-07-16T00:00:00Z
```

The default output is a compact table containing the lowest rating, conversation
name, triggering signal names, last activity time, and W&B link. Hydration is
read-only and never marks feedback as consumed.

Use JSON for the future Monopole Sessions and alerting boundary:

```bash
uv run weave-signal-monitor hydrate --json
```

The top-level JSON schema is version `1`:

```json
{
  "schema_version": 1,
  "generated_at": "2026-07-15T20:00:00Z",
  "entity": "weave-team",
  "project": "agent-sessions",
  "window": {
    "since": "2026-07-14T20:00:00Z",
    "until": "2026-07-15T20:00:00Z"
  },
  "conversations": []
}
```

Each conversation contains:

- `conversation_id` and a recorded name or bounded first-request preview;
- start and last-activity times;
- the lowest observed rating;
- ordered signal evidence with version, rating, reason, and turn identity;
- ordered triggering turn IDs; and
- a W&B link to the first triggering turn.

Conversations sort by lowest rating first, then most recent activity. Repeated
feedback for one signal and turn resolves to the newest completed score.

## Signal catalog

Catalog version `v1` scores every eligible Weave agent-turn signal span at sampling rate
`1.0`:

| Signal | Detects |
| --- | --- |
| `user-frustration` | Annoyance, impatience, confusion, or dissatisfaction expressed by the user |
| `user-correction-or-rejection` | Explicit correction, rejection, or corrective redirect |
| `explicit-repeat-or-rephrase-cue` | Explicit language that an unresolved request is being repeated or restated |
| `stalled-or-deferred-response` | Unnecessary handoff, postponement, or early stop by the agent |
| `low-quality-response` | A materially irrelevant, evasive, repetitive, unsupported, or incomplete response |

All signals use the same higher-is-better numeric anchors:

- `1.0`: no evidence of the problem;
- `0.75`: weak or ambiguous indication;
- `0.5`: plausible indication;
- `0.25`: strong indication; and
- `0.0`: explicit or severe evidence.

Ratings at or below `0.5` recommend conversation review. The scorer must also
return a short reason grounded in the visible turn.

## Failure behavior

Authentication failure, incomplete or repeated pagination, conflicting monitor
identity, malformed eligible feedback, missing turn identity, or incomplete
conversation hydration exits non-zero and emits no partial table or JSON.

An empty, completely queried window is successful. Ratings above the threshold,
unknown scorer versions, and incomplete scorer attempts are ignored.

Errors name the failed boundary without printing credentials, scorer prompts, or
trace content.

## Live compatibility smoke check

Before relying on a new Weave project or SDK version:

1. Run `install` twice. The first run creates or reuses five monitors; the second
   reuses all five without duplicates.
2. Generate or select one agent turn with an explicit low-signal cue.
3. Wait for every monitor attempt to finish.
4. Hydrate a narrow UTC window containing the turn with `--json`.
5. Confirm every monitor attached readable runnable feedback to the same turn,
   each output has an exact numeric anchor and reason, at least one rating is
   `<= 0.5`, and the expected conversation and W&B link are present.
6. Re-run hydration and confirm evidence selection and ordering are stable.
7. Query an adjacent empty window and confirm schema-v1 JSON with an empty
   `conversations` list and exit code `0`.

If activation, prompt-variable binding, feedback attachment, scorer identity, or
conversation hydration does not match this contract, stop and revise the design.
Do not add a private compatibility workaround.

## Deferred integrations

Slack alerts, scheduling, persistent acknowledgement state, automatic Monopole
runs, and Monopole Sessions UI changes are intentionally deferred. They should
consume schema-v1 JSON rather than importing this package's internals.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
```
