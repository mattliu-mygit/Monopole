# Spec 05: Score write-back

How scores are persisted as Weave feedback on agent trace refs. The write path is the mirror of spec 01's read path.

Ties to spec 01 (ref format), all scorer specs (02-04).

## Ref types

Turn and session scores attach to synthesized `weave:///{entity}/{project}/agent_turn/{trace_id}` and `.../agent_conversation/{conversation_id}` refs, respectively — there's no dedicated "session" entity in Weave itself (see spec 01); the ref is a convention this project defines. `conversation_id` can contain `/`, so it must be URL-encoded in the ref.

## Feedback type: custom, not `wandb.agent_monitor`

All scorers here (M1 deterministic and beyond) write a custom feedback type, `weave_agent_signals.<scorer_name>`, with a flat `payload` dict (rating, confidence, tags, reason, granularity, and scorer-specific detail under `payload.details`).

**Why not `wandb.agent_monitor`** (Weave's native Signals feedback type, with typed `scorer_ratings`/`scorer_tags` columns): it requires `runnable_ref` + `call_ref` + `trigger_ref` — refs to registered Signals infrastructure objects that don't exist for a scorer driven by an external CLI/API rather than a Weave-native trigger. Those typed columns are gated behind that infrastructure; a custom feedback type sidesteps the coupling while still landing in the queryable feedback table. If turn/session scorers are ever registered as native Weave Signals, `wandb.agent_monitor` and its typed columns become available — not before.

## Dedup and `--force`

Idempotent by construction: before writing, query existing feedback for the same ref + feedback_type. If any exists and `--force` isn't set, skip. If `--force` is set, delete the existing feedback then write fresh (overwrite, not append) — a query-then-delete, since the feedback purge endpoint only filters by `id`.

## Metadata conventions worth knowing

- `scorer_version` is stamped on every payload so scorer-logic changes can be tracked and old versions re-scored later.
- `config_version` and `git_branch` are denormalized onto every score for query convenience (avoids a join back to the turn).
- Scores also carry the turn's own `started_at` (as `turn_started_at`), separate from `scored_at` (when the feedback was written). Trend/regression detection orders by `turn_started_at` — otherwise a single backfill run would write a whole history's worth of scores at nearly the same wall-clock `scored_at`, collapsing the timeline.
