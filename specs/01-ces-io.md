# CES I/O

This boundary reads the agent sessions that `weave-agent-adapter` writes to the
CoreWeave Evaluations service (CES) and writes Monopole's scores back as CES
Scores. `ces_client.py` holds every request; `ces_catalog.py` names the Scorers.

## Identity model

A session is the adapter's session Trajectory, and its Trajectory ID is the
session's `conversation_id`. A turn is the session's Cases that share a
`turn_id`, ordered by `step_index`. The turn's `trace_id` is its first Case ID,
and the turn keeps every member Case ID in step order. A subagent is a child
Trajectory rooted at the `tool_call` Case that launched it.

Every turn resolves to one trace role from its `trace_role` field. Only a session
whose every turn is `agent_session` enters listing, detail, scoring, judging, or
evaluation-run selection. Mixed-role and non-agent sessions are excluded whole.

Cases and Scores never change. An evaluation run pins its session Trajectory IDs
and each turn's Case IDs (cohort schema v3). Every stage rereads the pinned
sessions, keeps only the pinned turns, and fails closed when a session or turn is
missing or its pinned metadata differs. Turns added after pinning are ignored.

## Reading sessions

`read_session` lists the session Trajectory's Cases and every child Trajectory's
Cases. Turn status maps `completed` to `OK`, `aborted` to `ERROR`, and anything
else to `UNSET`. A turn without a user prompt starts with a reply or tool call;
that span's ID gets a `/span` suffix so judge evidence IDs stay unique.

`find_sessions` reads turn rows newest first and stops at the first turn before
`since`, because CES filters cannot range-compare string fields.

## Scores

Each Monopole scorer is two CES Scorers: `<name>` holds the number or boolean
value, and `<name>.details` holds tags, reason, confidence, and scorer details as
JSON. Turn Scores target the turn's first Case; session Scores target the session
Trajectory. The `monopole turns` (Case) and `monopole sessions` (Trajectory)
Evaluations select them over the adapter's Dataset. Each judge rubric and panel
has its own Scorer, `<rubric>.panel-<digest>`, and only finished reviews write.

A target's Scores and their details go in one create-scores request keyed
`score-v2:<pass>:<kind>:<target>:<body digest>`. A run's pass is its run ID; a
command's pass is a new UUID. Reads return the newest Score per Scorer, so a
forced rescore writes newer Scores and never purges. Without force, a Scorer that
already has a Score on the target is skipped.

Turn Scores are read through the turns Evaluation's rows; session Scores through
its Trajectory rows, which include Scores written from anywhere.

Analysis, reflection, and the session pages read Scores as records shaped
`{id, feedback_type, target_id, payload}`. Reflection pins each record's Score
ID and digest.

## External correctness constraints

1. Every path sits under `/v1/namespaces/wandb/scopes/<scope>`, where the scope
   is the W&B project's internal ID that `weave-agent-adapter evaluations setup`
   records.
2. Filters name input fields with `i.`. String fields support only equality and
   pattern operators.
3. Creates are idempotent by `Idempotency-Key`; reusing a key with a different
   body is a conflict.
4. `429` and `503` are retried after `Retry-After`. CES answers `503
   capacity_unavailable` while it runs its few concurrent queries, so that error
   gets more attempts. It answers `503 storage_capacity_exceeded` when one page
   reads too much, which a session with large Cases hits at 1,000 Cases; lists
   and searches then halve their page size until CES accepts it, instead of
   resending the same page.
5. list-trajectory-rows has no filter, so session Score reads page every row.
6. CoreWeave Serverless Inference uses bearer W&B authentication plus the
   entity/project header; HTTP 402 means inference credits are unavailable.

Credentials come from `WANDB_API_KEY` or the `api.wandb.ai` netrc entry. The CES
URL, scope, and Dataset come from flags, `WEAVE_AGENT_SIGNALS_CES_*` env vars, or
the adapter's `~/.weave-agent-adapter/evaluations.json`.
