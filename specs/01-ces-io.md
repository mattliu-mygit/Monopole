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
its Trajectory rows. Until the Trajectory-row route deploys, a 404 or 405 falls
back to `ces_score_index.py`, a local index of the session Score IDs this machine
wrote. Delete that module and its fallbacks once the route is deployed.

Analysis, reflection, and the session pages read Scores as records shaped
`{id, feedback_type, target_id, payload}`. Reflection pins each record's Score
ID and digest.

## Queries

`ces_queries.py` asks CES for each config's count, mean, and spread per Scorer
with two saved EvaluationQueries. A Case query groups turn Scores by
`i.config_version`; a Trajectory query groups session Scores by the root Case's
`i.config_version` and `depth`. Turn Scores target only a turn's first Case, so
its other Cases have no Score and drop out of a mean and count. Every scored
Case belongs to the main agent, because turns are read from the session
Trajectory's own Cases and the adapter writes subagent Cases to child
Trajectories. The turn query does not group by `i.agent_id`: CES rejects grouping
by a field no Case has set, which is every Dataset that never ran a subagent.

The turn query runs on the turns Evaluation and the session query on the
sessions Evaluation, which already selects every session Scorer; judge panel
Scorers are found in its selection. A CES deployed before wandb/core#61599 runs
queries only on an Evaluation whose primary target kind is case, and answers a
session query on the sessions Evaluation with a 400 saying so. On that 400, and
only that one, Monopole falls back for the rest of the process: the turns
Evaluation also selects every session Scorer and runs the session query. The
fallback relies on the turns Evaluation accepting new Scorers: a fixed
Evaluation, whose Scorer list is chosen at creation, would reject them with 409.
Delete the fallback once wandb/core#61599 is deployed.

A query definition never changes. Its create is keyed
`monopole-query-v1:<evaluation>:<body digest>`, so a rerun replays to the same
query ID, a query saved on the turns Evaluation is never reused for the sessions
Evaluation, and a new Scorer such as a new judge panel makes a new query. A query
holds at most 100 metrics, so a Scorer list that needs more is split. Query calls
retry a busy CES's 503 with waits of 1, 2, 4, 8, and 16 seconds, because CES runs
only two queries at a time per instance.

## External correctness constraints

1. Every path sits under `/v1/namespaces/wandb/scopes/<scope>`, where the scope
   is the W&B project's internal ID that `weave-agent-adapter evaluations setup`
   records.
2. Filters name input fields with `i.`. String fields support only equality and
   pattern operators.
3. Creates are idempotent by `Idempotency-Key`; reusing a key with a different
   body is a conflict.
4. `429` and `503` are retried with doubling waits, never shorter than
   `Retry-After`.
5. list-trajectory-rows has no filter, so session Score reads page every row.
6. CoreWeave Serverless Inference uses bearer W&B authentication plus the
   entity/project header; HTTP 402 means inference credits are unavailable.

Credentials come from `WANDB_API_KEY` or the `api.wandb.ai` netrc entry. The CES
URL, scope, and Dataset come from flags, `WEAVE_AGENT_SIGNALS_CES_*` env vars, or
the adapter's `~/.weave-agent-adapter/evaluations.json`.
