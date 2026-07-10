# Spec 05: Score write-back

How scores are persisted as Weave feedback on agent trace refs. The write path is the mirror of spec 01's read path.

Ties to spec 01 (data flow — ref format), all scorer specs (02–04), future judge/gate scorers (FUTURE.md).

---

## API

### Endpoint

```
POST https://trace.wandb.ai/feedback/create
Authorization: Basic base64("api:" + WANDB_API_KEY)
Content-Type: application/json
```

Batch variant (preferred for backfill):

```
POST https://trace.wandb.ai/feedback/batch/create
```

### Auth

Same as spec 01 — W&B API key from `~/.netrc` or `WANDB_API_KEY`, HTTP Basic.

---

## Ref types

| Granularity | Ref format | Example |
|---|---|---|
| Turn | `weave:///{entity}/{project}/agent_turn/{trace_id}` | `weave:///mliu-wandb-weights-biases/agent-sessions/agent_turn/abc123` |
| Session | `weave:///{entity}/{project}/agent_conversation/{conversation_id}` | `weave:///mliu-wandb-weights-biases/agent-sessions/agent_conversation/conv456` |
| Individual span | `weave:///{entity}/{project}/agent_span/{span_id}` | `weave:///mliu-wandb-weights-biases/agent-sessions/agent_span/sp789` |

**Note**: `conversation_id` values may contain `/` — must be URL-encoded in the ref URI.

---

## Feedback types

### For M1 deterministic scorers

Custom feedback type: `weave_agent_signals.<scorer_name>`.

Rationale: `wandb.agent_monitor` requires `runnable_ref` + `call_ref` + `trigger_ref` (Signals infrastructure refs) which don't exist for external CLI-driven scorers. A custom type avoids this coupling while still landing in the queryable feedback table.

```json
{
  "project_id": "mliu-wandb-weights-biases/agent-sessions",
  "weave_ref": "weave:///mliu-wandb-weights-biases/agent-sessions/agent_turn/abc123",
  "feedback_type": "weave_agent_signals.outcome.test",
  "payload": {
    "scorer_version": "v1",
    "outcomes": [{"framework": "pytest", "passed": 42, "failed": 0}]
  },
  "scorer_ratings": {"_rating_": 1.0},
  "scorer_rating_reasons": {"_rating_": "All 42 tests passed"},
  "scorer_rating_confidences": {"_rating_": 1.0},
  "scorer_tags": ["test_pass"],
  "scorer_tag_reasons": {"test_pass": "pytest: 42 passed, 0 failed"},
  "scorer_tag_confidences": {"test_pass": 1.0}
}
```

### For L2 Signals-native judges (M2+)

Type: `wandb.agent_monitor` — requires the Signals infrastructure refs.

When we register custom Signals (M2), the scorer is a Weave op with a proper `runnable_ref`, and the trigger is a registered trigger object. These refs come from the Signals registration API, not from us.

### Scorer naming convention

| Scorer | `feedback_type` |
|---|---|
| Test outcome | `weave_agent_signals.outcome.test` |
| Build outcome | `weave_agent_signals.outcome.build` |
| Lint outcome | `weave_agent_signals.outcome.lint` |
| Git outcome | `weave_agent_signals.outcome.git` |
| Verified before done | `weave_agent_signals.outcome.verified_before_done` |
| Turn frustration | `weave_agent_signals.implicit.frustration` |
| Session frustration | `weave_agent_signals.implicit.session_frustration` |
| Abandonment | `weave_agent_signals.implicit.abandonment` |
| Correction density | `weave_agent_signals.implicit.correction_density` |
| Efficiency | `weave_agent_signals.efficiency` |
| Session efficiency | `weave_agent_signals.efficiency.session` |
| Routing gate | `weave_agent_signals.gate` |

---

## Typed scorer columns

The feedback schema supports structured output columns:

| Column | Type | Use |
|---|---|---|
| `scorer_ratings` | `dict[str, float]` | Continuous scores (0–1). Key `_rating_` for the primary score. |
| `scorer_rating_reasons` | `dict[str, str]` | Human-readable reason per rating |
| `scorer_rating_confidences` | `dict[str, float]` | Confidence per rating (0–1) |
| `scorer_tags` | `list[str]` | Categorical labels |
| `scorer_tag_reasons` | `dict[str, str]` | Reason per tag |
| `scorer_tag_confidences` | `dict[str, float]` | Confidence per tag |
| `payload` | `dict` | Freeform metadata (scorer-specific detail) |

All scores use `scorer_ratings["_rating_"]` as the primary numeric score, with tags for categorical classification.

---

## Dedup

Idempotent scoring: before writing, check if feedback with the same `feedback_type` already exists for this ref.

```
POST https://trace.wandb.ai/feedback/query
{
  "project_id": "...",
  "query": {
    "$expr": {
      "$and": [
        {"$eq": [{"$getField": "weave_ref"}, {"$literal": "<ref>"}]},
        {"$eq": [{"$getField": "feedback_type"}, {"$literal": "<type>"}]}
      ]
    }
  }
}
```

If feedback exists: skip (default) or `--force` to overwrite (delete old + create new).

**OPEN**: exact feedback query filter syntax — need to verify against live API. Alternative: maintain a local "scored" set in a state file and reconcile periodically.

---

## Batch write

For backfill efficiency, batch multiple scores into one request:

```json
{
  "batch": [
    {"weave_ref": "...", "feedback_type": "...", "scorer_ratings": {"_rating_": 0.95}, ...},
    {"weave_ref": "...", "feedback_type": "...", "scorer_ratings": {"_rating_": 0.0}, ...}
  ]
}
```

**OPEN**: batch size limits (if any) on the trace-server. Start conservative (50 per batch), increase if no errors.

---

## Score metadata conventions

Every score's `payload` includes:

```json
{
  "scorer_version": "v1",
  "scored_at": "2026-07-09T12:00:00Z",
  "config_version": "cafcafde5ec7",
  "git_branch": "main"
}
```

`scorer_version` enables re-scoring when the scorer logic changes (filter by version, re-score old versions). `config_version` and `git_branch` are denormalized from the turn for query convenience.
