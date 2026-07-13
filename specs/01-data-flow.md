# Spec 01: Data flow & span reader

How weave-agent-signals reads agent traces from Weave and hydrates them into scorable models. Ties to spec 05 (write-back is the mirror of this read path).

## Read path

All reads go directly against the Weave trace-server HTTP API via `httpx` — no SDK `weave.init()` needed for reading.

**Turns** (`agents/spans/query`): root turn spans (`invoke_agent` operation, no parent) carry per-turn token counts, model, status, and the adapter's custom attrs (`config_version`, `steering_count`, `denial_count`, `tool_error_count`, etc.).

**Gotcha**: custom attrs are only returned when explicitly requested via `custom_attr_columns` in the query — otherwise `custom_attrs` comes back empty. Easy to silently lose all adapter signal if a new query path forgets this (verified in M0' E2E). CLAUDE.md's `include_details` quirk is the same "ask or get nothing" shape, for tool arguments/results instead of custom attrs.

**Turn children** (tool calls, chat spans, subagents): queried by `trace_id`, not `conversation_id` — scoping to the single turn avoids pulling in spans from sibling turns in the same session.

**Sessions**: there's no session root span in this spans-only architecture. A session is just the set of turns sharing a `conversation_id` — grouping, not a distinct queryable entity. Session-level scores attach to a synthesized `agent_conversation` ref (spec 05).

**Incremental scoring**: to avoid re-scoring, query recent turns, check existing feedback for those refs, and score only what's missing. This is what makes `score` (frequent, small) cheap relative to `backfill` (full history).

## Hydration model

Raw spans are hydrated into typed models (`TurnSpan`, `ToolSpan`, `ChatSpan`, `SessionView` in `models.py`) that scorers consume directly rather than working with raw span dicts. Turn and session refs are derived deterministically from `trace_id` / `conversation_id`, since Weave's ref format isn't otherwise discoverable from the query response.

## Entity/project configuration

Hardcoded for the single-user case (`mliu-wandb-weights-biases/agent-sessions`). Not worth parameterizing until there's a second user or project.

## Auth

W&B API key from `~/.netrc` or `WANDB_API_KEY`, HTTP Basic — same credential the CLI and adapter already use.

## Rate limiting

No documented rate limits on trace-server reads. `backfill` pages through history via cursor-based pagination rather than one giant query, mainly to bound response size — not (yet) to dodge a rate limit.
