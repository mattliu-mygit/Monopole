# Spec 01: Data flow & span reader

How weave-agent-signals reads agent traces from Weave and hydrates them into scorable models. Ties to spec 05 (write-back).

---

## Read path

All reads go through the Weave trace-server HTTP API. No SDK `weave.init()` needed for reading — direct `httpx` calls with Basic auth (W&B API key).

### Turn-level: `agents/spans/query`

```
POST https://trace.wandb.ai/agents/spans/query
Authorization: Basic base64("api:" + WANDB_API_KEY)
Content-Type: application/json
```

```json
{
  "project_id": "mliu-wandb-weights-biases/agent-sessions",
  "limit": 100,
  "sort_by": [{"field": "started_at", "direction": "desc"}],
  "query": {
    "$expr": {
      "$and": [
        {"$eq": [{"$getField": "operation_name"}, {"$literal": "invoke_agent"}]},
        {"$not": [{"$getField": "parent_span_id"}]}
      ]
    }
  },
  "custom_attr_columns": [
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.config_version"},
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.git_branch"},
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.effort_level"},
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.session_id"},
    {"source": "custom_attrs_int", "key": "weave_agent_adapter.steering_count"},
    {"source": "custom_attrs_int", "key": "weave_agent_adapter.denial_count"},
    {"source": "custom_attrs_int", "key": "weave_agent_adapter.tool_error_count"}
  ]
}
```

**Critical**: custom attrs are returned ONLY when explicitly requested via `custom_attr_columns`. Without this, `custom_attrs` is empty (`{}`). Verified in M0' E2E.

Returns: list of root turn spans with `trace_id`, `conversation_id`, `started_at`, `ended_at`, `input_tokens`, `output_tokens`, `model`, `status_code`, custom attrs, and `events_dump` (JSON string of span events).

### Turn children: tool spans

Same endpoint, filtered by parent:

```json
{
  "query": {
    "$expr": {
      "$eq": [{"$getField": "conversation_id"}, {"$literal": "<conversation_id>"}]
    }
  }
}
```

Returns all spans in the conversation (root + children). Filter client-side by `operation_name`:
- `execute_tool` → tool call spans (have `gen_ai.tool.call.name`, `.arguments`, `.result`)
- `chat` → LLM call spans (have model, tokens, input/output messages)
- `invoke_agent` with `parent_span_id` → subagent spans

### Session-level: conversation grouping

No session root span exists in the spans-only architecture. Sessions are groups of turns sharing a `conversation_id`. To get all turns in a session:

```json
{
  "query": {
    "$expr": {
      "$and": [
        {"$eq": [{"$getField": "conversation_id"}, {"$literal": "<conversation_id>"}]},
        {"$not": [{"$getField": "parent_span_id"}]}
      ]
    }
  },
  "sort_by": [{"field": "started_at", "direction": "asc"}]
}
```

### Incremental scoring: "what's unscored?"

To avoid re-scoring turns that already have feedback, the scorer:
1. Queries recent turn spans (by `started_at` range or last N)
2. Queries existing feedback for those refs: `POST /feedback/query` with `weave_ref` filter
3. Scores only turns without feedback from this scorer

**OPEN**: exact shape of `feedback/query` filtering — need to verify `feedback_type` + `weave_ref` filter syntax.

---

## Hydration model

Raw spans are hydrated into typed models for scorer consumption:

```python
@dataclass
class TurnSpan:
    trace_id: str
    conversation_id: str
    started_at: datetime
    ended_at: datetime | None
    model: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    status_code: str              # "OK", "ERROR", "UNSET"

    # from custom attrs
    config_version: str | None
    git_branch: str | None
    effort_level: str | None
    session_id: str | None
    steering_count: int
    denial_count: int
    tool_error_count: int

    # from events_dump (JSON string → parsed)
    events: list[SpanEvent]

    # hydrated separately (child spans)
    tool_calls: list[ToolSpan]
    chat_spans: list[ChatSpan]
    subagents: list[SubagentSpan]

    @property
    def ref(self) -> str:
        return f"weave:///mliu-wandb-weights-biases/agent-sessions/agent_turn/{self.trace_id}"


@dataclass
class ToolSpan:
    span_id: str
    tool_name: str
    arguments: str      # JSON string, up to 32KB
    result: str         # JSON string, up to 32KB
    status_code: str
    started_at: datetime
    ended_at: datetime | None


@dataclass
class ChatSpan:
    span_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    finish_reason: str | None


@dataclass
class SessionView:
    conversation_id: str
    turns: list[TurnSpan]       # ordered by started_at
    config_version: str | None  # from first turn (or most common)
    git_branch: str | None

    @property
    def ref(self) -> str:
        return f"weave:///mliu-wandb-weights-biases/agent-sessions/agent_conversation/{self.conversation_id}"

    @property
    def total_tokens(self) -> int:
        return sum(t.input_tokens + t.output_tokens for t in self.turns)
```

---

## Entity/project configuration

Hardcoded for now (single user): `mliu-wandb-weights-biases/agent-sessions`. Parameterizable later via config or CLI flag.

## Auth

W&B API key from `~/.netrc` (entry for `api.wandb.ai`) or `WANDB_API_KEY` env var. HTTP Basic: `base64("api:" + key)`.

## Rate limiting

The trace-server API has no documented rate limits for reads. Backfill should still be polite — batch queries by conversation, sleep between pages. Write-back via `/feedback/batch/create` (spec 05) reduces round-trips.
