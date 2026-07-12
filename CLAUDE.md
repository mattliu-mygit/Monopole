# weave-agent-signals

Layered evaluation & weak RSI for Weave agent traces. Reads traces from `weave-agent-adapter`, scores them (deterministic → LLM-judge → patterns → self-improvement), writes feedback back to the same Weave project.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/         # lint
ruff format --check src/ tests/ # format
```

## Architecture

- `specs/DESIGN.md` — start here for the full system overview
- `specs/` — detailed specs per component (01–06)
- `src/weave_agent_signals/` — implementation
- Weave entity: `mliu-wandb-weights-biases`, project: `agent-sessions`
- All reads: `POST trace.wandb.ai/agents/spans/query` (custom_attr_columns required)
- All writes: `POST trace.wandb.ai/feedback/create` (or batch variant)

## Scorers

| Scorer | Granularity | Module |
|---|---|---|
| `outcome.test` | turn | `scorers/outcome.py` |
| `outcome.build` | turn | `scorers/outcome.py` |
| `outcome.lint` | turn | `scorers/outcome.py` |
| `outcome.git` | turn | `scorers/outcome.py` |
| `outcome.install` | turn | `scorers/outcome.py` |
| `outcome.command` | turn | `scorers/outcome.py` |
| `judge.verification` | turn | `judges/rubrics.py` |
| `judge.error_recovery` | turn | `judges/rubrics.py` |
| `judge.tool_choice` | turn | `judges/rubrics.py` |
| `judge.completion` | turn | `judges/rubrics.py` |
| `judge.session_outcome` | session | `judges/rubrics.py` |
| `judge.session_autonomy` | session | `judges/rubrics.py` |
| `efficiency` | turn | `scorers/efficiency.py` |
| `efficiency.session` | session | `scorers/efficiency.py` |
| `implicit.correction_density` | session | `scorers/implicit.py` |
| `implicit.abandonment` | session | `scorers/implicit.py` |

Unified API: `scorers.score_turn(turn)` and `scorers.score_session(session)`.

## CLI

```bash
weave-agent-signals score [--since DATE] [--limit N] [--dry-run] [--force]
weave-agent-signals backfill --start DATE [--end DATE] [--page-size N] [--dry-run] [--force]
weave-agent-signals judge [--since DATE] [--limit N] [--rubric NAME,...] [--panel-size N] [--dry-run] [--force]
weave-agent-signals analyze [--limit N] [--ab] [--trends] [--coaching]
weave-agent-signals reflect [--limit N] [--model MODEL] [--iterations N] [--project-root DIR] [--dry-run] [--apply]
weave-agent-signals monitor [--limit N] [--alert-webhook URL] [--state-file PATH] [--dry-run]
weave-agent-signals inspect [--recent N] [--session ID] [--feedback]
```

Exit codes: 0 success, 1 partial failure, 2 config error, 3 API error.

## Testing

Tests use synthetic span data (no live Weave calls). HTTP interactions mocked with `respx`.

```bash
pytest                          # full suite (282 tests)
pytest tests/test_outcome.py    # single module
pytest -x                       # stop on first failure
```

## Conventions

- Scorer names: `<category>.<subcategory>` (e.g. `outcome.test`, `implicit.correction_density`)
- Feedback types: `weave_agent_signals.<scorer_name>`
- Score values: 0.0–1.0 float or bool; stored as `payload.rating` in feedback
- All score payloads include `scorer_version`, `scored_at`; scorer-specific data under `payload.details`
- Scores carry `details.turn_started_at` (agent run time): trend detection orders by this, not `scored_at`, so a single backfill doesn't collapse the timeline
- Children query filters by `trace_id` to prevent cross-turn contamination
- Backfill paginates automatically via cursor-based `query_turns_paginated`
- `--force` deletes existing feedback before writing (not append)

## Weave API quirks

- Spans query: `parent_span_id` is empty string `""` not null; use `$eq ""` not `$not`
- Datetime literals: ClickHouse DateTime64(6) needs bare ISO without timezone suffix (no `Z`, no `+00:00`)
- Feedback query: response key is `result` (not `feedback` or `results`)
- Feedback purge: only accepts `$eq` on `id` field; filter by ref/type requires query-then-delete
- Feedback create: `scorer_*` fields only work with `wandb.agent_monitor` type (requires `runnable_ref`); custom types use `payload` only
- Python env: requires Python 3.11+; use `uv` to create venv: `uv venv .venv --python 3.12`
- **Tool results are JSON-wrapped**: Bash tool results come as `{"stdout": "...", "stderr": "..."}` not plain text. The `_extract_output_text()` helper unwraps this before passing to parsers. `status_code` is often `UNSET` (not `OK`/`ERROR`), so outcome parsers must infer success from output content when `exit_code` is None.
- **W&B Inference endpoint**: `https://api.inference.wandb.ai/v1` (NOT the proxy at `api.wandb.ai/proxy/inference/v1`). Requires `Authorization: Bearer {wandb_api_key}` AND `OpenAI-Project: {entity}/{project}` headers. Model names: `gpt-oss-20b`, `Llama-3.1-8B`, `granite-4.1-8b` etc. A 402 means the account needs inference credits enabled.
- **`include_details: true` is REQUIRED for tool args/results**: the spans query splits columns into a lightweight default set and heavy "detail-only" columns (`tool_call_arguments`, `tool_call_result`, `input_messages`, `output_messages`, `system_instructions`, `reasoning_content`, `*_refs`, `raw_span_dump`). Without `include_details`, those come back NULL — the outcome scorers get no command text or output. The children/hydration query sets it; the turn-list query does not (turn roots have no tool data, and details are expensive). Verified against Weave source: `agent_query_builder.py` `_SPANS_DETAILS_FIELD_NAMES` + `make_spans_list_query` (`details_projection` gated on `req.include_details`). Not allowed together with `group_by`.
