# weave-agent-signals

Layered evaluation & weak RSI for Weave agent traces. Reads traces from `weave-agent-adapter`, scores them (deterministic → LLM-judge → patterns → self-improvement), writes feedback back to the same Weave project.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

- `specs/DESIGN.md` — start here for the full system overview
- `specs/` — detailed specs per component (01–11)
- `src/weave_agent_signals/` — implementation
- Weave entity: `mliu-wandb-weights-biases`, project: `agent-sessions`
- All reads: `POST trace.wandb.ai/agents/spans/query` (custom_attr_columns required)
- All writes: `POST trace.wandb.ai/feedback/create` (or batch variant)

## Testing

Tests use synthetic span data (no live Weave calls). HTTP interactions mocked with `respx`.

```bash
pytest                          # full suite
pytest tests/test_outcome.py    # single module
pytest -x                       # stop on first failure
```

## Conventions

- Scorer names: `<category>.<subcategory>` (e.g. `outcome.test`, `implicit.frustration`)
- Feedback types: `weave_agent_signals.<scorer_name>`
- Score values: 0.0–1.0 float or bool; `scorer_ratings["_rating_"]` is the primary
- All scores include `scorer_version` in payload metadata
