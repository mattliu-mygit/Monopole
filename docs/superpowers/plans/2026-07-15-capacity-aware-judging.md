# Capacity-Aware Judging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route each session only to selected judges whose real context capacity can hold every intact turn and the complete sliding-window protocol.

**Architecture:** Model descriptors remain the authoritative source for context capacity and token-counter identity. Window planning uses the reviewer descriptor to select either `tiktoken` or the existing conservative byte heuristic, applies a 100K or 50K reserve tier, and records an authenticated per-reviewer planned-or-skipped disposition. The panel keeps selected order but excludes skipped reviewers from inference and aggregation.

**Tech Stack:** Python 3.11+, Pydantic, `tiktoken`, pytest, SQLite-backed run persistence.

## Global Constraints

- Do not split, truncate, deduplicate, or summarize an indivisible raw turn.
- Models with more than 200,000 context tokens reserve at least 100,000 tokens.
- Models with 200,000 context tokens or fewer reserve at least 50,000 tokens.
- Protocol overhead may increase the reserve but never reduce the tier reserve.
- Use `tiktoken` only for explicitly cataloged OpenAI encodings; use `utf8_bytes_div_3` elsewhere.
- Skipped reviewers are not inference failures and must remain visible in the pinned plan and attempt records.
- The local run database is disposable; bump schemas rather than adding compatibility adapters.

---

### Task 1: Pin model capacities and token counters

**Files:**
- Create: `src/weave_agent_signals/judges/tokens.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/weave_agent_signals/run_config.py`
- Modify: `src/weave_agent_signals/catalogs.py`
- Test: `tests/evaluation/judging/test_tokens.py`
- Test: `tests/evaluation/test_catalogs.py`
- Test: `tests/runs/test_config.py`

**Interfaces:**
- Produces: `TokenCounterName`, `count_tokens(text: str, counter: TokenCounterName) -> int`.
- Produces: `ModelDescriptor.token_counter` and corrected `max_input_tokens` values.

- [ ] **Step 1: Add failing counter and catalog tests**

```python
def test_utf8_counter_remains_conservative():
    assert count_tokens("", "utf8_bytes_div_3") == 0
    assert count_tokens("abcd", "utf8_bytes_div_3") == 2


def test_openai_counter_uses_pinned_encoding():
    assert count_tokens("hello world", "o200k_base") == 2


def test_large_models_publish_real_capacity_and_counter():
    catalog = build_model_catalog(which=lambda name: f"/bin/{name}")
    assert catalog.model("gpt-5.6-sol").max_input_tokens == 1_050_000
    assert catalog.model("gpt-5.6-sol").token_counter == "o200k_base"
    assert catalog.model("claude-sonnet-5").max_input_tokens == 1_000_000
    assert catalog.model("claude-sonnet-5").token_counter == "utf8_bytes_div_3"
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `.venv/bin/python -m pytest -q tests/evaluation/judging/test_tokens.py tests/evaluation/test_catalogs.py tests/runs/test_config.py`

Expected: failure because `tokens.py` and `ModelDescriptor.token_counter` do not exist.

- [ ] **Step 3: Add the dependency and counter boundary**

```python
TokenCounterName = Literal["utf8_bytes_div_3", "o200k_base", "o200k_harmony"]


def count_tokens(text: str, counter: TokenCounterName) -> int:
    if not text:
        return 0
    if counter == "utf8_bytes_div_3":
        return (len(text.encode("utf-8")) + 2) // 3
    return len(tiktoken.get_encoding(counter).encode(text))
```

Add `tiktoken>=0.12` to base dependencies and refresh `uv.lock` with `uv lock`.

- [ ] **Step 4: Extend and populate the authoritative model descriptor**

```python
class ModelDescriptor(StrictFrozenModel):
    id: StrictStr
    label: StrictStr
    family: StrictStr
    backend: StrictStr
    supported_roles: tuple[ModelRole, ...]
    max_input_tokens: Annotated[int, Field(strict=True, ge=1)] = 128_000
    token_counter: TokenCounterName = "utf8_bytes_div_3"
```

Set capacities to 1,050,000 for GPT-5.6 Sol, 1,000,000 for Claude Sonnet 5,
200,000 for Claude Haiku 4.5, 131,072 for GPT-OSS/Llama/Granite, and 128,000
for GPT-4o models. Pin GPT-5/GPT-4o to `o200k_base`, GPT-OSS to
`o200k_harmony`, and all other models to `utf8_bytes_div_3`.

Bump `PIPELINE_VERSION`, `MODEL_CATALOG_SCHEMA_VERSION`, and
`EFFECTIVE_RUN_CONFIG_SCHEMA_VERSION`; update `EffectiveRunConfig.schema_version`.
Remove the validator that compares every selected model to the retired fixed
`target_input_tokens` ceiling.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest -q tests/evaluation/judging/test_tokens.py tests/evaluation/test_catalogs.py tests/runs/test_config.py`

Expected: all pass.

- [ ] **Step 6: Commit the independently testable metadata boundary**

```bash
git add pyproject.toml uv.lock src/weave_agent_signals/judges/tokens.py src/weave_agent_signals/run_config.py src/weave_agent_signals/catalogs.py tests/evaluation/judging/test_tokens.py tests/evaluation/test_catalogs.py tests/runs/test_config.py
git commit -m "feat: pin judge context capacities"
```

### Task 2: Make window planning capacity-aware

**Files:**
- Modify: `src/weave_agent_signals/run_config.py`
- Modify: `src/weave_agent_signals/judges/windowing.py`
- Modify: `src/weave_agent_signals/judges/sliding.py`
- Modify: `src/weave_agent_signals/judges/sliding_contracts.py`
- Modify: `src/weave_agent_signals/routes/models.py`
- Test: `tests/evaluation/judging/test_windowing.py`
- Test: `tests/evaluation/judging/test_sliding.py`
- Test: `tests/evaluation/judging/test_sliding_contracts.py`

**Interfaces:**
- Consumes: `count_tokens` and `ModelDescriptor.token_counter` from Task 1.
- Produces: `WindowPlanInapplicable(reason: str)` for capacity-only planning rejection.
- Produces: reviewer-specific `build_window_plan(session: SessionView, policy: JudgingContextPolicy, model_limit: int, token_counter: TokenCounterName) -> dict[str, object]`.

- [ ] **Step 1: Replace fixed-target fixtures with reserve-tier tests**

```python
def test_large_model_reserves_100k():
    plan = build_window_plan(session, policy, model_limit=1_000_000,
                             token_counter="utf8_bytes_div_3")
    assert plan["capacity_reserve_tokens"] == 100_000
    assert plan["input_cap_tokens"] == 1_000_000


def test_small_model_reserves_50k():
    plan = build_window_plan(session, policy, model_limit=200_000,
                             token_counter="utf8_bytes_div_3")
    assert plan["capacity_reserve_tokens"] == 50_000


def test_intact_turn_that_exceeds_raw_budget_is_inapplicable():
    with pytest.raises(WindowPlanInapplicable) as caught:
        build_window_plan(huge_session, policy, model_limit=200_000,
                          token_counter="utf8_bytes_div_3")
    assert caught.value.reason == "insufficient_context_capacity"
```

- [ ] **Step 2: Run window tests and confirm they fail**

Run: `.venv/bin/python -m pytest -q tests/evaluation/judging/test_windowing.py tests/evaluation/judging/test_sliding.py tests/evaluation/judging/test_sliding_contracts.py`

Expected: failures reference the removed fixed target and missing counter argument.

- [ ] **Step 3: Replace fixed target policy with reserve tiers**

```python
class JudgingContextPolicy(StrictFrozenModel):
    contract_version: Literal["2"] = "2"
    large_model_threshold_tokens: int = 200_000
    large_model_reserve_tokens: int = 100_000
    small_model_reserve_tokens: int = 50_000
    prompt_reserve_tokens: int = 6_000
    output_reserve_tokens: int = 4_000
    safety_reserve_tokens: int = 8_000
    digest_max_tokens: int = 1_000
    finding_max_tokens: int = 750
    overlap_turns: Literal[1] = 1
    max_chunks: int = 40

    def capacity_reserve(self, model_limit: int) -> int:
        return (self.large_model_reserve_tokens
                if model_limit > self.large_model_threshold_tokens
                else self.small_model_reserve_tokens)
```

- [ ] **Step 4: Count rendered turns and requests with the reviewer counter**

Change `build_window_plan` to use the model limit directly, calculate each
turn with `count_tokens`, and compute the fixed point as:

```python
protocol_overhead = base_reserve + max(0, chunk_count - 1) * policy.digest_max_tokens
capacity_reserve = max(policy.capacity_reserve(model_limit), protocol_overhead)
raw_budget = model_limit - capacity_reserve
```

Keep merge bounded independently by
`base_reserve + chunk_count * (digest_max_tokens + finding_max_tokens)`.
Raise `WindowPlanInapplicable("insufficient_context_capacity")` only for
capacity cases; malformed evidence and non-convergence remain ordinary errors.
Persist `token_counter` and `capacity_reserve_tokens` in the window plan.

Update `SlidingReviewer._messages_fit` to call the same counter against
`judge.max_input_tokens`. Keep the conservative byte counter in
`sliding_contracts.py` for response-size contract bounds.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest -q tests/evaluation/judging/test_windowing.py tests/evaluation/judging/test_sliding.py tests/evaluation/judging/test_sliding_contracts.py`

Expected: all pass.

- [ ] **Step 6: Commit capacity-aware planning**

```bash
git add src/weave_agent_signals/run_config.py src/weave_agent_signals/judges/windowing.py src/weave_agent_signals/judges/sliding.py src/weave_agent_signals/judges/sliding_contracts.py src/weave_agent_signals/routes/models.py tests/evaluation/judging/test_windowing.py tests/evaluation/judging/test_sliding.py tests/evaluation/judging/test_sliding_contracts.py
git commit -m "feat: size judge windows by model capacity"
```

### Task 3: Skip incapable reviewers without failing the run

**Files:**
- Modify: `src/weave_agent_signals/judges/plan.py`
- Modify: `src/weave_agent_signals/judges/review.py`
- Modify: `src/weave_agent_signals/judges/runner.py`
- Modify: `src/weave_agent_signals/runs/stages/judging.py`
- Modify: `src/weave_agent_signals/runs/store.py`
- Modify: `src/weave_agent_signals/routes/models.py`
- Test: `tests/evaluation/judging/test_plan.py`
- Test: `tests/evaluation/judging/test_panel.py`
- Test: `tests/evaluation/judging/test_session_judge.py`
- Test: `tests/runs/stages/test_judging.py`
- Test: `tests/runs/test_store.py`

**Interfaces:**
- Consumes: `WindowPlanInapplicable` from Task 2.
- Produces: reviewer plan rows with `status`, `skip_reason`, nullable `window_plan`, and zero work bounds when skipped.
- Produces: `AttemptObservation(status="skipped", skip_reason="insufficient_context_capacity")`.

- [ ] **Step 1: Add failing mixed-capacity and all-skipped tests**

```python
def test_plan_skips_only_incapable_reviewer():
    plan = _plan(judges=(small_judge, large_judge), sessions=(huge_session,))
    assert [row["status"] for row in plan["sessions"][0]["reviewers"]] == [
        "skipped", "planned"
    ]
    assert plan["sessions"][0]["reviewers"][0]["skip_reason"] == \
        "insufficient_context_capacity"


def test_panel_aggregates_success_and_ignores_skip():
    outcome = execute_panel(_judges(2), _ScriptedInvoke((_skipped(), _success(0.8))),
                            threshold=0.5)
    assert outcome.status == "degraded"
    assert outcome.rating == 0.8
```

Add a stage test whose oversized session makes every selected reviewer
inapplicable; assert `not_evaluable_rubrics == 1` and that the fake chat
client records zero calls.

- [ ] **Step 2: Run focused plan/panel/stage tests and confirm they fail**

Run: `.venv/bin/python -m pytest -q tests/evaluation/judging/test_plan.py tests/evaluation/judging/test_panel.py tests/evaluation/judging/test_session_judge.py tests/runs/stages/test_judging.py tests/runs/test_store.py`

Expected: plan construction still raises and skipped observations are unsupported.

- [ ] **Step 3: Pin planned-or-skipped reviewer dispositions**

Catch only `WindowPlanInapplicable` in `build_judging_plan`. A skipped row keeps
the exact ordered judge descriptor, sets `window_plan` to null, records the
stable reason, and uses zero work bounds. Bump the judging plan schema and
window-plan contract versions. Update strict store validation rather than
adding a retired-schema adapter.

- [ ] **Step 4: Model skipped attempts and aggregate eligible reviewers**

```python
ObservationStatus = Literal["succeeded", "abstained", "failed", "skipped"]

if self.status == "skipped":
    if self.skip_reason != "insufficient_context_capacity":
        raise ValueError("skipped observations require a known skip reason")
    if any((self.resolved_model, self.score, self.rationale, self.error_type)):
        raise ValueError("skipped observations cannot contain inference output")
    return
```

`execute_panel` treats skipped reviewers as non-failures, returns `degraded`
when at least one eligible reviewer succeeds, and returns `not_evaluable` when
there are no eligible reviewers. Any eligible reviewer failure still fails the
panel. `judge_session` returns the synthetic skipped observation from the
pinned reviewer row and constructs `SlidingReviewer` only for planned rows.

Count only non-skipped attempts in progress attempt totals. Keep skipped rows
in attempt summaries so analysis and reflection inputs can distinguish absent
coverage from inference failure.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest -q tests/evaluation/judging/test_plan.py tests/evaluation/judging/test_panel.py tests/evaluation/judging/test_session_judge.py tests/runs/stages/test_judging.py tests/runs/test_store.py`

Expected: all pass.

- [ ] **Step 6: Commit reviewer applicability**

```bash
git add src/weave_agent_signals/judges/plan.py src/weave_agent_signals/judges/review.py src/weave_agent_signals/judges/runner.py src/weave_agent_signals/runs/stages/judging.py src/weave_agent_signals/runs/store.py src/weave_agent_signals/routes/models.py tests/evaluation/judging/test_plan.py tests/evaluation/judging/test_panel.py tests/evaluation/judging/test_session_judge.py tests/runs/stages/test_judging.py tests/runs/test_store.py
git commit -m "feat: skip judges without enough context"
```

### Task 4: Reconcile documentation, remove retired paths, and verify

**Files:**
- Modify: `specs/02-evaluation.md`
- Modify: `specs/04-evaluation-runs.md`
- Modify: `docs/superpowers/specs/2026-07-15-capacity-aware-judging-design.md`
- Delete after reconciliation: `docs/superpowers/plans/2026-07-15-capacity-aware-judging.md`

**Interfaces:**
- Consumes: final behavior from Tasks 1-3.
- Produces: canonical specs matching the implementation, with temporary plan removed.

- [ ] **Step 1: Search for stale fixed-limit and retired design language**

Run: `rg -n "target_input_tokens|128_000|schema_version.*2|token_estimator|dedup|calibration" src tests specs docs/superpowers`

Expected: only intentional historical or unrelated matches remain after cleanup.

- [ ] **Step 2: Update canonical specs and remove transitional documentation**

Document the observable 100K/50K reserve policy, model-specific counting,
intact-turn invariant, and reviewer skip behavior in canonical specs. Remove
the completed temporary plan after its durable decisions are consolidated.

- [ ] **Step 3: Run focused judging verification**

Run: `.venv/bin/python -m pytest -q tests/evaluation/judging tests/evaluation/test_catalogs.py tests/runs/test_config.py tests/runs/stages/test_judging.py tests/runs/test_store.py`

Expected: all pass.

- [ ] **Step 4: Run backend quality gates once**

Run: `.venv/bin/python -m pytest -q`

Run: `.venv/bin/ruff check src/ tests/`

Run: `.venv/bin/ruff format --check src/ tests/`

Expected: all commands exit zero.

- [ ] **Step 5: Review every changed file and commit cleanup**

Run: `git diff --check && git status --short && git diff --stat`

Read every changed source, test, and spec file. Remove dead fixed-target code,
duplicate helpers, and stale assertions before committing.

```bash
git add specs/02-evaluation.md specs/04-evaluation-runs.md docs/superpowers/specs/2026-07-15-capacity-aware-judging-design.md
git add -u docs/superpowers/plans/2026-07-15-capacity-aware-judging.md
git commit -m "docs: consolidate capacity-aware judging"
```
