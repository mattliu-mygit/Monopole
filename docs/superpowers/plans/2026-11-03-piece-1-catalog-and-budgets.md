# Piece 1: Catalog, Reading Budgets and Scaled Reserves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every catalog model correct context sizes and its own pinned reading budgets, and
make the reserves scale with model size so small models can be planned at all.

**Architecture:** Budgets live on `ModelDescriptor`, so they are pinned catalog data covered by
`catalog_version`. `JudgingContextPolicy` computes one reserve from the model's size, and both
`build_window_plan` and the frontend capacity estimate use it. The open-model route keeps the
internal provider id `wandb`, but users see it as CoreWeave Serverless Inference.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, ruff; React, TypeScript, vitest, generated
OpenAPI types.

**Spec:** `docs/superpowers/specs/2026-11-03-long-session-judging-design.md` (piece 1 and the
"Reading plans" section). Starting budgets come from
`docs/small-context-judging-techniques.md` section A7.

This is the first of ten per-piece plans. Pieces 2–10 get their own plans, written after this
piece lands.

## Global Constraints

- **Database:** no run database migrations. The database is disposable (`AGENTS.md`).
- **Pinning:** every descriptor or policy change changes `catalog_version`, which is expected.
  Runs pinned to an old catalog fail closed, as today.
- **Policy contract:** bump `JudgingContextPolicy.contract_version` from `"3"` to `"4"`, because
  the reserve semantics change.
- **Provider id:** stays `wandb`. The user-facing route name is exactly
  `CoreWeave Serverless Inference`.
- **Endpoint and key:** unchanged. `INFERENCE_BASE = "https://api.inference.wandb.ai/v1"`, with
  the existing key lookup (`auth.get_api_key`).
- **Clean replacement:** remove superseded code (`raw_window_target_tokens`, the reserve tiers)
  instead of keeping compatibility paths.
- **Commits:** match `git log` style (`feat:`, `fix:`, `docs:`) with the Devin trailer. Each task
  ends with backend tests, ruff, and the frontend checks where the frontend changed.

## Review Focus

1. **A context window of 18k or less**, such as an 8,192-token model, gets a positive usable
   budget instead of `WindowPlanInapplicable`. Test in Task 1.
2. **Budgets in the wrong order,** or larger than what fits after the reserve, are rejected when
   the catalog is built, with a message naming the model and field. Test in Task 2.
3. **Changing one model's budget changes `catalog_version`.** Test in Task 2.
4. **A run config naming a model removed from the catalog** fails validation, with a message
   naming the id. Test in Task 3.
5. **Frontend and backend reserve arithmetic match** for 8,192, 32,800, 131,072 and 1,000,000
   tokens. The same numbers are tested on both sides in Tasks 1 and 4.

---

### Task 1: Reserves scale with model size

**Files:**
- Modify: `src/weave_agent_signals/run_config.py:47-86` (`JudgingContextPolicy`)
- Modify: `src/weave_agent_signals/judges/windowing.py:285-326` (use the policy's reserve)
- Test: `tests/evaluation/judging/test_windowing.py`, plus a new
  `tests/evaluation/test_judging_context.py`

**Interfaces:**
- Produces: `JudgingContextPolicy.capacity_reserve(model_limit: int) -> int`, the single reserve
  for a model. It is the sum of three parts:
  - `prompt_reserve(model_limit) -> int`: `clamp(round(0.15 * model_limit), 2_000, 6_000)`
  - `generation_budget(model_limit) -> int`: `clamp(round(0.10 * model_limit), 1_000, cap)`,
    where `cap` is 4_000 at or below 200_000 tokens and 10_000 above
  - `safety_reserve(model_limit) -> int`: `clamp(round(0.05 * model_limit), 1_000, 8_000)`
- Removes: `large_model_reserve_tokens`, `small_model_reserve_tokens`, `prompt_reserve_tokens`,
  `output_reserve_tokens`, `large_model_output_reserve_tokens`, `safety_reserve_tokens`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize(
    ("limit", "prompt", "generation", "safety"),
    [(8_192, 2_000, 1_000, 1_000), (32_800, 4_920, 3_280, 1_640),
     (131_072, 6_000, 4_000, 6_554), (1_000_000, 6_000, 10_000, 8_000)],
)
def test_reserve_parts_scale_with_model_size(limit, prompt, generation, safety):
    policy = JudgingContextPolicy()
    assert (policy.prompt_reserve(limit), policy.generation_budget(limit),
            policy.safety_reserve(limit)) == (prompt, generation, safety)
    assert policy.capacity_reserve(limit) == prompt + generation + safety

def test_an_8k_model_gets_a_window_plan():
    # Reuse this file's existing session fixture, shrunk to one turn of about 1,000 tokens,
    # and call build_window_plan exactly as the neighbouring tests do, with an 8,192-token
    # limit. Before this task it raises WindowPlanInapplicable.
    plan = build_window_plan(...)
    assert plan is not None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/evaluation/test_judging_context.py tests/evaluation/judging/test_windowing.py -k "reserve or 8k"`

Expected: FAIL. `prompt_reserve` is undefined, and the 8k plan raises `WindowPlanInapplicable`.

- [ ] **Step 3: Implement the three reserve methods and `capacity_reserve`**

Do this in `run_config.py`. Bump `contract_version` to `"4"`. In `windowing.py`, replace the
`base_reserve` sum (`:291-295`) and the `max(tier_reserve, base_reserve)` line (`:323`) with
`policy.capacity_reserve(model_limit)`.

- [ ] **Step 4: Run the judging tests and fix the fixtures that pinned the old reserves**

Run: `.venv/bin/python -m pytest -q tests/evaluation`

Expected: PASS. Update only the assertions that pinned 18,000-token reserves, and say which ones
in the commit message.

- [ ] **Step 5: Commit**

Message: `feat: scale judging reserves with model size`

### Task 2: Per-model reading budgets

**Files:**
- Modify: `src/weave_agent_signals/run_config.py:92-111` (`ModelDescriptor`)
- Modify: `src/weave_agent_signals/catalogs.py` (`_wandb_model` and every descriptor)
- Modify: `src/weave_agent_signals/judges/windowing.py:315-326` (window size from the model's
  budget)
- Test: `tests/evaluation/test_catalogs.py`, `tests/evaluation/judging/test_windowing.py`

**Interfaces:**
- Produces these `ModelDescriptor` fields, all strict ints of at least 1:
  - `question_budget_tokens: int`: B_q, the evidence bundle for one yes/no question.
  - `window_budget_tokens: int | None`: B_win, the raw rubric window. `None` means plan R is
    unavailable for this model.
  - `whole_budget_tokens: int | None`: B_whole, the largest thread judged in one request. `None`
    means plan W is never used.
  - `merge_budget_tokens: int`: B_merge, the largest merge input.
- Validation on the descriptor:
  - `question_budget_tokens <= merge_budget_tokens`
  - `question_budget_tokens <= window_budget_tokens <= whole_budget_tokens`, when set
  - every budget `<= max_input_tokens - DEFAULT_JUDGING_CONTEXT_POLICY.capacity_reserve(max_input_tokens)`
  - errors name the model id and the field.
- Produces `catalogs.py` budget classes as one mapping, each value
  `(question, window, whole, merge)`:

```python
_BUDGETS = {
    "small_dense": (2_000, None, None, 4_000),        # 8k-class, or dense 8B and below
    "small_moe": (2_000, 8_000, 8_000, 16_000),       # about 32k, or 3B-5B active MoE
    "mid": (4_000, 32_000, 32_000, 32_000),           # dense 27B-70B, 128k-262k windows
    "frontier": (8_000, 128_000, 128_000, 128_000),   # frontier, 200k-1M windows
}
```

- Removes: `ModelDescriptor.raw_window_target_tokens`, and
  `JudgingContextPolicy.large_model_raw_target_tokens` and `small_model_raw_target_tokens`.
  `build_window_plan` uses `window_budget_tokens`, and raises `WindowPlanInapplicable` when it
  is `None`. Plan Q, in piece 6, picks those models up.

- [ ] **Step 1: Write the failing tests**

`test_budgets_out_of_order_are_rejected` checks that a descriptor with question 8,000 and window
4,000 raises a `ValueError` whose message contains its id and `question_budget_tokens`.

`test_budget_larger_than_usable_context_is_rejected` checks that a 32,800-token model with
window 30,000 raises.

`test_changing_a_budget_changes_the_catalog_version` builds the catalog twice with one budget
changed and checks that the versions differ.

`test_window_size_comes_from_the_model_budget` checks that a frontier descriptor's plan never
has a window larger than 128,000 raw tokens.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/evaluation/test_catalogs.py tests/evaluation/judging/test_windowing.py -k budget`

Expected: FAIL, because the fields don't exist yet.

- [ ] **Step 3: Implement the fields, validation and `_BUDGETS`**

Give `_wandb_model` a `budget_class: str` parameter in place of `raw_window_target_tokens`. Pass
a budget class for every descriptor. The Task 3 table fixes the class of each model in the
refreshed catalog.

- [ ] **Step 4: Run the backend suite**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Message: `feat: pin per-model reading budgets in the catalog`

### Task 3: Refresh the catalog and name the CoreWeave route

**Files:**
- Modify: `src/weave_agent_signals/catalogs.py:57-180` (`_WANDB_MODELS`), `:192-347` (other
  descriptors), `:356-400` (preferences)
- Modify: user-facing strings that say "W&B Inference" in `src/`, `frontend/src/`, `README.md`
  and `specs/`. Find them with `grep -rn "W&B Inference"`.
- Test: `tests/evaluation/test_catalogs.py`, `tests/interfaces/routes/test_run_config*.py`
  (whichever covers config validation)

**Interfaces:**
- Consumes: the `budget_class` parameter from Task 2.
- Produces: the `wandb:` descriptor set, which equals the live model list on
  `https://api.inference.wandb.ai/v1/models`.

- [ ] **Step 1: Record the live list and context sizes**

Run `.venv/bin/python` with `httpx.get(INFERENCE_BASE + "/models", headers={"Authorization": f"Bearer {get_api_key()}"})`.
Print each `id` and every field of the response that gives a context length. Use the listed
length where the response has one. Otherwise use the vendor model page cited in
`docs/long-session-judging-research.md`, and put the source in a comment next to the
descriptor.

The 24 ids listed on 2026-11-03:

`MiniMaxAI/MiniMax-M3`, `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.8-27B`,
`deepseek-ai/DeepSeek-V3.1`, `deepseek-ai/DeepSeek-V4-Flash-0731`,
`deepseek-ai/DeepSeek-V4-Pro-0813`, `deepseek-ai/DeepSeek-V4.1-Flash`,
`deepseek-ai/DeepSeek-V4.1-Flash-1017`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it`,
`ibm-granite/granite-4.2-8b`, `meta-llama/Llama-3.1-8B-Instruct`,
`meta-llama/Llama-3.3-70B-Instruct`, `moonshotai/Kimi-K2.6`, `moonshotai/Kimi-K2.7-Code`,
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B`,
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
`reflection/Beam-1.0-501B-A23B`, `zai-org/GLM-5.2`, `zai-org/GLM-5.3`, `zai-org/GLM-5.3-Flash`.

- [ ] **Step 2: Write the failing tests**

`test_wandb_descriptors_match_the_recorded_live_list` checks that the set of `provider_model`
values for `wandb` descriptors equals the 24 ids above.

`test_claude_sonnet_5_has_a_1m_window` checks that `max_input_tokens == 1_000_000`.

`test_retired_codex_models_are_gone` checks that `codex:gpt-5.4` and `codex:gpt-5.5` are absent.

`test_config_naming_a_removed_model_fails` checks that validating a run config whose judge is
`wandb:granite-4.1-8b` fails, with a message containing that id.

`test_recommended_judges_span_three_families` checks that the families of `recommended_judges`
are pairwise distinct.

- [ ] **Step 3: Rewrite `_WANDB_MODELS` from the recorded list**

- **Model ids:** keep today's id style, `wandb:<short name>`.
- **Budget classes:**
  - `small_dense`: Llama 3.1 8B, Granite 4.2 8B.
  - `small_moe`: GPT OSS 20B, Qwen3.6 35B A3B, Nemotron 3.5 Lightning, Gemma 4 26B A4B,
    GLM 5.3 Flash.
  - `mid`: Llama 3.3 70B, Gemma 4 31B, Qwen3.8 27B, GPT OSS 120B, Nemotron 3.5 Super VL.
  - `frontier`: DeepSeek V3.1 and the V4/V4.1 models, GLM 5.2 and 5.3, Kimi K2.6 and K2.7 Code,
    MiniMax M3, Nemotron 3 Ultra, Beam 1.0.
- **Other routes:**
  - Set `claude:claude-sonnet-5` `max_input_tokens=1_000_000`, class `frontier`.
  - Claude Haiku 4.5 and the Codex GPT-5.6 models are `frontier`.
  - Remove `codex:gpt-5.4` and `codex:gpt-5.5`.
- **Judge preferences:** order `_JUDGE_PREFERENCES` so the first three available models are
  from different families:
  1. `claude:claude-sonnet-5`
  2. `codex:gpt-5.6-sol`
  3. `wandb:DeepSeek-V4.1-Flash`
  4. `agy:gemini-3.1-pro-high`
  5. `wandb:GLM-5.3`
  6. `wandb:gpt-oss-120b`
  7. the rest, keeping their current relative order.
- **Route name:** replace the "W&B Inference" strings with "CoreWeave Serverless Inference".

- [ ] **Step 4: Run the backend suite and ruff**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests`

Expected: PASS.

- [ ] **Step 5: Commit**

Message: `feat: refresh the model catalog from CoreWeave Serverless Inference`

### Task 4: Frontend capacity uses the same reserves and budgets

**Files:**
- Modify: `frontend/src/features/runs/modelCapacity.ts:1-94`
- Regenerate: `frontend/src/generated/api.ts`, `frontend/src/generated/openapi.json`
- Test: `frontend/tests/state/modelCapacity.test.ts`

**Interfaces:**
- Consumes: the `ModelDescriptor` budget fields and the policy contract `"4"` from the generated
  types.
- Produces: `capacityReserve(modelLimit: number): number`, with the same formula and clamps as
  Task 1. The existing `estimateModelCapacity` uses it, and uses `window_budget_tokens` in place
  of the removed raw targets.

- [ ] **Step 1: Write the failing test**

```ts
it.each([[8_192, 4_000], [32_800, 9_840], [131_072, 16_554], [1_000_000, 24_000]])(
  'reserve for %i tokens is %i', (limit, reserve) => {
    expect(capacityReserve(limit)).toBe(reserve)
  })
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npm --prefix frontend test -- modelCapacity`

Expected: FAIL, because `capacityReserve` isn't exported.

- [ ] **Step 3: Regenerate the types and implement `capacityReserve`**

Run `npm --prefix frontend run generate:api`, then update `estimateModelCapacity`.

- [ ] **Step 4: Run the frontend checks**

Run: `npm --prefix frontend test && npm --prefix frontend run check:api && npm --prefix frontend run lint && npm --prefix frontend run build`

Expected: all pass.

- [ ] **Step 5: Commit**

Message: `feat: mirror scaled reserves and model budgets in run setup`

### Task 5: Specs and README

**Files:**
- Modify: `specs/02-evaluation.md` (the context policy, `:40-60` and `:224-236`, and the
  provider limits, `:243-245`)
- Modify: `README.md` (the model and provider setup sections)

- [ ] **Step 1: Update the spec**

- Reserves scale with model size, as policy contract 4.
- Each model has pinned reading budgets B_q, B_win, B_whole and B_merge, and experiment 1
  replaces the starting values.
- The open-model route is CoreWeave Serverless Inference, at the same endpoint.

State only behavior and invariants; leave the formulas to the code.

- [ ] **Step 2: Check that nothing still describes the removed fields**

Run: `grep -rn "raw_window_target\|reserve_tokens\|W&B Inference" specs README.md src frontend/src`

Expected: no matches.

- [ ] **Step 3: Commit**

Message: `docs: describe per-model budgets and the CoreWeave route`
