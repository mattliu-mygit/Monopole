# Sliding-Window Judging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace trigger-selected episode judging with exhaustive sliding-window session judging that emits one comparable score and evidence-cited behavioral feedback per rubric and session.

**Architecture:** Pin one versioned context policy and deterministic window plan per run. Each invoked reviewer creates its own cited chunk digests, examines every chunk as raw evidence with one-turn overlap, emits bounded findings, and performs one final rubric merge; durable artifacts make every paid step idempotent and resumable. Existing review escalation pools only final reviewer verdicts, and reflection consumes merged session scores plus bounded low-score feedback examples.

**Tech Stack:** Python 3.11+, Pydantic, SQLite, pytest, FastAPI route serialization, React/TypeScript, Vitest.

## Global Constraints

- Target approximately 100,000 input tokens per digest, window, and merge request.
- Split only at turn boundaries and include one neighboring raw turn on each available side.
- Every non-overlap turn must appear as raw evidence for every reviewer actually invoked.
- Use one session score per rubric; window findings never become standalone Weave scores.
- Keep reviewer digests independent; no reviewer consumes another reviewer's generated digest.
- Preserve primary, selective, and full-panel reviewer ordering and escalation behavior.
- Persist bounded behavioral feedback about agent behavior, never prescriptions for instruction-file edits.
- Fail closed on missing evidence, invalid citations, incomplete raw coverage, over-budget context, invalid artifacts, or failed final merge.
- Buffer Weave feedback writes until complete applicable coverage is known.
- Do not add compatibility adapters or database migrations; the pre-release run database is disposable.
- Use `.venv/bin/python -m pytest` for backend tests and the existing `npm` scripts from `frontend/`.

---

### Task 1: Pin the context policy and model input limits

**Files:**
- Modify: `src/weave_agent_signals/run_config.py`
- Modify: `src/weave_agent_signals/catalogs.py`
- Test: `tests/runs/test_config.py`
- Test: `tests/evaluation/test_catalogs.py`

**Interfaces:**
- Produces: `JudgingContextPolicy`, `DEFAULT_JUDGING_CONTEXT_POLICY`, `ModelDescriptor.max_input_tokens`, and `EffectiveRunConfig.judging_context`.
- Consumes: no new interfaces.

- [ ] **Step 1: Write failing configuration tests**

Add tests asserting the exact immutable policy and version changes:

```python
def test_effective_config_pins_sliding_window_context_policy():
    models, rubrics, request = _catalog_request()
    effective = resolve_run_config(request, model_catalog=models, rubric_catalog=rubrics)
    assert effective.schema_version == "2"
    assert effective.pipeline_version == "4"
    assert effective.judging_context.model_dump(mode="json") == {
        "contract_version": "1",
        "target_input_tokens": 100_000,
        "prompt_reserve_tokens": 6_000,
        "output_reserve_tokens": 4_000,
        "safety_reserve_tokens": 8_000,
        "digest_max_tokens": 1_000,
        "finding_max_tokens": 750,
        "overlap_turns": 1,
        "max_chunks": 40,
        "token_estimator": "utf8_bytes_div_3",
    }


def test_every_model_declares_enough_input_context():
    catalog = build_model_catalog(which=lambda _name: "/usr/bin/model")
    models = [
        *catalog.proposal.available_models,
        *(model for backend in catalog.judge_backends.values() for model in backend.available_models),
    ]
    assert models
    assert all(model.max_input_tokens >= 128_000 for model in models)
```

- [ ] **Step 2: Run the focused tests and confirm the schema mismatch failures**

Run:

```bash
.venv/bin/python -m pytest -q tests/runs/test_config.py tests/evaluation/test_catalogs.py
```

Expected: failures for missing policy/model fields and old version constants.

- [ ] **Step 3: Add the authoritative context policy and update descriptors**

Add a frozen Pydantic model with the exact defaults asserted above:

```python
class JudgingContextPolicy(StrictFrozenModel):
    contract_version: Literal["1"] = "1"
    target_input_tokens: Annotated[int, Field(strict=True, ge=1)] = 100_000
    prompt_reserve_tokens: Annotated[int, Field(strict=True, ge=1)] = 6_000
    output_reserve_tokens: Annotated[int, Field(strict=True, ge=1)] = 4_000
    safety_reserve_tokens: Annotated[int, Field(strict=True, ge=1)] = 8_000
    digest_max_tokens: Annotated[int, Field(strict=True, ge=1)] = 1_000
    finding_max_tokens: Annotated[int, Field(strict=True, ge=1)] = 750
    overlap_turns: Literal[1] = 1
    max_chunks: Annotated[int, Field(strict=True, ge=1)] = 40
    token_estimator: Literal["utf8_bytes_div_3"] = "utf8_bytes_div_3"


DEFAULT_JUDGING_CONTEXT_POLICY = JudgingContextPolicy()
```

Add a model validator that rejects policies whose prompt, output, safety, digest, and finding reserves leave no raw capacity under `target_input_tokens`. Add `max_input_tokens: int = 128_000` to `ModelDescriptor`, set every current descriptor explicitly to `128_000`, add `judging_context` to `EffectiveRunConfig`, and have `resolve_run_config` pin the default. Bump `PIPELINE_VERSION` to `"4"`, `MODEL_CATALOG_SCHEMA_VERSION` to `"2"`, and the effective-config schema literal to `"2"`. Leave rubric scope unchanged until Task 6 so this commit remains compatible with the current runner.

- [ ] **Step 4: Run the focused tests**

Run the Task 1 command again.

Expected: all selected tests pass.

- [ ] **Step 5: Commit the pinned policy boundary**

```bash
git add src/weave_agent_signals/run_config.py src/weave_agent_signals/catalogs.py tests/runs/test_config.py tests/evaluation/test_catalogs.py
git commit -m "feat: pin sliding judge context policy"
```

### Task 2: Build deterministic raw rendering, chunking, overlap, and budget validation

**Files:**
- Create: `src/weave_agent_signals/judges/windowing.py`
- Create: `tests/evaluation/judging/test_windowing.py`
- Modify: `src/weave_agent_signals/judges/digest.py`
- Modify: `tests/evaluation/judging/test_digest.py`

**Interfaces:**
- Consumes: `JudgingContextPolicy`, `SessionView`, and hydrated `TurnSpan` values.
- Produces: `estimate_tokens(text: str) -> int`, `render_raw_turn(turn: TurnSpan, position: int) -> str`, `build_window_plan(session: SessionView, policy: JudgingContextPolicy, model_limit: int) -> dict[str, object]`, and `render_raw_window(session: SessionView, window: Mapping[str, object]) -> JudgeDigest`.

- [ ] **Step 1: Write failing planner tests**

Cover stable ordering, no model identity leakage, every turn's raw coverage, exact one-turn overlap, oversized single-turn rejection, maximum-chunk rejection, and deterministic plan IDs:

```python
def test_window_plan_covers_every_turn_raw_and_overlaps_one_turn():
    session = _session_with_rendered_turn_sizes([18_000, 18_000, 18_000, 18_000])
    plan = build_window_plan(session, _small_policy(), model_limit=60_000)
    cores = [window["core_trace_ids"] for window in plan["windows"]]
    assert [trace for core in cores for trace in core] == ["t1", "t2", "t3", "t4"]
    assert plan["windows"][0]["raw_trace_ids"][-1] == "t3"
    assert plan["windows"][1]["raw_trace_ids"][0] == "t2"
    assert plan["raw_coverage_trace_ids"] == ["t1", "t2", "t3", "t4"]


def test_window_plan_rejects_a_turn_that_cannot_fit_with_reserves():
    session = _session_with_text("x" * 400_000)
    with pytest.raises(ValueError, match="single turn exceeds the raw window budget"):
        build_window_plan(session, _small_policy(), model_limit=60_000)
```

Define those three local helpers in the same test module: `_session_with_rendered_turn_sizes` builds hydrated turns whose rendered byte sizes are deterministic, `_session_with_text` builds one hydrated turn, and `_small_policy` returns a valid reduced-reserve `JudgingContextPolicy` that forces two chunks in the first test.

- [ ] **Step 2: Run the new test module and verify import failures**

```bash
.venv/bin/python -m pytest -q tests/evaluation/judging/test_windowing.py
```

Expected: collection fails because `windowing.py` does not exist.

- [ ] **Step 3: Implement the renderer and fixed-point chunk planner**

Use the versioned conservative estimator:

```python
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text.encode("utf-8")) + 2) // 3
```

Render complete captured user, assistant, tool, status, event, and subagent evidence with trace/span evidence IDs, while omitting the generating model identity. Build chunks greedily at turn boundaries. Recompute the raw-window budget as:

```python
raw_budget = min(policy.target_input_tokens, model_limit) - (
    policy.prompt_reserve_tokens
    + policy.output_reserve_tokens
    + policy.safety_reserve_tokens
    + chunk_count * policy.digest_max_tokens
)
```

Repeat partitioning until the chunk count and raw budget stabilize. While choosing each core boundary, budget the candidate core together with its available previous neighbor and prospective next neighbor so overlap cannot make an otherwise accepted window overflow. Materialize exactly one adjacent turn on each side, then validate every final raw window against the same budget. Also fail closed unless the worst-case merge input—prompt, reserves, every bounded digest, and every bounded finding bundle—fits the same input cap. Reject non-convergence, empty raw capacity, more than `max_chunks`, or any turn/window/merge that cannot fit. Hash the complete JSON plan body for `plan_id`.

- [ ] **Step 4: Remove obsolete bounded session rendering**

Keep `JudgeDigest`, but remove `build_session_digest`, `_selected_turn_indexes`, `_context_turn_indexes`, and the 400-character session summary path. Update digest tests to cover only raw turn/window rendering and citation visibility.

- [ ] **Step 5: Run planner and digest tests**

```bash
.venv/bin/python -m pytest -q tests/evaluation/judging/test_windowing.py tests/evaluation/judging/test_digest.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit deterministic window construction**

```bash
git add src/weave_agent_signals/judges/windowing.py src/weave_agent_signals/judges/digest.py tests/evaluation/judging/test_windowing.py tests/evaluation/judging/test_digest.py
git commit -m "feat: plan bounded raw judging windows"
```

### Task 3: Define strict digest, finding, merge, and behavioral-feedback contracts

**Files:**
- Create: `src/weave_agent_signals/judges/sliding_contracts.py`
- Create: `tests/evaluation/judging/test_sliding_contracts.py`
- Modify: `src/weave_agent_signals/judges/verdicts.py`
- Modify: `tests/evaluation/judging/test_verdicts.py`

**Interfaces:**
- Produces: `ChunkDigest`, `WindowFinding`, `WindowFindings`, `BehavioralFeedback`, `MergedVerdict`, their `JsonSchemaSpec` values, and strict parse functions.
- Consumes: allowed trace/span IDs and the configured text/token bounds.

- [ ] **Step 1: Write failing closed-contract tests**

Test successful parsing and rejection of unknown IDs, blank citations, duplicate findings, extra fields, over-limit text, invalid polarity, invalid score anchors, and feedback that prescribes file edits:

```python
def test_merged_verdict_keeps_behavioral_feedback_grounded():
    parsed = parse_merged_verdict(
        {
            "schema_version": 1,
            "status": "scored",
            "score": 0.5,
            "rationale": "The agent recovered but did not rerun the final check.",
            "evidence_ids": ["trace-2", "trace-3"],
            "feedback": {
                "success": "It changed approach after the failure.",
                "problem": "It claimed completion without rerunning the failed check.",
                "desired_behavior": "Rerun relevant checks after the final change.",
            },
        },
        allowed_evidence_ids=("trace-1", "trace-2", "trace-3"),
    )
    assert parsed.feedback.desired_behavior == "Rerun relevant checks after the final change."


def test_feedback_must_not_prescribe_managed_file_edits():
    payload = _valid_merged_payload()
    payload["feedback"]["desired_behavior"] = "Add a rule to AGENTS.md."
    with pytest.raises(ValueError, match="behavior rather than instruction edits"):
        parse_merged_verdict(payload, allowed_evidence_ids=("trace-1",))
```

Define `_valid_merged_payload` directly above the tests as the smallest valid scored payload, using `trace-1` as its only evidence ID.

- [ ] **Step 2: Run the contract tests and confirm missing imports**

```bash
.venv/bin/python -m pytest -q tests/evaluation/judging/test_sliding_contracts.py tests/evaluation/judging/test_verdicts.py
```

Expected: collection or assertion failures for the missing contract types.

- [ ] **Step 3: Implement the Pydantic contracts and JSON schemas**

Use strict, extra-forbid models. `WindowFinding` contains `finding_id`, `polarity` (`positive` or `negative`), `observation`, and unique `evidence_ids`. `WindowFindings` contains the window ID and at most four findings, bounded by the policy's 750-token output allowance. `BehavioralFeedback` contains nullable `success`, `problem`, and `desired_behavior`, with at least one non-null value. `MergedVerdict` preserves scored versus insufficient-evidence semantics and the five existing anchors.

Bound each finding observation to 350 normalized characters and each feedback field to 500 normalized characters. Reject desired-behavior text containing managed-file edit language such as `AGENTS.md`, `CLAUDE.md`, `SKILL.md`, "instruction file", "prompt file", or imperative create/update/delete language directed at a file.

- [ ] **Step 4: Run contract tests**

Run the Task 3 command again.

Expected: all selected tests pass.

- [ ] **Step 5: Commit the closed sliding contracts**

```bash
git add src/weave_agent_signals/judges/sliding_contracts.py src/weave_agent_signals/judges/verdicts.py tests/evaluation/judging/test_sliding_contracts.py tests/evaluation/judging/test_verdicts.py
git commit -m "feat: define sliding judge evidence contracts"
```

### Task 4: Persist immutable digest and finding artifacts for safe resume

**Files:**
- Modify: `src/weave_agent_signals/runs/store.py`
- Modify: `src/weave_agent_signals/routes/__init__.py`
- Test: `tests/runs/test_store.py`
- Test: `tests/interfaces/routes/test_runs.py`

**Interfaces:**
- Produces: `Run.judging_artifacts`, `RunStore.record_judging_artifact(run_id: str, artifact_id: str, artifact: Mapping[str, Any]) -> Run`, and route field `judging_artifacts`.
- Consumes: content-addressed digest/window/merge artifacts from Task 5.

- [ ] **Step 1: Write failing idempotency and validation tests**

```python
def test_judging_artifacts_are_content_addressed_and_idempotent(store):
    run = _started_judging_run(store)
    payload = {"chunk_id": "chunk-1", "text": "cited digest"}
    artifact = {
        "schema_version": "1",
        "kind": "chunk_digest",
        "content_digest": _payload_digest(payload),
        "payload": payload,
    }
    first = store.record_judging_artifact(run.run_id, "judge-1/session-1/digest/chunk-1", artifact)
    second = store.record_judging_artifact(run.run_id, "judge-1/session-1/digest/chunk-1", artifact)
    assert first.judging_artifacts == second.judging_artifacts

    changed_payload = {"chunk_id": "chunk-1", "text": "changed"}
    changed = {
        **artifact,
        "content_digest": _payload_digest(changed_payload),
        "payload": changed_payload,
    }
    with pytest.raises(RunStoreConflictError, match="artifact already exists with different content"):
        store.record_judging_artifact(run.run_id, "judge-1/session-1/digest/chunk-1", changed)
```

Use the existing run-start fixture pattern to define `_started_judging_run`. Define `_payload_digest` in the test module with the production canonical JSON separators and sorted keys so the asserted artifact is internally valid.

- [ ] **Step 2: Run store tests and confirm the missing-field failures**

```bash
.venv/bin/python -m pytest -q tests/runs/test_store.py tests/interfaces/routes/test_runs.py
```

Expected: failures because `Run` and the database do not expose judging artifacts.

- [ ] **Step 3: Add disposable-schema storage and strict encoding**

Bump `RUN_DB_SCHEMA_VERSION` to `5`, add a nullable `judging_artifacts` JSON column directly to `_SCHEMA`, and add the corresponding `Run` field. Do not add ALTER statements or migrations.

Validate artifact IDs as nonblank slash-delimited strings and artifact bodies as exactly `schema_version`, `kind`, `content_digest`, and `payload`. Recompute the SHA-256 digest of canonical `payload` JSON and reject mismatches. Within one immediate SQLite transaction, insert a missing artifact, accept an exact replay, and reject different content under an existing ID. Allow writes only while the run is actively judging.

- [ ] **Step 4: Expose artifacts in run detail and verify tests**

Return `judging_artifacts` from the run-detail route; run the Task 4 tests again.

Expected: all selected tests pass.

- [ ] **Step 5: Commit resumable artifact persistence**

```bash
git add src/weave_agent_signals/runs/store.py src/weave_agent_signals/routes/__init__.py tests/runs/test_store.py tests/interfaces/routes/test_runs.py
git commit -m "feat: persist sliding judge artifacts"
```

### Task 5: Execute one reviewer's digest-window-merge pipeline

**Files:**
- Create: `src/weave_agent_signals/judges/sliding.py`
- Create: `tests/evaluation/judging/test_sliding.py`
- Modify: `src/weave_agent_signals/judges/review.py`
- Modify: `tests/evaluation/judging/test_review_policy.py`

**Interfaces:**
- Consumes: window plans, contracts, `ChatClient`, `PositionedJudge`, and artifact load/record callbacks.
- Produces: `InferenceStepAudit`, extended `AttemptObservation.behavioral_feedback` and `.steps`, and `SlidingReviewer.review(rubric: RubricDescriptor) -> AttemptObservation`.

- [ ] **Step 1: Write failing orchestration tests**

Use a scripted client to assert call order and reuse:

```python
def test_reviewer_digests_once_then_reads_every_window_and_merges():
    reviewer = _reviewer_with_two_chunks()
    first = reviewer.review(_rubric("judge.verification"))
    second = reviewer.review(_rubric("judge.session_outcome"))

    assert [call["phase"] for call in reviewer.client.calls] == [
        "digest", "digest",
        "window", "window", "merge",
        "window", "window", "merge",
    ]
    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert first.evidence_ids == ("trace-1", "trace-2")
    assert first.behavioral_feedback is not None
```

Define `_reviewer_with_two_chunks` in the same module from a two-turn synthetic `SessionView`, a deterministic two-window plan, an in-memory artifact map, and the existing scripted `ChatClient` test double. Define `_rubric` by selecting the requested ID from `build_rubric_catalog()`.

Also test artifact replay makes zero repeated calls, each window replaces its own digest with raw evidence, chronological digest order, deduplication input, aggregated usage, cancellation between calls, malformed artifact rejection, over-budget prompt rejection, and a failed merge producing a failed observation rather than a score.

- [ ] **Step 2: Run orchestration tests and confirm missing imports**

```bash
.venv/bin/python -m pytest -q tests/evaluation/judging/test_sliding.py tests/evaluation/judging/test_review_policy.py
```

Expected: collection fails because `sliding.py` and the extended observation fields do not exist.

- [ ] **Step 3: Implement prompt construction and artifact replay**

`SlidingReviewer` owns one session, reviewer, window plan, client, cancellation callback, and artifact callbacks. Generate rubric-neutral chunk digests lazily once per reviewer. Protect the per-reviewer digest cache with one lock/future so concurrent rubric work cannot issue duplicate digest calls. For each rubric, call every window in order and persist parsed `WindowFindings`. Then call merge with the rubric criteria, ordered findings, ordered digest context, complete coverage manifest, and allowed evidence IDs.

Every call uses structured output, temperature zero, and the phase-appropriate token maximum. Validate the rendered request with `estimate_tokens` before transport. Aggregate portable usage counters across all calls and retain one `InferenceStepAudit` per call with phase, artifact ID, requested/resolved model, usage, output mode, schema, fallback reason, transport count, and raw-output digest.

- [ ] **Step 4: Extend pure review observations without changing escalation rules**

Add immutable optional `behavioral_feedback: Mapping[str, str | None]` and `steps: tuple[InferenceStepAudit, ...]` fields to `AttemptObservation`. Successful observations require feedback; abstained observations require no feedback; failed observations contain neither feedback nor evidence. Leave `execute_review` decision logic unchanged.

- [ ] **Step 5: Run orchestration and review-policy tests**

Run the Task 5 command again.

Expected: all selected tests pass.

- [ ] **Step 6: Commit the reviewer pipeline**

```bash
git add src/weave_agent_signals/judges/sliding.py src/weave_agent_signals/judges/review.py tests/evaluation/judging/test_sliding.py tests/evaluation/judging/test_review_policy.py
git commit -m "feat: execute sliding reviewer pipelines"
```

### Task 6: Replace episode judging in the runner, durable stage, and direct CLI

**Files:**
- Rewrite: `src/weave_agent_signals/judges/plan.py`
- Modify: `src/weave_agent_signals/judges/runner.py`
- Modify: `src/weave_agent_signals/runs/stages/judging.py`
- Modify: `src/weave_agent_signals/cli.py`
- Modify: `src/weave_agent_signals/judges/__init__.py`
- Test: `tests/evaluation/judging/test_plan.py`
- Test: `tests/evaluation/judging/test_session_judge.py`
- Test: `tests/runs/stages/test_judging.py`
- Test: `tests/interfaces/test_cli_judge.py`

**Interfaces:**
- Consumes: `build_window_plan`, `SlidingReviewer`, artifact persistence, and the unchanged `ReviewPolicy`.
- Produces: `build_judging_plan(..., context_policy, judge_models)`, session-only `judge_session(..., session_plan, artifact_loader, artifact_recorder, cancel_requested)`, and one `Score` per rubric/session.
- Removes: `judge_turn`, selected episodes, applicability heuristics, episode caps, and turn-target judge writes.
- Produces: all six rubric descriptors with `evaluation_unit == "session"`, version `v4`, and one authoritative `SESSION_RUBRICS` mapping.

- [ ] **Step 1: Rewrite plan tests around sessions and windows**

Assert schema version `2`, exact chunk/window manifests, six requested session rubrics, all-turn coverage, reviewer work bounds, and content-derived plan stability:

```python
def test_plan_counts_one_final_review_per_rubric_and_session():
    plan = build_judging_plan(
        [_session_with_turns(4)],
        cohort_id="cohort-1",
        rubrics=build_rubric_catalog().rubrics,
        review_depth="selective",
        judge_models=(_judge("judge-1", 1), _judge("judge-2", 2)),
        context_policy=DEFAULT_JUDGING_CONTEXT_POLICY,
    )
    assert plan["schema_version"] == "2"
    assert plan["totals"]["planned_rubrics"] == 6
    assert plan["totals"]["sessions_planned"] == 1
    assert plan["sessions"][0]["raw_coverage_trace_ids"] == ["turn-1", "turn-2", "turn-3", "turn-4"]
```

Define `_session_with_turns` with the existing synthetic span builders and `_judge` as a `PositionedJudge` factory in the same test module.

- [ ] **Step 2: Update runner tests for merged feedback and lazy selective reviewers**

Assert `judge_session` creates one shared `SlidingReviewer` per judge/session, invokes later reviewers only when `execute_review` requests them, stores reviewer feedback in attempt metadata, writes `granularity="session"`, and composes a bounded `Score.reason` from successful reviewer behavioral feedback.

- [ ] **Step 3: Run the focused integration tests and confirm old-shape failures**

```bash
.venv/bin/python -m pytest -q tests/evaluation/judging/test_plan.py tests/evaluation/judging/test_session_judge.py tests/runs/stages/test_judging.py tests/interfaces/test_cli_judge.py
```

Expected: failures referencing old episode plans and the old runner signature.

- [ ] **Step 4: Rewrite the plan and runner**

Build only session plans. Each session record contains ordered chunk/window records, raw coverage, requested rubrics, input-policy manifest, and reviewer attempt bounds. In the runner, resolve every descriptor against `SESSION_RUBRICS`, build one lazy reviewer cache keyed by positioned judge ID, and call `execute_review` once per rubric.

Change every model rubric to `evaluation_unit="session"`, bump their shared version to `v4`, consolidate them in `SESSION_RUBRICS`, and remove the old episode/cross-turn mappings in the same commit that removes their callers. Store final attempt audits, behavioral feedback, coverage, plan ID, rubric context, and reviewer context in `Score.metadata`. Bump `REVIEW_POLICY_VERSION` to `"3"` because one reviewer attempt now represents a multi-call pipeline.

- [ ] **Step 5: Rewrite the durable stage and CLI**

The durable stage iterates sessions only, supplies store-backed artifact callbacks, updates monotonic digest/window/merge counters, buffers every score, and writes only session refs after complete coverage. The direct CLI uses the same plan and runner with an in-memory artifact map, then reports sessions, windows, rubric scores, and writes.

Delete all episode selection, turn judge calls, selected-episode metadata, and now-unused imports.

- [ ] **Step 6: Run the focused integration tests**

Run the Task 6 command again.

Expected: all selected tests pass.

- [ ] **Step 7: Commit session-only judging integration**

```bash
git add src/weave_agent_signals/judges/plan.py src/weave_agent_signals/judges/runner.py src/weave_agent_signals/runs/stages/judging.py src/weave_agent_signals/cli.py src/weave_agent_signals/judges/__init__.py tests/evaluation/judging/test_plan.py tests/evaluation/judging/test_session_judge.py tests/runs/stages/test_judging.py tests/interfaces/test_cli_judge.py
git commit -m "feat: replace episode judging with sliding sessions"
```

### Task 7: Feed merged behavioral feedback into reflection coaching

**Files:**
- Modify: `src/weave_agent_signals/patterns.py`
- Modify: `tests/analysis/test_patterns.py`
- Modify: `tests/reflection/test_stage.py`

**Interfaces:**
- Consumes: session feedback `payload.reason`, `details.behavioral_feedback`, complete review status, rubric context, and evidence IDs.
- Produces: coaching sections containing at most three lowest-scoring feedback examples per rubric.

- [ ] **Step 1: Write failing representative-feedback tests**

```python
def test_coaching_digest_includes_bounded_low_score_behavioral_feedback():
    feedback = [
        _session_feedback(
            scorer="judge.verification",
            rating=0.25,
            conversation_id="session-1",
            behavioral_feedback={
                "success": None,
                "problem": "Completion was claimed before the final check.",
                "desired_behavior": "Run relevant checks after the final change.",
            },
            evidence_ids=["trace-7"],
        )
    ]
    text = coaching_digest(feedback)
    assert "## Behavioral feedback" in text
    assert "Completion was claimed before the final check." in text
    assert "Run relevant checks after the final change." in text
    assert "session-1" in text
    assert "trace-7" in text
```

Define `_session_feedback` in the test module by extending the existing feedback factory with complete-review metadata, `granularity="session"`, behavioral feedback, and evidence IDs.

Also assert incomplete/degraded/unresolved reviews are excluded, examples are ordered by rating then execution time, only three appear per rubric, and raw conversation text is absent.

- [ ] **Step 2: Run analysis/reflection tests and verify missing output**

```bash
.venv/bin/python -m pytest -q tests/analysis/test_patterns.py tests/reflection/test_stage.py
```

Expected: coaching assertions fail because reasons are not currently surfaced.

- [ ] **Step 3: Simplify representative analysis and add feedback examples**

Remove selected-episode diagnostic aggregation because new judge feedback is session-only. Keep deterministic score aggregation unchanged. From complete, comparable session judgments, select up to three low-scoring examples per rubric, normalize and bound each displayed field, and render session/evidence identities plus problem and desired behavior. Do not include raw window findings or raw conversation content.

- [ ] **Step 4: Run analysis/reflection tests**

Run the Task 7 command again.

Expected: all selected tests pass.

- [ ] **Step 5: Commit reflection coaching feedback**

```bash
git add src/weave_agent_signals/patterns.py tests/analysis/test_patterns.py tests/reflection/test_stage.py
git commit -m "feat: coach reflection from merged judge feedback"
```

### Task 8: Update run-detail types and progress UI for chunks, windows, and merge audits

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/features/runs/JudgingProgress.tsx`
- Modify: `frontend/tests/runs/JudgingProgress.test.tsx`
- Modify: `frontend/tests/pages/RunDetail.test.tsx`
- Modify: `frontend/tests/runs/runPolling.test.ts`

**Interfaces:**
- Consumes: plan schema `2`, digest/window/merge progress counters, session-only review summaries, and per-attempt step audits.
- Produces: accurate plan summary and expandable reviewer pipeline audit.

- [ ] **Step 1: Replace old fixtures with the new plan and progress shape**

Use a plan containing `sessions_planned`, `chunks_planned`, `windows_planned`, `planned_rubrics`, reviewer bounds, and per-session windows. Add progress counters `digests_completed`, `windows_completed`, and `merges_completed`. Add attempt `steps` and `behavioral_feedback` fixtures.

- [ ] **Step 2: Write failing UI assertions**

```tsx
expect(screen.getByText('2 chunks · 2 raw windows · 6 rubric merges')).toBeInTheDocument()
expect(screen.getByText('4 digests completed')).toBeInTheDocument()
expect(screen.getByText('Run relevant checks after the final change.')).toBeInTheDocument()
expect(screen.queryByText(/selected episodes/i)).not.toBeInTheDocument()
```

- [ ] **Step 3: Run focused frontend tests and confirm old-copy failures**

```bash
cd frontend && npm test -- --run tests/runs/JudgingProgress.test.tsx tests/pages/RunDetail.test.tsx tests/runs/runPolling.test.ts
```

Expected: type or assertion failures for the new plan and audit fields.

- [ ] **Step 4: Update TypeScript contracts and the progress component**

Remove episode plan types. Add chunk/window plan types, new totals, progress counters, `InferenceStepAudit`, and behavioral feedback. Render session/chunk/window totals, the three progress phases, final reviewer score, bounded feedback, and an expandable ordered transport audit. Keep failure and cancellation presentation intact.

- [ ] **Step 5: Run focused frontend tests**

Run the Task 8 command again.

Expected: all selected tests pass.

- [ ] **Step 6: Commit the updated run audit UI**

```bash
git add frontend/src/types.ts frontend/src/features/runs/JudgingProgress.tsx frontend/tests/runs/JudgingProgress.test.tsx frontend/tests/pages/RunDetail.test.tsx frontend/tests/runs/runPolling.test.ts
git commit -m "feat: show sliding judge progress"
```

### Task 9: Reconcile canonical specifications, remove temporary design artifacts, and verify all gates

**Files:**
- Modify: `specs/02-evaluation.md`
- Modify: `specs/03-analysis-monitoring.md`
- Modify: `specs/04-evaluation-runs.md`
- Delete: `docs/superpowers/specs/2026-07-15-sliding-window-judging-design.md`
- Delete: `docs/superpowers/plans/2026-07-15-sliding-window-judging.md`

**Interfaces:**
- Consumes: final implementation behavior.
- Produces: one non-duplicated canonical description of current architecture and a clean repository.

- [ ] **Step 1: Update canonical product behavior**

Replace selective-episode language with the final sliding-window behavior: pinned context policy, exhaustive raw chunk coverage, reviewer-specific digests, final merges, session-only rubric scores, behavioral feedback, complete-review comparability, reflection examples, artifact resume, and fail-closed writes. Keep specifications focused on intent, behavior, boundaries, invariants, and tradeoffs rather than code inventories or schemas.

- [ ] **Step 2: Remove superseded temporary documents and dead terminology**

Delete this plan and its design document after their lasting decisions are represented in canonical specs. Search production code, tests, frontend, and specs:

```bash
rg -n "selected_episodes|episodes_selected|planned_episode_rubrics|max_episodes_per_session|judge_turn|selection_kind.*deterministic_trigger" src tests frontend specs
```

Expected: no matches except an intentionally retained historical fixture; remove any such fixture instead of adding compatibility handling.

- [ ] **Step 3: Run focused backend quality gates**

```bash
.venv/bin/python -m pytest -q tests/evaluation/judging tests/runs/stages/test_judging.py tests/runs/test_store.py tests/analysis/test_patterns.py tests/reflection/test_stage.py tests/interfaces/test_cli_judge.py
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
```

Expected: all commands exit zero.

- [ ] **Step 4: Run the complete backend suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Run all frontend gates**

```bash
cd frontend && npm test
cd frontend && npm run lint
cd frontend && npm run build
```

Expected: all commands exit zero.

- [ ] **Step 6: Review the final diff for contract drift and dead code**

```bash
git diff --check
git status --short
git diff --stat
```

Confirm that only intended implementation, test, UI, and canonical-spec changes remain; no episode-judging path, compatibility layer, temporary document, generated build output, or unrelated user change is included.

- [ ] **Step 7: Commit final documentation and cleanup**

```bash
git add -A
git commit -m "docs: document sliding window evaluation"
```

