# Future phases (M2+): settled decisions, detailed at implementation time

Parking lot for research and decisions already made. Each section becomes a full spec when its milestone starts. M1 specs (01–06) are the current build surface.

**Build-vs-reuse verdict (3-agent research sweep, 2026-07-09):** all-in-Weave stands — every competing platform requires dual ingestion and strands scores outside Weave; their coach/insight agents are hard-wired to frontier models. Genuinely unbuilt anywhere: multi-family PoLL panel, deterministic test/build-outcome scoring, config_version cohort A/B, in-Weave ask-your-traces. What already exists is folded into the sections below as "reuse" notes.

---

## Routing gates (M2)

**Decide at M2 with real cost data**: at single-user volume (~100–200 turns/day, small judges ≈ $0.10–0.20/day) gates save pennies; their honest value is noise reduction, not cost. Default lean: skip gates in v1 — judge everything with small models, sample the panel — and add gates only if cost or score noise actually hurts.

Deterministic predicates deciding which turns/sessions get LLM judging and at what tier (`skip` / `small` / `panel`). Core idea: don't spend judge calls on turns already characterized by deterministic scores — clear test pass with no friction needs no judge; deterministic failure or high frustration warrants one. Gate on the already-stamped attrs (`steering_count`, `denial_count`, `tool_error_count`) + L1 scores; always panel the first N turns of a new `config_version` (baseline seeding); hash-based sampling for the rest. Thresholds are unknowable until we have score distributions — set them from backfill data, not guesses. Gate decisions persist as feedback for auditability.

## Judge fleet (M2–M3)

All judges on **W&B Inference** (OpenAI-compatible, W&B API key, no external keys).

- **Family-disjoint selection**: agent is Claude ⇒ non-Anthropic judges only (self-preference bias is training-distribution-driven, arXiv:2410.21819; `gpt-oss` counts as OpenAI-family). Per-unit: model → family map → exclude judges in the unit's family.
- **Tiers**: small (`gpt-oss-20b`, `Llama-3.1-8B`, `granite-4.1-8b`) for turn-level process rubrics; PoLL panel of 3 disjoint families (`gpt-oss-120b` / `DeepSeek-V4` / `Qwen3-30B` or `Llama-3.3-70B`) for sessions — mean-pool scores, max-pool flags, 7–8x cheaper than one frontier judge (arXiv:2404.18796); escalation tier on low panel agreement (Trust-or-Escalate, arXiv:2407.18370), flag for human review if still divergent.
- **Rubrics**: decomposed multi-dim (cuts self-preference ~31.5%, arXiv:2604.22891) with length controls. Process rubrics (verification discipline, error recovery, tool choice, planning, context management) and outcome rubrics (task completion, code quality, communication). Position-swap applies **only to pairwise comparisons** (e.g. judging one session against another across configs) — it's meaningless for absolute rubric scoring of a single digest; don't cargo-cult it there.
- **Write-back**: `wandb.agent_monitor` feedback (requires Signals-native `runnable_ref`/`call_ref`/`trigger_ref` from Signals registration).
- **Weave-native reality (verified in source 2026-07-09)**: turn-level judging infrastructure already ships — 13 preset classifier signals + 8 agent-signal templates (incl. User Frustration, Satisfaction), custom prompts, 0–1 sampling, W&B Inference judge picker (default `gpt-oss-20b`), and rationale + confidence persisted in `wandb.agent_monitor`. Custom-attr filters (`custom_attrs_string.*`) work via SDK-published `Monitor` objects even though the UI filter editor doesn't expose them. **So M2 = signal/rubric definitions published as code, not judging infrastructure.** Session scope does NOT exist: scoring fires only on `turn_ended`; `conversation_ended` is a commented-out TODO in W&B's worker — build M3 externally but design it to migrate onto the native trigger when it ships.
- **Ops patterns to adopt**: idle-timeout session-close trigger (LangSmith multi-turn evals / MLflow 5-min buffer — no explicit end event needed); "rewind" re-scoring from a timestamp after a scorer fix (Braintrust) — spec 05's `scorer_version` field is the hook.
- **Judge calibration**: align judges against human labels from Weave's annotation queues (already shipped) — the pattern LangSmith Align Evals / MLflow `align()` use; a natural GEPA extension (align the judges, not just the config).
- Reuse: `agentevals` trajectory judge, RAGAS `AgentGoalAccuracyWithoutReference`, `openevals` primitives, `phoenix.evals` + DeepEval agentic/conversational metrics as embeddable libraries, Verdict (MIT) for panel aggregation primitives.
- **OPEN**: exact W&B Inference base URL + model catalog.

## Digest builder (M3)

Raw whole-trace judging catches ~11–22% of issues (TRAIL arXiv:2505.08638, arXiv:2606.10315) — so sessions are judged via a structured digest, not the raw trace. Extract ~10–20 chronological key moments (outcomes, errors, steering/denials, key decisions, completion) + L1 scores + session stats into ~2K tokens for the panel. Idempotent + cached. The cheap-summarizer → strong-judge split is independently validated by LangSmith's Insights Agent architecture. Use TRAIL's failure taxonomy (reasoning / system-execution / planning-coordination, 20+ modes) as the rubric + clustering ontology rather than inventing one.

**Known risk — the digest is load-bearing and lossy**: whatever the extraction rules miss, the panel never sees, and it will score the digest with confident garbage. Mitigations: extraction rules are anchored on L1 facts (anomalies always included, not editorially chosen); validate digests against ~10 hand-read sessions before trusting panel scores (M3 exit criterion); keep the builder simple and auditable rather than clever.

## Pattern engine (M4)

Feedback + span queries → pandas. The analyses:

- **Emergent failure clustering** (soul-stealer pattern): don't impose a taxonomy — cluster negative-signal turns by features (tool sequences, error types, correction types) and let recurring shapes name themselves ("Bash retry spiral on pytest", "edit-without-read"). Growing clusters = RSI candidates.
- **Negative-space extraction**: for high-frustration turns, capture prompt-response correction records (what the agent did, how the user corrected it) — "corrected 12× for not testing before done" is actionable; a frustration average isn't. Full local transcripts are the source here (richer than Weave's redacted/capped spans).
- **A/B leaderboards by `config_version`**: the RSI measurement mechanism. Min-N per cohort, confidence intervals. **Be honest about what this is: quasi-experimental, not an RCT.** Task mix confounds everything (research vs implementation vs debugging sessions differ more than any config change), and single-user volume (~tens of sessions/week) means small effects are undetectable. Mitigations: stratify comparisons by session type (from L2 tags or task classification) rather than comparing raw cohort means; trust A/B only for large effects; treat it as directional evidence that the coach agent pairs with qualitative correction records, not as proof.
- **`config_version` churn — FIXED (adapter commit `6d6a334`, deployed 2026-07-09)**: the surface originally included the auto-memory directory, which changes nearly every session, flipping the cohort key constantly. Auto-memory is now excluded; `config_version` covers only deliberately-edited artifacts (CLAUDE.md, skills, commands, agents). A separate `memory_version` attr can be added later if memory-effects ever need measuring.
- Score trends, prompt-quality ↔ efficiency correlations, cost join via `ccusage --json` (**OPEN**: join key).
- **Documented limitations (accepted for v1)**: sessions ≠ tasks — one `conversation_id` can span several distinct tasks, blurring session-level scores (task segmentation deferred; stratification partially compensates). Frustration measures *the user*, whose standards drift upward over time — within-cohort comparisons are valid, long-horizon frustration trends are not.
- Output: materialized views the coach agent can query (see stretch goal) + EvaluationLogger rollups; scheduled refresh daily/weekly.

## RSI loop (M4)

Engine: **GEPA** `optimize_anything` (arXiv:2507.19457; `gepa.gskill` as reference recipe; Weave-native tracing). Judge rationales = textual feedback; L1+L3 scores = fitness.

- **Action space**: create / edit / **delete** over CLAUDE.md sections, skills, slash commands, prompt templates, memory entries. Baseline: no global CLAUDE.md/skills exist yet, so early iterations mostly create — clean A/B baseline. Delete fires when a cohort measures worse or an artifact goes unused.
- **Proposal validation** (SkillOpt pattern): weak-model pre-screen (syntax/contradiction/length/vagueness, ~$0.001) → held-in test (do the triggering failure cases improve?) → held-out test (do good sessions regress?). Validation evidence ships with the proposal.
- **Quality gates** (soul-stealer pattern): specific to a named failure cluster; evidence-grounded (real correction records, N occurrences); non-redundant vs existing artifacts; actionable (concrete procedure, not "be careful").
- **Review gate**: nothing applies without user approval — diff + evidence + validation results + measurement plan. Options: interactive CLI, review dashboard, or PR branch.
- **Measurement**: applied diff ⇒ `config_version` flips ⇒ same suite scores the new cohort ⇒ A/B leaderboard compares ⇒ bad changes get rollback proposals. Self-correcting.
- **Generated artifact format**: SKILL.md progressive disclosure (frontmatter ~100 tokens always loaded, body loaded on trigger); frontmatter carries `source: rsi` + `created_by_proposal` for audit.
- Rubric calibration: versioned rubrics, human-graded annotation-queue sample (EvalGen; Weave annotation queues are already shipped), held-out labeled slice as judge meta-eval.
- Never modifies auto-memory files; overlaps are flagged, not overwritten.
- **Optimizer choice (re-verified 2026-07-09)**: standalone `gepa` 0.1.x — `optimize_anything` + `gskill` exist and emit real `.claude/skills/<repo>/SKILL.md` with published wins; weave master already ships a GEPA tracing integration (`weave/integrations/gepa/` — candidate versioning + EvaluationLogger). Do NOT use DSPy's `dspy.GEPA` (pins pre-`optimize_anything` gepa, wrong shape for free-form artifacts). **ACE** (arXiv:2510.04618) is the credible alternative — incremental delta updates beat GEPA on agent benchmarks at ~¼ the rollouts and suit long CLAUDE.md files; borrow its delta-curation style into GEPA's mutation step (our itemized-delta proposal format already aligns) before considering a switch.
- **Two loops, two speeds**: GEPA's iterative search cannot run against live usage — one "rollout" would be days of real sessions. The **inner loop** (GEPA candidate evaluation) runs offline against replayed history and held-in/held-out splits (how `gskill` does it with synthetic tasks); the **outer loop** (live A/B on `config_version` cohorts) only measures the one applied winner, over days-to-weeks. Don't design anything that assumes fast live feedback.

### Reconciliation with W&B HiveMind's insights pipeline (REQUIRED before M4 design is final)

`hivemind insights list/apply` already ships a gated suggestions→context-file loop. From the selfhost repo + CLI (verified 2026-07-09): a two-stage pipeline — Stage 1 per-session extraction (LLM), Stage 2 cluster matching (`INSIGHTS_MATCHER_MODEL`) with 1536-dim embeddings in ClickHouse for insight clustering — feeding per-project suggestions with a `pending|applied|dismissed` lifecycle and an interactive `apply` that writes to the right context file. Self-host LLM config is OpenAI-compatible (explicitly supports W&B/CoreWeave inference).

What it does NOT have: outcome/score grounding (extraction is correction-heuristic, like `/insights` and claude-reflect), proposal validation, or A/B measurement. **Our L4 differentiation is the evidence + measurement side.** Design decision at M4: emit our validated, score-grounded proposals INTO hivemind's suggestion lifecycle (preferred if its API allows external suggestion sources — ask the hivemind team) rather than building a parallel queue; at minimum adopt its `pending|applied|dismissed` lifecycle semantics. Also reuse **`hivemind-query`** (`~/repos/hivemind-query`): a MapReduce YAML harness (map over transcripts → reduce) whose `agents-md.yaml` already does corrections→CLAUDE.md-rule extraction with a good genericness rubric ("timeless rules, not session summaries") — its map/reduce prompts are a tested starting point for our negative-space extraction.

## Coach agent / queryable knowledge base (stretch goal)

A subagent (or MCP server) Matt can converse with to understand what he's been doing and how he's been doing, grounded in the pipeline's data. The hivemind pattern (`@hivemind` subagent wrapping a CLI) applied to evaluations.

- **No new datastore**: Weave holds scores (feedback) + spans; hivemind holds searchable transcripts; local `~/.claude/projects` holds full-fidelity transcripts. The coach gets a thin query tool over Weave feedback+spans (`find_corrections`, `session_digest`, `compare_configs`, `trends`, raw query) plus `@hivemind` search plus local transcript reads.
- **Join facts (verified 2026-07-09)**: hivemind's daemon keys sessions by the local Claude Code session id (`~/.hivemind/state.json`), same id the adapter stamps as `weave_agent_adapter.session_id` — but hivemind's server assigns its own v5 UUID and its CLI doesn't expose the source id. So the hard join path is local transcripts ↔ Weave `session_id`; hivemind is search convenience. If a hard hivemind↔Weave join is ever needed, file a request for a `source_session_id` field rather than engineering around it.
- **Shape**: subagent config whose system prompt knows the score schema + scoring semantics; answers "what do I keep getting corrected on?", "did that CLAUDE.md change help?", "why do long sessions go worse?". Materialized pattern-engine views (M4) make these cheap, but a useful v0 needs only M1 scores + span queries — can land any time after M1.
- **Reuse from hivemind**: `hivemind export` (full session data as Parquet/DuckDB) is a ready analysis substrate, and `hivemind-query`'s YAML map/reduce harness (`guidance.yaml` is literally a coaching-feedback query) can drive the transcript-analysis half — the coach's novel part is only the join with eval scores. Nothing comparable exists inside Weave (no ask-your-traces feature shipped; ARIA/Wandbot don't touch traces).
- Defer embeddings/semantic search until a real query need forces it (DuckDB mirror is the escape hatch if Weave's query API proves too limited for analysis).

## Prior art (consolidated)

| Source | What we take |
|---|---|
| PoLL (arXiv:2404.18796) | 3 small disjoint-family judges; mean-pool scores, max-pool flags |
| arXiv:2410.21819 | Self-preference bias is family/training-distribution driven |
| arXiv:2604.22891 | Decomposed multi-dim rubrics cut self-preference ~31.5% |
| Trust-or-Escalate (arXiv:2407.18370) | Confidence-gated judge cascade |
| TRAIL (arXiv:2505.08638) | Raw-trace judging misses most issues → digest-then-judge |
| GEPA (arXiv:2507.19457) | `optimize_anything` + `gskill`; rationales as textual feedback |
| Activeloop Hivemind | SkillOpt held-in/held-out validation; proactive recall (`beforeSubmitPrompt` injection — deferred) |
| W&B soul-stealer | Emergent clustering; negative-space extraction; quality gates; calibration via replay |
| W&B HiveMind | `@subagent`-wraps-CLI access pattern; transcript search; insights lifecycle (`pending\|applied\|dismissed`) + Parquet/DuckDB export |
| W&B hivemind-query | Map/reduce YAML harness over transcripts; `agents-md.yaml` corrections→rules prompts; `guidance.yaml` coaching query |
| soul.md | Progressive-disclosure SKILL.md format; weak-model pre-screening |
| ACE (arXiv:2510.04618) | Incremental delta updates / curation for long context files (borrow into GEPA mutations) |
| Claude Code OTel + issue #42796 | L1 field semantics (`tool_decision` sources, `error_type`) + behavioral metric set (Read:Edit ratio, edit loops, interrupts) |
| claude-reflect (MIT) | Correction-detection patterns + confidence-scored review-queue UX |
| Docent (Apache-2.0) | Rubric/judge + clustering design over agent transcripts |
| Verdict (MIT) | Judge-ensemble aggregation primitives (voting, pooling, debate) |
| LangSmith / MLflow / Braintrust | Idle-timeout session-close trigger; judge alignment to human labels; rewind re-scoring |
