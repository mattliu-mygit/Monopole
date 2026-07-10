# weave-agent-signals: Layered evaluation & weak RSI for agent traces

> v1 design. Reads agent-session traces from Weave (emitted by [weave-agent-adapter](https://github.com/mattliu-mygit/Weave-Agent-Adapter)), scores them through four layers — deterministic, LLM-as-judge, pattern recognition, self-improvement — and writes structured feedback back to the same Weave project. The end goal: a closed loop that proposes, measures, and iterates on the user's prompts, skills, and configuration.

## 1. Purpose

Personal coding performance evaluation over Claude Code sessions at three granularities (per-turn, per-session, multi-session) with an improvement loop. Not a generic eval framework — opinionated about what matters for a single power user iterating on their own agent workflow.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Weave project: agent-sessions                          │
│  ┌───────────────┐         ┌──────────────────────┐     │
│  │ Agent traces   │         │ Feedback (scores)    │     │
│  │ (from adapter) │         │ on agent refs        │     │
│  └───────┬───────┘         └──────────▲───────────┘     │
└──────────┼────────────────────────────┼─────────────────┘
           │ agents/spans/query         │ feedback/create
           │ + custom_attr_columns      │ (batch)
           ▼                            │
┌──────────────────────────────────────────────────────────┐
│  weave-agent-signals                                     │
│                                                          │
│  ┌─────────────────┐                                     │
│  │ Span Reader      │ query turns + conversations        │
│  │ (spec 01)        │ from Weave, hydrate into Turn/     │
│  │                  │ Session models with tool results    │
│  └────────┬─────────┘                                    │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐    │
│  │ L1 Fact Extraction (M1)                          │    │
│  │  deterministic scorers (M1); micro-LLM (M2)     │    │
│  │  ├── outcome extractor (spec 02)                 │    │
│  │  │   test/build/lint/git → pass/fail/counts      │    │
│  │  ├── implicit feedback (spec 03)                 │    │
│  │  │   correction density + abandonment            │    │
│  │  └── efficiency (spec 04)                        │    │
│  │      error loops/repeated reads → waste ratio    │    │
│  └────────┬─────────────────────────────────────────┘    │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐    │
│  │ Routing Gates (M2)                                │    │
│  │  deterministic predicates → skip / small / panel │    │
│  └────────┬─────────────────────────────────────────┘    │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐    │
│  │ L2 LLM-as-Judge (M2–M3)                           │    │
│  │  ├── turn signals: process rubrics, small judges │    │
│  │  └── session panel:                              │    │
│  │      digest → 3-family PoLL → confidence cascade │    │
│  └────────┬─────────────────────────────────────────┘    │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐    │
│  │ L3 Pattern Engine (M4)                            │    │
│  │  emergent clustering, correction records,        │    │
│  │  A/B leaderboards keyed by config_version        │    │
│  └────────┬─────────────────────────────────────────┘    │
│           │                                              │
│  ┌────────▼─────────────────────────────────────────┐    │
│  │ L4 Weak RSI (M4)                                  │    │
│  │  GEPA reflector → create/edit/delete artifacts   │    │
│  │  → validation → review gate → config_version flip│    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Score Writer (spec 05)                           │    │
│  │ feedback/create on agent_turn / agent_conversation│    │
│  │ refs with typed scorer columns                   │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │ CLI (spec 06)                                    │    │
│  │ score · backfill · inspect · setup               │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘

Stretch goal: coach agent (see L4 section below).
```

## 3. Score model

Every scorer produces a `Score`:

```python
@dataclass
class Score:
    scorer: str           # e.g. "outcome.test", "implicit.correction_density"
    value: float | bool   # 0–1 continuous or binary
    tags: list[str]       # categorical labels, e.g. ["test_failure", "lint_clean"]
    confidence: float     # 0–1, used for routing and cascade
    metadata: dict        # scorer-specific detail (counts, commands, etc.)
    granularity: str      # "turn" | "session"
```

Scores are written as Weave feedback:
- **Turn scores** → `agent_turn/{trace_id}` ref
- **Session scores** → `agent_conversation/{conversation_id}` ref
- **Feedback type**: custom (`weave_agent_signals.<scorer>`) for M1 deterministic; `wandb.agent_monitor` for Signals-native L2 judges (requires `runnable_ref`/`call_ref`/`trigger_ref`)
- **Typed columns**: `scorer_ratings` for continuous, `scorer_tags` for categorical, with `_reasons` and `_confidences` maps
- **Dedup**: skip if feedback with same scorer+ref already exists (idempotent backfill)

## 4. Layers

### L1 Objective fact extraction (M1)

L1 is **fact extraction, not judgment**. The boundary with L2 is not the mechanism — it's fact vs. opinion. Three extraction classes:

- **Class 1 — environment verdicts** (deterministic, zero holes): exit codes, parsed test counts, git operations, the adapter's hook-event counters (`steering_count`, `denial_count`, `tool_error_count`), token/cache counts, compaction events. The environment already emitted a machine-readable verdict; an LLM could only add noise.
- **Class 2 — structural trace patterns** (deterministic, one rule): error loops, repeated reads, test-recency. Objectively-defined patterns. Rule: **L1 scores the pattern, never the blame** — "3 failing retries of the same command" is a fact; whether it was avoidable is L2's call.
- **Class 3 — micro-LLM fact extraction** (small model, single short text, structured output): facts latent in language — "does this final message claim completion?", "is this steering message a correction, a neutral addition, or a redirect?", "did this unparsed command output report test success?". LLM-as-parser/classifier, not judge: narrow tasks where small models are near-perfect, costing ~$0.001 and firing only on the relevant residue. (TRAIL's ~11–22% catch rate is about whole-trace judging — a different regime.)

What stays out of L1: rubrics, quality opinions, blame — that's L2.

1. **Verifiable environment outcomes** (spec 02): regex detection + framework parsers over `execute_tool` Bash spans for test/build/lint/git commands, with exit-code fallback. **M1**: Tier 1 (deterministic) only. **M2**: micro-LLM fallback on the unparsed residue, verification-before-done (deterministic test-recency + micro-LLM done-claim detection).

2. **Implicit human feedback** (spec 03): **M1**: correction density (factual ratio of turns with steering/denials) and abandonment detection (observable end-state). Raw hook-event counters available as span attrs. **M2**: micro-LLM classification of steering/denial content (correction vs. addition vs. redirect), frustration index over classified events, corrective follow-ups.

3. **Efficiency** (spec 04): error loops, repeated reads (with deterministic excuses: post-compaction re-reads, subagent contexts), waste ratio. Normalized — raw counts are anti-metrics (denominators for L3, not scores). Token efficiency is metadata, not a standalone score.

**Uses**: ground truth for judge calibration, L2 routing gates, L3 correlation features, coaching input.

### L2 LLM-as-Judge (M2, M3)

Two granularities with different judge strategies:

**Routing gates (M2)**: deterministic predicates deciding which turns get LLM judging. At single-user volume (~100–200 turns/day, small judges ≈ $0.10–0.20/day) gates save pennies; their value is noise reduction. Default lean: skip gates in v1 — judge everything with small models, sample the panel — and add gates only if cost or score noise actually hurts. Thresholds set from backfill data, not guesses.

**Turn-level (M2)**: Weave Signals fire on `turn_ended`, running small judges (`gpt-oss-20b`, `Llama-3.1-8B`, `granite-4.1-8b`) on process rubrics — verification discipline, error recovery, tool choice quality, user-prompt quality. Custom-attr filters (e.g. only judge turns with `tool_error_count > 0`) reduce cost. **Weave-native reality (verified 2026-07-09)**: turn-level judging infrastructure already ships — 13 preset classifier signals + 8 agent-signal templates, custom prompts, 0–1 sampling, W&B Inference judge picker. M2 = signal/rubric definitions published as code, not judging infrastructure. Also lands: micro-LLM classifiers for frustration scoring (spec 03) and verification-before-done (spec 02).

**Session-level (M3)**: raw whole-trace judging catches ~11–22% of issues (TRAIL, arXiv:2505.08638). Fix: **digest builder** extracts ~10–20 chronological key moments + L1 scores into ~2K tokens, then a **3-family PoLL panel** (arXiv:2404.18796) judges the digest. Panel: `gpt-oss-120b` / `DeepSeek-V4` / `Qwen3-30B` (or `Llama-3.3-70B`), mean-pooled scores, max-pooled flags. Confidence-gated escalation (Trust-or-Escalate, arXiv:2407.18370) on low agreement. **Known risk**: the digest is load-bearing and lossy — whatever extraction rules miss, the panel never sees. Mitigations: anchor on L1 facts (anomalies always included); validate digests against ~10 hand-read sessions before trusting panel scores (M3 exit criterion). Session scope does NOT exist in Weave Signals (`conversation_ended` is a commented-out TODO) — build M3 externally, migrate onto the native trigger when it ships.

**Bias controls**: agent is Claude ⇒ non-Anthropic judges only; `gpt-oss` counts as OpenAI-family (shared training distribution, arXiv:2410.21819); decomposed multi-dim rubrics cut self-preference ~31.5% (arXiv:2604.22891); position-swap for pairwise comparisons only (meaningless for absolute rubric scoring).

All judges via **W&B Inference** (OpenAI-compatible, auth via W&B API key, no external keys).

### L3 Pattern Recognition (M4)

Scheduled rollups via genai-spans-query + feedback queries → pandas:
- **Emergent failure clustering** (soul-stealer pattern) — don't impose a taxonomy; cluster negative-signal turns by features and let recurring shapes name themselves
- **Correction records** — structured prompt-response pairs of what got corrected and how (negative space), sourced from full local transcripts. Reuse `hivemind-query`'s map/reduce prompts (`agents-md.yaml`) as a tested starting point.
- Trend analysis, prompt-quality ↔ efficiency correlations
- **A/B leaderboards keyed by `config_version`** — quasi-experimental, not an RCT. Task mix confounds everything; trust only large effects; stratify by session type rather than comparing raw cohort means.

Output: materialized views (queryable by the coach agent) + EvaluationLogger rollups.

**Documented limitations**: sessions ≠ tasks (one `conversation_id` can span several distinct tasks); frustration measures the user, whose standards drift — within-cohort comparisons valid, long-horizon trends not.

### L4 Weak RSI (M4)

Engine: **`gepa.optimize_anything`** (MIT; arXiv:2507.19457, ICLR 2026 oral). Judge rationales = GEPA textual feedback; `gepa.gskill` as reference recipe; Weave has native GEPA tracing.

**Action space**: create, edit, and delete:
- CLAUDE.md sections
- Skills (`.claude/skills/`)
- Slash commands (`.claude/commands/`)
- Prompt templates
- Memory entries

Every proposal driven by evaluation outcomes (scores, judge rationales, A/B results), all behind a **review-queue gate**. Applied diff ⇒ `config_version` flips ⇒ same suite measures before/after. Rubrics versioned + recalibrated against a small human-graded annotation-queue sample (EvalGen).

**Proposal validation** (SkillOpt pattern): weak-model pre-screen (syntax/sanity, ~$0.001) → held-in test (do triggering failure cases improve?) → held-out test (do good sessions regress?). Only validated proposals enter the review queue.

**Two loops, two speeds**: GEPA's iterative search cannot run against live usage — one "rollout" would be days of real sessions. The **inner loop** (GEPA candidate evaluation) runs offline against replayed history and held-in/held-out splits; the **outer loop** (live A/B on `config_version` cohorts) only measures the one applied winner, over days-to-weeks.

**Generated artifact format** follows SKILL.md progressive disclosure: YAML frontmatter (~100 tokens, always loaded) + full instructions (~2–5K tokens, loaded when triggered). Prevents config-surface bloat.

Baseline: user currently has no global CLAUDE.md/skills/commands, only auto-memory — early iterations mostly *create* from observed patterns. Deletion proposals fire when an artifact's config_version cohort measures worse or goes unused.

**HiveMind reconciliation (required before M4 design is final)**: `hivemind insights list/apply` already ships a gated suggestions→context-file loop. It lacks outcome/score grounding, proposal validation, and A/B measurement — our L4 differentiation. Emit validated proposals INTO hivemind's suggestion lifecycle if its API allows external sources; at minimum adopt its `pending|applied|dismissed` lifecycle semantics.

### Stretch goal: coach agent

A subagent Matt can converse with to understand what he's been doing and how, grounded in the pipeline's data. No new datastore — Weave holds scores + spans; hivemind holds searchable transcripts; local `~/.claude/projects` holds full-fidelity transcripts. Join key: adapter stamps `weave_agent_adapter.session_id` which matches hivemind's daemon session key. Useful from M1 scores alone; gets smarter with each layer. Can slot in any time after M1.

## 5. Dependencies

| Dependency | Role |
|---|---|
| `weave-agent-adapter` | Produces the traces (custom attrs: `config_version`, `steering_count`, `denial_count`, `tool_error_count`, `git_branch`, `effort_level`) |
| `weave` SDK | Span queries, feedback write-back |
| `httpx` | Direct Weave trace-server API calls (agents/spans/query, feedback/create) |
| `openai` (optional) | W&B Inference judge calls (OpenAI-compatible) |
| `gepa` (optional) | L4 RSI loop |
| `pandas` | L3 rollups |
| `ccusage` (optional) | Cost axis join |
| `agentevals` | Referenceless trajectory judge (L2) |
| `openevals` | Judge primitives, pyright/code evaluators |

## 6. Weave project

Entity: `mliu-wandb-weights-biases`, project: `agent-sessions` — same project the adapter writes to. Scores land as feedback on the same traces they evaluate.

## 7. Scheduling

CLI-driven, schedulable via launchd:
- **`score`**: score recent unscored turns/sessions (incremental)
- **`backfill`**: score historical sessions in a date range
- **`inspect`** / **`setup`**: debugging and one-time configuration

Typical cron: `score` every 30min (or on session-end hook). Future commands (`reflect`, `propose`) add their own cadences.

## 8. Milestones

1. **M1 — deterministic layer** (specs 01–06): outcome extractor + implicit feedback + efficiency scorers + score write-back + backfill CLI. Exit: backfill recent history, spot-check against ~10 sessions.
2. **M2 — turn signals**: routing gates + preset/custom Weave Signals (process rubrics, small judges, sampling, custom-attr filters). Exit: live sessions → signals fire with tags/ratings.
3. **M3 — session panel**: digest builder + 3-family PoLL + feedback on `agent_conversation` refs + confidence-gated escalation. Exit: backfill panel scores, verify disagreement rates.
4. **M4 — pattern + RSI**: emergent clustering + A/B leaderboards, coaching digest, GEPA diff proposer with validation + review gate, annotation-queue calibration. Exit: dry-run reflector on history, apply one diff → config_version flips → A/B populates.

## 9. Prior art & reuse

| Library | Use |
|---|---|
| `agentevals` | Referenceless trajectory judge (MIT) |
| RAGAS | `AgentGoalAccuracyWithoutReference` |
| `openevals` | Judge primitives + pyright/code evaluators |
| `ccusage --json` | Cost axis |
| `claude-code-log` | Transcript parsing (prior art, not a dependency) |
| `gepa` | L4 artifact optimization |
| PoLL (arXiv:2404.18796) | 3 small disjoint-family judges, 7–8x cheaper than GPT-4 |
| Trust-or-Escalate (arXiv:2407.18370) | Confidence-gated cascade |
| Decomposed rubrics (arXiv:2604.22891) | Multi-dim scoring cuts self-preference ~31.5% |
| TRAIL (arXiv:2505.08638) | Motivates digest-then-judge over raw-trace judging |
| Activeloop Hivemind | SkillOpt held-in/held-out proposal validation |
| W&B soul-stealer / HiveMind | Emergent clustering, negative-space extraction, quality gates; subagent-wraps-CLI access pattern |
| W&B hivemind-query | Map/reduce YAML harness over transcripts; `agents-md.yaml` corrections→rules prompts |
| ACE (arXiv:2510.04618) | Incremental delta updates for long context files (borrow into GEPA mutations) |
| Claude Code OTel + issue #42796 | L1 field semantics + behavioral metric set (Read:Edit ratio, edit loops, interrupts) |
| Verdict (MIT) | Judge-ensemble aggregation primitives (voting, pooling, debate) |
| LangSmith / MLflow / Braintrust | Idle-timeout session-close trigger; judge alignment; rewind re-scoring |

**Novel contribution**: rubric-based agent-session scoring persisted as Weave feedback, closing into an eval-score-driven config loop with A/B measurement. Distinct from auto-memory/insights (heuristic, not score-driven).
