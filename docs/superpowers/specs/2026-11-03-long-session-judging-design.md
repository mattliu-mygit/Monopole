# Judging large sessions with any model, and step-by-step run setup

Status: draft for approval. Second revision, 2026-11-03.

This revision folds in the user's answers and five studies:

- [long-session-judging-research.md](../../long-session-judging-research.md)
- [windowed-judging-research.md](../../windowed-judging-research.md)
- [large-session-anatomy.md](../../large-session-anatomy.md)
- [small-model-judging-research.md](../../small-model-judging-research.md)
- [small-context-judging-techniques.md](../../small-context-judging-techniques.md)
- [large-session-handling-research.md](../../large-session-handling-research.md)

## Intent

The user wants to judge their real sessions with whichever model they choose, from any provider
and of any context size. Their real sessions are large:

- **Size:** the median is about 272k tokens, and the largest main thread is 2.2M.
- **Workflow sessions:** one has 605 subagents and comes to about 30M tokens once the
  subagents' own work is included.

Cost is not a concern; quality and model choice are.

Today most of that content goes unjudged:

- **Partial view:** a judge sees only each turn's first user message and last assistant message.
- **Skipped turns:** a turn bigger than one window is skipped.
- **Small models excluded:** small-context models can't be planned at all.
- **Workflow sessions:** subagents appear as one line each.

Run setup is one crowded page that offers models regardless of whether they can read the chosen
conversation.

## Goals

1. **Every message:** a judge sees every message of every turn.
2. **Any model:** any model in the catalog can judge any session. Each model gets a reading plan
   sized to how much context it actually uses well, not to its advertised window.
3. **Coverage:** every raw step is read in exactly one window. The pieces are combined into one
   verdict per rubric per thread.
4. **Workflow scores:** a Workflow session gets a session score, an orchestrator score, and a
   score for each subagent.
5. **Step-by-step setup:** setup picks the conversation first. The next step shows every
   available model with how it would read that conversation: its reading plan, request count,
   time, and a quality note. It never blocks.
6. **Measured quality:** whether a reading plan is good enough is measured on the user's own
   sessions, including the user's hand labels, before anything is presented as trustworthy.

## Non-goals

- Judging sessions while they are still running. Judging stays on complete sessions.
- Minimizing cost.
- Changing rubric content or thresholds.
- Changing the CES schema.

## Anti-goals

- Dropping content silently to make something fit. Every rule that shortens content is fixed,
  recorded in the pinned plan, and traceable by evidence ID.
- Presenting a small model's verdict as equivalent to a whole-thread verdict before experiments
  show it.
- Letting an experimental judge write official Scores.

## Constraints

From `AGENTS.md` and `specs/02-evaluation.md`:

- **Pinning:** runs pin identities, plans and protocols before work.
- **Coverage:** judging requires complete raw coverage, which this design extends from turns to
  steps.
- **Artifacts:** persisted artifacts are content-authenticated.
- **Scores:** only finished reviews write Scores.
- **Database:** no migrations of the run database.
- **Claude judges:** tool-disabled, except for the opt-in experiment, which is documented.

## What the evidence says

- **Effective context:** much smaller than advertised. In one study, judges up to 70B dropped
  below chance by 4k tokens of evidence. Llama 3.1 8B holds up to about 1k on a hard test. Even a
  frontier monitor fell from 99.7% to 69% detection between 100k and 800k tokens of Claude Code
  transcript.
- **Small models:** match large ones only on narrow, short jobs, such as checking one claim
  against its evidence or answering one checklist question. Their whole-thread verdicts barely
  agree with each other.
- **Reading strategies:** hiding old tool output matches summarizing it for working agents. But
  summaries hide failure signals, and a judge reading only final replies misses faults it catches
  step by step. So judges read raw steps.
- **Training:** recursive and memory-folding agents gain most from training. Prompted small
  models gain little, so for small models Monopole's planner decides what to read.
- **Session anatomy:** tool output is 50–88% of each session. The stretches between the agent's
  own compactions are 520k tokens or less, and every Workflow subagent is 161k or less.

## Approaches

| Approach | Tradeoff | Decision |
| --- | --- | --- |
| 1. Build in dependency order, one commit per piece | Each piece is testable alone, and the setup screen lands with correct plans | **Picked** ("most thorough") |
| 2. Setup screen first | Visible soonest, but plans would be wrong until judging changes | Rejected |
| 3. One rewrite of judging | Fewer intermediate states, but too large to review | Rejected |

Within approach 1, the user then chose model choice as the goal (not cost), so the design is
built around per-model reading plans rather than around 1M-token defaults.

## Design

### The layers

| Layer | Done by | Produces |
| --- | --- | --- |
| 0. Facts | code | thread tree; steps with evidence IDs and positions; checks, edits, errors, user corrections, subagent launches, compaction points; claim–check pairs |
| 1. Windows | code | per thread and model: whole, or structural windows cut at subagent, compaction stretch, turn, then step. Each window has a core read raw, read-only overlap, and a pinned prefix with the task and open ledger entries |
| 2. Per-window judging | model | a whole-thread verdict, rubric findings, or answers to yes/no questions plus step labels |
| 3. Cross-window memory | code and model | a cited ledger of claims, checks, edits, errors and corrections, folded in two passes; findings for pairs that span windows |
| 4. Aggregation | model, with code audits | verdicts per rubric per thread, merged in a tree when one merge doesn't fit; subagent, orchestrator and session scores; fixed-rule roll-ups stored only as audits |
| 5. Escalation | model | flagged items re-judged by a model the user included in the panel. An item is flagged on panel disagreement of 0.5 or more on a thread verdict, an abstention, a claim–check pair the ledger couldn't resolve, or a large gap between a session verdict and its parts. Layers 2–4 emit what these checks need |

### Reading plans

Each model has four budgets:

- **B_q:** the largest evidence bundle for one question.
- **B_win:** the largest raw window.
- **B_whole:** the largest thread it judges in one request.
- **B_merge:** the largest merge input.

Budgets start from the conservative values in the techniques study, and experiment 1 replaces
them with measured values, pinned in the catalog. Reserves scale with model size: today's fixed
18k reserve is 55% of a 32k model.

For each thread of each session, code picks one plan:

| Plan | When | How |
| --- | --- | --- |
| **W, whole** | thread ≤ B_whole | One request per rubric. Transcript first and rubric last, so the provider's cached copy of the prompt start is reused across rubrics |
| **R, rubric windows** | B_win ≥ 32k (a threshold to test) | A ledger pass, then one request per rubric per window, then resolution requests for claim–check pairs that span windows, then one merge per rubric, as a tree when needed |
| **Q, questions** | otherwise, or when the user picks it | Step labels, yes/no questions on evidence bundles within B_q (quoting the claim sentence, and covering the checks just before it), then verdicts from a merge tree within B_merge |

For 1M-token models, main threads too big for one request use one window per compaction
stretch. That costs about the same as packing windows to the budget, and each window holds
roughly what the agent itself held.

Every judge writes its own verdicts, including a small one. Its plan is recorded with the Scores,
so analysis can compare plans.

### Workflow sessions

- **Order:** subagents are judged first, then the orchestrator, then the session.
- **Orchestrator view:** a short verdict summary of each subagent at the call that launched it.
- **Session verdict:** written by a model from the parts' verdicts, not by a fixed rule. The mean
  and minimum of the parts are stored next to it as an audit, and a large gap flags the session.
- **Orchestrator rubrics:** verification, error recovery, tool choice and state consistency.
  Outcome and autonomy repeat the session score for the orchestrator, so they are skipped (the
  user's decision). The session gets all six rubrics, and each subagent is scored on its own
  work.
- **Score targets:** a subagent's Scores target the first Case of its Trajectory. Scorers are
  named by rubric and level, such as `verification.subagent` and `verification.orchestrator`.

### Partial work and reruns

- **Finished reviews:** each finished review persists as a content-authenticated artifact. That
  covers each window, subagent, orchestrator and merge.
- **What writes CES Scores:** only thread-level verdicts write Scores: each subagent, the
  orchestrator and the session, plus today's turn Scores. Window and merge reviews stay
  artifacts, so newest-per-target reads see one Score per rubric per thread.
- **Missing subagents:** a launch call whose subagent Trajectory isn't in CES marks the session's
  coverage partial and names the missing launch. Everything that does exist is still judged.
- **Partial verdicts:** a session verdict over incomplete inputs is written, labeled partial, and
  names what's missing.
- **Reruns:** a rerun reuses every finished review whose plan, inputs, prompt, model and protocol
  match. This replaces today's rule that any reviewer failure fails the session.

### Run setup

1. **Conversation:** load sessions and select one or more.
2. **Models:** every available model, meaning one whose credentials or CLI login are present,
   with the plan per selected session, the request count, the estimated time, and a quality note.
   - Time is estimated from the requests and the provider's concurrency (W&B 4 calls, Codex 2),
     using measured request times where the run history has them.
   - Setup auto-selects a recommended panel: different model families, chosen by measured
     budgets.
   - Setup never blocks.
3. **Rubrics and confirmation:** then the run starts.

### Tool-using judge experiment

- **Role:** Claude Code with read-only tools serves only as an escalation and audit judge on
  flagged items.
- **Comparison:** its results are compared against requests built by code.
- **Scores:** it never writes official Scores until the experiments show it is reliable.
- **Rule exception:** `AGENTS.md` documents it.

## Pieces, in build order

1. **Catalog and budgets:**
   - correct context sizes from vendor docs (Codex GPT-5.6 is 272k; Claude Sonnet 5 is 1M);
   - drop retired models and add the current ones W&B lists;
   - per-model budgets;
   - reserves that scale with size.
2. **Reader and renderer:**
   - every message;
   - subagent messages and nesting kept;
   - `agent_type` from the agent's type;
   - Codex tool programs parsed for their commands, patches and exit codes.
3. **Layer 0 facts and fixed compression:** plain-text tool output, no per-call counters, and
   pointers for repeated results.
4. **Plan W:** whole-thread requests, transcript first and rubric last, plus prompt caching.
5. **Structural windows, plan R, the ledger and merge trees:** steps split between messages and
   tool calls when needed.
6. **Plan Q:** questions, step labels, and merge trees for small models.
7. **Run setup:** conversation first, then models with their plans, never blocking.
8. **Workflow multi-level scores, and partial work and reruns.**
9. **Experiment harness:** planted probes, budget calibration, and the comparisons below.
10. **Tool-using judge experiment.**

### Related fixes, scoped separately

These fixes were found during the research and are not part of this project:

- **Confirmed:** the outcome scorer reads only `Bash` calls (`scorers/outcome.py:700-709`), so
  no Codex check ever reaches outcome scores.
- **Confirmed:** CES stores only a compaction's trigger, not its summary, so the agent's own
  summaries never reach a judge. This needs an adapter contract change.
- **Suspected:** a Codex `spawn_agent` subagent wasn't captured in CES.

## Validation

All of it runs on offline copies, and nothing is written to CES.

1. **Calibrate budgets per model:** plant probes with known answers at the start, middle and end
   of 1k–256k-token bundles. A model's budget is the largest size within 5 points of its 1k
   accuracy.
2. **Agreement with whole-thread references:** compare against whole-thread verdicts from two
   frontier families, on threads that fit. That means the 605 subagents, the compaction
   stretches, and ordinary sessions. Then force plans R and Q at smaller budgets and measure
   agreement, Krippendorff's α, and planted detection 0, 1 and 5+ windows apart.
3. **Prompt layout:** rubric last, versus rubric before and after the transcript, versus
   reminders between steps.
4. **What crosses windows:** today's digests versus the two-pass ledger.
5. **Escalation:** requests built by code versus the tool-using judge.
6. **Juries:** three families on one plan, versus one family on three plans.
7. **Human labels:** the user labels 20–40 cells where plans or models disagree by 0.5 or more,
   plus a few where they agree, blind to which plan produced them. A sign test reads the result:
   at 30 cells, the winner needs 21. This can show large gaps between plans, but not certify
   agreement.

## Decisions from the conversation

- **Every message:** judges see every message.
- **Workflow scores:** a session score, plus orchestrator and per-subagent scores.
- **Tool-using judge:** worth an experiment.
- **Partial work:** recorded, and reruns skip finished work.
- **Tool output:** stays cut at 12,000 characters by default.
- **Human review:** part of validation, with the user as the labeler.
- **Model choice:** the goal is choosing models and providers, and cost is not a concern.
- **Setup:** step 2 shows every model's reading plan and never blocks.
- **Build order:** approach 1, implemented thoroughly.
- **Providers:** existing routes only: the open-model route, the OpenAI API, and the Claude,
  Codex and Antigravity CLIs. The open-model route is CoreWeave Serverless Inference (Forge),
  where W&B Inference is moving. It has the same OpenAI-compatible endpoint Monopole already
  calls (`https://api.inference.wandb.ai/v1`, `judges/inference.py:23`), and the user's existing
  W&B key lists 24 models there today. Piece 1 renames the route, refreshes the catalog from the
  live model list, and takes prices from Forge's pricing page. A separate Forge key setting is
  added only if W&B keys stop working.
- **Run time:** long runs are accepted. That's about 9 hours per judge for the 605-subagent
  session on CoreWeave, and up to about a day with a full panel on one route.
  - Setup shows the estimate, and runs show progress.
  - Finished work survives restarts.
  - The parallel-call limits (CoreWeave 4, Codex 2; `specs/02-evaluation.md:243-245`) are raised
    where a provider's tested limit allows.
  - Judging all rubrics for a subagent in one request is tested in the experiments, not assumed.

## My calls

- **Budgets and reserves:** measured budgets are pinned catalog data, re-measured when the
  catalog, renderer or model version changes. Reserves scale with model size.
- **Small models:** a small judge writes its own verdicts under plan Q, labeled as such.
  Escalation goes only to a model the user included.
- **Prompt layout:** the rubric goes last by default. Reminders between steps are added only if
  experiment 3 shows a gain, and are marked as judge notes.
- **Juries:** panels may mix reading plans as well as families, depending on experiment 6.
- **1M models:** too-big main threads use one window per compaction stretch.
- **Verification questions:** they quote the claim sentence and cover the checks just before it.
- **Coverage:** every raw step is in exactly one window core. Merges read findings that cite
  evidence IDs.
- **Verification across levels:** an orchestrator's claim counts as verified when the check ran
  inside one of its subagents.
- **Replace, don't coexist:** the new protocol replaces today's digest, window and merge protocol
  under a new pipeline version. Runs pinned to the old version fail closed with a clear message,
  as incompatible versions do today. Experiment 4 compares digests and the ledger offline,
  without keeping both in the product.
- **Labeling:** the human review is offline. A command writes blind cells to a local file, the
  user fills in labels, and a command reads them back. It is not a UI feature.

## Left to the builder

- Exact prompt wording.
- The R/Q threshold and the starting budgets until measured.
- Where summaries go when one launch call starts 100+ subagents and the window is small.
- UI layout within the steps.
- Commit boundaries within each piece.

## Open questions for the user

None. The orchestrator rubrics, run time and providers were decided in the conversation.
