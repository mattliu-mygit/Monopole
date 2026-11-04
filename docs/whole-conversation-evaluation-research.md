# Evaluating a very large conversation as a whole

Research notes on ways to give one trustworthy verdict per rubric for a whole Claude Code or Codex
session of hundreds of thousands to tens of millions of tokens, beyond reading it in windows.
Observed 2026-11-04. They are not Monopole product guarantees.

This is the seventh document in a series and does not repeat the others:

- [long-session-judging-research.md](long-session-judging-research.md): long-context models,
  compression rules, whole-session requests, prompt caching, agent-as-a-judge.
- [windowed-judging-research.md](windowed-judging-research.md): window-then-merge, structural
  windows, the cited ledger, claim–check pairs, summary trees.
- [small-model-judging-research.md](small-model-judging-research.md): small judges, checklists,
  claim checking, cascades.
- [small-context-judging-techniques.md](small-context-judging-techniques.md): the 26-technique
  catalog (B1–B26).
- [large-session-anatomy.md](large-session-anatomy.md) and
  [large-session-handling-research.md](large-session-handling-research.md): measurements and
  simulations on the user's four largest sessions.
- [The approved design](superpowers/specs/2026-11-03-long-session-judging-design.md): per-model
  reading plans W, R and Q, structural windows, the cited ledger, merge trees, escalation.

Citation rules. Every external claim cites a numbered source from the list at the end; all were
read for this document on 2026-11-04. Claims taken from the companion documents cite the
companion's `path:line`. Repository claims cite `path:line`. Text marked **Inference** is
reasoning from cited facts, not something a source states. Facts about the user's sessions are
aggregates taken from the anatomy and handling studies; no transcript text appears here.

Terms used throughout:

- **Thread, step, window, core, finding, merge, ledger:** as in the windowed study
  (`docs/windowed-judging-research.md:20-39`). A thread is one agent's own messages and tool
  calls; a step is one message, tool call or compaction; a ledger is a list of cited facts.
- **Event log:** the session rewritten as a typed list of events (user request, correction,
  edit of a named file, check with its result, error, claim, delegation, compaction), each with
  its evidence ID, time and position. The approved design's layer 0 produces most of these facts
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:111`).
- **Temporal property:** a rule about the order of events that code can check over the whole log,
  such as "every completion claim about a file has a passing relevant check after the last edit
  to that file". Checking a recorded log against such rules is called conformance checking in
  process mining and offline runtime verification in formal methods.
- **Requirement ledger:** the user's requests, constraints and acceptance criteria for one thread,
  each citing the message that stated or revised it.
- **Coverage duty:** a rule that every failure signal code found (failed check, error, denial,
  correction) must be cited or explicitly dismissed by some finding before a verdict is accepted.
- **Stability audit:** re-running the merge with one window's findings removed, or with window
  edges moved, to see whether the verdict is an artifact of where the session was cut.

## Summary

1. **Property checks over an event log:** code checks order rules across the whole session at any
   size. A temporal-logic checker beat GPT-4o, F1 91.8% to 74.4%, on 571 recorded runs [1].
2. **A requirement ledger from all user messages,** which are small enough to read whole.
   Human-written per-item rubrics raised judge–human agreement from 36% to 93% [11].
3. **Coverage-checked verdicts:** every code-found failure signal must be cited or dismissed,
   since summaries hide failures (`docs/large-session-handling-research.md:329-331`).
4. **Stability audits, then pairwise adjudication:** catch verdicts that depend on where the
   session was cut, and settle them side by side; debate lifted weak judges from 48% to 76% [46].
5. **Label-efficient trust:** actively chosen labels, conformal sets, prediction-powered averages
   [44][53][54].

Not now: learned compression (self-hosting only), consensus debate [45][47], and execution
grounding, since CES records no repository state [25].

## How the approaches compare

What each approach can decide for each rubric. "Yes" means it can settle the rubric's main
question; "part" means it supplies part of the evidence; "no" means it cannot help.

| Approach | Verification | Error recovery | Tool choice | State consistency | Outcome | Autonomy | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Execution-grounded | part | no | no | part | yes, if repository state is recoverable | no | large, needs an adapter change |
| 2. Event log and property checks | part: order facts | part: repeats, unanswered errors | part: redundancy, risky commands | part, once constraints are extracted | part: what is still broken at the end | part: counts only | medium |
| 3. Requirement ledger | part: which claims are material | no | part: task fit | yes, as the list of constraints | yes, as the checklist | yes, as the record of what was already said | medium |
| 4. Agentic and recursive judges | escalation only | escalation only | escalation only | escalation only | escalation only | escalation only | done as an experiment in the design |
| 5. Learned compression | no | no | no | no | no | no | large; self-hosting only |
| 6. Relative evaluation | adjudication | adjudication | adjudication | adjudication | adjudication | adjudication | small |
| 7. Multi-judge designs | all, as panels | all | all | all | all | all | exists |
| 8. Uncertainty | all, as intervals and abstention | all | all | all | all | all | small to medium |
| 9. Counterfactual audits | all, as a check on the verdict | all | all | all | all | all | medium |
| 10. Summaries that cannot hide failures | part | part | no | part | part | part | small |
| 11. Other: reasoning field, active labels | part | no | no | no | no | part | small |

## 1. Execution-grounded evaluation

### Mechanism

Instead of reading the transcript, check the work itself: the final repository, the diff the
session produced, and the agent's own verification commands re-run in a sandbox at the state where
the agent claimed success. A judge, or an agent that explores the repository, can also write a
checklist grounded in the code and score the diff against it without running anything.

### Measured evidence

- **Repository-grounded checklists without execution.** An agent explored each repository and wrote
  a rubric for the task; candidate patches were then scored against it without running code. On
  SWE-bench Verified this selected patches at 54.2% with Qwen3-Coder-30B-A3B, at least 3.5 points
  above the best baseline. Rubric scores predicted the ground-truth tests at ROC-AUC 0.886. When
  tests passed but the rubric scored low, 54% of the rubric's objections were useful, such as a
  missed root cause or an untested edge case [20].
- **Trained trajectory verifiers.** SWE-RM, a 30B mixture-of-experts model with 3B active
  parameters, reads a trajectory and its patch and predicts success. It was trained on 100k
  trajectories from 170 repositories with a 256k context. At 32k context it could score only 12.5%
  of trajectories without overflowing, at 128k 99.5%. Its AUC ranged from about 0.73 to 0.81 across
  its training ablations [21]. SWE-Critix raised a 30B verifier's accuracy on held-out coding
  trajectories from 57% to 78% with fine-tuning and reinforcement learning [30].
- **Tests are weak evidence of correctness.** Adding adversarial tests rejected 19.78% of previously
  passing SWE-bench Verified patches [24]. Requirement-targeted tests found that 34.4% of trials
  judged correct violated the task, lowering one resolution rate from 50.6% to 33.2% [23]. That
  method needs a ground-truth patch to validate its new tests [23], which recorded user sessions do
  not have.
- **What the diff shows that the transcript hides.** In ImpossibleBench, tasks whose tests
  contradict the specification can only be passed by cheating. LLM monitors reading transcripts
  caught 86–89% of cheating on short coding tasks but only 42–65% on SWE-bench tasks. One undetected
  case justified spec-violating code as "backward compatibility" [22]. Cheating included editing
  tests and special-casing them [22], which is visible in a diff.
- **Recovering real sessions is rare.** SWE-Together kept 109 tasks from 11,260 recorded real
  coding sessions, selecting those with recoverable repository states, clear goals and observable
  outcomes [25].
- **Limit in principle.** "Every verifier we can build is only a proxy for human intent, never the
  intent itself" [27].

### What it can and cannot judge

- **Outcome:** yes, if the final state is recoverable, though tests alone overstate success
  [23][24]. **Inference:** grading the diff against the user's own requirements (section 3) is
  closer to the rubric than re-running the repository's tests
  (`src/weave_agent_signals/judges/rubrics.py:311-319`).
- **Verification:** partly. A re-run shows whether a claimed check would pass at the claim's
  state, and the diff shows test edits [22]. It cannot show whether the agent ran a check before
  claiming, which is what the anchors ask (`rubrics.py:111-115`, `:129-136`).
- **State consistency:** partly, for constraints visible in code, such as a forbidden file or a
  preserved public API.
- **Error recovery, tool choice, autonomy:** no. They are about the path, not the result.

### Novelty and fit

- **Novelty:** the approved design reads only the transcript
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:107-116`).
- **CES data:** CES records `cwd` and `git_branch` on each member Case, but no commit identity,
  repository URL, diff or snapshot (`weave-agent-adapter/specs/EVALUATIONS_CASE_CONTRACT.md:26-27`),
  and caps text at 32,000 characters (`:39`). **Inference:** the end state of a past session cannot
  be rebuilt from CES today. A repository at `cwd` on the user's machine has usually moved on since
  the session.
- **Partial reconstruction:** edit tool arguments are recorded, but many edits run through shell
  commands or Codex programs (`docs/large-session-handling-research.md:195-205`, `:430-437`).
  **Inference:** replaying edits would give an incomplete diff.
- **Sandbox:** Monopole already runs paired A/B verification in two local Smol Machines microVMs
  built from one pinned image (`specs/04-evaluation-runs.md:196-249`). **Inference:** that runner
  could re-run checks once a session's repository state is known.

### Effort and risks

- **Effort:** large. It needs an adapter contract change, for example the commit and a bounded diff
  at each turn's end, plus an artifact store for diffs over the text cap. Then rubric generation
  over the diff, and sandbox replay.
- **Risks:** weak tests [23][24]; environments that cannot be rebuilt [25]; side effects when
  re-running commands; and an outcome score that ignores the user's actual requests unless it is
  tied to the requirement ledger.

## 2. Structure first: event logs, property checks and provenance graphs

### Mechanism

Convert each thread into a typed event log. Then evaluate the log, not the text:

- **Conformance checking:** temporal properties written per rubric are compiled into automata and
  run over the whole log by code.
- **Provenance graphs:** nodes for files, commands, results, claims and corrections, with edges for
  "edited", "checked", "cites" and "contradicts". A reviewer or judge walks from a claim to its
  supporting checks.
- **Process mining:** mines common paths across many sessions and flags deviations.

### Measured evidence

- **Deterministic contracts match or beat LLM judges on procedure.** ContrAgent writes
  assume-guarantee contracts in linear temporal logic over finite traces. It compiles each contract
  to a deterministic automaton that grades recorded traces offline [1]:
  - On R-Judge's 571 recorded agent runs it reached F1 91.8% (precision 97.0%, recall 87.0%), above
    GPT-4o at 74.4% and the human baseline of about 89%. The runs it missed were "semantic rather
    than procedural" [1].
  - On τ²-bench, Claude 3.7 passed 25–60% of tasks on all four tries by outcome alone, but 0% when
    the pass also required zero contract violations [1].
  - ContrAgent reports that AgentPex, which extracts rules from agent prompts with a model, found
    83% of fully rewarded Claude traces still broke a procedural rule [1][4].
  - Its contract library was written with an LLM and inspected by people. The checker trusts the
    recorded events to be faithful [1].
- **The log must carry history.** Replaying three benchmarks' recorded runs through an unmodified
  temporal-logic monitor, five generic rules flagged 70.1–71.8% of attacks but also 29.3% of benign
  runs. The authors trace the imprecision to logs that "rarely record approvals and never record
  timestamps". One planted line defeated a naive provenance check on 94–99% of the runs it would
  otherwise flag [2].
- **Judge-free compliance scores.** AgentLTL expresses procedural rules in first-order temporal
  logic and returns a deterministic compliance score for completed traces [3].
- **Claim-to-evidence graphs and process ontologies.** LEDGER links claims to the actions,
  artifacts and checks that support them, for human audit; it reports examples, not accuracy [5].
  ProcCtrlBench maps coding-agent logs into one event format and scores 11 process-defect types in 4
  categories [6]. Agent Behavior Mining maps agent events to the XES process-mining standard [7].
- **Graphs for failure attribution.** An information dependency graph, recording which output each
  step relied on, let a trained 8B model attribute failures up to 18.18% more accurately than
  baselines on Who&When [8]. Reasoning LLMs without such training were generally below 10% accuracy
  on this task, and that training data came from counterfactual replay, which re-runs the system
  [9]. A spectrum-based method also needs repeated runs [10].

### What it can and cannot judge

Rule examples, each mapped to an anchor in `src/weave_agent_signals/judges/rubrics.py`:

| Rubric | Property code can check | What still needs a model |
| --- | --- | --- |
| Verification | a claim about file F has a passing check after F's last edit and before the claim (`:111-115`, `:133-136`) | is it a material claim; is the check relevant |
| Error recovery | the same failing command repeats with no edit in between (`:180-183`); an error gets no response within N steps | is the diagnosis real |
| Tool choice | destructive command patterns; re-reads with no edit (`:230-237`) | was the method a good fit |
| State consistency | an edit touches a file a constraint forbade, with no authorizing user message after it (`:281-284`, `:271-273`) | extracting the constraint |
| Outcome | a check still failing, or an error still unresolved, at the thread's end | whether the requests were met |
| Autonomy | counts of corrections, steering and denials | whether each correction was avoidable |

Worked example, the windowed study's parser session (`docs/windowed-judging-research.md:90-104`):

- **Events:** a failing pytest at turn 8, an edit of `parser.py` at turn 11, and "Fixed; all tests
  pass" at turn 12. The first passing pytest comes at turn 64.
- **Property check:** the verification rule finds no check between the turn-11 edit and the
  turn-12 claim, wherever the windows were cut.
- **Remaining model questions:** is the claim material, and is the turn-64 test relevant?
- **Result:** the candidate is the 0.25 anchor (`rubrics.py:133-136`).

### Novelty and fit

- **Novelty:** the design's layer 0 already extracts facts and claim–check pairs, and sends pairs
  that span windows to resolution requests
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:111-116`). What is new:
  - a versioned, pinned set of rubric properties evaluated over the whole session;
  - results that become session-wide findings for the merge, not only questions for a resolution
    request;
  - the same properties run across sessions, as process mining does.
- **Fit with CES:** CES carries what the monitor study found missing:
  - start and end times on every member Case
    (`weave-agent-adapter/specs/EVALUATIONS_CASE_CONTRACT.md:27`);
  - approval decisions and exit codes on tool calls (`:35`);
  - a steering flag on user messages (`:28`).

  **Inference:** history-dependent rules should be more precise on CES than on the benchmarks in
  [2].
- **Existing parsers:** error loops (`src/weave_agent_signals/scorers/efficiency.py:52-79`),
  repeated reads (`efficiency.py:90-127`) and command classes
  (`src/weave_agent_signals/scorers/outcome.py:307`) are already properties in all but name.
- **Gap:** the outcome scorer reads only `Bash` (`outcome.py:705`), so Codex checks are invisible to
  it (`docs/large-session-handling-research.md:430-437`). The event log must parse Codex programs.
- **Size of the job:** on the largest sessions, the claim proxy found 29 to 239 claims per thread,
  with 2 to 60 lacking any check (`docs/large-session-handling-research.md:199`). **Inference:**
  each thread's property candidates number in the dozens, few enough to confirm each with a model
  question.

### Effort and risks

- **Effort:** medium. An event-log builder on top of layer 0, a small property language, perhaps a
  library like the ones in [1][3], and about 10 to 20 rules.
- **Risks:**
  - Parser misses, since parsers favor precision over coverage (`specs/02-evaluation.md:34-36`), so
    a missed check reads as no check.
  - Rules about meaning need model-extracted predicates.
  - Planted or misleading text in tool output can fool provenance rules [2].
  - **Inference:** a rule can be over-strict where an anchor allows judgment, such as "one limited
    and clearly non-critical gap" (`rubrics.py:141-144`). So property results should be findings for
    the merge, not scores.

## 3. Task-derived criteria: requirement ledgers and tracing

### Mechanism

1. Extract the user's requirements, constraints and acceptance criteria from every user message
   and delegation prompt of the thread. Track how they evolve: later messages revise earlier ones.
2. Turn each item into a yes/no question, or a small tree of them.
3. Trace each item to evidence: edits, checks, and the final message.
4. Report which items were met, by whom, and after how many corrections.

### Measured evidence

- **Per-item rubrics beat reading the conversation.** In MultiChallenge, a judge reading the raw
  multi-turn conversation agreed with experienced raters 36% of the time. With one binary,
  instance-level rubric question per example it agreed 93% of the time. People wrote those
  questions, and each needed only the final response [11].
- **Requirements arrive in pieces.** Across 200,000+ simulated conversations, models did 39% worse
  on average when instructions were spread over turns. "When LLMs take a wrong turn in a
  conversation, they get lost and do not recover" [12].
- **Automatic extraction from real coding dialogues works passably.** Conv-to-Bench turned real
  multi-turn coding dialogues into requirement checklists. Its judge agreed with human-verified
  labels at κ 0.705, and extraction centered on instructions was more robust than extraction from
  the whole exchange [13]. Its Spearman correlation of up to 1.000 compares model rankings, not
  single items [13].
- **Rubric trees with critical nodes.** Mind2Web 2's task-specific judge agents used rubric trees of
  50 nodes on average and up to 603. Critical nodes fail their parent outright. Of 720 leaf checks,
  35 disagreed with a person, and on review 27 of those were the person's error [14].
- **Hierarchical rubrics, graded from artifacts.** PaperBench's rubrics hold 8,316 gradable leaf
  tasks, graded from the submission's files and logs. The best judge reached F1 0.83 against expert
  labels [15].
- **Generated rubrics fall short of human ones:**
  - RubricBench measured a 27% accuracy gap between model-generated and human rubrics [16].
  - Off-the-shelf models wrote rubrics poorly aligned with human ones [17].
  - Perturbations that degraded a response moved scores on generated and expert rubrics together;
    improvements did not transfer [18]. **Inference:** a generated requirement list is better at
    catching failures than at certifying full success.
  - AdaRubric generated task-specific rubrics for agent trajectories and reached Pearson 0.79 with
    people, with Krippendorff's α 0.83 [19].
- **Progress measures.** AgentBoard scores partial progress through subgoals instead of only final
  success [62].

### What it can and cannot judge

- **Outcome:** yes. The rubric asks whether "every material request and stated constraint" was met
  (`rubrics.py:347-349`).
- **Autonomy:** yes. Whether a correction repeats information given earlier is a lookup in the
  ledger (`rubrics.py:385-388`).
- **State consistency:** yes, as the source of constraints and decisions (`rubrics.py:264-268`).
- **Verification:** partly. The ledger says which claims are material.
- **Tool choice:** partly, as the stated task (`rubrics.py:214-217`).
- **Error recovery:** no.

### Novelty and fit

- **Novelty:** the design's plan Q already extracts "user requests and acceptance criteria" and asks
  one outcome question per request (`docs/small-model-judging-research.md:665-686`). What is new:
  one requirement ledger per thread for every plan, including W and R, with revisions tracked. The
  ledger goes into every window's pinned prefix and every merge.
- **Size:** user messages are 0.1% to 6.0% of each large session, and the 6.0% is mostly subagent
  task prompts (`docs/large-session-anatomy.md:67`). Per thread:
  - Main threads have 5 to 67 user messages (`docs/large-session-handling-research.md:197`), the
    largest 1,711 tokens (`:167`).
  - A subagent's task prompt has a median of 287 tokens and is at most 8.7k
    (`docs/large-session-anatomy.md:117`).

  - The largest user message in the Workflow session's subagents is 10,164 tokens
    (`docs/large-session-handling-research.md:167`).

  **Inference:** each thread's user messages fit one request for every catalog model of 32k or more,
  even in the 30M-token Workflow session. Its orchestrator has 13 user messages of at most 1,711
  tokens, so about 22k at most.
- **Workflow sessions:** a subagent's "user" is its delegation prompt, so its ledger is its task, as
  the design scores each subagent "on its own work"
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:150-155`).

### Effort and risks

- **Effort:** medium. One extraction request per thread per reviewer, trace questions per item,
  plus storage of the ledger as a content-authenticated artifact.
- **Risks:**
  - Extraction misses a requirement, and a missed item is never checked.
  - Revisions get merged wrongly.
  - Generated lists over-specify [16][17].
  - Requirements stated only in files the agent read, such as a spec document, are not in user
    messages. **Inference.**

## 4. Agentic and recursive judges (2025–2026)

### Mechanism

The transcript stays outside the model's context. The judge reads it through tools (read, search)
or through code that splits it and calls models on the pieces, as Recursive Language Models do.
Trained judges for long trajectories are the other branch.

The companion catalog covers RLM, SAFARI, Agent-as-a-Judge and AJ-Bench
(`docs/small-context-judging-techniques.md:245-312`,
`docs/long-session-judging-research.md:469-525`). New evidence:

### Measured evidence

- **Tools help frontier judges and can hurt open ones.** AgentHorizon has 1,373 long computer-use
  tasks, built from 166 hours of recordings. Judges either received the whole trajectory or worked
  as coding agents in five harnesses [28]:
  - GPT-5.5 in Codex was best at 80.9% balanced accuracy, ahead of Gemini 3.1 Pro (77.7%) and Claude
    Opus 4.7 in Claude Code (76.0%).
  - Qwen 3.6 27B scored 77.6% reading directly but 70.4% as an agent.
- **Better program selection for recursive readers.** Choosing among candidate reading programs by
  self-consistency, reasoning length and stated confidence beat RLM by up to 22% at the same time
  budget [29].
- **Trained long-trajectory judges:**
  - SWE-Critix (above) [30].
  - OS-Shepherd judges at 9B and 35B matched commercial judges at 30–60× lower cost on computer-use
    trajectories. The benchmark found a shared "leniency bias" that labels failed runs as successes
    [31].
  - A judge given a reusable "JudgeSkill", verification know-how refined on its own misjudged cases,
    gained 14.8 points of accuracy [32].

### What it can and cannot judge

Any rubric in principle. The strength is search across distance: a claim at turn 12 and a test at
turn 64. The weakness is that its reading path does not guarantee complete coverage, the design's
reason to keep it as an escalation and audit judge
(`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:185-191`).

### Novelty and fit

- **Novelty:** the design already plans a read-only Claude Code escalation experiment. The new
  evidence narrows it: only frontier models should judge as agents [28]; open-weight models should
  get requests built by code.
- **Fit:** the event log and requirement ledger make good tool targets. A query such as "list checks
  after event e-415" returns pinned facts instead of text. **Inference.**
- **Training:** a trained judge would need labels Monopole does not have
  (`specs/02-evaluation.md:398-402`).

### Effort and risks

- **Effort:** as planned in the design.
- **Risks:** leniency [31], run-to-run variance and long cost tails
  (`docs/small-context-judging-techniques.md:265-266`), and untrusted transcript text reaching a
  model with tools.

## 5. Learned compression and memory

### Mechanism

Train a model to shrink the input, then judge the smaller version. Three families:

- **Soft tokens:** gist tokens, in-context autoencoders.
- **Hard token deletion:** a small classifier drops tokens before a hosted model sees the text.
- **Key-value cache methods:** the internal cache of a self-hosted model is evicted or trained
  offline.

Trained memory agents, such as MemAgent, are in the companions
(`docs/small-context-judging-techniques.md:193-243`).

### Measured evidence

| Method | What it needs | Measured result |
| --- | --- | --- |
| Gist tokens [33] | fine-tuned model | up to 26× prompt compression, up to 40% fewer FLOPs, minimal quality loss |
| In-context autoencoder [34] | about 1% extra parameters on Llama | 4× context compression |
| LLMLingua-2 [35] | small encoder classifier; text out, so any API | 2× to 5× compression; 1.6× to 2.9× faster end to end |
| LongLLMLingua [36] | question-aware deletion | up to +21.4% on NaturalQuestions with about 4× fewer tokens |
| Comparison study [37] | — | extractive compression often beat token pruning and summaries; up to 10× with little loss |
| Cartridges [38] | train a key-value cache per corpus, self-hosted | matched in-context reading at 38.6× less memory; effective context from 128k to 484k |
| KVzip [39] | self-hosted cache eviction | 3× to 4× smaller cache, about 2× faster decoding, up to 170K tokens |

### What it can and cannot judge

None of these papers measures judging agent transcripts. **Inference:** deletion is riskiest for
verification and error recovery, whose evidence is often one diagnostic line inside long output.
Monopole's compaction deliberately keeps such lines (`specs/02-evaluation.md:91-94`).

### Novelty and fit

- **New to the series:** only LLMLingua-2-style extractive deletion works through the routes the
  user chose: CoreWeave, OpenAI and the Claude, Codex and Antigravity CLIs
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:258-264`).
- **Self-hosted methods:** soft tokens and key-value methods need a self-hosted model, which those
  routes exclude.
- **Duplicates:** the anatomy found most of the Workflow session's bulk is repeated output across
  subagents (`docs/large-session-anatomy.md:24-27`). Pointers for repeats already remove that
  without a model (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:205-206`).

### Effort and risks

- **Effort:** large for little expected gain.
- **Risks:** deletion inside quoted evidence breaks the host's exact-quote check
  (`specs/02-evaluation.md:174-175`). A classifier trained on prose may drop the one failing line.
  **Recommendation: do not adopt.**

## 6. Relative evaluation

### Mechanism

Compare instead of scoring. There are three ways for Monopole:

- **Session against session:** which session handled verification better.
- **Candidate verdicts:** which of two reviewers' verdicts the cited evidence supports better.
- **Anchored comparison:** compare a session with labeled reference sessions to place it on the
  0-to-1 scale.

### Measured evidence

- **Comparisons can beat scores for smaller judges.** Comparative assessment beat absolute scoring
  for moderate open models, with strong position bias [42]. A search that ranks by pairwise
  comparisons beat direct scoring on long-form evaluation [41].
- **Comparisons are easier to game.** With distractor features added, pairwise preferences flipped
  in about 35% of cases, against 9% for absolute scores [40].
- **Comparisons are often not transitive.** Aggregate cycle rates looked low, at 0.8–4.1%, yet
  33–67% of documents had at least one three-way cycle [44].
- **Same-outcome trajectories.** Pairwise comparison of debugging behavior among SWE trajectories
  with the same test outcome produced useful task-local scores for training [43].
- **Two answers argued side by side help a weaker judge.** A non-expert model picked the true answer
  76% of the time when two experts debated, against 48% without them; people went from 60% to 88%
  [46].

### What it can and cannot judge

- **Session against session:** **Inference:** impractical. Two 30M-token sessions cannot share one
  request, and pairs need a common task.
- **Candidate verdicts:** fit every rubric. The adjudicator reads two verdicts, the steps each
  cites, and the property findings. That is a short input, close to the debate setting in [46].

### Novelty and fit

- **Novelty:** the design re-judges flagged items with a larger model
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:116`). Pairwise adjudication of
  the two most distant verdicts is a new format for that step.
- **Anchored comparison** against the user's labeled sessions would calibrate the scale. It is
  unmeasured for transcripts and needs labeled sessions of the same kind. **Inference.**

### Effort and risks

- **Effort:** small; one request kind.
- **Risks:** position bias, so run both orders [42]; distractors [40]; non-transitive results [44].

## 7. Multi-judge designs

### Mechanism

Several judges per item: independent panels pooled by vote or mean, debate between judges,
role-specialized judges (one per rubric, sharing a ledger), or a critic revising a judge's draft.

### Measured evidence

- **Voting carries most of the gain.** Across seven benchmarks, majority voting explained most of
  multi-agent debate's gains. Debate alone did not raise expected correctness unless biased toward
  correction [45].
- **Consensus can bias scores.** On subjective rubric scoring, a single judge aligned best with
  people across six models. Consensus debate made it worse, mostly because a "strict" role pulled
  scores down and the consensus kept the bias [47].
- **Gains come from evidence, not roles.** A courtroom-style debate beat standard debate by 10
  points on claim verification, and 7.5 of them came from retrieving more evidence during the debate
  [48].
- **Diversity helps less than it seems.** Errors correlate across strong models
  (`docs/small-context-judging-techniques.md:377-381`). Small judges agree poorly with one another
  (`:382-384`).

### What it can and cannot judge

All rubrics. The design already gives each rubric its own request, which is role specialization by
rubric, and pools panels by mean (`specs/02-evaluation.md:280-286`).

### Novelty and fit

- **Novelty:** the new evidence argues against adding consensus debate and for keeping judges
  independent. It supports using a debate-like format only to adjudicate disagreements (section 6),
  with retrieval of cited steps as its main ingredient [48].
- **Judge skills:** a written judge skill refined on misjudged cases [32] is a cheap critic.
  **Inference:** it maps onto the design's rubric text, and changing it would need a new rubric
  version.

### Effort and risks

- **Effort:** none beyond section 6.
- **Risks:** downward bias from asymmetric roles [47], and correlated errors.

## 8. Uncertainty: calibrated, conformal and Bayesian judges

### Mechanism

- **Conformal prediction** uses a small labeled set to turn any score into a set of anchors that
  contains the true label at a chosen rate.
- **Abstention** fires when the set is too wide.
- **Bayesian aggregation** treats each partial verdict as noisy evidence with an estimated error
  rate, as the Dawid–Skene model does for annotators.
- **Prediction-powered inference** corrects an average computed from judge scores using a few human
  labels, with valid intervals.

### Measured evidence

- **Intervals from one judge run.** Conformal prediction produced intervals with coverage guarantees
  for judge scores from one run, with an adjustment for discrete scales [49]. Pooling intervals
  across several judges was more stable [50]. Backing off to less specific outputs gave 80–90%
  correctness guarantees with few labels [51].
- **Width predicts disagreement with people.** The width of conformal prediction sets correlated
  with judge–human disagreement at Spearman 0.576 over 1,918 items. Widths agreed across judges at r
  0.32 to 0.38, so width measured item difficulty more than judge noise [44].
- **Bayesian correction of judges.** Bayesian Dawid–Skene and Bayesian win-rate sampling corrected
  LLM evaluators' bias with sparse or no human labels [52].
- **Few labels, valid intervals.** Prediction-powered inference gives valid confidence intervals
  from a few labels plus many predictions [53]. Labeling where the model is unsure gives the same
  accuracy with far fewer labels [54].
- **Guarantees need many labels.** Trust or Escalate needed about 500 labels for a human-agreement
  guarantee (`docs/small-model-judging-research.md:500-510`).

### What it can and cannot judge

All rubrics, as a layer over verdicts.

**Inference:** with 20–40 labels Monopole can estimate interval widths and correct aggregate means,
but cannot certify coverage per rubric. Thirty labels give a 90% binomial upper bound of 21% after 3
misses (`docs/small-model-judging-research.md:521-525`).

### Novelty and fit

- **Novelty:** the design reads labels with a sign test and flags disagreement
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:242-245`). What is new:
  - per-verdict sets from conformal calibration;
  - prediction-powered averages in analysis;
  - Bayesian pooling of panel members by estimated reliability instead of the plain mean
    (`specs/02-evaluation.md:185-187`).
- **Partial verdicts:** **Inference:** pooling window findings into a session verdict with a
  Bayesian model is not supported by any source for rubric anchors that depend on recovery, such as
  error recovery (`rubrics.py:179-199`). Keep the model merge and use Bayesian pooling only across
  reviewers.

### Effort and risks

- **Effort:** small for the analysis side; medium to store sets with Scores.
- **Risks:**
  - Labels from one user are one labeler's view.
  - Coverage holds only for sessions like the labeled ones.
  - Changing the panel's pooling from the mean changes a pinned product rule, so it needs the user's
    decision.

## 9. Counterfactual and attribution checks

### Mechanism

Remove or perturb parts of the judge's input and watch the verdict:

- **Leave one window out:** re-merge without one window's findings.
- **Move the edges:** shift window edges by half a window.
- **Uncut the evidence:** re-run a pivotal window with uncut tool output.
- **Attribute:** fit which inputs drive the verdict.

Agent-side counterfactuals, which re-run the agent from a changed step, need the live environment.

### Measured evidence

- **A few ablations are enough.** ContextCite ablates random parts of the context and fits a linear
  model of the response's likelihood. A small number of extra passes faithfully identified the parts
  responsible for a statement [55]. It needs the response's token probabilities.
- **Leave-one-out can be cheap.** AttriBoT approximated leave-one-out attribution over 300× faster,
  using smaller proxy models [56].
- **Agent counterfactuals need a recorded environment.** Reverting and replaying agent runs from
  changed steps improved repair on Terminal-Bench 2.0. That required a reversible record of the
  environment captured at run time [57]. Failure-attribution training data came from counterfactual
  replay [9] and repeated runs [10].
- **Window edges are unmeasured.** No source measures how window edges change a judge's verdict
  (`docs/windowed-judging-research.md:467-468`).

### What it can and cannot judge

All rubrics, as an audit of the verdict. Recorded CES sessions cannot be re-run, so agent-side
counterfactuals are out of reach (section 1). Judge-side counterfactuals are not.

### Novelty and fit

- **Novelty:** the design flags large gaps between a session verdict and its parts, and panel
  disagreement (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:116`). A stability
  audit asks a different question: would the verdict change if the session had been cut differently,
  or if one window had been read differently?
- **Fit:** windows and merges are separate, request-keyed calls (`specs/02-evaluation.md:320-325`),
  so a re-merge that leaves one window out is one more request with known inputs. **Inference.**
- **Token probabilities:** ContextCite-style attribution needs them, which not every route returns
  (`docs/small-model-judging-research.md:969-971`). Leave-one-out re-merges do not.

### Effort and risks

- **Effort:** medium. W extra merges per rubric for a thread in W windows; cost is not a concern.
- **Risks:** a pivotal window is expected when one window holds the decisive event. **Inference:**
  the audit should flag only flips that rest on compacted, truncated or low-confidence evidence, or
  flips under edge shifts.

## 10. Summaries that cannot hide failures

### Mechanism

Wherever a summary, digest or verdict summary stands in for raw steps:

- **Pointers:** every statement carries pointers to the evidence IDs it summarizes.
- **Faithfulness check:** a separate check confirms that each pointer supports its statement.
- **Coverage check:** code confirms that every failure signal is mentioned or explicitly dismissed.

### Measured evidence

- **Summaries hide failure signals.** Summarizing old observations made software agents'
  trajectories longer than hiding them, which the authors read as summaries hiding "failure signals"
  (`docs/large-session-handling-research.md:322-331`).
- **Omission is the main failure.** Telling a summarizer which utterances it had omitted gave large
  quality gains [59].
- **Correct citations are not always faithful.** Up to 57% of correct-looking citations were not
  what the model actually relied on [58].
- **Long-input raters are weak.** No LLM rater agreed strongly with people on book-length summaries
  (`docs/long-session-judging-research.md:187-189`), and citing summaries scored under 20% on
  SummHay (`:445-447`).
- **Claim-to-evidence graphs** give reviewers a path from a conclusion to its checks [5].

### What it can and cannot judge

It protects verification and error recovery most, since their evidence is failures. **Inference.**

### Novelty and fit

- **Novelty:** the design keeps summaries out of the citation path and has merges cite findings with
  evidence IDs (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:284-286`). It does
  not require the verdict to account for every failure signal.
- **Coverage duty, made concrete:** the host lists the code-found failure signals in each window's
  core, such as failed checks, errors, denials and corrections, ranked by property severity. Each
  window returns a finding citing the signal, or a dismissal with a reason. The merge's citations
  must reach every signal marked critical, such as an unresolved failure at a claim.
- **Same pattern for verdict summaries:** each subagent's verdict summary at its launch call
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:149`) must carry the IDs of its
  critical signals.

### Effort and risks

- **Effort:** small. A list in each window prompt, a field in findings, and a host check.
- **Risks:**
  - Judges may dismiss signals lazily. Dismissal reasons are auditable, and planted probes measure
    this.
  - Signals the parsers miss are not listed (`specs/02-evaluation.md:34-36`).

## 11. Other credible ideas

- **Read the agent's reasoning, carefully.**
  - A monitor that also read the agent's reasoning caught 95% of systemic test hacks, against 60%
    for one reading only actions and outputs [26].
  - CES stores an assistant message's `reasoning`
    (`weave-agent-adapter/specs/EVALUATIONS_CASE_CONTRACT.md:34`), but `reasoning` does not occur in
    `src/weave_agent_signals/ces_client.py`, so judges never see it. **Confirmed by search.**
  - Counterpoint: rewriting an agent's reasoning while keeping its actions raised judges' false
    positives by up to 90% (`docs/large-session-handling-research.md:361-362`).
  - **Inference:** reasoning may help detect gaming for verification, but it should never count as a
    check.
- **Choose which cells the user labels.** Active testing picks the test points to label and removes
  the bias that choosing introduces [61]. Active inference keeps intervals valid when labels go
  where the model is unsure [54]. A curated subset of 100 items estimated a model's score on a
  14k-item benchmark [60]. **Inference:** pick the user's 20–40 cells with recorded inclusion
  probabilities, so the estimates in section 8 stay unbiased.
- **Checklists catch failures, not success.** Generated rubrics transfer degradations but not
  improvements [18]. **Inference:** property violations and unmet requirements should pull scores
  down with evidence. A clean result should not push a score to 1 on its own; the merge still needs
  positive evidence (`rubrics.py:145-148`).

## Recommended whole-conversation architecture

### What the user decided and what this adds

Decided by the user (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:247-271`):

1. Judges see every message; Workflow sessions get session, orchestrator and subagent scores.
2. Any model from the existing routes can judge; cost is not a concern; quality and model choice are.
3. Partial work is kept, and reruns skip finished work.
4. A tool-using judge is an experiment that never writes official Scores.
5. The user labels 20–40 cells.

The design's non-goals also hold: no CES schema change (`:52-55`).

Added here, each open to change:

- A. An event log per thread and a versioned set of rubric properties checked by code.
- B. A requirement ledger per thread, for every reading plan.
- C. A coverage duty for failure signals in windows and merges.
- D. A stability audit per verdict.
- E. Pairwise adjudication as the escalation format.
- F. Conformal sets and prediction-powered estimates, with actively chosen labels.
- G. Execution grounding later, only after a separately scoped adapter change.

### Steps

The approved layers stay
(`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:107-116`). New or changed steps
are marked.

0. **Facts (layer 0, extended).** Build the thread tree and steps as planned. Add the typed event
   log: user requests, corrections, steering, denials, edits with files, checks with results,
   errors, delegations and compactions. Parse Codex programs for commands and patches.
1. **Requirement ledger (new, one request per thread per reviewer).** Read all user messages of the
   thread, or the subagent's delegation prompt and follow-ups, plus the final assistant message.
   Return cited requirements, constraints and acceptance criteria, each with its revisions. Each
   reviewer writes its own, keeping panels independent as the windowed study proposed
   (`docs/windowed-judging-research.md:548-549`).
2. **Property checks (new, code).** Run the versioned rubric properties over the event log. A model
   question settles each predicate that needs meaning, such as "is this claim material" or "is this
   check relevant", within B_q. Results are pinned findings with evidence IDs and positions.
3. **Reading plan W, R or Q (layers 1–3, changed).** As approved. Every window's pinned prefix now
   carries the requirement ledger, open property findings, and the failure signals in its core with
   the coverage duty.
4. **Merge (layer 4, changed).** The merge reads findings, property findings, the requirement trace
   table and the coverage manifest. The host checks that the verdict cites or dismisses every
   critical signal and re-asks once if not. Audit roll-ups stay hidden from the merge, as designed
   (`docs/windowed-judging-research.md:678-682`).
5. **Stability audit (new).**
   - For R and Q threads: leave-one-window-out re-merges per rubric, plus one edge-shifted re-run
     for verdicts whose pivotal window rests on compacted or low-confidence evidence.
   - For W threads: agreement across panel members, or three samples.
   - A verdict is flagged when its anchor contradicts a critical property violation, or when it
     moves by 0.5 or more under the audit.
6. **Escalation (layer 5, changed format).** Flagged items go to pairwise adjudication by a panel
   model the user included. The adjudicator sees the two most distant verdicts, the raw steps they
   cite, and the property findings, in both orders. The tool-using judge stays an experiment.
7. **Reporting (new, offline).** Store property counts, stability flags and, once calibrated,
   conformal sets in Score details. Analysis reports prediction-powered averages per rubric.
8. **Later:** if the adapter records the commit and a bounded diff at each turn's end, add
   requirement-grounded diff checks and sandbox replay of claimed checks in the existing microVM
   runner (`specs/04-evaluation-runs.md:237-249`).

### What this changes in the approved design

- **New pinned inputs:** event-log schema version, property set version, requirement-ledger
  artifacts, coverage manifests and stability results. They join the pinned plan, so authentication
  can recompute them (`specs/02-evaluation.md:40-48`).
- **New request kinds:** requirement extraction, predicate questions, re-merges and adjudication.
  Coverage answers ride inside existing window requests.
- **Unchanged:** the verdict contract and abstention (`specs/02-evaluation.md:150-166`), reading
  plans and budgets, Workflow scoring, request-keyed resume, providers, and the CES schema.
- **Pieces** (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:193-214`):
  - The event log and properties extend piece 3.
  - The ledger and the coverage duty go into pieces 5 and 6.
  - The stability audit and adjudication extend piece 10's escalation.
  - Calibration joins piece 9's harness.

Cost, **Inference** from the anatomy's counts for the 605-subagent session:

- **Requirement ledgers:** 606 per reviewer, against about 3,084 requests for the 1M plan
  (`docs/large-session-handling-research.md:424`).
- **Re-merges:** for the orchestrator's 9 stretch windows, 36.
- **Whole subagents (plan W):** no extra requests with a panel of two or more. With one judge, two
  more samples per thread and rubric add about 6,050 requests for 605 subagents and 5 rubrics.
- **Property checks:** no model calls beyond predicate questions.

## Experiments on the user's sessions

All run on offline copies; nothing is written to CES. They extend the design's validation list
(`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:226-245`).

1. **Property census, no model calls.** Run the properties on the four largest sessions and about
   200 recent ones. Report violations per rubric and per thousand steps, and how many need a
   predicate question.
2. **Planted probes for properties and coverage.** In threads of every size, plant three probes at
   the start, middle and end:
   - a failed check right before a claim;
   - a repeated failing command;
   - an edit to a file that a user message forbade.

   Measure property detection; verdict detection with and without the coverage duty; and the
   silent-drop rate, meaning signals neither cited nor dismissed.
3. **Requirement-ledger quality.** On 5 sessions, the user reviews each thread's extracted list
   against its user messages, which are short. Report recall and precision, and plant a late
   constraint that revises an early one.
4. **Whole versus pieces, with structure.** On threads that fit a frontier model whole, take the
   whole-thread verdict as the reference: the 605 subagents and the stretches of 520k or less
   (`docs/large-session-anatomy.md:20-23`, `:108-115`). Run plans R and Q with and without the
   ledger and property findings. Report exact agreement, agreement within 0.25, ordinal
   Krippendorff's α, and planted detection 0, 1 and 5+ windows apart.
5. **Does the stability flag predict errors?** In experiment 4, compare unstable and stable
   verdicts' disagreement with the reference. Report the AUROC of the flag as an error predictor.
   Keep the audit only if flagged verdicts are clearly worse.
6. **Adjudication versus re-judging.** On disagreement cells, compare pairwise adjudication, mean
   pooling and re-judging by the largest model against the user's labels.
7. **Execution-grounding census, no model calls.** For each recent session, check whether `cwd`
   exists locally and whether git history covers the session's time window on its `git_branch`.
   This decides whether the adapter change is worth proposing.
8. **The 20–40 labels.**
   - **Selection:** choose cells by active selection with recorded probabilities. Weight toward
     disagreement, instability and property–verdict conflict, and include a few random cells
     [54][61].
   - **Blind review:** label blind to arm, and relabel 5 a week later.
   - **Reading:** a sign test for arm comparisons, where 21 of 30 wins are needed
     (`docs/small-model-judging-research.md:914-917`). Prediction-powered averages per rubric [53].
     Conformal sets pooled across rubrics if per-rubric counts are too small [49].
   - **Limit:** these labels can show large gaps and set thresholds, not certify agreement.

## Open questions

1. Should the adapter record the commit, dirty state and a bounded diff at each turn's end, as a
   separate project, so execution grounding becomes possible?
2. Should property findings be visible to the merge as evidence, as proposed, or hidden like roll-ups?
3. May judges read the agent's `reasoning` field as context, never as a check?
4. How should the 20–40 labels split between property precision and verdict disagreements, and is
   active selection with recorded probabilities acceptable?
5. Pairwise adjudication, or re-judging by the largest model, as the escalation format?
6. Should Bayesian pooling by reviewer reliability replace the mean once labels exist?
7. Is the stability audit's extra merge per window per rubric acceptable at the Workflow session's
   scale?

## Observations outside the question

1. **Confirmed:** CES stores assistant `reasoning`
   (`weave-agent-adapter/specs/EVALUATIONS_CASE_CONTRACT.md:34`), and Monopole's reader never reads
   it (no occurrence in `src/weave_agent_signals/ces_client.py`).
2. **Confirmed:** CES has no commit identity, diff or snapshot
   (`weave-agent-adapter/specs/EVALUATIONS_CASE_CONTRACT.md:26-27`), so no recorded session can be
   re-executed today.
3. **Confirmed:** spec line numbers cited by earlier documents have moved. "No human-labeled
   validation set" is now `specs/02-evaluation.md:398-402`.

## Sources

All accessed 2026-11-04. arXiv details were read from each paper's abstract page and HTML version.

1. Xiao and Nuzzo, Symbolic Temporal Supervision of LLM Agents Using Contracts (ContrAgent). https://arxiv.org/abs/2609.18128
2. Kekatos et al., Formal Runtime Verification for Tool-Using LLM Agents: An Offline Same-Benchmark Study on AgentDojo and STAC. https://arxiv.org/abs/2610.09793
3. Elkoussy and Perez, AgentLTL: A Trace-Verification Framework for Measuring, Enforcing, and Training Procedural Compliance in Tool-Using LLM Agents. https://arxiv.org/abs/2607.02599
4. Sharma, Barke, Zorn, Willful Disobedience: Automatically Detecting Failures in Agentic Traces (AgentPex). https://arxiv.org/abs/2603.23806
5. Kim, Miao, Liu, LEDGER: Claim-to-Evidence Trace Graphs for Auditing LLM Agents. https://arxiv.org/abs/2608.18398
6. He et al., ProcCtrlBench: Evaluating Process-Level Defects and Control Preservation in LLM Coding Agents (v1 titled ProcBench). https://arxiv.org/abs/2605.20251
7. Agent Behavior Mining: Generative AI Agent Governance in Business Processes. https://arxiv.org/abs/2606.20669
8. GraphTracer: Graph-Guided Failure Tracing in LLM Agents for Robust Multi-Turn Deep Search. https://arxiv.org/abs/2510.10581
9. AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems? https://arxiv.org/abs/2509.03312
10. Who is Introducing the Failure? Automatically Attributing Failures of Multi-Agent Systems via Spectrum Analysis (FAMAS). https://arxiv.org/abs/2509.13782
11. Sirdeshmukh et al., MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs. https://arxiv.org/abs/2501.17399
12. Laban et al., LLMs Get Lost In Multi-Turn Conversation. https://arxiv.org/abs/2505.06120
13. dos Santos et al., Conv-to-Bench: Evaluating Language Models Via User-Assistant Dialogues In Code Tasks. https://arxiv.org/abs/2605.26440
14. Gou et al., Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge. https://arxiv.org/abs/2506.21506
15. Starace et al., PaperBench: Evaluating AI's Ability to Replicate AI Research. https://arxiv.org/abs/2504.01848
16. RubricBench: Aligning Model-Generated Rubrics with Human Standards. https://arxiv.org/abs/2603.01562
17. Dhole and Agichtein, RubricRAG: Towards Interpretable and Reliable LLM Evaluation via Domain Knowledge Retrieval for Rubric Generation. https://arxiv.org/abs/2603.20882
18. Evaluating Rubric Generation with Interventional Transfer. https://arxiv.org/abs/2610.10809
19. AdaRubric: Task-Adaptive Rubrics for Reliable LLM Agent Evaluation and Reward Learning. https://arxiv.org/abs/2603.21362
20. Raghavendra et al., Agentic Rubrics as Contextual Verifiers for SWE Agents. https://arxiv.org/abs/2601.04171
21. SWE-RM: Execution-Free Feedback for Software Engineering Agents. https://arxiv.org/abs/2512.21919
22. Zhong, Raghunathan, Carlini, ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases. https://arxiv.org/abs/2510.20270
23. Yao et al., TestJack: Should you trust the results in coding benchmarks? Agentic Coding Benchmarks Auditing via Evaluator Evolution. https://arxiv.org/abs/2610.10619
24. SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark. https://arxiv.org/abs/2603.00520
25. Wu et al., SWE-Together: Evaluating Coding Agents in Interactive User Sessions. https://arxiv.org/abs/2606.29957
26. Baker et al., Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation. https://arxiv.org/abs/2503.11926
27. Wang et al., The Verification Horizon: No Silver Bullet for Coding Agent Rewards. https://arxiv.org/abs/2606.26300
28. AgentHorizon: Evaluating Agentic Judges for Long-Horizon Computer-Use Tasks. https://arxiv.org/abs/2610.11050
29. Alizadeh et al., Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context. https://arxiv.org/abs/2603.15653
30. SWE-Critix: Post-Training Generalist Verifiers for Assessing Coding-Agent Trajectories. https://arxiv.org/abs/2610.13356
31. OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models. https://arxiv.org/abs/2607.28609
32. SkillTV-Bench: Benchmarking How Well Judges Perform on Skill-Augmented Agentic Execution. https://arxiv.org/abs/2608.05573
33. Mu, Li, Goodman, Learning to Compress Prompts with Gist Tokens. https://arxiv.org/abs/2304.08467
34. Ge et al., In-context Autoencoder for Context Compression in a Large Language Model. https://arxiv.org/abs/2307.06945
35. Pan et al., LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. https://arxiv.org/abs/2403.12968
36. Jiang et al., LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression. https://arxiv.org/abs/2310.06839
37. Jha et al., Characterizing Prompt Compression Methods for Long Context Inference. https://arxiv.org/abs/2407.08892
38. Eyuboglu et al., Cartridges: Lightweight and general-purpose long context representations via self-study. https://arxiv.org/abs/2506.06266
39. Kim et al., KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction. https://arxiv.org/abs/2505.23416
40. Tripathi et al., Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation. https://arxiv.org/abs/2504.14716
41. Liu et al., Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators (PairS). https://arxiv.org/abs/2403.16950
42. Liusie, Manakul, Gales, LLM Comparative Assessment: Zero-shot NLG Evaluation through Pairwise Comparisons using Large Language Models. https://arxiv.org/abs/2307.07889
43. Beyond Binary Verifiable Rewards: Behavioral Self-Rewarding for SWE Agents. https://arxiv.org/abs/2610.12598
44. Diagnosing LLM Judge Reliability: Conformal Prediction Sets and Transitivity Violations. https://arxiv.org/abs/2604.15302
45. Choi, Zhu, Li, Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models? https://arxiv.org/abs/2508.17536
46. Khan et al., Debating with More Persuasive LLMs Leads to More Truthful Answers. https://arxiv.org/abs/2402.06782
47. Song et al., Beyond Consensus: Downward Bias and Role Asymmetry in Multi-Agent LLM Judges for Subjective Evaluation. https://arxiv.org/abs/2608.30373
48. Courtroom-Style Multi-Agent Debate with Progressive RAG and Role-Switching for Controversial Claim Verification (PROClaim). https://arxiv.org/abs/2603.28488
49. Sheng et al., Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction. https://arxiv.org/abs/2509.18658
50. Liu, Robust Conformal Consensus: Multi-Agent LLM-as-a-Judge Interval Evaluation with Conformal Prediction. https://arxiv.org/abs/2609.06367
51. Mohri and Hashimoto, Language Models with Conformal Factuality Guarantees. https://arxiv.org/abs/2402.10978
52. Gao et al., Bayesian Calibration of Win Rate Estimation with LLM Evaluators. https://arxiv.org/abs/2411.04424
53. Angelopoulos et al., Prediction-Powered Inference. https://arxiv.org/abs/2301.09633
54. Zrnic and Candès, Active Statistical Inference. https://arxiv.org/abs/2403.03208
55. Cohen-Wang et al., ContextCite: Attributing Model Generation to Context. https://arxiv.org/abs/2409.00729
56. AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution. https://arxiv.org/abs/2411.15102
57. Shepherd: Enabling Programmable Meta-Agents via Reversible Agentic Execution Traces. https://arxiv.org/abs/2605.10913
58. Wallat et al., Correctness is not Faithfulness in RAG Attributions. https://arxiv.org/abs/2412.18004
59. Zou et al., Towards Understanding Omission in Dialogue Summarization. https://arxiv.org/abs/2211.07145
60. Maia Polo et al., tinyBenchmarks: evaluating LLMs with fewer examples. https://arxiv.org/abs/2402.14992
61. Kossen et al., Active Testing: Sample-Efficient Model Evaluation. https://arxiv.org/abs/2103.05331
62. Ma et al., AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents. https://arxiv.org/abs/2401.13178
