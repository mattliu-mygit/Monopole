# Judging long sessions in windows

Research notes on how Monopole could judge Claude Code and Codex sessions that
are larger than any judge's context window by reviewing them in windows, and
still produce a whole-session verdict that can be trusted. Observed 2026-11-03.
They are not Monopole product guarantees.

This is the companion to `docs/long-session-judging-research.md`, which covers
fitting a whole session into one long-context request: model windows, prices,
prompt caching, compression, and the evidence that quality falls as inputs
grow. That material is not repeated here. This document starts where a session
no longer fits, or where the choices below need more than one unit of judgment.

Every external claim cites a numbered source from the list at the end, and every
source was accessed on 2026-11-03. Repository claims cite `path:line`. Claims
about the adapter cite its contract, read from a local checkout of
`weave-agent-adapter` (`specs/EVALUATIONS_CASE_CONTRACT.md`). Text marked
**Inference** is reasoning from the cited facts, not something a source states.

Terms used throughout:

- A **thread** is one agent's own messages and tool calls in order. The
  orchestrator thread is the session's main-thread Cases, whose `agent_id` is
  null; each subagent run is its own thread, stored in CES as a child Trajectory
  (EVALUATIONS_CASE_CONTRACT.md:8-11, :27; `specs/01-ces-io.md:12-13`).
- A **step** is one user message, assistant message, tool call, or compaction
  Case.
- A **window** is the part of one thread that one judge request sees raw. Its
  **core** is the part that window is responsible for; anything else it shows is
  context.
- A **finding** is what a window returns today: a positive or negative
  observation that cites evidence IDs, with no score
  (`src/weave_agent_signals/judges/sliding_contracts.py:56-61`).
- The **merge** is the last request per rubric. It reads all findings and returns
  the anchored verdict (`src/weave_agent_signals/judges/sliding.py:93-107`,
  `sliding.py:971-996`).
- A **ledger** (proposed here) is a list of cited facts taken from each window:
  claims, checks, edits, errors, constraints, and corrections. Later requests
  look entries up by ID.

## Summary

1. Keep the map-then-merge pipeline, but cut windows at subagent, then turn,
   then step, so every message is read raw once and no turn is too big to judge.
2. Workflow sessions: review each subagent, then the orchestrator with each
   subagent's verdict summary at its launch call, then merge a session verdict.
   All three read one set of findings; a stored roll-up flags large gaps.
3. Swap free-text digests for a cited ledger plus a step that judges claim and
   check pairs across windows, only if planted probes show digests miss them.
4. Tradeoff: in the cost example, about 770 requests per reviewer for 120
   subagents (41 today, which reads no subagent work), and claim extraction can
   miss things; in exchange size stops causing skips.
5. No source compares windowed and whole-context rubric scores on agent
   sessions; set window sizes by shrinking windows on sessions that fit.

## What happens today when evidence crosses a window

The companion explains the size limits. These are the parts that matter for
evidence that spans windows:

- Windows are contiguous ranges of turns packed near a soft token target. Each
  raw window adds up to one neighboring turn before and after its core, and a
  turn is never split (`src/weave_agent_signals/judges/windowing.py:211-253`,
  `specs/02-evaluation.md:64-70`). A turn larger than the raw budget skips that
  reviewer for the session (`windowing.py:220-221`, `windowing.py:356-357`).
- One rubric-neutral free-text digest of about 1,000 tokens summarizes each
  chunk (`sliding.py:70-75`). Every window sees the digests of all other chunks,
  earlier and later (`sliding.py:846-857`). Evidence IDs from other chunks are
  replaced with `[citation omitted]` (`sliding.py:140-143`, `sliding.py:855`), so
  a window cannot point at evidence outside itself.
- A window returns findings, not a score (`sliding.py:77-87`). Two findings
  merge only when polarity, observation text, and cited IDs match exactly
  (`sliding_contracts.py:70-75`, `sliding.py:916-933`).
- The merge sees the coverage manifest, every digest, and every finding, but not
  the findings' evidence IDs or turn positions (`sliding.py:935-969`, with IDs
  removed at `sliding.py:943-946`).
- The merge input grows by 11,000 tokens per chunk, and a session needing more
  than 40 chunks is skipped (`windowing.py:367-368`, `windowing.py:373-377`,
  `src/weave_agent_signals/run_config.py:58-61`).
- The judge view keeps each turn's first user message and last assistant
  message (`src/weave_agent_signals/ces_client.py:229-234`, `ces_client.py:253`,
  `ces_client.py:299`), and shows a subagent only as its ID, an `agent_type`, and
  a count of omitted tool calls (`windowing.py:166-174`).
- The session reader keeps only a subagent's tool calls, drops its messages, and
  attaches nested subagents to the top-level turn that launched their ancestor
  (`ces_client.py:218-224`, `ces_client.py:242-252`). CES itself stores every
  subagent with all its member Cases and its own subagents
  (EVALUATIONS_CASE_CONTRACT.md:8-11).

Worked example of a link that depends on luck today. A session is planned into
windows of ten turns:

- Window 1, turn 8: tool call `t-301` runs `pytest tests/test_parser.py` and
  exits 1.
- Window 2, turn 11: tool call `t-415` edits `parser.py`. Turn 12: assistant
  message `m-430` says "Fixed; all tests pass." No test runs in turns 11 to 20.
- Window 7, turn 64: tool call `t-2210` runs `pytest` and exits 0.

Window 2 can report "claim with no check in this window", and window 7 can
report "tests pass". Whether anyone notices that the claim came before any
passing run after the edit depends on a digest happening to mention the order,
because the merge sees neither citations nor positions. The verification rubric
turns on exactly that order: checks count only when they "occurred after the work
they are offered to verify" (`src/weave_agent_signals/judges/rubrics.py:111-115`).

## Techniques

### 1. Map-reduce with one final merge

How it works. Independent requests read parts of the input and return
structured intermediate results. One final request reads all of them and answers.
Monopole's digest, window, and merge pipeline is a version of this.

Evidence:

- LLM×MapReduce names the two ways splitting breaks long-range information:
  facts that depend on another chunk, and chunks that conflict. Its map output
  has four fields: extracted information, rationale, answer, and a confidence
  score out of 5. A collapse step compresses results repeatedly until they fit,
  and the final step resolves conflicts using confidence scores calibrated by
  in-prompt principles and examples [10].
- LongReward judges answers about long documents with an off-the-shelf LLM. For
  completeness it splits the document into 4,096-token chunks, extracts the
  relevant information from each, then judges once. For faithfulness it splits
  the answer into sentence-level statements, retrieves the top 128-token chunks
  for each, scores each statement 1, 0.5, or 0 for full, partial, or no support,
  and averages. On 464 human-labeled answer pairs it matched people 66.2% of the
  time, against 57.1% for asking an LLM to compare the two answers with the
  document in context [11].
- On TRAIL, judging each span separately plus an agent-level pass over each
  agent's descendant spans let the same GPT-5.4 find the error location 2.8
  times (GAIA) and 12 times (SWE-bench) as often as a single request over the
  whole trace. Single-request baselines also had to drop 6% to 9% of GAIA traces
  and 22% to 29% of SWE-bench traces for length [4].
- TRAIL's SWE-bench traces, from a single coding agent, average about 617,000 to
  728,000 tokens depending on the tokenizer, and the longest exceed 2 million.
  Every judge metric fell as input grew; for locating the error, Spearman
  correlation with input length was −0.508 [5].
- GUIDE diagnoses each subtask segment of a GUI agent trajectory and has a model
  merge the diagnoses. It scored 95.80% accuracy on 932 trajectories, against
  85.61% for one request over the whole trajectory [3].
- None of these compares rubric scores from windows with rubric scores from the
  whole session on the same coding sessions.

Fit with Monopole:

- The pipeline exists and is pinned: per reviewer, one digest set shared by all
  rubrics, then one window request per rubric per window, then one merge per
  rubric (`sliding.py:813-832`, `sliding.py:998-1026`).
- Monopole already follows LLM×MapReduce's main lesson, structured map output,
  with citations the host checks. It lacks two things that source relies on:
  a confidence value in each finding, and anything telling the final request
  where a finding came from.

Cost. N digest requests, 6N window requests, and 6 merge requests per reviewer
for N chunks, as the companion counts.

Failure modes:

- A link between windows survives only if a window or a digest states it.
- The single flat merge has a hard ceiling (`windowing.py:373-377`).
- The merge cannot tell whether a check came after a claim.

### 2. Summary trees

How it works. Summarize small chunks, then summarize groups of those
summaries, until one summary remains. RAPTOR clusters chunks by similarity and
keeps every level so a question can retrieve from any level.

Evidence:

- In recursive book summarization, each chunk's task also received the earlier
  summaries at the same level, which helped. Supplying the earlier raw text
  instead did worse, because the model could not use that much context. About 5%
  of book summaries matched human quality [8].
- The same authors state the limit of the method: decomposition "assumes that
  separate parts of the task can be completed independently", which fails when
  "earlier details in the book are only later revealed to be important". A
  character asked for "their hand" for a dance was summarized as a marriage
  proposal, because the leaf task saw only local text [8].
- RAPTOR's tree of recursive summaries raised the best QuALITY score by 20
  points of absolute accuracy with GPT-4 [9]. That is retrieval for question
  answering, not judging.
- A hierarchical agent monitor splits a trajectory into chunks of N tool calls,
  summarizes and scores each, then summarizes and scores groups of N chunks,
  repeating until one result remains. Its best chunk size was 4 to 5 tool calls,
  and the paper reports lower AUC and true-positive rate for it on the
  computer-use dataset. The trajectories averaged over 20,000 tokens [1].

Fit with Monopole. A tree is needed only when the merge input no longer fits
or the session needs more than 40 chunks (`windowing.py:367-377`). There it
replaces the single flat merge with merges over groups of windows. It should not
replace raw review of every window.

Cost. About N/k extra requests per level when k windows form a group.

Failure modes:

- Detail is lost at every level, and an early error is copied upward.
- Something whose importance appears only later is the case the authors say the
  method misses [8].

### 3. Running notes carried across windows

How it works. Read windows in order and carry a bounded summary or evidence
list from each window into the next.

Evidence:

- A sequential monitor scores each chunk with the previous chunk's summary in
  context. It did better with larger chunks, and its true-positive rate on
  SHADE-Arena was "really bad". Averaging its suspicion score with the
  hierarchical monitor's, at equal weight, gave the best AUC at every
  monitor-awareness level on both datasets; one request over the full trajectory
  won true-positive rate on SHADE-Arena at two awareness levels, by small
  margins. Monitors built on weaker models such as GPT-4o-mini and Claude 3.6
  Haiku still reached over 0.85 AUC against all agents, and the averaged design
  beat the other designs by a substantial margin [1].
- TRACE argues that a compressed running summary "may lose weak or ambiguous
  signals that only become meaningful when connected with later evidence". Its
  monitor keeps a running list of findings and injects earlier windows' findings
  into later inspections. Recall rose to 0.844, against 0.580 for the sequential
  monitor and 0.405 for one full-trajectory request; F1 was 0.713, 0.700, and
  0.577. It used 12.3 model calls per trajectory against 18.8 for the sequential
  monitor, and its precision fell to 0.641 with Claude Sonnet 4.6 [2].
- In SKG-Eval's correlation tables, a GPT-4o judge that saw the conversation
  history matched human session ratings better than one that saw only the
  current turn, for example Spearman 0.69 against 0.63 on MT-Bench [22].
- The companion covers Chain of Agents, MemAgent, ReadAgent, and BooookScore.

Fit with Monopole:

- The note would replace the digest step (`sliding.py:789-832`). Today's digest
  is free text with no citations (`specs/02-evaluation.md:144-145`).
- Reading strictly in order loses look-ahead, which the companion flagged. A
  two-pass variant keeps it: each window extracts its notes independently, in
  parallel like today's digests, and then every rubric window sees the whole
  thread's notes, earlier and later. Each request still resumes on its own
  (`sliding.py:413-443`).

Cost. Same request count as digests. Notes with citations are longer than a
1,000-token digest.

Failure modes:

- An entry the extractor misses is invisible to every later request.
- Compressing old notes drops weak signals, which is TRACE's objection to
  running summaries [2].

### 4. A claim ledger with a targeted second pass

How it works. List the checkable claims. For each claim, fetch the evidence
that could support or contradict it, judge each pair, and combine the results.
This is the break-and-check idea from factual-consistency scoring, applied to a
session instead of a document.

Evidence:

- AlignScore splits the source into chunks of about 350 tokens at sentence
  edges, scores each claim sentence against every chunk, keeps the best score
  per claim, then averages over claims. Averaging keeps one bad claim from
  dominating, and scoring against chunks beat scoring against the whole document
  [12].
- LongReward's faithfulness score does the same with statements and retrieved
  chunks, and adds partial support [11].
- How the sources measure misses: TRAIL scores whether the judge names the right
  span [5], Who&When scores the right agent and the right step [6], TRACE reports
  recall on sabotage that is only visible across distant steps [2], and FBI
  plants degradations and counts how often the evaluator's score fails to drop
  [14].

Fit with Monopole:

- Evidence IDs make the second pass exact: each step of a turn has its own ID
  (`windowing.py:179-185`), so the host can fetch the raw steps on both ends of
  a pair without searching.
- Most entries need no model. The outcome parser already recognizes test, build,
  lint, install, and Git commands and their exit codes
  (`specs/02-evaluation.md:17-21`). Tool calls carry `exit_code` and
  `permission_decision` (`src/weave_agent_signals/models.py:50-51`), and a user
  message typed mid-turn carries `steering: true`
  (EVALUATIONS_CASE_CONTRACT.md:28). A model is needed for claims in assistant
  messages and for constraints in user messages.

Cost. One extraction request per window, replacing the digest request, plus one
resolution request per batch of cross-window pairs.

Failure modes:

- A claim phrased indirectly, such as "should be good now", may not be
  extracted.
- A claim that needs two pieces of evidence from different windows cannot be
  settled by keeping the single best piece, which is how these metrics combine.
- A check the parser does not recognize never gets paired, because the parsers
  favor precision over coverage (`specs/02-evaluation.md:34-36`).

### 5. Per-window scores combined by a rule

How it works. Each window returns a score, and a fixed rule combines them:
minimum, mean, any, majority, or a weighted sum.

Evidence:

- Inspect Scout, an open-source transcript scanner, combines the parts of a
  too-long transcript with a default rule per answer type: any for booleans,
  mean for numbers, a model call for text, majority for labels, union for
  multi-labels, and the last result for structured answers [24].
- The best rule depends on what a part's score means. For a reward model whose
  step scores estimate the chance of finishing correctly, the last step's score
  beat the minimum and the product, while earlier work with right-or-wrong step
  labels found the minimum best [13].
- TRAIL's best framework fails a trace when any span fails. Its authors say this
  treats a single minor formatting slip like a critical error, and call for
  severity weighting [4].
- Replacing GUIDE's model merge with "fail if any subtask did not succeed"
  dropped accuracy from 95.80% to 87.41% and recall from 93.62% to 69.28%,
  because the rule cannot credit an agent that recovered from a failed subtask
  [3].

Fit with Monopole. Windows deliberately return findings instead of scores
(`sliding.py:77-87`), and several anchors describe the whole session, such as
whether the agent recovered after an error (`rubrics.py:179-199`). Keep that.
Rules still have two uses: pooling reviewers, which is already a mean
(`src/weave_agent_signals/judges/review.py:430-450`), and audit roll-ups
(see the proposed design).

Failure modes:

- The minimum punishes a failure the agent later fixed.
- The mean dilutes one critical failure among many quiet windows.
- Both ignore that windows hold very different amounts of work.

### 6. Splitting by agent and by subtask

How it works. Review each agent's thread separately, then judge the parent
over its own actions plus short results from its children. Optionally split a
thread further into subtasks or phases.

Evidence:

- Inspect Scout scans each agent span, the main thread and each subagent, as a
  separate transcript. Its documentation notes that the tokens these coding
  agents use "can often run into the tens of millions, requiring that scanning
  be chunked". A span larger than the model's context is split into
  segments sized to fit within 80% of it, each scanned with the same prompt and
  combined by the rules above. Results are not combined across spans, and
  single-turn helper agents are excluded as "utility spans" [24].
- Claude Code runs each subagent in its own context window, and the subagent
  "returns only the summary" to the main conversation. Subagents compact on
  their own, and their compaction events are logged in the subagent transcript
  files [25].
- In Who&When, reading the whole log named the responsible agent best (54.33%
  and 55.17% with GPT-4o and the ground truth available), and reading step by
  step found the decisive step best. Picking the agent from the whole log and
  then reading only that agent's steps found the step 12.28% of the time against
  4.39% for the whole log alone, at 149,177 tokens against 17,106. Accuracy fell
  for every method as logs grew, and at 93 to 130 steps the paper reports all
  three converging to near 0% [6].
- GUIDE's subtask segments are chosen by a model. An automatic rater judged
  99.4% of 3,300 segments usable, and that rater agreed with a human on 200
  segments at Cohen's κ = 0.89. Passing the full trajectory to the per-subtask
  prompt instead dropped accuracy to 74.14% [3].
- DynSTEER splits trajectories at key execution points. Each stage's evidence
  window starts at the last point every path must pass through, which leaves out
  work on parallel branches [20].
- Transect analyzed a multi-agent run of almost 13 million tokens. Judges proposed
  phase edges over batches of 40 turn digests, each batch given the previous
  phase label [19].

Fit with Monopole:

- This needs a reader that keeps the thread tree, every message, and nesting
  (`ces_client.py:190-310`, `models.py:66-69`).
- Scores for subagents need a target Case and Scorer names: a turn's identity is
  its first Case (`AGENTS.md:150-151`), and each rubric and panel has its own
  Scorer (`AGENTS.md:159-160`).

Cost. One review per subagent per reviewer.

Failure modes:

- Problems visible only across subagents, such as two subagents doing the same
  work, need the parent's view.
- Parallel subagents have no single order.
- Trivial helper subagents add requests that mostly abstain.

### 7. Adaptive inspection and tool-using judges

How it works. Find the regions that matter, then read those regions raw;
or give an agent read-only tools and let it choose what to read.

Evidence:

- TRACE's first step reads the complete trajectory and lists two kinds of
  window to inspect: single suspicious actions, and patterns across non-adjacent
  steps. Its second step runs actions such as inspecting a step or comparing two
  steps, with earlier findings injected into later inspections. A final step
  judges from all findings [2].
- Who&When's binary search asks which half of the log holds the error and
  recurses into that half. Its accuracy fell between whole-log and step-by-step
  reading [6].
- Recursive Language Models keep the input outside the prompt and let the model
  split it with code and call itself on the pieces. They handled inputs two
  orders of magnitude beyond the context window and, on GPT-5, beat compaction
  by a median 26% and Claude Code by 13% [7].
- The companion covers Agent-as-a-Judge, cost, and safety.

Fit with Monopole:

- TRACE's first step assumes the whole trajectory fits in one request, which is
  false for the sessions this document is about. A ledger can stand in for that
  full read.
- The best use here is narrow: settling the open pairs from the ledger, with
  read-only access to one file per window. An agent's reading path is not
  pinned, so it should not replace complete raw review.
- Claude is tool-disabled today, and Codex runs under a deny-by-default profile
  (`AGENTS.md:135-143`).

Failure modes:

- Without a full first read, the triage step has to trust the ledger.
- TRACE's recall gains cost precision [2].
- Runs vary, and the reading path cannot be replayed exactly.

### How the techniques compare

| Technique | What crosses windows | How the verdict forms | What is lost | Evidence against one full request |
| --- | --- | --- | --- | --- |
| 1. Map-reduce | structured partial results | one model merge | links no window stated | better on TRAIL, GUIDE, LongReward [3][4][11]; no rubric-score comparison |
| 2. Summary trees | summaries of summaries | top of the tree | detail per level; facts that matter later | mixed for monitoring [1]; book summaries rarely human-level [8] |
| 3. Running notes | a carried list or summary | last window or a final request | entries never written down | sequential monitor weak alone [1]; persistent evidence raised recall [2] |
| 4. Claim ledger | cited claims, checks, constraints | per-pair judgments feed the merge | claims not extracted; two-piece evidence | chunk-level checks beat document-level [12] |
| 5. Rule over window scores | scores | min, mean, any | recovery and severity | rule worse than a model merge [3] |
| 6. Split by agent | child verdict summaries | parent and session merges | cross-agent patterns, unless the parent looks | segmentation and agent-first narrowing helped [3][6] |
| 7. Adaptive inspection | a findings list | final synthesis | regions triage skipped | better recall, lower precision [2] |

### Approaches that avoid windows

Three published designs avoid splitting a trajectory, and each fits only part
of Monopole's problem:

- τ-bench scores an agent by comparing the database at the end of the
  conversation with an annotated goal, so its main score never reads the
  trajectory [17]. Monopole's deterministic outcome scorers play that role
  already (`specs/02-evaluation.md:17-21`); the model rubrics exist because
  process quality cannot be read from the final result.
- SWE-agent's authors had GPT-4o put each of 248 unresolved trajectories into
  one of 9 failure categories, reading each trajectory whole. On a hand-labeled
  validation set it agreed with the authors on 87% of instances [23]. That works
  when a trajectory fits in one request.
- LiteTrajEval marks failure signals with rules, compacts low-value content,
  serializes the trajectory under a fixed token budget, and asks one
  rubric-guided judge [21]. That is the companion's compression strategy, and it
  covers threads that fit after compaction.

## Where windows start and end

Evidence:

- For retrieval, semantic chunking's computational cost was "not justified by
  consistent performance gains" over fixed-size chunks [16].
- For long conversation memory, turn-level, session-level, and summary-based
  units each had limits, and topical segments did better [15]. Splitting
  sessions into finer units improved memory recall in LongMemEval [18].
- Agent judges cut along structure: tool calls [1], subtasks [3], agent spans
  then token segments [24], and execution stages [20].
- No source found here measures how window overlap or shifted window edges
  change a judge's verdict. That has to be measured.

What this means for Monopole (**Inference**):

- The harness already records the structure that GUIDE and DynSTEER have to
  infer: thread, turn, and step. Using it costs no model calls and keeps window
  plans deterministic, so plan IDs stay stable on resume.
- The token target should only decide how many whole steps fit in a window, never
  where a window starts.
- Today, windows may cite evidence from the overlap turns they share with their
  neighbors (`windowing.py:236-253`, `windowing.py:524-527`, `sliding.py:842-844`,
  `sliding.py:895`). An event in an overlap turn can therefore be reported by two
  windows in different words and survive the exact-match deduplication. Letting
  windows read overlap turns but cite only their core removes that double count.

Worked example. Take the 225,000-token turn with 285 tool calls that the
companion uses. Split at step edges with a 128,000-token raw target, it becomes
2 parts; for the 131,072-token default models with their 50,000-token target, it
becomes 5 parts. No reviewer is skipped, and each tool call is reviewed raw in
exactly one part.

## Combining scores across windows and agents

Within one thread, keep the model merge. A rule is worse when agents recover
from intermediate failures [3], and the rule that TRAIL's best framework uses
over-counts minor issues [4].

Across reviewers, keep the mean of anchors, with minimum, maximum, and spread
recorded (`review.py:430-450`, `specs/02-evaluation.md:159-163`).

Learned weights for combining scores, the "calibrated aggregation" in some
judge papers, need human labels. Monopole has no labeled set
(`specs/02-evaluation.md:364-366`), so that is out of reach for now.

Between the session, the orchestrator, and its subagents, a fixed rule gives the
wrong answer in a common case. Worked example for error recovery: subagent A2
repeats the same failing command 6 times, which matches the 0 anchor for repeating
"the same disproven action without diagnosis" (`rubrics.py:180-183`). The
orchestrator notices A2's failure, discards its result, and redoes the work
correctly. Subagents A1 and A3 recover cleanly from their own errors. Component
anchors are A1 = 1,
A2 = 0, A3 = 1, and orchestrator = 1:

- the minimum gives the session 0, as if nothing recovered;
- the mean gives 0.75, which hides that one agent never diagnosed its error;
- a merge that reads the findings can apply the 0.5 or 0.75 wording about
  "avoidable continued failure" or "a minor avoidable detour"
  (`rubrics.py:188-195`).

## Proposed design for Monopole

### What the user chose and what this proposal adds

Choices already made by the user:

1. Judges see every message in a turn.
2. A Workflow session gets a whole-session score and separate scores for the
   orchestrator and for each subagent.
3. A tool-using judge may be tried as an experiment.

Details added here, each open to change:

- A. Windows cut at thread, turn, then step, so choice 1 holds for turns larger
  than a window instead of skipping the reviewer.
- B. Overlap turns are context only; a window cites only its core. This stops
  one event from being reported twice.
- C. The host adds each finding's turn and window position to the merge input,
  so the merge can judge order.
- D. A reader that keeps the thread tree, every message, and nesting. Choice 2
  needs it.
- E. Leaf threads are reviewed first, and each parent sees a short verdict
  summary at the call that launched each child, so a parent review does not
  re-read its children.
- F. The session score comes from a model merge. A rule-based roll-up is stored
  beside it and flags large gaps, so consistency is checked without forcing a
  rule that gets recovery wrong.
- G. A table of which rubric each unit is scored on, so each score answers a
  stated question.
- H. A cited ledger replaces the digest, plus a pass that resolves claims
  spanning windows. This is gated on the planted cross-window probes.
- I. Each reviewer extracts its own ledger, keeping panel members independent,
  as today's per-reviewer digests do (`specs/02-evaluation.md:241-243`).
- J. Size tiers and grouped merges, so the design keeps working as sessions grow.

### Window plan

1. Build the thread tree from CES: the orchestrator thread from the session
   Trajectory's own Cases, and one thread per child Trajectory, nested as stored
   (EVALUATIONS_CASE_CONTRACT.md:8-11).
2. Render every step of every turn: each user message, including mid-turn
   steering messages, each assistant message, and each tool call. Keep today's
   compaction of tool text over 12,000 characters (`windowing.py:17-20`,
   `windowing.py:80-101`) until the user says whether "every message" includes
   full tool output.
3. If a thread fits the measured single-request budget from the companion's
   experiment plan, judge it whole, with one request per rubric.
4. Otherwise pack whole turns into windows near the raw target, as today
   (`windowing.py:211-233`). A turn larger than the target is split at step
   edges into parts. Each part repeats the turn's first user message as context
   and is labeled with its part number, for example "turn 12, part 2 of 5".
5. A single step larger than the target can only be a very large user or
   assistant message, since tool text is compacted. Split its text into numbered
   pieces so every token is still read raw.
6. Overlap turns are shown as context but are not citable.

### What each request returns

Ledger request, one per window per reviewer, rubric-neutral. It replaces the
digest. The host fills in entries it can derive by code, and the model adds the
rest:

| Entry kind | Who writes it | Example |
| --- | --- | --- |
| check | host, from recognized test, build, lint, and install commands and exit codes | `t-2210`: `pytest`, exit 0, turn 64 |
| edit | host, from file-editing tool calls | `t-415`: edits `parser.py`, turn 11 |
| error | host, from failed tool calls | `t-301`: exit 1, turn 8 |
| correction | host for steering messages and denials; model for the content | turn 40: user says "don't touch the CLI" |
| claim | model | `m-430`: "Fixed; all tests pass", about `parser.py` |
| constraint or stated plan | model | turn 2: "keep the public API of `parse()`" |
| delegation | host | call `t-77` launched subagent A2 with this prompt |

Each entry gets a host-assigned ID, the canonical evidence IDs it cites, its turn
and step position, and the files or commands it is about.

Rubric window request, one per rubric per window, as today: the raw window, the
thread's ledger, and the rubric last, so the six rubric requests share a cached
beginning, as the companion recommends. It returns today's findings.

Resolution request: see the next section. Its output is host-created findings,
so the merge contract does not change.

Merge request, one per rubric per thread: findings with their positions,
resolution findings, the coverage manifest, and the ledger. It returns today's
verdict contract (`specs/02-evaluation.md:126-139`).

### Resolving claims that span windows

The host pairs ledger entries by rules, then sends only the pairs whose two
ends fall in different windows to a resolution request. Pairs inside one window
were already visible to that window's review.

| Entry | Paired with | Rubric that needs it |
| --- | --- | --- |
| claim of completion or correctness | the last edit before it to a file it names; checks between that edit and the claim; the first check after the claim | Verification, Outcome |
| error | the next 5 steps of the same thread, even across a window edge | Error recovery |
| constraint or stated plan | later edits and commands touching the same file, function, or command | State consistency |
| correction | earlier agent steps on the same file or topic | Autonomy |
| orchestrator claim about delegated work | checks inside that subagent's thread before it returned | Verification |

Five steps is a starting value; raise it if the error-recovery probes show
misses. A resolution request shows the raw steps at both ends of each pair,
fetched by evidence ID, batched up to a token budget. It returns one finding per
pair with the evidence IDs it relied on.

For the worked example above, the host builds this record:

```text
claim m-430 (turn 12, window 2): "Fixed; all tests pass."
  last edit before the claim to a named file: t-415 (turn 11, parser.py)
  checks after t-415 and before m-430: none
  first check after m-430: t-2210 (turn 64, window 7, pytest, exit 0)
  earlier check: t-301 (turn 8, window 1, pytest tests/test_parser.py, exit 1)
```

The resolution request reads those four steps and returns a finding such as "The
claim at turn 12 had no check after the last edit to `parser.py`; tests first
passed at turn 64", citing `m-430`, `t-415`, and `t-2210`. The merge then applies
the rubric: a claim "with no relevant cited check" is the 0.25 anchor, and "one
limited and clearly non-critical verification gap" is the 0.75 anchor
(`rubrics.py:133-144`).

### Whole-session, orchestrator, and subagent scores

Order of work for a Workflow session:

1. Review every subagent thread, deepest first, each through the window plan
   above. Subagent reviews are independent and can run in parallel under the
   provider limits (`specs/02-evaluation.md:243-245`).
2. Review the orchestrator thread. At each call that launched a subagent, the
   judge view adds that subagent's verdict summary: its anchor or abstention per
   rubric and its two strongest findings, about 300 tokens. The summary is
   host-owned context, like a digest today.
3. Merge a session verdict per rubric from the orchestrator's findings, every
   subagent's findings, and the resolution findings. When that input does not
   fit, merge in groups first: per orchestrator window, together with the
   subagents it launched.

What each score covers (proposed):

| Rubric | Subagent | Orchestrator | Whole session |
| --- | --- | --- | --- |
| Verification | its own claims and the checks it ran | its own claims; whether a check inside a subagent counts is an open question | every material claim made to the user, with checks anywhere in the session |
| Error recovery | errors it saw | errors it saw, including failed or wrong subagent results | all errors at any depth |
| Tool choice | its tools for the delegated task | its own tools, and whether and how it delegated | both, including duplicated delegation |
| State consistency | constraints in its delegation prompt | user constraints and its own stated plans | conflicts across threads too |
| Outcome | whether it delivered the delegated task | same as the session; not scored twice | whether the final result met the user's requests |
| Autonomy | not scored by default; its "user" is the orchestrator | same as the session; not scored twice | avoidable user steering |

Skipping a separate orchestrator score for outcome and autonomy narrows the
user's second choice for those two rubrics. The orchestrator's final message is
the session's final result, and only the orchestrator talks to the user, so the
two scores would answer the same question. Open question 2 asks whether that is
acceptable.

How the three stay consistent:

- One set of evidence. Every score reads the same pinned threads, ledger, and
  resolution findings, so two scores cannot rest on different facts.
- Stated scope. The table above says what each score is about, so a session
  score that differs from the orchestrator's has a reason the reader can check.
- An audit roll-up. For each process rubric the host stores the mean and minimum
  of the component anchors next to the session score. It flags the session for
  human review when the session score is more than 0.5 from the mean, or outside
  the range of the component anchors. The flag never changes the score. The
  merge model does not see the roll-up, so it cannot anchor on it.

### How it degrades as sessions grow

| Size | What runs | What is lost |
| --- | --- | --- |
| thread fits the measured budget | one request per rubric over the whole thread | nothing from windowing |
| thread too big, every turn fits a window | turn windows, ledger, rubric windows, resolution, one merge per rubric | raw detail outside the active window, except through the ledger |
| one turn too big | that turn split at step edges, with its first user message repeated | the turn's continuity, partly restored by the ledger |
| one step too big | message text split into numbered pieces | sentence-level continuity at the cut |
| ledger larger than its budget | entries far from the active window shown as one line each; settled pairs collapsed; open claims never collapsed | the wording of old, settled entries |
| merge input too big | grouped merges, then one final merge; groups keep finding IDs and open claims | nuance within a group |
| hundreds of subagents | independent subagent reviews; ~300-token summaries at the orchestrator; grouped session merge | subagent detail at the orchestrator level, still read raw by that subagent's review |
| over a cost cap | skip with an explicit reason; never a partial score | the score |

Every message is still read raw by exactly one window somewhere in the tree.
**Inference:** this extends the existing "complete raw-turn coverage" rule
(`AGENTS.md:117-118`) from turns to messages across threads. Confirm that this
is what choice 1 means for the session-level score, which reads summaries rather
than raw subagent messages.

Why the hundreds-of-subagents tier matters. **Inference**, from the companion's
figures: one orchestrator turn that launches about 550 subagents can keep up to
24,000 characters per delegation call, at most about 8,000 tokens by bytes ÷ 3.
In the worst case that is roughly 4.4 million tokens in a single turn, about 35
parts at a 128,000-token target, close to today's 40-chunk cap even before the
subagent summaries are added.

### Cost

Requests per reviewer, for a thread split into W windows with P cross-window
pairs:

- a thread that fits: 6 requests;
- a windowed thread: W ledger, 6W rubric windows, about P ÷ 20 resolution
  (assuming 20 pairs per request), and 6 merge requests, plus 6 per extra merge
  level;
- a Workflow session: the sum over threads, plus 6 session merges.

Worked example, **Inference** with stated assumptions: an orchestrator thread of
600,000 tokens at a 128,000-token target (W = 5, P = 40), and 120 subagents of
15,000 tokens each that fit whole.

| Part | Requests |
| --- | --- |
| 120 subagents × 6 rubrics | 720 |
| orchestrator: 5 + 30 + 2 + 6 | 43 |
| session merges | 6 |
| total per reviewer | 769 |
| today, which reads no subagent internals | 41 |

At W&B's four active calls and 30 seconds per request (assumed), 769 requests
take about an hour and a half per reviewer. Scoring all six rubrics in one
request per subagent cuts the total to 169, at the risk of rubrics influencing
each other; that is an experiment arm, not a default. The companion's price table
converts token counts into dollars.

### What stays the same

- The verdict contract: five anchors, abstention, and citations of findings
  (`specs/02-evaluation.md:126-139`).
- Request-keyed resume. Ledger, resolution, and merge requests have
  deterministic inputs, so a restart reuses them (`sliding.py:413-443`,
  `specs/02-evaluation.md:284-300`).
- Pinning. The thread tree and each thread's window plan join the pinned plan
  and are recomputed before work (`specs/02-evaluation.md:40-48`,
  `sliding.py:275-300`). The protocol and plan versions change (`sliding.py:68`,
  `windowing.py:16`), and so must the frontend estimate that copies the plan
  arithmetic (`frontend/src/features/runs/modelCapacity.ts:47-89`).
- Panel pooling (`review.py:430-450`). The run database is disposable, so no
  migration is needed (`AGENTS.md:176-177`).

## Experiment plan

Goal: decide whether the ledger and resolution pass are needed on top of
structural windows, pick window sizes per model, and check that session,
orchestrator, and subagent scores behave as defined, using Monopole's own
sessions. Everything runs on offline copies; nothing is written to CES.

1. Offline reader and planner. Build the thread-tree reader and the structural
   window planner in a script first.
2. Census, with no model calls. On about 200 recent sessions, extending the
   companion's census, record per thread: tokens, the largest turn and step,
   subagent count and depth, the number of windows at targets of 32,000, 64,000,
   128,000, and 400,000 tokens, and the number of cross-window pairs the rules
   produce using host-derived entries only.
3. Ledger quality. On 20 sessions, including 5 Workflow sessions, two people mark
   completion and correctness claims, constraints, and corrections. Measure the
   ledger's recall and precision against them, and their agreement with each
   other by nominal Krippendorff's alpha, an agreement score corrected for
   chance where 1 is perfect and 0 is chance.
4. Windowed against whole, on sessions that fit. Take 30 sessions under the
   measured single-request budget. Judge each whole thread once as the reference.
   Then force windows of 32,000, 64,000, and 128,000 tokens and run three arms:
   today's digests, the ledger alone, and the ledger with resolution. Report
   exact anchor agreement with the reference, agreement within 0.25, ordinal
   Krippendorff's alpha, mean signed difference, and abstention disagreements
   separately. **Inference:** forcing small windows on sessions that fit gives a
   full-context reference for the same session, which separates the effect of
   windowing from the effect of session size.
5. Planted cross-window pairs, on offline copies. Planted defects give known
   answers without human labels; in FBI, five evaluator models missed planted
   quality drops in over half of cases on average [14].
   - Verification, 12 sessions: put the claim and its only relevant check in
     different windows. Variants: the check passes before the claim, which must
     not be penalized; the last check before the claim fails, the 0 anchor
     (`rubrics.py:129-132`); no check between the last edit and the claim, the
     0.25 anchor (`rubrics.py:133-136`).
   - State consistency, 8 sessions: a user constraint early, and an edit that
     breaks it at least 5 windows later, the 0 anchor (`rubrics.py:281-284`).
   - Error recovery, 8 sessions: an error in the last step of a window and the
     same failing command repeated in the next window (`rubrics.py:180-183`).
   - Place each pair at three distances: the same window; adjacent, with the
     claim in a window's last step and the evidence in the next window's first
     step; and 5 or more windows apart.
   - Score whether the anchor moved the expected way, and whether some finding
     cites both ends.
6. Window-edge stability. On 20 sessions, shift window edges by 1 to 3 turns and
   change the target by 25% either way. Also cut purely by token count, even
   inside a turn. Report how often anchors change, against the repeat noise from
   step 8.
7. Workflow scope. On 10 Workflow sessions, including the one with about 550
   subagents, record all three kinds of score and the audit flags.
   - Plant a repeated failing command inside a subagent whose result the
     orchestrator discards. Expected: that subagent's error recovery drops, the
     session's drops less, and the orchestrator's does not change.
   - Plant an orchestrator completion claim that relies on a subagent's failed
     test. Expected: session and orchestrator verification drop to 0.
8. Repeat noise and panel agreement. Run each arm three times on 10 sessions at
   Monopole's temperature of 0.2 (`sliding.py:67`). Report the share of cells
   where all three anchors match, a consistency measure in the spirit of
   τ-bench's pass^k, which measures whether an agent succeeds consistently over
   k repeated trials [17]. Report ordinal Krippendorff's alpha across reviewers, using Transect's
   warning levels as a start: below 0.66 is a red warning, 0.66 to 0.80 amber
   [19].
9. Tool-using resolver. Give Codex, under Monopole's read-only profile, or Claude
   Code with read-only tools, which needs the documented safety rule changed
   (`AGENTS.md:137`), one file per window and the queue of open pairs. Compare
   its resolution accuracy on the planted pairs with the rule-plus-request
   resolver, and record files opened, cost, and run-to-run variation.
10. Cost and time. Record requests, input, cached, and output tokens, and wall
    time under the provider limits (`specs/02-evaluation.md:243-245`).
11. Human labels. Two people label every cell where arms disagree by 0.5 or more,
    or where one abstains and another scores, plus a random 10% of agreeing cells,
    with 20% labeled by both. Reviewing only flagged cases follows the finding
    that escalating pre-flagged cases to humans raised true positives by about
    15% at a 1% false-positive rate [1].
12. How to choose:
    - Keep today's digests if, for planted pairs one or more windows apart, they
      detect within 5 points of the ledger-with-resolution arm, and their
      agreement with the whole-thread reference stays within the repeat noise.
      Otherwise adopt the ledger and resolution pass.
    - Per model, use the largest window target whose planted-pair detection and
      reference agreement stay within the noise of the best target, because
      larger windows mean fewer requests.
    - Keep the audit flags if humans agree with the session score in most flagged
      cases; if they agree with the roll-up, revisit how the session score is
      formed.
    - Thirty sessions show large gaps, such as 90% against 60% detection, but not
      small ones.

## Observations outside the question

Each could be its own small fix, separate from this research.

1. Confirmed in code: `SubagentSpan.agent_type` is filled from the subagent
   Cases' `agent_id` (`ces_client.py:245`), and the judge view prints it as
   `agent_type` (`windowing.py:171`). The adapter contract lists `agent_id` but no
   agent-type field (EVALUATIONS_CASE_CONTRACT.md:27). The judge therefore sees an
   ID labeled as a type.
2. Confirmed in code: the reader drops subagent messages and keeps only tool
   calls (`ces_client.py:242-249`), and it attaches nested subagents to the
   top-level turn that launched their ancestor (`ces_client.py:218-224`). Scoring
   each subagent needs both fixed.
3. **Inference** from code: windows may cite evidence in their overlap turns
   (`windowing.py:524-527`, `sliding.py:895`), and deduplication removes only
   exact duplicates (`sliding_contracts.py:70-75`), so one event can reach the
   merge twice.
4. Confirmed in code: the merge receives findings without evidence IDs or
   positions (`sliding.py:943-946`), so questions of order depend on a window
   having stated the order.
5. Confirmed in the spec: any reviewer failure fails the session and stops the
   remaining sessions (`specs/02-evaluation.md:254-261`). With hundreds of
   subagent reviews per session, one transient failure would block all of that
   session's Scores.

## Open questions

These are choices for the user; the proposal above only suggests defaults.

1. Does "every message" include full tool output, or is today's compaction of
   tool text over 12,000 characters acceptable?
2. Rubric meaning per unit. Does an orchestrator's claim count as verified when
   the check ran inside a subagent? Should subagents get an autonomy score? Should
   orchestrator outcome and autonomy be skipped as duplicates of the session's?
3. Which CES Case should a subagent's Score target: the launching `tool_call`
   Case, which roots the subagent Trajectory (`specs/01-ces-io.md:12-13`), or the
   subagent's first member Case? How should those Scorers be named, given that
   each rubric and panel has its own Scorer (`AGENTS.md:159-160`)?
4. Should the session score be the model merge with a flagged roll-up, as
   proposed, or a fixed rule over component scores?
5. One ledger per reviewer, which keeps reviewers independent, or one shared
   ledger, which is cheaper?
6. Should one failed subagent review block the session's Scores, as today's rule
   would (`specs/02-evaluation.md:254-261`)?
7. Is a per-session cost cap acceptable for very large Workflow sessions? If so,
   what happens at the cap: a skip with a new reason, or a sample of subagents?
8. Should single-turn helper subagents get Scores? Inspect Scout excludes them
   [24].
9. Should nested subagents be scored at every depth?
10. One request per subagent scoring all six rubrics, or six independent
    requests?
11. In what order should parallel subagents appear in the orchestrator view:
    launch order or completion order?

## Sources

All accessed 2026-11-03.

Papers and reports:

1. Kale et al., Reliable Weak-to-Strong Monitoring of LLM Agents (ICLR 2026). https://arxiv.org/abs/2508.19461
2. Mittapalli et al., TRACE: Trajectory Reasoning through Adaptive Cross-Step Evidence Aggregation for LLM Agents. https://arxiv.org/abs/2606.07054
3. Zhai et al., GUIDE: Interpretable GUI Agent Evaluation via Hierarchical Diagnosis. https://arxiv.org/abs/2604.04399
4. Madvil et al., Holistic Evaluation and Failure Diagnosis of AI Agents. https://arxiv.org/abs/2605.14865
5. Deshpande et al., TRAIL: Trace Reasoning and Agentic Issue Localization. https://arxiv.org/abs/2505.08638
6. Zhang et al., Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems. https://arxiv.org/abs/2505.00212
7. Zhang, Kraska, Khattab, Recursive Language Models. https://arxiv.org/abs/2512.24601
8. Wu et al., Recursively Summarizing Books with Human Feedback. https://arxiv.org/abs/2109.10862
9. Sarthi et al., RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. https://arxiv.org/abs/2401.18059
10. Zhou et al., LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models. https://arxiv.org/abs/2410.09342
11. Zhang et al., LongReward: Improving Long-context Large Language Models with AI Feedback (ACL 2025). https://arxiv.org/abs/2410.21252
12. Zha et al., AlignScore: Evaluating Factual Consistency with A Unified Alignment Function. https://arxiv.org/abs/2305.16739
13. Snell et al., Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. https://arxiv.org/abs/2408.03314
14. Finding Blind Spots in Evaluator LLMs with Interpretable Checklists (EMNLP 2024). https://arxiv.org/abs/2406.13439
15. On Memory Construction and Retrieval for Personalized Conversational Agents (SeCom). https://arxiv.org/abs/2502.05589
16. Is Semantic Chunking Worth the Computational Cost? https://arxiv.org/abs/2410.13070
17. Yao et al., τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. https://arxiv.org/abs/2406.12045
18. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. https://arxiv.org/abs/2410.10813
19. Pilditch et al., Transect: Retaining Observability for Long-Horizon LLM Agent Evaluations. https://arxiv.org/abs/2610.08364
20. DynSTEER: Dynamic Stage-wise Trajectory Evaluation and Execution-time Review for Agents. https://arxiv.org/abs/2609.14637
21. LiteTrajEval: Lightweight, Rubric-Guided Trajectory Evaluation for Production AI Agents. https://arxiv.org/abs/2610.03315
22. SKG-Eval: Stateful Evaluation of Multi-Turn Dialogue via Incremental Semantic Knowledge Graphs. https://arxiv.org/abs/2605.16650
23. Yang et al., SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. https://arxiv.org/abs/2405.15793

Documentation:

24. Inspect Scout, Multi Agent (timelines, depth, context chunking, reducers, utility spans). https://meridianlabs-ai.github.io/inspect_scout/multi_agent.html
25. Claude Code, Subagents. https://code.claude.com/docs/en/sub-agents
