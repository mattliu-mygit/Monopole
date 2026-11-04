# Judging multi-million-token sessions with small-context judges

## Summary

1. Layer the work: code extracts facts, windows are sized per model, a cited ledger crosses windows, a model merges each thread, and hard cases escalate.
2. Size windows from measured effective context: judges up to 70B fell below chance past 4k tokens [32], and frontier monitors missed events 2–30× more often after 800k tokens [12].
3. Repeat the task inside every window: the query before and after the data, and reminders between tool calls, raised detection at little cost, but not for subtle cases [5][12].
4. Bounded-memory agents keep only the task, distilled notes, and the latest step; their large gains come from training [16][17][18], and prompting alone gains less [19][20].
5. Tool-driven reading scales past the window: RLM scored 91.3% on 6–11M-token inputs that GPT-5 could not load [22], but a small root model needed training first [22].
6. Step-level verifiers keep each request small, yet the best model found the first error in only 65.8% of tool-use trajectories [25], so step labels should feed the verdict, not replace it.
7. Panels from different model families beat one large judge on short QA [27], but errors correlate more as models get stronger [28], so vary reading designs as well as families.
8. Today's policy cannot plan an 8k judge at all, and limits a 32,800-token judge to sessions that fit one 14,800-token window (computed from code, see Observations).
9. Next: calibrate each model's budgets on planted probes in the user's sessions, then spend the 20–40 hand labels on cells where reading plans disagree.

## How to read this

Research notes on how judges whose context windows hold about 8k to 131k
tokens could judge Claude Code and Codex sessions of millions of tokens, and how
the techniques combine into one plan that adapts to whichever model the user
picks, from 8k to 1M tokens. Observed 2026-11-03. They are not Monopole product
guarantees.

This is the fifth document in a series and does not repeat the others:

- `docs/long-session-judging-research.md` (the long-session study): long-context
  models, compression, whole-session requests, prompt caching,
  agent-as-a-judge.
- `docs/windowed-judging-research.md` (the windowed study): window-then-merge,
  structural windows, claim ledgers, summary trees, validation.
- `docs/small-model-judging-research.md` (the small-model study): small judges,
  checklists, chunked reading, claim checking, cascades.
- `docs/large-session-anatomy.md` (the anatomy): measurements of the four
  largest real sessions.
- `docs/superpowers/specs/2026-11-03-long-session-judging-design.md` (the draft
  design).

Part A covers the questions those documents left open. Part B is the
consolidated catalog: one entry per technique from all five documents and Part
A, then a stitched design and the experiments that would most reduce
uncertainty.

Citation rules. Every external claim cites a numbered source from the list at
the end. Sources 1–30 were read for this document on 2026-11-03. Sources 31 and
up were read for the companion documents on the same date; claims from them
also cite the companion line where the claim is worded. Repository claims cite
`path:line`. Text marked **Inference** is reasoning from the cited facts, not
something a source states.

Terms used throughout:

- **Context window**: the most tokens a model accepts in one request.
- **Effective context**: the input length up to which a model still does a task
  about as well as on short input. It depends on the task, and it is usually far
  below the context window.
- **Thread, step, window, core, finding, merge, ledger**: as defined in the
  windowed study (`docs/windowed-judging-research.md:20-39`). In short: a thread
  is one agent's own messages and tool calls; a step is one message, tool call,
  or compaction; a window is the part of a thread one request reads raw; its core
  is the part that window may cite; a ledger is a list of cited facts (claims,
  checks, edits, errors, constraints, corrections).
- **Pinned prefix**: text repeated in every window of a thread, such as the
  thread's task and the rubric. "Pinned" means the planner puts it there every
  time; it is not carried forward by the model.
- **Running notes**: notes a model writes in one window and the next window
  reads.
- **Reading plan** (proposed here): the per-model, per-thread choice of how a
  thread is read: whole in one request, in rubric windows, or as small
  questions, plus the window sizes and the escalation target.
- **Step verifier**: a judge that labels one step at a time, with only local
  context. A process reward model (PRM) is a trained step verifier.

## Part A: what the existing documents leave open

### A1. Sliding-window variants

What Monopole does today:

- It packs whole turns into windows near a soft target of 50,000 tokens for
  models at or below 200,000 tokens of capacity, and 128,000 above
  (`src/weave_agent_signals/run_config.py:49-53`).
- Each window adds up to one neighboring turn on each side when it fits
  (`run_config.py:60`, `src/weave_agent_signals/judges/windowing.py:236-253`).
  A window may cite those overlap turns, so one event can be reported twice
  (`docs/windowed-judging-research.md:477-481`).
- Every other chunk is shown as a free-text digest of about 1,000 tokens,
  earlier and later chunks alike (`src/weave_agent_signals/judges/sliding.py:70-76`,
  `sliding.py:846-857`).
- The rubric comes first: the system message holds it, and the user message
  lists the allowed evidence IDs and the criteria before the session
  (`sliding.py:77-92`, `sliding.py:858-873`).

**Overlap, also called stride.** The stride is how far each window moves; any
part it does not move is overlap.

- When a model reads text itself, read-only context before the scored part
  helps. GPT-2 large scored a perplexity of 19.44 on WikiText-2 with disjoint
  1,024-token windows, and 16.44 with a stride of 512, where the overlapping
  half is context only and is left out of the score [4]. This is language
  modeling, not judging.
- In retrieval, overlap raised token recall from 0.771 to 0.824 for 250-token
  chunks with 125 tokens of overlap, and lowered efficiency because the
  overlapping tokens repeat [8]. A three-page preprint with an 8B generator found
  that 10–20% overlap gave no measurable gain in question answering [9].
- No source found for this document or the windowed study measures how overlap
  changes a judge's verdict (`docs/windowed-judging-research.md:467-468`).

**Inference:** the evidence supports overlap as read-only context, which the
windowed study already proposed: windows read their neighbors but cite only
their core. Overlap should be counted in steps, not turns, because one turn can
hold 1.07M tokens (`docs/large-session-anatomy.md:93`).

**Window size.**

- In Chain of Agents, Claude 3 Haiku on NarrativeQA improved as each worker's
  window grew from 4k to 16k tokens, and stayed flat around 20 points up to
  128k [6].
- LLM×MapReduce ran Llama3-70B-Instruct, whose effective context the authors put
  at 8K, over chunks of InfiniteBench, whose inputs average over 100k tokens
  (`docs/small-model-judging-research.md:323`). It averaged
  68.66, against 57.34 for GPT-4 reading the whole input. Removing its
  structured map output cut the English-understanding average from 41.23 to
  25.93 [7].
- A hierarchical agent monitor worked best with chunks of 4 to 5 tool calls
  (`docs/windowed-judging-research.md:184-188`) [34].
- Opus 4.6 with thinking caught an obvious dangerous action 99.7% of the time
  inside a 100k-token Claude Code transcript, and 69% inside an 800k-token one
  [12].

**Inference:** quality rises with window size up to a model-specific plateau,
then falls off a cliff. A reading plan should sit on the plateau, and only a
measurement on Monopole's own probes can find it (experiment 1).

**Pinned prefix.** Several designs repeat the task in every request:

- Placing the query both before and after the data gave near-perfect key-value
  retrieval: GPT-3.5-Turbo (16K) became perfect at 300 key-value pairs, against a
  worst case of 45.6% without it. It barely changed multi-document question
  answering [5].
- Anthropic says queries placed after long inputs "can improve response quality
  by up to 30 percent in tests", especially with complex multi-document inputs,
  and recommends asking for relevant quotes first [11].
- Repeating the whole prompt twice won 47 of 70 comparisons with no losses when
  models answered without reasoning, and was neutral to slightly positive with
  reasoning [10].
- Inserting a reminder of the monitoring task between every tool call and its
  result cut missed obvious attacks for most models, did not help with subtle
  ones, added under 3% input tokens, and did not raise false positives [12].
- Pydantic AI's trajectory judge reads at most the most recent `window` tokens
  (20,000 by default). When the run is longer, it keeps the request that started
  the run at the top, cut to half the window if needed, and fills the rest with
  the latest steps [13].
- IterResearch keeps three things in each round: the question, an evolving
  report, and the last interaction [20]. Chain of Agents gives every worker the
  query along with the previous worker's notes [6]. SAFARI appends its memory to
  the end of every request [15].

**Inference**, what a Monopole window could pin:

- the rubric's question and anchors, after the transcript;
- the thread's task: its first user message and later user requests and
  constraints, which are 0.1% to 6.0% of a session's content
  (`docs/large-session-anatomy.md:67`);
- the ledger entries still open when the window starts, such as a completion
  claim not yet checked;
- optionally, a one-line reminder between steps. A reminder that names the
  rubric breaks the shared cached beginning across the six rubric requests
  (`docs/long-session-judging-research.md:543-552`). Cost is not a concern, but
  caching also cuts latency, so this is a tradeoff to measure (experiment 3).

**Running notes.** Chain of Agents read left to right best on most datasets
(MuSiQue 37.09, against 29.77 right to left), and dropping the final manager
cost over 10 points on MuSiQue (26.79) [6]. A sequential monitor that carries
only the previous chunk's summary was weak alone; a monitor that keeps a
persistent findings list raised recall
(`docs/windowed-judging-research.md:210-225`) [33][34].

**Attention sinks.** StreamingLLM keeps the internal key-value cache of the
first few tokens plus a window of recent tokens. That kept language modeling
stable for up to 4 million tokens on Llama-2, MPT, Falcon, and Pythia, without
fine-tuning [1]. Its authors warn that this does not extend memory: middle
tokens are discarded, and a model given a book "might only summarize the
concluding paragraphs" [2]. InfLLM adds memory units for distant context and
looks up the relevant ones for each token, without training, up to 1,024K
tokens [3].

**Inference:** both methods change the model's key-value cache. Hosted APIs and
the Claude, Codex, and Antigravity CLIs that Monopole calls do not expose that
cache, so Monopole cannot use them unless it self-hosts a model. The pinned
prefix is the prompt-level counterpart, and StreamingLLM's warning applies to
it: pinning the start and keeping the recent end forgets the middle unless a
ledger carries it.

### A2. Bounded-memory and context-folding agents

These agents work on long tasks inside a fixed context by rewriting their own
history:

| Method | What stays in context | What is dropped | Trained? | Result |
| --- | --- | --- | --- | --- |
| MEM1 [16] | one compact internal state, rewritten each turn | all earlier turns | yes, reinforcement learning | MEM1-7B: 3.5× the performance at 3.7× less memory than Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task |
| AgentFold [17] | several summaries at different scales, plus the latest action and observation in full | raw history, folded into summaries; whole finished sub-tasks folded into one line | yes, supervised fine-tuning of Qwen3-30B-A3B | 36.2% on BrowseComp, matching or beating DeepSeek-V3.1-671B; context about 7k tokens after 100 turns, scaling to 500 turns |
| Context-Folding [18] | the main thread; a side branch's steps until it returns a summary | the branch's intermediate steps | yes, reinforcement learning (FoldGRPO) | Seed-OSS-36B with a 32K active context: 62.0% BrowseComp-Plus and 58.0% SWE-Bench Verified, against 54.0% and 57.4% for the same model trained with a 327K context |
| ReSum [19] | the latest summary plus new turns | turns already summarized | optional | +4.5% over ReAct without training; +8.2% more with training |
| IterResearch [20] | the question, an evolving report, the last interaction | everything else | yes; also tested as a prompt | 2,048 interactions in a 40k context; as a prompt alone, up to +19.2 points over ReAct for frontier models |
| MemAgent [38] | 1,024 tokens of memory plus a 5,000-token chunk | the read text, once memory is updated | yes, reinforcement learning | trained 7B model: 71.9% at 3.5M tokens (`docs/small-model-judging-research.md:322`) |
| ARMT [21] | a recurrent memory passed between segments | the segments | yes, a custom architecture | 79.9% on single-fact BABILong questions over 50M tokens |
| SAFARI's memory [15] | task goal, hypotheses with evidence, open gaps, past tool calls | old tool results once the budget fills | no | see A3; slightly worse than reading everything when the trace fits |

What they keep and lose:

- They keep the task, distilled findings, and the newest raw step. They lose raw
  detail of everything folded, and anything the summarizer judged unimportant at
  the time. AgentFold's authors describe summarizing every step as risking "the
  irreversible loss of crucial details" [17].
- The large gains are trained. Without training, Context-Folding with a 32K
  context scored 42.0% and 49.2%, below 47.8% and 55.2% for the same untrained
  model reading 327K [18]. Prompt-only gains were smaller but real: ReSum +4.5%
  [19], IterResearch up to +19.2 points on frontier models [20].
- Code-heavy traces were harder to condense into memory than general agent traces
  [15].

Could a judge read a transcript the same way? **Inference:**

- Yes in shape. A judge reading a recorded session is a reader with a fixed
  budget, like MemAgent or IterResearch: pin the task, carry notes, read the next
  window raw.
- Three differences matter:
  - A judge must cite evidence, so its notes must carry evidence IDs. That is
    the windowed study's ledger, not a free-text summary.
  - The transcript is complete, so a judge can read twice: first extract notes
    from every window in parallel, then judge each window with the whole
    thread's notes, earlier and later. An agent cannot look ahead.
  - The best results come from training on the task. Monopole has no labeled
    set (`specs/02-evaluation.md:364-368`), so it can copy the layout but not
    the training.
- The agent's own compactions do not help directly. CES stores a compaction
  step with only a `trigger` field
  (`weave-agent-adapter/specs/EVALUATIONS_CASE_CONTRACT.md:30`), and Monopole's
  reader keeps only its timestamp (`src/weave_agent_signals/ces_client.py:272-276`).
  Their positions still make good fold points: every stretch between compactions
  in the four largest sessions is 520k tokens or less
  (`docs/large-session-anatomy.md:93-97`). The agent's own summaries would be
  self-reports in any case.

### A3. Recursive and agentic navigation

Here the transcript stays outside the model's context, and the model reads it
through code or tools.

**Recursive Language Models (RLM)** load the prompt into a Python environment as
a variable. The model writes code to inspect it, split it, and call itself on
the pieces [22]. Results from the paper's main table, with GPT-5 as the root and
GPT-5-mini for sub-calls [22]:

| Task (input size) | GPT-5 alone | Compaction agent | Retrieval agent (CodeAct + BM25) | RLM, depth 1 | Coding agent with the input in a file |
| --- | --- | --- | --- | --- | --- |
| BrowseComp-Plus, 1,000 documents (6M–11M tokens) | 0.0, input does not fit | 70.5 | 51.0 | 91.3 | Claude Code (Opus 4.1) 84.0; OpenCode 94.0 |
| CodeQA (23K–4.2M) | 24.0, some inputs did not fit | 58.0 | 22.0 | 62.0 | Claude Code 62.0; OpenCode 64.0 |
| OOLONG (131K) | 44.0 | 46.0 | 38.0 | 56.0 | Claude Code 48.0; OpenCode 52.0 |
| OOLONG-Pairs (32K) | 0.1 | 0.1 | 24.7 | 58.0 (76.0 at depth 3) | Claude Code 6.5; OpenCode 4.8 |

- With Qwen3-Coder-480B as the root, results were weaker and less stable;
  deeper recursion sometimes hurt, for example CodeQA fell from 66.0 at depth 0
  to 44.0 at depth 3 [22].
- Cost per query was long-tailed: the median RLM run was cheaper than the median
  base-model run, but the average was higher because of outlier runs [22].
- Small models: Qwen3-8B "struggles to solve long context tasks even in an RLM
  scaffold". Fine-tuning it on 1,000 trajectories from unrelated tasks raised it
  by a median 28.3%, approaching GPT-5 alone on three tasks [22].
- OOLONG-Pairs, which needs every pair of items compared, is the closest of
  these tasks to cross-window judging. There, coding agents with the input in a
  file stayed below 7, and only recursion did well [22]. **Inference.**

**SAFARI** is the closest published design to a navigating judge. It locates the
decisive fault in an agent trace with two tools, `read(offset, limit)` and
`search(pattern)`, and a short-term memory appended to every request [15]:

- Before concluding, it splits its hypothesis into up to three claims that cite
  quoted evidence. Separate evaluator calls, which cannot see the trace, score
  each claim, and the investigator must reach an average confidence of 70 [15].
- With Claude Opus 4.6 limited to a 25K budget on TRAIL's GAIA traces, it beat
  the best prior method by 7% in precision and 19% in strict precision. With a
  100K budget it matched that method at 1M. At 1M, where the traces fit, it was
  slightly worse, which the authors attribute to information lost in memory
  [15].
- With the fault five times beyond the budget, precision held at 0.58 on GAIA
  traces, where methods that must read everything score 0. On SWE-bench traces
  precision fell to 0.29–0.42 [15].

Other navigating readers are in the companion studies: Agent-as-a-Judge, TRACE's
adaptive inspection, MemWalker, ReadAgent, and GraphReader
(`docs/long-session-judging-research.md:469-498`,
`docs/windowed-judging-research.md:387-423`,
`docs/small-model-judging-research.md:313-337`).

Measured on multi-million-token inputs: RLM at 6M–11M tokens and CodeQA up to
4.2M [22]; SAFARI with faults five times past its budget [15]; LLM×MapReduce on
needle tests up to 1,280K tokens [7]; ARMT at 50M on synthetic questions [21].
None of them judges coding sessions against rubrics.

Fit with Monopole (**Inference**):

- A model's reading path is not pinned, so it cannot by itself meet "complete
  raw-turn coverage" (`AGENTS.md:117-118`). It fits best as a second reader:
  resolving cross-window pairs, or investigating an escalated verdict.
- Claude judges are tool-disabled and Codex runs under a deny-by-default
  profile (`AGENTS.md:137`); a navigating judge needs the documented exception
  the draft design proposes
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:150-157`).
- An 8B root model fails without training [22]. For small models, the recursion
  should live in Monopole's planner instead: code decides the slices and asks
  the model only local questions (catalog entry B19).

### A4. Step-level verifiers

A step verifier labels each step using only nearby context, so a request's size
depends on the step, not on the session. Evidence:

- ProcessBench (3,400 math solutions, experts marked the first wrong step):
  trained PRMs failed to generalize beyond GSM8K and MATH, and did worse than
  general models prompted to critique each step. QwQ-32B-Preview critiqued about
  as well as GPT-4o [24].
- AgentProcessBench (1,000 tool-use trajectories, 8,509 human step labels, 89.1%
  agreement between annotators) [25]:
  - The best model labeled 81.6% of steps correctly but found the first error
    in only 65.8% of trajectories. The best open model, Qwen3-30B-A3B-Thinking,
    labeled 68.5% of steps correctly; Qwen3-8B 57.1%, with 40.7% first-error
    accuracy.
  - Reasoning mode added 6.1 points of step accuracy for Qwen3-8B.
  - On the longest quarter of trajectories, step accuracy fell 3.8 to 13.5
    points and first-error accuracy 10.8 to 21.7 points, more for smaller open
    models.
  - Telling a neutral or exploratory step from a wrong one was the hardest part.
- SWE-PRM runs during an agent's work: every few steps it reads a short sliding
  window of past steps and gives feedback guided by a list of known
  inefficiencies. It raised SWE-bench Verified resolution from 40.0% to 50.6%
  with closed models, at an added inference cost as low as $0.2 [23].
- judgeman's free rules plus a 9B step judge, with a frontier model on flagged
  steps, reached Cohen's κ 0.72 against 0.61 for the frontier model alone; two
  careful humans agreed at 0.68 (`docs/small-model-judging-research.md:565-576`)
  [39].

How step scores become one verdict:

- Product, geometric mean, minimum, and last-step rules can rank the same step
  scores differently [26].
- For a PRM trained to estimate the chance of finishing correctly, the last
  step's score beat the minimum and the product; with right-or-wrong step labels,
  the minimum did best (`docs/windowed-judging-research.md:308-311`) [45].
- Fail-if-any-step-fails counts a minor slip like a critical error, and cannot
  credit recovery: replacing GUIDE's model merge with that rule cut accuracy from
  95.80% to 87.41% (`docs/windowed-judging-research.md:312-318`) [35][44].

Fit with Monopole (**Inference**): every rubric is session-level
(`specs/02-evaluation.md:50`), and several anchors ask whether the agent later
recovered (`src/weave_agent_signals/judges/rubrics.py:179-199`), which no single
step shows. Step labels belong in the ledger as findings for the merge, and a
rule over them belongs in the audit roll-up. One tool call with its result is at
most 12,000 characters per side after compaction (`windowing.py:17`,
`windowing.py:80-101`), roughly 8,000 tokens for arguments and result together
by bytes ÷ 3, so this step fits an 8k-class model only if one side is short;
otherwise the step itself must be split.

### A5. Juries of diverse models

- PoLL pooled three judges from different families (Command R, Claude 3 Haiku,
  GPT-3.5) by majority for QA correctness and by mean for Chatbot Arena scores
  [27]:
  - Cohen's κ with human labels on three KILT QA sets was 0.763, 0.906, and
    0.867 for the panel, against 0.627, 0.841, and 0.830 for GPT-4. On the third
    set Haiku alone scored 0.873, above the panel [27].
  - On Chatbot Arena rankings the panel correlated 0.917 with humans, against
    0.817 for GPT-4 [27].
  - The panel cost over seven times less than GPT-4, and pooling across
    families reduced each model's own scoring bias [27].
  - These are short answers, not transcripts.
- Errors are shared. Across more than 350 models, two models agreed on 60% of
  the questions they both got wrong on one leaderboard, where random wrong
  answers would agree a third of the time. Larger, more accurate models had
  more correlated errors even across architectures and providers, and this
  changed LLM-as-judge results [28].
- Small judges agree poorly with one another: Krippendorff's α was 0.24 and 0.17
  among 7B–9B judges, against 0.67 among 70B–123B judges
  (`docs/small-model-judging-research.md:252-255`) [41].
- Extra votes help less each time on agentic rubric checks
  (`docs/long-session-judging-research.md:673-674`) [51].
- Mixing reading designs can beat mixing models. Averaging a hierarchical
  monitor with a sequential one gave the best AUC at every setting, and monitors
  built on weaker models still exceeded 0.85 AUC
  (`docs/windowed-judging-research.md:210-218`) [34]. In Chain of Agents, a
  model choosing among reading orders beat a vote on summarization and code
  (16.41 against 9.51 on QMSum, 49.69 against 38.70 on RepoBench-P) [6].

Fit with Monopole (**Inference**): a panel already holds one to three judges,
warns on family overlap, and pools anchors by mean with minimum, maximum, and
spread kept (`specs/02-evaluation.md:193-200`,
`src/weave_agent_signals/judges/review.py:430-450`). For small models, the low
agreement makes disagreement a better escalation trigger than a vote. For large
models, the correlated-errors finding argues for diversity in how the session is
read (whole thread, windows, navigation), not only in who reads it.

### A6. Streaming and incremental evaluation

- Monitoring growing prefixes of a transcript, each ending at a tool call, and
  flagging the session if any check fires, improved detection for Opus 4.6 with
  and without thinking. A dangerous action was easiest to catch when it was the
  last thing the monitor saw, and hardest in the middle [12].
- Pydantic AI's trajectory judge runs every N model requests during the run, over
  the pinned request plus the latest steps, and can steer the agent [13]. SWE-PRM
  does the same for coding agents [23].
- A sequential monitor that sees only the previous chunk's summary was weak
  alone, but averaging it with a hierarchical monitor was best
  (`docs/windowed-judging-research.md:210-218`) [34]. A judge that saw the
  conversation so far matched human session ratings better than one that saw
  only the current turn (Spearman 0.69 against 0.63 on MT-Bench)
  (`docs/windowed-judging-research.md:226-228`) [58].
- CES already receives sessions incrementally: the adapter writes each finished
  turn, and writes a subagent's turns with the parent turn that ran them
  (`weave-agent-adapter/specs/EVALUATIONS_CASE_CONTRACT.md:3`, `:55`). Monopole
  evaluates only complete sessions (`specs/02-evaluation.md:8-13`).

Streaming against after the fact (**Inference**):

- A streaming judge sees only the past, so it cannot credit a recovery that comes
  later or use later context to understand an early step. It can judge each
  event while that event is the newest thing in its window, which is where [12]
  found detection best.
- An after-the-fact judge can look ahead and read twice.
- A middle path keeps Monopole's scope: within an after-the-fact run, read each
  thread's turns in order as prefix reviews that carry notes forward, then run
  one look-ahead pass. Judging during the session would change the product's
  scope, so it is listed as an open question.

### A7. Effective context of common judge models

| Model | Context window | RULER effective length [29] | NoLiMa effective length [31] |
| --- | --- | --- | --- |
| Llama 3.1 8B | 128K | 32K (score 77.0 at 128K) | 1K |
| Llama 3.1 70B | 128K | 64K (66.6 at 128K) | 2K |
| Llama 3.3 70B | 128K | not listed | 2K |
| Qwen3-8B | 128K | 64K (77.4 at 128K) | not listed |
| Qwen3-32B | 128K | over 128K (85.6 at 128K) | not listed |
| Qwen2.5-7B-Instruct-1M | 1M | over 128K (84.4 at 128K) | not listed |
| Mistral-Nemo | 128K | 16K (19.0 at 128K) | not listed |
| Phi3-medium (14B) | 128K | 32K (46.1 at 128K) | not listed |
| Gemma 3 27B | 128K | not listed | under 1K |
| GPT-4o | 128K | not listed | 8K |

RULER counts a length as effective while the model scores above Llama-2-7B's
85.6 at 4K, on mostly retrieval-style synthetic tasks [29]. NoLiMa counts it
while the model keeps 85% of its short-input score on questions that share few
words with their evidence (`docs/small-model-judging-research.md:94-107`) [31].
The NoLiMa entries come from the small-model study's table.

Other measurements that bear on judging:

- Every judge tested in LongRM, up to 70B, fell below 50%, the accuracy of
  guessing, once given 4k tokens of context
  (`docs/small-model-judging-research.md:108-112`) [32].
- BABILong: popular models used only 10–20% of their context effectively, and
  accuracy fell sharply as reasoning got harder [30].
- Oolong, which classifies and counts many small items, close to judging a whole
  session: GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro all scored under 50% at
  128k (`docs/long-session-judging-research.md:184-186`) [48].
- The longest trajectories cost the most first-error accuracy, more for small
  open models [25]. Frontier monitors missed events far more often deep in long
  transcripts [12].

**Inference**, starting points to measure, not settled values:

| Model class | Question bundle (one claim and its check) | Local reading (extraction, step labels) | Rubric window read raw | Merge input |
| --- | --- | --- | --- | --- |
| 8k-class or 8B dense | 2k–4k | 4k–6k | not used: questions instead | 4k–6k, as a merge tree |
| 32k-class, or 3B–5B active MoE | 2k–4k | 8k–16k | 8k–16k | 16k |
| 27B–70B dense, 128k–262k windows | 4k–8k | 16k–32k | 32k–64k | 32k |
| frontier, 200k–1M windows | 8k–16k | 64k–128k | the measured whole-thread budget, starting near 128k–256k | 128k |

The question-bundle column follows NoLiMa and LongRM, where linking two pieces
of evidence fails earliest [31][32]. The reading column follows the plateau in
Chain of Agents [6]. The frontier column stays well below 800k because of [12].
Experiment 1 replaces every cell with a measured value per model.

## Part B: catalog

Each entry gives the mechanism, the evidence and how strong it is, the smallest
context and model it is shown or expected to support, what it loses, its fit
with Monopole's code, and what it combines with. Evidence strength:

- **strong**: several controlled studies on closely related tasks;
- **medium**: one or two studies, or a related task such as agent monitoring or
  long-document QA;
- **weak**: indirect evidence, a single short preprint, or reasoning only.

The entries are grouped by the layer of the stitched design they serve.

### Layer 0: preparing the input with code

**B1. Deterministic fact extraction**

- How it works: code reads the thread tree and records each step's kind,
  evidence ID, thread, and position, plus facts: checks (test, build, lint,
  install, Git) with exit codes, edits and their files, failed calls, error
  loops, repeated reads, steering and denial events, delegations, and
  compaction boundaries. Code also pairs facts: a claim with the last edit before
  it and the checks around it, an error with the next steps.
- Evidence, medium: rule checks for web-agent success had 83.8% precision but
  55.9% recall (`docs/small-model-judging-research.md:454-459`) [66]. Rules that
  mark failure signals before one rubric judge improved failure localization by
  20 to 35 points (`docs/small-model-judging-research.md:460-464`) [57].
- Smallest context and model: none; it is code.
- Loses: whatever the parsers do not recognize, by design
  (`specs/02-evaluation.md:34-36`). Codex runs almost everything through `exec`
  (1,428 of 1,458 calls in one session, `docs/large-session-anatomy.md:55`), so
  its edits need parsing out of shell commands.
- Fit: most parsers exist (`docs/small-model-judging-research.md:436-446`).
  CES carries `exit_code`, `permission_decision`, and a `steering` flag
  (`weave-agent-adapter/specs/EVALUATIONS_CASE_CONTRACT.md:28`, `:35`).
- Combines with: everything. It writes the pairs for B8 and B10, the
  boundaries for B3, and the triggers for B17 and B21.

**B2. Deterministic compression of the judge view**

- How it works: plain-text tool output instead of JSON strings, no per-call
  token counters, pointers for repeated identical results, shorter caps for
  successful output without diagnostic lines.
- Evidence, medium for agents and indirect for judges: hiding old tool output
  halved an agent's cost and matched summarization
  (`docs/long-session-judging-research.md:229-230`) [61]; ACON cut peak tokens by
  26% to 54% (`:231-232`) [62]. In the four largest sessions, fixed rules saved
  5% to 20%, a 4k tool-text cap saved 42% on the biggest, and a session-wide
  pointer for already-seen results could remove about 9.7M tokens from the
  Workflow session (`docs/large-session-anatomy.md:14-16`, `:139-141`). Shorter
  relevant input scores better even when the model could find the evidence
  (`docs/long-session-judging-research.md:178-183`) [46][47].
- Smallest context and model: any.
- Loses: evidence a rubric needs; tool choice judges redundancy, so a
  deduplicated read must stay visible as a re-read.
- Fit: all rendering goes through `render_raw_turn` and `compact_tool_text`
  (`windowing.py:80-176`); the draft design's piece 3.
- Combines with: raises every budget below, so more threads qualify for B5.

**B3. Structural windows with read-only overlap**

- How it works: cut each thread at compaction stretches, then turns, then steps;
  pack whole steps up to the model's reading budget; show neighboring steps as
  read-only context; let a window cite only its core; split an oversized step
  into numbered pieces.
- Evidence, medium: agent judges cut along recorded structure
  (`docs/windowed-judging-research.md:465-466`). Diagnosing each subtask segment
  scored 95.80% against 85.61% for one request over the whole GUI trajectory
  (`:139-141`) [44]. Overlap evidence is in A1 [4][8][9].
- Smallest context and model: any; one step is the floor (A4).
- Loses: continuity at each cut, and order across windows unless positions travel
  with the findings.
- Fit: `build_window_plan` keeps turns indivisible and skips a reviewer when one
  turn exceeds the raw budget (`windowing.py:270-421`, `:356-357`); the draft
  design's piece 6 splits them.
- Combines with: B6 for what each window repeats, B13 for what crosses windows,
  B15 and B16 for the verdict.

**B4. Effective-context calibration per model**

- How it works: plant known-answer probes into offline copies of the user's
  sessions at several sizes and positions; for each model, find the largest
  size at which accuracy stays close to its short-input accuracy; store those
  budgets with the model.
- Evidence, strong that advertised context overstates usable context [29][30][31]
  [32][48]. Simple needle-in-a-haystack scores did not predict performance on
  real tasks (`docs/long-session-judging-research.md:174-175`) [50], and the best
  model answering LongBench v2 directly scored 50.1% (`:176-177`) [49]. No study
  has measured Monopole's own questions; NoLiMa's method shows how [31].
- Smallest context and model: any.
- Loses: nothing; budgets go stale when a model or the renderer changes.
- Fit: each model descriptor already has an optional
  `raw_window_target_tokens` (`src/weave_agent_signals/run_config.py:101`), applied
  as the minimum with the size-tier target (`windowing.py:319-322`), and set
  once, to 96,000 for DeepSeek V4 Pro (`src/weave_agent_signals/catalogs.py:100-108`).
- Combines with: feeds the reading-plan choice in the stitched design.

### Layer 1 and 2: how one request reads

**B5. Whole thread in one request**

- How it works: one request per rubric per thread, when the thread fits the
  model's measured whole-thread budget.
- Evidence, medium: with enough budget, full input beat retrieval
  (`docs/long-session-judging-research.md:280-282`). But one request over a whole
  trace found GAIA error locations 29.2% of the time against 82.3% for the same
  model inside a decomposed method (`:284-288`) [35], judge metrics fell as TRAIL
  traces grew (Spearman −0.508 with length for locating the error,
  `docs/windowed-judging-research.md:135-138`) [36], and detection fell from
  99.7% to 69% between 100k and 800k tokens [12].
- Smallest context and model: whatever holds the thread. Every subagent in the
  Workflow session is 161k tokens or less, and every stretch between compactions
  in the four largest sessions is 520k or less
  (`docs/large-session-anatomy.md:93-97`, `:108-115`).
- Loses: attention to the middle [5][12].
- Fit: `build_window_plan` already returns one window for a small session, but
  today that still costs a digest, six windows, and six merges
  (`docs/long-session-judging-research.md:296-298`).
- Combines with: B6, B7, and B22, as one of several reading designs.

**B6. Prompt layout: pinned prefix, query last, reminders**

- How it works: transcript first and the rubric last; the thread's task and the
  open ledger entries pinned in every window; optionally a short reminder between
  steps, or the question repeated before and after the transcript.
- Evidence, medium: A1 [5][6][10][11][12][13][15][20].
- Smallest context and model: any, but pinned text competes with the window.
  Pydantic AI caps its pinned request at half the window [13].
- Loses: reminders alter the transcript the judge sees; a rubric-specific
  reminder breaks the shared cached beginning across rubrics.
- Fit: today the rubric comes first (`sliding.py:858-873`); the draft design's
  piece 4 reverses that.
- Combines with: every entry that sends a window.

**B7. Quote evidence before scoring**

- How it works: the judge first copies the relevant steps with their evidence
  IDs, then reasons, then scores.
- Evidence, medium: asking Opus 4.6 to list the most suspicious actions word for
  word before deciding "aids detection massively" [12]; asking for the evidence
  first helped when length alone hurt (`docs/long-session-judging-research.md:181-183`)
  [46]; Anthropic recommends quotes first [11].
- Smallest context and model: any.
- Loses: output tokens only.
- Fit: findings already accept a short exact quote that the host keeps only if
  it occurs in the window (`specs/02-evaluation.md:149-151`, `sliding.py:83-84`).
- Combines with: B5, B8, B10.

**B8. Rubric questions on evidence bundles**

- How it works: code turns each pair from B1 and B10 into a yes/no question that
  names its evidence IDs, such as "Is test run `t-2210` relevant to the claim in
  `m-430`?". Each question goes in its own request with a bundle of 4k tokens or
  less, and the answer cites IDs.
- Evidence, medium on prose and untested on tool output: per-item checklists
  raised Llama-3-8B's agreement with people from 55.6% to 63.8%, while a fixed
  generic list lowered it to 46.9% (`docs/small-model-judging-research.md:235-244`)
  [40]; gains for 7B–9B judges were mixed in CheckEval (`:248-255`) [41]. Small
  readers invented answers when the bundle lacked them until trained to refuse
  (`:196-198`) [63].
- Smallest context and model: 8k; prompted 7B–9B models
  (`docs/small-model-judging-research.md:769`).
- Loses: anything nobody asked; the 0.75 and 1 anchors need coverage that a
  list cannot prove (`:289-291`).
- Fit: each rubric already separates allowed evidence, excluded concerns, and
  anchors (`src/weave_agent_signals/judges/rubrics.py:36-65`); the small-model
  study maps rubrics to questions (`docs/small-model-judging-research.md:679-686`).
- Combines with: B1, B10, B21 (two judges disagreeing escalates), B15 (answers
  become findings).

**B9. Step labels (step verifiers)**

- How it works: one request per tool-call step, with the thread's task, the
  agent's message before the step, the step, and one line about the next step;
  yes/no labels such as wasted, repeated, risky, failed and ignored, and "is a
  check".
- Evidence, medium: A4 [23][24][25][39]. Checklist-trained 3B and 8B web judges
  beat GPT-4o by about 30 points in-domain
  (`docs/small-model-judging-research.md:265-267`) [60].
- Smallest context and model: 8k if the step fits; a 9B judge with rules and
  escalation (`docs/small-model-judging-research.md:767`).
- Loses: facts that span steps, and recovery; telling neutral from wrong steps
  is hard [25].
- Fit: labels become ledger entries and findings; a rule over them goes to the
  audit roll-up.
- Combines with: B1, B17, B21.

**B10. Claim ledger with targeted resolution**

- How it works: a ledger of claims, checks, edits, errors, constraints, and
  corrections with evidence IDs and positions; code pairs entries, and pairs
  whose ends sit in different windows go to a resolution request that reads both
  ends raw.
- Evidence, medium and indirect: checking each claim against small chunks beat
  checking it against the whole document (`docs/windowed-judging-research.md:259-263`)
  [52]; statement-level checks with partial support matched people 66.2% of the
  time against 57.1% for whole-answer comparison (`:122-129`) [68]; a persistent
  findings list raised monitor recall to 0.844 from 0.580 (`:219-225`) [33].
- Smallest context and model: 8k; resolution bundles stay under 4k.
- Loses: claims never extracted, and claims that need two pieces of evidence
  from different places (`docs/windowed-judging-research.md:288-295`).
- Fit: every step already has its own evidence ID (`windowing.py:179-185`); the
  windowed study details the ledger (`docs/windowed-judging-research.md:575-637`).
- Combines with: B1, B8, B13, B15.

### Layer 3: what crosses windows

**B11. Free-text digests (today)**

- How it works: one rubric-neutral digest of about 1,000 tokens per chunk; each
  window sees all other digests; the merge sees digests and findings.
- Evidence, weak: no study tests digests for judging; a compressed summary "may
  lose weak or ambiguous signals" (`docs/windowed-judging-research.md:219-221`)
  [33].
- Smallest context and model: a model that can digest a whole raw window; today
  that is 50,000 tokens for any model at or below 200,000
  (`run_config.py:53`), about 50 times Llama 3.1 8B's NoLiMa length
  (`docs/small-model-judging-research.md:121-123`).
- Loses: citations, replaced with `[citation omitted]` (`sliding.py:140-143`,
  `:855`), and finding positions (`sliding.py:943-946`).
- Fit: exists (`sliding.py:789-832`).
- Combines with: kept as the baseline arm; B13 replaces it in the stitched
  design.

**B12. Running notes in one pass**

- How it works: read windows in order, carry notes forward, and judge from the
  notes at the end.
- Evidence, medium: Chain of Agents beat retrieval and full-context baselines by
  up to 10% with 8k workers [6]. With Llama 3.1 8B, plain chains lost information
  along the way (`docs/small-model-judging-research.md:321`, `:330-332`) [69];
  gists without look-ups lost accuracy (`:333`) [55].
- Smallest context and model: 8k windows [6]; small models need structure or
  training.
- Loses: look-ahead; an early miss is never corrected; results depend on order.
- Fit: sequential, so each step must be saved to resume
  (`docs/long-session-judging-research.md:411-412`).
- Combines with: B23, where it is the reader; B13 is the parallel alternative.

**B13. Two-pass cited ledger with folding**

- How it works: pass 1 extracts ledger entries from every window in parallel.
  Pass 2 judges each window with the whole thread's ledger, earlier and later,
  folded: settled entries shrink to one line, open entries stay verbatim, and
  long settled stretches fold into a short stretch note that keeps evidence IDs.
  This copies the layout of AgentFold and IterResearch for a reader that can
  look ahead.
- Evidence, weak to medium, all indirect: multi-scale folding held an agent's
  context near 7k tokens after 100 turns [17]; an evolving report plus the last
  interaction improved frontier models by up to 19.2 points as a prompt [20]; a
  memory appended to every request kept a 25K-budget investigator competitive
  [15]; persistent findings raised recall (`docs/windowed-judging-research.md:219-225`)
  [33].
- Smallest context and model: **Inference**, 8k–16k windows if the folded ledger
  stays under about a quarter of the window.
- Loses: the wording of settled entries, and anything pass 1 missed.
- Fit: replaces the digest step; each request still resumes on its own
  (`docs/windowed-judging-research.md:235-239`).
- Combines with: B3, B6 (open entries go in the pinned prefix), B10, B15.

**B14. Trained bounded-memory readers**

- How it works: fine-tune a small reader to update a fixed memory while reading,
  as MemAgent, MEM1, AgentFold, Context-Folding, and RLM-Qwen3-8B do.
- Evidence, strong for their own tasks: A2 and A3 [16][17][18][22][38].
- Smallest context and model: 7B–8B with 8k windows [38].
- Loses: generality. A critic trained on benchmark-style coding data was near
  random on production outcomes (`docs/small-model-judging-research.md:178-181`).
- Fit: Monopole has no labeled set (`specs/02-evaluation.md:364-368`); training
  on a large model's judgments of the user's own sessions is the small-model
  study's open question 8 (`docs/small-model-judging-research.md:976-978`).
- Combines with: could replace the reader in B12 or B13 later.

### Layer 4: forming the verdict

**B15. Model merge over structured findings**

- How it works: windows return structured findings; one request per rubric per
  thread reads them all and returns the anchored verdict. Each finding carries
  its evidence IDs, position, and source window, and optionally a confidence.
- Evidence, medium: removing LLM×MapReduce's structured map output cut its
  English-understanding average from 41.23 to 25.93 [7]; a model merge beat a
  fail-any rule 95.80% to 87.41% (`docs/windowed-judging-research.md:315-318`)
  [44]; decomposed methods beat one request on TRAIL (`docs/long-session-judging-research.md:284-288`)
  [35].
- Smallest context and model: the merge input must fit; today the plan reserves
  11,000 tokens per chunk (`windowing.py:373-377`).
- Loses: links that no window stated, and order, since the merge sees no IDs or
  positions today (`sliding.py:943-946`).
- Fit: exists; adding positions is the windowed study's detail C
  (`docs/windowed-judging-research.md:534-535`).
- Combines with: B16 when the input is too big; B17 beside it.

**B16. Merge trees**

- How it works: merge groups of windows first, such as one compaction stretch,
  then merge the group results.
- Evidence, mixed: in recursive book summaries, a leaf misread an early detail
  that mattered only later, and about 5% of summaries matched human quality
  (`docs/windowed-judging-research.md:172-180`) [53]; merging chunk summaries was
  more coherent but held less detail than updating one summary
  (`docs/long-session-judging-research.md:349-350`) [54].
- Smallest context and model: any; the number of levels grows with the log of
  the window count.
- Loses: detail at each level, and importance revealed late [53].
- Fit: replaces the 40-chunk cap and the merge-fit skip (`windowing.py:367-377`).
- Combines with: B15, B18.

**B17. Rule roll-ups as audit**

- How it works: minimum, mean, any, or last over component scores or step labels,
  stored beside the model verdict and hidden from it; a large gap flags the
  verdict for review.
- Evidence, medium: rules do worse than a model merge when agents recover
  [35][44]; Inspect Scout offers any, mean, median, mode, max, min, union,
  majority, last, and model reducers within one span [14].
- Smallest context and model: none.
- Loses: recovery and severity, which is why it stays an audit.
- Fit: panels already keep mean, minimum, maximum, and spread
  (`src/weave_agent_signals/judges/review.py:430-450`); the windowed study's
  detail F.
- Combines with: B21, as a trigger.

**B18. Split by agent**

- How it works: judge each subagent thread against its delegated task; the
  orchestrator sees a verdict summary of about 300 tokens at each launch call;
  a session merge reads both.
- Evidence, medium: Inspect Scout scans each agent span separately and excludes
  single-turn helper agents [14]; picking the agent first and then reading only
  its steps found the decisive step more often (`docs/windowed-judging-research.md:352-358`)
  [37].
- Smallest context and model: the median subagent is 26.5k tokens
  (`docs/large-session-anatomy.md:110`), so 8k-class models also need windows
  inside subagents.
- Loses: patterns across subagents unless the parent's review looks for them.
- Fit: the reader must keep subagent messages and nesting
  (`src/weave_agent_signals/ces_client.py:218-252`).
- Combines with: B3, B15, B16.

### Layer 5: navigation, escalation, and panels

**B19. Recursion driven by the planner**

- How it works: Monopole's planner plays the role of RLM's root model. Code
  decides the slices and the order, sends each model request one local job (a
  question, a step, a window), and combines answers in further requests.
- Evidence, weak, **Inference** from [22]: RLM's gains come from recursion, and an
  8B root failed without training. Llama 3.1 8B in a structured multi-agent
  design at 2k windows beat the same model reading 128k
  (`docs/small-model-judging-research.md:327-329`) [56].
- Smallest context and model: 8k.
- Loses: the model cannot choose where to look; the rules in B1 must be good.
- Fit: keeps plans pinned and coverage complete (`AGENTS.md:117-118`).
- Combines with: B1, B3, B8, B10, B16.

**B20. Navigation driven by the model**

- How it works: the transcript sits in files or a variable; the model reads,
  searches, and recurses, keeps a memory appended to each request, and verifies
  its claims before concluding.
- Evidence, medium to strong on QA and fault finding: A3 [15][22]; Agent-as-a-Judge
  agreed with people on 84% to 92% of requirements against 60% to 84% for
  LLM-as-a-Judge (`docs/long-session-judging-research.md:482-486`) [43];
  MemWalker's walk down a summary tree beat long-context baselines (`:351-353`)
  [64].
- Smallest context and model: a frontier model with a 25K budget [15]; an 8B
  root only after training [22].
- Loses: guaranteed coverage; run-to-run variance and a long cost tail [22];
  untrusted transcript text reaches a model with tools
  (`docs/long-session-judging-research.md:522-523`).
- Fit: Claude is tool-disabled (`AGENTS.md:137`); the draft design makes a
  tool-using judge an opt-in experiment that never writes official Scores
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:150-157`).
- Combines with: B21 as the escalation target; B10 for resolving pairs.

**B21. Cascades and escalation**

- How it works: a cheaper or smaller reader goes first; an item moves up when two
  judges disagree, when a judge says "not in this evidence" while code says it
  is, when the merge and the roll-up differ by 0.5 or more, or when an anchor is
  0 or 0.25.
- Evidence, medium: a learned cascade matched GPT-4 at up to 98% lower cost
  (`docs/small-model-judging-research.md:496-497`) [70]; Trust or Escalate kept
  over 80% human agreement while deciding almost 80% of items, after calibrating
  on about 500 labels (`:500-510`) [42]; judgeman sent about a third of steps to
  the frontier model (`:511-512`) [39].
- Smallest context and model: set by the first tier.
- Loses: confident mistakes never escalate, and small judges lean lenient
  (`docs/small-model-judging-research.md:535`).
- Fit: an escalated item is an ordinary request with its own key, so it resumes
  and pins like any other (`docs/small-model-judging-research.md:526-528`).
- Combines with: B20, B22.

**B22. Juries of models and of reading designs**

- How it works: two or three judges score the same item and their anchors are
  pooled, or disagreement escalates. Members can differ by model family or by
  reading plan.
- Evidence, medium for short items and none for long transcripts: A5
  [6][27][28][34][41][51].
- Smallest context and model: each member's own.
- Loses: correlated errors make agreement look stronger than it is [28].
- Fit: a panel holds one to three judges with family-overlap warnings and mean
  pooling (`specs/02-evaluation.md:193-200`, `src/weave_agent_signals/judges/families.py:5-34`,
  `review.py:430-450`).
- Combines with: B5 next to B13 or B19, as different reading designs.

**B23. Incremental prefix judging**

- How it works: read a thread's turns in order and judge each growing prefix,
  carrying notes forward, so each event is judged while it is the newest thing in
  the window; a final pass adds look-ahead.
- Evidence, medium for monitoring: A6 [12][13][23][34][58].
- Smallest context and model: the reader's window.
- Loses: look-ahead, unless the final pass runs.
- Fit: Monopole judges complete sessions only (`specs/02-evaluation.md:8-13`),
  while CES receives finished turns as they happen
  (`weave-agent-adapter/specs/EVALUATIONS_CASE_CONTRACT.md:3`).
- Combines with: B12 as the reader; B13 as the final pass.

### Other techniques

**B24. Small fine-tuned judges and checkers**

- How it works: a model trained to judge, or to check a claim against a
  document, answers narrow questions.
- Evidence, medium for claim checking on prose: Bespoke-MiniCheck-7B scored 77.4
  balanced accuracy against 75.9 for GPT-4o on LLM-AggreFact
  (`docs/small-model-judging-research.md:379-393`) [71]; fine-tuned judges behave
  as task-specific classifiers and generalize poorly (`:168-170`) [72].
- Smallest context and model: 0.4B–9B checkers for one claim and its evidence.
- Loses: generality; stack traces and test logs are not prose
  (`docs/small-model-judging-research.md:414-416`).
- Fit: could answer B8 and B10 questions locally.
- Combines with: B8, B10, B21.

**B25. Key-value cache memory (attention sinks, InfLLM)**

- How it works and evidence: A1 [1][2][3].
- Smallest context and model: any self-hosted model.
- Loses: the middle of the input, unless a memory lookup is added [2][3].
- Fit: not usable through hosted APIs or CLIs (**Inference**); listed so it is
  not mistaken for a prompt technique.

**B26. Prompt caching**

- How it works: a request that starts with exactly the same tokens as an earlier
  one reuses the provider's processed copy at a lower price.
- Evidence: provider documentation (`docs/long-session-judging-research.md:527-565`)
  [67].
- Fit: cost is not a concern here, but caching also shortens latency and eases
  rate limits on the many requests small-model plans make. It needs transcript
  first and rubric last (B6).

### How the techniques compare

| Entry | Smallest context | Smallest model with evidence | Main loss | Evidence |
| --- | --- | --- | --- | --- |
| B1 facts by code | none | none | unparsed evidence | medium |
| B2 compression | any | any | evidence a rubric needs | medium, indirect |
| B3 structural windows | one step | any | continuity at cuts | medium |
| B4 calibration | any | any | budgets go stale | strong that it is needed |
| B5 whole thread | the thread's size | frontier | the middle | medium |
| B6 prompt layout | any | any | caching, altered transcript | medium |
| B7 quote first | any | any | none | medium |
| B8 questions | 4k bundles | 7B–9B prompted | what nobody asked | medium on prose |
| B9 step labels | one step | 9B with escalation | cross-step facts | medium |
| B10 claim ledger | 4k bundles | 8B | unextracted claims | medium, indirect |
| B11 digests | a whole window | not shown | citations, order | weak |
| B12 one-pass notes | 8k | frontier; small needs structure | look-ahead | medium |
| B13 folded ledger | 8k–16k (**Inference**) | not shown | pass-1 misses | weak to medium |
| B14 trained readers | 8k | 7B trained | generality | strong in-domain |
| B15 model merge | merge input | 70B | unstated links | medium |
| B16 merge trees | any | any | detail per level | mixed |
| B17 rule roll-ups | none | none | recovery, severity | medium |
| B18 split by agent | subagent size | any | cross-agent patterns | medium |
| B19 planner recursion | 8k | 8B (**Inference**) | model cannot steer | weak |
| B20 model navigation | 25K | frontier; 8B trained | coverage, variance | medium to strong |
| B21 escalation | first tier | 7B first tier | confident errors | medium |
| B22 juries | each member | small panels | correlated errors | medium, short items |
| B23 prefix judging | the window | frontier | look-ahead | medium |
| B24 small checkers | one claim | 0.4B–9B | logs are not prose | medium |
| B25 key-value memory | any | self-hosted | the middle | not applicable |
| B26 caching | any | any | none | provider docs |

## Stitched design (proposal)

### What the user decided and what this adds

Decisions already made by the user:

1. Judges see every message
   (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:204`).
2. A Workflow session gets a session score plus separate orchestrator and
   subagent scores (`:205`).
3. Partial work is recorded, and reruns skip finished work (`:216`).
4. A tool-using judge is worth an experiment (`:213`).
5. The user labels 20–40 disagreement cases (`:197-200`, and this request).
6. Any model the user picks, from any provider and of any context size, must be
   able to judge these sessions well. Cost is not a concern; quality and model
   choice are (this request).

Details added here, each open to change:

- A. A reading plan per model and thread, chosen by code from measured budgets.
- B. Budgets measured per model on the user's own sessions and stored with the
  model.
- C. Transcript first, rubric last, with the thread's task and open ledger
  entries pinned in every window; reminders only if experiment 3 supports them.
- D. Overlap counted in steps and read-only; windows cite only their core.
- E. A two-pass folded ledger replaces free-text digests.
- F. For small models, recursion lives in the planner, not in the model.
- G. Navigating investigators only as an escalation target.
- H. Juries that mix reading plans as well as model families.
- I. Escalation triggers with a per-session budget.
- J. No judging during a live session (open question 6).

### Layers

| Layer | Who | Reads | Produces | Changes with model size |
| --- | --- | --- | --- | --- |
| 0. Facts | code (B1, B2) | CES Cases | thread tree; steps with evidence IDs and positions; checks, edits, errors, steering, delegations, compaction points; pairs | no |
| 1. Windows | code (B3, B6) | layer 0 and the model's budgets | per thread: cores, read-only overlap, pinned prefix | yes: sizes |
| 2. Per-window judging | model (B5, B7, B8, B9) | windows | a whole-thread verdict, rubric findings, or question answers and step labels | yes: which kind |
| 3. Cross-window memory | code and model (B10, B13) | layer 2 outputs | the folded cited ledger, and findings for pairs that cross windows | partly |
| 4. Aggregation | model and code (B15–B18) | findings, answers, ledger, positions | thread verdicts per rubric; merge trees; subagent, orchestrator, and session scores; audit roll-ups | yes: tree depth |
| 5. Escalation | model (B20–B22) | triggers from layers 2–4 | re-judged items from a larger or different model, or an investigator | the target is chosen per panel |

Every raw step is read in exactly one window core, which extends today's
complete raw-turn coverage (`AGENTS.md:117-118`) to steps. Verdict requests read
findings and answers, as today's merge does (`specs/02-evaluation.md:78-90`).

### Choosing a reading plan

Each model gets four budgets from experiment 1, or the conservative starting
values in A7 until measured:

- `B_q`: the largest evidence bundle for one linking question;
- `B_win`: the largest raw rubric window;
- `B_whole`: the largest thread it may judge in one request;
- `B_merge`: the largest merge input.

For each thread of each session, code picks one plan:

1. **Whole (W)** when the thread is at most `B_whole`: one request per rubric.
2. **Rubric windows (R)** when `B_win` is at least 32k tokens (a threshold to
   test): ledger pass, one request per rubric per window, resolution requests
   for cross-window pairs, and one merge per rubric, as a tree when the input
   exceeds `B_merge`.
3. **Questions (Q)** otherwise: step labels, extraction over runs of messages,
   and yes/no questions within `B_q`; verdicts from a merge tree within
   `B_merge`, or from the panel's largest-budget model if the user allows that
   (open question 1).

The escalation target is the panel member with the largest `B_whole`, unless
the user names one. The plan, budgets, and chosen layouts join the pinned
judging plan, so authentication can recompute them before work
(`specs/02-evaluation.md:40-48`).

Worked example, **Inference**, for one 100,000-token thread, the size of the
Workflow session's 90th-percentile subagent (`docs/large-session-anatomy.md:110`),
with assumed budgets:

| Judge | Assumed budgets | Plan | Requests per reviewer |
| --- | --- | --- | --- |
| a 1M-token frontier model | `B_whole` 256k | W | 6 |
| GPT OSS 120B, 131,072 tokens, today's first default (`catalogs.py:169-175`, `:356-357`) | `B_win` 32k | R | about 35: 4 ledger, 24 rubric windows, 1 resolution, 6 merges |
| Qwen3 14B, 32,800 tokens (`catalogs.py:68-74`) | `B_win` 16k, `B_q` 4k | Q | one step label per tool call, about 2 extraction requests for its messages, two judges per question, plus merge levels |
| an 8k-class model | `B_q` 2k–4k, `B_merge` 6k | Q | as above, with smaller bundles and one more merge level |

For the Workflow session's orchestrator thread (1.92M tokens with every message,
in 9 stretches between compactions, the largest 520k;
`docs/large-session-anatomy.md:80`, `:95-96`), a 1M-token judge with a measured
`B_whole` of 256k would read the smaller stretches whole, use plan R inside the
larger ones, and merge across stretches in a tree.

What stays the same: the verdict contract and abstention
(`specs/02-evaluation.md:126-139`), request-keyed resume (`:284-300`), and panel
pooling (`review.py:430-450`). What changes: new request kinds and protocol
versions (`sliding.py:68`, `windowing.py:16`), budgets in the catalog, and the
frontend estimate that copies the plan arithmetic
(`frontend/src/features/runs/modelCapacity.ts:35-94`).

## Experiments that would most reduce uncertainty

All run on offline copies; nothing is written to CES.

1. **Calibrate budgets per model.** Plant probes with known answers into the
   user's sessions: a claim with a failing check, a repeated failing command, an
   edit that breaks an earlier user constraint. Place each at the start, middle,
   and end of bundles of 1k, 4k, 16k, 64k, and 256k tokens, as far as each model
   allows. For each model and job, the budget is the largest size within 5
   points of its 1k accuracy. This replaces every cell of the A7 table.
2. **Agree with a whole-thread reference on threads that fit.** The Workflow
   session's 605 subagents (each 161k or less) and the stretches between
   compactions (520k or less) give references from plan W with two frontier
   model families. Then force plans R and Q at smaller budgets, with the same
   and smaller models. Report exact anchor agreement, agreement within 0.25,
   ordinal Krippendorff's α, and planted detection at 0, 1, and 5 or more windows
   apart. This separates the effect of windowing from the effect of size
   (`docs/windowed-judging-research.md:779-781`).
3. **Prompt layout.** On plans R and Q, compare rubric last, rubric before and
   after, and rubric last plus reminders between steps, using probes in the
   middle of windows. Record latency and cache hits as well as detection.
4. **What crosses windows.** Compare today's digests, one-pass notes, and the
   two-pass folded ledger on probe pairs placed in different windows, and score
   the ledger's recall against the user's own marks on a few sessions.
5. **Escalation.** On escalated items only, compare resolution requests built by
   code with a navigating investigator that has read and search tools. Record
   detection, run-to-run variation, steps opened, and whether every cited step
   was actually read.
6. **Juries.** Compare three model families on one plan with one strong family
   on three plans (W, R, Q), against the human labels and the probes.
7. **Human labels.** Choose the 20–40 cells where plans or model sizes disagree
   by 0.5 or more, or where one abstains and another scores, plus 5 to 8 agreeing
   cells. Label blind to the arm, relabel 5 a week later, and read the result
   with a sign test: with 30 cells where exactly one arm matches the user, the
   winner needs 21 for p < 0.05 (`docs/small-model-judging-research.md:914-917`).
   Prediction-powered inference can correct each arm's overall estimate with
   these labels (`:918-921`) [65]; Transect's warning level of α below 0.66
   (`docs/windowed-judging-research.md:814-816`) [59] is a reasonable starting
   bar for agreement.

## Observations outside the question

Each could be its own small fix, separate from this research.

1. Computed from code: today's policy cannot plan any model whose context window
   is 18,000 tokens or less. The base reserve is 6,000 prompt plus 4,000 output
   plus 8,000 safety (`src/weave_agent_signals/run_config.py:54-57`,
   `src/weave_agent_signals/judges/windowing.py:291-295`), and a window at or
   below it raises `WindowPlanInapplicable` (`windowing.py:323-325`). An 8k
   judge is therefore always skipped.
2. Computed from code: `wandb:Qwen3-14B-Instruct`, at 32,800 tokens
   (`catalogs.py:68-74`), gets a raw budget of 14,800. The merge must fit 18,000
   plus 11,000 per chunk (`windowing.py:373-377`, `run_config.py:58-59`), which is
   40,000 at two chunks, so this judge is skipped for every session that needs
   more than one 14,800-token window.
3. Confirmed in code: the per-model `raw_window_target_tokens`
   (`run_config.py:101`) can only lower the size-tier target
   (`windowing.py:321-322`), and the reserve and merge arithmetic ignore it.
   **Inference:** calibrated budgets need fields that can also raise targets and
   describe question bundles and whole-thread budgets.
4. Confirmed in the adapter contract and code: a CES compaction step stores only
   its `trigger` (`weave-agent-adapter/specs/EVALUATIONS_CASE_CONTRACT.md:30`),
   and Monopole keeps only its timestamp (`src/weave_agent_signals/ces_client.py:272-276`),
   so the agent's own compaction summaries never reach a judge.

## Open questions

These are choices for the user; the proposal only suggests defaults.

1. May a model with a tiny effective context write thread verdicts itself, or
   must verdicts go to the panel member with the largest budget?
2. Should measured budgets become pinned catalog data, re-measured when the
   catalog, the renderer, or a model version changes?
3. Rubric before and after the transcript, which may help attention [5], or
   rubric last only, which keeps the cached beginning shared across rubrics?
4. Are reminders inserted between steps acceptable, given that they change the
   transcript the judge sees?
5. Should a navigating investigator stay escalation-only and audit-only, as the
   draft design rules for the tool-using judge
   (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:49`,
   `:155-156`)?
6. Should Monopole ever judge prefixes of a live session as CES receives its
   turns, or stay with complete sessions (`specs/02-evaluation.md:8-13`)?
7. Should a panel diversify reading plans as well as model families?
8. If the user picks only an 8k-class model and nothing larger to escalate to,
   should the run proceed with plan Q and label its verdicts as such, or refuse?

## Sources

All accessed 2026-11-03.

Read for this document:

1. Xiao et al., Efficient Streaming Language Models with Attention Sinks
   (StreamingLLM). https://arxiv.org/abs/2309.17453
2. MIT Han Lab, streaming-llm repository README, FAQ. https://github.com/mit-han-lab/streaming-llm
3. Xiao et al., InfLLM: Training-Free Long-Context Extrapolation for LLMs with an
   Efficient Context Memory. https://arxiv.org/abs/2402.04617
4. Hugging Face, Transformers documentation, Perplexity of fixed-length models.
   https://huggingface.co/docs/transformers/perplexity
5. Liu et al., Lost in the Middle: How Language Models Use Long Contexts,
   section 4.2. https://arxiv.org/abs/2307.03172
6. Zhang et al., Chain of Agents: Large Language Models Collaborating on
   Long-Context Tasks, Tables 7–8 and Appendix D. https://arxiv.org/abs/2406.02818
7. Zhou et al., LLM×MapReduce: Simplified Long-Sequence Processing using Large
   Language Models, Table 2 and section 3.4. https://arxiv.org/abs/2410.09342
8. Smith and Troynikov, Evaluating Chunking Strategies for Retrieval (Chroma
   technical report, 2024-07-03). https://research.trychroma.com/evaluating-chunking
9. Bennani and Moslonka, A Systematic Analysis of Chunking Strategies for
   Reliable Question Answering. https://arxiv.org/abs/2601.14123
10. Prompt Repetition Improves Non-Reasoning LLMs. https://arxiv.org/abs/2512.14982
11. Anthropic, Prompting best practices, "Long context prompting".
    https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
12. Martin and Roger, Classifier Context Rot: Monitor Performance Degrades with
    Context Length. https://arxiv.org/abs/2605.12366
13. Pydantic AI Harness, Trajectory Judge. https://pydantic.dev/docs/ai/harness/trajectory-judge/
14. Inspect Scout, Multi Agent (context chunking, reducers, utility spans).
    https://meridianlabs-ai.github.io/inspect_scout/multi_agent.html
15. Zhu et al., SAFARI: Scaling Long Horizon Agentic Fault Attribution via Active
    Investigation. https://arxiv.org/abs/2606.24626
16. Zhou et al., MEM1: Learning to Synergize Memory and Reasoning for Efficient
    Long-Horizon Agents. https://arxiv.org/abs/2506.15841
17. Ye et al., AgentFold: Long-Horizon Web Agents with Proactive Context
    Management. https://arxiv.org/abs/2510.24699
18. Sun et al., Scaling Long-Horizon LLM Agent via Context-Folding,
    https://arxiv.org/abs/2510.11967, with the results table on the project page
    https://context-folding.github.io/
19. Wu et al., ReSum: Unlocking Long-Horizon Search Intelligence via Context
    Summarization. https://arxiv.org/abs/2509.13313
20. IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling,
    https://arxiv.org/abs/2511.07327, and repository README
    https://github.com/Chen-GX/IterResearch
21. Rodkin et al., Associative Recurrent Memory Transformer.
    https://arxiv.org/abs/2407.04841
22. Zhang, Kraska, Khattab, Recursive Language Models (version 3, 2026-05-11),
    Table 1 and Appendix A. https://arxiv.org/abs/2512.24601
23. Gandhi et al., When Agents go Astray: Course-Correcting SWE Agents with PRMs.
    https://arxiv.org/abs/2509.02360
24. Zheng et al., ProcessBench: Identifying Process Errors in Mathematical
    Reasoning. https://arxiv.org/abs/2412.06559
25. Fan et al., AgentProcessBench: Diagnosing Step-Level Process Quality in
    Tool-Using Agents (KDD 2026). https://arxiv.org/abs/2603.14465
26. StrictPRMBench: A Replayable Evaluation Benchmark for Auditing Fixed-Score PRM
    Aggregation (abstract). https://arxiv.org/abs/2610.41333
27. Verga et al., Replacing Judges with Juries: Evaluating LLM Generations with a
    Panel of Diverse Models (PoLL), Tables 1–2. https://arxiv.org/abs/2404.18796
28. Kim et al., Correlated Errors in Large Language Models (ICML 2025).
    https://arxiv.org/abs/2506.07962
29. NVIDIA, RULER repository README, results table. https://github.com/NVIDIA/RULER
30. Kuratov et al., BABILong: Testing the Limits of LLMs with Long Context
    Reasoning-in-a-Haystack. https://arxiv.org/abs/2406.10149

Read for the companion documents, cited here with the companion line:

31. Modarressi et al., NoLiMa: Long-Context Evaluation Beyond Literal Matching,
    https://arxiv.org/abs/2502.05167, and README https://github.com/adobe-research/NoLiMa
32. Tang et al., LongRM: Revealing and Unlocking the Context Boundary of Reward
    Modeling. https://arxiv.org/abs/2510.06915
33. Mittapalli et al., TRACE: Trajectory Reasoning through Adaptive Cross-Step
    Evidence Aggregation for LLM Agents. https://arxiv.org/abs/2606.07054
34. Kale et al., Reliable Weak-to-Strong Monitoring of LLM Agents.
    https://arxiv.org/abs/2508.19461
35. Madvil et al., Holistic Evaluation and Failure Diagnosis of AI Agents.
    https://arxiv.org/abs/2605.14865
36. Deshpande et al., TRAIL: Trace Reasoning and Agentic Issue Localization.
    https://arxiv.org/abs/2505.08638
37. Zhang et al., Which Agent Causes Task Failures and When?
    https://arxiv.org/abs/2505.00212
38. Yu et al., MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based
    Memory Agent. https://arxiv.org/abs/2507.02259
39. Lam, How Far Can You Trust a Step-Level Judge? (judgeman working paper,
    2026-10-14). https://doi.org/10.5281/zenodo.23354648
40. Wei et al., RocketEval: Efficient Automated LLM Evaluation via Grading
    Checklist. https://arxiv.org/abs/2503.05142
41. Lee et al., CheckEval. https://arxiv.org/abs/2403.18771
42. Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement.
    https://arxiv.org/abs/2407.18370
43. Zhuge et al., Agent-as-a-Judge: Evaluate Agents with Agents.
    https://arxiv.org/abs/2410.10934
44. Zhai et al., GUIDE: Interpretable GUI Agent Evaluation via Hierarchical
    Diagnosis. https://arxiv.org/abs/2604.04399
45. Snell et al., Scaling LLM Test-Time Compute Optimally can be More Effective
    than Scaling Model Parameters. https://arxiv.org/abs/2408.03314
46. Du et al., Context Length Alone Hurts LLM Performance Despite Perfect
    Retrieval. https://arxiv.org/abs/2510.05381
47. Hong, Troynikov, Huber, Context Rot (Chroma technical report).
    https://research.trychroma.com/context-rot
48. Bertsch et al., Oolong: Evaluating Long Context Reasoning and Aggregation
    Capabilities. https://arxiv.org/abs/2511.02817
49. Bai et al., LongBench v2. https://arxiv.org/abs/2412.15204
50. Yen et al., HELMET: How to Evaluate Long-Context Language Models Effectively
    and Thoroughly. https://arxiv.org/abs/2410.02694
51. Peng et al., Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?
    (RuVerBench). https://arxiv.org/abs/2606.29920
52. Zha et al., AlignScore. https://arxiv.org/abs/2305.16739
53. Wu et al., Recursively Summarizing Books with Human Feedback.
    https://arxiv.org/abs/2109.10862
54. Chang et al., BooookScore. https://arxiv.org/abs/2310.00785
55. Lee et al., A Human-Inspired Reading Agent with Gist Memory of Very Long
    Contexts (ReadAgent). https://arxiv.org/abs/2402.09727
56. Graph of Agents: Principled Long Context Modeling by Emergent Multi-Agent
    Collaboration. https://arxiv.org/abs/2509.21848
57. LiteTrajEval: Lightweight, Rubric-Guided Trajectory Evaluation for Production
    AI Agents. https://arxiv.org/abs/2610.03315
58. SKG-Eval: Stateful Evaluation of Multi-Turn Dialogue via Incremental Semantic
    Knowledge Graphs. https://arxiv.org/abs/2605.16650
59. Pilditch et al., Transect: Retaining Observability for Long-Horizon LLM Agent
    Evaluations. https://arxiv.org/abs/2610.08364
60. Web-Shepherd: Advancing PRMs for Reinforcing Web Agents.
    https://arxiv.org/abs/2505.15277
61. Lindenbauer et al., The Complexity Trap: Simple Observation Masking Is as
    Efficient as LLM Summarization for Agent Context Management.
    https://arxiv.org/abs/2508.21433
62. Kang et al., ACON: Optimizing Context Compression for Long-horizon LLM Agents.
    https://arxiv.org/abs/2510.00615
63. Zhao et al., LongAgent: Scaling Language Models to 128k Context through
    Multi-Agent Collaboration. https://arxiv.org/abs/2402.11550
64. Chen et al., Walking Down the Memory Maze (MemWalker).
    https://arxiv.org/abs/2310.05029
65. AutoEval Done Right: Using Synthetic Data for Model Evaluation.
    https://arxiv.org/abs/2403.07008
66. AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories.
    https://arxiv.org/abs/2504.08942
67. Anthropic, Prompt caching. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
68. Zhang et al., LongReward: Improving Long-context Large Language Models with AI
    Feedback. https://arxiv.org/abs/2410.21252
69. Tree of Agents: Improving Long-Context Capabilities of Large Language Models
    through Multi-Perspective Reasoning. https://arxiv.org/abs/2509.06436
70. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving
    Performance. https://arxiv.org/abs/2305.05176
71. LLM-AggreFact leaderboard. https://llm-aggrefact.github.io/
72. An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge
    Model is not a General Substitute for GPT-4. https://arxiv.org/abs/2403.02839
