# Handling the largest sessions with any judge model

Parts 2 and 3 of the large-session study, measured 2026-11-03 from prod CES, read-only.
Part 1, [large-session-anatomy.md](large-session-anatomy.md), measured what makes the four
largest sessions large. This document simulates how Monopole would judge them with models of
32k to 1M tokens of context (Part 2), and collects what others report about oversized turns and
huge tool outputs (Part 3). It builds on
[small-model-judging-research.md](small-model-judging-research.md),
[windowed-judging-research.md](windowed-judging-research.md),
[long-session-judging-research.md](long-session-judging-research.md), and the draft design
([2026-11-03-long-session-judging-design.md](superpowers/specs/2026-11-03-long-session-judging-design.md)),
and does not repeat them.

Every external claim cites a numbered source; all were accessed 2026-11-03. Repository claims
cite `path:line`. Text marked **Inference** is reasoning from cited facts or from the simulation's
stated assumptions. Only aggregate statistics are reported.

## Summary

- **32k** (the catalog's Qwen3 14B, `catalogs.py:68-74`): only 126 of the Workflow session's
  605 subagents fit whole. Use the decomposed pipeline: every request fits, at 487 to 3,583
  requests for the single-thread sessions and 28,537 (about 18 hours) for the Workflow session.
  Quality should hold for yes/no questions on 4k of evidence and drop for thread verdicts and
  claim extraction written by a 14B model, which no source supports.
- **131k and 200k** (GPT OSS, Llama, Granite; Claude Haiku 4.5): the main threads need 4 to 36
  windows of up to 90k (131k) or 146k (200k). The Workflow session takes 3,150 to 3,500 requests
  and 9 to 10 hours. For 8B to 70B models, use the decomposed pipeline instead: a 90k window is
  45 to 90 times their measured effective length.
- **262k and 272k** (W&B Qwen, Kimi, GLM, Nemotron; Codex GPT-5.x): every subagent fits whole.
  The main threads need 4 to 17 windows of up to about 200k. The Workflow session takes 3,121
  requests, about 9 hours, or about twice that under Codex's limit of 2 concurrent calls. Quality
  should hold best for the whole-subagent verdicts and is unproven for the 110k–200k main-thread
  windows.
- **1M** (Claude Sonnet 5 at its vendor size, Gemini, DeepSeek V4): two of the four sessions, C
  and D, fit whole,
  6 requests each, minutes. Judge the other two, A and B, one compaction stretch per window: 18
  windows of at most 211k for A and 9 of at most 371k for B's orchestrator. Each window then
  holds roughly what the agent held at that time.
- **Everywhere:** no raw step is larger than about 10k tokens. Splitting at step boundaries
  therefore always works, and only host-added subagent summaries ever need a step split.
- **Literature:** for software agents, hiding old tool output matched LLM summaries, and the
  summaries hid failure signals [12]. A judge shown only the final reply missed faults that the
  same judge caught step by step [15]. So judges should read raw steps, each in exactly one
  window, rather than summaries of them.

## Decisions and additions

Decided by the user:

1. Monopole judges large sessions with whatever model the user picks, from 32k to 1M tokens of
   context. Cost is not a concern.
2. A thread that fits is judged whole. A thread that doesn't is judged in windows, or decomposed
   as in the small-model study.
3. Budget = context minus about 18,000 reserved tokens: 14,768 for 32k; 113,072 for 131k;
   244,000 for 262k; 254,000 for 272k; 982,000 for 1M.
4. Earlier choices still hold. Judges see every message, a Workflow session gets session,
   orchestrator and subagent scores, and finished work is kept on reruns
   (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:202-218`).

Added here, each open to change:

- **Window core:** a window's own steps fill at most 80% of the budget. The rest holds read-only
  overlap and the ledger.
- **One reviewer:** all counts are for one judge. A three-judge panel triples them.
- **200k class:** added because the catalog's Claude entries are 200k (`catalogs.py:193-201`,
  `:262-270`).
- **Latency model:** an assumption, since Monopole has no per-model timing yet (Part 2 method).
- **Claim counts:** a word-pattern proxy, since claims come from a model in the real pipeline.
- **Session D snapshot:** D has grown since Part 1, from 709 to 816 Cases and from 4 to 6 turns.
  The simulation uses its first 4 turns, the 709 Cases Part 1 measured, so the two parts stay
  comparable.

## Part 2: simulation on the four sessions

### Method

- **Data and rendering:** Part 1's loaders and renderers, reused unchanged, reading CES
  read-only. Sizes use the compressed rendering, which is the draft design's piece 3
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:94-102`). It matches Part 1:
  2.13M (A), 28.7M (B, all threads), 562k (C), 325k (D).
- **Orchestrator view:** B's main-thread steps plus 300 tokens of verdict summary for each
  subagent at the call that launched it, 1.83M in total. Part 1's figure of 1.86M also counted
  one line per subagent.
- **Whole fit:** a thread is judged whole when its rendered size fits the budget.
- **Windows:** a thread that doesn't fit is cut at compaction stretches first, then turns, then
  steps. Each cut happens only where a unit is too big. Consecutive whole units are packed
  together. A step larger than a window is split into numbered pieces. Each step is in exactly
  one window core.
- **Cross-window pairs:** the small-model study's pairing rules
  (`docs/small-model-judging-research.md:627-632`), counted where the two ends of a pair fall in
  different windows.
- **Code-derived facts:**
  - Checks: commands that the repo's `classify_command` labels as test, build or lint
    (`src/weave_agent_signals/scorers/outcome.py:307-335`). For Claude Code this is Bash's
    `command`. For Codex it is the `cmd` strings inside its JavaScript tool programs (see
    Observations).
  - Errors: failed calls, error results, or nonzero exit codes.
  - Edits: file-editing tools, Codex `apply_patch`, and `sed -i`.
  - Corrections: steering messages and denials.
  - Claims: assistant messages containing completion words such as "fixed", "passes" or
    "verified". This proxy over-counts and does not split messages that hold several claims.
- **Question evidence:** bundles as specified in the small-model study
  (`docs/small-model-judging-research.md:679-686`), plus 200 tokens of framing, against the 4k
  cap.
- **Requests:**
  - Windowed threads follow the windowed study's formula
    (`docs/windowed-judging-research.md:712-719`): one ledger request per window, one rubric
    request per rubric per window, one resolution request per batch of pairs, and merges.
  - Rubrics per thread: 6 for a single-thread session, 5 for a subagent, 4 for the orchestrator,
    plus 6 session merges for a Workflow session.
  - At 32k, today's 11,000 tokens of findings and digest per window
    (`src/weave_agent_signals/judges/windowing.py:373-377`) can't be merged in a 14,768-token
    budget. The simulation caps them at a quarter of the budget, so merges combine 4 windows at a
    time.
  - The pipeline follows the small-model study's steps
    (`docs/small-model-judging-research.md:618-760`). Two judges answer each question, and 20%
    of verdicts escalate (`:789-790`).
- **Wall time:**
  - Each request takes 2 s + input ÷ 10,000 tokens/s + output ÷ 60 tokens/s.
  - Assumed outputs: 200 tokens for a label or question, 400 for an outcome question, 800 for
    extraction, 1,500 for a ledger, merge or verdict, and 2,000 for window findings or a whole
    verdict. Hidden reasoning is excluded.
  - Requests run on 4 concurrent slots and respect dependencies: subagents before the
    orchestrator, ledgers before rubric windows. Earlier studies assumed flat 4 s and 30 s per
    request instead (`docs/small-model-judging-research.md:809-811`,
    `docs/windowed-judging-research.md:733-734`).
  - Ranges use a slow provider (5,000 and 40 tokens/s) and a fast one (20,000 and 120).
  - W&B allows 4 concurrent calls and Codex 2 (`specs/02-evaluation.md:243-245`).

### 1. Whole fit and structural windows

Cells show the window count, then the median and largest window. "Whole" means one request per
rubric.

| Thread (size) | 32k | 131k | 200k | 262k | 272k | 1M |
| --- | --- | --- | --- | --- | --- | --- |
| A (2.13M) | 226 (10k / 12k) | 36 (68k / 90k) | 26 (78k / 142k) | 17 (132k / 193k) | 16 (148k / 198k) | 3 (695k / 749k); or 18, one per stretch (120k / 211k) |
| B orchestrator (1.83M) | 181, plus 12 pieces | 27 (89k / 90k) | 18 (113k / 146k) | 16 (113k / 195k) | 16 (113k / 203k) | 3 (667k / 692k); or 9, one per stretch (211k / 371k) |
| B subagents (605; median 25.5k, largest 150k) | 126 whole; 479 in 2,926 windows (median 6 and up to 15 each) | 588 whole; 17 in 49 windows | all whole | all whole | all whole | all whole |
| C (563k) | 54 | 9 (79k / 90k) | 6 (112k / 145k) | 5 (145k / 192k) | 5 (135k / 203k) | whole |
| D (325k) | 33 | 6 (62k / 90k) | 4 (90k / 145k) | 4 (65k / 195k) | 4 (61k / 203k) | whole |

Cross-window pairs, which the resolution step must judge:

| Thread (all pairs) | 32k | 131k | 200k | 262k | 272k | 1M |
| --- | --- | --- | --- | --- | --- | --- |
| A (1,199) | 896 | 616 | 571 | 501 | 495 | 282; 515 per stretch |
| B orchestrator (487) | 269 | 161 | 122 | 117 | 118 | 20; 69 per stretch |
| B subagents (3,037) | 1,741 | 37 | 0 | 0 | 0 | 0 |
| C (365) | 241 | 104 | 51 | 68 | 68 | 0 |
| D (57) | 30 | 9 | 7 | 7 | 7 | 0 |

- **Subagent sizes:** the median and largest subagent sizes come from Part 1
  (`docs/large-session-anatomy.md:104-118`).
- **The 12 pieces at 32k:** five orchestrator calls each launched 36 to 100 subagents. With
  their summaries, those five steps are 12.5k to 32.2k tokens each, more than a 32k window core
  of 11.8k holds. These are the only oversized steps found.
- **Window sizes at 32k:** windows are 10k to 12k tokens everywhere.

### 2. Largest single steps

Tokens in the compressed rendering. In the every-message rendering, the same steps are at most
3% larger for messages and 12% larger for tool calls.

| | A | B main | B subagents | C | D |
| --- | --- | --- | --- | --- | --- |
| Largest user message | 1,633 | 1,711 | 10,164 | 100 | 676 |
| Largest assistant message | 7,071 | 3,800 | 9,612 | 2,089 | 2,716 |
| Largest tool call | 4,659 | 4,350 | 4,467 | 4,059 | 4,090 |
| Tool calls over 4k tokens | 18 | 7 | 68 | 1 | 2 |
| Messages with text: count, tokens | 217, 70.8k | 411, 137k | 1,500, 3.54M | 204, 56.7k | 79, 40.2k |
| Messages over 8k tokens | 0 | 0 | 38 | 0 | 0 |
| Tool results the adapter marked truncated | 101 of 1,458 | 21 of 1,735 | 623 of 16,447 | 0 | 0 |

- **No raw step exceeds the smallest budget, 14,768.** Tool calls stay under about 5k tokens,
  because the renderer cuts each argument and result at 12,000 characters
  (`windowing.py:17`). The largest argument found was 3,991 tokens and the largest result 3,996.
  Only one call had both above 2k.
- **The largest steps are messages in subagents:** delegated task prompts and returned reports
  of up to about 10k tokens.
- **A split between messages and tool calls**, as the small-model pipeline does it:
  - Tool calls: every call fits a step-label request at every context size, including 32k
    (4.7k step + 3k context + 1k prompt).
  - Messages: extraction runs of 16k need no splits. The 8k runs that a 32k model needs would
    split 38 subagent messages into two pieces each.
  - This study's step assumption: the small-model study assumed steps of 4k or less
    (`docs/small-model-judging-research.md:645`, `:767`). 96 calls exceed that, by at most 0.7k.

### 3. What code can count

| | A (Codex) | B orchestrator | B subagents (605) | C | D |
| --- | --- | --- | --- | --- | --- |
| Checks (test, build, lint) | 97: pytest 43, make 52, lint 2 | 8: builds 7, test 1 | 1,541: go test 1,250, golangci-lint 179, builds 112 | 43 | 0 |
| Checks whose result code can't read | 15 | 0 | 0 | 0 | 0 |
| Edits (file unknown) | 76 (0) | 242 (8) | 469 (270) | 41 (31) | 62 (2) |
| Errors | 82 | 38 | 442 | 11 | 4 |
| User messages, steering, denials | 67, 4, 0 | 13, 2, 2 | 1,053, 0, 45 | 6, 0, 1 | 5, 1, 0 |
| Subagent launches | 0 (1 Codex spawn, no thread) | 605 in 52 calls | 0 | 0 | 0 |
| Claim proxy, and claims with no check | 62, 4 | 113, 25 | 239, 60 | 90, 2 | 29, 29 |
| Checks paired per claim: median, p90, largest | 1, 15, 87 | 1, 1, 5 | 3, 8, 18 | 2, 4, 27 | 0 |

- **Check results:** Claude Code reports every check's result through the call status.
  15 of A's checks have neither an exit code nor a parseable summary, so each needs a "did it
  pass?" question.
- **Edits with no file named:** mostly `sed -i`. They pair with no claim or constraint.
- **The orchestrator's own claims:** 25 have no check in its own thread. Checks inside the
  subagents it launched could settle them, through the cross-thread pair the windowed study
  proposes (`docs/windowed-judging-research.md:615`). That pair type is not counted here.
- **Long check runs:** A sometimes runs many checks between an edit and a claim. Asking about
  every one gives one claim up to 87 relevance questions.

### 4. Question evidence under the 4k cap

| | A | B orchestrator | B subagents | C | D |
| --- | --- | --- | --- | --- | --- |
| Questions (each answered by two judges) | 1,011 | 433 | 2,092 | 342 | 45 |
| Bundle median / p90 | 1.2k / 3.9k | 2.1k / 3.5k | 3.5k / 8.4k | 1.4k / 2.4k | 1.2k / 1.6k |
| Bundles over 4k | 9.7% | 3.9% | 48.2% | 0.3% | 0% |
| Main cause | autonomy: 68% of its bundles | verification: 8% | verification: 93%; state consistency: 18 of 36 | — | — |
| Largest bundle of any kind after quoting the claim instead of the whole message, and sending only the last 3 user messages | 3.9k | 3.6k | 5.7k: 18 state-consistency bundles stay over 4k | 2.5k | 2.2k |

- **Verification:** a subagent's claims sit in its final report, which has a median of 5,886
  tokens. Sending the whole message, as the bundle definition says
  (`docs/small-model-judging-research.md:681`), overflows 4k. Quoting the claim sentence keeps
  every verification bundle under 4k.
- **Autonomy:** "all earlier user messages" (`:686`) grows with the session. A has 67 user
  messages, so most of its autonomy bundles overflow.
- **Middle of long outputs:** the 2k head-and-tail cap on outputs cuts the middle of 6 of A's 97
  check results. It also cuts 87 of the 442 error outputs in B's subagents. Those are the
  questions where the deciding line could fall in the cut.

### 5. Small-model pipeline requests per step and per rubric

For 131k and larger models; values for 32k are in brackets where they differ. Question rows
count requests, with two judges per question.

| Step | A | B orchestrator | B subagents | C | D |
| --- | --- | --- | --- | --- | --- |
| Step labels | 1,458 | 1,735 | 16,447 | 678 | 377 |
| Extraction runs, 16k [8k] | 9 [18] | 14 [27] | 611 [858] | 6 [11] | 4 [8] |
| Verification questions | 1,420 | 426 | 2,132 | 594 | 58 |
| Error-recovery questions | 328 | 152 | 1,768 | 44 | 16 |
| Tool choice: destructive-command checks | 0 | 32 | 212 | 24 | 0 |
| State-consistency questions | 4 | 208 | 72 | 0 | 0 |
| Autonomy questions | 270 | 48 | not scored | 22 | 16 |
| Outcome questions (one mid-size request each) | 63 | 11 | 605 | 6 | 4 |
| Verdicts, including grouped ones | 8 [18] | 31 [61] | 3,025 | 6 [8] | 6 |
| Escalations, plus B's session merges | 2 [4] | B total: 620 [634] escalations, 42 [84] session merges | | 2 | 2 |
| Total | 3,562 [3,583] | B total: 28,191 [28,537] | | 1,382 [1,389] | 483 [487] |

- **Grouped verdicts:** the orchestrator's verdict inputs are about 200k tokens each, mostly 605
  subagent summaries. They are therefore written in groups of 32k (or 14.7k at 32k) before a
  final verdict. Most subagents' verdict inputs are 3k to 8k tokens.
- **Measured against the small-model study's estimate:** that study assumed 10 questions per
  subagent and 300 for the orchestrator, for about 35,500 requests in total
  (`docs/small-model-judging-research.md:784-801`). The real data give 3.5 per subagent and 433
  for the orchestrator, for 28,191 requests. Step labels are 65% of the total.
- **Per subagent:** a typical subagent makes 12 tool calls (`docs/large-session-anatomy.md:116`).
  The pipeline therefore spends about 26 requests on it, against 5 for judging it whole.
  **Inference**, from the averages above.

### 6. Requests and wall time per context size

One reviewer, at 4 concurrent calls. Times are the base estimate, with the slow-to-fast range in
brackets for the Workflow session.

| Session | Plan | 32k | 131k | 200k | 262k | 272k | 1M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | whole or windows | 2,492, 5.8 h | 313, 0.9 h | 229, 0.7 h | 151, 0.5 h | 143, 0.5 h | 42, 0.25 h; per stretch 158, 0.65 h |
| A | pipeline | 3,583, 1.5 h | 3,562, 1.5 h | same | same | same | same |
| B | whole or windows | 26,373, 62 h [33–92] | 3,504, 9.8 h [5–15] | 3,152, 8.9 h | 3,121, 8.8 h [4.6–13.6] | same | 3,051, 8.7 h [4.5–13.4]; per stretch 3,084, 8.8 h |
| B | pipeline | 28,537, 18 h [11–25] | 28,191, 17.6 h [10.7–24.8] | same | same | same | same |
| C | whole or windows | 613, 1.4 h | 75, 0.23 h | 51, 0.17 h | 45, 0.16 h | 45, 0.16 h | whole: 6, 0.05 h |
| C | pipeline | 1,389, 0.6 h | 1,382, 0.6 h | same | same | same | same |
| D | whole or windows | 324, 0.8 h | 49, 0.15 h | 35, 0.12 h | 35, 0.12 h | 35, 0.12 h | whole: 6, 0.04 h |
| D | pipeline | 487, 0.2 h | 483, 0.2 h | same | same | same | same |

- **Whole session in one request per rubric:** only C and D fit, and only at 1M.
- **B's mix:** B's subagents are judged whole from 200k up, and 588 of them at 131k. Its
  orchestrator never fits.
- **What the time measures:** with thousands of requests, wall time is about total request time
  ÷ 4. The four slots, not the context size, set the time. Codex's limit of 2 roughly doubles
  each figure.

### What Part 2 shows

All **Inference**, from the tables above.

- **32k:** windowing is the wrong tool at this size. B needs 26,000 requests and 62 hours, and
  merges four levels deep. The pipeline fits every request and takes about 18 hours.
- **131k and above:** whole-or-windows needs at least 8 times fewer requests than the pipeline,
  and at most two-thirds of its wall time. What decides is quality per window, which no study
  measures on coding sessions.
- **1M:** per-stretch windows cost almost nothing over packed windows (3,084 against 3,051
  requests for B). They keep each window at or below 371k instead of 692k. They also match what
  the agent held between its own compactions.
- **Pipeline inputs to fix:** the claim quote, a cap on checks per claim, and recent user
  messages for autonomy. With these, all but 18 questions fit 4k; those 18 are state-consistency
  questions whose constraint is a subagent's long task prompt. Without them, 48% of B's subagent
  questions overflow.

## Part 3: how others handle oversized turns and huge tool outputs

The companion studies cite observation masking [12], ACON [13], and Anthropic's tool-result
clearing in one line each (`docs/long-session-judging-research.md:229-235`). This part adds the
mechanics and defaults, the measured effects, and the evidence on judges.

### Vendor mechanisms

| Mechanism | Who uses it | What remains in context | Defaults | Source |
| --- | --- | --- | --- | --- |
| Tool-result clearing | Anthropic API (`clear_tool_uses_20250919`) | Every tool call. The oldest results are replaced with placeholder text, and the last 3 tool uses are kept. | Triggers at 100,000 input tokens. Calls are cleared only if `clear_tool_inputs` is set. Runs on the server; the client keeps the full history. | [1] |
| Tool-result clearing | Claude Code | Not documented beyond the fact that it happens. The cost display counts "clearing old tool results from context" as an expected cache rebuild. | — | [6] |
| Compaction (model-written summary) | Anthropic API (`compact_20260112`) | A summary block. The API drops everything before it. | Triggers at 150,000 input tokens (at least 50,000). The default prompt asks for "state, next steps, learnings". | [2][3] |
| Compaction | Claude Code | A summary. Docs: it "keeps your requests and key code snippets, but detailed instructions from early in the conversation may be lost". | Automatic near the limit; idle compaction from 200,000 tokens; `/compact` can be given a focus. | [4][5] |
| Compaction | Codex | A handoff summary plus the most recent user messages, up to 20,000 tokens of them. | The model's default threshold, configurable. | [7][8][9] |
| Output limits | Claude Code | Output up to the limit; the docs read don't say which part is kept | Bash 30,000 characters; file reads 25,000 tokens; MCP results 25,000 tokens | [5] |
| Output limits | Codex | Start and end of each output. The middle is cut, with a header giving the original token count. | 10,000 tokens per output in every listed model | [10][11] |

### Masking versus summarization for software agents

- **The Complexity Trap** [12] ran SWE-agent on SWE-bench Verified with five model
  configurations, with up to 250 turns.
  - Masking hid observations older than the last 10 turns behind a placeholder.
  - Summarization condensed 21 turns at a time and kept the last 10.
  - Masking halved cost against the raw agent and matched or slightly beat summarization's
    solve rate. With Qwen3-Coder 480B it was 52% cheaper than the raw agent, with a 2.6% higher
    solve rate (v3 text).
  - Summarization made trajectories longer, by 15% against the raw agent and 13% against masking
    with Qwen3-Coder. The authors read this as summaries hiding "failure signals that would
    otherwise prompt earlier termination".
  - A hybrid of the two cut cost a further 7% to 11%.
- **ACON** [13] learns compression guidelines from the agent's failures. It cut peak tokens by
  26% to 54% while raising task success on AppWorld, OfficeBench and multi-objective QA. That is
  not software engineering.
- **Measure Before You Manage** [14] studied 55 archived coding-agent trajectories.
  - Instructions, artifacts, tool outputs and the agent's own notes were retained and compressed
    differently.
  - Gains from calibrating a compression policy did not always carry over to new tasks.
  - The same token budget did not deliver the same context.

### Judging with masked, truncated or summarized evidence

No source measures rubric-judge agreement on coding transcripts as tool output is masked,
truncated or summarized. The closest evidence:

- **Outcome-only view** [15]: a support-desk testbed of 400 exactly labeled trajectories.
  - A 14B judge shown only the request and the final reply could not tell faulty runs from clean
    ones when the fault left the reply unchanged. Its "recall" equaled its rate of flagging
    clean runs.
  - Told to check each step, the same judge flagged every such fault and 0 of 100 clean runs.
- **Subsampled observations** [16]: on 931 mobile-agent trajectories, the number of screenshots
  was the input that mattered most. At equal cost, 16 screenshots at quarter resolution beat 192
  at one sixty-fourth (86.5% against 82.3% with Qwen, 91.2% against 86.6% with GPT).
- **Budgeted compaction** [17]: LiteTrajEval warns that "aggressively truncating it can break
  the causal chain".
  - It compacts by tool status, shrinks the longest outputs first, and picks evidence inside a
    step.
  - It improved failure localization by 20 to 35 points on Magentic-One trajectories against a
    baseline.
- **Reasoning over evidence** [18]: rewriting agents' reasoning, with actions and observations
  unchanged, raised judges' false positives by up to 90%.
- **Granularity and detectors** [19][20]:
  - Fine-grained judging beat holistic judging by over 30 points in several settings [19].
  - A trace-level verification detector agreed with human review on 67 of 81 traces [20].

### What each method loses for a judge

| Method | Lost | Risk for Monopole's rubrics | Evidence |
| --- | --- | --- | --- |
| Clearing or masking old tool output | old results; calls remain | Verification and error recovery depend on what the results said. | agents unharmed [12]; no judge data |
| Model-written summary | details and failure signals | Error recovery and verification credit work the summary smooths over. | longer trajectories, hidden failures [12] |
| Head-and-tail truncation | the middle of long outputs | A failing test in the middle. Monopole keeps diagnostic lines from the middle (`windowing.py:55-101`). | causal chains break [17] |
| Outcome-only or reply-only view | every step | Process rubrics become guesses. | zero discrimination [15] |

**Inference:**

- **What the judge reads:** keep raw steps, each in exactly one window. Use summaries only as
  read-only context, as the draft design does
  (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:126-129`).
- **What CES already holds:** the agents' own truncation is in CES already. The adapter marked
  623 of the 16,447 tool results in B's subagents as truncated, and Codex keeps 10,000 tokens per
  output [11]. Monopole's 12,000-character cut comes on top.
- **What the agent knew:** after each compaction the agent no longer had the earlier details,
  only the summary. A judge reading across a compaction boundary can blame the agent for losing
  details it was not shown. Per-stretch windows avoid that.

## Per-model reading plan

How a plan forms once the user picks a model. This is proposed here and open to change. Steps 1
to 4 reuse existing pinned inputs; the other steps are new.

1. **Budget:** the model's pinned context capacity minus the reserve
   (`src/weave_agent_signals/run_config.py:50-57`; this study used 18,000 throughout, as asked).
2. **Reading size:** the largest window this model reads well. Use the catalog's per-model
   window target where one exists (`run_config.py:101`, `specs/02-evaluation.md:57-60`);
   otherwise use the budget until the experiment measures one.
3. **Thread tree:** subagents first. The orchestrator view carries each subagent's verdict
   summary at its launch call.
4. **Per thread:** judge it whole if the rendered thread fits the budget. Otherwise cut
   structural windows with cores of at most 80% of the reading size, at compaction stretches
   first.
5. **Decomposed instead of windows** when any of these holds:
   - the model is 70B or smaller and has no measured effective length above the window size;
   - merges would need more than two levels, as for every main thread here at 32k;
   - the user selects it.
6. **Requests and time:** count requests by the formulas above. Divide the estimated request
   time by the provider's concurrent-call limit (W&B 4, Codex 2).
7. **What setup shows:** for each session and model, the plan, request count, time, and quality
   note, as in the table below. That replaces today's yes/no "fits" check
   (`specs/02-evaluation.md:224-236`), and resolves the draft design's open question in favor of
   "windows count" (`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:242-249`).
   **Inference** from the user's goal.

What setup would show for the two largest sessions:

| Model chosen (example) | Session B plan | Session A plan | Requests, B / A | Time, B / A | Quality note |
| --- | --- | --- | --- | --- | --- |
| Qwen3 14B, 32,800 | decomposed | decomposed | 28,537 / 3,583 | 18 h / 1.5 h | questions supported; verdicts and extraction by this model unvalidated |
| GPT OSS 120B, 131,072 | 588 subagents whole, 17 in 49 windows; orchestrator in 27 windows | 36 windows up to 90k | 3,504 / 313 | 10 h / 0.9 h | no effective-length data; decomposed is the fallback |
| Claude Haiku 4.5, 200k | all subagents whole; orchestrator 18 windows up to 146k | 26 windows up to 142k | 3,152 / 229 | 9 h / 0.7 h | Claude CLI concurrency not specified |
| Kimi K2.6, 262k | all whole; orchestrator 16 windows up to 195k | 17 windows up to 193k | 3,121 / 151 | 9 h / 0.5 h | windows in the range where long-context quality drops |
| GPT-5.6 Sol, 272k (Codex) | as above; 16 windows up to 203k | 16 windows up to 198k | 3,121 / 143 | about 18 h / 1 h | 2 concurrent calls |
| A 1M model, for example DeepSeek V4 Flash | all whole; orchestrator 9 stretch windows up to 371k | 18 stretch windows up to 211k | 3,084 / 158 | 9 h / 0.65 h | C and D whole in 6 requests |

## Observations outside the question

Each could be its own small fix.

1. **Codex checks invisible to the outcome scorer.** Confirmed in code and data.
   - The deterministic outcome scorer reads only `Bash` calls and their `command` argument
     (`src/weave_agent_signals/scorers/outcome.py:645-652`, `:702-708`).
   - Session A's Codex `exec` calls carry JavaScript programs instead: 1,410 of 1,428 start with
     `const`, and 1,157 call `tools.exec_command`.
   - So none of A's 97 checks reaches the outcome scores. The small-model study's open question
     about Codex edits (`docs/small-model-judging-research.md:981-983`) applies to these
     programs, not to plain shell commands.
2. **Exit codes.** Confirmed in data.
   - Claude Code Bash calls in CES carry no exit code; status alone marks failure.
   - Codex `exec` has an exit code on 344 of 1,428 calls. 184 more print `"exit_code"` in their
     output.
3. **Missing Codex subagent thread.** Suspicion: A's one `spawn_agent` call has no child
   Trajectory in CES, so Codex subagent work may not be recorded.
4. **Codex context window.** Confirmed in source: Codex's `models.json` lists 272,000-token
   windows for GPT-5.6 Sol, Terra and Luna and for GPT-5.5 [11]. That answers the window part of
   the long-session study's open question 5 (`docs/long-session-judging-research.md:711-712`).
5. **Sessions keep growing.** Confirmed in data: D grew from 709 to 816 Cases between Part 1 and
   this run, so studies and runs should pin Case counts.

## Open questions

1. With one small model chosen, should it write verdicts too, or should setup require a second,
   larger verdict and escalation model?
2. For 1M models, should windows be one per compaction stretch, as recommended here, or packed
   up to the budget?
3. Should verification questions quote the claim sentence instead of the whole message, and ask
   about only the last few checks before a claim?
4. Should the extractor parse commands and patches out of Codex's JavaScript tool programs?
5. At 32k, where should 100 subagent summaries on one launch call go?
6. Is about 9 hours per reviewer acceptable for the Workflow session (18 hours with Codex, three
   times the requests with a full panel)?
7. Should the 18,000-token reserve shrink for 32k models, where it is 55% of the context?
8. Can the run UI's measured request times
   (`specs/02-evaluation.md:269-270`) replace the assumed latency model, per model?

## Sources

All accessed 2026-11-03.

Vendor documentation and source code:

1. Anthropic, Context editing. https://platform.claude.com/docs/en/build-with-claude/context-editing
2. Anthropic, Compaction overview. https://platform.claude.com/docs/en/build-with-claude/compaction
3. Anthropic, Compaction at a token threshold. https://platform.claude.com/docs/en/build-with-claude/compaction-threshold
4. Claude Code, How Claude Code works. https://code.claude.com/docs/en/how-claude-code-works
5. Claude Code, Environment variables. https://code.claude.com/docs/en/env-vars
6. Claude Code, Manage costs. https://code.claude.com/docs/en/costs
7. OpenAI, Codex configuration reference. https://developers.openai.com/codex/config-reference
8. OpenAI Codex source, compaction prompt. https://github.com/openai/codex/blob/main/codex-rs/prompts/templates/compact/prompt.md
9. OpenAI Codex source, `compact.rs` (`COMPACT_USER_MESSAGE_MAX_TOKENS`). https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs
10. OpenAI Codex source, output truncation. https://github.com/openai/codex/blob/main/codex-rs/utils/output-truncation/src/lib.rs
11. OpenAI Codex source, `models.json` (context windows, truncation policy). https://github.com/openai/codex/blob/main/codex-rs/models-manager/models.json

Papers:

12. Lindenbauer et al., The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management. https://arxiv.org/abs/2508.21433 (details from https://arxiv.org/html/2508.21433v3)
13. Kang et al., ACON: Optimizing Context Compression for Long-horizon LLM Agents. https://arxiv.org/abs/2510.00615
14. Chen et al., Measure Before You Manage: Evaluating Agent Working Memory in Coding Agents. https://arxiv.org/abs/2608.31057
15. trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories. https://arxiv.org/abs/2609.00038
16. Benchmarking LLM Judges for Mobile Agent Evaluation. https://arxiv.org/abs/2608.11434
17. Lightweight, Rubric-Guided Trajectory Evaluation for Production AI Agents (LiteTrajEval). https://arxiv.org/abs/2610.03315
18. Gaming the Judge: Unfaithful Chain of Thought Can Undermine Agent Evaluation. https://arxiv.org/abs/2601.14691
19. Time to Reflect: Can We Trust LLM Judges for Evidence-based Research Agents? https://arxiv.org/abs/2605.19196
20. What the Trace Shows and What the Label Says: Auditing Behavioral Detectors in Agent Evaluation. https://arxiv.org/abs/2610.30491
