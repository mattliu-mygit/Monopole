# Judging long sessions as a whole

Research notes on how Monopole could judge very long Claude Code and Codex
sessions in one piece, instead of skipping them or judging them through short
digests. Observed 2026-11-03. They are not Monopole product guarantees.

Every external claim cites a numbered source from the list at the end, and every
source was accessed on 2026-11-03. Repository claims cite `path:line`. Text
marked **Inference** is reasoning from the cited facts, not something a source
states. Vendor prices and model lists change often; re-check them before acting.

## Summary

1. Most "doesn't fit" skips come from Monopole's own settings rather than from
   today's models. The default judge panel is three 131,072-token models, so
   every default judge skips any turn above about 113,000 tokens and any session
   above about 500,000 tokens. Several catalog entries also understate the
   current windows; Claude Sonnet 5 is 1M tokens, not 200,000.
2. Recommended order of work:
   - Move the default panel to 1M-context judges from different families, and
     check the Claude token estimate against real token counts.
   - Shrink the judge view with deterministic rules before any model call.
   - Judge a session in one request per rubric when it fits a measured budget.
     Put the transcript first and the rubric last, so the six rubric requests
     share one cached prompt beginning (prompt caching).
   - Keep digest/window/merge for sessions above that budget, and split
     Workflow sessions by subagent.
3. Run agent-as-a-judge (Claude Code or Codex reading the transcript from files
   with read-only tools) as an experiment, not as the default.
4. Main tradeoff: one long request is the simplest and, with caching, the
   cheapest way to judge a session as a whole. But every long-context study
   found here shows quality falling well before the advertised limit. The
   size budget for single-request judging therefore has to come from the
   experiment plan below, not from the context-window number.

## Where the limits come from today

What the judge sees. `render_raw_turn`
(`src/weave_agent_signals/judges/windowing.py:116-176`) writes each turn as
text. The text holds the turn's first user message, its last assistant message,
and one block of token counters per model call. It also holds every tool call,
with its arguments and result written as JSON strings. Tool text over 12,000
characters is cut down to its head, its tail, its diagnostic lines, its original
length, and a hash (`windowing.py:17-20`, `windowing.py:80-101`). A subagent
appears only as its ID and its count of internal tool calls. The parent's
delegation call still carries what the subagent was asked and what it returned
(`windowing.py:166-174`, `specs/02-evaluation.md:104-109`).

How a session is split. `JudgingContextPolicy`
(`src/weave_agent_signals/run_config.py:47-86`) sets these limits:

- It reserves at least 18,000 tokens for the prompt, the output, and a safety
  margin.
- It aims each raw window at 128,000 tokens for models above 200,000 tokens of
  capacity, and at 50,000 for smaller models.
- It summarizes every other chunk in a digest of about 1,000 tokens, and lets
  each window return up to 10,000 tokens of findings.
- It allows at most 40 chunks per session.

A turn is never split. A reviewer is skipped with
`insufficient_context_capacity` in three cases:

- one turn is larger than the raw budget (`windowing.py:356-357`);
- the session needs more than 40 chunks (`windowing.py:367-368`);
- the final merge request would not fit. That request needs the reserve plus
  11,000 tokens per chunk (`windowing.py:373-377`).

The run UI repeats the same arithmetic to grey out models
(`frontend/src/features/runs/modelCapacity.ts:35-94`).

Worked example with the code's numbers, for the 225,000-token turn with 285
tool calls described in the request:

| Judge (catalog capacity) | Raw budget per window | Turn of 225k tokens | Longest session the merge allows |
| --- | --- | --- | --- |
| Default panel: `wandb:gpt-oss-120b`, `wandb:Llama-3.1-8B`, `wandb:granite-4.1-8b` (131,072 each) | 131,072 − 18,000 = 113,072 | skipped by all three | 10 chunks of ~50k, about 500k tokens |
| `claude:claude-sonnet-5` (200,000) | 182,000 | skipped | 16 chunks of ~50k, about 800k tokens |
| `codex:gpt-5.6-sol` (272,000) | 248,000 | fits as its own window | 22 chunks of ~128k, about 2.8M tokens |
| 1,048,576-token entries (agy Gemini, W&B DeepSeek V4) | about 1,024,000 | fits | 40-chunk cap, about 5.1M tokens |

The default panel comes from the first three judge-capable entries of
`_JUDGE_PREFERENCES` (`src/weave_agent_signals/catalogs.py:356-368`,
`catalogs.py:434-443`, `catalogs.py:522-528`); users can override it. The table
shows three separate causes of skips:

- the small default models;
- catalog capacities that are out of date (next section);
- the policy itself. Even a 1M-token model sees only about 128,000 raw tokens
  at a time, and sees the rest of the session as 1,000-token digests.

Workflow sessions hit the "turns are indivisible" rule hardest. **Inference:**
when one turn launches hundreds of subagents (one session had about 550), every
delegation call lands in that single turn. Each call can keep up to 24,000
characters of input and result, so the turn can outgrow every model's raw
budget.

Work per session. For each reviewer, Monopole makes N digest requests, 6 × N
window requests (one per rubric per window), and 6 merge requests
(`src/weave_agent_signals/judges/sliding.py:813-832`, `sliding.py:998-1026`).
The six rubrics are in `src/weave_agent_signals/judges/rubrics.py:406-416`.
**Inference:** a 1M-token session at the 128,000-token target has N of about 8.
That is 62 requests and about 8M input tokens per reviewer.

Prompt order. In a window request, the system message holds the rubric. The
user message starts with an example evidence ID, the allowed evidence IDs, and
the rubric criteria, and only then lists the session (`sliding.py:858-873`).
The requests for the six rubrics over one window therefore start differently,
so they cannot share a cached prompt beginning. This order is also the reverse
of what Anthropic and Google recommend for long inputs [5][18].

## What long-context models offer now

| Model | How Monopole can reach it | Context window | Max output | Price per 1M tokens (input / output) | Long-input pricing | Cache read price | Sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Sonnet 5.5 | Claude CLI, not in catalog | 1M | 128k | $2 / $10 | standard across 1M | 0.05× input | [1][2][3] |
| Claude Opus 5.5 | Claude CLI, not in catalog | 1M | 128k | $4 / $20 | standard across 1M | 0.05× input | [1][3] |
| Claude Haiku 5.5 | Claude CLI, not in catalog | 1M | 128k | $0.10 / $0.50 | $0.50 / $2.50 for prompts over 100k | 0.1× input | [1][3] |
| Claude Sonnet 5 | Claude CLI, catalog says 200,000 | 1M | 128k | $2 / $10 | standard across 1M | $0.20 | [2][3][7] |
| Claude Haiku 4.5 | Claude CLI, catalog 200,000 | 200k | not checked | $1 / $5 | none | $0.10 | [2][3] |
| GPT-6.1 Sol | Codex or OpenAI API, not in catalog | 1.05M (922k max input; 500k by default in Codex) | 128k | $2 / $10 | 2× input and cache, 1.5× output when input is over 272k | 0.05× input | [9][10] |
| GPT-6 Astra / GPT-6.1 Terra / GPT-6 Luna | Codex or OpenAI API, not in catalog | 1.05M | 128k | $10 / $50, $1 / $5, $0.10 / $0.50 | not checked | not checked | [9] |
| GPT-5.6 Sol | Codex, catalog says 272,000 | 1.05M in the API (922k max input) | 128k | $4 / $20 (promotional) | 2× input, 1.5× output when input is over 272k | $0.40 | [11] |
| Gemini 4 Argon | Gemini API; agy not checked | 1,048,576 shared by input, thinking, and output | 65,536 by default | $2 / $10 through 2026-11-30, then $4 / $20 | flat | $0.10, then $0.20 (implicit caching only) | [15][16][17][18] |
| Gemini 3.1 Pro Preview | agy, catalog 1,048,576 | 1M input plus 64k output | 64k | $2 / $12 | $4 / $18 for prompts over 200k | $0.20, $0.40 over 200k | [17][18] |
| W&B: GLM 5.3, GLM 5.3 Flash, GLM 5.2, DeepSeek V4-Pro-0813, DeepSeek V4.1-Flash, Nemotron 3.5 Super VL, Reflection Beam 1.0 | W&B Inference | 1,049k | not documented | for example DeepSeek V4.1-Flash $0.20 / $0.65, GLM 5.3 Flash $0.15 / $0.50, DeepSeek V4-Pro-0813 $1.31 / $3.96, GLM 5.2 $0.76 / $2.42 | none listed | $0.03, $0.05, $0.044, $0.14 | [20][21] |
| W&B: GPT OSS 120B (current default judge) | W&B Inference | 131k | not documented | $0.03 / $0.17 | none listed | no cache price listed | [20][21] |

Notes:

- Anthropic: 1M is the default for every 1M-window model, with no beta header,
  and long requests cost the standard rate, except on Haiku 5.5 [2][3].
- Claude Code runs Sonnet 5 and later at 1M by default [7]. Monopole's Claude
  entry is therefore planned at a fifth of what the CLI accepts
  (`catalogs.py:193-201`).
- Catalog entries that differ from the vendor lists:
  - GLM 5.2 is listed at 262,000 tokens (`catalogs.py:177`); the CoreWeave
    list of generally available models says 1,049k [20]. The W&B Inference web
    page still shows 262k for GLM-5.2 in a June 2026 listing [23], so the two
    sources disagree.
  - 13 of the 24 W&B catalog entries are not on that list, and its deprecated
    list is empty [20]. They include `granite-4.1-8b` from the default panel.
  - GPT-5.4, GPT-5.4 mini, and GPT-5.5 left Codex for ChatGPT sign-in on
    2026-08-31 and 2026-10-14; API-key sessions are not affected [13].
  - The Codex window for the GPT-5.6 entries is not documented in the pages read
    here. Codex lets a user set `model_context_window` [14].
- Token counting. Monopole estimates non-OpenAI models as UTF-8 bytes ÷ 3
  (`src/weave_agent_signals/judges/tokens.py:17-22`). Anthropic says Claude 4.7
  and later use a tokenizer that produces about 30% more tokens for the same
  text [3], and that 1M tokens is roughly 2.5M characters [1]. **Inference:** for
  plain ASCII text that is 2.5 characters per token, so bytes ÷ 3 would
  undercount Claude by about 17%. Measure this with the token counting API
  before planning requests near 1M.
- Quality caveats in vendor docs:
  - Anthropic: "As token count grows, accuracy and recall degrade, a phenomenon
    known as context rot" [2].
  - Google: finding several separate facts in a long input is less accurate
    than finding one, and the query belongs at the end [18].
  - The OpenAI model pages read here give no long-context quality caveat
    [9][10][11].

## Effective versus advertised context

Effective context is the length up to which a model still does a task about as
well as it does on short input. The studies below find it far shorter than the
advertised window for tasks like judging:

- In Lost in the Middle, accuracy was highest when the relevant text sat at the
  beginning or end of the input, and dropped when it sat in the middle [24].
- Of 17 models in RULER that claim 32k tokens or more, only half kept
  satisfactory performance at 32k [25].
- In NoLiMa, the question and the relevant text share few words. At 32k tokens,
  11 of 13 models fell below half of their short-input score; GPT-4o fell from
  99.3% to 69.7% [26].
- In HELMET, simple needle-in-a-haystack scores did not predict performance on
  real tasks [27].
- In LongBench v2, the best model answering directly scored 50.1%, and a
  reasoning model scored 57.7% [28].
- Chroma's report tested 18 models. On LongMemEval, every model did much better
  with only the relevant ~300 tokens than with the full ~113k-token chat
  history [29].
- Even when the model is given all relevant information and the filler is
  whitespace or masked out, performance fell 13.9% to 85% as the input grew.
  Asking the model to recite the evidence first helped [30].
- Oolong needs a model to classify many small pieces and count across them,
  which is close to judging a whole session. GPT-5, Claude Sonnet 4, and Gemini
  2.5 Pro all scored under 50% at 128k [31].
- LLM faithfulness raters were sensitive to document order, with a U-shaped
  pattern that neglects the middle [32]. On book-length input, no LLM rater
  agreed strongly with humans [33].
- On TRAIL, a benchmark of finding and classifying errors in long agent traces,
  the best model scored 11% [34].

What this means for Monopole's judges (**Inference**):

- A rubric verdict over a session needs many pieces of evidence: every
  completion claim, every failed check, every user correction. The evidence
  rarely uses the rubric's words. These are the conditions under which the
  studies above show the largest drops, so a session that fits in 1M tokens may
  still be judged worse in one request than in pieces.
- Middle turns get the least attention. A failure that the agent made and
  recovered from in the middle of a long session is the case most likely to be
  missed.
- Removing tokens the rubric does not need helps even when the model could find
  the evidence [29][30]. Compression is worth doing before any other strategy.
- Put the transcript first and the rubric last [5][18], and ask the judge to
  quote or cite evidence before it scores [5][30].

## Strategies

### 1. Compress the transcript

How it works. Shrink the judge view with deterministic rules before planning,
keeping every evidence ID:

- Drop the per-call token counters, or roll them up into one line per turn.
- Write tool arguments and results as plain text blocks instead of JSON-escaped
  strings.
- Replace a repeated read of unchanged content with a pointer such as "same
  content as evidence `e17`, sha256 …". Monopole's efficiency scorer already
  detects repeated reads without an edit in between
  (`specs/02-evaluation.md:25-26`).
- Keep failing output at today's 12,000-character limit, and cut successful
  output that has no diagnostic lines to a shorter head and tail.
- Optionally add a model-written summary of each subagent's work. This is
  lossy, so keep it out of the citation path.

Evidence:

- In SWE-agent, hiding old tool output halved cost and matched or slightly beat
  LLM summarization [49].
- ACON cut peak tokens by 26% to 54% while keeping or improving task success
  [50].
- These two results measure agent context, not judges. The judge-side evidence
  is indirect: shorter relevant input scores better [29][30].
- Anthropic offers tool-result clearing as a context-editing feature [2].

Measured here, on a synthetic turn only (not a real session). I built a turn
with 285 tool calls whose results are this repository's own Python files,
plus short shell output, and rendered it with `render_raw_turn`:

| Measurement | Result |
| --- | --- |
| Rendered size | 1,298,046 characters; 344,273 tokens with `o200k_base`; 432,709 by bytes ÷ 3 |
| Token counters for each model call | 3.9% to 4.5% of tokens |
| Tool results as JSON-escaped strings vs plain text | 253,865 vs 216,443 tokens (escaping adds 17%) |

I built repeated reads into the synthetic turn on purpose, so its duplicate
share says nothing about real sessions. Measure that on CES data.

Fit with Monopole:

- All rendering goes through `render_raw_turn` and `compact_tool_text`
  (`windowing.py:80-176`), and evidence IDs survive.
- Changing the rendering changes plan and request identities, which is fine
  because the run database is disposable (`AGENTS.md`).
- The judge view keeps only the first user message and the last assistant
  message of each turn (`src/weave_agent_signals/ces_client.py:229-234`,
  `ces_client.py:253`, `ces_client.py:299`). Adding the intermediate assistant
  messages costs tokens but gives the judge the claims that the verification
  rubric checks. See "Observations outside the question".

Cost. No model calls for the rules above; subagent summaries cost one request
each.

Risks:

- A rule can drop evidence a rubric needs. Tool choice judges redundancy, so a
  deduplicated read must stay visible as "re-read of unchanged file".
- Summaries can be unfaithful [33].

### 2. Judge the whole session in one request

How it works. When the compressed session plus prompt fits under a measured
budget, send the whole session once per rubric. Put the session first and the
rubric and output rules last, and ask the judge to cite or quote evidence IDs
before it scores [5][30].

Evidence:

- Every major vendor now offers 1M-token windows (table above).
- When enough budget is available, full long input beat retrieval on average,
  but at much higher cost [48].
- The degradation evidence above says the usable budget is below the window.
- For finding errors in long traces, a monolithic judge did far worse than the
  same model inside a decomposed method. On GAIA traces, GPT-5.4 found the
  error location 29.2% of the time alone and 82.3% decomposed. Monolithic
  baselines also had to exclude 6% to 9% of GAIA traces and 22% to 29% of
  SWE-bench traces for length [35].

Fit with Monopole:

- `build_window_plan` already produces one window when a session is under the
  raw target. Raising the target for 1M models is a policy change, and the spec
  says the current numbers are cost bounds, not measured optimums
  (`specs/02-evaluation.md:364-368`).
- Today a one-window session still makes 13 requests per reviewer: one digest,
  six windows, six merges. A whole-session mode could make six, where the
  window request returns the verdict.
- Both changes need new pinned protocol and plan versions (`sliding.py:68`,
  `windowing.py:16`).
- The frontend estimate (`modelCapacity.ts`) copies the plan arithmetic and
  must change with it.

Cost per reviewer for a 600,000-token session with six rubrics. Assumptions:
a 6,000-token prompt, 3,000 output tokens per request including thinking, one
cache write followed by five cache reads, and list prices from [3][10][17][21].
**Inference**, from list prices:

| Setup | Requests | Estimated cost |
| --- | --- | --- |
| Current pipeline on Claude Sonnet 5.5, 128k windows, no caching | 41 | about $11 |
| Current pipeline, window prompts reordered so the six rubrics share a cached beginning | 41 | about $5 |
| Whole session per rubric, Claude Sonnet 5.5, no caching | 6 | about $7.40 |
| Whole session per rubric, Claude Sonnet 5.5, cached | 6 | about $2.00 |
| Whole session per rubric, GPT-6.1 Sol API (over-272k surcharge), cached | 6 | about $3.90 |
| Whole session per rubric, Gemini 4 Argon until 2026-11-30, implicit cache hits | 6 | about $1.70 |
| Whole session per rubric, W&B DeepSeek V4.1-Flash, cache hits | 6 | about $0.22 |

A three-judge panel triples these numbers. Claude and Codex CLI judges bill
against a subscription or an API key, so for them the cost shows up as usage
limits rather than dollars.

Risks:

- Quality between 300k and 900k tokens is unproven and probably lower.
- Evidence in the middle of the session is under-weighted [24][32].
- One provider failure loses the whole reviewer for that session.
- Token estimates near the limit must be right; see the Claude token note.

### 3. Hierarchical (map-reduce) judging

How it works. The map step judges parts of the session independently and
returns findings with evidence IDs. The reduce step is one final request that
reads all findings and returns the verdict. Monopole's pipeline is already a
version of this: rubric-neutral digests, rubric findings per window with
digests of the other chunks as context, and a merge. The structural variant
maps over subagent trajectories in Workflow sessions: judge each subagent
against its delegated task and the rubrics, then judge the parent using those
findings in place of the raw delegation calls.

Evidence:

- Judging each span separately plus a top-down pass beat monolithic judges on
  TRAIL. The authors list a limitation of their own roll-up rule: any span
  error fails the whole trace, so minor issues count like critical ones [35].
- LLM×MapReduce names the two failure types of splitting: facts that depend
  on another chunk, and chunks that conflict. It handles them with a structured
  output format and a confidence score in each chunk's answer [43].
- Book summaries built by merging chunk summaries were more coherent than ones
  built by updating a running summary, but held less detail [41].
- MemWalker builds a tree of summaries and lets the model walk down it to the
  passages a question needs. It beat long-context, recurrence, and retrieval
  baselines on long-text question answering [46].
- Decomposition can also hurt whole-trace judgments. In Who&When, with GPT-4o,
  the whole-log method named the failing agent about 54% of the time, against
  about 35% for step-by-step reading. The paper's best method named the
  responsible agent 53.5% of the time but the decisive step only 14.2% of the
  time [36].
- No source found here compares map-reduce rubric scores with whole-context
  rubric scores on the same long agent sessions. That agreement has to be
  measured.

Fit with Monopole:

- The digest, window, and merge code exists, and its plans and artifacts are
  pinned.
- Workflow sessions need a map over subagent trajectories. Turns above the raw
  budget need splitting between tool calls instead of being skipped.
- The merge sees only ~1,000-token digests and the findings, so a link between
  a claim in chunk 3 and a failing test in chunk 7 survives only if a window
  reported it.

Cost. Today N + 6N + 6 requests per reviewer. The subagent map adds one
request per subagent, up to about 550. **Inference:** a cheaper model can do the
map step, since a panel of smaller models outperformed a single large judge
[52]; verify this per rubric.

Risks:

- Links between chunks get lost.
- The roll-up rule from findings to an anchor changes scores.
- More requests mean more failure points, and Monopole fails the session when
  any reviewer fails (`specs/02-evaluation.md:254-261`).

### 4. Incremental judging with running notes

How it works. Read chunks in order and carry a bounded ledger forward: claims
made, checks run and their results, open errors, decisions, and constraints.
Each step updates the ledger, and a final request scores each rubric from the
ledger and the last chunk.

Evidence:

- Chain of Agents passes notes along a chain of workers that each read one
  segment. It beat retrieval, full-context, and multi-agent baselines by up to
  10% [42].
- MemAgent, trained with an 8k context on 32k-token texts, kept accuracy loss
  under 5% on 3.5M-token questions. That is a trained model, not prompting
  [44].
- ReadAgent extended effective context 3.5 to 20 times with short "gist"
  notes plus look-ups [45].
- Updating a running summary kept more detail than merging chunk summaries but
  was less coherent [41].
- With GPT-5, Recursive Language Models (RLM) beat repeated summarization
  ("compaction") by a median 26% [40].
- None of these sources tests judging.

Fit with Monopole. Today's digests are independent per chunk, and each window
sees digests of both earlier and later chunks. A ledger shows only the past.
That loses the look-ahead that state consistency and outcome may need.
Resuming works only if each ledger step is saved, since changing an early step
invalidates every later one.

Cost. N sequential requests plus the verdicts. No parallelism, so wall time is
the sum of the requests.

Risks:

- An early miss is never corrected.
- The verdict depends on reading order.
- Sequential latency.

### 5. Extract rubric evidence before the verdict

How it works. Select what each rubric needs, by code where possible, and judge
that smaller view:

| Rubric | What to extract |
| --- | --- |
| Verification | completion claims, and the checks after the last edit |
| Error recovery | failed calls and the next few actions |
| Autonomy | user messages, steering events, and denial events |
| State consistency | stated decisions and constraints, and later actions that touch them |
| Tool choice | the call sequence, with results mostly elided |
| Outcome | the request, the final changes, and the last checks |

If the judge returns `insufficient_evidence`, retry with the full view.

Evidence:

- Routing questions between retrieval and full context cut cost 65% for
  Gemini 1.5 Pro and 39% for GPT-4o, at accuracy comparable to full context
  [48].
- Focused input beat full input [29].
- Writing a summary that cites its sources is hard. Long-context models with no
  retriever scored under 20% on SummHay [47].
- In Agent-as-a-Judge, a module that pulls relevant segments out of long
  trajectories added nothing on DevAI: alignment went from 90.44% to 90.16%
  [37].

Fit with Monopole:

- Deterministic outcome and efficiency parsers already exist
  (`specs/02-evaluation.md:15-36`).
- Each rubric already lists its "Allowed evidence" (`rubrics.py:102-404`).
- Evidence IDs stay canonical.

Cost. Lowest of all: extraction is code, judge requests are small, and only
abstentions trigger a second request.

Risks:

- The parsers favor precision over coverage by design
  (`specs/02-evaluation.md:34-36`), so extraction will miss evidence.
- An abstention caused by extraction looks like a real abstention unless the
  pipeline records which view was judged.
- State consistency needs wide context.

### 6. Agent-as-a-judge

How it works. Write the session into a read-only workspace: an index of turns
with their sizes and evidence IDs, one file per turn, and one file per
subagent. Run Claude Code with `--tools "Read,Grep,Glob"`, `--json-schema`,
`--max-turns`, and `--max-budget-usd` [6], or Codex with Monopole's existing
read-only permission profile. The judge returns the same verdict JSON with
evidence IDs, and the host validates them. A variant keeps the transcript in a
Python variable that the model inspects and splits with code, calling itself on
the pieces (RLM) [40].

Evidence:

- On DevAI (55 tasks, 365 requirements), Agent-as-a-Judge agreed with the human
  judgments on 84% to 92% of requirements, against 60% to 84% for
  LLM-as-a-Judge across three agents and two settings. The paper notes that the
  highest LLM-as-a-Judge number is inflated by imbalanced labels, and reports
  about 97% lower time and cost than three human experts [37].
- DevAI checks requirements against the files each agent produced, and the
  module that pulls segments out of trajectories did not improve agreement
  there [37]. So DevAI says little about reading very long transcripts.
- AJ-Bench found consistent gains over LLM-as-a-Judge, with large open problems
  [38]. A 2026 survey maps the follow-up work [39].
- RLM handled inputs more than ten times larger than the model's context
  window. With GPT-5 it beat compaction by a median 26% and Claude Code by
  13%. Its per-query cost was long-tailed and highly variable. Its Claude Code
  baseline used Claude Opus 4.1 with the context in a file or in the prompt
  [40].
- The TRAIL follow-up concludes that "evaluation methodology, not model
  capability, is the bottleneck" [35].

Fit with Monopole:

- Codex's permission profile already lets the judge read its workspace root
  (`src/weave_agent_signals/judges/process.py:28-42`). That workspace
  (`cli_backend.py:83`) is empty today (`AGENTS.md`), so the transcript files
  would go there.
- Claude runs with `--tools ""` (`cli_backend.py:978-1001`), and `AGENTS.md`
  documents "Claude is tool-disabled". Giving it read tools changes a
  documented safety rule.
- Monopole pins "complete raw-turn coverage" and resumes per request
  (`sliding.py:413-443`). An agent's reading path is neither pinned nor
  guaranteed to cover every turn. The run can pin the input files and record
  the agent's tool-call log as an audit record, but it cannot be reproduced
  exactly.

Cost. Unpredictable. **Inference:** about 30 tool turns over a 600k-token
session on Sonnet 5.5 is roughly $1 to $2 per rubric, mostly cache reads. One
agent run could score all six rubrics to save cost, at the risk of rubrics
influencing each other.

Risks:

- The transcript is untrusted text that now reaches a model with tools.
  Read-only access on a copy, with no network, limits the damage.
- The agent may stop reading early and abstain or overstate its confidence.
- Scores vary from run to run, with a long cost tail [40].

### 7. Prompt caching

How it works. The provider keeps the processed beginning of a prompt. A later
request that starts with exactly the same tokens pays a reduced rate for that
part.

| Provider | Cache write | Cache read | Lifetime | Minimum prompt | Notes | Sources |
| --- | --- | --- | --- | --- | --- | --- |
| Anthropic | 1.25× for 5 minutes, 2× for 1 hour | 0.1×; 0.05× on Opus and Sonnet 5.5 | 5 minutes or 1 hour | 512 to 4,096 tokens by model; 1,024 for Sonnet 5 | up to 4 cache breakpoints; checks back 20 blocks; a parallel request can reuse the cache only after the first response starts | [3][4] |
| OpenAI (GPT-5.6 and later) | 1.25× for 30 minutes, 2.5× for 12 hours | 0.1×; 0.05× on GPT-6.1 Sol and Terra | 30 minutes or 12 hours | 1,024 tokens | changing the response format or reasoning effort can break reuse | [12] |
| Gemini | no write charge listed | per-model price, for example $0.10 on Gemini 4 Argon | not documented | 4,096 tokens on current models | Gemini 4 Argon has implicit caching only | [17][19] |
| W&B / CoreWeave | none listed | per-model "cache hit" price | not documented | not documented | automatic on supported models; `cache_salt` limits reuse to one tenant; cached counts appear in usage on some models | [21][22] |
| Claude Code | — | — | — | — | caches by default (`DISABLE_PROMPT_CACHING` turns it off); holds back workflow fan-outs so siblings reuse one cached beginning | [8] |

Fit with Monopole:

- Today the six rubric requests over a window begin with rubric text
  (`sliding.py:858-873`), so they share no cached beginning. The fix: a
  rubric-neutral system message, then the session sections, then the rubric
  criteria and allowed IDs last.
- The response schema of a window depends only on that window's evidence IDs
  (`sliding.py:895`), so it is the same for all six rubrics. That matters for
  OpenAI, where a different response format can break reuse [12].
- Rubric panels run concurrently (`specs/02-evaluation.md:238-246`). To get
  hits, send one rubric's request first and the rest after its response
  starts [4].
- The Claude CLI receives the rubric through `--system-prompt`, and Codex
  receives it at the start of its prompt (`cli_backend.py:999-1003`). Both
  break reuse across rubrics in the same way.

Cost. **Inference:** in the 600k example, this cuts the price of repeated
input in the window step by about 75%, and makes whole-session judging about a
quarter of its uncached cost.

Risks:

- A miss is silent: no error, just full price. Log the cache-read counts from
  provider usage on every request.
- W&B reuse is best effort, "on the same backend" [22].

## Comparison

| Strategy | Longest session it can judge | Requests per reviewer, six rubrics | Relative cost | Main quality risk | Fits Monopole's pinned plans | Effort |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Compression | raises every other ceiling by the share it saves | none added | lowers all others | drops evidence a rubric needs | yes, rules are deterministic | small |
| 2. Whole session | model input limit minus reserve; the quality limit is unknown | 6 | low with caching | context rot; middle evidence missed | yes | small to medium |
| 3. Hierarchical | unbounded | N + 6N + 6, plus one per subagent | high without caching | links between chunks lost; roll-up rule | yes, exists today | medium |
| 4. Incremental | unbounded | N sequential, plus 6 | medium, and slow | early misses never corrected; order effects | yes, but resumes step by step | medium |
| 5. Evidence extraction | unbounded if the extract is small | 6, plus retries on abstain | lowest | extraction misses; false abstentions | yes | medium |
| 6. Agent-as-a-judge | unbounded | 6 agent runs (or 1 shared), each many turns | variable, long tail | reads too little; prompt injection; variance | partly; reading path not pinned | medium to large |
| 7. Prompt caching | no change | same | cuts repeated input price by about 80% to 95% on hits, by provider | silent misses | yes, after reordering prompts | small |

## Experiment plan

Goal: pick a default per session size, using Monopole's own sessions, and
measure the size budget for single-request judging on our data.

1. Size census, with no model calls.
   - Read about 200 recent complete sessions.
   - For each session record: rendered tokens under each counter, the largest
     turn, the number of turns, tool calls, and subagents.
   - Record which catalog models skip it and why: turn too big, merge, or the
     chunk cap. `build_window_plan` can run in a script.
   - Apply each compression rule and record the tokens it saves.
   - Output: a size histogram showing what share of sessions each strategy
     could cover.
2. Token calibration.
   - Count 20 rendered windows with Anthropic's token counting API, W&B
     `usage.prompt_tokens`, and `o200k_base`. Compare with bytes ÷ 3 and set
     counters or safety factors from the result.
3. Probe set.
   - Pick 40 sessions, 10 per size bucket: under 128k, 128k to 400k, 400k to
     1M, and over 1M. Include the 225k-token-turn session and the
     ~550-subagent Workflow session.
4. Planted evidence, on offline copies only, never written to CES.
   - In 16 sessions with a completion claim, change the last relevant test
     result before the claim from pass to fail, so the claim is now
     contradicted. Verification should drop to 0, the anchor for a claim made
     despite a failed check (`rubrics.py:129-132`).
   - In 16 sessions with an error, insert a repeat of the same failing command
     right after it. Error recovery should drop, since repeating a disproven
     action is its 0 anchor (`rubrics.py:180-183`).
   - Choose the edited claim or error so that it falls in the first, middle,
     or last third of the session. These probes give known answers without
     human labels, and they measure position bias [24][32].
5. Strategies to compare. Use one judge model for all of them, for example
   Claude Sonnet 5.5 or GLM 5.3, and a second model family on 15 sessions.

   | Name | What it runs |
   | --- | --- |
   | **current** | today's pipeline with today's default panel, to count skips |
   | **current-1M** | today's pipeline on a 1M model with the catalog fixed |
   | **wide-windows** | current-1M with the raw target raised to about 400k |
   | **whole** | one request per rubric, session first, cached, only when it fits |
   | **whole-compressed** | whole, run on the compressed view |
   | **by-subagent** | hierarchical by subagent, Workflow sessions only |
   | **agent** | Claude Code or Codex with read-only file tools, capped by turns and budget |

6. Record for each session, rubric, and strategy:
   - the anchor or abstention, and how many cited evidence IDs pass host
     validation;
   - input, cache-read, cache-write, and output tokens, and dollars at list
     price;
   - wall time, and whether the reviewer was skipped or failed.
7. Stability. Run each strategy three times on 10 sessions at Monopole's
   temperature of 0.2 (`sliding.py:67`). How much those repeats agree is the
   noise floor; a single sample can mislead [56].
8. Agreement:
   - the share of exactly equal anchors, and of anchors within 0.25;
   - Krippendorff's alpha, an agreement score corrected for chance where 1 is
     perfect and 0 is chance. The ordinal version treats 0.25 against 0.5 as
     closer than 0 against 1 [60];
   - the mean signed difference, to show whether a strategy scores higher
     overall;
   - abstention disagreements, counted separately.
9. Prompt order. On 10 sessions, run **whole** with the rubric first, as today,
   and with the rubric last, to measure the order effect that vendors describe
   [5][18].
10. Human labels.
    - Two people label every cell where strategies disagree by 0.5 or more, or
      where one abstains and another scores.
    - They also label a random 10% of the agreeing cells.
    - 20% of cells get both raters, to measure agreement between humans.
    - Monopole has no labeled set today (`specs/02-evaluation.md:364-366`), and
      judges should be checked against people per task [51][57]. Expect raters
      to refine the rubric wording as they read [59].
11. Decision rule.
    - The size budget is the largest size bucket where **whole** or
      **whole-compressed** stays within the noise floor of the best decomposed
      strategy, and detects planted middle-third evidence within 5 points of
      the best strategy.
    - Below that budget, use the cheaper of the two whole strategies. Above it,
      use the best decomposed strategy.
    - Adopt **agent** only if it beats the others on planted evidence and on the
      human-labeled disagreements by more than the noise floor, at an
      acceptable cost cap.
    - 40 sessions show large gaps, such as 90% against 60% detection, but not
      small ones.

Further validation methods:

- Position-consistency and repeat-stability metrics, and the finding that
  position bias depends on how different the candidates' quality is [53].
- Averaging over orderings and asking for evidence before the score [54].
- A catalog of judge biases [55].
- Panels of judges from different model families [52].
- For rubric checks in agent tasks, LLM judges are good but noisy, and majority
  voting helps less with each added vote [58].

## Observations outside the question

Each of these could be its own small fix, separate from this research.

1. Confirmed in code: the default recommended judges are three 131,072-token
   models (`catalogs.py:356-368`). One of them, `granite-4.1-8b`, is not on W&B's
   current list of available models [20], so the default may point at a model
   that no longer serves requests. Check `/v1/models` to confirm.
2. Confirmed in code and docs: the catalog understates current windows. Claude
   Sonnet 5 is 1M [2][7] and GLM 5.2 is 1,049k [20]. It also lists 13 W&B models
   missing from W&B's current list of available models [20], and three Codex
   models that Codex no longer offers with ChatGPT sign-in [13].
3. Confirmed in code; whether it was intended is unknown. The judge view keeps
   only the first user message and the last assistant message of each turn
   (`ces_client.py:229-234`, `ces_client.py:253`, `ces_client.py:299`), even
   though each assistant message is its own Case. The spec says the raw
   rendering includes captured user and assistant evidence
   (`specs/02-evaluation.md:104`). Claims made mid-turn never reach the judge.
4. **Inference**, needs measurement: bytes ÷ 3 probably undercounts tokens for
   Claude 4.7 and later [1][3].
5. Confirmed in code: window prompts put the rubric first, which blocks cache
   reuse across rubrics and goes against vendor long-input guidance
   (`sliding.py:858-873`; [5][18]).

## Open questions

1. Which rubrics need subagents' internal tool calls, and which need only the
   delegated input and returned output?
2. Was dropping intermediate assistant messages from the judge view
   intentional?
3. Should the Claude judge get read-only tools? That changes the documented
   "Claude is tool-disabled" rule in `AGENTS.md`.
4. How would an agent judge meet Monopole's "complete raw-turn coverage" rule?
   Options include requiring the agent to open every turn file, or accepting
   less coverage.
5. What context window does Codex actually use for the GPT-5.6 entries, and
   does `codex exec` report cached tokens?
6. Do separate `claude -p` processes reuse the cache when their prompts begin
   identically? **Inference:** yes, because the cache lives on the provider
   side. Check `cache_read_input_tokens` in `--output-format json`.
7. How do subscription usage limits behave for 1M-token CLI requests?
8. Should a Workflow session be scored as one unit, or should the orchestrator
   and each subagent get their own Scores?
9. Do W&B models with no listed cache price, such as GPT OSS 120B, cache at
   all?

## Sources

All accessed 2026-11-03.

Vendor documentation:

1. Anthropic, Models overview. https://platform.claude.com/docs/en/about-claude/models/overview
2. Anthropic, Context windows. https://platform.claude.com/docs/en/build-with-claude/context-windows
3. Anthropic, Pricing. https://platform.claude.com/docs/en/about-claude/pricing
4. Anthropic, Prompt caching. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
5. Anthropic, Prompting best practices, "Long context prompting". https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
6. Claude Code, CLI reference. https://code.claude.com/docs/en/cli-reference
7. Claude Code, Model configuration. https://code.claude.com/docs/en/model-config
8. Claude Code, Environment variables. https://code.claude.com/docs/en/env-vars
9. OpenAI, Models. https://platform.openai.com/docs/models
10. OpenAI, GPT-6.1 Sol. https://platform.openai.com/docs/models/gpt-6.1-sol
11. OpenAI, GPT-5.6 Sol. https://platform.openai.com/docs/models/gpt-5.6-sol
12. OpenAI, Prompt caching. https://platform.openai.com/docs/guides/prompt-caching
13. OpenAI, Codex models. https://developers.openai.com/codex/models
14. OpenAI, Codex configuration reference. https://developers.openai.com/codex/config-reference
15. Google, Gemini models. https://ai.google.dev/gemini-api/docs/models
16. Google, Gemini 4 Argon. https://ai.google.dev/gemini-api/docs/models/gemini-4-argon
17. Google, Gemini Developer API pricing. https://ai.google.dev/gemini-api/docs/pricing
18. Google, Long context. https://ai.google.dev/gemini-api/docs/long-context
19. Google, Context caching. https://ai.google.dev/gemini-api/docs/caching
20. CoreWeave, Serverless Inference available models (redirected from docs.wandb.ai). https://docs.wandb.ai/guides/inference/models/
21. CoreWeave, Forge Inference pricing. https://coreweave.com/forge-pricing/inference-agents
22. CoreWeave, Serverless Inference prefix caching. https://docs.coreweave.com/products/inference/serverless/response-settings/prefix-caching
23. W&B, Serverless Inference hosted models. https://wandb.ai/inference

Papers and reports:

24. Liu et al., Lost in the Middle: How Language Models Use Long Contexts. https://arxiv.org/abs/2307.03172
25. Hsieh et al., RULER: What's the Real Context Size of Your Long-Context Language Models? https://arxiv.org/abs/2404.06654
26. Modarressi et al., NoLiMa: Long-Context Evaluation Beyond Literal Matching. https://arxiv.org/abs/2502.05167
27. Yen et al., HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly. https://arxiv.org/abs/2410.02694
28. Bai et al., LongBench v2. https://arxiv.org/abs/2412.15204
29. Hong, Troynikov, Huber, Context Rot: How Increasing Input Tokens Impacts LLM Performance (Chroma technical report, July 2025). https://research.trychroma.com/context-rot
30. Du et al., Context Length Alone Hurts LLM Performance Despite Perfect Retrieval. https://arxiv.org/abs/2510.05381
31. Bertsch et al., Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities. https://arxiv.org/abs/2511.02817
32. Wan et al., On Positional Bias of Faithfulness for Long-form Summarization. https://arxiv.org/abs/2410.23609
33. Kim et al., FABLES: Evaluating faithfulness and content selection in book-length summarization. https://arxiv.org/abs/2404.01261
34. Deshpande et al., TRAIL: Trace Reasoning and Agentic Issue Localization. https://arxiv.org/abs/2505.08638
35. Madvil et al., Holistic Evaluation and Failure Diagnosis of AI Agents. https://arxiv.org/abs/2605.14865
36. Zhang et al., Which Agent Causes Task Failures and When? https://arxiv.org/abs/2505.00212 and the per-method table at https://ag2ai.github.io/Agents_Failure_Attribution/
37. Zhuge et al., Agent-as-a-Judge: Evaluate Agents with Agents. https://arxiv.org/abs/2410.10934
38. Shi et al., AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation. https://arxiv.org/abs/2604.18240
39. You et al., A Survey on Agent-as-a-Judge. https://arxiv.org/abs/2601.05111
40. Zhang, Kraska, Khattab, Recursive Language Models. https://arxiv.org/abs/2512.24601
41. Chang et al., BooookScore: A systematic exploration of book-length summarization in the era of LLMs. https://arxiv.org/abs/2310.00785
42. Zhang et al., Chain of Agents: Large Language Models Collaborating on Long-Context Tasks. https://arxiv.org/abs/2406.02818
43. Zhou et al., LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models. https://arxiv.org/abs/2410.09342
44. Yu et al., MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent. https://arxiv.org/abs/2507.02259
45. Lee et al., A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts (ReadAgent). https://arxiv.org/abs/2402.09727
46. Chen et al., Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading (MemWalker). https://arxiv.org/abs/2310.05029
47. Laban et al., Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems. https://arxiv.org/abs/2407.01370
48. Li et al., Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach. https://arxiv.org/abs/2407.16833
49. Lindenbauer et al., The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management. https://arxiv.org/abs/2508.21433
50. Kang et al., ACON: Optimizing Context Compression for Long-horizon LLM Agents. https://arxiv.org/abs/2510.00615
51. Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. https://arxiv.org/abs/2306.05685
52. Verga et al., Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL). https://arxiv.org/abs/2404.18796
53. Shi et al., Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge. https://arxiv.org/abs/2406.07791
54. Wang et al., Large Language Models are not Fair Evaluators. https://arxiv.org/abs/2305.17926
55. Ye et al., Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge. https://arxiv.org/abs/2410.02736
56. Schroeder and Wood-Doughty, Can You Trust LLM Judgments? Reliability of LLM-as-a-Judge. https://arxiv.org/abs/2412.12509
57. Bavaresco et al., LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks. https://arxiv.org/abs/2406.18403
58. Peng et al., Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios? (RuVerBench). https://arxiv.org/abs/2606.29920
59. Shankar et al., Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences. https://arxiv.org/abs/2404.12272
60. Krippendorff, Computing Krippendorff's Alpha-Reliability (2011). https://repository.upenn.edu/asc_papers/43/
