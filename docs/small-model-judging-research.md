# Judging large sessions with small models

Research notes on how Monopole could judge very large Claude Code and Codex
sessions with the smallest practical models: open models of about 8B to 70B
parameters with 32k to 262k context windows, as W&B Inference serves them.
Observed 2026-11-03. They are not Monopole product guarantees.

This is the fourth document in a series, and it does not repeat the others:

- `docs/long-session-judging-research.md` covers long-context models,
  compression, whole-session requests, prompt caching, and agent-as-a-judge.
- `docs/windowed-judging-research.md` covers window-then-merge designs, claim
  ledgers, structural windows, and validation.
- `docs/large-session-anatomy.md` measures the four largest real sessions.
- `docs/superpowers/specs/2026-11-03-long-session-judging-design.md` is the
  draft design.

Every external claim cites a numbered source from the list at the end, and every
source was accessed on 2026-11-03. Repository claims, including claims taken
from the documents above, cite `path:line`. Text marked **Inference** is
reasoning from the cited facts, not something a source states. Prices and model
lists change often; re-check them before acting.

Terms used throughout:

- **Small model**: a dense model of about 8B parameters, or a mixture-of-experts
  (MoE) model that runs only 3B to 5B of its parameters for each token, even if
  it stores up to about 35B. Examples: Llama 3.1 8B, Granite 4.2 8B, GPT OSS 20B.
- **Mid-size model**: a 27B to 70B dense model, or a larger MoE. Examples: Gemma
  4 31B, Llama 3.3 70B, GPT OSS 120B.
- **Large model**: the 1M-token frontier and MoE models from the long-session
  study.
- **Effective context**: the input length up to which a model still does a task
  about as well as it does on short input.
- **Question**: one yes/no check about named evidence, answered in its own
  request. Example: "Is test run `t-2210` relevant to the claim in message
  `m-430`?"
- **Evidence bundle**: the raw steps one question needs, fetched by evidence ID.
- **Step label**: a yes/no tag that a judge puts on one step, such as "wasted"
  or "risky".
- **Escalation**: sending a question, a thread verdict, or a session to a larger
  model or to the user.
- Thread, step, window, finding, merge, and ledger mean what the windowed study
  defines (`docs/windowed-judging-research.md:20-39`).

## Summary

1. Pipeline: code extracts checks, edits, errors, steering, and delegations;
   small models label each step and answer yes/no questions on evidence of 4k
   tokens or less; a 27B–70B model writes each thread's verdict; a large model
   sees only escalated items.
2. For: small models match large ones on narrow, short jobs: checking a claim
   against a document [5], answering per-item yes/no questions [20], and labeling
   steps with a frontier model behind them on flagged steps [50].
3. Against: every judge LongRM tested, up to 70B, fell below 50% (guessing) past
   4k tokens of context [17]; Llama 3.1 8B keeps 85% of its short-input score
   only up to 1k tokens [4]; small judges agree poorly with one another [19].
4. Tradeoffs: about 35,000 requests and 10 hours for the 30M-token Workflow
   session; extraction misses stay silent; and small is not cheap on W&B (GPT OSS
   120B input $0.03 per 1M tokens, Llama 3.1 8B $0.22 [2]).
5. Unmeasured: no study compares this with a large whole-thread judge on coding
   sessions; the experiment plan does, with the user's 20–40 labels.

## What "small" means on W&B today

Models of 70B or fewer parameters, or with few active parameters, that W&B lists
as generally available [1], with prices from CoreWeave [2]:

| Model | Active / total parameters | Context | Price per 1M tokens: input / output / cache hit | In Monopole's catalog |
| --- | --- | --- | --- | --- |
| Meta Llama 3.1 8B | 8B dense | 131k | $0.22 / $0.22 / none listed | yes (`catalogs.py:124-130`) |
| IBM Granite 4.2 8B | 8B dense | 131k | $0.10 / $0.15 / $0.05 | no; the catalog has `granite-4.1-8b` (`catalogs.py:110-116`), which W&B no longer lists |
| OpenAI GPT OSS 20B | 3.6B / 20B | 131k | $0.03 / $0.13 / none listed | yes (`catalogs.py:160-167`) |
| NVIDIA Nemotron 3.5 Lightning | 3B / 30B | 262k | $0.07 / $0.20 / $0.04 | no |
| Google Gemma 4 26B A4B | 4B / 26B | 262k | $0.10 / $0.30 / $0.05 | no |
| Qwen3.6 35B A3B | 3B / 35B | 262k | $0.25 / $1.25 / none listed | yes (`catalogs.py:91`) |
| Qwen3.8 27B | 27B dense | 262k | $0.40 / $3.00 / $0.15 | no; the catalog has Qwen3.6 27B (`catalogs.py:90`) |
| Google Gemma 4 31B | 31B dense | 262k | $0.10 / $0.34 / none listed | yes (`catalogs.py:109`) |
| Meta Llama 3.3 70B | 70B dense | 128k | $0.71 / $0.71 / none listed | yes (`catalogs.py:131-137`) |
| OpenAI GPT OSS 120B | 5.1B / 117B | 131k | $0.03 / $0.17 / none listed | yes; first default judge (`catalogs.py:357`) |
| For comparison: DeepSeek V4.1-Flash | 16B / 552B | 1,049k | $0.20 / $0.65 / $0.03 | no; the catalog has DeepSeek V4 Flash (`catalogs.py:93-99`) |

Two facts shape everything below.

Price does not follow size. GPT OSS 120B's input price is about a seventh of
Llama 3.1 8B's [2]. DeepSeek V4.1-Flash reads 1M tokens, costs about the same per
input token as Llama 3.1 8B, and also has a cache-hit price of $0.03 [2]. **Inference:** on W&B,
picking the smallest model saves little money. The reasons to use small models
are self-hosting, throughput, independence from one vendor, or the goal itself.
Open question 1 asks which one matters.

Advertised context is not usable context. Measured effective lengths:

| Model | Advertised | RULER effective length [3] | NoLiMa effective length [4] | NoLiMa score at 4k / 8k / 32k, short-input score in brackets [4] |
| --- | --- | --- | --- | --- |
| Llama 3.1 8B | 128K | 32K | 1K | 44.1 / 31.9 / 14.2 (76.7) |
| Llama 3.1 70B | 128K | 64K | 2K | 71.2 / 62.7 / 43.2 (94.5) |
| Llama 3.3 70B | 128K | not listed | 2K | 81.5 / 72.1 / 42.7 (97.3) |
| Gemma 3 27B | 128K | not listed | under 1K | 48.1 / 32.7 / 9.5 (88.6) |
| GPT-4o mini | 128K | not listed | under 1K | 44.1 / 32.6 / 13.7 (84.9) |
| GPT-4o | 128K | not listed | 8K | 95.7 / 89.2 / 69.7 (99.3) |

- RULER counts a length as effective while the model stays above Llama-2-7B's
  score at 4K, 85.6 [3].
- NoLiMa counts a length as effective while the model keeps 85% of its
  short-input score. Its questions share few words with the evidence they need
  [4].
- LongRM tested judges that pick the better of two answers given a long context.
  Under 1k tokens of context, most did well; Skywork-Critic-Llama-3.1-70B was
  near 100% on single-document questions. At 4k, every tested model fell below
  50%, the accuracy of guessing. A trained 8B judge from the same paper beat 70B
  baselines and matched Gemini 2.5 Pro [17].
- None of the current W&B small models (Gemma 4, Qwen3.6 and 3.8, GPT OSS,
  Granite 4.2, Nemotron 3.5) has an effective-length number in the sources read.
  The experiment plan measures it.

What this means for Monopole (**Inference**):

- Judging is closer to NoLiMa than to RULER. "Fixed; all tests pass" shares
  almost no words with `pytest -q` and `exit_code: 1`.
- Today's policy plans 50,000-token raw windows for every model at or below
  200,000 tokens (`run_config.py:53`). The default 8B judge therefore reads
  windows about 50 times its NoLiMa effective length.
- For a small model the question is not what fits, but what it can still
  connect. A question that links two pieces of evidence should see 2k to 4k
  tokens. Local reading, such as listing the claims in a run of messages, can
  take 8k to 16k.

## Strategy families

### 1. Small and fine-tuned judges

How it works. Either prompt a small general model as the judge, or use a small
model fine-tuned to judge (an evaluator model). Examples with their bases:

- Prometheus 2, on Mistral-7B and Mixtral-8x7B [8].
- JudgeLM, at 7B, 13B, and 33B [9].
- Auto-J, at 13B [10].
- FLAMe, at 24B [11].
- Skywork-Critic, on Llama-3.1 8B and 70B; JudgeBench evaluates it [12].
- Atla Selene Mini, on Llama-3.1 8B [15].
- J1, at 8B, 32B, and 70B, trained with reinforcement learning [16].

Section 4 covers small claim checkers.

Evidence on size:

- Prometheus 2. On FLASK, correlation with human scores (Pearson) was:
  GPT-4 0.679, Prometheus-2-8x7B 0.555, Prometheus-2-7B 0.545,
  Mistral-7B-Instruct 0.377, Llama-2-70B-Chat 0.317 [8]. On its own test set,
  drawn from its training data, it correlated 0.88 to 0.90 with GPT-4 [8].
- Judging the Judges. Thirteen judges graded TriviaQA answers, a task where
  humans agree. Only GPT-4 Turbo, Llama-3.1 70B, and Llama-3 70B aligned well
  with humans. Llama-3 8B agreed with humans on over 80% of answers but had a
  chance-corrected agreement (Scott's π) of only 59 out of 100 [14]. The lesson:
  raw percent agreement hides real disagreement.
- JudgeBench. These are hard pairs where one answer is objectively correct, so
  guessing scores 50% [12]. Accuracy:
  - Prometheus2-7B 34.9%, JudgeLM-7B 25.1%;
  - Skywork-Critic 8B 53.4%, Skywork-Critic 70B 57.4%;
  - GPT-4o 50.9% with a plain prompt;
  - with the Arena-Hard judge prompt, Claude-3.5-Sonnet 64.3% and o3-mini (high)
    80.9%.

  For a fixed model, judging accuracy closely tracked that model's accuracy at
  solving the problem [12]. Two fine-tuned judges have 2,048-token windows, so
  the benchmark authors had to truncate the answers [12].
- Huang et al. Fine-tuned judges beat GPT-4 on their in-domain test sets but fall
  behind on generalization, fairness, and adaptability. They behave as
  task-specific classifiers [13].
- Short-item benchmarks:
  - FLAMe-RM-24B 87.8% against GPT-4o 84.7% on RewardBench [11];
  - Selene Mini was the top 8B generative model on RewardBench, and beat
    GPT-4o-mini across 11 out-of-distribution benchmarks [15];
  - J1-Qwen-32B beat o1-mini and DeepSeek-R1 on some benchmarks [16].

  None of these benchmark items is a trace.
- OpenHands trained a small critic on coding-agent trajectories. Trained on
  benchmark-style data only, it was near random on real production outcomes
  (AUC 0.45 to 0.48). Training on production signals raised it to 0.58 (PR
  merged) and 0.69 (code survived) [7].

Known failure modes of small judges:

- Inconsistency. Asked each checklist question three times, Qwen-2-1.5B
  contradicted itself on over 50% of questions; smaller models were less
  consistent [20].
- Order effects. When checklist items were answered in one conversation, earlier
  answers swayed later ones, more so for smaller models [20]. In pairwise
  judging, swapping the two answers changed the verdict on 62% of MT-Bench
  writing pairs for LLaMA-2-7B-chat, 52% for LLaMA-2-70B-chat, and 17% for GPT-4
  [21].
- Length. Where humans preferred the shorter answer, LLaMA-2-7B-chat picked the
  longer one 55% of the time and GPT-4 39% [21].
- Leniency, and sensitivity to prompt length and complexity [14].
- Invented answers when the evidence is missing. LongAgent's 7B readers answered
  from chunks that lacked the answer until trained to refuse. Training raised
  correct refusals from 51.0% to 78.6% [24].
- Long input: the LongRM drop above [17].

Fit with Monopole:

- Monopole judges one session at a time, not pairs, so pairwise position bias
  does not apply directly. Position inside a long input does (long-session
  study, `docs/long-session-judging-research.md:167-206`).
- The evaluator models above expect a short instruction, a response, and a
  rubric; none was built to read traces. Two of the older ones have 2,048-token
  windows [12], against a median subagent of 26.5k tokens
  (`docs/large-session-anatomy.md:108-110`).
- A fine-tuned judge would need training on Monopole's own sessions to avoid the
  benchmark-to-production gap OpenHands measured [7]. That needs labels Monopole
  does not have (`specs/02-evaluation.md:364-366`).

Cost. Lowest per request only where the price is low (table above).

Failure modes. The list above. For Monopole the decisive one is long input.

### 2. Breaking a rubric into yes/no questions

How it works. Replace one overall score with a list of yes/no questions, answer
each one separately, and combine the answers by a rule or by a model. Variants:

- TICK has a model write a checklist for each instruction [18].
- CheckEval writes questions for each evaluation dimension, filtered by people
  [19].
- RocketEval has a strong model write 5 to 10 questions for each item once. A
  lightweight model then answers each question in its own request, and the
  probability of "Yes" against "No" serves as a soft score [20].
- InFoBench splits an instruction into requirements and scores the fraction met
  [22].
- Branch-Solve-Merge plans criteria, judges each, and merges [21].

Evidence:

- RocketEval, on MT-Bench human judgments, where two people agree 64.7% of the
  time and GPT-4o agrees with people 66.6% [20]:
  - Llama-3-8B rose from 55.6% with reasoning-then-score prompting to 63.8% with
    checklists. Qwen2.5-1.5B rose from 31.1% to 60.7%, and Gemma-2-2B from 37.9%
    to 57.9%.
  - A fixed, generic list of six questions did not help and sometimes hurt;
    Llama-3-8B fell to 46.9%.
  - Giving small judges GPT-4o's written analysis raised Llama-3-8B's agreement
    with GPT-4o from 48.4% to 74.0%. The authors conclude that small judges fail
    mainly at analysis, not at scoring.
  - Ranking chat models on WildBench, Gemma-2-2B with checklists reached
    Spearman 0.965 against human rankings. That is agreement on rankings, not on
    single items.
- CheckEval, with 12 judges on SummEval and Topical-Chat [19]:
  - Llama-3.1-8B's correlation with people rose from 0.24 to 0.41 on SummEval
    and from 0.11 to 0.46 on Topical-Chat. Gemma2-9B and Qwen2.5-7B moved by 0.01
    to 0.05.
  - Agreement among judges of similar size rose for every size. The 7B–9B group
    stayed low, at Krippendorff's α 0.24 and 0.17, against 0.67 for the
    70B–123B group. Krippendorff's α is agreement corrected for chance, where 1
    is perfect and 0 is chance.
- Branch-Solve-Merge, on MT-Bench writing questions [21]:
  - LLaMA-2-70B-chat's agreement with people rose from 0.43 to 0.55, against
    0.59 for GPT-4. Verdict changes on swapped order fell from 52% to 17%.
  - LLaMA-2-7B-chat's agreement moved only from 0.39 to 0.41.
  - Across domains, the 70B model gained up to 26 points.
- TICK, with GPT-4o as the judge. Exact agreement with trained annotators rose
  from 46.4% (direct score) to 52.2% (checklist). Showing the checklists to human
  raters raised their agreement with each other from 0.194 to 0.256 [18]. This is
  not small-model evidence.
- Web-Shepherd. Web-agent judges built on Qwen2.5-3B and Qwen3-8B, trained with
  checklists, scored about 30 points above GPT-4o on the authors' benchmark,
  WebRewardBench [46]. They were trained in-domain.

Fit with Monopole:

- Each rubric already separates allowed evidence, excluded concerns, and five
  anchors (`rubrics.py:36-65`). That structure maps onto questions (table under
  Step 3).
- RocketEval's negative result matters most. A fixed question list per rubric
  did not help; questions about the item's own content did [20]. Monopole can
  write per-session questions with code: the ledger names the claim, the edit,
  and the check, and the question names them too. **Inference.**
- Answers cite evidence IDs, so a merge can cite them as findings, and the
  verdict contract stays as it is (`specs/02-evaluation.md:126-139`).
- RocketEval's soft score needs token probabilities from the provider. Whether
  W&B returns them for these models was not checked (open question 4).

Cost. One request per question. RocketEval used prefix caching so that all
questions about one item share a cached prompt start [20]. W&B lists cache
prices for some small models [2].

Failure modes:

- A question list misses what nobody asked. The 0.75 and 1 anchors ("every
  material claim", "every relevant established decision") need coverage, which a
  list cannot prove (`rubrics.py:145-148`, `rubrics.py:297-300`).
- Two small judges agreeing is weak evidence, since small judges agree poorly
  [19]. Disagreement between them is a cheap uncertainty signal.
- Answers still have to become an anchor. RocketEval's best combiner was trained
  on labeled data [20].

### 3. Reading long inputs through small windows

How it works. Split the input into chunks the model handles well:

- workers read chunks in order and pass notes to a manager (Chain of Agents)
  [23];
- a leader questions many chunk readers and resolves their conflicts (LongAgent)
  [24];
- an agent reads short "gist" notes and opens original pages on demand
  (ReadAgent) [25];
- a model pages its own memory in and out with function calls (MemGPT) [26];
- a tree of summaries (RAPTOR) [27].

The companion studies describe the designs. This section adds window and model
sizes.

| Method | Window per read | Models | Compared with | Result |
| --- | --- | --- | --- | --- |
| Chain of Agents [23] | 8k | PaLM 2 (8k), Gemini Ultra (32k), Claude 3 Haiku, Sonnet, Opus | Claude 3 reading 200k at once; retrieval | 8k chains beat 200k Claude 3 on NarrativeQA and BookSum; the gain grew from Haiku to Opus |
| LongAgent [24] | 4k | fine-tuned LLaMA2-7B | GPT-4 (128k), Claude 2.1 | 81.53% against 62.00% on single-document needle questions; 55.33% against 50.37% on multi-document |
| ReadAgent [25] | 8k | PaLM 2-L; GPT-3.5 in an appendix | full text, retrieval | 3.5 to 20 times the effective context with PaLM 2-L; with GPT-3.5 on QuALITY, gists alone 66.1%, full text 73.3%, gists plus look-ups 72.1% |
| MemGPT [26] | 8,192 (GPT-4) | GPT-4, GPT-4 Turbo, GPT-3.5 | fixed-context models | accuracy stayed flat as more documents were retrieved; the only method that solved nested key-value lookups beyond 2 levels |
| GraphReader [28] | 4k, chunks of at most 2k | GPT-4-128k | the same model reading the full text | beat full-text GPT-4 from 16k to 256k on LV-Eval |
| Graph of Agents [29] | 2k | Llama 3.1 8B, Qwen3 8B | retrieval; fixed agent chains; Llama 3.1 8B at 128k | 5.7% better F1 than retrieval and 16.35% better than fixed chains; at 2k it beat 128k Llama 3.1 8B on LongBench |
| Tree of Agents [30] | chunks | Llama 3.1 8B, DeepSeek-V3 | baselines including Chain of Agents | 8B results comparable to Gemini 1.5 Pro; Chain of Agents did poorly with Llama 3.1 8B, which the authors blame on information decaying along the chain |
| MemAgent [31] | 8k: 5,000 text, 1,024 memory | Qwen2.5 7B and 14B, trained with reinforcement learning | Qwen2.5-7B and 14B 1M-context instruct models | 71.9% for 7B at 3.5M tokens; the 1M-context instruct models fell to 0% at 896k |
| LLM×MapReduce [32] | chunks | Llama3-70B, Qwen2-72B | GPT-4, Claude 2, LongAgent, Chain of Agents | beat representative long-context models on InfiniteBench, whose inputs average over 100k tokens |

Lessons that carry over:

- Small windows beat long windows for 7B–8B models. Graph of Agents running Llama
  3.1 8B at 2k beat the same model reading 128k [29], and MemAgent's trained 7B
  reader beat 1M-context 7B models [31].
- The strongest 7B–8B results came from training (LongAgent, MemAgent) or a
  carefully built structure (Graph of Agents). Plain chains of small models lost
  information [30].
- Gists without look-ups lose accuracy [25].
- A reader asked about a chunk that lacks the answer invents one unless trained
  to refuse [24].
- Giving Chain of Agents' manager every worker's output hurt, because the outputs
  conflicted [23].

Fit with Monopole:

- These methods answer one question about one document. Monopole asks six rubric
  questions, each needing many pieces of evidence, about a document whose
  structure is recorded. **Inference:** what carries over is the window size (2k
  to 8k) and the reader's right to say "not in this window", not a free-form
  reading agent.
- Monopole already reads in windows. For small models, the change is the window
  size and what windows return: ledger entries and question answers instead of
  free-text digests and findings.
- MemGPT-style paging depends on reliable function calls. How well 8B–30B models
  page through hundreds of pages was not measured in the sources read.

Cost. Each pass reads the input once. Chain of Agents grows linearly with input
length times window size, instead of with the square of the input [23].

Failure modes. Information decays along chains [30], worker outputs conflict
[23], readers invent answers [24], and gists drop detail [25].

### 4. Claim extraction and verification

How it works. Split text into small claims, fetch evidence for each claim, and
check each claim against its evidence:

- FActScore splits text into atomic facts [33].
- SAFE checks each fact against Google Search results [34].
- VeriScore keeps only claims that can be checked [36].
- RefChecker writes claims as subject–predicate–object triplets [38].
- Small trained checkers do the checking: AlignScore (355M, in the windowed
  study, `docs/windowed-judging-research.md:259-263`) and MiniCheck [35].

Evidence:

- FActScore's automatic estimate, built on retrieval and a strong model, was
  within 2% of human scores [33].
- SAFE agreed with crowd annotators on 72% of about 16,000 facts and was right in
  76 of 100 disagreements, at over 20 times lower cost [34].
- MiniCheck. A 770M Flan-T5 checker reached GPT-4's accuracy on LLM-AggreFact at
  about 400 times lower cost. With its trained checker, splitting claims into
  atomic facts was not needed [35].
- The LLM-AggreFact leaderboard today, balanced accuracy over 11 datasets of
  claims checked against documents [5]:

  | Checker | Size | Balanced accuracy |
  | --- | --- | --- |
  | nimble-v4-max | 27B | 79.5 |
  | nimble-v4 | 9B | 77.8 |
  | Bespoke-MiniCheck-7B | 7B | 77.4 |
  | Claude-3.5 Sonnet | not stated | 77.2 |
  | Granite Guardian 3.3 | 8B | 76.5 |
  | GPT-4o | not stated | 75.9 |
  | FactCG-DeBERTa-L | 0.4B | 75.6 |
  | MiniCheck-Flan-T5-L | 0.8B | 75.0 |
  | Llama-3.3-70B-Instruct | 70B | 74.5 |
  | Llama-3.1-405B-Instruct | 405B | 74.4 |

- VeriScore works with closed models or with fine-tuned open-weight models [36].
  Claimify extracts a claim only when it is confident how the text should be read
  [37]. Triplet-level claims found hallucinations better than whole responses,
  sentences, or sentence parts [38].

Fit with Monopole:

- Monopole's claims are about the session's own actions, not about the world.
  "Fixed; all tests pass" is checked against recorded tool calls, which code
  fetches by evidence ID and position (`windowing.py:179-185`). The retrieval
  step that these systems need a model or a search engine for is exact here.
  **Inference.**
- Claims sit in assistant messages, which are 2.1% to 13.1% of a session's
  content (`docs/large-session-anatomy.md:62-68`). Extraction reads a small part
  of the session.
- Most checks are commands with exit codes that code already parses
  (`scorers/outcome.py:149-196`, `scorers/outcome.py:655`). A model is needed for
  relevance ("does this test cover the claimed change?") and for output the
  parsers do not recognize.
- LLM-AggreFact's documents are prose. Stack traces and test summaries are a
  different kind of text, so the leaderboard does not transfer directly.
  **Inference.**

Cost. One extraction request per run of messages and one check per claim and
evidence pair. Trained checkers can run locally.

Failure modes:

- Indirect claims such as "should be good now" escape extraction
  (`docs/windowed-judging-research.md:290-291`).
- A claim that needs two pieces of evidence from different places
  (`docs/windowed-judging-research.md:292-293`).
- Checkers trained on prose may misread logs.

### 5. Deterministic features and hybrid code-plus-model judging

How it works. Compute everything code can compute exactly, give it to the judge
as facts, and let rules flag cases for model review.

What code can compute from Monopole's sessions:

| Feature | Where | Status |
| --- | --- | --- |
| Test, build, lint, install, and git commands and their outcomes | `scorers/outcome.py:149-196`, `outcome.py:702`, `outcome.py:760` | exists |
| Exit code and permission decision per tool call | `models.py:41-51`, `ces_client.py:91-93` | exists |
| Error loops: three or more consecutive calls to one tool with similar arguments, at least 60% failed | `scorers/efficiency.py:52-79` | exists |
| Repeated reads with no edit in between, reset at compactions | `scorers/efficiency.py:90-127` | exists |
| Steering, denial, and tool-error counts per turn | `models.py:94-96` | exists |
| Compaction boundaries | span events, `scorers/efficiency.py:92-109` | exists |
| Edits and the files they touch | `Edit` and `Write` calls (`efficiency.py:117-121`); edits made through Codex `exec` commands need parsing | partial |
| Order facts: the last edit before a claim, the checks in between | step positions | new, cheap |
| Results repeated across subagents (73% of result lines in the Workflow session) | `docs/large-session-anatomy.md:131-141` | measured, not in code |

Evidence:

- R2E-Gym, picking the best of several agent attempts on SWE-Bench Verified.
  Test-based checks and a model judging the trajectory each topped out around
  42% to 43%; together they reached 51%. Tests separated candidates poorly; the
  model judge leaned on style [43].
- AgentRewardBench, judging whether web agents succeeded [44]:
  - The benchmarks' official rule-based checks had 83.8% precision but 55.9%
    recall. Rules miss real successes.
  - The authors' model judges had 61.5% to 69.8% precision and 79% to 90% recall.
    They credit some failures. Llama 3.3 70B had 67.7% precision, GPT-4o 69.8%.
  - Two expert annotators agreed on 89.3% of success labels.
- LiteTrajEval. Rules mark failure signals, the trajectory is cut to a fixed
  budget, and one rubric-guided judge writes the report. Agreement with human
  failure localization improved by 20 to 35 points over a baseline on
  Magentic-One trajectories, at about 6 times lower cost and over 8 times less
  time [49].
- judgeman (described under section 7): free rules and a 9B judge see every
  step, and a frontier model sees the flagged steps [50]. Its rules lean on the
  command's return code [6].

Fit with Monopole:

- Code can fill most of the windowed study's ledger: checks, edits, errors,
  delegations, and the steering flag
  (`docs/windowed-judging-research.md:579-590`). Small models are needed only
  where meaning matters.
- The parsers favor precision over coverage (`specs/02-evaluation.md:34-36`), so
  a check they do not recognize never enters a pair. The step labels in Step 3
  are the backstop, because a model reads every step.

Cost. None for the code.

Failure modes:

- Rules differ across harnesses. Codex runs nearly everything through `exec`
  (1,428 of 1,458 calls in session A, `docs/large-session-anatomy.md:55`).
- An unrecognized check looks like no check. Verification scores a claim with no
  supporting check at 0.25 (`rubrics.py:121-127`, `rubrics.py:133-136`), so a
  parser miss can lower a score.

### 6. Cascades and escalation

How it works. Start with a cheap judge. Send an item up when the cheap judge is
unsure, when two cheap judges disagree, or when a rule contradicts the judge.

Evidence:

- FrugalGPT. A learned cascade of models matched GPT-4 at up to 98% lower cost,
  or beat it by 4% at the same cost [39].
- RouteLLM. Routers trained on preference data cut cost by over half in some
  cases without losing quality [40].
- Trust or Escalate [41]:
  - The judges were a cascade of Mistral-7B, GPT-3.5, and GPT-4. Each judge
    passes an item up when its confidence is below a threshold, and the
    thresholds are set on a human-labeled calibration set.
  - On a Chatbot Arena subset where GPT-4 alone almost never reached 80% human
    agreement, the cascade guaranteed over 80% agreement while still deciding
    almost 80% of the items.
  - Calibration sets had 500 labeled items, or 392 for one smaller benchmark.
  - A method called Simulated Annotators, which asks the model to answer as
    several different annotators, estimated confidence better than token
    probabilities or the model's stated confidence, especially for weak judges.
- judgeman: the frontier model judged only the third of steps that rules or the
  9B judge flagged [6][50].
- Kale et al. Escalating pre-flagged cases to humans raised true positives by
  about 15% at a 1% false-positive rate (`docs/windowed-judging-research.md:824-828`).

Fit with Monopole:

- The windowed design proposes a rule-based roll-up beside each session verdict
  that flags large gaps for review (`docs/windowed-judging-research.md:678-682`).
  Escalation adds more triggers of the same kind.
- A formal guarantee needs hundreds of labels. Suppose a judge disagrees with 3
  of 30 human labels (10%). The exact binomial 90% upper bound on its true
  disagreement rate is then 21%; with 50 of 500 it would be 12%. **Inference**,
  computed. The user's 20–40 labels can set thresholds and expose large gaps, but
  they cannot certify an agreement level.
- An escalated item is an ordinary judge request with its own request key, so
  it resumes and pins like any other (`specs/02-evaluation.md:284-300`).
  **Inference.**

Cost. Set by the escalation rate. In judgeman, about a third of steps reached the
frontier model [6].

Failure modes:

- Confident mistakes never escalate, and small judges lean lenient [14].
- Unusual sessions can escalate far more than usual, so cost needs a cap.

### 7. What agent-trajectory benchmarks show about small judges

Evidence:

- TRAIL, which asks judges to find and classify errors in long agent traces [47].
  On the GAIA traces:

  | Judge | Error-category F1 | Error-location accuracy |
  | --- | --- | --- |
  | Llama-4-Scout (17B active parameters) | 0.041 | 0.000 |
  | Llama-4-Maverick (17B active parameters) | 0.122 | 0.023 |
  | GPT-4.1 | 0.218 | 0.107 |
  | Claude-3.7-Sonnet | 0.254 | 0.204 |
  | o3 | 0.296 | 0.535 |

  The best model overall scored 11%. Several models could not take the
  SWE-bench traces at all because they exceeded the context window [47].
- AgentRewardBench. Llama 3.3 70B, judging web-agent success from text, came
  within about 2 points of GPT-4o's precision [44].
- WebJudge, a model judge for web agents, agreed with people about 85% of the
  time. A 7B version, Qwen2.5-VL-7B trained to pick the key screenshots, kept
  comparable or higher agreement at lower cost [45].
- Web-Shepherd. 3B and 8B checklist-trained judges beat GPT-4o by about 30
  points on WebRewardBench [46].
- Counsel. People rated open-weight judges' step-level critiques of τ-bench and
  DA-Code traces. More capable judges and more reasoning effort raised agreement;
  the best reached about 88% on error location and about 65% on reasoning [48].
- judgeman, a working paper with an open-source tool. It labels 516 steps from
  mini-SWE-agent runs on SWE-bench Verified on five yes/no axes: progress,
  redundant, risky, unverified completion, and outcome-process mismatch [50]:
  - Free rules and a 9B open judge see every step, and a frontier model sees only
    the flagged steps. That pipeline reached Cohen's κ 0.72 (95% interval 0.50 to
    0.86), against 0.61 for the frontier model alone, at about a third of the
    cost.
  - Two careful human labelers agreed at κ 0.68 on whether a step was wasted.
  - Larger open models, few-shot examples, hindsight context, a tighter rubric,
    and two fine-tunes did not close the gap to that ceiling.
  - Its tool reads Claude Code logs. It treats one tool call and its result as a
    step, and costs about $3.50 for a 400-step session [6].

What this says (**Inference**):

- Small models fail at the open-ended job: find and classify every error in a
  long trace (TRAIL). They do well at bounded jobs: a yes/no about one step, a
  success label from a short summary, a checklist item (judgeman, WebJudge,
  Web-Shepherd).
- The strongest small results were trained in-domain. Prompted small models need
  the most structure.

## Proposed stitched pipeline for Monopole

### What the user chose and what this proposal adds

Choices already made by the user
(`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:202-218`, and
this request):

1. Judges see every message.
2. Workflow sessions get session, orchestrator, and per-subagent scores.
3. Partial work is recorded, and reruns skip finished work.
4. The user labels 20–40 disagreement cases for validation.
5. Large threads should be judgeable by as small a model as possible.

Details added here, each open to change:

- Small models read each step raw once, instead of reading 50,000-token windows.
- Code writes the questions from the ledger, and small models answer them in
  separate requests.
- Two small judges from different model families answer each question, and a
  disagreement escalates it. Panels of smaller models from different families
  beat a single large judge in PoLL
  (`docs/long-session-judging-research.md:374-376`).
- A mid-size model writes each thread's verdict. 8B-class models never do.
- Code also derives a candidate anchor from the answers. It is stored for audit,
  hidden from the merge, and used to trigger escalation.
- A large model sees only escalated items, and serves as the experiment's
  reference judge.
- Window sizes per model come from its measured effective context on Monopole's
  own questions.

### Step 1. Extract facts by code

1. Build the thread tree with the reader the windowed study proposes
   (`docs/windowed-judging-research.md:554-556`).
2. Record each step's kind, evidence ID, thread, and position: user message,
   assistant message, tool call, or compaction.
3. Write the code-derived ledger entries: checks, edits, errors, error loops,
   repeated reads, pointers to repeated results, steering and denial events,
   delegations, and compaction boundaries (section 5 table).
4. Build pairs with the windowed study's rules
   (`docs/windowed-judging-research.md:609-615`). A claim pairs with the last edit
   before it, the checks in between, and the first check after it. An error pairs
   with the next five steps. A constraint pairs with later edits to the same file.
   A correction pairs with earlier steps on the same file. Claims and constraints
   come from Step 3.

No model runs in this step.

### Step 2. Cut structural windows

Cut in this order: thread, then compaction stretch, then turn, then step.

| Unit | Size in the measured sessions | Role for small models |
| --- | --- | --- |
| Thread | subagents: median 26.5k, largest 161k; orchestrator 1.9M (`docs/large-session-anatomy.md:17-19`, `:108-113`) | the unit that gets a verdict |
| Compaction stretch | 520k or less (`docs/large-session-anatomy.md:93-97`) | groups merges; marks where the agent's own context was reset |
| Turn | up to 1.07M (`docs/large-session-anatomy.md:93`) | groups tallies for audit |
| Step | one tool call with its result, capped at 12,000 characters, about 4,000 tokens by bytes ÷ 3 (`windowing.py:17`); or one message | the raw reading unit |
| Message run | consecutive user and assistant messages, with each tool call shown as a one-line host summary (name, command, exit code, evidence ID), packed to 8k–16k tokens | the extraction unit |

Coverage. One step-label request reads each tool call raw, and one extraction
request reads each message raw. That keeps "judges see every message" while no
request exceeds a small model's effective context. It extends today's rule of
complete raw-turn coverage (`AGENTS.md:117`). Verdict requests read answers, not
raw steps, as today's merge reads findings.

### Step 3. What a small model does in each window

Step labels, one request per tool-call step, with no rubric:

- Input: the thread's task (its first user message, cut to 1k tokens), the
  agent's message just before the step, the step, and a one-line summary of the
  next step.
- Output: yes/no for wasted, repeated, risky, failed-and-ignored, and "is a
  check" with passed or failed, each with a short reason.
- The axes adapt judgeman's [50]. "Is a check" catches checks the parsers miss.

Extraction, one request per message run, with no rubric. It lists:

- completion and correctness claims, with the files or features they are about;
- constraints and stated plans;
- the content of user corrections;
- user requests and acceptance criteria.

Each entry cites one message ID. The host merges these entries with the code
entries into one ledger.

Questions, one request per question. Code turns each pair into a question naming
its evidence IDs. The bundle is 4k tokens or less. The answer is yes, no, or
"not in this evidence", with cited IDs and a short exact quote.

| Rubric | Questions, and who answers | Bundle |
| --- | --- | --- |
| Verification (`rubrics.py:102-152`) | Is message M a material completion or correctness claim, and about what? (small) Is check C relevant to M's subject? (small) Did C pass? (code; small only when unparsed) Was a relevant passing check run after the last edit and before M? (code, from the answers) | M, the user request, C's command and the head and tail of its output (2k) |
| Error recovery (`rubrics.py:154-203`) | Does the action after error E respond to E's content, or repeat it unchanged? (small) Is E resolved or contained within the next five steps? (small) | E's output (2k) plus one-line summaries of the next five steps and the first relevant output |
| Tool choice (`rubrics.py:205-253`) | the step labels; was a destructive command needed? (code flags by pattern, small confirms) | one step |
| State consistency (`rubrics.py:255-304`) | Does edit or command Z contradict constraint K? (small) Did the user authorize the change between K and Z? (small) | K's text, Z's arguments or diff (4k), and the user messages in between |
| Outcome (`rubrics.py:306-354`) | For each request R: do the final message and final state show R done? (mid-size first; small as an experiment arm) Are errors unresolved at the end? (code) | R, the final assistant message, and the last checks (8k) |
| Autonomy (`rubrics.py:356-404`) | Is user message U a correction, redirect, or re-explanation? (small) Was U's information given earlier? (small) Could the agent have avoided its question to the user? (small) | U plus all earlier user messages, which are 0.1% to 6.0% of content (`docs/large-session-anatomy.md:67`) |

Candidate anchor, by code. A fixed rule maps answers to an anchor. Example for
verification: a material claim with a relevant, unresolved failing check gives 0
(`rubrics.py:129-132`). A material claim with no relevant check gives 0.25
(`rubrics.py:133-136`). The candidate is stored for audit and never shown to the
merge, as the windowed design keeps its roll-up hidden
(`docs/windowed-judging-research.md:678-682`).

### Step 4. Resolve claims and checks across windows

Code builds every question from host pairs. A pair whose two ends fall in
different windows is just a question whose bundle comes from two places. No
digest is needed to carry it.

**Inference:** for small models, this replaces today's digests outright. A
1,000-token free-text digest of a 50,000-token window, written by an 8B model,
sits far outside that model's measured effective context. That is a stronger
case than the windowed study's gate, which keeps digests unless planted probes
show they miss links (`docs/windowed-judging-research.md:546-547`).

Worked example, the windowed study's parser claim
(`docs/windowed-judging-research.md:90-104`):

1. Extraction lists `m-430` ("Fixed; all tests pass.") as a completion claim
   about `parser.py`.
2. Code finds:
   - the last edit to `parser.py` before `m-430`: `t-415`, turn 11;
   - checks between `t-415` and `m-430`: none;
   - the first check after the claim: `t-2210`, `pytest`, exit 0, turn 64;
   - an earlier check: `t-301`, failed, turn 8.
3. A small judge answers "Is `t-2210` relevant to the claim in `m-430`?" with
   yes, citing `t-2210`. Code records that the only relevant passing check came
   after the claim.
4. The candidate anchor is 0.25: a claim with no relevant check behind it.
5. The mid-size merge sees the claim entry, the answer, and both positions. It
   returns 0.25, or 0.75 if it judges the gap "limited and clearly non-critical"
   (`rubrics.py:141-144`). A gap of 0.5 between merge and candidate triggers
   escalation.

### Step 5. Aggregate into turn, subagent, orchestrator, and session scores

- Turn. Code tallies, per turn, the claims, claims without checks, errors,
  recovered errors, and flagged steps. These are stored as audit artifacts.
  Monopole's rubrics are session-level (`specs/02-evaluation.md:50`), so turn
  Scores are not proposed (open question 6).
- Thread verdict. For each rubric, a mid-size model reads the thread's ledger,
  question answers, and step-label tallies with the flagged steps and their
  positions, within 32k tokens. It returns today's verdict contract
  (`specs/02-evaluation.md:126-139`). When the input exceeds 32k, as for the
  orchestrator, it merges each compaction stretch first and then the thread.
- Subagent scores. The verdicts of subagent threads, on the rubrics the windowed
  study's scope table assigns (`docs/windowed-judging-research.md:657-664`).
- Orchestrator scores. Its thread verdict. At each launch call it sees the
  subagent's verdict summary of about 300 tokens
  (`docs/windowed-judging-research.md:646-649`).
- Session score. A grouped merge over orchestrator stretches and the subagents
  each one launched (`docs/windowed-judging-research.md:650-653`). The stored
  roll-up and its flags sit beside it.

### Step 6. Escalate

| Trigger | Goes to |
| --- | --- |
| The two small judges disagree on a question, or one answers "not in this evidence" when code says the evidence is in the bundle | a mid-size model, same bundle |
| The merge anchor and the candidate anchor differ by 0.5 or more | a large model, for that thread and rubric: the same inputs plus the raw bundles of the cited steps |
| A thread scores 0 or 0.25 on verification or error recovery | a large model confirms; an experiment arm, kept only if it pays off |
| The session score lies outside the range of component anchors, or more than 0.5 from their mean | the user's review queue (`docs/windowed-judging-research.md:678-682`) |
| The large reference judge and the small pipeline differ by 0.5 or more | the user's 20–40 labels |

Each session gets an escalation budget. Beyond it, the remaining triggers are
recorded but not sent, and each affected verdict keeps the small pipeline's
answer and lists the escalations it skipped. This fits the draft design's rule
that finished work is kept and gaps are named
(`docs/superpowers/specs/2026-11-03-long-session-judging-design.md:159-171`).

### Model size and window per step

| Step | Job | Smallest model the evidence supports | Window | Evidence | Confidence |
| --- | --- | --- | --- | --- | --- |
| 1 | facts and pairs | none: code | not applicable | existing parsers (section 5) | high for recognized commands |
| 3 | step labels | 9B, with rules and escalation behind it | one step: 4k or less, plus 1k–2k of context | judgeman [50] | medium: one working paper, one agent harness |
| 3 | extraction of claims, constraints, requests | not established below GPT-4-class models; test 8B against 27B–31B | 8k–16k of messages | FActScore and SAFE used large models [33][34]; VeriScore worked with fine-tuned open models [36] | low |
| 3 | linking questions (claim and check, constraint and edit, error and next action) | 7B–9B prompted with per-session questions; 0.4B–9B if trained | 2k–4k | RocketEval [20], LLM-AggreFact [5], LongRM [17], NoLiMa [4] | medium on prose; untested on tool output |
| 3 | outcome per request | 27B or larger, as an assumption | 8k | no direct evidence | low |
| 5 | thread verdict | 27B–70B | 32k | Branch-Solve-Merge [21], CheckEval [19], Judging the Judges [14] | medium |
| 5 | stretch and session merges | 27B–70B; a large model on flags | 32k per group | same | low |
| 6 | escalation | large: the 1M-token models in the long-session study | one bundle or one thread | Trust or Escalate [41], judgeman [50] | medium |

### Cost and time, worked on the Workflow session

All **Inference**, from these assumptions:

- Session B has 18,182 tool calls across the main thread and its 605 subagents
  (`docs/large-session-anatomy.md:55`; the main thread has only 3,188 Cases).
  Messages are about 10% of its 36.8M raw tokens (`:64-68`).
- A step label reads the step plus 2,000 tokens of prompt and context, and
  writes 150 tokens.
- Ten questions per subagent and 300 for the orchestrator, 3,000 tokens each,
  each answered by two small judges.
- Five rubric verdicts per subagent and about 40 orchestrator and session
  merges. Inputs average 8,000 tokens for a subagent verdict and 20,000 for a
  merge.
- 20% of verdicts escalate to DeepSeek V4-Pro-0813 at 20,000 tokens in and 1,000
  out ($1.31 and $3.96 per 1M [2]).
- Output counts exclude hidden reasoning. GPT OSS models reason by default,
  which can multiply output tokens.

| Part | Requests | Input tokens | Output tokens |
| --- | --- | --- | --- |
| Step labels | 18,182 | about 63M (27M of tool text read once, plus 36M of overhead) | 2.7M |
| Extraction | about 1,000 | about 4.5M | 0.5M |
| Questions, two judges each | about 12,700 | about 38M | 1.3M |
| Verdicts and merges | about 3,070 | about 25M | 2.5M |
| Escalations | about 600 | about 12M | 0.6M |
| Total | about 35,500 | about 143M | about 7.6M |

Dollars at W&B list prices [2]:

- GPT OSS 20B for the small steps and Gemma 4 31B for verdicts: about $7, plus
  about $18 of escalation.
- Llama 3.1 8B and Llama 3.3 70B instead: about $44, plus the same escalation.

Wall time. Monopole runs at most four W&B requests at once
(`specs/02-evaluation.md:243-245`). At an assumed 4 seconds per short request,
35,500 requests take about 10 hours.

For comparison, a large whole-thread judge on the same subagents: DeepSeek
V4.1-Flash reads each subagent once at $0.20 per 1M tokens and four more times
from cache at $0.03. That is about 28.5M × $0.32 per 1M ≈ $9, plus about $3 of
output, in 3,025 requests, roughly 6 hours at 30 seconds each. The orchestrator
still has to be judged by compaction stretch. Today's pipeline makes 41 requests
per reviewer on the windowed study's example and reads no subagent internals
(`docs/windowed-judging-research.md:731`).

What this shows (**Inference**): the stitched small-model pipeline is not cheaper
than a cheap 1M-context MoE judge on W&B, and it needs more than ten times the
requests. Its case rests on quality per thread, which the experiment measures,
and on being able to run on small models at all.

### What stays the same and what changes

Stays the same:

- The verdict contract and abstention (`specs/02-evaluation.md:126-139`).
- Request-keyed resume. Each question, label, and verdict is its own reusable
  call, so a rerun skips finished ones, as the user chose
  (`specs/02-evaluation.md:284-300`).
- Pinning. The thread tree, ledger rules, question templates, and window sizes
  join the pinned plan.

Changes:

- New phases. Step labels, extraction, questions, and escalation are new request
  types, with new protocol and plan versions (`sliding.py:68`, `windowing.py:16`).
- New roles. Today a panel has one to three reviewers, each running the whole
  pipeline (`AGENTS.md:122`). Here one reviewer uses several models: small
  question judges, a mid-size verdict writer, and a large escalation model. The
  catalog and run setup need role assignments.
- The frontend estimate that copies the plan arithmetic
  (`frontend/src/features/runs/modelCapacity.ts:35-94`).

## Experiment plan

Goal. Decide whether the stitched small-model pipeline can replace or support a
large whole-thread judge. Find the smallest model for each step, and set window
sizes, all on the user's own sessions. Everything runs on offline copies; nothing
is written to CES.

1. Units. Use threads that a large model can read whole, so that each one has a
   large-model reference:
   - session B's 605 subagents, each 161k tokens or less;
   - session A's 18 compaction stretches, each 219k or less;
   - session B's 9 orchestrator stretches, each 520k or less;
   - sessions C and D whole;
   - about 50 ordinary recent sessions under 128k.

   Each thread is scored on each rubric
   (`docs/large-session-anatomy.md:93-97`, `:108-115`).
2. Arms:

   | Arm | What runs |
   | --- | --- |
   | **large** | one 1M-token model judges each whole thread, one request per rubric, transcript first and rubric last, cached (`docs/long-session-judging-research.md:271-276`) |
   | **stitched** | the pipeline above, with no escalation |
   | **stitched+escalation** | the pipeline with escalation; record the escalation rate |
   | **small-today** | the same small model in today's digest, window, and merge pipeline at its default 50k windows, to measure what stitching adds |

   A size ladder runs the step-label and question steps at four sizes, plus the
   large arm's model on the same questions:
   - 8B dense: Llama 3.1 8B, Granite 4.2 8B;
   - MoE with 3B–5B active: GPT OSS 20B, Nemotron 3.5 Lightning, Gemma 4 26B A4B;
   - 27B–31B dense: Qwen3.8 27B, Gemma 4 31B;
   - 70B: Llama 3.3 70B.
3. Planted questions with known answers. Reuse the windowed study's planted pairs
   (`docs/windowed-judging-research.md:782-797`) and add:
   - verification: a relevant against an irrelevant check (a test of another
     module), passing against failing, before against after the edit;
   - error recovery: the same failing command repeated against a changed command;
   - state consistency: an edit to a file the user forbade against a permitted
     file;
   - autonomy: a correction that repeats earlier information against one that
     adds new information.

   Plant each at three bundle sizes, 1k, 4k, and 16k tokens, by adding
   neighboring steps. This measures each model's effective window on Monopole's
   own questions, as NoLiMa does on its own [4].
4. Record:
   - question accuracy on planted answers;
   - step-label agreement with the large model's labels on the same steps;
   - for thread anchors: exact agreement, agreement within 0.25, ordinal
     Krippendorff's α, mean signed difference, and abstention disagreements
     counted separately (`docs/long-session-judging-research.md:634-641`);
   - repeat noise: three runs on 30 threads at temperature 0.2 (`sliding.py:67`);
   - requests, input, cached, and output tokens, dollars, wall time, and the
     escalation rate.
5. The user's 20–40 labels:
   - Selection. Take the thread-and-rubric cells where **stitched+escalation**
     and **large** differ by 0.5 or more, or where one abstains. Spread them
     across the six rubrics and across subagents, orchestrator stretches, and
     ordinary sessions. Add 5 to 8 cells where the two agree, to check that
     agreement means both are right.
   - Blind review. The user sees the thread view and the evidence both verdicts
     cite, not which arm wrote which.
   - Each label records the anchor or abstention, and the evidence IDs that
     decided it.
   - Self-consistency. With one labeler, inter-rater agreement cannot be
     measured, so relabel 5 cells a week later.
   - Analysis. Count, in disagreement cells, which arm matched the user. With 30
     cells where exactly one arm matches, one arm must win at least 21 for a
     two-sided sign test to reach p < 0.05 (computed). Report exact binomial
     intervals; for example, 3 misses in 30 has a 90% upper bound of 21%.
   - Use the labels with prediction-powered inference to correct each arm's
     overall score estimate for its bias on the full set, with honest intervals.
     In AutoEval Done Right, this raised the effective number of human labels by
     up to 50% in GPT-4 experiments [42].
   - What 20–40 labels cannot do: certify an agreement guarantee, which used 500
     labels in Trust or Escalate [41]; or fit learned weights for combining
     answers, as RocketEval's trained combiner needed labeled data [20].
6. How to choose:
   - For each step, use the smallest model whose planted-question accuracy is
     within 5 points of the large model's, and whose repeat agreement is within
     the large model's repeat noise.
   - For that model, use the largest bundle size where it stays within 5 points
     of its own 1k score.
   - Adopt **stitched+escalation** for threads above the large model's measured
     budget, such as session B's orchestrator, only if all three hold:
     - its planted detection is within 5 points of **large**;
     - its anchor agreement with **large** is within the large arm's repeat noise,
       on threads both can read;
     - the user's labels do not significantly favor **large**.

     Otherwise keep **large** for whole threads, and use small models only for
     step labels and screening.
   - Sample sizes. 605 subagent threads detect differences of a few points;
     30 human labels detect only large ones.

## Observations outside the question

Each could be its own small fix, separate from this research.

1. Confirmed by comparing code with W&B's list. W&B lists Granite 4.2 8B, Gemma
   4 26B A4B, Nemotron 3.5 Lightning, Qwen3.8 27B, and DeepSeek V4.1-Flash [1];
   none of them is in the catalog. Two catalog entries, `granite-4.1-8b` and
   `Qwen3.6-27B`, are not on W&B's list (`catalogs.py:90`, `catalogs.py:110-116`).
   This adds specific small models to the catalog drift that the long-session
   study found (`docs/long-session-judging-research.md:680-687`).
2. Confirmed in code; the impact is **Inference** and needs measurement. Models
   at or below 200,000 tokens get a 50,000-token raw window target
   (`run_config.py:53`). That is about 50 times Llama 3.1 8B's NoLiMa effective
   length [4], and Llama 3.1 8B is in the default panel (`catalogs.py:358`).
3. Confirmed from the price list. The default panel's Llama 3.1 8B costs more
   per token than its GPT OSS 120B ($0.22 against $0.03 input per 1M) [2].

## Open questions

1. What should "small" optimize: dollars, self-hosting or privacy, throughput, or
   independence from large vendors? On W&B, small dense models are not the
   cheapest.
2. Is a 27B–70B verdict writer acceptable, with 8B-class models limited to step
   labels and questions?
3. Should code write the questions from fixed templates per rubric, or should a
   larger model write per-session questions, as in RocketEval [20]?
4. Does W&B return token log-probabilities for these models? RocketEval's soft
   yes/no score needs them. Without them, use two model families or repeated
   samples.
5. Two small judges per question, which doubles question requests, or one judge
   sampled several times?
6. Should per-turn tallies become CES Scores?
7. What escalation budget per session, and which large model?
8. Would the user later train a small judge on the large model's answers for
   Monopole's own sessions? Trained small models win in-domain [17][45][46], but
   benchmark-trained critics failed on production data [7].
9. Should a small judge's "not in this evidence" answer count as an abstention,
   or always escalate?
10. Codex edits arrive inside `exec` shell commands (1,428 of 1,458 calls in
    session A, `docs/large-session-anatomy.md:55`). Should the extractor parse
    patches out of shell commands?

## Sources

All accessed 2026-11-03.

Vendor documentation, leaderboards, and code:

1. CoreWeave, Serverless Inference available models (redirected from docs.wandb.ai). https://docs.wandb.ai/guides/inference/models/
2. CoreWeave, Forge Inference pricing. https://coreweave.com/forge-pricing/inference-agents
3. NVIDIA, RULER repository README, effective-length table. https://github.com/NVIDIA/RULER
4. Adobe Research, NoLiMa repository README, effective-length table. https://github.com/adobe-research/NoLiMa
5. LLM-AggreFact leaderboard. https://llm-aggrefact.github.io/
6. judgeman repository README. https://github.com/Christopher4113/JudgeMan
7. OpenHands, Learning to Verify AI-Generated Code (2026-03-05). https://www.openhands.dev/blog/20260305-learning-to-verify-ai-generated-code

Papers and reports:

8. Kim et al., Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models. https://arxiv.org/abs/2405.01535
9. JudgeLM: Fine-tuned Large Language Models are Scalable Judges. https://arxiv.org/abs/2310.17631
10. Generative Judge for Evaluating Alignment (Auto-J). https://arxiv.org/abs/2310.05470
11. Foundational Autoraters: Taming Large Language Models for Better Automatic Evaluation (FLAMe). https://arxiv.org/abs/2407.10817
12. Tan et al., JudgeBench: A Benchmark for Evaluating LLM-based Judges. https://arxiv.org/abs/2410.12784
13. An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Model is not a General Substitute for GPT-4. https://arxiv.org/abs/2403.02839
14. Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges. https://arxiv.org/abs/2406.12624
15. Atla Selene Mini: A General Purpose Evaluation Model. https://arxiv.org/abs/2501.17195
16. J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning. https://arxiv.org/abs/2505.10320
17. Tang et al., LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling. https://arxiv.org/abs/2510.06915
18. Cook et al., TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and Generation. https://arxiv.org/abs/2410.03608
19. Lee et al., CheckEval: A reliable LLM-as-a-Judge framework for evaluating text generation using checklists. https://arxiv.org/abs/2403.18771
20. Wei et al., RocketEval: Efficient Automated LLM Evaluation via Grading Checklist. https://arxiv.org/abs/2503.05142
21. Branch-Solve-Merge Improves Large Language Model Evaluation and Generation. https://arxiv.org/abs/2310.15123
22. InFoBench: Evaluating Instruction Following Ability in Large Language Models. https://arxiv.org/abs/2401.03601
23. Zhang et al., Chain of Agents: Large Language Models Collaborating on Long-Context Tasks. https://arxiv.org/abs/2406.02818
24. Zhao et al., LongAgent: Scaling Language Models to 128k Context through Multi-Agent Collaboration. https://arxiv.org/abs/2402.11550
25. Lee et al., A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts (ReadAgent). https://arxiv.org/abs/2402.09727
26. MemGPT: Towards LLMs as Operating Systems. https://arxiv.org/abs/2310.08560
27. Sarthi et al., RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. https://arxiv.org/abs/2401.18059
28. GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models. https://arxiv.org/abs/2406.14550
29. Graph of Agents: Principled Long Context Modeling by Emergent Multi-Agent Collaboration. https://arxiv.org/abs/2509.21848
30. Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning. https://arxiv.org/abs/2509.06436
31. Yu et al., MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent. https://arxiv.org/abs/2507.02259
32. Zhou et al., LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models. https://arxiv.org/abs/2410.09342
33. Min et al., FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. https://arxiv.org/abs/2305.14251
34. Long-form factuality in large language models (SAFE). https://arxiv.org/abs/2403.18802
35. MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. https://arxiv.org/abs/2404.10774
36. VeriScore: Evaluating the factuality of verifiable claims in long-form text generation. https://arxiv.org/abs/2406.19276
37. Towards Effective Extraction and Evaluation of Factual Claims (Claimify). https://arxiv.org/abs/2502.10855
38. RefChecker: Reference-based Fine-grained Hallucination Checker and Benchmark for Large Language Models. https://arxiv.org/abs/2405.14486
39. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. https://arxiv.org/abs/2305.05176
40. RouteLLM: Learning to Route LLMs with Preference Data. https://arxiv.org/abs/2406.18665
41. Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement. https://arxiv.org/abs/2407.18370
42. AutoEval Done Right: Using Synthetic Data for Model Evaluation. https://arxiv.org/abs/2403.07008
43. R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents. https://arxiv.org/abs/2504.07164
44. AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories. https://arxiv.org/abs/2504.08942
45. An Illusion of Progress? Assessing the Current State of Web Agents (Online-Mind2Web, WebJudge). https://arxiv.org/abs/2504.01382
46. Web-Shepherd: Advancing PRMs for Reinforcing Web Agents. https://arxiv.org/abs/2505.15277
47. Deshpande et al., TRAIL: Trace Reasoning and Agentic Issue Localization. https://arxiv.org/abs/2505.08638
48. Counsel: A Meta-Evaluation Dataset for Agentic Tasks. https://arxiv.org/abs/2606.21627
49. Lightweight, Rubric-Guided Trajectory Evaluation for Production AI Agents (LiteTrajEval). https://arxiv.org/abs/2610.03315
50. Lam, How Far Can You Trust a Step-Level Judge? Layered judging for coding agents at a third of the cost (working paper, 2026-10-14). https://doi.org/10.5281/zenodo.23354648

Sources cited through the companion documents, listed for convenience:

- Kale et al., Reliable Weak-to-Strong Monitoring of LLM Agents. https://arxiv.org/abs/2508.19461
- Verga et al., Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL). https://arxiv.org/abs/2404.18796
