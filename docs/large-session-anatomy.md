# Anatomy of the largest recorded sessions

Measured 2026-11-03 from prod CES, read-only. This is Part 1 of the large-session study: what
makes the biggest sessions big. Ways of handling them are out of scope. Companion studies:
[long-session-judging-research.md](long-session-judging-research.md) and
[windowed-judging-research.md](windowed-judging-research.md).

## Summary

- **Tool output is most of every session:** tool results are 50–88% of each session's content,
  and tool arguments another 10–37%. Messages are small.
- **Showing every message barely changes size:** rendering every message instead of each turn's
  first and last adds 0.5–11%.
- **Fixed compression rules help modestly:** plain-text output, dropping counters, and pointers
  to repeated reads save 5–20%. Only shorter tool-output caps cut much: a 4k cap saves 42% on
  the biggest session.
- **The Workflow session is about 30M tokens once subagents' own work is included:** its
  main thread alone is 1.9M. Every one of its 605 subagents fits a 254k-token budget on its own
  (the largest is 161k). The orchestrator's view doesn't fit 1M, even compressed.
- **Compactions split each main thread into stretches of 520k tokens or less:** compactions are
  where the agent summarized its own context. The agent itself never held more than 933k tokens
  in one model call. A 1M-token judge can read every stretch whole, which is close to what the
  agent itself saw.
- **Repetition is mostly across subagents:** in the Workflow session, 73% of tool-result lines
  had already appeared earlier in the session, mostly because subagents re-read the same files
  and outputs. Within one thread, repetition is small, except for repeated polling commands in
  one session (26%).

## Method

- **Sessions:** the four largest sessions of the last 30 days. Each was read whole: every Case,
  and every subagent Trajectory.
- **Sizes:** in tokens from the renderer's counter (UTF-8 bytes ÷ 3). Per the first study, this
  undercounts Claude 4.7+ by about 17%.
- **Budgets:** one request's budget is the context size minus about 18k reserved for prompt,
  output and safety: 113,072 for 131k models, 254,000 for 272k, 982,000 for 1M.
- **Rendering variants:**

| Variant | Meaning |
| --- | --- |
| current | today's renderer: each turn's first user message and last assistant message, tool calls, one line per subagent |
| every message | every message and step of the turn, tool text cut at 12,000 characters as today |
| compressed | every message, plus plain-text tool output, no per-call token counters, and pointers for repeated identical results |
| cap 4k / cap 2k | compressed, with tool text cut at 4,000 or 2,000 characters (head plus tail) |
| no cap | every message, tool text uncut |

## The sessions

| | A | B | C | D |
| --- | --- | --- | --- | --- |
| Harness | Codex | Claude Code (Workflow) | Claude Code | Claude Code |
| Main-thread turns | 63 | 11 | 6 | 4 |
| Subagent threads | 0 | 605 (depth 1) | 0 | 0 |
| Cases | 2,008 | 3,188 main, 32,870 in subagents | 1,360 | 709 |
| Tool calls | 1,458 (exec 1,428) | 18,182 (Bash 14,926, Read 1,903) | 678 (Bash 560) | 377 (Bash 212) |
| Compactions | 17 | 8 (6 auto, 2 manual) | 2 manual | 1 manual |

## Where the tokens are

Shares of raw content (bytes ÷ 3, tool text as the adapter stored it):

| Content | A | B | C | D |
| --- | --- | --- | --- | --- |
| Tool results | 88.1% | 75.6% (72.2% in subagents) | 65.5% | 49.9% |
| Tool arguments | 9.7% | 14.5% (12.6% in subagents) | 25.4% | 36.7% |
| Assistant messages (all) | 2.1% | 3.8% | 9.1% | 13.1% |
| User messages | 0.1% | 6.0% (subagent task prompts) | 0.1% | 0.3% |
| Raw total | 3.53M | 36.8M | 0.61M | 0.29M |

The largest tool results come from shell commands (`exec`, `Bash`) in every session. In B, file
reads inside subagents add another 32%.

## Sizes by rendering

Whole main thread. For B, "+ subagents" also counts every subagent thread.

| Variant | A | B main | B + subagents | C | D |
| --- | --- | --- | --- | --- | --- |
| current | 2.23M | 1.80M | (one line per subagent) | 650k | 314k |
| every message | 2.25M | 1.92M | 30.4M | 701k | 350k |
| compressed | 2.13M | 1.68M | 28.7M | 562k | 325k |
| cap 4k | 1.30M | 1.28M | 20.2M | 484k | 280k |
| cap 2k | 0.98M | 1.05M | 15.8M | 409k | 243k |
| no cap | 3.81M | 2.41M | 40.7M | 726k | 353k |

Against the 1M budget (982,000): C and D fit whole in every variant. A fits only with the 2k
cap, and then by 0.5%. B's main thread doesn't fit in any variant.

## Turns and stretches between compactions

| | A | B | C | D |
| --- | --- | --- | --- | --- |
| Largest turn (every message) | 200k | 1.07M | 361k | 312k |
| Turns over 113k / 254k / 982k | 2 / 0 / 0 | 3 / 3 / 1 | 2 / 2 / 0 | 1 / 1 / 0 |
| Stretches between compactions | 18 | 9 | 3 | 2 |
| Largest stretch (every message / compressed) | 219k / 210k | 520k / 357k | 361k / 338k | 312k / 290k |
| Largest context the agent held in one model call | 487k | 933k | 420k | 635k |

B's largest turn ran for 103 hours. It has 1,791 steps and 6 compactions, and launched 579 of
the 605 subagents.

## The Workflow session's subagents (B)

- **Launch:** 584 subagents were launched by `Workflow` and 21 by `Agent`. All are one level
  deep.
- **Size per subagent:**

| Variant | median | 90th percentile | largest | total |
| --- | --- | --- | --- | --- |
| every message | 26.5k | 100k | 161k | 28.5M |
| compressed | 25.5k | 94.7k | 150k | 27.0M |
| cap 4k | 17.6k | 66.4k | 122k | 18.9M |
| no cap | 35.6k | 134k | 198k | 38.3M |

- **Fit:** 27 subagents exceed 113k (17 when compressed). None exceeds 254k.
- **Shape:** a typical subagent has 2 user messages (the task and one follow-up) and 12 tool
  calls. The busiest has 133. Task prompts are 287 tokens at the median and up to 8.7k.
- **Orchestrator view** (main thread plus about 300 tokens of summary per subagent):

| Variant | Whole view | Largest turn plus its summaries |
| --- | --- | --- |
| every message | 2.10M | 1.25M |
| compressed | 1.86M | 1.19M |
| cap 2k | 1.23M | 0.80M |

## Repetition

Tool results that repeat exactly (same tool, arguments and result, at least 300 characters), and
lines already seen earlier:

| | A | B | C | D |
| --- | --- | --- | --- | --- |
| Exact repeats, share of result tokens | 0.2% | 0.8% | 26.4% | 0% |
| Repeated identical command output | 1 call | 121 calls | 130 calls | 0 |
| Unchanged file re-read | 0 | 19 calls | 1 call | 0 |
| Result lines already seen earlier in the same thread | 25.2% | main 27.7%, a subagent's own 1.4% | 37.4% | 6.8% |
| Result lines already seen anywhere earlier in the session | (same) | 73.2% | (same) | (same) |

In B, 4,938 tool results consist of at least 90% lines already seen elsewhere in the session.
They total about 9.7M tokens at the 12k cap. Most come from different subagents reading the same
files and command outputs.

## What this implies

All inferences, to be tested by the handling study.

- **Tool output:** how it is shown decides almost everything. Showing every message is cheap
  enough to do everywhere.
- **Compression:** the fixed rules are worth having but don't change what fits. Pointers for
  repeated results matter for C, which polls, and very little for the others.
- **Option D for B:** "each part fits one request" works for B's subagents only with 272k-token
  or larger judges. The 131k defaults would reject 27 of them.
- **B's orchestrator:** it can't be judged whole. Its stretches between compactions all fit a 1M
  budget (largest 520k), so compactions are natural window boundaries. A judge reading one
  stretch sees roughly what the agent held at that time.
- **A:** it fits whole only with a 2k tool-text cap, which cuts most command output. Its 18
  stretches between compactions (largest 219k) fit any 272k-token or larger model.
- **Whole-session view of B:** a session-wide pointer for results already shown elsewhere could
  remove up to about 9.7M tokens. Per-subagent reviews wouldn't gain from it, since each
  subagent's own repetition is only 1.4%.

## Caveats

- **Sample:** four sessions from one user over 30 days, so not representative.
- **Stored text:** the adapter stores already shortened tool text, so "no cap" is a floor, not
  the true raw size.
- **Summary size:** the orchestrator figures assume 300-token summaries per subagent, as the
  windowed study proposed.
- **Token counts:** counts come from the bytes ÷ 3 estimator; a model's own tokenizer differs.
