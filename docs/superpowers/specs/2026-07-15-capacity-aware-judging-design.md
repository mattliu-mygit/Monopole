# Capacity-Aware Judging

## Goal

Let each selected judge evaluate as much complete session evidence as its real
context window safely supports, while skipping incapable reviewers instead of
failing the run or silently removing unique evidence.

## Current problem

Judging currently caps every model at 128,000 input tokens and then applies a
fixed 100,000-token target. Those catalog values are stale: GPT-5.6 Sol has a
1.05M context window and Claude Sonnet 5 has a 1M context window, while Claude
Haiku 4.5 and the current open models remain around 128K-200K. A failed cohort
also contains individual rendered turns as large as 2.45 MB because root turns
include nested subagent tool results.

The current UTF-8-bytes-divided-by-three estimator is conservative but not a
model token count. Tokenization differs by family and model version. Correct
capacity routing is therefore the primary fix; evidence compression and
deduplication are explicitly deferred.

## Context policy

Model context capacity, pinned in the model catalog and effective run
configuration, is the authoritative upper bound. The fixed
`target_input_tokens` ceiling is removed.

- Models with more than 200,000 input tokens reserve a minimum 100,000-token
  buffer.
- Models with 200,000 input tokens or fewer reserve a minimum 50,000-token
  buffer.
- The buffer covers prompts, outputs, safety margin, surrounding digests, and
  findings. If calculated protocol overhead exceeds the tier buffer, the
  calculated overhead wins.
- A reviewer's raw budget is its model capacity minus the greater of its tier
  buffer and its calculated protocol overhead.
- The active raw chunk replaces its own digest, so a window budgets only the
  other `chunk_count - 1` digests.
- Merge input remains independently bounded using every digest and every
  bounded window finding.

The initial catalog correction sets GPT-5.6 Sol to 1,050,000 tokens, Claude
Sonnet 5 to 1,000,000, Claude Haiku 4.5 to 200,000, and keeps model-specific
published limits for the remaining models.

## Reviewer applicability

Window planning remains reviewer-specific. A reviewer is applicable to a
session only when every indivisible turn fits its raw budget and the complete
sliding-window and merge plans fit its context capacity.

An inapplicable reviewer is recorded as skipped with a stable reason such as
`insufficient_context_capacity`; it is not a failed inference attempt. Other
eligible reviewers continue. If no selected reviewer is applicable, that
session is recorded as unevaluable and the run continues. Progress, artifacts,
results, analysis, and reflection distinguish skipped coverage from failed or
missing work.

Judges never receive incomplete evidence while claiming a whole-session
verdict. This version does not truncate, deduplicate, split, or otherwise
compress an indivisible turn.

## Token counting strategy

Token counting is an explicit model-catalog capability. The effective
configuration pins the counter used for every reviewer.

The first implementation adds OpenAI's `tiktoken` dependency:

- GPT-5 and GPT-4o-family models use the explicitly pinned `o200k_base`
  encoding.
- GPT-OSS models use the explicitly pinned `o200k_harmony` encoding.
- Other models use the existing conservative UTF-8 byte estimator.

Claude, Llama, and Granite keep the conservative estimator because adding API
calls, model downloads, or a broad tokenizer abstraction would add operational
complexity without solving the immediate capacity-routing problem. The
100K/50K buffers protect against request-format overhead and estimation error.

Existing provider-reported token usage from original Weave traces and completed
judge calls remains ordinary accounting data. It is not used for planning
because it either describes a different payload or arrives after the request.
No calibration subsystem is added.

## Persistence and observability

The judging plan and retained artifacts expose:

- model context capacity and buffer tier;
- token counter identity;
- estimated raw and overhead tokens;
- per-reviewer applicability or skip reason;
- per-session reviewer coverage used by analysis and reflection.

Existing plan and artifact hashes bind these new inputs so changes produce new
identities.

## Verification

Tests cover capacity-tier boundaries, corrected catalog limits, the two token
counting paths, reviewer skipping, zero-applicable-reviewer sessions, progress
totals, and reflection exclusion of unevaluated evidence.

A fixture based on the observed oversized shape must show that GPT-5.6 Sol and
Claude Sonnet 5 can plan the roughly 816K estimated turn under their large-model
buffers, while 128K-200K reviewers are skipped without failing the run.

## Research basis

- [OpenAI model catalog](https://developers.openai.com/api/docs/models) lists
  GPT-5.6 Sol with a 1.05M context window.
- [Claude Sonnet 5 guidance](https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5)
  lists a 1M context window and its tokenizer change.
- [OpenAI token counting](https://developers.openai.com/api/docs/guides/token-counting)
  and [Claude token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)
  explain why request-aware model-specific counts are preferable to character
  heuristics when available.
- [`tiktoken`](https://github.com/openai/tiktoken) maps GPT-5 to `o200k_base`
  and GPT-OSS to `o200k_harmony`.
- [RULER](https://arxiv.org/abs/2404.06654) and
  [Lost in the Middle](https://aclanthology.org/anthology-files/pdf/tacl/2024.tacl-1.9.pdf)
  support retaining substantial headroom instead of treating advertised
  context size as uniformly effective capacity.
