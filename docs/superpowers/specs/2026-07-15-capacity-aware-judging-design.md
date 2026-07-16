# Capacity-Aware Judging and Lossless Evidence Deduplication

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
model token count. Tokenization differs by family and model version; Claude
Sonnet 5, for example, uses a new tokenizer that produces approximately 30%
more tokens than its predecessor for the same input.

Exact duplicate tool results account for about 18% of tool-result bytes in the
observed failed cohort. Deduplication helps several large turns substantially,
but saves only about 14 KB from the single largest turn, so correct capacity
routing remains necessary.

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

Window planning remains reviewer-specific. After lossless deduplication, a
reviewer is applicable to a session only when every indivisible turn fits its
raw budget and the complete sliding-window and merge plans fit its context
capacity.

An inapplicable reviewer is recorded as skipped with a stable reason such as
`insufficient_context_capacity`; it is not a failed inference attempt. Other
eligible reviewers continue. If no selected reviewer is applicable, that
session is recorded as unevaluable and the run continues. Progress, artifacts,
results, analysis, and reflection distinguish skipped coverage from failed or
missing work.

Judges never receive incomplete unique evidence while claiming a whole-session
verdict.

## Lossless duplicate-result representation

Deduplication is scoped to one session and applies only to nonempty tool result
payloads with identical UTF-8 bytes. Semantic similarity is not sufficient.

The first occurrence in session order remains the canonical full result. A
later duplicate keeps its tool identity, name, arguments, status, timestamps,
and position, but replaces only the repeated result bytes with a deterministic
marker containing:

- the canonical evidence ID;
- the exact content SHA-256 digest;
- the original byte length.

This preserves the fact, location, outcome, and context of every repeated tool
call while presenting the identical result content raw once. Every canonical
result still receives raw coverage in one core window, and surrounding digests
plus final finding merge carry its relevance across the session.

No truncation, semantic deduplication, token pruning, or generated summary of
unique tool evidence is introduced in this version.

## Token counting strategy

Token counting is an explicit model-catalog capability, not inferred from
family names at runtime. The effective configuration and judging plan pin the
counter identity and version used for every reviewer.

The first implementation adds OpenAI's `tiktoken` dependency:

- GPT-5 and GPT-4o-family models use the explicitly pinned `o200k_base`
  encoding.
- GPT-OSS models use the explicitly pinned `o200k_harmony` encoding.
- Unknown OpenAI model IDs fail closed unless their catalog descriptor names an
  encoding; automatic fallback to a different encoding is not allowed.

Claude has no authoritative local tokenizer package. Its official token-count
endpoint is accurate for a selected model, but the local CLI backend does not
share API credentials and planning must remain deterministic and offline.
Claude therefore keeps the conservative UTF-8 byte estimator initially, with
its estimator identity pinned and the 100K/50K capacity buffer protecting
request-format overhead and estimation error.

Open-weight Llama and Granite tokenizers are available through Hugging Face,
but `transformers` plus runtime model downloads would add a large dependency
and nondeterministic network/cache behavior. They retain the conservative
fallback initially. A later change may pin tokenizer JSON assets and use the
smaller `tokenizers` runtime after measured benefit.

LiteLLM is not adopted. It provides broad tokenizer helpers, but unsupported
models can silently fall back to an OpenAI tokenizer and its model metadata is
community-maintained. That behavior is too permissive for a reproducible
evaluation plan.

Provider count-token endpoints remain useful for offline calibration tests.
Calibration compares representative rendered evidence against the pinned local
counter or byte estimator and verifies that the configured buffer covers the
observed error. It does not become an unpinned runtime planning dependency.

W&B provides two useful actual-usage signals, neither of which replaces
preflight counting. Hydrated Weave chat spans contain provider-reported token
usage for the original agent calls, and W&B Inference chat completions return
usage for a judge call after it finishes. The former describes a different
payload from the rendered judge window; the latter arrives too late to decide
whether that request fits. Post-call judge usage is therefore retained for
estimator calibration and monitoring, while the pinned local counter remains
authoritative for window planning. W&B Inference currently documents chat
completions and model listing, but no preflight token-count endpoint.

## Persistence and observability

The judging plan and retained artifacts expose:

- model context capacity and buffer tier;
- token counter identity and version;
- estimated raw and overhead tokens;
- duplicate-result canonical references and saved bytes;
- per-reviewer applicability or skip reason;
- per-session reviewer coverage used by analysis and reflection.

Existing plan and artifact hashes bind these new inputs so changes produce new
identities.

## Verification

Tests cover capacity-tier boundaries, corrected catalog limits, exact duplicate
matching, non-deduplication of near matches, canonical session ordering,
tamper-resistant duplicate markers, model-specific token counters, unknown
model failure, reviewer skipping, zero-applicable-reviewer sessions, progress
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
- [LiteLLM token usage documentation](https://docs.litellm.ai/docs/completion/token_usage)
  documents both its broad tokenizer support and unsupported-model fallback.
- [W&B Serverless Inference API](https://docs.wandb.ai/inference/api-reference)
  documents chat completions and model listing, with no preflight token-count
  method; judge response usage remains a post-call calibration signal.
- [RULER](https://arxiv.org/abs/2404.06654) and
  [Lost in the Middle](https://aclanthology.org/anthology-files/pdf/tacl/2024.tacl-1.9.pdf)
  support retaining substantial headroom instead of treating advertised
  context size as uniformly effective capacity.
- [Fundamental Limits of Prompt Compression](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ac8fbba029dadca99d6b8c3f913d3ed6-Abstract-Conference.html)
  supports deferring rubric-agnostic lossy compression because useful
  information depends on the downstream query.
