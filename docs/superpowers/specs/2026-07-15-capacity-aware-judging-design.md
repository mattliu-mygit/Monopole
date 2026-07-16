# Capacity-Aware Judging

## Goal

Let every selected judge evaluate complete session evidence when its pinned
context capacity permits, while preserving incapable reviewers as auditable
skips instead of failing the run or silently removing evidence.

## Capacity and counting

The model descriptor is the authority for context capacity and token-counter
identity, and the effective run configuration pins both. OpenAI catalog entries
use explicitly selected `tiktoken` encodings: GPT-5 and GPT-4o use
`o200k_base`, while GPT-OSS uses `o200k_harmony`. Other model families use the
conservative UTF-8 byte estimator to avoid remote tokenizer calls, model
downloads, and a broader tokenizer abstraction.

Models above 200,000 input tokens reserve at least 100,000 tokens. Models at or
below 200,000 reserve at least 50,000. Prompt, output, safety, digest, and
finding overhead may increase this reserve. Raw-window and merge inputs are
bounded independently against the model's capacity.

Provider-reported token usage remains accounting data rather than a planning
input because it describes a different payload or arrives after inference.

## Evidence and applicability

Window planning is reviewer-specific. Contiguous core chunks retain complete
raw-turn coverage and may add neighboring whole turns as overlap when they fit.
An individual captured raw turn is never split, truncated, deduplicated, or
summarized.

Each selected reviewer is recorded as planned or skipped for each session. A
reviewer is skipped only as `insufficient_context_capacity` when an intact turn,
the bounded chunk plan, or the bounded merge cannot fit. Other planning errors
remain failures. Skipped reviewers perform no inference, do not count as failed
or completed reviewer attempts, and retain zero model-work bounds.

Applicable reviewers continue in selected order. A score based on fewer than
all selected reviewers is degraded. If no reviewer is applicable, the session's
rubrics are not evaluable, no feedback is written, and the run continues.

## Authentication and downstream use

The content-authenticated judging plan binds the exact session, ordered model
descriptors, context policy, token counters, reviewer dispositions, window
plans, work bounds, and protocol. Before external work, authentication
recomputes reviewer applicability and exact window plans from those pinned
inputs; a forged or stale planned-or-skipped disposition fails closed.

Attempt summaries retain capacity skips for audit while progress counts only
applicable reviewer work. Analysis and reflection exclude not-evaluable
judgments, so skipped coverage cannot be mistaken for model feedback.
