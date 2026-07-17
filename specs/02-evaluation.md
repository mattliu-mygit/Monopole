# Evaluation

Monopole combines deterministic evaluation with model judgment. Deterministic
scorers cover evidence that can be interpreted with high precision. Model
judges cover process and session qualities that require contextual assessment.
Both write ratings on the same higher-is-better `[0, 1]` scale.

Ordinary evaluation applies only to complete `agent_session` conversations.
Signal scorer, judge, reflection, other-system, and mixed-role traces are
operational evidence, not members of the agent-quality population. Discovery
enforces that boundary before child hydration, inference, cohort pinning, or
feedback writes. Rehydrating a pinned run also rejects role drift before
external work.

## Deterministic evaluation

Outcome evaluation recognizes observed test, build, lint, installation,
mutating Git, and definite generic-command failures. Explicit numeric exit codes
are authoritative. An explicit null exit code is unknown; framework output or
span status is used only when it can establish an outcome safely. Unknown
evidence produces no invented success or failure.

Process evaluation produces:

- turn and session efficiency, based on observable repeated failing attempts and
  repeated reads without intervening edits;
- session correction-free rate, the complement of the fraction of turns with a
  steering or denial event; and
- session completion, the complement of a conservative abandonment signal.

Raw correction density, abandonment, token counts, duration, tool counts, and
model tier remain explanatory context rather than negative-direction ratings.

Deterministic parsers favor precision over coverage. They inspect only retained
tool evidence and do not claim intent, semantic code quality, test quality, or
causal attribution for all work that led to an observed result.

## Planned model evaluation

Before inference, an evaluation run derives and pins a versioned judging plan
from its immutable cohort. The plan pins the context policy, ordered model
descriptors including context capacity and token-counter identity, rubric
descriptors, complete session trace identities, each reviewer's planned-or-
skipped disposition, reviewer-specific window manifests, and the exact
digest/window/merge protocol. Changes to any of those inputs produce a different
plan or artifact identity. Authentication recomputes each reviewer's
applicability and exact window plan from the pinned session, descriptor, and
policy before external work.

All model rubrics are session-level. Each applicable reviewer gets a window plan
sized to its pinned context capacity. Models above 200,000 input tokens reserve
at least 100,000 tokens; models at or below 200,000 reserve at least 50,000.
Prompt, output, safety, surrounding-digest, and finding overhead may increase
that reserve. Raw windows have separate soft targets of 128,000 tokens above
the same model threshold and 50,000 tokens at or below it. OpenAI catalog
entries use their explicitly pinned `tiktoken` encoding; other model families
use the conservative UTF-8 byte estimator.

The hard raw budget remains a capacity ceiling. Contiguous core chunks are
packed near the soft target, assigning a boundary-crossing turn to the side
closer to that target, so every session turn has exact core coverage. Each raw
window includes up to one neighboring turn on either side when it remains
within the soft target; overlap can shrink to zero without removing core
coverage. Turn boundaries remain indivisible, and a single oversized turn may
exceed the soft target only when it still fits the hard budget. Before planning,
the judge view deterministically compacts only oversized tool arguments and
results, preserving their head, diagnostic excerpts, tail, original character
count, and content hash. Captured user and assistant messages are not compacted. When an intact
projected turn, the required chunk count, or the worst-case merge cannot fit,
that reviewer alone is skipped for the session with
`insufficient_context_capacity`.

For each reviewer, the pipeline first creates one rubric-neutral factual digest
per core chunk. A rubric evaluation then slides across every window: the active
chunk is supplied as raw captured evidence while the rest of the session is
represented by that reviewer's ordered digests. Every window returns a bounded
set of positive or negative findings, not a score. Findings are deduplicated in
session order, then one final merge sees the coverage manifest, ordered digests,
and findings and returns the reviewer's anchored session verdict and behavioral
feedback.

Raw rendering includes captured user, assistant, tool, event, status, token, and
trace-identity evidence while omitting the generating model's identity from the
judge-facing content. Parent-session judging retains each subagent's identity
and internal tool-call count but omits its internal tool payloads; the parent's
delegation call still carries the input and returned output that influenced the
parent. The complete hydrated trace remains available outside this judge view.
Digests, findings, verdicts, and behavioral feedback are schema-bounded. Each
inference schema binds the expected chunk or window identity and enumerates the
exact evidence identities authorized for that phase; the same constraints are
validated again after inference. Complete
window-finding artifacts receive a 4,000-token budget, and each final behavioral
feedback field is limited to 10,000 characters.

The current session rubrics evaluate verification, error recovery, tool choice,
state consistency, outcome, and autonomy. A rubric may return
`insufficient_evidence` when its required opportunity is absent even though the
session itself has complete raw coverage. That abstention remains distinct from
missing, malformed, or unauthenticated pipeline evidence.

## Verdict contract

Every reviewer's final merge returns the closed versioned verdict contract:

- status is `scored` or `insufficient_evidence`;
- a scored verdict uses exactly one anchor from
  `0, 0.25, 0.5, 0.75, 1`;
- an insufficient-evidence verdict has no score;
- rationale is required;
- a scored verdict cites at least one evidence ID from the complete session and
  includes at least one nonblank success, problem, or desired-behavior field;
  and
- an insufficient-evidence verdict has no citations or behavioral feedback.

Chunk digests must cite their exact core evidence. Window findings cite only
evidence visible in that raw window, are capped in count and size, and carry
finding identities unique within that window. Aggregation scopes each identity
to its authenticated window before deduplicating exact semantic findings in
session order. An unknown citation, blank required text, duplicate semantic
finding within one window, or oversized artifact fails closed. The merge prompt
asks for feedback about what the agent did or should do, while reflection
separately decides whether and how to edit managed instructions. Imperfect
semantic wording is not itself a validation failure.

Boolean, missing, nonnumeric, non-finite, out-of-range, non-anchor, malformed,
or improperly cited output is rejected. Scores are never clamped or coerced.
Abstention remains distinct from transport or validation failure. Successful
reviewer anchors are mean-pooled, so a final multi-reviewer rating may be any
finite value in `[0, 1]`.

Structured output is requested when the backend supports it. A recorded JSON
object fallback is allowed only when the provider explicitly rejects structured
schema capability. Invalid model content does not trigger a looser parsing
mode. Transient transport failures and provider exhaustion while producing
schema-valid output may retry the same strict request; retries never remove its
identity or evidence constraints. Backend-specific schema emission may omit a keyword that the backend
rejects when the same invariant remains enforced by the canonical runtime
validator; the canonical schema and artifact contract are not weakened.

## Judge panel

Runs use guided model catalogs with recommended selections, but the saved
ordered choices are explicit and user-overridable. Start-time validation pins
the model and rubric catalog versions, exact descriptors, rubric thresholds,
the ordered panel, and visible family-overlap warnings. A panel contains one,
two, or three unique judges. The runtime honors the chosen order, plans or skips
each selected judge per session, and never silently replaces one.

One versioned catalog is authoritative for proposal writers, run judges, B/C
verification judges, and proposal evaluators. Each globally unique model ID is
provider-qualified, while its descriptor separately pins the provider's exact
model name. Role capability is part of the descriptor. A panel may therefore
mix providers without a separate backend selection, and execution dispatches
each selected model through its pinned provider.

Only planned reviewers call inference. A capacity skip remains in the judging
plan and attempt audit, but does not count as a completed reviewer attempt or an
inference failure. The one through three selected panel positions run
concurrently for each rubric, so concurrency is bounded by the pinned panel
size; rubrics and sessions remain sequential. A scored rubric is complete when
every selected reviewer returns a valid score. It is `degraded` when at least
one reviewer scores and the rest either validly abstain or were skipped; its
rating is the arithmetic mean of available scores, with minimum, maximum,
spread, abstentions, and skips retained for audit. When no reviewer is
applicable, or every applicable reviewer validly abstains, the rubric is not
evaluable and writes no feedback.

A failed invocation or invalid attempt by an applicable reviewer still fails
coverage even when another reviewer returned a valid score. After the bounded
transport retries for that attempt are exhausted, the runtime cancels
outstanding panel work when the transport supports active cancellation, waits
for started work to settle safely, and stops the remaining rubrics and sessions.
An already-started request may complete before cancellation takes effect.

Every attempt retains its requested and resolved model, outcome, rationale or
safe error, evidence citations, structured-output mode, usage, bounded
behavioral feedback, and ordered digest/window/merge step audit. Scores are
buffered until every session rubric resolves to a complete or degraded rating,
or a unanimous not-evaluable outcome. Only the resulting ratings are written
under the run's cancellation barrier.

## Artifact resume and failure behavior

Successful digest, window, and merge outputs are persisted as immutable,
content-authenticated artifacts. Their identities bind the exact window plan,
reviewer, protocol content, rubric where relevant, phase, and source identity.
On restart, matching artifacts are validated again and reused; an artifact with
the wrong envelope, digest, model, phase, schema, or request provenance fails
closed instead of being treated as compatible work.

Progress counts unique persisted artifacts, so reuse cannot inflate completed
work. Concurrent panel activity and artifact completion are serialized before
persistence. Partial model outputs never become scores, and cancelled or
unattempted work after a fail-fast stop is not reported as completed. Any
planned rubric failure prevents all judge feedback writes for that run;
successful scores are written only after complete coverage is known. Forced
replacement creates new feedback before deleting prior matches, and write or
cleanup failures remain visible.

## Inference trust boundary

Catalog-backed HTTP evaluation supports direct OpenAI and W&B Inference. Local
evaluation supports Claude, Codex, and Antigravity (`agy`) from the same
provider-qualified catalog. Every local provider receives a minimal child
environment and an empty working directory. Claude is tool-disabled. Codex
uses isolated state and a deny-by-default filesystem profile; its
repository-check override only removes a CLI precondition. Antigravity runs in
non-mutating plan mode with its sandbox enabled and uses the existing user home
only for Antigravity/Gemini authentication and state. These controls do not
weaken evidence pinning or permit arbitrary parent secrets and proxy variables
to enter the child environment.

Prompts, raw model output, credentials, and arbitrary parent environment values
are not written to application logs or run errors. Audit records retain safe
structured outcomes and content digests instead.

## Research basis and limits

The design is informed by:

- [PoLL](https://arxiv.org/abs/2404.18796), for heterogeneous judges and score
  pooling;
- [MT-Bench judge analysis](https://arxiv.org/abs/2306.05685), for
  conversation context and boundary sensitivity;
- [self-recognition and preference](https://arxiv.org/abs/2404.13076), for
  hiding generator identity from judge-facing evidence;
- [Agent-as-a-Judge](https://arxiv.org/abs/2410.10934), for inspectable
  trajectory evidence;
- [PReMISE](https://arxiv.org/abs/2605.30803), for rubric-specific abstention;
  and
- [RuVerBench](https://arxiv.org/abs/2606.29920), for the limits and diminishing
  returns of agentic coding judges.

These papers motivate the architecture but do not calibrate this product. There
is no human-labeled validation set, and neither the three-judge cap nor the
context-budget policy is claimed as an empirical optimum. These are product
cost bounds, not claims that larger contexts or another correlated judge could
never add information.
