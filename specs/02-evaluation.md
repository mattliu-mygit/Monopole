# Evaluation

Monopole combines deterministic evaluation with model judgment. Deterministic
scorers cover evidence that can be interpreted with high precision. Model
judges cover process and session qualities that require contextual assessment.
Both write ratings on the same higher-is-better `[0, 1]` scale.

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
from its immutable cohort. The plan pins the context policy, ordered judges,
review policy, rubric descriptors, complete session trace identities,
reviewer-specific window manifests, and exact digest/window/merge protocol.
Changes to any of those inputs produce a different plan or artifact identity.

All model rubrics are session-level. Each reviewer gets a window plan sized to
the smaller of the configured target and that model's context limit. Reserved
space for prompts, outputs, safety, surrounding digests, and findings leaves a
bounded raw budget. Contiguous core chunks partition every session turn exactly
once; each raw window also includes one available neighboring turn on either
side for continuity. A single turn that cannot fit, too many required chunks,
or a worst-case merge that exceeds the input cap fails planning before model
work begins.

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
judge-facing content. Digests, findings, verdicts, and behavioral feedback are
bounded and may cite only evidence identities authorized for their phase.

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
stable finding identities. A reused finding identity with conflicting content,
unknown citation, blank required text, duplicate semantic finding, or oversized
artifact fails closed. Desired behavior must describe how the agent should act;
it cannot prescribe edits to managed instruction files.

Boolean, missing, nonnumeric, non-finite, out-of-range, non-anchor, malformed,
or improperly cited output is rejected. Scores are never clamped or coerced.
Abstention remains distinct from transport or validation failure. Successful
reviewer anchors are mean-pooled, so a final multi-reviewer rating may be any
finite value in `[0, 1]`.

Structured output is requested when the backend supports it. A recorded JSON
object fallback is allowed only when the provider explicitly rejects structured
schema capability. Invalid model content does not trigger a looser parsing
mode.

## Judge choice and review depth

Runs use guided model catalogs with recommended selections, but the saved
ordered choices are explicit and user-overridable. Start-time validation pins
the model and rubric catalog versions, exact descriptors, rubric thresholds,
review policy, and visible family-overlap warnings. The runtime honors the
chosen order and never silently replaces a judge.

Review depth applies independently to every session rubric:

- primary uses one judge;
- selective uses two or three configured judges and normally stops after the
  first clear result; and
- full panel uses all three judges on every rubric.

Selective review calls Judge 2 when Judge 1 fails, abstains, or scores within the
configured margin of the rubric boundary. With a third configured judge, it may
continue after threshold disagreement, a lone near-boundary success, or two
unsuccessful attempts. The default margin is an explicit cost/latency heuristic,
not calibrated confidence.

All successful scores are mean-pooled. A result is:

- complete when every attempted reviewer succeeded and no two-way threshold
  split remains;
- degraded when a rating survives a failed or abstained attempt;
- unresolved when exactly two successful reviewers split across the threshold
  without a successful third opinion; or
- failed when no reviewer supplies a valid score.

Every attempt retains its trigger, requested and resolved model, outcome,
rationale or safe error, evidence citations, structured-output mode, usage,
bounded behavioral feedback, and ordered digest/window/merge step audit. A
rubric with zero successful reviewers fails stage coverage. Scores are buffered
until every planned session rubric has an accepted outcome, then written under
the run's cancellation barrier.

## Artifact resume and failure behavior

Successful digest, window, and merge outputs are persisted as immutable,
content-authenticated artifacts. Their identities bind the exact window plan,
reviewer, protocol content, rubric where relevant, phase, and source identity.
On restart, matching artifacts are validated again and reused; an artifact with
the wrong envelope, digest, model, phase, schema, or request provenance fails
closed instead of being treated as compatible work.

Progress counts unique persisted artifacts, so reuse cannot inflate completed
work. Partial model outputs never become scores. Any planned rubric failure
prevents all judge feedback writes for that run; successful scores are written
only after complete coverage is known. Forced replacement creates new feedback
before deleting prior matches, and write or cleanup failures remain visible.

## Inference trust boundary

Catalog-backed HTTP evaluation supports direct OpenAI and W&B Inference. Local
Claude and Codex evaluation runs with tools and customization disabled, a
minimal child environment, isolated state, and bounded model-readable
filesystem access. The Codex repository-check override only removes its CLI
precondition; it does not weaken the sandbox or evidence pinning. Local model
families without a verified equivalent confinement mode remain disabled.

Prompts, raw model output, credentials, and arbitrary parent environment values
are not written to application logs or run errors. Audit records retain safe
structured outcomes and content digests instead.

## Research basis and limits

The design is informed by:

- [PoLL](https://arxiv.org/abs/2404.18796), for heterogeneous judges and score
  pooling;
- [Trust or Escalate](https://arxiv.org/abs/2407.18370), for selective review
  cascades;
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
is no human-labeled validation set, and neither the selective margin nor the
context-budget policy is claimed as an empirical optimum.
