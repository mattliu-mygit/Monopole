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
from its immutable cohort. It bounds selected episodes per session, records why
each episode was chosen, names the exact evidence traces, and marks each
requested rubric as applicable or not applicable. Whole-session rubrics are
planned separately.

Selection targets high-information moments such as tool errors and recovery,
user corrections or denials, modifications, verification opportunities, and
the final substantive turn. A rubric with no valid opportunity is not
applicable and emits no score. Missing evidence for an applicable judgment is a
failure, not a neutral rating.

Judge-facing digests contain bounded user, assistant, tool, status, and
trace-identity evidence. They omit the generating model's identity to reduce
source-recognition bias. Cross-turn rubrics receive their pinned predecessor
evidence. Session digests cover the beginning, end, prioritized episodes, and
explicit included, omitted, and missing counts rather than pretending a
truncated prefix is complete.

The current rubrics evaluate verification, error recovery, tool choice, and
state consistency for selected episodes, plus outcome and autonomy for whole
sessions.

## Verdict contract

Every judge returns the closed versioned verdict contract:

- status is `scored` or `insufficient_evidence`;
- a scored verdict uses exactly one anchor from
  `0, 0.25, 0.5, 0.75, 1`;
- an insufficient-evidence verdict has no score;
- rationale is required; and
- scored verdicts group one or more nonblank observations under evidence IDs
  supplied in that request.

Each evidence ID is canonicalized to one group. If a model repeats a supplied
ID, its distinct observations are merged in first-seen order and exact duplicate
observations are retained once. Unknown IDs, blank observations, and empty
groups still fail closed. Observation bodies are validated transiently; durable
review audit keeps the unique cited IDs, score, rationale, model, usage, output
mode, and content digest without storing raw model output or observation text.

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

Review depth applies to every selected-episode and whole-session rubric:

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
rationale or safe error, evidence citations, structured-output mode, and usage.
An applicable rubric with zero successful reviewers fails stage coverage.
Scores are buffered until complete applicable coverage is known, then written
under the run's cancellation barrier.

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
- [PReMISE](https://arxiv.org/abs/2605.30803), for explicit rubric
  applicability; and
- [RuVerBench](https://arxiv.org/abs/2606.29920), for the limits and diminishing
  returns of agentic coding judges.

These papers motivate the architecture but do not calibrate this product. There
is no human-labeled validation set, and neither the selective margin nor episode
cap is claimed as an empirical optimum.
