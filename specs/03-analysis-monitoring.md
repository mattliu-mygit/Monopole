# Analysis and monitoring

Analysis turns score feedback into summaries, configuration comparisons, time
trends, coaching input, and regression alerts. Its central constraint is that
evidence must be comparable before it is aggregated.

## Eligible evidence

Deterministic scores cover every eligible hydrated trace and can support
ordinary summaries when their rating is valid.

Sliding judge scores describe complete sessions: every captured turn receives
raw coverage in one core chunk, while the final verdict is merged from all
windows. They may enter ordinary analysis only when their review is complete
and their comparison context is fully identified. Degraded, failed, or
incompletely described judgments remain available for audit but do not
support ordinary quality claims. Historical non-session judgments and judge
records without a current session evaluation unit are likewise audit-only.
Complete coverage improves comparability; it does not make the observed
sessions a random or causally representative sample.

## Comparable cohorts

Judged session scores are compared only when they share:

- rubric version and threshold;
- panel contract version; and
- ordered requested judge selection.

The runtime-dependent number of reviewer attempts is evidence about a review,
not a cohort key. A score missing required context stays visible in raw feedback
but is excluded from aggregate analysis.

Every accepted rating must be a real finite number in `[0, 1]`; booleans,
numeric strings, non-finite values, and out-of-range values are rejected rather
than coerced. Because every persisted score is higher-is-better, downward
movement always has one interpretation.

## Analysis surfaces

Summaries report sample count, mean, tags, and an interval appropriate to binary
or continuous ratings. A pass rate is shown only when every observed rating is
binary. Small samples are labeled as low confidence instead of being presented
as conclusive.

Configuration analysis groups feedback by the adapter's observed configuration
version, while keeping incomparable judge contexts separate. It describes
association, not causal attribution.

Trend analysis orders scores by agent execution time stored in feedback
details, not by the later feedback-write time. It compares older and recent
windows only after the evidence passes the representative/comparability rules.

The coaching digest combines eligible summaries, configuration context, recent
trends, and bounded low-scoring behavioral examples. For each rubric it selects
at most three complete, comparable session judgments below that rubric's pinned
threshold, ordered deterministically by score and execution context. Examples
include bounded session and evidence identities plus merged reviewer success,
problem, and desired-behavior fields. They never include raw conversation
windows or raw window findings.

The digest is input to reflection, not an independent score or proof of the
cause of a regression. Reflection separately pins the exact feedback records it
consumed, so later feedback changes cannot silently alter an active proposal
evaluation.

## Monitoring

Monitoring reports only new statistically significant downward movements in
eligible time trends or configuration comparisons. Alerts identify the score,
sample context, magnitude, and comparison rather than claiming a root cause.

When a state file is configured, monitoring retains stable keys for regressions
that are currently significant. An ongoing regression is not sent repeatedly;
observed recovery removes its active key so a later recurrence can alert again.
A failed delivery is not marked active and remains retryable. Monitoring state
is optional and separate from evaluation-run state.

Low W&B Agent Signals are high-recall recommendations for human review, not
Monopole quality judgments. The Sessions page and evaluation-run session
selector call out affected conversations with the lowest observed rating and
signal names. The callout does not reorder, select, or exclude a session and
does not assert that the conversation failed an evaluation.

## Limits

- Intervals summarize observed samples; they do not correct trace-selection,
  missing-evidence, or model-judge bias.
- Configuration comparisons are observational.
- Complete session coverage does not correct cohort-selection or model-judge
  bias.
- A lack of an alert means the configured statistical and sample requirements
  were not met; it does not prove the absence of a regression.
