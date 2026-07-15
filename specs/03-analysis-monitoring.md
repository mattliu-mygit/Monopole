# Analysis and monitoring

Analysis turns score feedback into summaries, configuration comparisons, time
trends, coaching input, and regression alerts. Its central constraint is that
evidence must be comparable before it is aggregated.

## Representative and diagnostic evidence

Deterministic scores cover every eligible hydrated trace and can support
ordinary summaries when their rating is valid.

Selected-episode judgments intentionally oversample high-information moments.
They are diagnostic examples, not a random sample of turns. They are excluded
from population pass rates, time-trend claims, configuration-regression claims,
and confidence intervals. Coaching may show their selected-sample mean only
under an explicit diagnostic label.

Whole-session judgments may enter ordinary analysis only when their review is
complete and their comparison context is fully identified. Degraded,
unresolved, failed, or incompletely described judgments remain available for
audit but do not support ordinary quality claims.

## Comparable cohorts

Judged session scores are compared only when they share:

- rubric version and threshold;
- review depth and policy version;
- selective-review margin; and
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

The coaching digest combines representative summaries, aggregate
selected-episode diagnostics, configuration context, and recent trends. It is
input to reflection, not an independent score or proof of the cause of a
regression.

## Monitoring

Monitoring reports only new statistically significant downward movements in
eligible time trends or configuration comparisons. Alerts identify the score,
sample context, magnitude, and comparison rather than claiming a root cause.

When a state file is configured, monitoring retains stable keys for regressions
that are currently significant. An ongoing regression is not sent repeatedly;
observed recovery removes its active key so a later recurrence can alert again.
A failed delivery is not marked active and remains retryable. Monitoring state
is optional and separate from evaluation-run state.

## Limits

- Intervals summarize observed samples; they do not correct trace-selection,
  missing-evidence, or model-judge bias.
- Configuration comparisons are observational.
- Trigger-selected judgments remain useful for debugging and coaching even
  though they cannot support population claims.
- A lack of an alert means the configured statistical and sample requirements
  were not met; it does not prove the absence of a regression.
