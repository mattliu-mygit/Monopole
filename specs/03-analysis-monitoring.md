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

## Where the numbers come from

`analyze`, `monitor`, and the Analyze page read two kinds of evidence:

| Numbers | Source | Scope |
| --- | --- | --- |
| Summaries, the A/B leaderboard, config regressions | CES EvaluationQueries (`ces_queries.py`) | every Score the Evaluations hold, by config version |
| Trends, config order, coaching examples, summary tag counts | raw Scores (`read_all_scores`) | the newest `--limit` turns and their sessions |
| Reflection's digest (runs and `reflect`) | the raw Scores it pins | that cohort or the newest `--limit` turns |

The queries count only the main agent: turn Scores only target main-agent
Cases, and the session query keeps depth 0. A summary over every Score differs
from the older newest-`--limit` summary when older turns scored differently.
The turn query runs on the turns Evaluation and the session query on the
sessions Evaluation. On a CES that runs queries only on Case Evaluations, the
session query falls back to the turns Evaluation, as
[`01-ces-io.md`](01-ces-io.md#queries) describes.

Trends wait for a day field from the adapter, because CES queries have no time.
Coaching examples are session judge Scores, and a filtered session-row search
waits on a CES deploy. Configs are ordered by their latest turn among the newest
`--limit` turns, so a config missing from that window is not compared.

Until CES ships its spread aggregation (`stddev`), a query cannot give a
continuous Scorer's interval. Then continuous Scorers' summaries, leaderboard
entries, and comparisons come from the raw newest-`--limit` Scores, and binary
Scorers still come from queries. Monopole finds out once per process, when CES
rejects a query that names `stddev`.

## Analysis surfaces

Summaries report sample count, mean, and an interval appropriate to binary or
continuous ratings: Wilson for binary Scorers, the normal approximation for
continuous ones. A pass rate is shown only for binary Scorers, the outcome
Scorers and `implicit.completion`, whose every value is 0 or 1. Queries cannot
count tags, so a summary's tag counts come from the raw newest-`--limit` Scores
even when its count and mean cover every Score.
Small samples are labeled as low confidence instead of being presented as
conclusive.

Configuration analysis groups Scores by the adapter's observed configuration
version. Each judge rubric and panel is its own Scorer, so incomparable judge
contexts stay separate. A rubric threshold change within one rubric version and
panel is not separated. It describes association, not causal attribution.

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

## Query insights

`analyze --subagents` adds a Trajectory query
over every Trajectory, run on the sessions Evaluation with the same fallback and
create keys as the session query ([`01-ces-io.md`](01-ces-io.md#queries)). It is
CLI-only; `monitor` and the Analyze page don't run it.

### Subagents

The query groups every Trajectory by the root Case's `i.config_version` and
`depth`, and reports each group's Trajectory count, scored or not: depth 0 is
the session, depth 1 and deeper its subagents. No Monopole Scorer scores a
subagent Trajectory, because turn, session, and judge Scores all target the
session or its Cases, so the insight shows counts only, such as "config v3: 40
sessions, 12 subagents (0.30 per session; depth 1: 10, depth 2: 2)", not a
main-versus-subagent score gap. A subagent's root Case is the parent's Case
that launched it, which carries the same config version.

A query returns at most 10,000 groups, which this query stays far below.

## Monitoring

Monitoring reports only new statistically significant downward movements in
eligible time trends or configuration comparisons. Alerts identify the score,
sample context, magnitude, and comparison rather than claiming a root cause.

A configuration comparison takes each Scorer's newest two configs and reports a
drop of at least 0.2. A trend compares the older and recent windows and reports
a change of at least 0.2 either way. Both call a change significant when the 95%
interval on the difference (newer minus older) excludes 0: Newcombe's interval
for binary Scorers, and Welch's standard error with the normal approximation for
continuous ones, which need at least two ratings on each side. This replaced
asking whether the two sides' own intervals were disjoint with at least five
ratings each, a test stricter than 95% that missed real drops. For example, 40
of 50 passing then 30 of 50 has overlapping intervals but a difference interval
of -0.36 to -0.02, so it now alerts. Without the five-rating floor, a small but
clean change can be significant: 2 of 2 passing then 0 of 2 has a difference
interval of -1.0 to -0.07.

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
