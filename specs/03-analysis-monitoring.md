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
| Summaries, the A/B leaderboard, config regressions, session success and spend | CES EvaluationQueries (`ces_queries.py`) | every Score the Evaluations hold, by config version |
| Trends, config order, coaching examples, summary tag counts | raw Scores (`read_all_scores`) | the newest `--limit` turns and their sessions |
| Reflection's digest (runs and `reflect`) | the raw Scores it pins | that cohort or the newest `--limit` turns |

The queries count only the main agent: turn Scores only target main-agent
Cases, and the session query keeps depth 0. They name only rating Scorers, and
raw Score records leave out measurements and dimensions, so neither ever enters
a summary, leaderboard, trend, alert, or coaching digest. A summary over every Score differs
from the older newest-`--limit` summary when older turns scored differently.
The turn query runs on the turns Evaluation and the session query on the
sessions Evaluation ([`01-ces-io.md`](01-ces-io.md#queries)).

Trends still read raw Scores, because CES queries have no time; a query
grouped by the `dim.day` Score could replace them.
Coaching examples are session judge Scores, and a filtered session-row search
waits on a CES deploy. Configs are ordered by their latest turn among the newest
`--limit` turns, so a config missing from that window is not compared.

Until CES ships its spread aggregation (`stddev`), the config query gives a
continuous Scorer's mean and count but not its spread. Monopole finds out once
per process, when CES rejects a query that names `stddev`. Then each continuous
Scorer that has Scores gets one more query, grouped by config and by the
Scorer's own value, counting each value. That histogram gives the exact mean
and sample stddev over every Score. Example: ratings 0.5, 0.5, and 1.0 come back
as 0.5 twice and 1.0 once, so the mean is 2/3. No summary reads raw Scores.

When the histogram passes 5,000 result rows, or CES refuses it as too large,
the Scorer keeps the config query's mean and count with the largest stddev
ratings in [0, 1] with that mean can have: mean·(1−mean), times n/(n−1) for the
sample. Its intervals are wider than the data's own and are labeled
"conservative" in the CLI's leaderboard and digest. The Scorers round to 3 or 4
decimals, so a config has at most about 1,000 or 10,000 distinct values.

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

`analyze`'s summary, the coaching digest or `--ab`, starts with each config's
session success and spend from one query. It groups sessions by config, depth,
and `outcome.session_tests_passed`, keeps depth 0, and takes each
measurement's mean and count, whose product is its sum. A session succeeds when
its last test run passed. The success rate covers sessions that ran a test,
with a Wilson interval; sessions with no test run, including ones Monopole
never scored, are counted apart. Spend per success counts only sessions with a
test outcome: it divides the tokens, turns, and seconds of every session that
ran a test by the successes, so failed attempts count as the cost of the
successes, and a session that ran no test adds neither spend nor success. The
seconds are `measure.duration_s`, agent working time inside turns. A session's
tokens include its subagents', so subagent spend is counted once. For example,
3 passing sessions of 1,000 tokens, 2 failing of 4,000, and 5 untested of 50
give 60% success and 11,000 / 3 ≈ 3,667 tokens per success.

The digest is input to reflection, not an independent score or proof of the
cause of a regression. Reflection separately pins the exact feedback records it
consumed, so later feedback changes cannot silently alter an active proposal
evaluation.

## Query insights

`analyze --subagents`, `--judges`, and `--segments` each add Trajectory queries
over every session or Trajectory, run on the sessions Evaluation with the same
create keys as the session query ([`01-ces-io.md`](01-ces-io.md#queries)). They
are CLI-only; `monitor` and the Analyze page don't run them.

### Subagents

The query groups every Trajectory by the root Case's `i.config_version` and
`depth`, with the session outcome Scorers as metrics. It reports each group's
Trajectory count, scored or not: depth 0 is the session, depth 1 and deeper its
subagents, such as "config v3: 40 sessions, 12 subagents (0.30 per session;
depth 1: 10, depth 2: 2)". A subagent's root Case is the parent's Case that
launched it, which carries the same config version.

For each session outcome it then compares subagents, every depth below 0
pooled, with sessions: each side's rate and Newcombe's interval on subagents
minus sessions. For example, 35 of 50 sessions against 12 of 30 subagents whose
last test run passed is -0.30, interval -0.49 to -0.08. A session's outcomes
include its subagents' tool calls, so both sides can share a test run; that
correlation makes the independent-samples interval wider than the truth, not
narrower.

### Judge agreement

Each judge panel Scorer is compared with a reference Scorer on the sessions
both rated: with each session outcome Scorer, with `implicit.completion`, and
with every other panel of the same rubric. Each pair is one query grouped by both Score columns, which returns
exact counts of each (reference, judge) rating pair. A panel's rating is the
mean of one to three judges' five anchors, so a column has at most 13 values and
needs no bucketing. From the pairs the insight reports:

- agreement: the share of sessions where both pass or both fail, with a Wilson
  interval. A rating passes at or above its rubric's catalog threshold, and a
  true session outcome or `implicit.completion` passes. A query sees only ratings, so a run that
  pinned another threshold is judged by the catalog's.
- the judge's mean where the reference passes versus where it fails, with a
  Welch difference interval computed from the exact pairs, so it needs no CES
  `stddev`.

Only session Scorers are compared: the turn outcome Scorers (`outcome.test` and
the rest) score turns, which a Trajectory query can't read. The session outcome
Scorers carry the same evidence per session. `implicit.completion` is weaker: it
says only that the last turn didn't end in an error, and every session of two
turns or fewer counts as completed. Subagents have outcome Scores but no judge
Score, so they pair with nothing. Agreement is raw, not corrected for chance: two judges
that both pass 90% of sessions agree on at least 80% by construction.

### Segments

The query groups session Scores by the root Case's `i.config_version`, depth,
and one dimension Score, `dim.repo` by default or `dim.git_branch` or `dim.day`,
and keeps depth 0. Old sessions derive their dimensions from their Cases and new
ones copy the adapter's root-Case fields ([`02-evaluation.md`](02-evaluation.md)),
so both take this one path; a session without the dimension Score, such as one
not yet backfilled, is the unknown segment. Each
segment is compared with the rest of its config's sessions, not the whole
config, so the two sides are independent; the difference interval is the one
config regressions use. A segment is shown only when it and the rest each have
at least 10 ratings from the Scorer. Continuous Scorers have no interval until
CES has `stddev`.

A query returns at most 10,000 groups, which the subagent and judge queries stay
far below. `dim.day` segments add a group per day for each config and depth,
so a history of years across many configs could reach it.

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
