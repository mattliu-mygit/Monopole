# Sliding-window judging design

## Goal

Evaluate every part of a large agent conversation without forcing one judge call
to interpret the entire raw trace at once. Preserve local raw evidence, retain
enough global context to interpret it, and produce one comparable score plus
actionable behavioral feedback for each rubric and session.

This replaces trigger-selected model judgments with exhaustive session judging.
Deterministic scoring remains unchanged.

## Observable behavior

Each selected session is divided at turn boundaries into ordered chunks. Every
chunk receives a bounded, evidence-cited digest. For each rubric and reviewer,
judging advances through the session so that every chunk appears as raw evidence
in at least one window. A window contains the active raw chunk, a small raw
neighbor overlap, and the ordered digests for the rest of the session.

Each window reports a bounded set of structured findings rather than a rating.
A finding names the rubric, describes one relevant observed behavior, states
whether it is positive or negative, and cites only evidence identifiers visible
in that window. Findings may also report that the window contains no relevant
evidence.

After every chunk has been examined raw, the same reviewer receives the complete
ordered finding set and session-level digest context. It deduplicates overlapping
findings and returns one final verdict for that rubric and session. Existing
primary, selective, and full-panel review policies operate on these final
reviewer verdicts; successful reviewer ratings continue to be pooled under the
existing policy.

The persisted score is attached to the session, not to an individual turn.
Window findings remain run audit evidence and do not become independent Weave
scores.

## Context budgeting

The target input size for each window is approximately 100,000 tokens, but the
effective configuration pins the exact budget and evidence-rendering contract.
The available raw-evidence budget is the target input budget minus rubric and
protocol instructions, all included digests, the merge/output reserve, and a
safety reserve. Chunk boundaries are chosen only after this budget is known.

The first version requires all ordered chunk digests to fit beside a useful raw
window. It never silently drops context or exceeds a model's declared supported
input limit. A session whose digests cannot fit this contract is rejected as
unsupported before paid window judging begins; hierarchical digesting is not
part of this design.

Neighbor overlap exists only to preserve behavior that crosses a chunk boundary.
Overlapping raw evidence keeps its original evidence identifiers so the merge
can recognize duplicates. Digests are context, not substitutes for exhaustive
raw coverage: every non-overlap portion of every chunk must appear raw.

## Reviewer independence and digest provenance

Each configured reviewer creates and uses its own chunk digests. This avoids
making later reviewers depend on summaries produced by the first reviewer. The
run pins the digest prompt version, reviewer identity, chunk membership, ordered
evidence identifiers, content digests, token counts, and generated digest for
every reviewer.

A digest must distinguish observed facts from omissions and must cite the source
trace identifiers for every retained claim. A missing, malformed, uncited, or
over-budget digest fails that reviewer attempt; it cannot be treated as neutral
evidence.

## Final score and behavioral feedback

The final merge uses the existing anchored rating scale and insufficient-evidence
semantics. In addition, it returns concise evidence-cited behavioral feedback:

- the most important successful behavior, when present;
- the most important problem, when present; and
- the desired agent behavior that would address the problem.

Feedback describes agent behavior and must not prescribe edits to managed
instruction files. Reflection owns the choice of instruction changes.

The final rating, rationale, behavioral feedback, cited evidence identifiers,
window coverage, merge provenance, and reviewer-attempt audit are written as one
session feedback record per rubric. Text fields are validated and bounded before
persistence. Raw prompts, raw model output, and unrestricted finding text are
not persisted.

## Reflection and analysis

Merged session judgments are representative evaluation evidence when their
coverage and review are complete. Every session contributes at most one score
per rubric, so long sessions do not receive extra statistical weight merely
because they contain more windows.

The coaching digest continues to summarize comparable scores and tags. It also
includes a bounded set of low-scoring merged behavioral-feedback examples per
rubric. These examples retain evidence identifiers and session identity so the
proposal writer can understand the observed failure without receiving raw
conversation content. Reflection consumes the feedback but remains responsible
for proposing and evaluating coherent instruction-bundle changes.

Incomplete window coverage, a failed merge, or an unresolved review remains
available for audit but cannot enter ordinary trends, configuration comparisons,
regression alerts, or reflection coaching.

## Pinned execution and failure behavior

Before inference, the judging plan pins the complete session partition, overlap,
context budgets, digest limits, requested rubrics, and minimum and maximum
reviewer work. Retries restore the same plan and completed immutable artifacts;
they do not repartition the session or admit newly recorded turns.

Progress counts digest generation, raw windows, merges, and final rubric
coverage monotonically. Cancellation applies between every external call and
before the protected feedback-write batch. Scores remain buffered until every
applicable session rubric has complete window coverage and an accepted final
review result.

Any missing pinned trace, changed evidence identity, invalid citation, malformed
finding, missing raw coverage, digest failure, merge failure, or reviewer with
no valid final verdict fails the affected coverage. It never produces a neutral
score or a partial representative result.

## Tradeoffs

This design costs more than selective episode judging because every session
chunk is examined raw by each reviewer that is actually invoked. In exchange,
it removes trigger-selection blind spots and produces comparable session-level
signals for reflection and monitoring.

Generated digests can still omit or distort cross-window context. Exhaustive raw
coverage, neighbor overlap, evidence citations, reviewer-specific digests, and a
final merge limit that risk without introducing an unrestricted agentic judge.

## Validation

Focused tests cover deterministic chunking, token-budget enforcement, boundary
overlap, full raw coverage, citation validation, duplicate finding merge,
reviewer independence, retry stability, cancellation, and fail-closed writes.

Evaluation fixtures include long synthetic sessions with decisive behavior at
the beginning, middle, end, and across chunk boundaries. Controlled mutations
add or remove failures, recovery, corrections, constraints, and verification.
The new pipeline must outperform the current bounded digest on detecting those
mutations without regressing short-session judgments before it becomes the
default pipeline version.
