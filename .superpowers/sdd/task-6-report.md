# Task 6 report

Implemented the session-only sliding-window judging replacement.

## Result

- Judging plans now use schema version 2 and pin the complete session cohort, six v4
  session rubric descriptors, context policy, sliding protocol, per-positioned-reviewer
  model identity, exact window manifests, raw coverage, reviewer work bounds, and the
  selective-review second-opinion margin.
- `SlidingReviewer` authenticates the full content-derived judging plan before resolving
  the exact reviewer ordinal and rebuilding its pinned window manifest from current
  evidence.
- The runner exposes only `judge_session`, lazily creates one reviewer per
  judge/session, uses the unchanged escalation policy, and returns one merged session
  score per rubric with behavioral feedback, bounded reason text, coverage, plan,
  rubric, reviewer, attempt, inference-step, and artifact-reuse provenance.
- The durable stage iterates sessions only, persists immutable artifacts through the
  run store, reports unique digest/window/merge progress, buffers all results until
  coverage is complete, and writes only conversation refs.
- The direct CLI uses the same plan and runner with a conflict-checking in-memory
  artifact map and reports sessions, reviewer windows, rubric scores, and writes.
- Store validation now accepts only the schema-v2 session/window plan and validates its
  complete raw cohort coverage and reviewer ordinals.

## Removed original framework

- Deleted `judge_turn` and all turn-target judging calls and exports.
- Deleted episode selection, salience triggers, applicability heuristics, episode caps,
  selected/omitted episode records, episode attempt totals, and their runtime paths.
- Deleted the old raw turn/context/session digest builders and their compatibility tests.
- Deleted the old single-call judge verdict schema/parser/exports and its tests; the
  sliding contracts now own their score-anchor and strict-input validation directly.
- Removed split `RUBRICS`/`CROSS_TURN_RUBRICS` maps in favor of one authoritative
  `SESSION_RUBRICS` map.
- Replaced old episode runner, stage, CLI, plan, rubric, and store fixtures with
  session-window coverage.

## Verification

- Review-fix focused judging/CLI/stage/store suite: 94 passed.
- Final backend suite: 798 passed, one existing Starlette deprecation warning.
- `ruff check src/ tests/`: passed.
- `ruff format --check src/ tests/`: passed.
- `git diff --check`: passed.

## Review hardening

- Removed the retired schema-v3 verdict layout from rubric prompts, leaving structured
  output layout exclusively to sliding phase schemas and prompts.
- Made direct CLI writes fail closed across the complete selected session/rubric scope.
- Added pre-inference runner authentication for plan digest/schema, review depth,
  second-opinion margin, ordered positioned judges, context policy, exact rubric rows,
  and attempt bounds.
- Persist phase progress after each immutable artifact commit, reconstruct unique phase
  counts on resume, and use artifact identity rather than attempt audits as the
  authoritative completed-work source.
- Strengthened store validation for exact plan/window/reviewer/rubric/policy/protocol
  shapes, hashes, coverage geometry, work bounds, and recomputed totals.
- Closed the remaining self-rehash gap by enforcing maximum chunk count, merge/input
  and raw-window budgets, canonical raw-turn SHA-256 digests, and the tight raw-window
  token range derivable from per-turn estimates plus separator overhead.
- Restored session-only safety coverage for cancellation, durable cancellation/write
  races, fail-closed buffering, force replacement ordering, write and cleanup failure
  behavior, bounded audits/failures, and monotonic resumed artifact progress.
- Made the selective-review margin part of the schema-v2 plan digest, validated it
  strictly on persistence, and required the runner to authenticate it before creating
  any reviewer. Primary and full-panel plans canonically pin a null margin.
