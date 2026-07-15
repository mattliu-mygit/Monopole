# Task 6 report

Implemented the session-only sliding-window judging replacement.

## Result

- Judging plans now use schema version 2 and pin the complete session cohort, six v4
  session rubric descriptors, context policy, sliding protocol, per-positioned-reviewer
  model identity, exact window manifests, raw coverage, and reviewer work bounds.
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

- Focused plan/runner/stage/CLI tests: 14 passed.
- Broader judging/catalog/stage/CLI tests: 233 passed before final cleanup.
- Final backend suite: 783 passed, one existing Starlette deprecation warning.
- `ruff check src/ tests/`: passed.
- `ruff format --check src/ tests/`: passed.
- `git diff --check`: passed.
