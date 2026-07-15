# Task 8: sliding-window judging frontend

## Implemented

- Replaced episode-era judging types with the backend schema-v2 session plan:
  context policy, protocol manifest, reviewer-specific window plans, raw/core trace
  coverage, and digest/window/merge work bounds.
- Aligned effective run configuration with backend schema v2 and required model
  context limits.
- Reworked the judging stage UI to show session/turn/window scope, reviewer
  window manifests, artifact progress, final scores, behavioral feedback, and
  ordered inference-step audits including reuse and transport details.
- Preserved persisted failure, failed-attempt, truncation, and run cancellation
  presentation while removing all episode-only branches and copy.
- Updated frontend fixtures to use only session rubrics and current config,
  plan, progress, and result payloads. No compatibility fields were added.

## TDD evidence

- The new `JudgingProgress` fixture first failed type checking on the removed
  schema-v1 totals/session fields and missing behavioral feedback contract.
- The pinned-config audit test first failed because schema-v2 context limits
  were not rendered, then passed after the UI exposed those values.

## Verification

- `npm test` — 21 files, 124 tests passed.
- `npm run lint` — passed.
- `npm run build` — passed.
- `rg` found no episode, selected-episode, episode-cap, applicability, or skip
  compatibility fields under `frontend/`.

## Review fixes

- Behavioral feedback categories now match the nullable backend contract and
  the UI omits absent categories.
- The principal fixture now satisfies selective-review cardinality, full raw
  coverage, reviewer work bounds, recomputed totals, and the stable
  `merged_verdict` schema identity.
- The shared plan fixture is emitted by the backend plan constructor with
  canonical hashes, then checked through persisted-plan and cohort validation
  in a backend cross-contract test. Run-detail tests consume that same valid
  plan and cohort instead of hand-authored approximations.
