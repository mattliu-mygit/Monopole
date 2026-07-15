# Task 7 report: reflection coaching from session feedback

## Result

- Reflection coaching now includes at most three low-scoring session examples per rubric.
- Eligibility requires a complete session judgment with the full comparable evaluation context.
- Low score is evaluated against each judgment's pinned `rubric_threshold`.
- Examples sort deterministically by rating, execution time, and session identity.
- Per-reviewer behavioral feedback is normalized and deduplicated before rendering.
- Session IDs, evidence IDs, success, problem, and desired behavior are bounded; raw reasons, conversations, window findings, and attempt artifacts are never rendered in the feedback section.
- The retired trigger-selected episode diagnostic aggregation and its compatibility parameter/tests were removed.

## Review hardening

- Comparable context validation now enforces the persisted session markers and the same depth, margin, threshold, judge-count, uniqueness, and JSON-list invariants as the review policy.
- Reviewer feedback accepts only the exact persisted three-field object shape and examines no more than the authenticated reviewer count, capped at three, so oversized lists cannot expand work or output.
- Execution ordering parses timestamps as timezone-aware instants. Invalid or missing timestamp values use a deterministic typed fallback and cannot produce heterogeneous comparison errors.
- Adversarial tests cover invalid thresholds and review policies, conflicting session markers, duplicate and tuple judge selections, a 10,000-item feedback tail, malformed feedback objects, offset-equivalent timestamps, and mixed invalid timestamp types.

## TDD evidence

The new analysis tests first failed because the coaching digest did not contain a behavioral-feedback section (three expected assertion failures). The production aggregation was then implemented against the persisted session score shape where `details.behavioral_feedback` is a list of reviewer objects.

The review-hardening tests then failed in 13 expected cases: invalid policy contexts remained comparable, oversized reviewer lists rendered unbounded output, and heterogeneous timestamp values raised `TypeError`. The tightened validation, bounded iteration, and typed chronological sort made those cases pass.

## Verification

- `.venv/bin/python -m pytest -q tests/analysis/test_patterns.py tests/reflection/test_stage.py` — 86 passed
- `.venv/bin/python -m pytest -q` — 811 passed, one pre-existing Starlette deprecation warning
- `.venv/bin/ruff check src/weave_agent_signals/patterns.py tests/analysis/test_patterns.py tests/reflection/test_stage.py` — passed
- `.venv/bin/ruff format --check src/weave_agent_signals/patterns.py tests/analysis/test_patterns.py tests/reflection/test_stage.py` — passed
- `git diff --check` — passed
