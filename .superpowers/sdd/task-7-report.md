# Task 7 report: reflection coaching from session feedback

## Result

- Reflection coaching now includes at most three low-scoring session examples per rubric.
- Eligibility requires a complete session judgment with the full comparable evaluation context.
- Low score is evaluated against each judgment's pinned `rubric_threshold`.
- Examples sort deterministically by rating, execution time, and session identity.
- Per-reviewer behavioral feedback is normalized and deduplicated before rendering.
- Session IDs, evidence IDs, success, problem, and desired behavior are bounded; raw reasons, conversations, window findings, and attempt artifacts are never rendered in the feedback section.
- The retired trigger-selected episode diagnostic aggregation and its compatibility parameter/tests were removed.

## TDD evidence

The new analysis tests first failed because the coaching digest did not contain a behavioral-feedback section (three expected assertion failures). The production aggregation was then implemented against the persisted session score shape where `details.behavioral_feedback` is a list of reviewer objects.

## Verification

- `.venv/bin/python -m pytest -q tests/analysis/test_patterns.py tests/reflection/test_stage.py` — 72 passed
- `.venv/bin/python -m pytest -q` — 797 passed, one pre-existing Starlette deprecation warning
- `.venv/bin/ruff check src/weave_agent_signals/patterns.py tests/analysis/test_patterns.py tests/reflection/test_stage.py` — passed
- `.venv/bin/ruff format --check src/weave_agent_signals/patterns.py tests/analysis/test_patterns.py tests/reflection/test_stage.py` — passed
- `git diff --check` — passed
