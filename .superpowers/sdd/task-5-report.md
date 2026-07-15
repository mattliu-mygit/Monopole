# Task 5 report

Implemented one reviewer's resumable digest-window-merge pipeline.

## Result

- Added rubric-neutral, reviewer-specific chunk digests guarded by a shared lock.
- Added ordered raw-window inference with chronological surrounding digests.
- Added ordered cross-window finding deduplication and a full raw-coverage merge manifest.
- Added strict semantic parsing, exact pinned-plan validation, and immutable artifact replay.
- Added request-budget and between-call cancellation checks.
- Added aggregate portable usage plus one `InferenceStepAudit` per actual inference call.
- Extended `AttemptObservation` with immutable behavioral feedback and inference steps while
  leaving `execute_review` escalation decisions unchanged.
- Failed, malformed, and over-budget pipelines return failed observations without feedback or
  evidence; artifact replay makes zero repeated model calls.

## Verification

- `.venv/bin/python -m pytest -q tests/evaluation/judging/test_sliding.py tests/evaluation/judging/test_review_policy.py` — 47 passed.
- `.venv/bin/python -m pytest -q tests/evaluation/judging/test_sliding.py tests/evaluation/judging/test_review_policy.py tests/evaluation/judging/test_sliding_contracts.py tests/evaluation/judging/test_windowing.py tests/runs/test_store.py` — 167 passed.
- Ruff check and format check passed for all four changed source/test files.
- `git diff --check` passed.

The broader pre-Task-6 judging suite has 15 expected transitional failures because the old
`judge_turn` and `judge_session` callers construct successful observations without behavioral
feedback. Task 6 replaces those callers with `SlidingReviewer`; this task intentionally did not
modify the old runner or framework.
