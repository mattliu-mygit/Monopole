"""Tests for persisted reflection activity snapshots."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from weave_agent_signals.runs.progress import ReflectionProgressRecorder


def _iter_clock(*values: str):
    times = iter(datetime.fromisoformat(value) for value in values)
    return lambda: next(times)


def _incrementing_clock():
    current = datetime(2026, 7, 14, 19, 20, tzinfo=timezone.utc)

    def now():
        nonlocal current
        value = current
        current += timedelta(seconds=1)
        return value

    return now


def _persisted_snapshot():
    return {
        "phase": "candidate_evaluated",
        "status_message": "Earlier work",
        "started_at": "2026-07-14T19:00:00+00:00",
        "attempted": 3,
        "valid": 2,
        "rejected": 1,
        "scored": 1,
        "total_attempts": 3,
        "events": [
            {
                "id": 7,
                "at": "2026-07-14T19:01:00+00:00",
                "phase": "candidate_evaluated",
                "message": "Earlier work",
            }
        ],
    }


def test_recorder_persists_complete_snapshots_with_metadata():
    saved = []
    recorder = ReflectionProgressRecorder(
        saved.append,
        total_attempts=3,
        context={
            "proposal_writer": "gpt",
            "proposal_evaluator": "llama",
            "no_improvement_patience": 2,
        },
        clock=_iter_clock(
            "2026-07-14T19:20:00+00:00",
            "2026-07-14T19:20:02+00:00",
            "2026-07-14T19:20:04+00:00",
        ),
    )

    recorder.record(
        "generating_candidate",
        "Generating candidate 1 of 3 — gpt",
        candidate=1,
        model="gpt",
        acting_role="proposal_writer",
        attempt_id="attempt-1",
        attempted=1,
        valid=0,
        rejected=0,
        scored=0,
    )
    recorder.handle(
        {
            "phase": "candidate_evaluated",
            "message": "Candidate 1 scored 0.81 with llama",
            "candidate": 1,
            "model": "llama",
            "acting_role": "proposal_evaluator",
            "attempt_id": "attempt-1",
            "evaluation_id": "evaluation-2",
            "score": 0.81,
            "attempted": 1,
            "valid": 1,
            "rejected": 0,
            "scored": 1,
        }
    )

    snapshot = saved[-1]
    assert snapshot == {
        "proposal_writer": "gpt",
        "proposal_evaluator": "llama",
        "no_improvement_patience": 2,
        "phase": "candidate_evaluated",
        "status_message": "Candidate 1 scored 0.81 with llama",
        "started_at": "2026-07-14T19:20:00+00:00",
        "attempted": 1,
        "valid": 1,
        "rejected": 0,
        "scored": 1,
        "total_attempts": 3,
        "events": [
            {
                "id": 1,
                "at": "2026-07-14T19:20:02+00:00",
                "phase": "generating_candidate",
                "message": "Generating candidate 1 of 3 — gpt",
                "candidate": 1,
                "model": "gpt",
                "acting_role": "proposal_writer",
                "attempt_id": "attempt-1",
            },
            {
                "id": 2,
                "at": "2026-07-14T19:20:04+00:00",
                "phase": "candidate_evaluated",
                "message": "Candidate 1 scored 0.81 with llama",
                "candidate": 1,
                "model": "llama",
                "acting_role": "proposal_evaluator",
                "attempt_id": "attempt-1",
                "evaluation_id": "evaluation-2",
                "score": 0.81,
            },
        ],
    }


def test_recorder_caps_history_without_reusing_ids():
    saved = []
    recorder = ReflectionProgressRecorder(saved.append, clock=_incrementing_clock())

    for index in range(105):
        recorder.record("working", f"Event {index}")

    events = saved[-1]["events"]
    assert len(events) == 100
    assert [event["id"] for event in events] == list(range(6, 106))


def test_recorder_snapshots_do_not_share_mutable_event_lists():
    saved = []
    recorder = ReflectionProgressRecorder(saved.append, clock=_incrementing_clock())

    recorder.record("one", "First")
    recorder.record("two", "Second")

    assert [event["message"] for event in saved[0]["events"]] == ["First"]


def test_recorder_never_decreases_completed_counters():
    saved = []
    recorder = ReflectionProgressRecorder(
        saved.append,
        total_attempts=3,
        clock=_incrementing_clock(),
    )

    recorder.record(
        "candidate_scored",
        "Second scored",
        attempted=3,
        valid=2,
        rejected=1,
        scored=2,
    )
    recorder.record(
        "scoring_candidate",
        "Rechecking first",
        attempted=1,
        valid=1,
        rejected=0,
        scored=1,
    )

    assert saved[-1]["attempted"] == 3
    assert saved[-1]["valid"] == 2
    assert saved[-1]["rejected"] == 1
    assert saved[-1]["scored"] == 2


def test_recorder_never_decreases_the_proposal_attempt_limit():
    saved = []
    recorder = ReflectionProgressRecorder(
        saved.append,
        total_attempts=3,
        clock=_incrementing_clock(),
    )

    recorder.record("starting", "Starting reflection", total_attempts=3)
    recorder.record("working", "Writer is still working", total_attempts=1)

    assert saved[-1]["total_attempts"] == 3


def test_recorder_resumes_persisted_events_counters_and_ids():
    saved = []
    initial = _persisted_snapshot()
    recorder = ReflectionProgressRecorder(
        saved.append,
        initial_snapshot=initial,
        clock=_incrementing_clock(),
    )

    recorder.record(
        "resumed",
        "Continuing",
        attempted=1,
        valid=1,
        rejected=0,
        scored=0,
    )

    assert saved[-1]["started_at"] == initial["started_at"]
    assert saved[-1]["attempted"] == 3
    assert saved[-1]["valid"] == 2
    assert saved[-1]["rejected"] == 1
    assert saved[-1]["scored"] == 1
    assert [event["id"] for event in saved[-1]["events"]] == [7, 8]


def test_recorder_rejects_incomplete_persisted_progress():
    initial = _persisted_snapshot()
    del initial["attempted"]

    with pytest.raises(ValueError, match="complete current progress contract"):
        ReflectionProgressRecorder(lambda _snapshot: None, initial_snapshot=initial)


def test_recorder_rejects_legacy_persisted_progress_fields():
    initial = _persisted_snapshot()
    initial["generated"] = 2

    with pytest.raises(ValueError, match="retired progress fields"):
        ReflectionProgressRecorder(lambda _snapshot: None, initial_snapshot=initial)


def test_progress_persistence_failure_is_observational_and_does_not_abort_work():
    attempts = []

    def flaky_sink(snapshot):
        attempts.append(snapshot)
        if len(attempts) == 1:
            raise OSError("transient persistence failure")

    recorder = ReflectionProgressRecorder(flaky_sink, clock=_incrementing_clock())

    recorder.record("loading", "Loading inputs")
    recorder.record("loaded", "Inputs loaded")

    assert len(attempts) == 2
    assert [event["message"] for event in attempts[-1]["events"]] == [
        "Loading inputs",
        "Inputs loaded",
    ]


def test_rejected_proposal_diagnostics_are_persisted_without_legacy_counters():
    saved = []
    recorder = ReflectionProgressRecorder(
        saved.append,
        total_attempts=3,
        clock=_incrementing_clock(),
    )

    recorder.record(
        "candidate_rejected",
        "Proposal path is outside managed scope",
        attempt_id="attempt-1",
        model="writer",
        acting_role="proposal_writer",
        error_type="BundleValidationError",
        changed_paths=("../secret.md",),
        response_digest=f"sha256:{'a' * 64}",
        response_excerpt='{"path":"../secret.md"}',
        attempted=1,
        valid=0,
        rejected=1,
        scored=0,
    )

    snapshot = saved[-1]
    assert set(snapshot) >= {
        "attempted",
        "valid",
        "rejected",
        "scored",
        "total_attempts",
    }
    assert "generated" not in snapshot
    assert "total_candidates" not in snapshot
    assert snapshot["events"][-1] == {
        "id": 1,
        "at": "2026-07-14T19:20:01+00:00",
        "phase": "candidate_rejected",
        "message": "Proposal path is outside managed scope",
        "model": "writer",
        "acting_role": "proposal_writer",
        "attempt_id": "attempt-1",
        "error_type": "BundleValidationError",
        "changed_paths": ["../secret.md"],
        "response_digest": f"sha256:{'a' * 64}",
        "response_excerpt": '{"path":"../secret.md"}',
    }


def test_record_and_resume_reject_event_keys_outside_the_contract():
    recorder = ReflectionProgressRecorder(lambda _snapshot: None, total_attempts=3)

    with pytest.raises(ValueError, match="unexpected progress event fields"):
        recorder.record("working", "Working", raw_output="must not persist")

    initial = _persisted_snapshot()
    initial["events"][0]["raw_output"] = "must not persist"
    with pytest.raises(ValueError, match="unexpected progress event fields"):
        ReflectionProgressRecorder(lambda _snapshot: None, initial_snapshot=initial)


@pytest.mark.parametrize(
    ("details", "message"),
    [
        ({"attempted": True}, "attempted"),
        ({"valid": "1"}, "valid"),
        ({"score": True}, "score"),
        ({"score": float("nan")}, "score"),
        ({"score": 1.01}, "score"),
        ({"candidate": 0}, "candidate"),
        ({"changed_paths": ["ok.md", 1]}, "changed_paths"),
        ({"response_digest": "sha256:abc"}, "response_digest"),
    ],
)
def test_record_rejects_malformed_typed_details(details, message):
    recorder = ReflectionProgressRecorder(lambda _snapshot: None, total_attempts=3)

    with pytest.raises(ValueError, match=message):
        recorder.record("working", "Working", **details)


def test_resume_applies_the_same_strict_event_validation_as_record():
    initial = _persisted_snapshot()
    initial["events"][0]["score"] = "0.8"

    with pytest.raises(ValueError, match="score"):
        ReflectionProgressRecorder(lambda _snapshot: None, initial_snapshot=initial)


def test_recorder_bounds_and_sanitizes_untrusted_event_text():
    saved = []
    recorder = ReflectionProgressRecorder(
        saved.append,
        total_attempts=3,
        clock=_incrementing_clock(),
    )

    recorder.record(
        "candidate_rejected",
        "token=message-secret\x00 " + "m" * 1_000,
        error_type="BundleValidationError",
        error="password=error-secret " + "e" * 1_000,
        changed_paths=("folder/unsafe\x00name.md" + "p" * 1_000,),
        response_digest=f"sha256:{'a' * 64}",
        response_excerpt="api_key=excerpt-secret " + "x" * 2_000,
        attempted=1,
        valid=0,
        rejected=1,
        scored=0,
    )

    event = saved[-1]["events"][-1]
    assert "message-secret" not in event["message"]
    assert "error-secret" not in event["error"]
    assert "excerpt-secret" not in event["response_excerpt"]
    assert "\x00" not in event["changed_paths"][0]
    assert len(event["message"]) <= 500
    assert len(event["error"]) <= 500
    assert len(event["changed_paths"][0]) <= 500
    assert len(event["response_excerpt"]) <= 1_000


def test_recorder_rejects_malformed_snapshot_context_and_counter_invariants():
    initial = _persisted_snapshot()
    initial["attempted"] = 1
    initial["valid"] = 1
    initial["rejected"] = 1

    with pytest.raises(ValueError, match="completed attempts"):
        ReflectionProgressRecorder(lambda _snapshot: None, initial_snapshot=initial)

    unknown_context = deepcopy(_persisted_snapshot())
    unknown_context["candidate_budget"] = 3
    with pytest.raises(ValueError, match="retired progress fields"):
        ReflectionProgressRecorder(lambda _snapshot: None, initial_snapshot=unknown_context)
