from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from weave_agent_signals.runs.judging_activity import JudgingActivityLog


def test_concurrent_activity_records_keep_unique_ordered_ids() -> None:
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    first_clock_entered = threading.Event()
    release_first_clock = threading.Event()
    clock_lock = threading.Lock()
    clock_calls = 0

    def clock() -> datetime:
        nonlocal clock_calls
        with clock_lock:
            clock_calls += 1
            call = clock_calls
        if call == 1:
            first_clock_entered.set()
            assert release_first_clock.wait(timeout=1)
        return now + timedelta(seconds=call)

    activity = JudgingActivityLog(
        initial_snapshot={"started_at": now.isoformat()},
        clock=clock,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(activity.record, {"phase": "judge_started", "message": "one"})
        assert first_clock_entered.wait(timeout=1)
        second = executor.submit(
            activity.record,
            {"phase": "judge_started", "message": "two"},
        )
        time.sleep(0.05)
        release_first_clock.set()
        first.result(timeout=1)
        second.result(timeout=1)

    assert [event["id"] for event in activity.snapshot()["events"]] == [1, 2]
