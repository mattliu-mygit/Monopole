from __future__ import annotations

import threading

import pytest

from weave_agent_signals.runs import RunStatus, RunStore


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs.db")


def test_create_run(store):
    run = store.create()
    assert run.run_id.startswith("run-")
    assert run.status == RunStatus.CREATED
    assert run.data_selection is None


def test_get_run(store):
    created = store.create()
    fetched = store.get(created.run_id)
    assert fetched.run_id == created.run_id


def test_get_missing_returns_none(store):
    assert store.get("run-nonexistent") is None


def test_list_runs(store):
    store.create()
    store.create()
    runs = store.list()
    assert len(runs) == 2


def test_set_selection(store):
    run = store.create()
    store.set_selection(
        run.run_id,
        since="2026-07-01",
        until="2026-07-12",
        session_ids=["conv-abc", "conv-def"],
    )
    updated = store.get(run.run_id)
    assert updated.data_selection is not None
    assert updated.data_selection.since == "2026-07-01"
    assert updated.data_selection.session_ids == ["conv-abc", "conv-def"]


def test_advance_status(store):
    run = store.create()
    store.set_selection(run.run_id, since="2026-07-01")
    store.update(run.run_id, status=RunStatus.SCORING)
    updated = store.get(run.run_id)
    assert updated.status == RunStatus.SCORING


def test_invalid_transition_raises(store):
    run = store.create()
    with pytest.raises(ValueError, match="Invalid transition"):
        store.update(run.run_id, status=RunStatus.JUDGING)


def test_update_step_results(store):
    run = store.create()
    store.update(run.run_id, status=RunStatus.SCORING)
    store.update(
        run.run_id,
        scoring_progress={"scored": 5, "total": 10},
    )
    updated = store.get(run.run_id)
    assert updated.scoring_progress["scored"] == 5


def test_failed_from_any_state(store):
    run = store.create()
    store.update(run.run_id, status=RunStatus.FAILED, error="boom")
    updated = store.get(run.run_id)
    assert updated.status == RunStatus.FAILED
    assert updated.error == "boom"


def test_no_transition_out_of_failed(store):
    run = store.create()
    store.update(run.run_id, status=RunStatus.FAILED, error="boom")
    with pytest.raises(ValueError, match="Invalid transition"):
        store.update(run.run_id, status=RunStatus.SCORING)


def test_no_transition_out_of_complete(store):
    run = store.create()
    store.update(run.run_id, status=RunStatus.SCORING)
    store.update(run.run_id, status=RunStatus.JUDGING)
    store.update(run.run_id, status=RunStatus.REFLECTING)
    store.update(run.run_id, status=RunStatus.COMPLETE)
    with pytest.raises(ValueError, match="Invalid transition"):
        store.update(run.run_id, status=RunStatus.SCORING)


def test_create_stores_config_version_and_auto_run(store):
    run = store.create(config_version="cfg-7", auto_run=True)
    assert run.config_version == "cfg-7"
    assert run.auto_run is True
    fetched = store.get(run.run_id)
    assert fetched.config_version == "cfg-7"
    assert fetched.auto_run is True


def test_update_unknown_field_raises(store):
    run = store.create()
    with pytest.raises(ValueError, match="Unknown field"):
        store.update(run.run_id, bogus="nope")


def test_concurrent_create_is_thread_safe(store):
    errors = []

    def _create():
        # Collect exceptions instead of letting them escape a non-main thread,
        # where pytest would never see them and the test would hang or pass
        # silently.
        try:
            store.create()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_create) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(store.list(limit=100)) == 20
    assert len({run.run_id for run in store.list(limit=100)}) == 20


def test_update_nonexistent_raises(store):
    with pytest.raises(ValueError, match="not found"):
        store.update("run-nonexistent", error="boom")


def test_set_selection_nonexistent_raises(store):
    with pytest.raises(ValueError, match="not found"):
        store.set_selection("run-nonexistent", since="2026-07-01")
