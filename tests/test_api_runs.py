"""Tests for the run API endpoints and pipeline execution (spec 09).

Covers the 5 CRUD-ish endpoints from the task brief plus the pipeline
execution logic the brief describes in prose: the advance endpoint's status
transitions and background dispatch, the step functions' data-selection
filtering, and the auto-run cascade in `_execute_run_step`.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import weave_agent_signals.api as api_mod
from weave_agent_signals.models import Score
from weave_agent_signals.reflector import Artifact, Proposal
from weave_agent_signals.runs import DataSelection, Run, RunStatus, RunStore


@pytest.fixture
def client(tmp_path):
    """Swap the module-level _run_store for a tmp_path-backed one.

    The task brief's reference fixture routes through `patch(...)` plus a
    manual `mock_store_ref.__class__` mutation that's immediately discarded
    (it assigns a real RunStore over the mock before any test runs, and the
    trailing manual restore is redundant with what `patch`'s own __exit__
    already does). Swapping the module attribute directly is equivalent and
    clearer.
    """
    original = api_mod._run_store
    api_mod._run_store = RunStore(tmp_path / "runs.db")
    with TestClient(api_mod.app) as tc:
        yield tc
    api_mod._run_store = original


# ---------------------------------------------------------------------------
# Brief's 5 endpoint tests
# ---------------------------------------------------------------------------


def test_create_run(client):
    resp = client.post("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"].startswith("run-")
    assert data["status"] == "created"


def test_list_runs(client):
    client.post("/api/runs")
    client.post("/api/runs")
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert len(resp.json()["runs"]) == 2


def test_get_run(client):
    created = client.post("/api/runs").json()
    resp = client.get(f"/api/runs/{created['run_id']}")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == created["run_id"]


def test_get_missing_run_404(client):
    resp = client.get("/api/runs/run-nonexistent")
    assert resp.status_code == 404


def test_set_selection(client):
    run = client.post("/api/runs").json()
    resp = client.put(
        f"/api/runs/{run['run_id']}/selection",
        json={"since": "2026-07-01", "session_ids": ["conv-abc"]},
    )
    assert resp.status_code == 200
    updated = client.get(f"/api/runs/{run['run_id']}").json()
    assert updated["data_selection"]["since"] == "2026-07-01"


# ---------------------------------------------------------------------------
# _serialize_run
# ---------------------------------------------------------------------------


def test_serialize_run_converts_enum_and_selection_to_plain_types():
    run = Run(
        run_id="run-abc",
        status=RunStatus.SCORING,
        created_at="2026-07-01T00:00:00+00:00",
        data_selection=DataSelection(since="2026-07-01", session_ids=["c1"]),
    )
    d = api_mod._serialize_run(run)
    assert d["status"] == "scoring"
    assert isinstance(d["status"], str)
    assert d["data_selection"] == {
        "since": "2026-07-01",
        "until": None,
        "session_ids": ["c1"],
        "excluded_session_ids": [],
    }


# ---------------------------------------------------------------------------
# advance: guard rails (404 / 400s)
# ---------------------------------------------------------------------------


def test_advance_missing_run_404(client):
    resp = client.post("/api/runs/run-nonexistent/advance")
    assert resp.status_code == 404


def test_advance_without_selection_400(client):
    run = client.post("/api/runs").json()
    resp = client.post(f"/api/runs/{run['run_id']}/advance")
    assert resp.status_code == 400
    assert "selection" in resp.json()["detail"].lower()
    # status must not have moved
    assert client.get(f"/api/runs/{run['run_id']}").json()["status"] == "created"


def test_advance_from_terminal_status_400(client):
    run = client.post("/api/runs").json()
    run_id = run["run_id"]
    # Drive straight to a terminal status via the store, bypassing execution.
    api_mod._run_store.update(run_id, status=RunStatus.FAILED, error="boom")
    resp = client.post(f"/api/runs/{run_id}/advance")
    assert resp.status_code == 400
    assert "cannot advance" in resp.json()["detail"].lower()


def test_advance_concurrent_race_returns_409(client, monkeypatch):
    """Two near-simultaneous advances on the same run: the loser's own
    status-check read is stale by the time its update() runs, since the
    winner's update already landed. RunStore.update() raises ValueError for
    that (its transition check is atomic); the endpoint must turn that into
    a 409, not an unhandled 500."""
    run = client.post("/api/runs").json()
    run_id = run["run_id"]
    api_mod._run_store.set_selection(run_id, since="2026-07-01")

    real_get = api_mod._run_store.get
    calls = {"n": 0}

    def racing_get(rid):
        calls["n"] += 1
        stale = real_get(rid)
        if calls["n"] == 1:
            # Simulate a concurrent request winning the race right after our
            # read but before our update() below runs.
            api_mod._run_store.update(rid, status=RunStatus.SCORING)
        return stale

    monkeypatch.setattr(api_mod._run_store, "get", racing_get)

    resp = client.post(f"/api/runs/{run_id}/advance")
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# advance: dispatch behavior (synchronous status transition + background call)
# ---------------------------------------------------------------------------


def test_advance_transitions_status_and_dispatches_step(client, monkeypatch):
    called = threading.Event()
    captured = {}

    def fake_execute(run_id, step_status, req):
        captured["run_id"] = run_id
        captured["step_status"] = step_status
        captured["req"] = req
        called.set()

    monkeypatch.setattr(api_mod, "_execute_run_step", fake_execute)

    run = client.post("/api/runs").json()
    run_id = run["run_id"]
    client.put(f"/api/runs/{run_id}/selection", json={"since": "2026-07-01"})

    resp = client.post(f"/api/runs/{run_id}/advance")
    assert resp.status_code == 200
    assert resp.json()["status"] == "scoring"

    assert called.wait(timeout=2), "background step was never dispatched"
    assert captured["run_id"] == run_id
    assert captured["step_status"] == RunStatus.SCORING


def test_advance_to_complete_does_not_dispatch(client, monkeypatch):
    """REFLECTING -> COMPLETE is a terminal marker; no step function runs."""
    called = threading.Event()
    monkeypatch.setattr(api_mod, "_execute_run_step", lambda *a, **kw: called.set())

    run = client.post("/api/runs").json()
    run_id = run["run_id"]
    api_mod._run_store.set_selection(run_id, since="2026-07-01")
    api_mod._run_store.update(run_id, status=RunStatus.SCORING)
    api_mod._run_store.update(run_id, status=RunStatus.JUDGING)
    api_mod._run_store.update(run_id, status=RunStatus.REFLECTING)

    resp = client.post(f"/api/runs/{run_id}/advance")
    assert resp.status_code == 200
    assert resp.json()["status"] == "complete"
    assert not called.is_set()


# ---------------------------------------------------------------------------
# _execute_run_step: failure handling + auto-run cascade
# ---------------------------------------------------------------------------


def test_execute_run_step_marks_failed_on_exception(monkeypatch, tmp_path):
    store = RunStore(tmp_path / "runs.db")
    monkeypatch.setattr(api_mod, "_run_store", store)
    run = store.create()
    store.set_selection(run.run_id, since="2026-07-01")
    store.update(run.run_id, status=RunStatus.SCORING)

    def boom(run_id, req):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(api_mod, "_run_scoring_step", boom)

    api_mod._execute_run_step(run.run_id, RunStatus.SCORING, api_mod.AdvanceRequest())

    updated = store.get(run.run_id)
    assert updated.status == RunStatus.FAILED
    assert "kaboom" in updated.error


def test_execute_run_step_auto_run_cascades_to_complete(monkeypatch, tmp_path):
    store = RunStore(tmp_path / "runs.db")
    monkeypatch.setattr(api_mod, "_run_store", store)
    run = store.create(auto_run=True)
    store.set_selection(run.run_id, since="2026-07-01")
    store.update(run.run_id, status=RunStatus.SCORING)

    # Stub out the actual step work — this test is about the state-machine
    # cascade, not scoring/judging/reflecting internals (covered separately).
    monkeypatch.setattr(api_mod, "_run_scoring_step", lambda run_id, req: None)
    monkeypatch.setattr(api_mod, "_run_judging_step", lambda run_id, req: None)
    monkeypatch.setattr(api_mod, "_run_reflecting_step", lambda run_id, req: None)

    api_mod._execute_run_step(run.run_id, RunStatus.SCORING, api_mod.AdvanceRequest())

    assert store.get(run.run_id).status == RunStatus.COMPLETE


def test_execute_run_step_no_auto_run_stops_after_one_step(monkeypatch, tmp_path):
    store = RunStore(tmp_path / "runs.db")
    monkeypatch.setattr(api_mod, "_run_store", store)
    run = store.create(auto_run=False)
    store.set_selection(run.run_id, since="2026-07-01")
    store.update(run.run_id, status=RunStatus.SCORING)

    monkeypatch.setattr(api_mod, "_run_scoring_step", lambda run_id, req: None)

    api_mod._execute_run_step(run.run_id, RunStatus.SCORING, api_mod.AdvanceRequest())

    assert store.get(run.run_id).status == RunStatus.SCORING


# ---------------------------------------------------------------------------
# _select_turns_for_run: data-selection filtering
# ---------------------------------------------------------------------------


def _turn(trace_id, conv_id, started_at, config_version="cfg"):
    t = MagicMock()
    t.trace_id = trace_id
    t.conversation_id = conv_id
    t.started_at = started_at
    t.config_version = config_version
    t.git_branch = "main"
    t.ref_for.return_value = f"weave:///e/p/agent_turn/{trace_id}"
    return t


def _dt(day):
    return datetime(2026, 7, day, 12, 0, tzinfo=timezone.utc)


def test_select_turns_for_run_filters_until_and_sessions():
    turns = [
        _turn("t1", "c1", _dt(1)),
        _turn("t2", "c2", _dt(5)),
        _turn("t3", "c3", _dt(10)),
    ]
    fake_client = MagicMock()
    fake_client.query_turns_paginated.return_value = turns

    selection = DataSelection(since="2026-07-01", until="2026-07-06", session_ids=[])
    result = api_mod._select_turns_for_run(fake_client, selection)

    assert [t.trace_id for t in result] == ["t1", "t2"]  # t3 excluded by `until`
    fake_client.query_turns_paginated.assert_called_once()
    _, kwargs = fake_client.query_turns_paginated.call_args
    assert kwargs["since"] == datetime(2026, 7, 1, tzinfo=timezone.utc)


def test_select_turns_for_run_session_id_whitelist():
    turns = [_turn("t1", "c1", _dt(1)), _turn("t2", "c2", _dt(2)), _turn("t3", "c3", _dt(3))]
    fake_client = MagicMock()
    fake_client.query_turns_paginated.return_value = turns

    selection = DataSelection(session_ids=["c1", "c3"])
    result = api_mod._select_turns_for_run(fake_client, selection)
    assert {t.conversation_id for t in result} == {"c1", "c3"}


def test_select_turns_for_run_excluded_sessions_applied_after_whitelist():
    turns = [_turn("t1", "c1", _dt(1)), _turn("t2", "c2", _dt(2)), _turn("t3", "c3", _dt(3))]
    fake_client = MagicMock()
    fake_client.query_turns_paginated.return_value = turns

    # Broad default (no whitelist), except one session explicitly excluded.
    selection = DataSelection(excluded_session_ids=["c2"])
    result = api_mod._select_turns_for_run(fake_client, selection)
    assert {t.conversation_id for t in result} == {"c1", "c3"}


def test_select_turns_for_run_exclusion_overrides_whitelist_on_conflict():
    """session_ids and excluded_session_ids both set, with overlap: exclusion
    wins. This is the brief's literal wording ("filter to session_ids if set,
    AND exclude excluded_session_ids") applied unconditionally, not an
    either/or choice between the two filters."""
    turns = [_turn("t1", "c1", _dt(1)), _turn("t2", "c2", _dt(2)), _turn("t3", "c3", _dt(3))]
    fake_client = MagicMock()
    fake_client.query_turns_paginated.return_value = turns

    selection = DataSelection(session_ids=["c1", "c2"], excluded_session_ids=["c2"])
    result = api_mod._select_turns_for_run(fake_client, selection)
    assert {t.conversation_id for t in result} == {"c1"}


# ---------------------------------------------------------------------------
# _run_scoring_step / _run_judging_step / _run_reflecting_step
# ---------------------------------------------------------------------------


def _score(name, value=1.0):
    return Score(scorer=name, value=value, tags=[], confidence=1.0, metadata={}, granularity="turn")


def test_run_scoring_step_writes_scores_and_updates_result(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    with patch.object(api_mod, "_run_store", store):
        run = store.create()
        store.set_selection(run.run_id, since="2026-07-01", session_ids=["c1"])

        turn = _turn("t1", "c1", _dt(2))
        fake_client = MagicMock()
        fake_client.query_turns_paginated.return_value = [turn]
        fake_client.query_existing_feedback.return_value = []

        with (
            patch.object(api_mod, "_make_client") as mock_make_client,
            patch.object(api_mod, "score_turn", return_value=[_score("outcome.test")]),
            patch.object(api_mod, "score_session", return_value=[_score("efficiency.session")]),
        ):
            mock_make_client.return_value.__enter__.return_value = fake_client
            api_mod._run_scoring_step(run.run_id, api_mod.AdvanceRequest())

        updated = store.get(run.run_id)
        assert updated.scoring_result["turns_scored"] == 1
        assert updated.scoring_result["sessions_scored"] == 1
        assert updated.scoring_result["scores_written"] == 2  # 1 turn score + 1 session score
        assert updated.scoring_result["errors"] == 0
        assert fake_client.write_score.call_count == 2


def test_run_scoring_step_dry_run_does_not_write(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    with patch.object(api_mod, "_run_store", store):
        run = store.create()
        store.set_selection(run.run_id, since="2026-07-01")

        turn = _turn("t1", "c1", _dt(2))
        fake_client = MagicMock()
        fake_client.query_turns_paginated.return_value = [turn]

        with (
            patch.object(api_mod, "_make_client") as mock_make_client,
            patch.object(api_mod, "score_turn", return_value=[_score("outcome.test")]),
            patch.object(api_mod, "score_session", return_value=[]),
        ):
            mock_make_client.return_value.__enter__.return_value = fake_client
            api_mod._run_scoring_step(run.run_id, api_mod.AdvanceRequest(dry_run=True))

        assert fake_client.write_score.call_count == 0
        assert store.get(run.run_id).scoring_result["dry_run"] is True


def test_run_scoring_step_no_turns_completes_cleanly(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    with patch.object(api_mod, "_run_store", store):
        run = store.create()
        store.set_selection(run.run_id, since="2026-07-01")

        fake_client = MagicMock()
        fake_client.query_turns_paginated.return_value = []

        with patch.object(api_mod, "_make_client") as mock_make_client:
            mock_make_client.return_value.__enter__.return_value = fake_client
            api_mod._run_scoring_step(run.run_id, api_mod.AdvanceRequest())

        result = store.get(run.run_id).scoring_result
        assert result == {
            "turns_scored": 0,
            "sessions_scored": 0,
            "scores_written": 0,
            "errors": 0,
            "dry_run": False,
        }


def test_run_scoring_step_without_selection_raises(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    with patch.object(api_mod, "_run_store", store):
        run = store.create()
        with pytest.raises(ValueError, match="no data selection"):
            api_mod._run_scoring_step(run.run_id, api_mod.AdvanceRequest())


def test_run_judging_step_writes_scores_and_updates_result(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    with patch.object(api_mod, "_run_store", store):
        run = store.create()
        store.set_selection(run.run_id, since="2026-07-01")

        turn = _turn("t1", "c1", _dt(2))
        fake_client = MagicMock()
        fake_client.query_turns_paginated.return_value = [turn]
        fake_client.query_existing_feedback.return_value = []
        fake_judge = MagicMock()

        with (
            patch.object(api_mod, "_make_client") as mock_make_client,
            patch.object(api_mod, "_make_judge_client") as mock_make_judge,
            patch.object(api_mod, "judge_turn", return_value=[_score("judge.verification")]),
            patch.object(api_mod, "judge_session", return_value=[_score("judge.session_outcome")]),
        ):
            mock_make_client.return_value.__enter__.return_value = fake_client
            mock_make_judge.return_value.__enter__.return_value = fake_judge
            api_mod._run_judging_step(run.run_id, api_mod.AdvanceRequest())

        updated = store.get(run.run_id)
        assert updated.judging_result["turns_judged"] == 1
        assert updated.judging_result["scores_written"] == 2
        assert fake_client.write_score.call_count == 2


def test_run_judging_step_no_turns_completes_cleanly(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    with patch.object(api_mod, "_run_store", store):
        run = store.create()
        store.set_selection(run.run_id, since="2026-07-01")

        fake_client = MagicMock()
        fake_client.query_turns_paginated.return_value = []

        with patch.object(api_mod, "_make_client") as mock_make_client:
            mock_make_client.return_value.__enter__.return_value = fake_client
            api_mod._run_judging_step(run.run_id, api_mod.AdvanceRequest())

        result = store.get(run.run_id).judging_result
        assert result == {
            "turns_judged": 0,
            "scores_written": 0,
            "errors": 0,
            "dry_run": False,
        }


def test_run_reflecting_step_filters_feedback_to_run_scope(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    with patch.object(api_mod, "_run_store", store):
        run = store.create()
        store.set_selection(run.run_id, since="2026-07-01", session_ids=["c1"])

        turn = _turn("t1", "c1", _dt(2))
        fake_client = MagicMock()
        fake_client.query_turns_paginated.return_value = [turn]
        in_scope = {
            "weave_ref": "weave:///e/p/agent_turn/t1",
            "feedback_type": "weave_agent_signals.outcome.test",
            "payload": {"rating": 1.0, "tags": [], "details": {}},
        }
        out_of_scope = {
            "weave_ref": "weave:///e/p/agent_turn/other-turn",
            "feedback_type": "weave_agent_signals.outcome.test",
            "payload": {"rating": 0.0, "tags": [], "details": {}},
        }
        fake_client.query_project_feedback.return_value = [in_scope, out_of_scope]

        captured_feedback = {}

        def fake_coaching_digest(feedback):
            captured_feedback["feedback"] = feedback
            return "digest"

        with (
            patch.object(api_mod, "_make_client") as mock_make_client,
            patch.object(api_mod, "coaching_digest", side_effect=fake_coaching_digest),
            patch.object(api_mod, "extract_artifacts", return_value=[]),
        ):
            mock_make_client.return_value.__enter__.return_value = fake_client
            api_mod._run_reflecting_step(run.run_id, api_mod.AdvanceRequest())

        assert captured_feedback["feedback"] == [in_scope]
        result = store.get(run.run_id).reflecting_result
        assert result == {"proposal": None, "reason": "No artifacts found"}


def test_run_reflecting_step_no_feedback_in_scope(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    with patch.object(api_mod, "_run_store", store):
        run = store.create()
        store.set_selection(run.run_id, since="2026-07-01")

        fake_client = MagicMock()
        fake_client.query_turns_paginated.return_value = []
        fake_client.query_project_feedback.return_value = []

        with patch.object(api_mod, "_make_client") as mock_make_client:
            mock_make_client.return_value.__enter__.return_value = fake_client
            api_mod._run_reflecting_step(run.run_id, api_mod.AdvanceRequest())

        result = store.get(run.run_id).reflecting_result
        assert result == {"proposal": None, "reason": "No feedback found"}


def test_run_reflecting_step_dry_run(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    with patch.object(api_mod, "_run_store", store):
        run = store.create()
        store.set_selection(run.run_id, since="2026-07-01")

        turn = _turn("t1", "c1", _dt(2))
        fake_client = MagicMock()
        fake_client.query_turns_paginated.return_value = [turn]
        fb = {
            "weave_ref": "weave:///e/p/agent_turn/t1",
            "feedback_type": "weave_agent_signals.outcome.test",
            "payload": {"rating": 1.0, "tags": [], "details": {}},
        }
        fake_client.query_project_feedback.return_value = [fb]

        with (
            patch.object(api_mod, "_make_client") as mock_make_client,
            patch.object(
                api_mod, "extract_artifacts", return_value=[Artifact("CLAUDE.md", "CLAUDE.md", "x")]
            ),
            patch.object(api_mod, "run_reflection") as mock_reflect,
        ):
            mock_make_client.return_value.__enter__.return_value = fake_client
            api_mod._run_reflecting_step(run.run_id, api_mod.AdvanceRequest(dry_run=True))

        mock_reflect.assert_not_called()
        result = store.get(run.run_id).reflecting_result
        assert result["dry_run"] is True
        assert result["proposal"] is None


def test_run_reflecting_step_happy_path_stores_proposal(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    with patch.object(api_mod, "_run_store", store):
        run = store.create()
        store.set_selection(run.run_id, since="2026-07-01")

        turn = _turn("t1", "c1", _dt(2))
        fake_client = MagicMock()
        fake_client.query_turns_paginated.return_value = [turn]
        fb = {
            "weave_ref": "weave:///e/p/agent_turn/t1",
            "feedback_type": "weave_agent_signals.outcome.test",
            "payload": {"rating": 1.0, "tags": [], "details": {}},
        }
        fake_client.query_project_feedback.return_value = [fb]

        original = Artifact("CLAUDE.md", "CLAUDE.md", "old content")
        proposal = Proposal(
            artifacts=[Artifact("CLAUDE.md", "CLAUDE.md", "new content")],
            rationale="because reasons",
            score_delta=0.2,
        )

        with (
            patch.object(api_mod, "_make_client") as mock_make_client,
            patch.object(api_mod, "extract_artifacts", return_value=[original]),
            patch.object(api_mod, "run_reflection", return_value=proposal),
            patch.object(api_mod, "_make_judge_client") as mock_make_judge,
            patch.object(api_mod, "judge_default_model", return_value="gpt-4o"),
        ):
            mock_make_judge.return_value.__enter__.return_value = MagicMock()
            mock_make_client.return_value.__enter__.return_value = fake_client
            api_mod._run_reflecting_step(run.run_id, api_mod.AdvanceRequest())

        result = store.get(run.run_id).reflecting_result
        assert result["rationale"] == "because reasons"
        assert result["score_delta"] == 0.2
        assert "new content" in result["diff"]
