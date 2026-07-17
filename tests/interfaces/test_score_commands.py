from __future__ import annotations

import argparse
from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest

from weave_agent_signals import cli
from weave_agent_signals.models import Score, TraceRole


def _score() -> Score:
    return Score(
        scorer="outcome.test",
        value=1.0,
        tags=["verified"],
        metadata={},
        granularity="turn",
    )


def test_write_stats_writes_missing_feedback():
    client = MagicMock()
    client.query_existing_feedback.return_value = []
    stats = cli._WriteStats()

    stats.write_score(client, _score(), "weave:///turn", force=False)

    client.write_score.assert_called_once()
    assert stats.total_scored == 1
    assert stats.total_written == 1
    assert stats.all_tags == ["verified"]


def test_write_stats_skips_existing_feedback_unless_forced():
    client = MagicMock()
    existing = [{"id": "feedback-old"}]
    client.query_existing_feedback.return_value = existing
    stats = cli._WriteStats()

    stats.write_score(client, _score(), "weave:///turn", force=False)

    client.delete_feedback_ids.assert_not_called()
    client.write_score.assert_not_called()
    assert stats.skipped == 1


def test_write_stats_force_creates_before_purging_existing_feedback():
    client = MagicMock()
    existing = [{"id": "feedback-old"}, {"id": "feedback-duplicate"}]
    client.query_existing_feedback.return_value = existing
    stats = cli._WriteStats()
    score = _score()

    stats.write_score(client, score, "weave:///turn", force=True)

    assert client.method_calls == [
        call.query_existing_feedback(
            "weave:///turn",
            "weave_agent_signals.outcome.test",
        ),
        call.write_score(score, "weave:///turn"),
        call.delete_feedback_ids(existing),
    ]
    assert stats.total_written == 1


def test_write_stats_create_failure_does_not_purge_prior_feedback():
    client = MagicMock()
    client.query_existing_feedback.return_value = [
        {"id": "feedback-old"},
        {"id": "feedback-duplicate"},
    ]
    client.write_score.side_effect = RuntimeError("create failed")
    stats = cli._WriteStats()

    with pytest.raises(RuntimeError, match="create failed"):
        stats.write_score(client, _score(), "weave:///turn", force=True)

    assert client.method_calls == [
        call.query_existing_feedback(
            "weave:///turn",
            "weave_agent_signals.outcome.test",
        ),
        call.write_score(_score(), "weave:///turn"),
    ]
    assert stats.total_written == 0


def test_score_and_backfill_return_cleanly_for_empty_queries(monkeypatch, capsys):
    client = MagicMock()
    client.__enter__.return_value.query_turns.return_value = []
    client.__enter__.return_value.query_turns_paginated.return_value = []
    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: client)

    score_rc = cli.cmd_score(
        argparse.Namespace(
            entity="entity",
            project="project",
            since=None,
            limit=10,
            force=False,
        )
    )
    backfill_rc = cli.cmd_backfill(
        argparse.Namespace(
            entity="entity",
            project="project",
            start=None,
            end=None,
            page_size=10,
            force=False,
        )
    )

    assert (score_rc, backfill_rc) == (0, 0)
    assert "No turns found" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["score", "backfill"])
def test_score_commands_exclude_entire_mixed_role_session_before_hydration(
    monkeypatch,
    command,
):
    agent_turn = MagicMock(
        trace_id="agent-turn",
        conversation_id="mixed-session",
        trace_role=TraceRole.AGENT_SESSION,
        started_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    evaluator_turn = MagicMock(
        trace_id="evaluator-turn",
        conversation_id="mixed-session",
        trace_role=TraceRole.SIGNAL_EVALUATION,
        started_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.query_turns.return_value = [agent_turn, evaluator_turn]
    client.query_turns_paginated.return_value = [agent_turn, evaluator_turn]
    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: client)
    monkeypatch.setattr(cli, "score_turn", lambda _turn: pytest.fail("must not score"))
    monkeypatch.setattr(cli, "score_session", lambda _session: pytest.fail("must not score"))

    if command == "score":
        rc = cli.cmd_score(
            argparse.Namespace(
                entity="entity",
                project="project",
                since=None,
                limit=10,
                force=False,
            )
        )
    else:
        rc = cli.cmd_backfill(
            argparse.Namespace(
                entity="entity",
                project="project",
                start=None,
                end=None,
                page_size=10,
                force=False,
            )
        )

    assert rc == 0
    client.hydrate_turn_children.assert_not_called()
    client.write_score.assert_not_called()
