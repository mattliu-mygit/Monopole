from __future__ import annotations

import argparse
from unittest.mock import MagicMock, call

import pytest

from weave_agent_signals import cli
from weave_agent_signals.models import Score


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
