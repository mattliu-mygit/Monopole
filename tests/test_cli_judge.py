"""Regression tests for cmd_judge orchestration (hydration, backend wiring)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from weave_agent_signals import cli


def _fake_turn(tid, conv="c1"):
    t = MagicMock()
    t.trace_id = tid
    t.conversation_id = conv
    t.started_at = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    t.model = "claude-opus-4-8"
    t.config_version = "cfg"
    t.git_branch = "main"
    t.ref_for.return_value = "weave:///ref"
    return t


def _args(**over):
    base = dict(
        entity="e",
        project="p",
        since=None,
        limit=10,
        rubric=None,
        judge_backend="cli",
        panel_size=1,
        dry_run=True,
        force=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


@patch("weave_agent_signals.cli._make_judge_client")
@patch("weave_agent_signals.cli.judge_session", return_value=[])
@patch("weave_agent_signals.cli.WeaveClient")
def test_session_only_judge_still_hydrates_turns(mock_wc, mock_js, mock_mjc):
    """Session digests are built from turn children, so a session-only rubric
    must hydrate turns even though the turn-judging loop is skipped."""
    client = MagicMock()
    turns = [_fake_turn("t1"), _fake_turn("t2")]
    client.query_turns.return_value = turns
    mock_wc.return_value.__enter__.return_value = client
    mock_mjc.return_value.__enter__.return_value = MagicMock()

    rc = cli.cmd_judge(_args(rubric="judge.session_outcome"))

    assert rc == 0
    # both turns hydrated despite only a session rubric being requested
    assert client.hydrate_turn_children.call_count == 2
    mock_js.assert_called()


@patch("weave_agent_signals.cli._make_judge_client")
@patch("weave_agent_signals.cli.judge_turn", return_value=[])
@patch("weave_agent_signals.cli.judge_session", return_value=[])
@patch("weave_agent_signals.cli.WeaveClient")
def test_default_judge_hydrates_once_per_turn(mock_wc, mock_js, mock_jt, mock_mjc):
    client = MagicMock()
    turns = [_fake_turn("t1"), _fake_turn("t2")]
    client.query_turns.return_value = turns
    mock_wc.return_value.__enter__.return_value = client
    mock_mjc.return_value.__enter__.return_value = MagicMock()

    cli.cmd_judge(_args(rubric=None))

    # exactly one hydration per turn (no double-hydrate now that it's hoisted)
    assert client.hydrate_turn_children.call_count == 2
