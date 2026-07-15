"""Tests for the monitor command: alert on a real regression, then dedup."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

from weave_agent_signals import cli


def _fb(scorer, rating, config, scored_at):
    return {
        "feedback_type": f"weave_agent_signals.{scorer}",
        "payload": {
            "rating": rating,
            "tags": [],
            "details": {"config_version": config, "scored_at": scored_at},
        },
    }


def _regressing_feedback():
    # old config scored 1.0; newer config scored 0.0 → significant config regression
    old = [_fb("outcome.test", 1.0, "v_old", "2026-07-01T00:00:00") for _ in range(6)]
    new = [_fb("outcome.test", 0.0, "v_new", "2026-07-09T00:00:00") for _ in range(6)]
    return old + new


def _improving_feedback():
    # single config, scores rise over time → a significant IMPROVEMENT, not a drop
    old = [_fb("outcome.test", 0.0, "v1", "2026-07-01T00:00:00") for _ in range(6)]
    new = [_fb("outcome.test", 1.0, "v1", "2026-07-09T00:00:00") for _ in range(6)]
    return old + new


def _capturing_send():
    sent = []

    def fake_send(found, webhook=None):
        sent.extend(found)
        return found  # cmd_monitor dedups on what send() reports delivered

    return sent, fake_send


def _args(**over):
    base = dict(
        entity="e",
        project="p",
        limit=1000,
        alert_webhook=None,
        state_file=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


@patch("weave_agent_signals.cli.WeaveClient")
def test_monitor_alerts_on_significant_regression_then_dedups(mock_wc, tmp_path, monkeypatch):
    client = MagicMock()
    client.query_project_feedback.return_value = _regressing_feedback()
    mock_wc.return_value.__enter__.return_value = client

    sent, fake_send = _capturing_send()
    monkeypatch.setattr(cli.alerts, "send", fake_send)

    state = str(tmp_path / "seen.json")
    rc1 = cli.cmd_monitor(_args(state_file=state))
    assert rc1 == 0
    # the drop is both a trend-down and a config regression → both fire, once each
    keys = {a.key for a in sent}
    assert keys == {"trend:outcome.test:regression", "config:v_new:outcome.test"}

    sent.clear()
    rc2 = cli.cmd_monitor(_args(state_file=state))
    assert rc2 == 0
    assert sent == []  # same regressions deduped on rerun


@patch("weave_agent_signals.cli.WeaveClient")
def test_monitor_realerts_when_a_regression_recurs_after_recovery(
    mock_wc,
    tmp_path,
    monkeypatch,
):
    client = MagicMock()
    client.query_project_feedback.side_effect = [
        _regressing_feedback(),
        _improving_feedback(),
        _regressing_feedback(),
    ]
    mock_wc.return_value.__enter__.return_value = client

    sent, fake_send = _capturing_send()
    monkeypatch.setattr(cli.alerts, "send", fake_send)
    state = str(tmp_path / "active.json")

    assert cli.cmd_monitor(_args(state_file=state)) == 0
    assert len(sent) == 2

    sent.clear()
    assert cli.cmd_monitor(_args(state_file=state)) == 0
    assert sent == []
    assert cli.alerts.load_active(state) == set()

    assert cli.cmd_monitor(_args(state_file=state)) == 0
    assert {alert.key for alert in sent} == {
        "trend:outcome.test:regression",
        "config:v_new:outcome.test",
    }


@patch("weave_agent_signals.cli.WeaveClient")
def test_monitor_does_not_alert_on_improvement(mock_wc, monkeypatch):
    client = MagicMock()
    client.query_project_feedback.return_value = _improving_feedback()
    mock_wc.return_value.__enter__.return_value = client
    sent, fake_send = _capturing_send()
    monkeypatch.setattr(cli.alerts, "send", fake_send)

    rc = cli.cmd_monitor(_args())
    assert rc == 0
    assert sent == []  # a significant improvement is not a regression
