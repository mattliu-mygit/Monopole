"""Tests for the monitor command: alert on a real regression, then dedup."""

from __future__ import annotations

import argparse

from weave_agent_signals import cli
from weave_agent_signals.patterns import ConfigStats, Stat


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
    stats = ConfigStats(
        {
            "v_old": {"outcome.test": Stat(6, 1.0, True)},
            "v_new": {"outcome.test": Stat(6, 0.0, True)},
        },
        {"v_old": 6, "v_new": 6},
    )
    return stats, old + new


def _improving_feedback():
    # single config, scores rise over time → a significant IMPROVEMENT, not a drop
    old = [_fb("outcome.test", 0.0, "v1", "2026-07-01T00:00:00") for _ in range(6)]
    new = [_fb("outcome.test", 1.0, "v1", "2026-07-09T00:00:00") for _ in range(6)]
    stats = ConfigStats({"v1": {"outcome.test": Stat(12, 0.5, True)}}, {"v1": 12})
    return stats, old + new


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


def test_monitor_alerts_on_significant_regression_then_dedups(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli.ces_commands, "read_analysis", lambda _args, feedback, **_: _regressing_feedback()
    )

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


def test_monitor_realerts_when_a_regression_recurs_after_recovery(
    tmp_path,
    monkeypatch,
):
    feeds = iter(
        [
            _regressing_feedback(),
            _improving_feedback(),
            _regressing_feedback(),
        ]
    )
    monkeypatch.setattr(cli.ces_commands, "read_analysis", lambda _args, feedback, **_: next(feeds))

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


def test_monitor_does_not_alert_on_improvement(monkeypatch):
    monkeypatch.setattr(
        cli.ces_commands, "read_analysis", lambda _args, feedback, **_: _improving_feedback()
    )
    sent, fake_send = _capturing_send()
    monkeypatch.setattr(cli.alerts, "send", fake_send)

    rc = cli.cmd_monitor(_args())
    assert rc == 0
    assert sent == []  # a significant improvement is not a regression


def test_monitor_alerts_on_a_drop_whose_config_intervals_overlap(monkeypatch):
    """Behavior change: 40/50 then 30/50 passing. The configs' own 95% intervals overlap, which
    the old rule read as no regression; the interval on the difference lies below 0."""
    feedback = [
        _fb("outcome.test", 1.0, "v_old", "2026-07-01T00:00:00"),
        _fb("outcome.test", 1.0, "v_new", "2026-07-09T00:00:00"),
    ]
    stats = ConfigStats(
        {
            "v_old": {"outcome.test": Stat(50, 0.8, True)},
            "v_new": {"outcome.test": Stat(50, 0.6, True)},
        },
        {"v_old": 50, "v_new": 50},
    )
    monkeypatch.setattr(cli.ces_commands, "read_analysis", lambda _args, **_: (stats, feedback))
    sent, fake_send = _capturing_send()
    monkeypatch.setattr(cli.alerts, "send", fake_send)

    assert cli.cmd_monitor(_args()) == 0
    assert [alert.key for alert in sent] == ["config:v_new:outcome.test"]
    assert "95% CI -0.36 to -0.02" in sent[0].text
