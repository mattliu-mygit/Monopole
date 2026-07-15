"""Alerting for the `monitor` command.

Turns significant regressions (from patterns.py) into messages and sends them to
a log line and, optionally, a webhook (e.g. a Slack incoming webhook). A small
state file tracks active alert keys so a scheduled run does not repeat an
ongoing regression but can alert again after recovery.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger("weave_agent_signals.monitor")


@dataclass
class Alert:
    key: str  # stable id for one active regression class
    text: str  # human-readable message


def trend_alert(reg: dict) -> Alert:
    """Build an alert from a detect_regressions() entry (drop over time)."""
    return Alert(
        key=f"trend:{reg['scorer']}:{reg['direction']}",
        text=(
            f"trend down: {reg['scorer']} {reg['older_mean']:.2f} -> "
            f"{reg['recent_mean']:.2f} (delta {reg['delta']:+.2f}, n={reg['sample_count']})"
        ),
    )


def config_alert(reg: dict) -> Alert:
    """Build an alert from a detect_config_regressions() entry (worse new config)."""
    return Alert(
        key=f"config:{reg['config']}:{reg['scorer']}",
        text=(
            f"config {reg['config']} worse than {reg['prev_config']}: {reg['scorer']} "
            f"{reg['prev_mean']:.2f} -> {reg['new_mean']:.2f} "
            f"(delta {reg['delta']:+.2f}, n={reg['sample_count']})"
        ),
    )


def send(alerts: list[Alert], *, webhook: str | None = None) -> list[Alert]:
    """Emit each alert to the log and, if configured, POST it to a webhook.

    Returns the alerts that were actually delivered. A failed or rejected webhook
    POST is not counted, so the caller won't dedup (and thus permanently drop) an
    alert that never reached its destination — it retries on the next run.
    """
    delivered: list[Alert] = []
    for a in alerts:
        log.warning("ALERT %s", a.text)
        if webhook:
            try:
                resp = httpx.post(
                    webhook,
                    json={"text": f"[weave-agent-signals] {a.text}"},
                    timeout=10.0,
                )
                resp.raise_for_status()
            except Exception as e:  # a down webhook must not crash the monitor
                log.warning("Alert webhook failed for %s: %s", a.key, e)
                continue
        delivered.append(a)
    return delivered


def load_active(path: str | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):  # ValueError covers JSON + UTF-8 decode
        return set()
    if not isinstance(data, list) or any(not isinstance(item, str) for item in data):
        return set()
    return set(data)


def save_active(path: str, active: set[str]) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sorted(active)))
    except OSError as e:  # a bad state path must not crash the monitor after alerting
        log.warning("Could not persist alert state to %s: %s", path, e)
