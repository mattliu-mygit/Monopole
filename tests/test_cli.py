from __future__ import annotations

import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "weave_agent_signals.cli", "--help"],
        capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": ""},
    )
    assert result.returncode == 0
    assert "score" in result.stdout
    assert "backfill" in result.stdout
    assert "inspect" in result.stdout


def test_score_help():
    result = subprocess.run(
        [sys.executable, "-m", "weave_agent_signals.cli", "score", "--help"],
        capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": ""},
    )
    assert result.returncode == 0
    assert "--since" in result.stdout
    assert "--dry-run" in result.stdout


def test_backfill_help():
    result = subprocess.run(
        [sys.executable, "-m", "weave_agent_signals.cli", "backfill", "--help"],
        capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": ""},
    )
    assert result.returncode == 0
    assert "--from" in result.stdout or "--start" in result.stdout
