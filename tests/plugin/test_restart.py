"""Restarting Monopole's server after a new OpenAI key: monopole-run's SIGUSR1, run against a fake
server command, and the rule that finds the lock holder and spares a run mid-stage."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tests.plugin.fixtures import add_run, started
from weave_agent_signals.plugin import server
from weave_agent_signals.plugin.settings import state_dir

ROOT = Path(__file__).resolve().parents[2]
CREATED_AT = "2026-11-05T22:00:00+00:00"
# Stands in for bin/monopole: records each start and stop, and stops on SIGTERM as uv does.
FAKE_SERVER = """#!/usr/bin/env bash
echo "start $$" >> "$SERVER_LOG"
trap 'echo "stop $$" >> "$SERVER_LOG"; exit 0' TERM
while true; do sleep 0.05; done
"""


def _wait_for(check, seconds: float = 10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        found = check()
        if found:
            return found
        time.sleep(0.05)
    raise AssertionError("timed out")


def _log_lines(log: Path) -> list[str]:
    return log.read_text().split("\n")[:-1] if log.exists() else []


def test_monopole_run_restarts_its_server_on_sigusr1_and_keeps_the_lock(tmp_path):
    folder = tmp_path / "plugin" / "bin"
    folder.mkdir(parents=True)
    shutil.copy(ROOT / "bin" / "monopole-run", folder / "monopole-run")
    (folder / "monopole").write_text(FAKE_SERVER)
    (folder / "monopole").chmod(0o755)
    log = tmp_path / "server.log"
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "SERVER_LOG": str(log),
    }
    run = subprocess.Popen(["bash", str(folder / "monopole-run")], env=environment)
    try:
        lock_pid = tmp_path / "state" / "monopole" / "run.lock" / "pid"
        _wait_for(lambda: len(_log_lines(log)) == 1 and lock_pid.exists())
        assert server.lock_holder(tmp_path / "state" / "monopole") == run.pid

        run.send_signal(signal.SIGUSR1)

        lines = _wait_for(lambda: len(_log_lines(log)) == 3 and _log_lines(log))
        first, second = lines[0].split()[1], lines[2].split()[1]
        # The old server stopped before the new one started.
        assert lines == [f"start {first}", f"stop {first}", f"start {second}"]
        assert first != second
        assert run.poll() is None
        assert lock_pid.read_text().strip() == str(run.pid)
    finally:
        run.terminate()
        run.wait(10)
    # monopole-run's exit stops the server; the fake logs it after its current sleep.
    _wait_for(lambda: _log_lines(log)[-1] == f"stop {second}")
    assert not lock_pid.parent.exists()


def test_lock_holder_ignores_a_pid_file_naming_another_program(state):
    lock = state_dir() / "run.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(f"{os.getpid()}\n")

    assert server.lock_holder(state_dir()) is None


@pytest.fixture
def signals(monkeypatch):
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(server, "lock_holder", lambda _state: 4242)
    monkeypatch.setattr(server.os, "kill", lambda pid, number: sent.append((pid, number)))
    return sent


def test_a_new_key_restarts_the_lock_holder(state, signals):
    # A run waiting for Continue isn't mid-stage.
    add_run(state / "runs.db", "run-1", "scoring", CREATED_AT, **started(), auto_run=0)

    said = server.restart_for_new_key(state / "runs.db")

    assert signals == [(4242, signal.SIGUSR1)]
    assert said == "Restarted Monopole's server so it uses the key."


def test_a_new_key_does_not_restart_the_server_while_a_run_is_mid_stage(state, signals):
    add_run(state / "runs.db", "run-1", "judging", CREATED_AT, **started(), auto_run=1)

    said = server.restart_for_new_key(state / "runs.db")

    assert signals == []
    assert "restart Monopole in Prompting Time after the current run" in said


def test_a_new_key_says_when_no_server_runs(state, monkeypatch):
    monkeypatch.setattr(server.os, "kill", lambda *_: pytest.fail("signalled"))

    said = server.restart_for_new_key(state / "runs.db")

    assert said == "Monopole's server isn't running; it uses the key when it starts."
