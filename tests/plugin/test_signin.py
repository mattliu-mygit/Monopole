"""Sign-in runs fake `claude`, `codex`, and `devin` scripts from a temporary PATH. No real login
command runs: starting a real `codex login` deletes the user's Codex login."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time

import pytest

from weave_agent_signals.cli import main
from weave_agent_signals.plugin import providers, signin
from weave_agent_signals.plugin.settings import state_dir

CHECKED = {"format": 1, "checked_at": "2026-11-26T18:00:00+00:00", "providers": []}

FAKE_CLAUDE = """#!/bin/sh
echo "Opening browser to sign in..."
echo "If the browser didn't open, visit: https://claude.com/cai/oauth/authorize?code=true&state=abc"
printf "Paste code here if prompted > "
read code
echo "$code" > "$FAKE_OUT"
echo "Login successful."
"""
FAKE_CODEX_FAILS = """#!/bin/sh
echo "Starting local login server on http://localhost:1455"
echo "Error: login was not completed" >&2
exit 1
"""
# What codex login --device-auth prints, colored as Codex colors it; it exits once $FAKE_OUT exists,
# as the real one does once the code is entered in the browser.
FAKE_CODEX_DEVICE = """#!/bin/sh
[ "$*" = "login --device-auth" ] || exit 2
echo "Follow these steps to sign in with ChatGPT using device code authorization:"
echo
echo "1. Open this link in your browser and sign in to your account"
printf "   \\033[94mhttps://auth.openai.com/codex/device\\033[0m\\n"
echo
echo "2. Enter this one-time code \\033[90m(expires in 15 minutes)\\033[0m"
printf "   \\033[94mBUT6-0M8K4\\033[0m\\n"
while [ ! -e "$FAKE_OUT" ]; do sleep 0.02; done
echo "Successfully logged in"
"""
FAKE_WAITS = """#!/bin/sh
echo "Waiting for the browser..."
sleep 30
"""


@pytest.fixture
def clis(monkeypatch, state, tmp_path):
    """A PATH whose claude, codex, and devin are the fake scripts, a temporary HOME, and a
    provider check that runs nothing."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {"claude": FAKE_CLAUDE, "codex": FAKE_CODEX_FAILS, "devin": FAKE_WAITS}
    for name, text in scripts.items():
        path = bin_dir / name
        path.write_text(text)
        path.chmod(0o755)
    path = f"{bin_dir}:/usr/bin:/bin"
    monkeypatch.setenv("PATH", path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FAKE_OUT", str(tmp_path / "typed-code"))
    for name in scripts:
        # Guard: the real CLIs must not be reachable.
        assert shutil.which(name, path=path) == str(bin_dir / name)
    checks = []
    monkeypatch.setattr(providers, "check_providers", lambda: checks.append(1) or CHECKED)
    monkeypatch.setattr(signin, "POLL_SECONDS", 0.02)
    monkeypatch.setattr(signin, "STOP_GRACE_SECONDS", 2)
    return tmp_path, checks


def _wait_for(condition, seconds: float = 5) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.02)


def _start(provider: str, results: list) -> threading.Thread:
    thread = threading.Thread(target=lambda: results.append(signin.sign_in(provider)))
    thread.start()
    _wait_for(lambda: (signin.current() or {}).get("state") == "waiting")
    return thread


def test_codex_sign_in_confirms_with_the_warning_and_print_runs_nothing(clis, capsys):
    assert main(["plugin", "sign-in", "--item", "codex", "--print"]) == 0

    out = capsys.readouterr().out
    fingerprint = re.fullmatch(r"fingerprint: ([0-9a-f]{64})", out.strip().splitlines()[-1])
    assert fingerprint
    request = json.loads((state_dir() / "sign-in-request.json").read_text())
    assert request["runs"] == "codex login"
    assert request["warning"] == signin.CODEX_WARNING
    assert signin.current() is None
    assert clis[1] == []


def test_claude_sign_in_shows_the_link_takes_a_pasted_code_and_rechecks_providers(clis, capsys):
    tmp_path, checks = clis
    results: list = []
    thread = _start("claude", results)
    _wait_for(lambda: signin.current()["asks_for_code"])
    waiting = signin.current()
    assert waiting["link"] == "https://claude.com/cai/oauth/authorize?code=true&state=abc"

    assert main(["plugin", "sign-in-code", "--item", "claude", "--code", "abc123"]) == 0
    thread.join(10)

    assert (tmp_path / "typed-code").read_text() == "abc123\n"
    assert not (state_dir() / signin.CODE_FILE).exists()
    finished = results[0]
    assert finished["sign_in"]["state"] == "signed_in"
    assert finished["sign_in"]["message"].endswith("Login successful.")
    assert finished["providers"] == CHECKED and checks == [1]
    assert json.loads((state_dir() / "sign-in.json").read_text())["state"] == "signed_in"


def test_a_login_that_exits_non_zero_reads_as_failed_with_its_last_line(clis, capsys):
    assert main(["plugin", "sign-in", "--item", "codex", "--print"]) == 0
    fingerprint = capsys.readouterr().out.strip().splitlines()[-1].split()[-1]

    assert main(["plugin", "sign-in", "--item", "codex", "--expect", fingerprint]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["sign_in"]["state"] == "failed"
    assert printed["sign_in"]["message"] == (
        "codex login exited with 1: Error: login was not completed"
    )


def test_cancel_stops_the_waiting_login(clis, capsys):
    results: list = []
    thread = _start("devin", results)

    assert main(["plugin", "sign-in-cancel", "--item", "devin"]) == 0
    thread.join(10)

    assert not thread.is_alive()
    assert results[0]["sign_in"]["state"] == "cancelled"


def test_a_login_that_never_finishes_stops_at_the_timeout(clis):
    finished = signin.sign_in("devin", timeout=0.3)

    assert finished["sign_in"]["state"] == "timed_out"


def test_a_code_or_cancel_needs_a_waiting_sign_in_for_that_provider(clis, capsys):
    assert main(["plugin", "sign-in-code", "--item", "claude", "--code", "abc"]) == 1
    assert "No Claude sign-in is waiting for a code" in capsys.readouterr().err
    assert main(["plugin", "sign-in-cancel", "--item", "codex"]) == 1
    assert not (state_dir() / signin.CODE_FILE).exists()


def test_a_sign_in_whose_process_is_gone_reads_as_stopped(clis):
    state = {"provider": "claude", "state": "waiting", "pid": 999_999_999}
    (state_dir()).mkdir(parents=True, exist_ok=True)
    (state_dir() / "sign-in.json").write_text(json.dumps(state))

    assert signin.current()["state"] == "stopped"


def test_codex_device_code_sign_in_shows_the_link_and_the_code(clis, capsys):
    tmp_path, _ = clis
    (tmp_path / "bin" / "codex").write_text(FAKE_CODEX_DEVICE)
    assert main(["plugin", "sign-in", "--item", "codex", "--device-auth", "--print"]) == 0
    request = json.loads((state_dir() / "sign-in-request.json").read_text())
    assert request["runs"] == "codex login --device-auth"
    assert request["warning"] == signin.CODEX_WARNING

    results: list = []
    thread = threading.Thread(
        target=lambda: results.append(signin.sign_in("codex", device_auth=True))
    )
    thread.start()
    _wait_for(lambda: (signin.current() or {}).get("device_code"))
    waiting = signin.current()
    assert (waiting["link"], waiting["device_code"]) == (
        "https://auth.openai.com/codex/device",
        "BUT6-0M8K4",
    )
    assert waiting["asks_for_code"] is False

    (tmp_path / "typed-code").write_text("entered in the browser")
    thread.join(10)
    assert results[0]["sign_in"]["state"] == "signed_in"


def test_only_codex_signs_in_with_a_device_code(clis, capsys):
    assert main(["plugin", "sign-in", "--item", "claude", "--device-auth", "--print"]) == 1
    assert "Only Codex signs in with a device code" in capsys.readouterr().err
