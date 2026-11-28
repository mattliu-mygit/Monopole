"""Provider sign-in from Prompting Time: runs a local CLI's own login command without a terminal.

`sign-in --item <provider>` is a confirmed action. It starts `claude auth login`, `codex login`, or
`devin auth login` with the user's own environment, so the login lands where judging reads it,
and waits until the command exits, the user cancels, or SIGN_IN_TIMEOUT_SECONDS pass. Each CLI
opens the browser and waits for a redirect to a localhost port. `--item codex --device-auth` runs
`codex login --device-auth` instead, which prints a link and a one-time code to enter there, for
when the redirect can't reach this machine. Starting either Codex login deletes the current Codex
login at once, even when the sign-in is then cancelled.

While it waits, it keeps `<state folder>/sign-in.json` current: the provider, the first https link
the CLI printed (the fallback when the browser doesn't open, or the device-code page), the
one-time code a device-code sign-in printed, and whether the CLI asks for a pasted code.
Prompting Time gives an action no stdin, so a pasted code reaches the CLI through a second action:
`sign-in-code` writes `<state folder>/sign-in-code` (mode 600), which the waiting action reads,
deletes, and types into the CLI's stdin. `sign-in-cancel` writes
`<state folder>/sign-in-cancel` the same way, and the waiting action stops the CLI. When the CLI
exits, the action checks every provider again and prints both results.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from weave_agent_signals.plugin import providers
from weave_agent_signals.plugin.confirm import Refused
from weave_agent_signals.plugin.runs import DETAIL_LIMIT, one_line
from weave_agent_signals.plugin.settings import process_alive, state_dir, write_atomic

# Codex's device code lasts 15 minutes; a browser sign-in normally takes under one.
SIGN_IN_TIMEOUT_SECONDS = 600
POLL_SECONDS = 0.25
STOP_GRACE_SECONDS = 5

COMMANDS = {
    "claude": ["claude", "auth", "login"],
    "codex": ["codex", "login"],
    "devin": ["devin", "auth", "login"],
}
# Codex's sign-in with a one-time code entered in any browser, rather than a localhost redirect.
DEVICE_AUTH_COMMAND = ["codex", "login", "--device-auth"]
CODEX_WARNING = "Starting sign-in removes the current Codex login, even if you cancel."

STATE_FILE = "sign-in.json"
CODE_FILE = "sign-in-code"
CANCEL_FILE = "sign-in-cancel"
_LINK = re.compile(r"https://[^\s\"'<>]+")
_ASKS_FOR_CODE = re.compile(r"paste|enter (the )?code", re.IGNORECASE)
# `codex login --device-auth` prints "Enter this one-time code (expires in 15 minutes)" and, on the
# next line, the code, such as BUT6-0M8K4, possibly colored.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_DEVICE_CODE = re.compile(r"one-time code[^\n]*\n\s*([A-Z0-9]+(?:-[A-Z0-9]+)+)\b")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider(provider: str, device_auth: bool = False) -> list[str]:
    if provider not in COMMANDS:
        raise Refused(f"Monopole signs in to {', '.join(COMMANDS)} only, not {provider}.")
    if device_auth:
        if provider != "codex":
            raise Refused("Only Codex signs in with a device code.")
        return DEVICE_AUTH_COMMAND
    return COMMANDS[provider]


def current() -> dict[str, Any] | None:
    """The last sign-in's state, for the page. A waiting sign-in whose action process is gone,
    as when Prompting Time quit, reads as stopped."""

    try:
        found = json.loads((state_dir() / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if found.get("state") == "waiting" and not process_alive(found.get("pid")):
        found = {**found, "state": "stopped", "message": "The sign-in stopped without finishing."}
    return found


def _waiting(provider: str | None = None) -> dict[str, Any] | None:
    found = current()
    if found is None or found["state"] != "waiting":
        return None
    if provider is not None and found["provider"] != provider:
        return None
    return found


def sign_in_request(provider: str, device_auth: bool = False) -> str:
    argv = _provider(provider, device_auth)
    request = {
        "action": "sign-in",
        "provider": providers.PROVIDER_TITLES[provider],
        "runs": " ".join(argv),
        "waits": f"until the browser sign-in finishes, at most {SIGN_IN_TIMEOUT_SECONDS // 60} min",
    }
    if provider == "codex":
        request["warning"] = CODEX_WARNING
    return json.dumps(request, indent=2, ensure_ascii=False) + "\n"


def _write_state(state: dict[str, Any]) -> None:
    write_atomic(state_dir() / STATE_FILE, json.dumps(state, indent=2) + "\n")


def _stop(process: subprocess.Popen) -> None:
    """SIGTERM to the CLI's process group, then SIGKILL after the grace period."""

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(STOP_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            continue


def _read_output(process: subprocess.Popen, chunks: list[str]) -> None:
    """Collect the CLI's output as it comes, including a prompt with no newline after it."""

    fd = process.stdout.fileno()
    while True:
        data = os.read(fd, 4096)
        if not data:
            return
        chunks.append(data.decode(errors="replace"))


def _take(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    path.unlink(missing_ok=True)
    return text


def sign_in(
    provider: str, *, device_auth: bool = False, timeout: float = SIGN_IN_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Run the provider's login command to its end, then check every provider again."""

    argv = _provider(provider, device_auth)
    if _waiting() is not None:
        raise Refused("Another sign-in is waiting; finish or cancel it first.")
    env = {**os.environ, "WEAVE_AGENT_ADAPTER_DISABLE": "1"}
    program = shutil.which(argv[0], path=env.get("PATH"))
    if program is None:
        raise Refused(f"{argv[0]} is not on PATH, so Monopole can't start its sign-in.")
    folder = state_dir()
    for name in (CODE_FILE, CANCEL_FILE):
        (folder / name).unlink(missing_ok=True)

    state: dict[str, Any] = {
        "format": 1,
        "provider": provider,
        "state": "waiting",
        "started_at": _now(),
        "finished_at": None,
        "pid": os.getpid(),
        "link": None,
        "device_code": None,
        "asks_for_code": False,
        "message": None,
    }
    _write_state(state)
    process = subprocess.Popen(
        [program, *argv[1:]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=tempfile.gettempdir(),
        start_new_session=True,
    )
    chunks: list[str] = []
    reader = threading.Thread(target=_read_output, args=(process, chunks), daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    outcome = None
    try:
        while process.poll() is None:
            output = _ANSI.sub("", "".join(chunks))
            link = state["link"] or next(iter(_LINK.findall(output)), None)
            asks = state["asks_for_code"] or bool(_ASKS_FOR_CODE.search(output))
            device = _DEVICE_CODE.search(output) if device_auth else None
            device_code = state["device_code"] or (device.group(1) if device else None)
            found = (link, asks, device_code)
            if found != (state["link"], state["asks_for_code"], state["device_code"]):
                state.update(link=link, asks_for_code=asks, device_code=device_code)
                _write_state(state)
            code = _take(folder / CODE_FILE)
            if code is not None:
                try:
                    process.stdin.write(code.strip().encode() + b"\n")
                    process.stdin.flush()
                except BrokenPipeError:
                    pass
            if _take(folder / CANCEL_FILE) is not None:
                outcome = ("cancelled", "You cancelled the sign-in.")
                _stop(process)
                break
            if time.monotonic() > deadline:
                outcome = ("timed_out", f"The sign-in didn't finish in {int(timeout)} s.")
                _stop(process)
                break
            time.sleep(POLL_SECONDS)
    finally:
        if process.poll() is None:
            _stop(process)
        reader.join(STOP_GRACE_SECONDS)
    if outcome is None:
        last = next((line for line in reversed("".join(chunks).splitlines()) if line.strip()), None)
        if process.returncode == 0:
            outcome = ("signed_in", one_line(last, DETAIL_LIMIT))
        else:
            message = f"{' '.join(argv)} exited with {process.returncode}: {last or ''}"
            outcome = ("failed", one_line(message, DETAIL_LIMIT))
    state.update(state=outcome[0], message=outcome[1], finished_at=_now())
    _write_state(state)
    for name in (CODE_FILE, CANCEL_FILE):
        (folder / name).unlink(missing_ok=True)
    check = providers.check_providers()
    return {"format": 1, "sign_in": state, "providers": check}


def send_code(provider: str, code: str) -> dict[str, Any]:
    """Hand a pasted code to the waiting sign-in, which types it into the CLI."""

    _provider(provider)
    code = code.strip()
    if not code or any(char.isspace() for char in code):
        raise Refused("Paste the code as one word, with no spaces.")
    if _waiting(provider) is None:
        raise Refused(f"No {providers.PROVIDER_TITLES[provider]} sign-in is waiting for a code.")
    # write_atomic's temporary file is created with mode 600 and keeps it.
    write_atomic(state_dir() / CODE_FILE, code + "\n")
    return {"format": 1, "sent": True}


def cancel(provider: str) -> dict[str, Any]:
    _provider(provider)
    if _waiting(provider) is None:
        raise Refused(f"No {providers.PROVIDER_TITLES[provider]} sign-in is waiting.")
    write_atomic(state_dir() / CANCEL_FILE, "cancel\n")
    return {"format": 1, "cancelling": True}
