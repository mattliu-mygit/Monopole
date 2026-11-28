"""API keys for W&B Inference and OpenAI, saved from Prompting Time.

Prompting Time gives an action only its arguments, no stdin or environment, so the page hands a key
over in two calls that keep it out of the process list for all but a moment:

1. `stage-key --item wandb|openai --key <key>` writes the key, unchecked, to
   `<state folder>/staged-<provider>-key` (mode 600) and exits, about 0.2 s after it started.
2. `set-key --item wandb|openai` takes that file, deleting it at once, checks the key with
   `GET /v1/models`, which spends nothing, and saves it only when the provider accepts it.

A held key older than five minutes, left by a hand-off the page abandoned, is deleted by the next
`set-key` or `status` and never used. From a terminal, `set-key` with nothing staged reads the key
from stdin instead. Nothing prints or logs the key. A real fix needs Prompting Time to pass a
secret on stdin.

- W&B goes where Monopole already reads it: the `api.wandb.ai` entry of `~/.netrc`. Only that
  entry's password changes; the file is rewritten whole through a temporary file with mode 600.
  `WANDB_API_KEY`, when set, still wins over the file.
- OpenAI goes to `<state folder>/openai-key` (mode 600). Every plugin action, including the start
  command's server, exports it as OPENAI_API_KEY, over `.env` and the environment. Saving it
  restarts Monopole's server, unless a run is mid-stage, so the server reads it.
"""

from __future__ import annotations

import getpass
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from weave_agent_signals import ces_commands
from weave_agent_signals.judges.inference import INFERENCE_BASE, OPENAI_BASE
from weave_agent_signals.plugin import providers, server
from weave_agent_signals.plugin.confirm import Refused
from weave_agent_signals.plugin.settings import run_db_path, state_dir, write_atomic

KEY_PROVIDERS = ("wandb", "openai")
NETRC_MACHINE = "api.wandb.ai"
OPENAI_KEY_FILE = "openai-key"
CHECK_TIMEOUT_SECONDS = 10
# A held key older than this belongs to a hand-off the page abandoned; the next `set-key` or
# `status` deletes it.
STAGED_KEY_MAX_AGE_SECONDS = 300

# How Python's netrc module, which Monopole and wandb read ~/.netrc with, reads the file: tokens
# split on whitespace; `machine`, `default`, and `macdef` start an entry only in keyword position,
# never as the value of `login` (or `user`), `account`, or `password`; a `#` in keyword position
# comments out the rest of its line; and a macro's body runs from the line after its name to the
# next blank line.
_TOKEN = re.compile(r"\S+")
_VALUE_KEYWORDS = frozenset({"login", "user", "account", "password"})


def read_key(text: str) -> str:
    key = text.strip()
    if not key:
        raise Refused("No key given; nothing saved.")
    if any(char.isspace() for char in key):
        raise Refused("A key has no spaces; nothing saved.")
    return key


def _staged_path(provider: str) -> Path:
    return state_dir() / f"staged-{provider}-key"


def stage_key(provider: str, key: str) -> dict[str, Any]:
    """Hold the key, unchecked, for the `set-key` call that follows, replacing any key held
    before."""

    # mkstemp, under write_atomic, creates the file with mode 600.
    write_atomic(_staged_path(provider), read_key(key) + "\n")
    return {"format": 1, "staged": provider}


def discard_stale_staged_keys() -> None:
    """Delete every held key older than STAGED_KEY_MAX_AGE_SECONDS."""

    for path in state_dir().glob("staged-*-key"):
        try:
            if time.time() - path.stat().st_mtime > STAGED_KEY_MAX_AGE_SECONDS:
                path.unlink()
        except FileNotFoundError:
            pass


def take_key(provider: str) -> str:
    """The key `stage-key` held for `provider` within STAGED_KEY_MAX_AGE_SECONDS, deleted now so
    it never outlives this call; with none held, the key typed or piped on stdin."""

    discard_stale_staged_keys()
    path = _staged_path(provider)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return read_key(getpass.getpass("Key: ") if sys.stdin.isatty() else sys.stdin.read())
    path.unlink(missing_ok=True)
    return read_key(text)


def _check(provider: str, key: str, client: httpx.Client) -> None:
    if provider == "wandb":
        entity, project = ces_commands.inference_project(
            os.environ.get("WANDB_ENTITY"), os.environ.get("WANDB_PROJECT")
        )
        url = f"{INFERENCE_BASE}/models"
        headers = {"Authorization": f"Bearer {key}", "OpenAI-Project": f"{entity}/{project}"}
    else:
        url = f"{OPENAI_BASE}/models"
        headers = {"Authorization": f"Bearer {key}"}
    try:
        response = client.get(url, headers=headers, timeout=CHECK_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise Refused(f"GET {url} failed ({type(exc).__name__}); nothing saved.") from exc
    if response.status_code in (401, 403):
        raise Refused(f"{url} refused the key (HTTP {response.status_code}); nothing saved.")
    if response.status_code != 200:
        raise Refused(f"GET {url} answered HTTP {response.status_code}; nothing saved.")


_Pair = tuple[re.Match[str], re.Match[str] | None]


def _netrc_entries(text: str) -> list[list[_Pair]]:
    """Each `machine` or `default` entry of a ~/.netrc's text as (keyword, value) token pairs,
    starting with `machine` and its host, or `default` and None."""

    entries: list[list[_Pair]] = []
    pending: re.Match[str] | None = None
    position = 0
    while match := _TOKEN.search(text, position):
        position = match.end()
        token = match.group()
        if pending is not None:
            if entries:
                entries[-1].append((pending, match))
            pending = None
        elif token.startswith("#"):
            line_end = text.find("\n", match.start())
            position = len(text) if line_end == -1 else line_end
        elif token == "macdef":
            name = _TOKEN.search(text, position)
            name_line_end = text.find("\n", name.end()) if name else -1
            body_end = -1 if name_line_end == -1 else text.find("\n\n", name_line_end)
            position = len(text) if body_end == -1 else body_end + 2
        elif token in ("machine", "default"):
            entries.append([])
            if token == "machine":
                pending = match
            else:
                entries[-1].append((match, None))
        elif token in _VALUE_KEYWORDS:
            pending = match
        elif entries:
            entries[-1].append((match, None))
    if pending is not None and entries:
        entries[-1].append((pending, None))
    return entries


def netrc_with_key(text: str, machine: str, password: str) -> str:
    """`text` with `machine`'s entry holding `password`: only the password's value changes, and
    every other byte, including the entry's other tokens and comments, stays. With several
    entries for `machine`, the last, which netrc readers use, changes. An entry without a password
    gets one, and a login "user" when it has none; a file without the entry gets it appended."""

    found = [
        entry
        for entry in _netrc_entries(text)
        if entry[0][0].group() == "machine" and entry[0][1] and entry[0][1].group() == machine
    ]
    if not found:
        separator = "" if not text or text.endswith("\n") else "\n"
        return f"{text}{separator}machine {machine}\n  login user\n  password {password}\n"
    entry = found[-1]
    for keyword, value in entry:
        if keyword.group() == "password":
            if value is None:
                return f"{text[: keyword.end()]} {password}{text[keyword.end() :]}"
            return text[: value.start()] + password + text[value.end() :]
    keywords = {keyword.group() for keyword, _ in entry}
    login = "" if keywords & {"login", "user"} else " login user"
    end = max((value or keyword).end() for keyword, value in entry)
    return f"{text[:end]}{login} password {password}{text[end:]}"


def _save_wandb(key: str) -> str:
    path = Path(os.path.realpath(Path.home() / ".netrc"))
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    write_atomic(path, netrc_with_key(text, NETRC_MACHINE, key))
    os.chmod(path, 0o600)
    return f"Saved in the {NETRC_MACHINE} entry of ~/.netrc."


def _save_openai(key: str) -> str:
    write_atomic(state_dir() / OPENAI_KEY_FILE, key + "\n")
    os.chmod(state_dir() / OPENAI_KEY_FILE, 0o600)
    return f"Saved in Monopole's state folder. {server.restart_for_new_key(run_db_path())}"


def set_key(provider: str, key: str, *, client: httpx.Client | None = None) -> dict[str, Any]:
    """Check the key, save it, and check every provider again. The result names where the key
    went, never the key."""

    if provider not in KEY_PROVIDERS:
        raise Refused(f"Monopole saves keys for {' and '.join(KEY_PROVIDERS)} only.")
    owned = client is None
    http = client or httpx.Client()
    try:
        _check(provider, key, http)
    finally:
        if owned:
            http.close()
    saved = _save_wandb(key) if provider == "wandb" else _save_openai(key)
    if provider == "wandb" and os.environ.get("WANDB_API_KEY"):
        saved += " WANDB_API_KEY is set in Monopole's environment and still wins over ~/.netrc."
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = key
    return {"format": 1, "saved": saved, "providers": providers.check_providers()}


def apply_saved_keys() -> None:
    """Export the saved OpenAI key, if any, for this action and what it starts."""

    path = state_dir() / OPENAI_KEY_FILE
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return
    if key:
        os.environ["OPENAI_API_KEY"] = key
