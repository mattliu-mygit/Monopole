"""API keys for W&B Inference and OpenAI, saved from Prompting Time.

`set-key --item wandb|openai` takes the key from `--key` (Prompting Time gives actions no stdin)
or from stdin, checks it with `GET /v1/models`, which spends nothing, and saves it only when the
provider accepts it. Nothing prints or logs the key.

- W&B goes where Monopole already reads it: the `api.wandb.ai` entry of `~/.netrc`. Only that
  entry changes; the file is rewritten whole through a temporary file with mode 600.
  `WANDB_API_KEY`, when set, still wins over the file.
- OpenAI goes to `<state folder>/openai-key` (mode 600). Every plugin action, including the start
  command's server, exports it as OPENAI_API_KEY, over `.env` and the environment. The running
  server reads it only when it starts again.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

from weave_agent_signals import ces_commands
from weave_agent_signals.judges.inference import INFERENCE_BASE, OPENAI_BASE
from weave_agent_signals.plugin import providers
from weave_agent_signals.plugin.confirm import Refused
from weave_agent_signals.plugin.settings import state_dir, write_atomic

KEY_PROVIDERS = ("wandb", "openai")
NETRC_MACHINE = "api.wandb.ai"
OPENAI_KEY_FILE = "openai-key"
CHECK_TIMEOUT_SECONDS = 10

# Where an entry of ~/.netrc starts: `machine <host>`, `default`, or a `macdef` macro, whose body
# runs to the next blank line.
_ENTRY_START = re.compile(r"(?m)(?<!\S)(machine|default|macdef)(?!\S)")


def read_key(argument: str | None) -> str:
    key = (argument if argument is not None else sys.stdin.read()).strip()
    if not key:
        raise Refused("No key given; nothing saved.")
    if any(char.isspace() for char in key):
        raise Refused("A key has no spaces; nothing saved.")
    return key


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


def netrc_with_key(text: str, machine: str, password: str) -> str:
    """`text` with `machine`'s entry holding `password`, keeping its login (else "user") and
    every other entry byte for byte. Appends the entry when the file has none."""

    starts = []
    for match in _ENTRY_START.finditer(text):
        if starts and starts[-1][0] == "macdef":
            body_end = text.find("\n\n", starts[-1][1])
            if body_end != -1 and match.start() < body_end:
                continue
        starts.append((match.group(1), match.start()))
    for index, (kind, start) in enumerate(starts):
        if kind != "machine":
            continue
        end = starts[index + 1][1] if index + 1 < len(starts) else len(text)
        tokens = text[start:end].split()
        if len(tokens) < 2 or tokens[1] != machine:
            continue
        values = dict(zip(tokens[2::2], tokens[3::2]))
        login = values.get("login", "user")
        trailing = text[start:end][len(text[start:end].rstrip()) :] or "\n"
        entry = f"machine {machine}\n  login {login}\n  password {password}{trailing}"
        return text[:start] + entry + text[end:]
    separator = "" if not text or text.endswith("\n") else "\n"
    return f"{text}{separator}machine {machine}\n  login user\n  password {password}\n"


def _save_wandb(key: str) -> str:
    path = Path(os.path.realpath(Path.home() / ".netrc"))
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    write_atomic(path, netrc_with_key(text, NETRC_MACHINE, key))
    os.chmod(path, 0o600)
    return f"Saved in the {NETRC_MACHINE} entry of ~/.netrc."


def _save_openai(key: str) -> str:
    write_atomic(state_dir() / OPENAI_KEY_FILE, key + "\n")
    os.chmod(state_dir() / OPENAI_KEY_FILE, 0o600)
    return "Saved in Monopole's state folder. Monopole's server uses it after it starts again."


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
