"""Checks whether each model provider Monopole calls is signed in, without calling a model.

Each local CLI runs its own status command in the environment judging gives it, so the check
sees the same stored login a judge call would. `claude auth status` and `devin models` ask the
provider; `codex login status` only reads Codex's saved file, so an expired Codex login still
reads as signed in. W&B and OpenAI keys are checked with `GET /models`, which spends nothing.
The result goes to `<state folder>/providers.json`, which `status` reads for its blockers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from weave_agent_signals import ces_commands
from weave_agent_signals.auth import get_api_key
from weave_agent_signals.judges.cli_backend import cli_environment
from weave_agent_signals.judges.inference import INFERENCE_BASE, OPENAI_BASE
from weave_agent_signals.models import ConfigurationError
from weave_agent_signals.plugin.runs import DETAIL_LIMIT, one_line
from weave_agent_signals.plugin.settings import state_dir, write_atomic

CHECK_TIMEOUT_SECONDS = 10
# A check older than this no longer blocks anything in Prompting Time; the user rechecks it.
BLOCKER_MAX_AGE = timedelta(hours=24)

PROVIDER_TITLES = {
    "claude": "Claude",
    "codex": "Codex",
    "devin": "Devin",
    "wandb": "W&B Inference",
    "openai": "OpenAI",
}
CODEX_NOTE = "Codex checks only its saved login file, so an expired login still reads as signed in."

Run = Callable[..., subprocess.CompletedProcess]


def _result(provider: str, state: str, detail: str | None = None) -> dict[str, Any]:
    return {
        "id": provider,
        "title": PROVIDER_TITLES[provider],
        "state": state,
        "detail": one_line(detail, DETAIL_LIMIT),
    }


def _first_line(*texts: str) -> str | None:
    return next((line for text in texts for line in text.splitlines() if line.strip()), None)


def _check_cli(provider: str, argv: list[str], run: Run) -> dict[str, Any]:
    env = cli_environment(provider)
    if shutil.which(argv[0], path=env.get("PATH")) is None:
        return _result(provider, "not_installed", f"{argv[0]} is not on PATH")
    try:
        finished = run(
            argv,
            env=env,
            cwd=tempfile.gettempdir(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return _result(
            provider, "unknown", f"{' '.join(argv)} did not answer in {CHECK_TIMEOUT_SECONDS} s"
        )
    except OSError as exc:
        return _result(provider, "unknown", f"{' '.join(argv)} could not start: {exc}")
    if provider == "claude":
        # The JSON's loggedIn decides: one report has the command exit non-zero while signed in.
        try:
            signed_in = json.loads(finished.stdout)["loggedIn"] is True
        except (ValueError, KeyError, TypeError):
            signed_in = finished.returncode == 0
    else:
        signed_in = finished.returncode == 0
    if signed_in:
        return _result(provider, "signed_in", CODEX_NOTE if provider == "codex" else None)
    return _result(
        provider,
        "signed_out",
        _first_line(finished.stderr, finished.stdout) or f"{' '.join(argv)} reports no login",
    )


def _check_key(
    provider: str, url: str, headers: dict[str, str], client: httpx.Client
) -> dict[str, Any]:
    try:
        response = client.get(url, headers=headers, timeout=CHECK_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        return _result(provider, "unknown", f"GET {url} failed: {exc}")
    if response.status_code == 200:
        return _result(provider, "signed_in")
    if response.status_code in (401, 403):
        return _result(provider, "signed_out", f"The key was refused (HTTP {response.status_code})")
    return _result(provider, "unknown", f"GET {url} answered HTTP {response.status_code}")


def _check_wandb(client: httpx.Client) -> dict[str, Any]:
    try:
        key = get_api_key()
    except ConfigurationError as exc:
        return _result("wandb", "no_key", str(exc))
    entity, project = ces_commands.inference_project(
        os.environ.get("WANDB_ENTITY"), os.environ.get("WANDB_PROJECT")
    )
    headers = {"Authorization": f"Bearer {key}", "OpenAI-Project": f"{entity}/{project}"}
    return _check_key("wandb", f"{INFERENCE_BASE}/models", headers, client)


def _check_openai(client: httpx.Client) -> dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return _result("openai", "no_key", "OPENAI_API_KEY is not set")
    headers = {"Authorization": f"Bearer {key}"}
    return _check_key("openai", f"{OPENAI_BASE}/models", headers, client)


def check_providers(
    *, run: Run = subprocess.run, client: httpx.Client | None = None
) -> dict[str, Any]:
    """Run every check at once, save the result in the state folder, and return it."""

    owned = client is None
    http = client or httpx.Client()
    checks: list[Callable[[], dict[str, Any]]] = [
        lambda: _check_cli("claude", ["claude", "auth", "status"], run),
        lambda: _check_cli("codex", ["codex", "login", "status"], run),
        # `devin auth status` only looks for the credentials file and exits 0 either way.
        lambda: _check_cli("devin", ["devin", "models"], run),
        lambda: _check_wandb(http),
        lambda: _check_openai(http),
    ]
    try:
        with ThreadPoolExecutor(max_workers=len(checks)) as pool:
            providers = list(pool.map(lambda check: check(), checks))
    finally:
        if owned:
            http.close()
    result = {
        "format": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "providers": providers,
    }
    write_atomic(state_dir() / "providers.json", json.dumps(result, indent=2) + "\n")
    return result


def last_check() -> dict[str, Any]:
    """The last saved check, or an empty one before the first."""

    path = state_dir() / "providers.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"format": 1, "checked_at": None, "providers": []}


def blockers(now: datetime | None = None) -> list[dict[str, Any]]:
    """Status-format blockers for the providers a check in the last day found signed out, and
    for a missing W&B key, which every run needs for CES. A missing OpenAI key or CLI blocks
    nothing: only runs that choose those models need them."""

    check = last_check()
    checked_at = check.get("checked_at")
    if not checked_at:
        return []
    now = now or datetime.now(timezone.utc)
    if now - datetime.fromisoformat(checked_at) > BLOCKER_MAX_AGE:
        return []
    found = []
    for provider in check.get("providers", []):
        state = provider.get("state")
        if state == "signed_out":
            title = f"{provider['title']} is signed out"
        elif state == "no_key" and provider.get("id") == "wandb":
            title = "No W&B API key"
        else:
            continue
        found.append(
            {
                "id": f"{provider['id']}-{state.replace('_', '-')}",
                "title": title,
                "dot": "error",
                "route": "/providers",
                "action": "check-providers",
            }
        )
    return found
