"""Shared isolation boundary for local coding-agent CLI subprocesses."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping

_PASSTHROUGH_ENV_KEYS = (
    # Executable discovery and OS account/config-store lookup.
    "PATH",
    "USER",
    "LOGNAME",
    "SHELL",
    # Non-secret process/runtime essentials.
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
)

# Codex permission profiles are deny-by-default. The judge may execute tools only
# against the empty isolated workspace and the minimal OS runtime paths needed to
# start commands. In particular, CODEX_HOME is deliberately outside the readable
# workspace root: the Codex host can authenticate from it, but model-generated
# commands cannot inspect it.
CODEX_CONFINED_ARGS = (
    "--strict-config",
    "-c",
    'default_permissions="judge"',
    "-c",
    'permissions.judge.filesystem={":minimal"="read",":workspace_roots"={"."="read"}}',
    "-c",
    "permissions.judge.network.enabled=false",
    "-c",
    "allow_login_shell=false",
    "-c",
    'web_search="disabled"',
    "-c",
    "features.apps=false",
)


# Devin's CLI can't turn its tools off, so its judge config denies every tool; a deny rule wins
# in every permission mode. `setup_complete` stops Devin's first-run banner from reaching stdout
# ahead of the answer; with it set, Devin leaves the file as written.
DEVIN_JUDGE_CONFIG = {
    "version": 1,
    "shell": {"setup_complete": True},
    "permissions": {"deny": ["read", "edit", "grep", "glob", "exec", "Fetch(*)", "mcp__*"]},
}
_DEVIN_CREDENTIALS = os.path.join(".local", "share", "devin", "credentials.toml")


def devin_config_path(devin_home: str) -> str:
    """The judge's Devin user config: the default location inside its isolated home."""
    return os.path.join(devin_home, ".config", "devin", "config.json")


def _link_credential(real: str, isolated: str) -> None:
    """Link a CLI's stored login into an isolated home, leaving an existing link alone."""
    if not os.path.exists(real) or os.path.lexists(isolated):
        return
    os.makedirs(os.path.dirname(isolated), exist_ok=True)
    try:
        os.symlink(real, isolated)
    except FileExistsError:
        pass


def _write_if_changed(path: str, content: str) -> None:
    """Replace a file atomically, and only when its content differs, so a CLI starting
    concurrently never reads it half-written."""
    try:
        with open(path, encoding="utf-8") as stream:
            if stream.read() == content:
                return
    except FileNotFoundError:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".config-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        os.remove(temporary)
        raise


def prepare_cli_subprocess(
    *,
    home: str,
    codex_home: str,
    devin_home: str,
    cwd: str,
    provider: str,
    source_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Create provider state and a minimal local-CLI child environment.

    Local CLIs authenticate from their existing on-disk/keychain stores. API keys,
    OAuth tokens, cloud-provider credentials, proxies, and unrelated parent values
    are intentionally not inherited. Codex and Devin get an isolated `HOME` that links
    only their stored login, so the user's own config, rules, and plugins never load.
    """
    if provider not in {"claude", "codex", "agy", "devin"}:
        raise ValueError(f"unsupported local CLI provider: {provider}")

    os.makedirs(cwd, exist_ok=True)
    child_home = home
    if provider == "codex":
        child_home = os.path.join(cwd, "home")
        os.makedirs(child_home, exist_ok=True)
        _link_credential(
            os.path.join(home, ".codex", "auth.json"), os.path.join(codex_home, "auth.json")
        )
    elif provider == "devin":
        child_home = devin_home
        _link_credential(
            os.path.join(home, _DEVIN_CREDENTIALS), os.path.join(devin_home, _DEVIN_CREDENTIALS)
        )
        _write_if_changed(devin_config_path(devin_home), json.dumps(DEVIN_JUDGE_CONFIG))

    source = os.environ if source_env is None else source_env
    env = {key: source[key] for key in _PASSTHROUGH_ENV_KEYS if source.get(key)}
    env.setdefault("PATH", os.defpath)
    env.update(
        {
            "HOME": child_home,
            "WEAVE_AGENT_ADAPTER_DISABLE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    if provider == "codex":
        env["CODEX_HOME"] = codex_home
    elif provider == "claude" and source.get("CLAUDE_CONFIG_DIR"):
        env["CLAUDE_CONFIG_DIR"] = source["CLAUDE_CONFIG_DIR"]
    return env
