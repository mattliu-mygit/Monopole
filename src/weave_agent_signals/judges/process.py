"""Shared isolation boundary for local coding-agent CLI subprocesses."""

from __future__ import annotations

import os
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


def prepare_cli_subprocess(
    *,
    home: str,
    codex_home: str,
    cwd: str,
    provider: str,
    source_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Create provider state and a minimal local-CLI child environment.

    Local CLIs authenticate from their existing on-disk/keychain stores. API keys,
    OAuth tokens, cloud-provider credentials, proxies, and unrelated parent values
    are intentionally not inherited.
    """
    if provider not in {"claude", "codex", "agy"}:
        raise ValueError(f"unsupported local CLI provider: {provider}")

    os.makedirs(cwd, exist_ok=True)
    child_home = home
    if provider == "codex":
        child_home = os.path.join(cwd, "home")
        os.makedirs(child_home, exist_ok=True)
        real_auth = os.path.join(home, ".codex", "auth.json")
        isolated_auth = os.path.join(codex_home, "auth.json")
        if os.path.exists(real_auth) and not os.path.exists(isolated_auth):
            try:
                os.symlink(real_auth, isolated_auth)
            except FileExistsError:
                pass

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
