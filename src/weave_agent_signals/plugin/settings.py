"""The plugin's settings and state folder.

Prompting Time draws a settings form from `schemas/plugin-settings.json`, reads the values with
`get-settings`, and saves the fields the user changed with `save-settings '<json>'`. Monopole keeps
them in `<state folder>/settings.json` and applies them to the environment the server and the
other actions read, so a blank field keeps the usual source: `.env`, the environment, or the
adapter's `~/.weave-agent-adapter/evaluations.json`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from weave_agent_signals.models import ConfigurationError
from weave_agent_signals.runs.store import DEFAULT_DB_PATH

PLUGIN_NAME = "monopole"
DEFAULT_PORT = 8787

# Each text setting and the environment variable it sets when it is not blank.
_PATH_SETTINGS = {
    "run_db": "WEAVE_AGENT_SIGNALS_RUN_DB",
    "target_registry": "TARGET_REGISTRY",
    "sandbox_runtime": "SMOLMACHINES_RUNTIME",
}
_CES_SETTINGS = {
    "url": "WEAVE_AGENT_SIGNALS_CES_URL",
    "scope_ref": "WEAVE_AGENT_SIGNALS_CES_SCOPE_REF",
    "dataset_id": "WEAVE_AGENT_SIGNALS_CES_DATASET_ID",
}


def state_dir() -> Path:
    """`$XDG_STATE_HOME/monopole`, else `~/.local/state/monopole`: the folder Prompting Time
    watches and passes XDG_STATE_HOME for."""

    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base and os.path.isabs(base) else Path.home() / ".local" / "state"
    return root / PLUGIN_NAME


def write_atomic(path: Path, text: str) -> None:
    """Write through a temporary file and a rename, so a reader never sees half a file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _defaults() -> dict[str, Any]:
    return {
        **{key: "" for key in _PATH_SETTINGS},
        "port": DEFAULT_PORT,
        "ces": {key: "" for key in _CES_SETTINGS},
    }


def load_settings() -> dict[str, Any]:
    """The saved settings over the defaults: blank text and port 8787 where nothing is saved."""

    settings = _defaults()
    path = state_dir() / "settings.json"
    if not path.exists():
        return settings
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"{path} is not readable JSON: {exc}") from exc
    return _merge(settings, saved, path)


def _merge(settings: dict[str, Any], changes: object, source: object) -> dict[str, Any]:
    """`settings` with `changes` applied, after checking each changed value's type."""

    if not isinstance(changes, dict):
        raise ConfigurationError(f"{source} must be a JSON object")
    merged = json.loads(json.dumps(settings))
    for key, value in changes.items():
        if key in _PATH_SETTINGS:
            if not isinstance(value, str):
                raise ConfigurationError(f"{key} must be text")
            merged[key] = value.strip()
        elif key == "port":
            if type(value) is not int or not 1 <= value <= 65535:
                raise ConfigurationError("port must be a whole number from 1 to 65535")
            merged[key] = value
        elif key == "ces":
            if not isinstance(value, dict):
                raise ConfigurationError("ces must be an object")
            for ces_key, ces_value in value.items():
                if ces_key not in _CES_SETTINGS or not isinstance(ces_value, str):
                    raise ConfigurationError(f"ces.{ces_key} is not a text setting")
                merged["ces"][ces_key] = ces_value.strip()
        else:
            raise ConfigurationError(f"{key} is not one of Monopole's settings")
    return merged


def save_settings(changes_json: str) -> dict[str, Any]:
    """Check the changed fields, refuse a path that does not exist, and save the result."""

    try:
        changes = json.loads(changes_json)
    except ValueError as exc:
        raise ConfigurationError(f"the settings are not JSON: {exc}") from exc
    settings = _merge(load_settings(), changes, "the settings")
    for key in ("target_registry", "sandbox_runtime"):
        if settings[key] and not Path(settings[key]).expanduser().is_file():
            raise ConfigurationError(f"{key} names {settings[key]}, which is not a file")
    write_atomic(state_dir() / "settings.json", json.dumps(settings, indent=2) + "\n")
    return settings


def apply_settings(settings: dict[str, Any]) -> None:
    """Set the environment variable of each setting that is not blank, over `.env`."""

    for key, variable in _PATH_SETTINGS.items():
        if settings[key]:
            os.environ[variable] = str(Path(settings[key]).expanduser())
    for key, variable in _CES_SETTINGS.items():
        if settings["ces"][key]:
            os.environ[variable] = settings["ces"][key]


def run_db_path() -> Path:
    """The run database the server uses: WEAVE_AGENT_SIGNALS_RUN_DB, else RunStore's default."""

    value = os.environ.get("WEAVE_AGENT_SIGNALS_RUN_DB")
    if value:
        return Path(value).expanduser()
    return DEFAULT_DB_PATH
