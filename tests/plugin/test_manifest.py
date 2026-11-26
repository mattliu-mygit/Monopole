from __future__ import annotations

import json
import os
from pathlib import Path

from tests.plugin.schema_check import check_schema

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "prompting-time-plugin.json").read_text())


def test_the_manifest_fits_prompting_times_manifest_schema():
    check_schema(
        MANIFEST, json.loads((ROOT / "schemas" / "prompting-time-plugin.json").read_text())
    )


def test_the_manifest_names_executable_commands_and_declared_row_actions():
    commands = [MANIFEST["start"][0], *(action["command"][0] for action in MANIFEST["actions"])]
    assert all(os.access(ROOT / command, os.X_OK) for command in commands)
    names = {action["name"] for action in MANIFEST["actions"]}
    assert set(MANIFEST["status"]["item_actions"]) <= names
    assert {"status", "get-settings", "save-settings", "check-providers"} <= names
    # Delete runs only after Prompting Time's confirmation.
    assert next(a for a in MANIFEST["actions"] if a["name"] == "delete")["confirm"] is True


def test_the_settings_schema_names_only_the_settings_the_plugin_saves():
    schema = json.loads((ROOT / MANIFEST["settings"]).read_text())
    assert set(schema["properties"]) == {
        "target_registry",
        "sandbox_runtime",
        "run_db",
        "port",
        "ces",
    }
    assert set(schema["properties"]["ces"]["properties"]) == {"url", "scope_ref", "dataset_id"}
