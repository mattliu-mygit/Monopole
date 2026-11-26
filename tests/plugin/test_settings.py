from __future__ import annotations

import json
import os

import pytest

from weave_agent_signals.cli import main
from weave_agent_signals.models import ConfigurationError
from weave_agent_signals.plugin.settings import (
    apply_settings,
    load_settings,
    run_db_path,
    save_settings,
    state_dir,
)


def test_get_settings_prints_blank_fields_and_the_default_port(state, capsys):
    assert main(["plugin", "get-settings"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "run_db": "",
        "target_registry": "",
        "sandbox_runtime": "",
        "port": 8787,
        "ces": {"url": "", "scope_ref": "", "dataset_id": ""},
    }


def test_save_settings_merges_the_changed_fields_and_applies_them(state, monkeypatch):
    registry = state / "targets.json"
    registry.write_text("{}")
    save_settings(json.dumps({"target_registry": str(registry), "port": 8790}))
    save_settings(
        json.dumps({"ces": {"dataset_id": " dataset-1 "}, "run_db": str(state / "v11.db")})
    )

    settings = load_settings()
    assert settings["target_registry"] == str(registry)
    assert settings["port"] == 8790
    assert settings["ces"] == {"url": "", "scope_ref": "", "dataset_id": "dataset-1"}
    assert json.loads((state_dir() / "settings.json").read_text()) == settings

    apply_settings(settings)
    assert os.environ["TARGET_REGISTRY"] == str(registry)
    assert os.environ["WEAVE_AGENT_SIGNALS_CES_DATASET_ID"] == "dataset-1"
    assert run_db_path() == state / "v11.db"
    # A blank field leaves its variable to the environment and .env.
    assert "WEAVE_AGENT_SIGNALS_CES_URL" not in os.environ


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"port": 0}, "port must be"),
        ({"port": True}, "port must be"),
        ({"target_registry": "/no/such/file.json"}, "not a file"),
        ({"ces": {"region": "x"}}, "ces.region"),
        ({"surprise": 1}, "not one of Monopole's settings"),
    ],
)
def test_save_settings_refuses_what_does_not_fit_and_keeps_the_saved_file(state, changes, message):
    with pytest.raises(ConfigurationError, match=message):
        save_settings(json.dumps(changes))
    assert not (state_dir() / "settings.json").exists()


def test_a_refused_setting_exits_2_with_the_reason(state, capsys):
    assert main(["plugin", "save-settings", '{"port": "x"}']) == 2
    assert "port must be a whole number" in capsys.readouterr().err


def test_serve_needs_the_registry_and_runtime(state, capsys):
    assert main(["plugin", "serve"]) == 2
    assert "needs a target registry and a sandbox runtime" in capsys.readouterr().err
