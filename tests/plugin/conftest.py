from __future__ import annotations

import pytest

from tests.plugin.fixtures import new_db


@pytest.fixture
def state(monkeypatch, tmp_path):
    """An empty state home and run database, with every variable the plugin's settings set
    restored after the test."""

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    db = new_db(tmp_path / "runs.db")
    monkeypatch.setenv("WEAVE_AGENT_SIGNALS_RUN_DB", str(db))
    for name in (
        "TARGET_REGISTRY",
        "SMOLMACHINES_RUNTIME",
        "WEAVE_AGENT_SIGNALS_CES_URL",
        "WEAVE_AGENT_SIGNALS_CES_SCOPE_REF",
        "WEAVE_AGENT_SIGNALS_CES_DATASET_ID",
        "OPENAI_API_KEY",
    ):
        # setenv first, so monkeypatch records the variable and removes whatever the test
        # sets directly through os.environ (apply_settings, set-key) when it ends.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return tmp_path
