"""set-key with mocked HTTP and a temporary HOME, so the real ~/.netrc and keys are untouched."""

from __future__ import annotations

import io
import json
import netrc
import os
import stat
import time

import httpx
import pytest

from weave_agent_signals.cli import main
from weave_agent_signals.plugin import keys, providers
from weave_agent_signals.plugin.settings import state_dir

CHECKED = {"format": 1, "checked_at": "2026-11-26T18:00:00+00:00", "providers": []}
NETRC = """machine github.com
  login octocat
  password gh-token

machine api.wandb.ai login matt password old-key

macdef init
machine api.wandb.ai inside a macro stays

default login anonymous password guest
"""


@pytest.fixture
def home(monkeypatch, state, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_PROJECT", "evals")
    monkeypatch.setattr(providers, "check_providers", lambda: CHECKED)
    return home


@pytest.fixture
def answers(monkeypatch):
    """The HTTP status each host answers GET /models with, and the requests it saw."""

    seen: list[httpx.Request] = []
    statuses = {"api.inference.wandb.ai": 200, "api.openai.com": 200}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(statuses[request.url.host], json={"data": []})

    real = httpx.Client
    monkeypatch.setattr(
        keys.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handle), **kw)
    )
    return statuses, seen


def test_a_wandb_key_that_works_replaces_only_the_api_wandb_ai_entry(home, answers, capsys):
    (home / ".netrc").write_text(NETRC)
    (home / ".netrc").chmod(0o644)

    assert main(["plugin", "stage-key", "--item", "wandb", "--key", "new-key-123"]) == 0
    assert json.loads(capsys.readouterr().out) == {"format": 1, "staged": "wandb"}
    staged = state_dir() / "staged-wandb-key"
    assert stat.S_IMODE(os.stat(staged).st_mode) == 0o600
    # stage-key only held the key; nothing was checked or saved yet.
    assert answers[1] == [] and (home / ".netrc").read_text() == NETRC

    assert main(["plugin", "set-key", "--item", "wandb"]) == 0

    out = capsys.readouterr().out
    assert "new-key-123" not in out
    assert not staged.exists()
    assert json.loads(out)["saved"] == "Saved in the api.wandb.ai entry of ~/.netrc."
    text = (home / ".netrc").read_text()
    assert text == NETRC.replace("password old-key", "password new-key-123")
    parsed = netrc.netrc(str(home / ".netrc"))
    assert parsed.authenticators("api.wandb.ai") == ("matt", "", "new-key-123")
    assert parsed.authenticators("github.com") == ("octocat", "", "gh-token")
    assert stat.S_IMODE(os.stat(home / ".netrc").st_mode) == 0o600
    request = answers[1][0]
    assert str(request.url) == "https://api.inference.wandb.ai/v1/models"
    assert request.headers["Authorization"] == "Bearer new-key-123"
    assert request.headers["OpenAI-Project"] == "team/evals"


def test_a_wandb_key_is_appended_when_netrc_has_no_entry_or_no_file(home, answers):
    keys.set_key("wandb", "first-key")
    assert (
        home / ".netrc"
    ).read_text() == "machine api.wandb.ai\n  login user\n  password first-key\n"

    (home / ".netrc").write_text("machine github.com login a password b")
    keys.set_key("wandb", "second-key")
    assert (home / ".netrc").read_text() == (
        "machine github.com login a password b\n"
        "machine api.wandb.ai\n  login user\n  password second-key\n"
    )


def test_a_refused_key_saves_nothing(home, answers, capsys):
    (home / ".netrc").write_text(NETRC)
    answers[0]["api.inference.wandb.ai"] = 401

    assert main(["plugin", "stage-key", "--item", "wandb", "--key", "bad-key"]) == 0
    assert main(["plugin", "set-key", "--item", "wandb"]) == 1

    err = capsys.readouterr().err
    assert "refused the key (HTTP 401); nothing saved" in err and "bad-key" not in err
    assert (home / ".netrc").read_text() == NETRC
    # The held copy is gone even though the key was refused.
    assert not (state_dir() / "staged-wandb-key").exists()


def test_an_openai_key_goes_to_a_600_file_that_every_action_exports(
    home, answers, monkeypatch, capsys
):
    monkeypatch.setattr("sys.stdin", io.StringIO("sk-test-key\n"))

    assert main(["plugin", "set-key", "--item", "openai"]) == 0

    assert json.loads(capsys.readouterr().out)["saved"] == (
        "Saved in Monopole's state folder. "
        "Monopole's server isn't running; it uses the key when it starts."
    )
    saved = state_dir() / "openai-key"
    assert saved.read_text() == "sk-test-key\n"
    assert stat.S_IMODE(os.stat(saved).st_mode) == 0o600
    assert str(answers[1][0].url) == "https://api.openai.com/v1/models"
    monkeypatch.delenv("OPENAI_API_KEY")
    keys.apply_saved_keys()
    assert os.environ["OPENAI_API_KEY"] == "sk-test-key"


def test_an_empty_key_or_one_with_spaces_is_refused(home, answers, monkeypatch, capsys):
    assert main(["plugin", "stage-key", "--item", "openai", "--key", "  "]) == 1
    assert main(["plugin", "stage-key", "--item", "openai", "--key", "a b"]) == 1
    assert not (state_dir() / "staged-openai-key").exists()
    # Prompting Time gives set-key no stdin, so with nothing held it has no key.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["plugin", "set-key", "--item", "openai"]) == 1
    assert "No key given" in capsys.readouterr().err
    assert answers[1] == []


def test_a_new_wandb_key_changes_only_the_password_and_keeps_tokens_and_comments():
    text = (
        "# machine api.wandb.ai login old password commented-out\n"
        "machine example.com login machine password api.wandb.ai\n"
        "machine api.wandb.ai\n"
        "  # personal key\n"
        "  login matt\n"
        "  account team\n"
        "  password old-key # rotated monthly\n"
        "machine github.com login a password b\n"
    )

    updated = keys.netrc_with_key(text, "api.wandb.ai", "new-key")

    assert updated == text.replace("password old-key", "password new-key")


def test_a_wandb_entry_without_a_password_or_login_gets_them_in_place():
    assert (
        keys.netrc_with_key("machine api.wandb.ai login matt\nmachine b\n", "api.wandb.ai", "k")
        == "machine api.wandb.ai login matt password k\nmachine b\n"
    )
    assert keys.netrc_with_key("machine api.wandb.ai", "api.wandb.ai", "k") == (
        "machine api.wandb.ai login user password k"
    )


def test_set_key_and_status_delete_a_held_key_the_page_abandoned(
    home, answers, monkeypatch, capsys
):
    monkeypatch.setattr(keys.providers, "blockers", lambda now=None: [])
    assert main(["plugin", "stage-key", "--item", "openai", "--key", "sk-old"]) == 0
    assert main(["plugin", "stage-key", "--item", "wandb", "--key", "old-wandb"]) == 0
    old = time.time() - keys.STAGED_KEY_MAX_AGE_SECONDS - 1
    for provider in ("openai", "wandb"):
        os.utime(state_dir() / f"staged-{provider}-key", (old, old))
    capsys.readouterr()

    # Prompting Time gives set-key no stdin, so with the stale key gone it has no key.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["plugin", "set-key", "--item", "openai"]) == 1
    assert "No key given" in capsys.readouterr().err
    assert answers[1] == []
    assert not list(state_dir().glob("staged-*-key"))

    assert main(["plugin", "stage-key", "--item", "wandb", "--key", "old-wandb"]) == 0
    os.utime(state_dir() / "staged-wandb-key", (old, old))
    assert main(["plugin", "stage-key", "--item", "openai", "--key", "sk-fresh"]) == 0
    assert main(["plugin", "status"]) == 0
    assert [path.name for path in state_dir().glob("staged-*-key")] == ["staged-openai-key"]
