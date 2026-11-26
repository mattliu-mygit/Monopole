from __future__ import annotations

import json
import subprocess

import httpx
import pytest

from weave_agent_signals.plugin import providers
from weave_agent_signals.plugin.settings import state_dir


class _FakeRun:
    """Answers each CLI status command as a signed-in or signed-out CLI does."""

    def __init__(self, answers: dict[str, object]):
        self.answers = answers
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        answer = self.answers[argv[0]]
        if isinstance(answer, BaseException):
            raise answer
        code, stdout, stderr = answer
        return subprocess.CompletedProcess(argv, code, stdout, stderr)


@pytest.fixture
def environment(monkeypatch, state):
    monkeypatch.setattr(
        providers, "cli_environment", lambda provider: {"PATH": "/fake/bin", "HOME": f"/{provider}"}
    )
    monkeypatch.setattr(providers.shutil, "which", lambda name, path=None: f"/fake/bin/{name}")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_PROJECT", "evals")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    return state


def _client(statuses: dict[str, int], seen: list[httpx.Request]) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(statuses[request.url.host])

    return httpx.Client(transport=httpx.MockTransport(handle))


def _by_id(result: dict) -> dict[str, dict]:
    return {provider["id"]: provider for provider in result["providers"]}


def test_each_provider_is_checked_the_way_its_research_found_reliable(environment):
    run = _FakeRun(
        {
            # Claude's JSON decides, even when its exit code disagrees.
            "claude": (1, json.dumps({"loggedIn": True, "authMethod": "claude.ai"}), ""),
            "codex": (0, "Logged in using ChatGPT", ""),
            "devin": (1, "", "Error: Not logged in. Run devin auth login\n"),
        }
    )
    seen: list[httpx.Request] = []

    result = providers.check_providers(
        run=run, client=_client({"api.inference.wandb.ai": 200, "api.openai.com": 401}, seen)
    )

    states = {key: value["state"] for key, value in _by_id(result).items()}
    assert states == {
        "claude": "signed_in",
        "codex": "signed_in",
        "devin": "signed_out",
        "wandb": "signed_in",
        "openai": "signed_out",
    }
    assert _by_id(result)["codex"]["detail"] == providers.CODEX_NOTE
    assert _by_id(result)["devin"]["detail"] == "Error: Not logged in. Run devin auth login"
    assert sorted(argv for argv, _ in run.calls) == [
        ["claude", "auth", "status"],
        ["codex", "login", "status"],
        ["devin", "models"],
    ]
    # Each CLI runs in judging's environment, with a short timeout.
    assert all(kwargs["env"]["HOME"] == f"/{argv[0]}" for argv, kwargs in run.calls)
    assert all(kwargs["timeout"] == providers.CHECK_TIMEOUT_SECONDS for _, kwargs in run.calls)
    wandb = next(request for request in seen if request.url.host == "api.inference.wandb.ai")
    assert str(wandb.url) == "https://api.inference.wandb.ai/v1/models"
    assert wandb.headers["Authorization"] == "Bearer wandb-key"
    assert wandb.headers["OpenAI-Project"] == "team/evals"
    saved = json.loads((state_dir() / "providers.json").read_text())
    assert saved == result
    assert providers.last_check() == result


def test_missing_programs_keys_and_slow_answers(environment, monkeypatch):
    monkeypatch.setattr(
        providers.shutil, "which", lambda name, path=None: None if name == "devin" else name
    )
    monkeypatch.delenv("WANDB_API_KEY")
    monkeypatch.setattr(providers, "get_api_key", _no_key)
    monkeypatch.delenv("OPENAI_API_KEY")
    run = _FakeRun(
        {
            "claude": subprocess.TimeoutExpired(["claude"], 10),
            "codex": (1, "Not logged in", ""),
        }
    )

    result = providers.check_providers(run=run, client=_client({}, []))

    by_id = _by_id(result)
    assert {key: value["state"] for key, value in by_id.items()} == {
        "claude": "unknown",
        "codex": "signed_out",
        "devin": "not_installed",
        "wandb": "no_key",
        "openai": "no_key",
    }
    assert by_id["claude"]["detail"] == "claude auth status did not answer in 10 s"


def _no_key():
    raise providers.ConfigurationError("No W&B API key found.")


def test_an_unexpected_answer_is_unknown_not_signed_out(environment):
    run = _FakeRun({"claude": (0, "{}", ""), "codex": (0, "", ""), "devin": (0, "", "")})

    result = providers.check_providers(
        run=run, client=_client({"api.inference.wandb.ai": 500, "api.openai.com": 200}, [])
    )

    by_id = _by_id(result)
    assert by_id["wandb"]["state"] == "unknown"
    assert by_id["wandb"]["detail"].endswith("answered HTTP 500")
    # Claude's JSON without loggedIn falls back to the exit code.
    assert by_id["claude"]["state"] == "signed_in"
