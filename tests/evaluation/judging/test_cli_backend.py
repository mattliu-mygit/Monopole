"""Tests for the confined local CLI judge backend."""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
from unittest.mock import MagicMock

import pytest

from weave_agent_signals.judges import cli_backend, inference
from weave_agent_signals.judges.cli_backend import CliJudgeClient, _extract_json
from weave_agent_signals.judges.inference import (
    ChatClient,
    InferenceCancelled,
    InferenceClient,
)


class _FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _msgs(system="sys prompt", user="judge this turn"):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _schema():
    return inference.JsonSchemaSpec(
        name="judge_verdict",
        schema={
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
            "additionalProperties": False,
        },
    )


def test_http_and_cli_clients_satisfy_chat_client_protocol(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cli_client = CliJudgeClient(runner=lambda *_args, **_kwargs: _FakeProc(stdout='{"score":0}'))
    http_client = InferenceClient(backend="openai")

    try:
        assert isinstance(cli_client, ChatClient)
        assert isinstance(http_client, ChatClient)
    finally:
        http_client.close()


def test_http_client_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(ValueError, match="unsupported judge backend"):
        InferenceClient(backend="unsupported")


# --- routing: requested model family → which CLI ---


def test_claude_family_model_routes_to_claude_cli():
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        seen["input"] = kw.get("input")
        return _FakeProc(
            stdout=json.dumps({"result": json.dumps({"score": 0.8, "rationale": "ok"})})
        )

    client = CliJudgeClient(runner=fake_run)
    parsed, resp = client.chat_json(model="claude-sonnet-5", messages=_msgs())

    assert seen["argv"][0] == "claude"
    assert parsed["score"] == 0.8
    assert resp.model == "claude-sonnet-5"


def test_openai_family_model_routes_to_codex_cli():
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return _FakeProc(stdout='{"score": 0.5, "rationale": "meh"}')

    client = CliJudgeClient(runner=fake_run)
    parsed, resp = client.chat_json(model="gpt-5.1", messages=_msgs())

    assert seen["argv"][0] == "codex"
    assert parsed["score"] == 0.5


def test_google_family_model_is_rejected_without_a_confined_cli_mode():
    client = CliJudgeClient(runner=lambda *_args, **_kwargs: pytest.fail("must not execute"))

    with pytest.raises(RuntimeError, match="verified confined mode"):
        client.chat_json(model="gemini-2.5-pro", messages=_msgs())


def test_system_and_user_prompts_reach_the_cli():
    seen = {}

    def fake_run(argv, **kw):
        seen["blob"] = " ".join(argv) + " " + (kw.get("input") or "")
        return _FakeProc(stdout='{"score": 1.0, "rationale": "x"}')

    client = CliJudgeClient(runner=fake_run)
    client.chat_json(model="gpt-5.1", messages=_msgs(system="SYSTEM_MARKER", user="USER_MARKER"))
    assert "SYSTEM_MARKER" in seen["blob"]
    assert "USER_MARKER" in seen["blob"]


def test_claude_receives_schema_and_reads_structured_output():
    seen = {}

    def fake_run(argv, **_kwargs):
        seen["argv"] = argv
        return _FakeProc(
            stdout=json.dumps(
                {
                    "result": "unstructured result must not be used",
                    "structured_output": {"score": 0.8},
                }
            )
        )

    parsed, response = CliJudgeClient(runner=fake_run).chat_json(
        model="claude-sonnet-5",
        messages=_msgs(),
        response_schema=_schema(),
    )

    schema_arg = seen["argv"][seen["argv"].index("--json-schema") + 1]
    assert json.loads(schema_arg) == _schema().schema
    assert parsed == {"score": 0.8}
    assert response.output_mode == "json_schema"
    assert response.schema_name == "judge_verdict"


def test_codex_uses_and_removes_temporary_output_schema(tmp_path, monkeypatch):
    judge_cwd = tmp_path / "judge-home" / "sandbox"
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "judge-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(judge_cwd))
    seen = {}

    def fake_run(argv, **_kwargs):
        schema_path = argv[argv.index("--output-schema") + 1]
        seen["schema_path"] = schema_path
        assert os.path.commonpath([schema_path, str(judge_cwd)]) == str(judge_cwd)
        assert os.path.exists(schema_path)
        with open(schema_path, encoding="utf-8") as schema_file:
            assert json.load(schema_file) == _schema().schema
        return _FakeProc(stdout='{"score": 0.9}')

    parsed, response = CliJudgeClient(runner=fake_run).chat_json(
        model="gpt-5.1",
        messages=_msgs(),
        response_schema=_schema(),
    )

    assert parsed == {"score": 0.9}
    assert response.output_mode == "json_schema"
    assert not os.path.exists(seen["schema_path"])


def test_cli_schema_fallback_keeps_the_same_model(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "judge-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(tmp_path / "judge-home" / "sandbox"))
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        if len(calls) == 1:
            return _FakeProc(
                stderr="error: --output-schema is not supported by this CLI",
                returncode=2,
            )
        return _FakeProc(stdout='{"score": 0.6}')

    parsed, response = CliJudgeClient(runner=fake_run).chat_json(
        model="gpt-5.1",
        messages=_msgs(),
        response_schema=_schema(),
    )

    assert parsed == {"score": 0.6}
    assert [argv[argv.index("--model") + 1] for argv in calls] == ["gpt-5.1", "gpt-5.1"]
    assert "--output-schema" in calls[0]
    assert "--output-schema" not in calls[1]
    assert response.output_mode == "json_object_fallback"
    assert response.schema_name == "judge_verdict"
    assert response.schema_fallback_reason == "schema_output_unsupported"
    assert response.transport_request_count == 2


def test_cli_failed_fallback_preserves_both_transport_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "judge-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(tmp_path / "judge-home" / "sandbox"))
    calls = 0

    def fake_run(_argv, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _FakeProc(
                stderr="error: --output-schema is not supported by this CLI",
                returncode=2,
            )
        return _FakeProc(stderr="fallback process failed", returncode=7)

    with pytest.raises(RuntimeError) as captured:
        CliJudgeClient(runner=fake_run).chat_json(
            model="gpt-5.1",
            messages=_msgs(),
            response_schema=_schema(),
        )

    assert calls == 2
    assert getattr(captured.value, "_transport_request_count", None) == 2


@pytest.mark.parametrize(
    "message",
    [
        "--output-schema validation failed: unsupported property type",
        "--output-schema validation failed: property type is not supported",
    ],
)
def test_cli_schema_validation_failure_does_not_fallback(tmp_path, monkeypatch, message):
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "judge-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(tmp_path / "judge-home" / "sandbox"))
    schema_paths: list[str] = []

    def fake_run(argv, **_kwargs):
        schema_paths.append(argv[argv.index("--output-schema") + 1])
        return _FakeProc(
            stderr=message,
            returncode=2,
        )

    with pytest.raises(RuntimeError, match="error_category=schema_validation_error"):
        CliJudgeClient(runner=fake_run).chat_json(
            model="gpt-5.1",
            messages=_msgs(),
            response_schema=_schema(),
        )

    assert len(schema_paths) == 1
    assert not os.path.exists(schema_paths[0])


@pytest.mark.parametrize(
    "message",
    [
        "error: unknown argument '--safe-mode'\nUsage: claude [--json-schema <JSON>]",
        "429 rate limit: --output-schema is not supported while overloaded",
    ],
)
def test_cli_unrelated_or_rate_process_error_does_not_fallback(tmp_path, monkeypatch, message):
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "judge-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(tmp_path / "judge-home" / "sandbox"))
    calls = 0

    def fake_run(_argv, **_kwargs):
        nonlocal calls
        calls += 1
        return _FakeProc(stderr=message, returncode=2)

    with pytest.raises(RuntimeError):
        CliJudgeClient(runner=fake_run).chat_json(
            model="gpt-5.1",
            messages=_msgs(),
            response_schema=_schema(),
        )

    assert calls == 1


def test_cli_schema_dialect_error_does_not_trigger_capability_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "judge-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(tmp_path / "judge-home" / "sandbox"))
    calls = 0

    def fake_run(_argv, **_kwargs):
        nonlocal calls
        calls += 1
        return _FakeProc(
            stderr="output-schema unsupported property type integer",
            returncode=2,
        )

    with pytest.raises(RuntimeError, match="error_category=schema_validation_error"):
        CliJudgeClient(runner=fake_run).chat_json(
            model="gpt-5.1",
            messages=_msgs(),
            response_schema=_schema(),
        )

    assert calls == 1


@pytest.mark.parametrize(
    "output",
    [
        '```json\n{"score": 0.7}\n```',
        'Here is the result: {"score": 0.7}',
        '{"score": 0.7}\n{"score": 0.8}',
    ],
)
def test_strict_schema_path_rejects_fenced_or_embedded_json(output, tmp_path, monkeypatch):
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "judge-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(tmp_path / "judge-home" / "sandbox"))
    client = CliJudgeClient(runner=lambda *_args, **_kwargs: _FakeProc(stdout=output))

    parsed, response = client.chat_json(
        model="gpt-5.1",
        messages=_msgs(),
        response_schema=_schema(),
    )

    assert parsed == {}
    assert response.output_mode == "json_schema"
    assert response.transport_request_count == 1


@pytest.mark.parametrize(
    ("model", "expected_auth_path", "other_auth_path"),
    [
        ("gpt-5.1", "CODEX_HOME", "CLAUDE_CONFIG_DIR"),
        ("claude-sonnet-5", "CLAUDE_CONFIG_DIR", "CODEX_HOME"),
    ],
)
def test_cli_judge_subprocess_environment_is_allowlisted(
    model, expected_auth_path, other_auth_path, tmp_path, monkeypatch
):
    seen = {}
    home = tmp_path / "home"
    claude_config = tmp_path / "claude-config"
    codex_home = tmp_path / "codex-home"
    sandbox = codex_home / "sandbox"
    home.mkdir()
    claude_config.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/safe/bin")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_config))
    monkeypatch.setenv("SECRET_TOKEN", "must-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "must-not-leak")
    monkeypatch.setenv("HTTPS_PROXY", "https://user:secret@example.test")
    monkeypatch.setenv("WANDB_API_KEY", "must-not-leak")
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(codex_home))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(sandbox))
    monkeypatch.setattr(cli_backend, "_HOME", str(home))

    def fake_run(_argv, **kwargs):
        seen["env"] = kwargs["env"]
        seen["cwd"] = kwargs.get("cwd")
        return _FakeProc(stdout='{"score": 0.5, "rationale": "ok"}')

    CliJudgeClient(runner=fake_run).chat_json(model=model, messages=_msgs())

    env = seen["env"]
    assert seen["cwd"] == str(sandbox)
    expected_home = sandbox / "home" if model == "gpt-5.1" else home
    assert env["HOME"] == str(expected_home)
    assert env["PATH"] == "/safe/bin"
    assert env["LANG"] == "en_US.UTF-8"
    expected_path = codex_home if expected_auth_path == "CODEX_HOME" else claude_config
    assert env[expected_auth_path] == str(expected_path)
    assert other_auth_path not in env
    assert env["WEAVE_AGENT_ADAPTER_DISABLE"] == "1"
    assert env["WANDB_MODE"] == "disabled"
    for forbidden in (
        "SECRET_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "HTTPS_PROXY",
        "WANDB_API_KEY",
    ):
        assert forbidden not in env


def test_cli_judge_argv_hardens_stored_credential_execution():
    client = CliJudgeClient(runner=lambda *_args, **_kwargs: _FakeProc(stdout='{"score":0}'))

    codex_argv, _, _ = client._build("gpt-5.1", "system", "user")
    claude_argv, _, _ = client._build("claude-sonnet-5", "system", "user")

    assert "--ignore-user-config" in codex_argv
    assert "--ignore-rules" in codex_argv
    assert "--ephemeral" in codex_argv
    assert "--strict-config" in codex_argv
    assert "--sandbox" not in codex_argv
    assert 'default_permissions="judge"' in codex_argv
    assert (
        'permissions.judge.filesystem={":minimal"="read",'
        '":workspace_roots"={"."="read"}}' in codex_argv
    )
    assert "permissions.judge.network.enabled=false" in codex_argv
    assert "allow_login_shell=false" in codex_argv
    assert 'web_search="disabled"' in codex_argv
    assert "features.apps=false" in codex_argv
    assert "-C" in codex_argv
    assert codex_argv[codex_argv.index("-C") + 1] == cli_backend._JUDGE_CWD
    assert "--safe-mode" in claude_argv
    assert "--disable-slash-commands" in claude_argv
    assert "--no-chrome" in claude_argv
    assert "--no-session-persistence" in claude_argv
    assert claude_argv[claude_argv.index("--tools") + 1] == ""


def test_non_anthropic_or_openai_family_is_rejected_by_local_cli_backend():
    client = CliJudgeClient(runner=lambda *_args, **_kwargs: pytest.fail("must not execute"))

    with pytest.raises(RuntimeError, match="supports only Anthropic and OpenAI"):
        client.chat_json(model="deepseek-r1", messages=_msgs())


# --- JSON extraction from messy CLI output ---


def test_extract_json_plain():
    assert _extract_json('{"score": 0.4, "rationale": "r"}') == {
        "score": 0.4,
        "rationale": "r",
    }


def test_extract_json_from_fenced_block():
    text = 'Here is my verdict:\n```json\n{"score": 0.9, "rationale": "good"}\n```\ndone'
    assert _extract_json(text) == {"score": 0.9, "rationale": "good"}


def test_extract_json_with_leading_prose():
    text = 'Sure. {"score": 0.2, "rationale": "weak"}'
    assert _extract_json(text) == {"score": 0.2, "rationale": "weak"}


def test_extract_json_returns_empty_on_garbage():
    assert _extract_json("no json here") == {}


def test_extract_json_skips_stray_braces_before_object():
    # codex-style output: status/reasoning text with stray braces, then the answer.
    text = 'thinking... {not valid} then answer\n{"score": 0.9, "rationale": "ok"}'
    assert _extract_json(text) == {"score": 0.9, "rationale": "ok"}


def test_extract_json_prefers_object_with_score():
    text = '{"meta": 1}\nsome log line\n{"score": 0.5, "rationale": "r"}'
    assert _extract_json(text) == {"score": 0.5, "rationale": "r"}


# --- error handling ---


def test_nonzero_exit_raises():
    def fake_run(argv, **kw):
        return _FakeProc(stdout="", stderr="command failed", returncode=1)

    client = CliJudgeClient(runner=fake_run)
    with pytest.raises(Exception):
        client.chat_json(model="gpt-5.1", messages=_msgs())


def test_cli_success_logs_only_safe_output_metadata(caplog, tmp_path, monkeypatch):
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "judge-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(tmp_path / "judge-home" / "sandbox"))
    prompt_secret = "SENTINEL_SYSTEM_INSTRUCTION"
    output_secret = "SENTINEL_MODEL_OUTPUT"
    raw = json.dumps({"score": 0.5, "rationale": output_secret})
    client = CliJudgeClient(runner=lambda *_args, **_kwargs: _FakeProc(stdout=raw))

    with caplog.at_level("INFO", logger="weave_agent_signals.judges"):
        client.chat_json(
            model="gpt-5.1",
            messages=_msgs(system=prompt_secret),
            response_schema=_schema(),
        )

    assert prompt_secret not in caplog.text
    assert output_secret not in caplog.text
    assert f"output_len={len(raw)}" in caplog.text
    assert hashlib.sha256(raw.encode()).hexdigest() in caplog.text
    assert "output_mode=json_schema" in caplog.text
    assert "request_count=1" in caplog.text


def test_cli_failure_logs_and_exception_exclude_process_output(caplog):
    secret = "SENTINEL_PROCESS_OUTPUT"
    client = CliJudgeClient(
        runner=lambda *_args, **_kwargs: _FakeProc(
            stdout=f"stdout {secret}",
            stderr=f"stderr {secret}",
            returncode=7,
        )
    )

    with caplog.at_level("WARNING", logger="weave_agent_signals.judges"):
        with pytest.raises(RuntimeError) as captured:
            client.chat_json(model="gpt-5.1", messages=_msgs())

    assert secret not in caplog.text
    assert secret not in str(captured.value)
    assert "exit=7" in caplog.text
    assert "error_category=process_error" in caplog.text
    assert "stdout_len=" in caplog.text
    assert "stderr_len=" in caplog.text
    assert hashlib.sha256(f"stderr {secret}stdout {secret}".encode()).hexdigest() in caplog.text


def test_retryable_process_failure_retries_once(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(tmp_path / "sandbox"))
    monkeypatch.setattr(cli_backend.time, "sleep", lambda _delay: None)
    client = CliJudgeClient()
    run = MagicMock(
        side_effect=[
            ("", "429 rate limit", 1),
            ('{"score": 0.6, "rationale": "retried"}', "", 0),
        ]
    )
    monkeypatch.setattr(client, "_run_with_cancel", run)

    parsed, _ = client.chat_json(model="gpt-5.1", messages=_msgs())

    assert parsed["score"] == 0.6
    assert run.call_count == 2


def test_exhausted_cli_retries_preserve_all_transport_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(tmp_path / "sandbox"))
    monkeypatch.setattr(cli_backend.time, "sleep", lambda _delay: None)
    client = CliJudgeClient()
    run = MagicMock(return_value=("", "429 rate limit", 1))
    monkeypatch.setattr(client, "_run_with_cancel", run)

    with pytest.raises(RuntimeError) as captured:
        client.chat_json(model="gpt-5.1", messages=_msgs())

    assert run.call_count == cli_backend.MAX_CLI_RETRIES + 1
    assert getattr(captured.value, "_transport_request_count", None) == 3


def test_process_cancellation_is_not_converted_to_a_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(tmp_path / "sandbox"))
    client = CliJudgeClient()
    run = MagicMock(side_effect=InferenceCancelled("cancelled"))
    monkeypatch.setattr(client, "_run_with_cancel", run)

    with pytest.raises(InferenceCancelled, match="cancelled"):
        client.chat_json(model="gpt-5.1", messages=_msgs())

    run.assert_called_once()


# --- real end-to-end via a fake CLI executable on PATH ---


def test_end_to_end_with_fake_cli_on_path(tmp_path, monkeypatch):
    fake = tmp_path / "claude"
    fake.write_text(
        textwrap.dedent("""\
        #!/usr/bin/env python3
        import sys, json
        sys.stdin.read()  # consume the prompt
        print(json.dumps({"result": json.dumps({"score": 0.7, "rationale": "e2e"})}))
    """)
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])

    client = CliJudgeClient()  # real subprocess.run
    parsed, resp = client.chat_json(model="claude-sonnet-5", messages=_msgs())
    assert parsed["score"] == 0.7
    assert parsed["rationale"] == "e2e"


def test_codex_subprocess_skips_repo_check_from_isolated_cwd(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "codex"
    fake.write_text(
        textwrap.dedent("""\
        #!/usr/bin/env python3
        import json, os, sys
        sys.stdin.read()
        print(json.dumps({
            "score": 0.5,
            "argv": sys.argv[1:],
            "cwd": os.getcwd(),
        }))
    """)
    )
    fake.chmod(0o755)

    judge_home = tmp_path / "judge-home"
    judge_cwd = judge_home / "sandbox"
    monkeypatch.setattr(cli_backend, "_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_backend, "_JUDGE_CODEX_HOME", str(judge_home))
    monkeypatch.setattr(cli_backend, "_JUDGE_CWD", str(judge_cwd))
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])

    parsed, _ = CliJudgeClient().chat_json(model="gpt-5.1", messages=_msgs())

    assert "--skip-git-repo-check" in parsed["argv"]
    assert parsed["cwd"] == str(judge_cwd)


def test_is_context_manager_and_has_backend_tag():
    with CliJudgeClient(runner=lambda *a, **k: _FakeProc(stdout='{"score":0}')) as c:
        assert c.backend == "cli"
