"""Tests for the TEMPORARY CLI judge backend (shells out to claude/codex)."""
from __future__ import annotations

import json
import os
import textwrap

import pytest

from weave_agent_signals.judges.cli_backend import CliJudgeClient, _extract_json


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


# --- routing: agent family → which CLI ---

def test_claude_family_model_routes_to_claude_cli():
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        seen["input"] = kw.get("input")
        return _FakeProc(stdout=json.dumps({"result": json.dumps({"score": 0.8, "rationale": "ok"})}))

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


def test_google_family_model_routes_to_gemini_cli():
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return _FakeProc(stdout='{"score": 0.7, "rationale": "gem"}')

    client = CliJudgeClient(runner=fake_run)
    parsed, resp = client.chat_json(model="gemini-2.5-pro", messages=_msgs())

    assert seen["argv"][0] == "gemini"
    assert parsed["score"] == 0.7


def test_system_and_user_prompts_reach_the_cli():
    seen = {}

    def fake_run(argv, **kw):
        seen["blob"] = " ".join(argv) + " " + (kw.get("input") or "")
        return _FakeProc(stdout='{"score": 1.0, "rationale": "x"}')

    client = CliJudgeClient(runner=fake_run)
    client.chat_json(model="gpt-5.1", messages=_msgs(system="SYSTEM_MARKER", user="USER_MARKER"))
    assert "SYSTEM_MARKER" in seen["blob"]
    assert "USER_MARKER" in seen["blob"]


# --- JSON extraction from messy CLI output ---

def test_extract_json_plain():
    assert _extract_json('{"score": 0.4, "rationale": "r"}') == {"score": 0.4, "rationale": "r"}


def test_extract_json_from_fenced_block():
    text = "Here is my verdict:\n```json\n{\"score\": 0.9, \"rationale\": \"good\"}\n```\ndone"
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


# --- real end-to-end via a fake CLI executable on PATH ---

def test_end_to_end_with_fake_cli_on_path(tmp_path, monkeypatch):
    fake = tmp_path / "claude"
    fake.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import sys, json
        sys.stdin.read()  # consume the prompt
        print(json.dumps({"result": json.dumps({"score": 0.7, "rationale": "e2e"})}))
    """))
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])

    client = CliJudgeClient()  # real subprocess.run
    parsed, resp = client.chat_json(model="claude-sonnet-5", messages=_msgs())
    assert parsed["score"] == 0.7
    assert parsed["rationale"] == "e2e"


def test_is_context_manager_and_has_backend_tag():
    with CliJudgeClient(runner=lambda *a, **k: _FakeProc(stdout='{"score":0}')) as c:
        assert c.backend == "cli"
