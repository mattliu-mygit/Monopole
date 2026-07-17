"""Tests for the attempt-scoped Antigravity prompt-file transport."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from weave_agent_signals.judges import agy_transport


def test_agy_invocation_writes_private_prompt_and_cleans_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agy_transport.tempfile, "tempdir", str(tmp_path))

    with agy_transport.agy_prompt_invocation(
        prompt="private prompt",
        model="Gemini 3.5 Flash (High)",
        home=str(tmp_path / "home"),
    ) as invocation:
        prompt_path = Path(invocation.prompt_path)
        workspace = Path(invocation.cwd)
        isolated_home = Path(invocation.home)
        assert prompt_path.read_text(encoding="utf-8") == "private prompt"
        assert stat.S_IMODE(prompt_path.stat().st_mode) == 0o600
        assert isolated_home.parent == workspace.parent
        assert isolated_home != workspace
        assert not isolated_home.is_relative_to(workspace)
        assert stat.S_IMODE(isolated_home.stat().st_mode) == 0o700
        assert "private prompt" not in invocation.argv
        assert invocation.argv[:3] == [
            "/usr/bin/sandbox-exec",
            "-f",
            invocation.profile_path,
        ]
        assert invocation.argv[3:6] == ["agy", "--print", invocation.bootstrap]
        assert invocation.argv[invocation.argv.index("--model") + 1] == ("Gemini 3.5 Flash (High)")
        assert invocation.argv[invocation.argv.index("--mode") + 1] == "accept-edits"
        assert "--sandbox" in invocation.argv
        assert "--dangerously-skip-permissions" in invocation.argv
        assert invocation.argv[invocation.argv.index("--add-dir") + 1] == str(workspace)
        assert os.path.commonpath([invocation.profile_path, str(workspace)]) == str(workspace)

    assert not workspace.exists()
    assert not isolated_home.exists()


def test_agy_invocation_streams_oauth_token_once_into_isolated_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agy_transport.tempfile, "tempdir", str(tmp_path))
    host_home = tmp_path / "host-home"
    host_state = host_home / ".gemini" / "antigravity-cli"
    host_state.mkdir(parents=True)
    (host_state / "antigravity-oauth-token").write_text("credential", encoding="utf-8")
    (host_state / "conversation_summaries.db").write_text("history", encoding="utf-8")

    with agy_transport.agy_prompt_invocation(
        prompt="prompt",
        model="GPT-OSS 120B (Medium)",
        home=str(host_home),
    ) as invocation:
        isolated_state = Path(invocation.home) / ".gemini" / "antigravity-cli"
        token = isolated_state / "antigravity-oauth-token"
        assert stat.S_ISFIFO(token.stat().st_mode)
        assert stat.S_IMODE(token.stat().st_mode) == 0o600
        assert token.read_text(encoding="utf-8") == "credential"
        assert not token.exists()
        assert not (isolated_state / "conversation_summaries.db").exists()


def test_agy_invocation_closes_unused_oauth_token_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agy_transport.tempfile, "tempdir", str(tmp_path))
    host_state = tmp_path / "host-home" / ".gemini" / "antigravity-cli"
    host_state.mkdir(parents=True)
    (host_state / "antigravity-oauth-token").write_text("credential", encoding="utf-8")

    with agy_transport.agy_prompt_invocation(
        prompt="prompt",
        model="GPT-OSS 120B (Medium)",
        home=str(tmp_path / "host-home"),
    ):
        pass


def test_agy_invocation_rejects_oversized_oauth_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agy_transport.tempfile, "tempdir", str(tmp_path))
    host_state = tmp_path / "host-home" / ".gemini" / "antigravity-cli"
    host_state.mkdir(parents=True)
    (host_state / "antigravity-oauth-token").write_bytes(
        b"x" * (agy_transport.MAX_OAUTH_TOKEN_BYTES + 1)
    )

    with pytest.raises(OSError, match="exceeds"):
        with agy_transport.agy_prompt_invocation(
            prompt="prompt",
            model="GPT-OSS 120B (Medium)",
            home=str(tmp_path / "host-home"),
        ):
            pass


def test_agy_invocation_rejects_symlinked_oauth_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agy_transport.tempfile, "tempdir", str(tmp_path))
    host_home = tmp_path / "host-home"
    host_state = host_home / ".gemini" / "antigravity-cli"
    host_state.mkdir(parents=True)
    target = tmp_path / "target"
    target.write_text("credential", encoding="utf-8")
    (host_state / "antigravity-oauth-token").symlink_to(target)

    with pytest.raises(OSError):
        with agy_transport.agy_prompt_invocation(
            prompt="prompt",
            model="GPT-OSS 120B (Medium)",
            home=str(host_home),
        ):
            pass


def test_agy_invocation_rejects_missing_host_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agy_transport, "_SANDBOX_EXEC", str(tmp_path / "missing"))

    with pytest.raises(RuntimeError, match="host confinement is unavailable"):
        with agy_transport.agy_prompt_invocation(
            prompt="x",
            model="Gemini 3.5 Flash (High)",
            home=str(tmp_path),
        ):
            pass


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS-only")
def test_host_profile_reads_prompt_but_denies_unrelated_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    home = tmp_path / "home"
    auth_home = tmp_path / "auth-home"
    workspace.mkdir()
    home.mkdir()
    auth_home.mkdir()
    allowed = workspace / "prompt.txt"
    denied = tmp_path / "outside.txt"
    allowed.write_text("allowed", encoding="utf-8")
    denied.write_text("denied", encoding="utf-8")

    profile = agy_transport._write_sandbox_profile(
        workspace=str(workspace),
        host_home=str(home),
        auth_home=str(auth_home),
    )

    allowed_result = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", profile, "/bin/cat", str(allowed)],
        capture_output=True,
        text=True,
        check=False,
    )
    denied_result = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", profile, "/bin/cat", str(denied)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert allowed_result.stdout == "allowed"
    assert allowed_result.returncode == 0
    assert denied_result.stdout == ""
    assert denied_result.returncode != 0


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS-only")
def test_host_profile_isolates_gemini_state_from_persistent_home(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    home = tmp_path / "home"
    auth_home = tmp_path / "auth-home"
    isolated_config = auth_home / ".gemini" / "config"
    persistent_config = home / ".gemini" / "config"
    workspace.mkdir()
    isolated_config.mkdir(parents=True)
    persistent_config.mkdir(parents=True)
    allowed = isolated_config / "config.json"
    denied = persistent_config / "config.json"
    allowed.write_text("isolated-state", encoding="utf-8")
    denied.write_text("persistent-state", encoding="utf-8")

    profile = agy_transport._write_sandbox_profile(
        workspace=str(workspace),
        host_home=str(home),
        auth_home=str(auth_home),
    )
    allowed_result = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", profile, "/bin/cat", str(allowed)],
        capture_output=True,
        text=True,
        check=False,
    )
    denied_result = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", profile, "/bin/cat", str(denied)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert allowed_result.stdout == "isolated-state"
    assert allowed_result.returncode == 0
    assert denied_result.stdout == ""
    assert denied_result.returncode != 0


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS-only")
def test_host_profile_allows_keychain_reads_but_denies_other_library_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    home = tmp_path / "home"
    auth_home = tmp_path / "auth-home"
    keychains = home / "Library" / "Keychains"
    unrelated = home / "Library" / "Application Support"
    workspace.mkdir()
    auth_home.mkdir()
    keychains.mkdir(parents=True)
    unrelated.mkdir(parents=True)
    allowed = keychains / "login.keychain-db"
    denied = unrelated / "secret.txt"
    allowed.write_text("keychain", encoding="utf-8")
    denied.write_text("unrelated-library-data", encoding="utf-8")

    profile = agy_transport._write_sandbox_profile(
        workspace=str(workspace),
        host_home=str(home),
        auth_home=str(auth_home),
    )
    allowed_result = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", profile, "/bin/cat", str(allowed)],
        capture_output=True,
        text=True,
        check=False,
    )
    denied_result = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", profile, "/bin/cat", str(denied)],
        capture_output=True,
        text=True,
        check=False,
    )
    null_result = subprocess.run(
        ["/usr/bin/sandbox-exec", "-f", profile, "/usr/bin/tee", "/dev/null"],
        input="keyring-helper-output",
        capture_output=True,
        text=True,
        check=False,
    )

    assert allowed_result.stdout == "keychain"
    assert allowed_result.returncode == 0
    assert denied_result.stdout == ""
    assert denied_result.returncode != 0
    assert null_result.returncode == 0
