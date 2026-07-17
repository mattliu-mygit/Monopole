"""Attempt-scoped prompt-file transport for confined Antigravity inference."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
MAX_OAUTH_TOKEN_BYTES = 8 * 1024


@dataclass(frozen=True)
class AgyInvocation:
    """One isolated Antigravity process invocation."""

    argv: list[str]
    cwd: str
    home: str
    prompt_path: str
    profile_path: str
    bootstrap: str


def _write_private_text(path: str, value: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as destination:
        destination.write(value)


def _write_sandbox_profile(*, workspace: str, host_home: str, auth_home: str) -> str:
    keychains = os.path.join(host_home, "Library", "Keychains")
    workspace_literal = json.dumps(os.path.realpath(workspace))
    auth_home_literal = json.dumps(os.path.realpath(auth_home))
    keychains_literal = json.dumps(os.path.realpath(keychains))
    temp_root_literal = json.dumps(os.path.realpath(tempfile.gettempdir()))
    path = os.path.join(workspace, "agy.sb")
    profile = f"""(version 1)
(deny default)
(allow process*)
(allow signal)
(allow sysctl-read)
(allow mach-lookup)
(allow network*)
(allow system-socket)
(allow file-read-metadata)
(allow file-read*
    (require-all
        (require-not (subpath "/Users"))
        (require-not (subpath "/Volumes"))
        (require-not (subpath {temp_root_literal}))))
(allow file-read* (subpath {workspace_literal}))
(allow file-write* (subpath {workspace_literal}))
(allow file-read* (subpath {auth_home_literal}))
(allow file-write* (subpath {auth_home_literal}))
(allow file-write* (literal "/dev/null"))
(allow file-read* (subpath {keychains_literal}))
"""
    _write_private_text(path, profile)
    return path


@contextmanager
def _oauth_token_pipe(*, host_home: str, auth_home: str) -> Iterator[None]:
    host_token = os.path.join(
        host_home,
        ".gemini",
        "antigravity-cli",
        "antigravity-oauth-token",
    )
    try:
        token_fd = os.open(host_token, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        yield
        return
    with os.fdopen(token_fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise OSError("Antigravity OAuth token is not a regular file")
        token = source.read(MAX_OAUTH_TOKEN_BYTES + 1)
        if len(token) > MAX_OAUTH_TOKEN_BYTES:
            raise OSError("Antigravity OAuth token exceeds the safe pipe size")

    isolated_gemini = os.path.join(auth_home, ".gemini")
    isolated_state = os.path.join(isolated_gemini, "antigravity-cli")
    os.mkdir(isolated_gemini, 0o700)
    os.mkdir(isolated_state, 0o700)
    pipe_path = os.path.join(isolated_state, "antigravity-oauth-token")
    os.mkfifo(pipe_path, 0o600)
    errors: list[BaseException] = []

    def serve_token() -> None:
        try:
            destination_fd = os.open(pipe_path, os.O_WRONLY)
            os.unlink(pipe_path)
            with os.fdopen(destination_fd, "wb") as destination:
                destination.write(token)
        except BrokenPipeError:
            pass
        except BaseException as error:  # surfaced on the invoking thread below
            errors.append(error)

    thread = threading.Thread(target=serve_token, daemon=True)
    thread.start()
    try:
        yield
    finally:
        reader_fd: int | None = None
        if thread.is_alive():
            try:
                reader_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
            except FileNotFoundError:
                pass
        try:
            thread.join(timeout=1)
        finally:
            if reader_fd is not None:
                os.close(reader_fd)
        if thread.is_alive():
            raise RuntimeError("Antigravity OAuth token pipe did not close")
        if errors:
            raise errors[0]


@contextmanager
def agy_prompt_invocation(*, prompt: str, model: str, home: str) -> Iterator[AgyInvocation]:
    """Yield one private prompt-file invocation and remove it on exit."""

    if sys.platform != "darwin" or not os.path.isfile(_SANDBOX_EXEC):
        raise RuntimeError("Agy host confinement is unavailable")

    with tempfile.TemporaryDirectory(prefix="weave-agent-signals-agy-") as root:
        os.chmod(root, 0o700)
        workspace = os.path.join(root, "workspace")
        isolated_home = os.path.join(root, "home")
        os.mkdir(workspace, 0o700)
        os.mkdir(isolated_home, 0o700)
        prompt_path = os.path.join(workspace, "prompt.txt")
        _write_private_text(prompt_path, prompt)
        profile_path = _write_sandbox_profile(
            workspace=workspace,
            host_home=home,
            auth_home=isolated_home,
        )
        bootstrap = f"Read {prompt_path} completely and follow its instructions exactly."
        argv = [
            _SANDBOX_EXEC,
            "-f",
            profile_path,
            "agy",
            "--print",
            bootstrap,
            "--model",
            model,
            "--mode",
            "accept-edits",
            "--sandbox",
            "--dangerously-skip-permissions",
            "--add-dir",
            workspace,
        ]
        with _oauth_token_pipe(host_home=home, auth_home=isolated_home):
            yield AgyInvocation(
                argv=argv,
                cwd=workspace,
                home=isolated_home,
                prompt_path=prompt_path,
                profile_path=profile_path,
                bootstrap=bootstrap,
            )
