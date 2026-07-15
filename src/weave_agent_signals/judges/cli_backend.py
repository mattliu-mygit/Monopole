"""Confined local-CLI implementation of the judge chat interface.

The configured model determines whether the installed ``claude`` or ``codex``
CLI is invoked. Model selection happens before this transport boundary; this
module only executes the exact requested model with a restricted environment.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable

from weave_agent_signals.judges.families import model_family
from weave_agent_signals.judges.inference import (
    SCHEMA_FALLBACK_UNSUPPORTED,
    InferenceCancelled,
    JsonSchemaSpec,
    JudgeResponse,
    _add_transport_request_count,
    _explicit_schema_rejection_reason,
    _parse_exact_json_object,
    _raw_output_digest,
)
from weave_agent_signals.judges.process import CODEX_CONFINED_ARGS, prepare_cli_subprocess

log = logging.getLogger("weave_agent_signals.judges")

# Full model ids → the alias the `claude` CLI's --model flag expects.
_CLAUDE_MODEL_ALIASES = {
    "claude-sonnet-5": "sonnet",
    "claude-opus-4-8": "opus",
    "claude-haiku-4-5": "haiku",
}


def _extract_json(text: str) -> dict:
    """Pull a judge JSON object out of a CLI/LLM text response.

    Agentic CLIs (esp. ``codex exec``) interleave status/reasoning text with the
    answer, so a naive first-``{``-to-last-``}`` slice can span unrelated braces
    and fail. Instead, scan for the first *valid* JSON object anywhere in the
    output, preferring one that carries a ``score`` key. Returns an empty dict
    when nothing parses — callers treat that as "no score".
    """
    if not text:
        return {}
    # Fenced ```json blocks first — the most explicit signal.
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    # Otherwise parse the first valid object at each '{'; prefer one with a score.
    decoder = json.JSONDecoder()
    best: dict | None = None
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            if "score" in obj:
                return obj
            if best is None:
                best = obj
        idx = text.find("{", idx + 1)
    return best if best is not None else {}


_HOME = os.path.expanduser("~")
_JUDGE_CODEX_HOME = os.path.join(_HOME, ".codex-judge")
_JUDGE_CWD = os.path.join(_HOME, ".codex-judge", "sandbox")

_RETRYABLE_PATTERNS = [
    "at capacity",
    "rate limit",
    "too many requests",
    "overloaded",
    "503",
    "429",
]

MAX_CLI_RETRIES = 2
RETRY_BASE_DELAY = 5.0


def _is_retryable(output: str) -> bool:
    lower = output.lower()
    return any(p in lower for p in _RETRYABLE_PATTERNS)


def _process_error_category(
    output: str,
    response_schema: JsonSchemaSpec | None,
) -> str:
    if response_schema is not None and _explicit_schema_rejection_reason(output):
        return "schema_output_unsupported"
    if response_schema is not None and any(
        marker in output.lower()
        for marker in (
            "validation failed",
            "validation error",
            "unsupported property",
            "property type is not supported",
        )
    ):
        return "schema_validation_error"
    if _is_retryable(output):
        return "retryable_process_error"
    return "process_error"


def _build_env(model: str) -> dict[str, str]:
    return prepare_cli_subprocess(
        home=_HOME,
        codex_home=_JUDGE_CODEX_HOME,
        cwd=_JUDGE_CWD,
        family=model_family(model),
    )


class _SchemaOutputUnsupported(RuntimeError):
    def __init__(self, reason: str, request_count: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.request_count = request_count


class CliJudgeClient:
    """Judge client that invokes a confined local coding-agent CLI.

    It exposes the same ``chat_json`` boundary as the HTTP client and is a
    context manager, so judge execution is transport-independent.
    """

    backend = "cli"

    def __init__(self, timeout: float = 360.0, runner: Callable[..., Any] | None = None):
        self._timeout = timeout
        self._runner = runner
        self._procs: set[subprocess.Popen] = set()
        self._procs_lock = threading.Lock()
        self._cancel: threading.Event | None = None

    def set_cancel(self, cancel: threading.Event) -> None:
        self._cancel = cancel

    def abort(self) -> None:
        with self._procs_lock:
            procs = list(self._procs)
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: JsonSchemaSpec | None = None,
    ) -> tuple[dict[str, Any], JudgeResponse]:
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        user = "\n\n".join(m["content"] for m in messages if m.get("role") == "user")
        family = model_family(model)
        if family == "google":
            raise RuntimeError(
                "Google local CLI judges are disabled until Gemini has a verified confined mode"
            )
        if family not in {"anthropic", "openai"}:
            raise RuntimeError(
                "The local CLI judge backend supports only Anthropic and OpenAI families"
            )
        env = _build_env(model)
        schema_path: str | None = None
        if response_schema is not None and family == "openai":
            schema_path = self._write_schema_file(response_schema)

        try:
            try:
                parsed, content, raw_output, request_count = self._invoke(
                    model=model,
                    system=system,
                    user=user,
                    env=env,
                    response_schema=response_schema,
                    schema_path=schema_path,
                )
                output_mode = "json_schema" if response_schema is not None else "json_object"
                fallback_reason = None
            except _SchemaOutputUnsupported as error:
                if response_schema is None:
                    raise
                try:
                    parsed, content, raw_output, fallback_count = self._invoke(
                        model=model,
                        system=system,
                        user=user,
                        env=env,
                        response_schema=None,
                        schema_path=None,
                    )
                except Exception as fallback_error:
                    _add_transport_request_count(fallback_error, error.request_count)
                    raise
                request_count = error.request_count + fallback_count
                output_mode = "json_object_fallback"
                fallback_reason = error.reason

            return parsed, JudgeResponse(
                content=content,
                model=model,
                usage={},
                output_mode=output_mode,
                schema_name=response_schema.name if response_schema is not None else None,
                schema_fallback_reason=fallback_reason,
                transport_request_count=request_count,
                raw_output_digest=_raw_output_digest(raw_output),
            )
        finally:
            if schema_path is not None:
                try:
                    os.remove(schema_path)
                except FileNotFoundError:
                    pass

    def _write_schema_file(self, response_schema: JsonSchemaSpec) -> str:
        """Write a Codex output schema inside the confined workspace."""
        schema_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".judge-output-schema-",
            suffix=".json",
            dir=_JUDGE_CWD,
            delete=False,
        )
        try:
            with schema_file:
                json.dump(response_schema.schema, schema_file)
        except Exception:
            try:
                os.remove(schema_file.name)
            except FileNotFoundError:
                pass
            raise
        return schema_file.name

    def _invoke(
        self,
        *,
        model: str,
        system: str,
        user: str,
        env: dict[str, str],
        response_schema: JsonSchemaSpec | None,
        schema_path: str | None,
    ) -> tuple[dict[str, Any], str, str, int]:
        argv, stdin_text, mode = self._build(
            model,
            system,
            user,
            response_schema=response_schema,
            schema_path=schema_path,
        )
        log.info(
            "CLI judge started: backend=cli model=%s family=%s prompt_len=%d output_mode=%s",
            model,
            model_family(model),
            len(stdin_text),
            "json_schema" if response_schema is not None else "json_object",
        )

        if self._runner:
            proc = self._runner(
                argv,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=env,
                cwd=_JUDGE_CWD,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            if getattr(proc, "returncode", 0) != 0:
                self._log_process_failure(
                    model=model,
                    returncode=proc.returncode,
                    elapsed=None,
                    stdout=stdout,
                    stderr=stderr,
                    attempt=1,
                    response_schema=response_schema,
                )
                self._raise_process_error(
                    argv,
                    proc.returncode,
                    stdout,
                    stderr,
                    response_schema=response_schema,
                    request_count=1,
                )
            parsed, content = self._decode_output(
                stdout,
                mode,
                strict=response_schema is not None,
            )
            log.info(
                "CLI judge completed: backend=cli model=%s exit=0 output_len=%d "
                "output_sha256=%s output_mode=%s request_count=1",
                model,
                len(stdout),
                _raw_output_digest(stdout),
                "json_schema" if response_schema is not None else "json_object",
            )
            return parsed, content, stdout, 1

        last_err: RuntimeError | None = None
        for attempt in range(MAX_CLI_RETRIES + 1):
            t0 = time.monotonic()
            stdout, stderr, returncode = self._run_with_cancel(
                argv,
                stdin_text,
                env,
            )
            elapsed = time.monotonic() - t0

            if returncode == 0:
                parsed, content = self._decode_output(
                    stdout,
                    mode,
                    strict=response_schema is not None,
                )
                log.info(
                    "CLI judge completed: backend=cli model=%s exit=0 elapsed=%.1fs "
                    "output_len=%d output_sha256=%s output_mode=%s request_count=%d",
                    model,
                    elapsed,
                    len(stdout),
                    _raw_output_digest(stdout),
                    "json_schema" if response_schema is not None else "json_object",
                    attempt + 1,
                )
                return parsed, content, stdout, attempt + 1

            combined = (stderr or "") + (stdout or "")
            self._log_process_failure(
                model=model,
                returncode=returncode,
                elapsed=elapsed,
                stdout=stdout,
                stderr=stderr,
                attempt=attempt + 1,
                response_schema=response_schema,
            )
            try:
                self._raise_process_error(
                    argv,
                    returncode,
                    stdout,
                    stderr,
                    response_schema=response_schema,
                    request_count=attempt + 1,
                )
            except _SchemaOutputUnsupported:
                raise
            except RuntimeError as error:
                last_err = error
            if attempt < MAX_CLI_RETRIES and _is_retryable(combined):
                delay = RETRY_BASE_DELAY * (2**attempt)
                log.info("Retryable error for %s, waiting %.0fs...", model, delay)
                time.sleep(delay)
                continue
            break

        raise last_err  # type: ignore[misc]

    def _log_process_failure(
        self,
        *,
        model: str,
        returncode: int,
        elapsed: float | None,
        stdout: str,
        stderr: str,
        attempt: int,
        response_schema: JsonSchemaSpec | None,
    ) -> None:
        combined = (stderr or "") + (stdout or "")
        category = _process_error_category(combined, response_schema)
        elapsed_text = "unknown" if elapsed is None else f"{elapsed:.1f}s"
        log.warning(
            "CLI judge failed: backend=cli model=%s exit=%d elapsed=%s "
            "stdout_len=%d stderr_len=%d output_sha256=%s output_mode=%s "
            "request_count=%d error_category=%s",
            model,
            returncode,
            elapsed_text,
            len(stdout or ""),
            len(stderr or ""),
            _raw_output_digest(combined),
            "json_schema" if response_schema is not None else "json_object",
            attempt,
            category,
        )

    def _raise_process_error(
        self,
        argv: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
        *,
        response_schema: JsonSchemaSpec | None,
        request_count: int,
    ) -> None:
        combined = (stderr or "") + (stdout or "")
        if response_schema is not None:
            reason = _explicit_schema_rejection_reason(combined)
            if reason is not None:
                raise _SchemaOutputUnsupported(
                    SCHEMA_FALLBACK_UNSUPPORTED,
                    request_count,
                )
        error = RuntimeError(
            f"{argv[0]} judge exited {returncode}: "
            f"error_category={_process_error_category(combined, response_schema)} "
            f"process_output_sha256={_raw_output_digest(combined)}"
        )
        _add_transport_request_count(error, request_count)
        raise error

    def _decode_output(self, stdout: str, mode: str, *, strict: bool) -> tuple[dict, str]:
        if strict and mode == "claude":
            try:
                envelope = json.loads(stdout)
            except json.JSONDecodeError:
                return {}, stdout
            if not isinstance(envelope, dict) or "structured_output" not in envelope:
                return {}, stdout
            structured = envelope["structured_output"]
            if isinstance(structured, str):
                content = structured
            else:
                content = json.dumps(structured, ensure_ascii=False)
            return _parse_exact_json_object(content), content

        content = self._extract_text(stdout, mode)
        if strict:
            return _parse_exact_json_object(content), content
        return _extract_json(content), content

    def _run_with_cancel(
        self, argv: list[str], stdin_text: str, env: dict[str, str]
    ) -> tuple[str, str, int]:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=_JUDGE_CWD,
        )
        with self._procs_lock:
            self._procs.add(proc)
        try:

            def _feed_stdin():
                try:
                    if stdin_text:
                        proc.stdin.write(stdin_text)
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            def _drain(pipe, chunks):
                try:
                    for chunk in iter(lambda: pipe.read(8192), ""):
                        chunks.append(chunk)
                except (ValueError, OSError):
                    pass

            writer = threading.Thread(target=_feed_stdin, daemon=True)
            reader_out = threading.Thread(
                target=_drain,
                args=(proc.stdout, stdout_chunks),
                daemon=True,
            )
            reader_err = threading.Thread(
                target=_drain,
                args=(proc.stderr, stderr_chunks),
                daemon=True,
            )
            writer.start()
            reader_out.start()
            reader_err.start()

            deadline = time.monotonic() + self._timeout
            while proc.poll() is None:
                if self._cancel and self._cancel.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise InferenceCancelled("cancelled")
                if time.monotonic() > deadline:
                    proc.kill()
                    raise subprocess.TimeoutExpired(argv, self._timeout)
                time.sleep(0.5)

            writer.join(timeout=5)
            reader_out.join(timeout=5)
            reader_err.join(timeout=5)
            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)
            return stdout, stderr, proc.returncode
        except InferenceCancelled:
            raise
        except Exception:
            if proc.poll() is None:
                proc.kill()
            raise
        finally:
            with self._procs_lock:
                self._procs.discard(proc)

    def _build(
        self,
        model: str,
        system: str,
        user: str,
        *,
        response_schema: JsonSchemaSpec | None = None,
        schema_path: str | None = None,
    ) -> tuple[list[str], str, str]:
        """Build (argv, stdin_text, mode) for the given judge model.

        Anthropic routes to ``claude`` and OpenAI routes to ``codex``. Other local
        families fail closed. ``mode`` is "claude" (answer is inside a JSON
        envelope) or "plain" (Codex prints the answer directly). The prompt is
        passed on stdin to avoid OS argument-length limits on big digests.
        """
        fam = model_family(model)
        if fam == "anthropic":
            alias = _CLAUDE_MODEL_ALIASES.get(model, model)
            argv = [
                "claude",
                "-p",
                "--output-format",
                "json",
                "--model",
                alias,
                "--tools",
                "",
                "--strict-mcp-config",
                "--no-session-persistence",
                "--safe-mode",
                "--disable-slash-commands",
                "--no-chrome",
            ]
            if response_schema is not None:
                argv += [
                    "--json-schema",
                    json.dumps(response_schema.schema, separators=(",", ":")),
                ]
            if system:
                argv += ["--system-prompt", system]
            return argv, user, "claude"

        if fam == "google":
            raise RuntimeError(
                "Google local CLI judges are disabled until Gemini has a verified confined mode"
            )

        if fam != "openai":
            raise RuntimeError(
                "The local CLI judge backend supports only Anthropic and OpenAI families"
            )

        argv = [
            "codex",
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            *CODEX_CONFINED_ARGS,
            "--model",
            model,
        ]
        if response_schema is not None:
            if schema_path is None:
                raise ValueError("Codex schema mode requires an output schema file")
            argv += ["--output-schema", schema_path]
        argv += ["-C", _JUDGE_CWD, "-"]
        prompt = f"{system}\n\n{user}" if system else user
        return argv, prompt, "plain"

    def _extract_text(self, stdout: str, mode: str) -> str:
        """Unwrap the assistant text from the CLI's stdout."""
        if mode == "claude":
            try:
                env = json.loads(stdout)
            except json.JSONDecodeError:
                return stdout
            if isinstance(env, dict) and "result" in env:
                return str(env["result"])
        return stdout

    def close(self) -> None:
        self.abort()

    def __enter__(self) -> "CliJudgeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
