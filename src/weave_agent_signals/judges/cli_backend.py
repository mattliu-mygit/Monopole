"""Confined local-provider implementation of the judge chat interface.

The selected descriptor binds this client to ``claude``, ``codex``, or ``agy``.
Model selection happens before this transport boundary; this module only
executes the exact provider model with a restricted environment.
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
from contextlib import contextmanager
from typing import Any, Callable, Literal, Mapping

from weave_agent_signals.judges.agy_transport import agy_prompt_invocation
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


def _unwrap_whole_json_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(\{.*\})\s*```\s*", text, re.DOTALL)
    return match.group(1) if match is not None else text


_HOME = os.path.expanduser("~")
_JUDGE_CODEX_HOME = os.path.join(_HOME, ".codex-judge")
_JUDGE_CWD = os.path.join(_HOME, ".codex-judge", "sandbox")

_RETRYABLE_PATTERNS = [
    "error_max_structured_output_retries",
    "at capacity",
    "rate limit",
    "too many requests",
    "overloaded",
    "timed out",
    "timeout",
    "connection error",
    "connection reset",
    "network error",
    "econnreset",
    "internal server error",
    " 500",
    " 502",
    "503",
    " 504",
    "429",
]

MAX_CLI_RETRIES = 2
RETRY_BASE_DELAY = 5.0
_PROCESS_DIAGNOSTIC_TAIL_CHARACTERS = 16_000
_PROVIDER_TEXT_CHARACTERS = 500
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|secret|password|authorization|credential)\b\s*[:=]\s*)"
    r"[^\s,;}]+"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[^\s,;}]+")
_PROVIDER_CODE = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")


def _process_diagnostic(output: str) -> str:
    return output[-_PROCESS_DIAGNOSTIC_TAIL_CHARACTERS:]


def _failure_diagnostic(stdout: str, stderr: str, prompt: str) -> str:
    """Remove Codex's exact prompt echo before retaining failure metadata."""
    combined = (stderr or "") + (stdout or "")
    return combined.replace(prompt, "") if prompt else combined


def _safe_provider_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    text = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", text)
    text = _BEARER_SECRET.sub("Bearer [REDACTED]", text)
    return text[:_PROVIDER_TEXT_CHARACTERS] or None


def _safe_provider_code(value: object) -> str | None:
    if not isinstance(value, str) or _PROVIDER_CODE.fullmatch(value) is None:
        return None
    return value


def _canonical_provider_message(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lower = value.lower()
    for message, patterns in (
        ("Provider request timed out.", ("timed out", "timeout")),
        (
            "Provider connection failed.",
            ("connection error", "connection reset", "network error", "econnreset"),
        ),
        ("Provider rate limit was reached.", ("rate limit", "too many requests", "429")),
        (
            "Provider service was unavailable.",
            ("at capacity", "overloaded", "internal server error", " 500", " 502", "503", " 504"),
        ),
        (
            "Provider authentication failed.",
            ("authentication", "unauthorized", "invalid api key"),
        ),
    ):
        if any(pattern in lower for pattern in patterns):
            return message
    return None


def _provider_issue(output: str) -> dict[str, object]:
    """Extract only allowlisted fields from a CLI provider error envelope."""
    decoder = json.JSONDecoder()
    issue: dict[str, object] = {}
    index = output.find("{")
    while index != -1:
        try:
            payload, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping) and isinstance(payload.get("error"), Mapping):
            error = payload["error"]
            candidate: dict[str, object] = {}
            status = payload.get("status")
            if type(status) is int and 100 <= status <= 599:
                candidate["provider_status"] = status
            code = _safe_provider_text(error.get("code"))
            if code is not None:
                candidate["provider_error_code"] = code
            message = _safe_provider_text(error.get("message"))
            if message is not None:
                candidate["provider_error_message"] = message
            if candidate:
                issue = candidate
        elif isinstance(payload, Mapping) and payload.get("is_error") is True:
            candidate = {}
            code = _safe_provider_code(payload.get("subtype"))
            if code is not None:
                candidate["provider_error_code"] = code
            message = (
                "Provider could not produce schema-valid output."
                if code == "error_max_structured_output_retries"
                else _canonical_provider_message(payload.get("result"))
            )
            if message is not None:
                candidate["provider_error_message"] = message
            if candidate:
                issue = candidate
        index = output.find("{", index + 1)
    return issue


def _codex_compatible_schema(value: Any) -> Any:
    """Copy a JSON Schema while dropping keywords Codex rejects."""
    if isinstance(value, dict):
        return {
            key: _codex_compatible_schema(item)
            for key, item in value.items()
            if key != "uniqueItems"
        }
    if isinstance(value, list):
        return [_codex_compatible_schema(item) for item in value]
    return value


def _is_retryable(output: str) -> bool:
    lower = _process_diagnostic(output).lower()
    return any(p in lower for p in _RETRYABLE_PATTERNS)


def _retry_reason(output: str) -> str | None:
    lower = _process_diagnostic(output).lower()
    for reason, patterns in (
        (
            "structured_output",
            ("error_max_structured_output_retries",),
        ),
        ("rate_limit", ("rate limit", "too many requests", "429")),
        ("capacity", ("at capacity",)),
        ("overloaded", ("overloaded",)),
        ("timeout", ("timed out", "timeout")),
        (
            "connection",
            ("connection error", "connection reset", "network error", "econnreset"),
        ),
        ("server_error", ("internal server error", " 500", " 502", "503", " 504")),
    ):
        if any(pattern in lower for pattern in patterns):
            return reason
    return None


def _process_error_category(
    output: str,
    response_schema: JsonSchemaSpec | None,
) -> str:
    diagnostic = _process_diagnostic(output)
    if "error_max_structured_output_retries" in diagnostic.lower():
        return "structured_output_retry_exhausted"
    if response_schema is not None and _explicit_schema_rejection_reason(diagnostic):
        return "schema_output_unsupported"
    if response_schema is not None and any(
        marker in diagnostic.lower()
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


def _missing_required_schema_fields(
    parsed: Mapping[str, Any],
    response_schema: JsonSchemaSpec | None,
) -> tuple[str, ...]:
    if response_schema is None:
        return ()
    required = response_schema.schema.get("required")
    if not isinstance(required, list):
        return ()
    return tuple(field for field in required if isinstance(field, str) and field not in parsed)


def _build_env(provider: str) -> dict[str, str]:
    return prepare_cli_subprocess(
        home=_HOME,
        codex_home=_JUDGE_CODEX_HOME,
        cwd=_JUDGE_CWD,
        provider=provider,
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

    def __init__(
        self,
        provider: str,
        timeout: float = 360.0,
        runner: Callable[..., Any] | None = None,
    ):
        if provider not in {"claude", "codex", "agy"}:
            raise ValueError(f"unsupported local CLI provider: {provider}")
        self.backend = provider
        self._provider = provider
        self._timeout = timeout
        self._runner = runner
        self._activity: Callable[[dict[str, object]], None] | None = None
        self._procs: set[subprocess.Popen] = set()
        self._procs_lock = threading.Lock()
        self._cancel: threading.Event | None = None
        self._abort_requested = threading.Event()

    def set_cancel(self, cancel: threading.Event) -> None:
        self._cancel = cancel

    def set_activity(self, callback: Callable[[dict[str, object]], None]) -> None:
        self._activity = callback

    def _emit_activity(self, event: dict[str, object]) -> None:
        if self._activity is None:
            return
        try:
            self._activity(dict(event))
        except Exception as error:
            log.warning(
                "CLI judge activity callback failed: error_type=%s",
                type(error).__name__,
            )

    def abort(self) -> None:
        self._abort_requested.set()
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
        reasoning: Literal["default", "disabled"] = "default",
    ) -> tuple[dict[str, Any], JudgeResponse]:
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        user = "\n\n".join(m["content"] for m in messages if m.get("role") == "user")
        env = _build_env(self._provider)
        schema_path: str | None = None
        if response_schema is not None and self._provider == "codex":
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
                json.dump(_codex_compatible_schema(response_schema.schema), schema_file)
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
        if self._provider == "agy":
            request_prompt = self._agy_prompt(
                system,
                user,
                response_schema=response_schema,
            )
            base_argv: list[str] | None = None
            base_stdin = ""
            mode = "plain"
        else:
            base_argv, base_stdin, mode = self._build(
                model,
                system,
                user,
                response_schema=response_schema,
                schema_path=schema_path,
            )
            request_prompt = base_stdin

        @contextmanager
        def attempt_invocation():
            if self._provider == "agy":
                with agy_prompt_invocation(
                    prompt=request_prompt,
                    model=model,
                    home=_HOME,
                ) as invocation:
                    attempt_env = dict(env)
                    attempt_env["HOME"] = invocation.home
                    yield invocation.argv, "", invocation.cwd, attempt_env
                return
            if base_argv is None:  # pragma: no cover - guarded by provider branch above
                raise RuntimeError("local CLI argv was not built")
            yield base_argv, base_stdin, _JUDGE_CWD, env

        log.info(
            "CLI judge started: provider=%s model=%s family=%s prompt_len=%d output_mode=%s",
            self._provider,
            model,
            model_family(model),
            len(request_prompt),
            "json_schema" if response_schema is not None else "json_object",
        )

        if self._runner:
            with attempt_invocation() as (argv, stdin_text, cwd, attempt_env):
                proc = self._runner(
                    argv,
                    input=stdin_text,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    env=attempt_env,
                    cwd=cwd,
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
                    prompt=request_prompt,
                )
                self._raise_process_error(
                    argv,
                    proc.returncode,
                    stdout,
                    stderr,
                    response_schema=response_schema,
                    request_count=1,
                    prompt=request_prompt,
                )
            parsed, content = self._decode_output(
                stdout,
                mode,
                strict=response_schema is not None,
            )
            log.info(
                "CLI judge completed: provider=%s model=%s exit=0 output_len=%d "
                "output_sha256=%s output_mode=%s request_count=1",
                self._provider,
                model,
                len(stdout),
                _raw_output_digest(stdout),
                "json_schema" if response_schema is not None else "json_object",
            )
            return parsed, content, stdout, 1

        last_err: RuntimeError | None = None
        invocation_started = time.monotonic()
        for attempt in range(MAX_CLI_RETRIES + 1):
            t0 = time.monotonic()
            with attempt_invocation() as (argv, stdin_text, cwd, attempt_env):
                if self._provider == "agy":
                    stdout, stderr, returncode = self._run_with_cancel(
                        argv,
                        stdin_text,
                        attempt_env,
                        cwd=cwd,
                    )
                else:
                    stdout, stderr, returncode = self._run_with_cancel(
                        argv,
                        stdin_text,
                        attempt_env,
                    )
            elapsed = time.monotonic() - t0

            if returncode == 0:
                parsed, content = self._decode_output(
                    stdout,
                    mode,
                    strict=response_schema is not None,
                )
                missing_fields = (
                    _missing_required_schema_fields(parsed, response_schema)
                    if self._provider == "agy"
                    else ()
                )
                if missing_fields:
                    output_sha256 = _raw_output_digest(stdout)
                    if attempt < MAX_CLI_RETRIES:
                        self._emit_activity(
                            {
                                "phase": "transport_retry",
                                "message": (
                                    f"{model} returned schema-invalid output after "
                                    f"{elapsed:.1f}s; retrying attempt {attempt + 2} of "
                                    f"{MAX_CLI_RETRIES + 1}"
                                ),
                                "model": model,
                                "request_attempt": attempt + 2,
                                "max_attempts": MAX_CLI_RETRIES + 1,
                                "elapsed_seconds": round(elapsed, 3),
                                "error_category": "schema_validation_error",
                                "retry_reason": "schema_output",
                                "exit_code": 0,
                                "stdout_chars": len(stdout),
                                "stderr_chars": len(stderr),
                                "prompt_characters": len(request_prompt),
                                "output_mode": "json_schema",
                                "output_sha256": output_sha256,
                                "missing_required_field_count": len(missing_fields),
                            }
                        )
                        delay = RETRY_BASE_DELAY * (2**attempt)
                        log.info(
                            "Agy returned schema-invalid output for %s, waiting %.0fs...",
                            model,
                            delay,
                        )
                        time.sleep(delay)
                        continue
                    self._emit_activity(
                        {
                            "phase": "transport_failed",
                            "message": (f"{model} failed after {attempt + 1} request attempts"),
                            "model": model,
                            "request_attempt": attempt + 1,
                            "max_attempts": MAX_CLI_RETRIES + 1,
                            "elapsed_seconds": round(
                                time.monotonic() - invocation_started,
                                3,
                            ),
                            "error_category": "schema_validation_error",
                            "retry_reason": "schema_output",
                            "exit_code": 0,
                            "stdout_chars": len(stdout),
                            "stderr_chars": len(stderr),
                            "prompt_characters": len(request_prompt),
                            "output_mode": "json_schema",
                            "output_sha256": output_sha256,
                            "missing_required_field_count": len(missing_fields),
                        }
                    )
                    error = RuntimeError(
                        "agy judge returned schema-invalid output: "
                        "error_category=schema_validation_error"
                    )
                    _add_transport_request_count(error, attempt + 1)
                    raise error
                log.info(
                    "CLI judge completed: provider=%s model=%s exit=0 elapsed=%.1fs "
                    "output_len=%d output_sha256=%s output_mode=%s request_count=%d",
                    self._provider,
                    model,
                    elapsed,
                    len(stdout),
                    _raw_output_digest(stdout),
                    "json_schema" if response_schema is not None else "json_object",
                    attempt + 1,
                )
                if attempt > 0:
                    self._emit_activity(
                        {
                            "phase": "transport_recovered",
                            "message": (
                                f"{model} recovered on request attempt {attempt + 1} of "
                                f"{MAX_CLI_RETRIES + 1}"
                            ),
                            "model": model,
                            "request_attempt": attempt + 1,
                            "max_attempts": MAX_CLI_RETRIES + 1,
                            "elapsed_seconds": round(
                                time.monotonic() - invocation_started,
                                3,
                            ),
                            "stdout_chars": len(stdout),
                            "stderr_chars": len(stderr),
                            "prompt_characters": len(request_prompt),
                            "output_mode": (
                                "json_schema" if response_schema is not None else "json_object"
                            ),
                            "output_sha256": _raw_output_digest(stdout),
                        }
                    )
                return parsed, content, stdout, attempt + 1

            diagnostic = _failure_diagnostic(stdout, stderr, request_prompt)
            provider_issue = _provider_issue(_process_diagnostic(diagnostic))
            self._log_process_failure(
                model=model,
                returncode=returncode,
                elapsed=elapsed,
                stdout=stdout,
                stderr=stderr,
                attempt=attempt + 1,
                response_schema=response_schema,
                prompt=request_prompt,
            )
            try:
                self._raise_process_error(
                    argv,
                    returncode,
                    stdout,
                    stderr,
                    response_schema=response_schema,
                    request_count=attempt + 1,
                    prompt=request_prompt,
                )
            except _SchemaOutputUnsupported:
                raise
            except RuntimeError as error:
                last_err = error
            category = _process_error_category(diagnostic, response_schema)
            output_sha256 = _raw_output_digest(diagnostic)
            if attempt < MAX_CLI_RETRIES and _is_retryable(diagnostic):
                self._emit_activity(
                    {
                        "phase": "transport_retry",
                        "message": (
                            f"{model} request failed after {elapsed:.1f}s; retrying attempt "
                            f"{attempt + 2} of {MAX_CLI_RETRIES + 1}"
                        ),
                        "model": model,
                        "request_attempt": attempt + 2,
                        "max_attempts": MAX_CLI_RETRIES + 1,
                        "elapsed_seconds": round(elapsed, 3),
                        "error_category": category,
                        "retry_reason": _retry_reason(diagnostic),
                        "exit_code": returncode,
                        "stdout_chars": len(stdout),
                        "stderr_chars": len(stderr),
                        "prompt_characters": len(request_prompt),
                        "output_mode": (
                            "json_schema" if response_schema is not None else "json_object"
                        ),
                        "output_sha256": output_sha256,
                        **provider_issue,
                    }
                )
                delay = RETRY_BASE_DELAY * (2**attempt)
                log.info("Retryable error for %s, waiting %.0fs...", model, delay)
                time.sleep(delay)
                continue
            self._emit_activity(
                {
                    "phase": "transport_failed",
                    "message": (
                        f"{model} failed after {attempt + 1} request attempt"
                        f"{'s' if attempt else ''}"
                    ),
                    "model": model,
                    "request_attempt": attempt + 1,
                    "max_attempts": MAX_CLI_RETRIES + 1,
                    "elapsed_seconds": round(
                        time.monotonic() - invocation_started,
                        3,
                    ),
                    "error_category": category,
                    "retry_reason": _retry_reason(diagnostic),
                    "exit_code": returncode,
                    "stdout_chars": len(stdout),
                    "stderr_chars": len(stderr),
                    "prompt_characters": len(request_prompt),
                    "output_mode": (
                        "json_schema" if response_schema is not None else "json_object"
                    ),
                    "output_sha256": output_sha256,
                    **provider_issue,
                }
            )
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
        prompt: str,
    ) -> None:
        diagnostic = _failure_diagnostic(stdout, stderr, prompt)
        category = _process_error_category(diagnostic, response_schema)
        provider_issue = _provider_issue(_process_diagnostic(diagnostic))
        elapsed_text = "unknown" if elapsed is None else f"{elapsed:.1f}s"
        log.warning(
            "CLI judge failed: provider=%s model=%s exit=%d elapsed=%s "
            "stdout_len=%d stderr_len=%d output_sha256=%s output_mode=%s "
            "request_count=%d error_category=%s provider_status=%s "
            "provider_error_code=%s provider_error_message=%s",
            self._provider,
            model,
            returncode,
            elapsed_text,
            len(stdout or ""),
            len(stderr or ""),
            _raw_output_digest(diagnostic),
            "json_schema" if response_schema is not None else "json_object",
            attempt,
            category,
            provider_issue.get("provider_status"),
            provider_issue.get("provider_error_code"),
            provider_issue.get("provider_error_message"),
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
        prompt: str,
    ) -> None:
        diagnostic = _failure_diagnostic(stdout, stderr, prompt)
        if response_schema is not None:
            reason = _explicit_schema_rejection_reason(_process_diagnostic(diagnostic))
            if reason is not None:
                raise _SchemaOutputUnsupported(
                    SCHEMA_FALLBACK_UNSUPPORTED,
                    request_count,
                )
        error = RuntimeError(
            f"{argv[0]} judge exited {returncode}: "
            f"error_category={_process_error_category(diagnostic, response_schema)} "
            f"process_output_sha256={_raw_output_digest(diagnostic)}"
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
            if self._provider == "agy":
                content = _unwrap_whole_json_fence(content)
            return _parse_exact_json_object(content), content
        return _extract_json(content), content

    def _run_with_cancel(
        self,
        argv: list[str],
        stdin_text: str,
        env: dict[str, str],
        *,
        cwd: str | None = None,
    ) -> tuple[str, str, int]:
        if self._abort_requested.is_set() or (self._cancel and self._cancel.is_set()):
            raise InferenceCancelled("cancelled")
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=_JUDGE_CWD if cwd is None else cwd,
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
            timed_out = False
            while proc.poll() is None:
                if self._abort_requested.is_set() or (self._cancel and self._cancel.is_set()):
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise InferenceCancelled("cancelled")
                if time.monotonic() > deadline:
                    proc.kill()
                    proc.wait(timeout=5)
                    timed_out = True
                    break
                time.sleep(0.5)

            if self._abort_requested.is_set() or (self._cancel and self._cancel.is_set()):
                raise InferenceCancelled("cancelled")

            writer.join(timeout=5)
            reader_out.join(timeout=5)
            reader_err.join(timeout=5)
            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)
            if timed_out:
                timeout_message = f"process timed out after {self._timeout:g}s"
                stderr = f"{stderr}\n{timeout_message}".strip()
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

        The client is bound to one explicit provider. ``mode`` is "claude"
        (answer is inside a JSON envelope) or "plain" (Codex prints the answer
        directly). Claude and Codex read prompts from stdin. Antigravity uses
        the separate attempt-scoped prompt-file transport.
        """
        if self._provider == "claude":
            argv = [
                "claude",
                "-p",
                "--output-format",
                "json",
                "--model",
                model,
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

        prompt = f"{system}\n\n{user}" if system else user
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
        return argv, prompt, "plain"

    @staticmethod
    def _agy_prompt(
        system: str,
        user: str,
        *,
        response_schema: JsonSchemaSpec | None,
    ) -> str:
        sections = [system] if system else []
        if response_schema is not None:
            sections.append(
                "Return exactly one JSON object matching this JSON Schema:\n"
                + json.dumps(response_schema.schema, separators=(",", ":"))
            )
        sections.append(user)
        return "\n\n".join(sections)

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
