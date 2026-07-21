"""Chat-completions client for the catalog-backed OpenAI and W&B providers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from weave_agent_signals.client import _get_api_key

log = logging.getLogger("weave_agent_signals.judges")

INFERENCE_BASE = "https://api.inference.wandb.ai/v1"
OPENAI_BASE = "https://api.openai.com/v1"

DEFAULT_ENTITY = "weave-team"
DEFAULT_PROJECT = "agent-sessions"
SCHEMA_FALLBACK_UNSUPPORTED = "schema_output_unsupported"
SCHEMA_FALLBACK_RETRY = "retry_recovery"


@dataclass(frozen=True)
class JsonSchemaSpec:
    """Named JSON Schema requested at the shared model transport boundary."""

    name: str
    schema: Mapping[str, Any]
    examples: tuple[Mapping[str, Any], ...] = ()


class InferenceResponseDiagnostic(BaseModel):
    """Bounded metadata for one provider response, excluding model output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finish_reason: str | None = Field(default=None, max_length=100)
    usage: dict[str, StrictInt] = Field(default_factory=dict)
    completion_details: dict[str, StrictInt] = Field(default_factory=dict)
    content_characters: int = Field(ge=0)

    @field_validator("usage", "completion_details")
    @classmethod
    def _valid_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key.strip() for key in value):
            raise ValueError("diagnostic token keys must be nonblank")
        if any(type(count) is not int or count < 0 for count in value.values()):
            raise ValueError("diagnostic token counts must be nonnegative integers")
        return value


@dataclass
class JudgeResponse:
    content: str
    model: str
    usage: Mapping[str, Any]
    output_mode: Literal["json_object", "json_schema", "json_object_fallback"] = "json_object"
    schema_name: str | None = None
    schema_fallback_reason: str | None = None
    transport_request_count: int = 1
    raw_output_digest: str | None = None
    response_diagnostics: tuple[InferenceResponseDiagnostic, ...] = ()


@runtime_checkable
class ChatClient(Protocol):
    """Minimal inference interface consumed by judge execution."""

    backend: str

    def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: JsonSchemaSpec | None = None,
        reasoning: Literal["default", "disabled"] = "default",
    ) -> tuple[dict[str, Any], JudgeResponse]: ...


class InferenceCancelled(RuntimeError):
    """Inference stopped because the owning run was cancelled."""


class InferenceContextExceeded(RuntimeError):
    """The provider explicitly rejected a request for exceeding context capacity."""

    def __init__(self) -> None:
        super().__init__("provider context capacity exceeded")


class InferenceOutputExceeded(RuntimeError):
    """The provider spent its full generation budget without valid JSON output."""

    def __init__(self, response: JudgeResponse) -> None:
        super().__init__("provider output limit exhausted")
        self.response = response
        self._transport_request_count = response.transport_request_count


_SCHEMA_TOKEN = (
    r"(?:json[_ -]?schema|response[_ -]?format|structured[- ]outputs?|"
    r"schema[- ]output|output[- ]schema|--json-schema|--output-schema)"
)
_WHOLE_FEATURE_END = (
    r"(?:$|[.;\n]|(?:by|for|on)\s+(?:(?:this|the)\s+)?"
    r"(?:model|backend|provider|cli|command|version|endpoint)\b)"
)
_SCHEMA_REJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"(?:unknown|unexpected|unrecognized)\s+(?:option|argument|parameter)\s*"
        rf"[:=]?\s*['\"]?{_SCHEMA_TOKEN}",
        rf"{_SCHEMA_TOKEN}(?:[ \t]+(?:mode|feature|type))?[ \t]+(?:"
        rf"(?:(?:is|are|was|were)[ \t]+)?not[ \t]+supported|"
        rf"(?:(?:is|are|was|were)[ \t]+)?(?:unsupported|unavailable)"
        rf"(?=[ \t]*{_WHOLE_FEATURE_END}))",
        rf"(?:model|backend|provider|cli|command|version|endpoint)\b[^.;\n]{{0,80}}\b"
        rf"(?:does\s+not|doesn't|cannot|can't)\s+support\s+(?:the\s+)?{_SCHEMA_TOKEN}",
        rf"\bunsupported\s+(?:(?:option|argument|parameter|feature)\s*[:=]?\s*)?"
        rf"['\"]?{_SCHEMA_TOKEN}(?=[ \t]*{_WHOLE_FEATURE_END})",
        rf"\bsupport\s+for\s+(?:the\s+)?{_SCHEMA_TOKEN}\s+"
        rf"(?:is\s+)?(?:unsupported|unavailable)",
    )
)
_NON_FALLBACK_FAILURE_TERMS = (
    "authentication",
    "api key",
    "unauthorized",
    "forbidden",
    "rate limit",
    "too many requests",
    "429",
    "server error",
    "internal server",
    "service unavailable",
    "overloaded",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "cancelled",
    "canceled",
)
_SCHEMA_VALIDATION_FAILURE_TERMS = (
    "validation failed",
    "validation error",
    "failed validation",
    "failed to validate",
    "invalid_json_schema",
    "invalid json schema",
    "invalid schema",
    "schema is invalid",
    "schema mismatch",
    "does not conform",
)


def _raw_output_digest(raw_output: str) -> str:
    return hashlib.sha256(raw_output.encode("utf-8")).hexdigest()


_OUTPUT_CONTRACT_MARKER = "REQUIRED_JSON_SCHEMA:"


def _output_contract_messages(
    messages: list[dict[str, str]],
    schema: Mapping[str, Any],
    examples: tuple[Mapping[str, Any], ...] = (),
) -> list[dict[str, str]]:
    recovered = [dict(message) for message in messages]
    if any(
        message.get("role") == "system" and _OUTPUT_CONTRACT_MARKER in message.get("content", "")
        for message in recovered
    ):
        return recovered
    sections = [
        _OUTPUT_CONTRACT_MARKER
        + "\n"
        + json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    ]
    sections.extend(
        f"CANONICAL_JSON_EXAMPLE_{index}:\n"
        + json.dumps(
            example,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        for index, example in enumerate(examples, 1)
    )
    instruction = "\n".join(sections)
    if recovered and recovered[0].get("role") == "system":
        recovered[0]["content"] = f"{recovered[0].get('content', '')}\n\n{instruction}"
    else:
        recovered.insert(0, {"role": "system", "content": instruction})
    return recovered


def json_output_contract_messages(
    messages: list[dict[str, str]],
    response_schema: JsonSchemaSpec,
) -> list[dict[str, str]]:
    """Attach one exact schema and its canonical examples to model messages."""

    return _output_contract_messages(
        messages,
        response_schema.schema,
        response_schema.examples,
    )


def _json_object_recovery_messages(
    messages: list[dict[str, str]],
    response_format: Mapping[str, Any],
) -> list[dict[str, str]]:
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, Mapping) or not isinstance(json_schema.get("schema"), Mapping):
        return messages
    return _output_contract_messages(messages, json_schema["schema"])


def _wandb_retry_body(body: dict[str, Any]) -> dict[str, Any]:
    recovered = dict(body)
    recovered["chat_template_kwargs"] = {"enable_thinking": False}
    response_format = recovered.get("response_format")
    if isinstance(response_format, Mapping) and response_format.get("type") == "json_schema":
        recovered["response_format"] = {"type": "json_object"}
        recovered["messages"] = _json_object_recovery_messages(body["messages"], response_format)
    return recovered


def _token_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: count
        for key, count in value.items()
        if isinstance(key, str) and key.strip() and type(count) is int and count >= 0
    }


def _parse_exact_json_object(text: str) -> dict[str, Any]:
    """Parse one complete JSON object without searching prose or fences."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _completion_limit_reached(response: JudgeResponse, max_tokens: int) -> bool:
    completion_tokens = response.usage.get("completion_tokens")
    return type(completion_tokens) is int and completion_tokens >= max_tokens


def _combined_usage(*values: Mapping[str, Any]) -> dict[str, int]:
    combined: dict[str, int] = {}
    for value in values:
        for key, count in value.items():
            if isinstance(key, str) and key.strip() and type(count) is int and count >= 0:
                combined[key] = combined.get(key, 0) + count
    return combined


def _explicit_schema_rejection_reason(text: str) -> str | None:
    """Return a bounded explicit schema-unsupported diagnostic, if present."""
    reason = " ".join(text.split())
    lowered = reason.lower()
    if any(term in lowered for term in _NON_FALLBACK_FAILURE_TERMS):
        return None
    if "invalid_json_schema" in lowered and re.search(
        r"['\"][A-Za-z][A-Za-z0-9_-]*['\"]\s+is not permitted\b",
        reason,
        re.IGNORECASE,
    ):
        return reason[:500]
    if any(term in lowered for term in _SCHEMA_VALIDATION_FAILURE_TERMS):
        return None
    if not any(pattern.search(text) for pattern in _SCHEMA_REJECTION_PATTERNS):
        return None
    return reason[:500]


def _http_schema_rejection_reason(error: httpx.HTTPStatusError) -> str | None:
    """Recognize only explicit client-side schema capability rejections."""
    response = error.response
    if response.status_code not in {400, 404, 405, 415, 422}:
        return None

    reason = response.text
    message = reason
    param = ""
    code = ""
    error_type = ""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, Mapping):
        detail = payload.get("error", payload)
        if isinstance(detail, Mapping):
            if isinstance(detail.get("message"), str):
                message = reason = detail["message"]
            if isinstance(detail.get("param"), str):
                param = detail["param"]
            if isinstance(detail.get("code"), str):
                code = detail["code"]
            if isinstance(detail.get("type"), str):
                error_type = detail["type"]

    category_text = " ".join((message, code, error_type)).lower()
    if any(term in category_text for term in _NON_FALLBACK_FAILURE_TERMS):
        return None
    if any(term in category_text for term in _SCHEMA_VALIDATION_FAILURE_TERMS):
        return None
    if re.search(_SCHEMA_TOKEN, param, re.IGNORECASE) and code.lower() in {
        "unsupported_value",
        "unsupported_parameter",
        "unknown_parameter",
        "unsupported_response_format",
    }:
        return SCHEMA_FALLBACK_UNSUPPORTED
    if _explicit_schema_rejection_reason(message) is None:
        return None
    return SCHEMA_FALLBACK_UNSUPPORTED


def _is_http_context_rejection(error: httpx.HTTPStatusError) -> bool:
    response = error.response
    if response.status_code not in {400, 413, 422}:
        return False
    text = response.text
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, Mapping):
        detail = payload.get("error", payload)
        if isinstance(detail, Mapping):
            text = " ".join(str(detail.get(key, "")) for key in ("message", "code", "type"))
    lowered = " ".join(text.lower().split())
    return bool(re.search(r"max_tokens must be at least 1, got -\d+", lowered)) or any(
        marker in lowered
        for marker in (
            "context_length_exceeded",
            "maximum context length",
            "context window",
            "too many tokens",
            "input is too long",
            "prompt is too long",
        )
    )


def _add_transport_request_count(error: Exception, additional: int) -> None:
    """Attach a cumulative safe transport-attempt count to a raised error."""

    current = getattr(error, "_transport_request_count", 0)
    if type(current) is not int or current < 0:
        current = 0
    error._transport_request_count = current + additional  # type: ignore[attr-defined]


def _resolve_backend(
    backend: str | None = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> tuple[str, dict[str, str], str]:
    """Resolve backend string to (base_url, headers, backend_name)."""
    backend = backend or os.environ.get("JUDGE_BACKEND", "openai")

    if backend == "wandb":
        api_key = _get_api_key()
        return (
            INFERENCE_BASE,
            {
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Project": f"{entity}/{project}",
                "Content-Type": "application/json",
            },
            "wandb",
        )

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Export it or use --judge-backend=wandb")
        return (
            OPENAI_BASE,
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "openai",
        )

    raise ValueError(f"unsupported judge backend: {backend}")


class InferenceClient:
    """Chat-completions client restricted to supported catalog backends."""

    backend: str

    def __init__(
        self,
        entity: str = DEFAULT_ENTITY,
        project: str = DEFAULT_PROJECT,
        backend: str | None = None,
    ):
        resolved_url, headers, self.backend = _resolve_backend(backend, entity, project)

        self._http = httpx.Client(
            base_url=resolved_url,
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0),
        )
        self._activity = threading.local()
        self._schema_recovery: set[tuple[str, str]] = set()

    def set_activity(self, callback: Callable[[dict[str, object]], None]) -> None:
        self._activity.callback = callback

    def _emit_activity(self, event: dict[str, object]) -> None:
        callback = getattr(self._activity, "callback", None)
        if callback is None:
            return
        try:
            callback(dict(event))
        except Exception as error:
            log.warning(
                "HTTP judge activity callback failed: error_type=%s",
                type(error).__name__,
            )

    def _post_with_retry(
        self,
        path: str,
        body: dict,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> tuple[dict, int]:
        request_count = 0
        started = time.monotonic()
        model = str(body.get("model") or "inference model")
        wandb_retry = self.backend == "wandb"
        schema_retry = (
            wandb_retry
            and isinstance(body.get("response_format"), Mapping)
            and body["response_format"].get("type") == "json_schema"
        )
        retry_note = " in concise JSON recovery mode" if wandb_retry else ""
        retry_fields = {"output_mode": "json_object_fallback"} if schema_retry else {}
        while True:
            request_count += 1
            retry_mode = wandb_retry and request_count > 1
            request_body = _wandb_retry_body(body) if retry_mode else body
            output_mode = "json_object_fallback" if retry_mode and schema_retry else None
            attempt_note = retry_note if retry_mode else ""
            attempt_started = time.monotonic()
            self._emit_activity(
                {
                    "phase": "transport_attempt_started",
                    "message": (
                        f"{model} provider request attempt {request_count} started{attempt_note}"
                    ),
                    "model": model,
                    "request_attempt": request_count,
                    "max_attempts": max_attempts,
                    "elapsed_seconds": round(attempt_started - started, 3),
                    **({"output_mode": output_mode} if output_mode else {}),
                }
            )
            try:
                resp = self._http.post(path, json=request_body)
            except httpx.TransportError as error:
                attempt_elapsed = time.monotonic() - attempt_started
                self._emit_activity(
                    {
                        "phase": "transport_attempt_completed",
                        "message": (
                            f"{model} provider request attempt {request_count} failed "
                            f"after {attempt_elapsed:.1f}s"
                        ),
                        "model": model,
                        "request_attempt": request_count,
                        "max_attempts": max_attempts,
                        "elapsed_seconds": round(attempt_elapsed, 3),
                        "error_category": type(error).__name__,
                        "status": "failed",
                    }
                )
                elapsed = time.monotonic() - started
                if request_count < max_attempts:
                    self._emit_activity(
                        {
                            "phase": "transport_retry",
                            "message": (
                                f"{model} {type(error).__name__} after {elapsed:.1f}s; "
                                f"retrying attempt {request_count + 1} of {max_attempts}"
                                f"{retry_note}"
                            ),
                            "model": model,
                            "request_attempt": request_count + 1,
                            "max_attempts": max_attempts,
                            "elapsed_seconds": round(elapsed, 3),
                            "error_category": type(error).__name__,
                            "retry_reason": "transport_error",
                            **retry_fields,
                        }
                    )
                    log.info(
                        "Transient inference failure, retrying in %.1fs (attempt %d/%d)",
                        base_delay,
                        request_count,
                        max_attempts,
                    )
                    time.sleep(base_delay)
                    continue
                error._transport_request_count = request_count  # type: ignore[attr-defined]
                self._emit_activity(
                    {
                        "phase": "transport_failed",
                        "message": (
                            f"{model} failed after {request_count} provider attempts: "
                            f"{type(error).__name__}"
                        ),
                        "model": model,
                        "request_attempt": request_count,
                        "max_attempts": max_attempts,
                        "elapsed_seconds": round(elapsed, 3),
                        "error_category": type(error).__name__,
                        "retry_reason": "transport_error",
                    }
                )
                raise
            attempt_elapsed = time.monotonic() - attempt_started
            response_succeeded = resp.status_code < 400
            self._emit_activity(
                {
                    "phase": "transport_attempt_completed",
                    "message": (
                        f"{model} provider request attempt {request_count} completed "
                        f"in {attempt_elapsed:.1f}s"
                    ),
                    "model": model,
                    "request_attempt": request_count,
                    "max_attempts": max_attempts,
                    "elapsed_seconds": round(attempt_elapsed, 3),
                    "provider_status": resp.status_code,
                    "status": "succeeded" if response_succeeded else "failed",
                    **({"output_mode": output_mode} if output_mode else {}),
                }
            )
            if resp.status_code == 429 and request_count < max_attempts:
                delay = base_delay * (2 ** (request_count - 1))
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                elapsed = time.monotonic() - started
                self._emit_activity(
                    {
                        "phase": "transport_retry",
                        "message": (
                            f"{model} received HTTP 429 after {elapsed:.1f}s; retrying "
                            f"attempt {request_count + 1} of {max_attempts}"
                            f"{retry_note}"
                        ),
                        "model": model,
                        "request_attempt": request_count + 1,
                        "max_attempts": max_attempts,
                        "elapsed_seconds": round(elapsed, 3),
                        "error_category": "rate_limit",
                        "retry_reason": "http_429",
                        "provider_status": 429,
                        **retry_fields,
                    }
                )
                log.info(
                    "Rate limited, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    request_count,
                    max_attempts,
                )
                time.sleep(delay)
                continue
            if resp.status_code in {500, 502, 503, 504} and request_count < max_attempts:
                elapsed = time.monotonic() - started
                self._emit_activity(
                    {
                        "phase": "transport_retry",
                        "message": (
                            f"{model} received HTTP {resp.status_code} after {elapsed:.1f}s; "
                            f"retrying attempt {request_count + 1} of {max_attempts}"
                            f"{retry_note}"
                        ),
                        "model": model,
                        "request_attempt": request_count + 1,
                        "max_attempts": max_attempts,
                        "elapsed_seconds": round(elapsed, 3),
                        "error_category": "server_error",
                        "retry_reason": f"http_{resp.status_code}",
                        "provider_status": resp.status_code,
                        **retry_fields,
                    }
                )
                log.info(
                    "Transient inference HTTP %d, retrying in %.1fs (attempt %d/%d)",
                    resp.status_code,
                    base_delay,
                    request_count,
                    max_attempts,
                )
                time.sleep(base_delay)
                continue
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as error:
                elapsed = time.monotonic() - started
                self._emit_activity(
                    {
                        "phase": "transport_failed",
                        "message": (
                            f"{model} failed after {request_count} provider attempts: "
                            f"HTTP {resp.status_code}"
                        ),
                        "model": model,
                        "request_attempt": request_count,
                        "max_attempts": max_attempts,
                        "elapsed_seconds": round(elapsed, 3),
                        "error_category": "http_error",
                        "retry_reason": f"http_{resp.status_code}",
                        "provider_status": resp.status_code,
                    }
                )
                if _is_http_context_rejection(error):
                    capacity_error = InferenceContextExceeded()
                    capacity_error._transport_request_count = request_count  # type: ignore[attr-defined]
                    raise capacity_error from None
                error._transport_request_count = request_count  # type: ignore[attr-defined]
                raise
            try:
                data = resp.json()
            except ValueError as error:
                _add_transport_request_count(error, request_count)
                elapsed = time.monotonic() - started
                self._emit_activity(
                    {
                        "phase": "transport_failed",
                        "message": f"{model} returned a non-JSON provider response",
                        "model": model,
                        "request_attempt": request_count,
                        "max_attempts": max_attempts,
                        "elapsed_seconds": round(elapsed, 3),
                        "error_category": "invalid_provider_response",
                        "retry_reason": "response_json",
                        "provider_status": resp.status_code,
                    }
                )
                raise
            if request_count > 1:
                elapsed = time.monotonic() - started
                self._emit_activity(
                    {
                        "phase": "transport_recovered",
                        "message": (
                            f"{model} recovered on provider attempt {request_count}{retry_note}"
                        ),
                        "model": model,
                        "request_attempt": request_count,
                        "max_attempts": max_attempts,
                        "elapsed_seconds": round(elapsed, 3),
                        **({"output_mode": output_mode} if output_mode else {}),
                    }
                )
            return data, request_count

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Mapping[str, Any] | None = None,
        reasoning: Literal["default", "disabled"] = "default",
    ) -> JudgeResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format
        if self.backend == "wandb" and reasoning == "disabled":
            body["chat_template_kwargs"] = {"enable_thinking": False}

        data, request_count = self._post_with_retry("/chat/completions", body)

        try:
            choice = data["choices"][0]
            usage = data.get("usage", {})
            content = choice["message"]["content"]
            if not isinstance(content, str):
                content = ""
            finish_reason = choice.get("finish_reason")
            if not isinstance(finish_reason, str):
                finish_reason = None
            else:
                finish_reason = finish_reason[:100]
            usage_mapping = usage if isinstance(usage, Mapping) else {}
            used_schema_retry = (
                self.backend == "wandb"
                and request_count > 1
                and isinstance(response_format, Mapping)
                and response_format.get("type") == "json_schema"
            )
            return JudgeResponse(
                content=content,
                model=data.get("model", model),
                usage=usage_mapping,
                transport_request_count=request_count,
                raw_output_digest=_raw_output_digest(content),
                response_diagnostics=(
                    InferenceResponseDiagnostic(
                        finish_reason=finish_reason,
                        usage=_token_counts(usage_mapping),
                        completion_details=_token_counts(
                            usage_mapping.get("completion_tokens_details")
                        ),
                        content_characters=len(content),
                    ),
                ),
                output_mode=("json_object_fallback" if used_schema_retry else "json_object"),
                schema_fallback_reason=(SCHEMA_FALLBACK_RETRY if used_schema_retry else None),
            )
        except Exception as error:
            _add_transport_request_count(error, request_count)
            raise

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
        started = time.monotonic()
        if response_schema is None:
            response_format: Mapping[str, Any] = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.name,
                    "strict": True,
                    "schema": response_schema.schema,
                },
            }
            messages = json_output_contract_messages(messages, response_schema)

        fallback_reason: str | None = None
        schema_request_count = 0
        recovery_key = (
            (model, response_schema.name)
            if self.backend == "wandb" and response_schema is not None
            else None
        )
        reuse_recovery = recovery_key is not None and recovery_key in self._schema_recovery
        request_messages = messages
        request_format = response_format
        request_reasoning = reasoning
        if reuse_recovery:
            fallback_reason = SCHEMA_FALLBACK_RETRY
            request_messages = _json_object_recovery_messages(messages, response_format)
            request_format = {"type": "json_object"}
            request_reasoning = "disabled"
            self._emit_activity(
                {
                    "phase": "schema_recovery_reused",
                    "message": (
                        f"{model} is reusing concise JSON recovery mode for {response_schema.name}"
                    ),
                    "model": model,
                    "retry_reason": "prior_schema_recovery",
                    "output_mode": "json_object_fallback",
                }
            )
        try:
            resp = self.chat(
                model=model,
                messages=request_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=request_format,
                reasoning=request_reasoning,
            )
        except httpx.HTTPStatusError as error:
            if response_schema is None:
                raise
            fallback_reason = _http_schema_rejection_reason(error)
            if fallback_reason is None:
                raise
            schema_request_count = getattr(error, "_transport_request_count", 1)
            try:
                resp = self.chat(
                    model=model,
                    messages=_json_object_recovery_messages(messages, response_format),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    reasoning=reasoning,
                )
            except Exception as fallback_error:
                _add_transport_request_count(fallback_error, schema_request_count)
                raise

        if fallback_reason is not None:
            resp.output_mode = "json_object_fallback"
            resp.schema_fallback_reason = fallback_reason
            resp.transport_request_count += schema_request_count
        elif response_schema is not None and resp.output_mode != "json_object_fallback":
            resp.output_mode = "json_schema"
        resp.schema_name = response_schema.name if response_schema is not None else None
        if recovery_key is not None and resp.output_mode == "json_object_fallback":
            self._schema_recovery.add(recovery_key)

        parsed = _parse_exact_json_object(resp.content)
        if not parsed and _completion_limit_reached(resp, max_tokens):
            first = resp
            first_diagnostic = first.response_diagnostics[-1]
            retry_reasoning: Literal["default", "disabled"] = (
                "disabled" if self.backend == "wandb" else reasoning
            )
            retry_note = " with thinking disabled" if retry_reasoning != reasoning else ""
            retry_mode_note = (
                " in concise JSON recovery mode" if self.backend == "wandb" else retry_note
            )
            self._emit_activity(
                {
                    "phase": "transport_retry",
                    "message": (
                        f"{model} exhausted {max_tokens} output tokens without valid JSON; "
                        f"retrying structured output attempt 2 of 2{retry_mode_note}"
                    ),
                    "model": model,
                    "request_attempt": 2,
                    "max_attempts": 2,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "error_category": "output_limit",
                    "retry_reason": "structured_output",
                    "finish_reason": first_diagnostic.finish_reason,
                    "completion_tokens": first_diagnostic.usage.get("completion_tokens"),
                    "reasoning_tokens": first_diagnostic.completion_details.get("reasoning_tokens"),
                    "output_sha256": first.raw_output_digest,
                    "output_mode": (
                        "json_object_fallback" if self.backend == "wandb" else first.output_mode
                    ),
                }
            )
            retry_format = (
                {"type": "json_object"}
                if self.backend == "wandb" or fallback_reason is not None
                else response_format
            )
            try:
                retry_messages = (
                    _json_object_recovery_messages(messages, response_format)
                    if retry_format["type"] == "json_object"
                    else messages
                )
                retry = self.chat(
                    model=model,
                    messages=retry_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=retry_format,
                    reasoning=retry_reasoning,
                )
            except Exception as error:
                error._inference_response = first  # type: ignore[attr-defined]
                _add_transport_request_count(error, first.transport_request_count)
                raise
            retry_exhausted = _completion_limit_reached(retry, max_tokens)
            retry.output_mode = (
                "json_object_fallback" if self.backend == "wandb" else first.output_mode
            )
            retry.schema_name = first.schema_name
            retry.schema_fallback_reason = (
                SCHEMA_FALLBACK_RETRY if self.backend == "wandb" else first.schema_fallback_reason
            )
            retry.transport_request_count += first.transport_request_count
            retry.usage = _combined_usage(first.usage, retry.usage)
            retry.response_diagnostics = first.response_diagnostics + retry.response_diagnostics
            resp = retry
            parsed = _parse_exact_json_object(resp.content)
            if parsed and recovery_key is not None and resp.output_mode == "json_object_fallback":
                self._schema_recovery.add(recovery_key)
            if not parsed and retry_exhausted:
                final_diagnostic = resp.response_diagnostics[-1]
                self._emit_activity(
                    {
                        "phase": "transport_failed",
                        "message": (
                            f"{model} exhausted {max_tokens} output tokens on both "
                            "structured output attempts"
                        ),
                        "model": model,
                        "request_attempt": 2,
                        "max_attempts": 2,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "error_category": "output_limit",
                        "retry_reason": "structured_output",
                        "finish_reason": final_diagnostic.finish_reason,
                        "completion_tokens": final_diagnostic.usage.get("completion_tokens"),
                        "reasoning_tokens": final_diagnostic.completion_details.get(
                            "reasoning_tokens"
                        ),
                        "output_sha256": resp.raw_output_digest,
                        "output_mode": resp.output_mode,
                    }
                )
                raise InferenceOutputExceeded(resp)
            if parsed:
                self._emit_activity(
                    {
                        "phase": "transport_recovered",
                        "message": f"{model} returned valid JSON on structured output attempt 2",
                        "model": model,
                        "request_attempt": 2,
                        "max_attempts": 2,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "output_sha256": resp.raw_output_digest,
                        "output_mode": resp.output_mode,
                    }
                )
        if not parsed:
            log.warning(
                "Judge returned invalid JSON object: backend=%s model=%s "
                "output_len=%d output_sha256=%s output_mode=%s request_count=%d",
                self.backend,
                resp.model,
                len(resp.content),
                resp.raw_output_digest,
                resp.output_mode,
                resp.transport_request_count,
            )
        return parsed, resp

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> InferenceClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
