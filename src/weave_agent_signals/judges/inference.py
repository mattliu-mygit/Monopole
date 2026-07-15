"""Chat-completions client for the catalog-backed OpenAI and W&B providers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import httpx

from weave_agent_signals.client import _get_api_key

log = logging.getLogger("weave_agent_signals.judges")

INFERENCE_BASE = "https://api.inference.wandb.ai/v1"
OPENAI_BASE = "https://api.openai.com/v1"

DEFAULT_ENTITY = "weave-team"
DEFAULT_PROJECT = "agent-sessions"
SCHEMA_FALLBACK_UNSUPPORTED = "schema_output_unsupported"


@dataclass(frozen=True)
class JsonSchemaSpec:
    """Named JSON Schema requested at the shared model transport boundary."""

    name: str
    schema: Mapping[str, Any]


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
    ) -> tuple[dict[str, Any], JudgeResponse]: ...


class InferenceCancelled(RuntimeError):
    """Inference stopped because the owning run was cancelled."""


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


def _parse_exact_json_object(text: str) -> dict[str, Any]:
    """Parse one complete JSON object without searching prose or fences."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _explicit_schema_rejection_reason(text: str) -> str | None:
    """Return a bounded explicit schema-unsupported diagnostic, if present."""
    reason = " ".join(text.split())
    lowered = reason.lower()
    if any(term in lowered for term in _NON_FALLBACK_FAILURE_TERMS):
        return None
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
            timeout=60.0,
        )

    def _post_with_retry(
        self,
        path: str,
        body: dict,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> tuple[dict, int]:
        for attempt in range(max_retries + 1):
            resp = self._http.post(path, json=body)
            if resp.status_code == 429 and attempt < max_retries:
                delay = base_delay * (2**attempt)
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                log.info(
                    "Rate limited, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
                continue
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as error:
                error._transport_request_count = attempt + 1  # type: ignore[attr-defined]
                raise
            try:
                return resp.json(), attempt + 1
            except ValueError as error:
                _add_transport_request_count(error, attempt + 1)
                raise
        raise AssertionError("HTTP retry loop exhausted without a response")

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Mapping[str, Any] | None = None,
    ) -> JudgeResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        data, request_count = self._post_with_retry("/chat/completions", body)

        try:
            choice = data["choices"][0]
            usage = data.get("usage", {})
            content = choice["message"]["content"]
            if not isinstance(content, str):
                content = ""
            return JudgeResponse(
                content=content,
                model=data.get("model", model),
                usage=usage if isinstance(usage, Mapping) else {},
                transport_request_count=request_count,
                raw_output_digest=_raw_output_digest(content),
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
    ) -> tuple[dict[str, Any], JudgeResponse]:
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

        fallback_reason: str | None = None
        schema_request_count = 0
        try:
            resp = self.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
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
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception as fallback_error:
                _add_transport_request_count(fallback_error, schema_request_count)
                raise

        if fallback_reason is not None:
            resp.output_mode = "json_object_fallback"
            resp.schema_fallback_reason = fallback_reason
            resp.transport_request_count += schema_request_count
        elif response_schema is not None:
            resp.output_mode = "json_schema"
        resp.schema_name = response_schema.name if response_schema is not None else None

        parsed = _parse_exact_json_object(resp.content)
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
