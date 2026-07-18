"""Tests for strict structured JSON over the HTTP inference transport."""

from __future__ import annotations

import hashlib

import httpx
import pytest

from weave_agent_signals.judges import inference


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "Return a verdict."}]


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


def _completion(content: str = '{"score":0.75}') -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "model": "gpt-test-resolved",
        "usage": {"total_tokens": 7},
    }


def _http_error(
    status: int,
    message: str,
    *,
    code: str = "unsupported_value",
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/chat/completions")
    response = httpx.Response(
        status,
        request=request,
        json={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": "response_format",
                "code": code,
            }
        },
    )
    return httpx.HTTPStatusError(message, request=request, response=response)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    value = inference.InferenceClient(backend="openai")
    try:
        yield value
    finally:
        value.close()


def test_http_chat_json_sends_strict_named_schema(client, monkeypatch):
    calls: list[dict] = []

    def fake_post(_path, body):
        calls.append(body)
        return _completion(), 1

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    parsed, response = client.chat_json(
        model="gpt-test",
        messages=_messages(),
        response_schema=_schema(),
    )

    assert parsed == {"score": 0.75}
    assert calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_verdict",
            "strict": True,
            "schema": _schema().schema,
        },
    }
    assert response.output_mode == "json_schema"
    assert response.schema_name == "judge_verdict"


def test_http_transport_allows_long_reasoning_responses(client):
    assert client._http.timeout.read == 180.0


def test_http_chat_json_falls_back_only_for_explicit_schema_rejection(client, monkeypatch):
    calls: list[dict] = []
    rejection = "response_format json_schema is not supported for this model"

    def fake_post(_path, body):
        calls.append(body)
        if len(calls) == 1:
            raise _http_error(400, rejection)
        return _completion('{"score":0.5}'), 1

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    parsed, _ = client.chat_json(
        model="gpt-pinned",
        messages=_messages(),
        response_schema=_schema(),
    )

    assert parsed == {"score": 0.5}
    assert [call["model"] for call in calls] == ["gpt-pinned", "gpt-pinned"]
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "invalid API key"),
        (429, "rate limit exceeded"),
        (500, "internal server error"),
    ],
)
def test_http_chat_json_does_not_fallback_for_auth_rate_or_server_errors(
    client, monkeypatch, status, message
):
    calls: list[dict] = []

    def fake_post(_path, body):
        calls.append(body)
        raise _http_error(status, message)

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            response_schema=_schema(),
        )

    assert len(calls) == 1
    assert calls[0]["response_format"]["type"] == "json_schema"


def test_http_schema_fallback_is_audited(client, monkeypatch):
    secret = "SENTINEL_PRIVATE_PROVIDER_DETAIL"
    rejection = f"response_format json_schema is not supported for this model: {secret}"
    calls = 0

    def fake_post(_path, _body):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(400, rejection)
        return _completion('{"score":1}'), 1

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    _, response = client.chat_json(
        model="gpt-pinned",
        messages=_messages(),
        response_schema=_schema(),
    )

    assert response.output_mode == "json_object_fallback"
    assert response.schema_name == "judge_verdict"
    assert response.schema_fallback_reason == "schema_output_unsupported"
    assert secret not in response.schema_fallback_reason
    assert response.transport_request_count == 2
    assert response.raw_output_digest == hashlib.sha256(b'{"score":1}').hexdigest()


@pytest.mark.parametrize(
    "message",
    [
        "response_format json_schema validation failed: unsupported property type",
        "response_format json_schema validation failed: property type is not supported",
    ],
)
def test_http_chat_json_does_not_fallback_for_schema_validation_error(client, monkeypatch, message):
    calls = 0

    def fake_post(_path, _body):
        nonlocal calls
        calls += 1
        raise _http_error(
            400,
            message,
            code="invalid_json_schema",
        )

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            response_schema=_schema(),
        )

    assert calls == 1


def test_http_transport_request_count_includes_rate_limit_retries(client, monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    responses = iter(
        [
            httpx.Response(429, request=request),
            httpx.Response(200, request=request, json=_completion()),
        ]
    )
    calls: list[dict] = []

    def fake_post(_path, *, json):
        calls.append(json)
        return next(responses)

    monkeypatch.setattr(client._http, "post", fake_post)
    monkeypatch.setattr(inference.time, "sleep", lambda _delay: None)

    _, response = client.chat_json(
        model="gpt-pinned",
        messages=_messages(),
        response_schema=_schema(),
    )

    assert len(calls) == 2
    assert all(call["response_format"]["type"] == "json_schema" for call in calls)
    assert response.output_mode == "json_schema"
    assert response.transport_request_count == 2


def test_http_explicit_context_rejection_has_typed_capacity_error(client, monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={
            "error": {
                "message": "This model's maximum context length is 131072 tokens.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        },
    )
    monkeypatch.setattr(client._http, "post", lambda _path, **_kwargs: response)

    with pytest.raises(inference.InferenceContextExceeded) as raised:
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            response_schema=_schema(),
        )

    assert str(raised.value) == "provider context capacity exceeded"
    assert raised.value._transport_request_count == 1


def test_http_wandb_negative_available_max_tokens_is_context_rejection(client, monkeypatch):
    request = httpx.Request("POST", "https://api.inference.wandb.ai/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        json={
            "error": {
                "message": "max_tokens must be at least 1, got -318993.",
                "type": "BadRequestError",
                "param": "max_tokens",
                "code": 400,
            }
        },
    )
    monkeypatch.setattr(client._http, "post", lambda _path, **_kwargs: response)

    with pytest.raises(inference.InferenceContextExceeded):
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            response_schema=_schema(),
        )


def test_http_unrelated_unsupported_capability_does_not_trigger_schema_fallback(
    client, monkeypatch
):
    calls = 0

    def fake_post(_path, _body):
        nonlocal calls
        calls += 1
        raise _http_error(
            400,
            "The selected model does not support tool calls",
            code="invalid_request_error",
        )

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            response_schema=_schema(),
        )

    assert calls == 1


def test_http_schema_dialect_error_does_not_trigger_capability_fallback(client, monkeypatch):
    calls = 0

    def fake_post(_path, _body):
        nonlocal calls
        calls += 1
        raise _http_error(
            400,
            "json_schema unsupported keyword defs",
            code="invalid_request_error",
        )

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            response_schema=_schema(),
        )

    assert calls == 1


def test_http_schema_fallback_count_includes_prior_rate_retry(client, monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    rejection = "response_format json_schema is not supported for this model"
    responses = iter(
        [
            httpx.Response(429, request=request),
            httpx.Response(
                400,
                request=request,
                json={
                    "error": {
                        "message": rejection,
                        "param": "response_format",
                        "code": "unsupported_value",
                        "type": "invalid_request_error",
                    }
                },
            ),
            httpx.Response(200, request=request, json=_completion('{"score":0.5}')),
        ]
    )
    calls: list[dict] = []

    def fake_post(_path, *, json):
        calls.append(json)
        return next(responses)

    monkeypatch.setattr(client._http, "post", fake_post)
    monkeypatch.setattr(inference.time, "sleep", lambda _delay: None)

    parsed, response = client.chat_json(
        model="gpt-pinned",
        messages=_messages(),
        response_schema=_schema(),
    )

    assert parsed == {"score": 0.5}
    assert [call["response_format"]["type"] for call in calls] == [
        "json_schema",
        "json_schema",
        "json_object",
    ]
    assert response.output_mode == "json_object_fallback"
    assert response.transport_request_count == 3


def test_http_failed_fallback_preserves_all_transport_attempts(client, monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    responses = iter(
        [
            httpx.Response(
                400,
                request=request,
                json={
                    "error": {
                        "message": "response_format json_schema is not supported",
                        "param": "response_format",
                        "code": "unsupported_value",
                        "type": "invalid_request_error",
                    }
                },
            ),
            httpx.Response(429, request=request),
            httpx.Response(500, request=request),
        ]
    )
    calls = 0

    def fake_post(_path, *, json):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(client._http, "post", fake_post)
    monkeypatch.setattr(inference.time, "sleep", lambda _delay: None)

    with pytest.raises(httpx.HTTPStatusError) as captured:
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            response_schema=_schema(),
        )

    assert calls == 3
    assert getattr(captured.value, "_transport_request_count", None) == 3


def test_http_malformed_json_preserves_rate_limit_attempts(client, monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    responses = iter(
        [
            httpx.Response(429, request=request),
            httpx.Response(200, request=request, content=b"not-json"),
        ]
    )
    calls = 0

    def fake_post(_path, *, json):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(client._http, "post", fake_post)
    monkeypatch.setattr(inference.time, "sleep", lambda _delay: None)

    with pytest.raises(ValueError) as captured:
        client.chat_json(model="gpt-pinned", messages=_messages())

    assert calls == 2
    assert getattr(captured.value, "_transport_request_count", None) == 2


def test_http_malformed_response_shape_preserves_rate_limit_attempts(client, monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    responses = iter(
        [
            httpx.Response(429, request=request),
            httpx.Response(200, request=request, json={"choices": []}),
        ]
    )
    calls = 0

    def fake_post(_path, *, json):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(client._http, "post", fake_post)
    monkeypatch.setattr(inference.time, "sleep", lambda _delay: None)

    with pytest.raises(IndexError) as captured:
        client.chat_json(model="gpt-pinned", messages=_messages())

    assert calls == 2
    assert getattr(captured.value, "_transport_request_count", None) == 2


def test_http_schema_fallback_malformed_success_preserves_all_attempts(client, monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    responses = iter(
        [
            httpx.Response(
                400,
                request=request,
                json={
                    "error": {
                        "message": "response_format json_schema is not supported",
                        "param": "response_format",
                        "code": "unsupported_value",
                        "type": "invalid_request_error",
                    }
                },
            ),
            httpx.Response(429, request=request),
            httpx.Response(200, request=request, content=b"not-json"),
        ]
    )
    calls = 0

    def fake_post(_path, *, json):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(client._http, "post", fake_post)
    monkeypatch.setattr(inference.time, "sleep", lambda _delay: None)

    with pytest.raises(ValueError) as captured:
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            response_schema=_schema(),
        )

    assert calls == 3
    assert getattr(captured.value, "_transport_request_count", None) == 3


def test_http_invalid_output_log_contains_metadata_not_model_output(client, monkeypatch, caplog):
    secret = "SENTINEL_PRIVATE_INSTRUCTION"
    output = f"not json: {secret}"
    monkeypatch.setattr(client, "_post_with_retry", lambda _path, _body: (_completion(output), 1))

    with caplog.at_level("WARNING", logger="weave_agent_signals.judges"):
        parsed, response = client.chat_json(
            model="gpt-test",
            messages=_messages(),
            response_schema=_schema(),
        )

    assert parsed == {}
    assert secret not in caplog.text
    assert f"output_len={len(output)}" in caplog.text
    assert hashlib.sha256(output.encode()).hexdigest() in caplog.text
    assert "output_mode=json_schema" in caplog.text
    assert "request_count=1" in caplog.text
    assert response.raw_output_digest == hashlib.sha256(output.encode()).hexdigest()
