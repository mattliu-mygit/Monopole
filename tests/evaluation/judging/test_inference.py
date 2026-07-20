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


def _completion(
    content: str = '{"score":0.75}',
    *,
    finish_reason: str = "stop",
    completion_details: dict[str, int] | None = None,
) -> dict:
    usage: dict[str, object] = {"total_tokens": 7}
    if completion_details is not None:
        usage["completion_tokens_details"] = completion_details
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "model": "gpt-test-resolved",
        "usage": usage,
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


def test_wandb_can_disable_reasoning_for_one_request(client, monkeypatch):
    calls: list[dict] = []
    client.backend = "wandb"
    monkeypatch.setattr(
        client,
        "_post_with_retry",
        lambda _path, body: (calls.append(body) or _completion(), 1),
    )

    client.chat_json(
        model="Qwen/Qwen3.6-35B-A3B",
        messages=_messages(),
        response_schema=_schema(),
        reasoning="disabled",
    )
    client.chat_json(
        model="Qwen/Qwen3.6-35B-A3B",
        messages=_messages(),
        response_schema=_schema(),
    )

    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in calls[1]


def test_http_chat_captures_response_diagnostics(client, monkeypatch):
    payload = _completion(completion_details={"reasoning_tokens": 2})
    payload["usage"].update(
        prompt_tokens=8,
        completion_tokens=3,
        total_tokens=11,
    )
    monkeypatch.setattr(client, "_post_with_retry", lambda _path, _body: (payload, 1))

    response = client.chat(model="gpt-test", messages=_messages())

    assert [item.model_dump() for item in response.response_diagnostics] == [
        {
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
            "completion_details": {"reasoning_tokens": 2},
            "content_characters": len('{"score":0.75}'),
        }
    ]


def test_http_transport_allows_long_reasoning_responses(client):
    assert client._http.timeout.read == 240.0
    assert client._http.timeout.connect == 10.0
    assert client._http.timeout.write == 60.0
    assert client._http.timeout.pool == 10.0


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
    assert "REQUIRED_JSON_SCHEMA:" in calls[1]["messages"][0]["content"]
    assert '"required":["score"]' in calls[1]["messages"][0]["content"]


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


def test_http_transport_retries_one_read_timeout(client, monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    responses = iter(
        [
            httpx.ReadTimeout("provider stalled", request=request),
            httpx.Response(200, request=request, json=_completion()),
        ]
    )
    calls = 0
    now = 0.0

    def fake_post(_path, **_kwargs):
        nonlocal calls, now
        calls += 1
        now += 5.0 if calls == 1 else 2.0
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    events: list[dict[str, object]] = []
    client.set_activity(events.append)
    monkeypatch.setattr(client._http, "post", fake_post)
    monkeypatch.setattr(inference.time, "monotonic", lambda: now)

    def verify_live_retry(_delay):
        assert events[-1]["phase"] == "transport_retry"

    monkeypatch.setattr(inference.time, "sleep", verify_live_retry)

    _, response = client.chat_json(
        model="gpt-pinned",
        messages=_messages(),
        response_schema=_schema(),
    )

    assert calls == 2
    assert response.transport_request_count == 2
    assert [event["phase"] for event in events] == [
        "transport_attempt_started",
        "transport_attempt_completed",
        "transport_retry",
        "transport_attempt_started",
        "transport_attempt_completed",
        "transport_recovered",
    ]
    assert events[0]["request_attempt"] == 1
    assert events[1]["elapsed_seconds"] == 5.0
    assert events[1]["status"] == "failed"
    assert events[2]["request_attempt"] == 2
    assert events[2]["max_attempts"] == 3
    assert events[2]["error_category"] == "ReadTimeout"
    assert events[2]["retry_reason"] == "transport_error"
    assert events[3]["request_attempt"] == 2
    assert events[4]["elapsed_seconds"] == 2.0
    assert events[4]["status"] == "succeeded"
    assert events[5]["request_attempt"] == 2


def test_wandb_transport_retry_uses_concise_json_mode(client, monkeypatch):
    request = httpx.Request("POST", "https://api.inference.wandb.ai/v1/chat/completions")
    responses = iter(
        [
            httpx.ReadTimeout("provider stalled", request=request),
            httpx.ReadTimeout("provider still stalled", request=request),
            httpx.Response(200, request=request, json=_completion()),
        ]
    )
    calls: list[dict] = []
    events: list[dict[str, object]] = []
    client.backend = "wandb"
    client.set_activity(events.append)

    def fake_post(_path, *, json):
        calls.append(json)
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(client._http, "post", fake_post)
    monkeypatch.setattr(inference.time, "sleep", lambda _delay: None)

    parsed, response = client.chat_json(
        model="google/gemma-3-27b-it",
        messages=_messages(),
        response_schema=_schema(),
    )

    assert parsed == {"score": 0.75}
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert "chat_template_kwargs" not in calls[0]
    assert [call["response_format"] for call in calls[1:]] == [
        {"type": "json_object"},
        {"type": "json_object"},
    ]
    assert [call["chat_template_kwargs"] for call in calls[1:]] == [
        {"enable_thinking": False},
        {"enable_thinking": False},
    ]
    assert "REQUIRED_JSON_SCHEMA:" not in calls[0]["messages"][0]["content"]
    assert all('"required":["score"]' in call["messages"][0]["content"] for call in calls[1:])
    assert response.output_mode == "json_object_fallback"
    assert events[2]["output_mode"] == "json_object_fallback"
    assert "concise JSON recovery mode" in str(events[2]["message"])
    assert events[3]["output_mode"] == "json_object_fallback"
    assert events[5]["output_mode"] == "json_object_fallback"
    assert events[6]["output_mode"] == "json_object_fallback"


def test_wandb_reuses_recovery_mode_for_same_model_and_schema(client, monkeypatch):
    calls: list[dict] = []
    events: list[dict[str, object]] = []
    client.backend = "wandb"
    client.set_activity(events.append)

    def fake_post(_path, body):
        calls.append(body)
        return _completion(), 2 if len(calls) == 1 else 1

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    client.chat_json(
        model="google/gemma-4-31B-it",
        messages=_messages(),
        response_schema=_schema(),
    )
    _, response = client.chat_json(
        model="google/gemma-4-31B-it",
        messages=_messages(),
        response_schema=_schema(),
    )

    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert calls[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert '"required":["score"]' in calls[1]["messages"][0]["content"]
    assert response.output_mode == "json_object_fallback"
    assert response.schema_fallback_reason == "retry_recovery"
    assert events[-1]["phase"] == "schema_recovery_reused"


def test_http_transport_stops_after_three_read_timeout_attempts(client, monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    calls = 0

    def fail(_path, **_kwargs):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("provider stalled", request=request)

    events: list[dict[str, object]] = []
    client.set_activity(events.append)
    monkeypatch.setattr(client._http, "post", fail)
    monkeypatch.setattr(inference.time, "sleep", lambda _delay: None)

    with pytest.raises(httpx.ReadTimeout) as captured:
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            response_schema=_schema(),
        )

    assert calls == 3
    assert captured.value._transport_request_count == 3
    assert [event["phase"] for event in events] == [
        "transport_attempt_started",
        "transport_attempt_completed",
        "transport_retry",
        "transport_attempt_started",
        "transport_attempt_completed",
        "transport_retry",
        "transport_attempt_started",
        "transport_attempt_completed",
        "transport_failed",
    ]
    assert events[-1]["error_category"] == "ReadTimeout"
    assert events[-1]["request_attempt"] == 3


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

    assert calls == 4
    assert getattr(captured.value, "_transport_request_count", None) == 4


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


def test_http_output_exhaustion_retries_once_and_combines_audit(client, monkeypatch):
    calls: list[dict] = []
    client.backend = "wandb"
    responses = iter(
        [
            (
                _completion("unfinished reasoning"),
                {"prompt_tokens": 100, "completion_tokens": 4, "total_tokens": 104},
            ),
            (
                _completion('{"score":0.75}'),
                {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
            ),
            (
                _completion('{"score":0.5}'),
                {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
            ),
        ]
    )

    def fake_post(_path, body):
        calls.append(body)
        payload, usage = next(responses)
        payload["usage"] = usage
        return payload, 1

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    parsed, response = client.chat_json(
        model="gpt-pinned",
        messages=_messages(),
        max_tokens=4,
        response_schema=_schema(),
    )

    assert parsed == {"score": 0.75}
    assert len(calls) == 2
    assert "chat_template_kwargs" not in calls[0]
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert '"required":["score"]' in calls[1]["messages"][0]["content"]
    assert response.output_mode == "json_object_fallback"
    assert response.usage == {
        "prompt_tokens": 200,
        "completion_tokens": 6,
        "total_tokens": 206,
    }
    assert response.transport_request_count == 2

    learned, learned_response = client.chat_json(
        model="gpt-pinned",
        messages=_messages(),
        max_tokens=4,
        response_schema=_schema(),
    )

    assert learned == {"score": 0.5}
    assert calls[2]["response_format"] == {"type": "json_object"}
    assert calls[2]["chat_template_kwargs"] == {"enable_thinking": False}
    assert learned_response.output_mode == "json_object_fallback"


def test_http_repeated_output_exhaustion_raises_with_combined_audit(client, monkeypatch):
    calls = 0
    events: list[dict[str, object]] = []
    client.set_activity(events.append)

    def fake_post(_path, _body):
        nonlocal calls
        calls += 1
        payload = _completion(
            "unfinished reasoning",
            finish_reason="length",
            completion_details={"reasoning_tokens": 4 if calls == 1 else 3},
        )
        payload["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 4,
            "total_tokens": 104,
            "completion_tokens_details": {"reasoning_tokens": 4 if calls == 1 else 3},
        }
        return payload, 1

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    with pytest.raises(inference.InferenceOutputExceeded) as captured:
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            max_tokens=4,
            response_schema=_schema(),
        )

    assert calls == 2
    assert str(captured.value) == "provider output limit exhausted"
    assert captured.value.response.usage == {
        "prompt_tokens": 200,
        "completion_tokens": 8,
        "total_tokens": 208,
    }
    assert captured.value.response.transport_request_count == 2
    assert [
        diagnostic.finish_reason for diagnostic in captured.value.response.response_diagnostics
    ] == ["length", "length"]
    assert [
        diagnostic.completion_details for diagnostic in captured.value.response.response_diagnostics
    ] == [{"reasoning_tokens": 4}, {"reasoning_tokens": 3}]
    assert [event["phase"] for event in events] == ["transport_retry", "transport_failed"]
    assert events[0]["error_category"] == "output_limit"
    assert events[0]["finish_reason"] == "length"
    assert events[0]["completion_tokens"] == 4
    assert events[0]["reasoning_tokens"] == 4


def test_http_malformed_output_below_limit_does_not_retry(client, monkeypatch):
    calls = 0

    def fake_post(_path, _body):
        nonlocal calls
        calls += 1
        payload = _completion("malformed")
        payload["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 3,
            "total_tokens": 103,
        }
        return payload, 1

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    parsed, _ = client.chat_json(
        model="gpt-pinned",
        messages=_messages(),
        max_tokens=4,
        response_schema=_schema(),
    )

    assert parsed == {}
    assert calls == 1


def test_http_exhaustion_retry_failure_preserves_first_response_audit(client, monkeypatch):
    calls = 0
    retry_error = RuntimeError("retry failed")
    retry_error._transport_request_count = 1  # type: ignore[attr-defined]

    def fake_post(_path, _body):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise retry_error
        payload = _completion("unfinished reasoning")
        payload["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 4,
            "total_tokens": 104,
        }
        return payload, 1

    monkeypatch.setattr(client, "_post_with_retry", fake_post)

    with pytest.raises(RuntimeError, match="retry failed") as captured:
        client.chat_json(
            model="gpt-pinned",
            messages=_messages(),
            max_tokens=4,
            response_schema=_schema(),
        )

    assert calls == 2
    assert captured.value._transport_request_count == 2
    assert captured.value._inference_response.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 4,
        "total_tokens": 104,
    }
