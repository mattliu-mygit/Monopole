"""OpenAI-compatible chat completions client.

Supports W&B Inference, direct OpenAI, or any OpenAI-compatible endpoint.
Backend selection via JUDGE_BACKEND env var or --judge-backend CLI flag:
  - "wandb"  → W&B Inference (api.inference.wandb.ai), uses WANDB_API_KEY
  - "openai" → OpenAI API (api.openai.com), uses OPENAI_API_KEY
  - custom URL → any OpenAI-compatible endpoint, uses JUDGE_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from weave_agent_signals.client import _get_api_key

log = logging.getLogger("weave_agent_signals.judges")

INFERENCE_BASE = "https://api.inference.wandb.ai/v1"
OPENAI_BASE = "https://api.openai.com/v1"

DEFAULT_ENTITY = "mliu-wandb-weights-biases"
DEFAULT_PROJECT = "agent-sessions"


@dataclass
class JudgeResponse:
    content: str
    model: str
    usage: dict[str, int]


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

    api_key = os.environ.get("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError(f"JUDGE_API_KEY or OPENAI_API_KEY required for backend {backend}")
    return (
        backend,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        "custom",
    )


class InferenceClient:
    """OpenAI-compatible chat completions client."""

    def __init__(
        self,
        base_url: str | None = None,
        entity: str = DEFAULT_ENTITY,
        project: str = DEFAULT_PROJECT,
        backend: str | None = None,
    ):
        if base_url:
            api_key = os.environ.get("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            resolved_url = base_url
            self.backend = "openai" if base_url == OPENAI_BASE else "custom"
        else:
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
    ) -> dict:
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
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ) -> JudgeResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        data = self._post_with_retry("/chat/completions", body)

        choice = data["choices"][0]
        return JudgeResponse(
            content=choice["message"]["content"],
            model=data.get("model", model),
            usage=data.get("usage", {}),
        )

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[dict[str, Any], JudgeResponse]:
        resp = self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(resp.content)
        except json.JSONDecodeError:
            log.warning("Judge returned non-JSON: %.200s", resp.content)
            parsed = {}
        if not isinstance(parsed, dict):
            # Valid JSON can be null/list/scalar; callers expect a dict to .get().
            log.warning("Judge returned non-object JSON: %.200s", resp.content)
            parsed = {}
        return parsed, resp

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> InferenceClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
