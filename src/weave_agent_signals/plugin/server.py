"""Calls to Monopole's running server, which the start command runs on 127.0.0.1.

Starting a run goes through the server's own run routes, as Monopole's web UI does, so the server
validates the request against its catalogs and runs the stages on its executor. An action process
can't run stages itself: it exits when the action ends, and the server would fail a run it finds
half done when it next starts.
"""

from __future__ import annotations

from typing import Any

import httpx

from weave_agent_signals.plugin.confirm import Refused

# Listing sessions scans CES, which can take most of a minute on a large Dataset.
REQUEST_TIMEOUT_SECONDS = 120


class Server:
    def __init__(self, port: int, client: httpx.Client | None = None) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Server:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def call(
        self,
        method: str,
        path: str,
        *,
        body: object = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """The route's JSON answer. A refusal carries the server's own message, such as
        "judge model codex:gpt-5 is unknown or unavailable"."""

        try:
            response = self._client.request(method, f"{self.base}{path}", json=body, params=params)
        except httpx.TransportError as exc:
            raise Refused(
                f"Monopole's server isn't answering on {self.base} ({exc}). Start Monopole in "
                "Prompting Time and try again."
            ) from exc
        if response.status_code >= 400:
            raise Refused(f"{method} {path}: {_message(response)}")
        return response.json() if response.content else None

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.call("GET", path, params=params)


def _message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, dict) and detail.get("message"):
        return str(detail["message"])
    if isinstance(detail, list) and detail:
        # FastAPI's validation errors: a list of {loc, msg}.
        return "; ".join(str(item.get("msg", item)) for item in detail if isinstance(item, dict))
    if isinstance(detail, str):
        return detail
    return f"HTTP {response.status_code}"


def open_server(port: int) -> Server:
    """The server on `port`, from the plugin's settings. Tests replace this."""

    return Server(port)
