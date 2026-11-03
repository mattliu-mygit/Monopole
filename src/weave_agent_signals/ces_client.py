"""The single place Monopole talks to CES (CoreWeave Evaluations).

Sessions come from the Dataset that weave-agent-adapter writes: a session Case
roots the session Trajectory S, whose member Cases each hold one user message,
assistant message, tool call, or compaction, and each subagent run is a child
Trajectory. The adapter's `specs/EVALUATIONS_CASE_CONTRACT.md` defines the fields.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from weave_agent_signals.client import _get_api_key
from weave_agent_signals.models import (
    ChatSpan,
    SessionView,
    SpanEvent,
    SubagentSpan,
    ToolSpan,
    TurnSpan,
    resolve_trace_role,
)

ATTEMPTS = 3
MAX_RETRY_AFTER_S = 30.0
PAGE_LIMIT = 1000
_STATUS = {"completed": "OK", "aborted": "ERROR"}


def _ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _tool_span(case: dict[str, Any]) -> ToolSpan:
    case_in, out = case["input"], case.get("output") or {}
    arguments = case_in.get("arguments")
    return ToolSpan(
        span_id=case["case_id"],
        tool_name=case_in.get("tool_name") or "",
        arguments="" if arguments is None else json.dumps(arguments, ensure_ascii=False),
        result=out.get("output") or out.get("error") or "",
        status_code="ERROR"
        if out.get("error") is not None or out.get("status") == "failed"
        else "OK"
        if out.get("status") == "completed"
        else "UNSET",
        started_at=_ts(case_in.get("started_at")) or datetime.now(timezone.utc),
        ended_at=_ts(case_in.get("ended_at")),
    )


def session_view(
    trajectory_id: str, cases: list[dict], children: dict[str, list[dict]]
) -> SessionView:
    """Rebuild Monopole's turns from S's Cases and its subagents' Cases.

    A turn is the Cases sharing `turn_id`, ordered by `step_index`. Its first Case
    carries the turn totals, and its ID is the turn's identity, where Weave used a
    trace ID. A subagent's Cases attach to the turn whose tool call launched it.
    """
    members = [c for c in cases if c["input"].get("kind") != "session"]
    turns: dict[str, list[dict]] = {}
    for case in sorted(
        members,
        key=lambda c: (_int(c["input"].get("turn_index")), _int(c["input"].get("step_index"))),
    ):
        turns.setdefault(case["input"]["turn_id"], []).append(case)
    launched = {
        child: case["case_id"]
        for case in [*members, *(c for cs in children.values() for c in cs)]
        for child in case.get("child_trajectory_ids") or []
    }
    owner = {c["case_id"]: child for child, cs in children.items() for c in cs}

    def launch_case(child: str) -> str | None:
        """The S Case a subagent descends from, following nested subagents up to S."""
        case_id, seen = launched.get(child), {child}
        while case_id in owner and owner[case_id] not in seen:
            seen.add(owner[case_id])
            case_id = launched.get(owner[case_id])
        return case_id

    built = []
    for steps in turns.values():
        first, totals = steps[0]["input"], steps[0].get("output") or {}
        texts = [c["input"].get("text") for c in steps if c["input"]["kind"] == "user_message"]
        answers = [
            (c.get("output") or {}).get("text")
            for c in steps
            if c["input"]["kind"] == "assistant_message"
        ]
        stamps = [
            s
            for c in steps
            for s in (_ts(c["input"].get("started_at")), _ts(c["input"].get("ended_at")))
            if s
        ]
        step_ids = {c["case_id"] for c in steps}
        subagents = [
            SubagentSpan(
                span_id=child,
                agent_type=None,
                tool_calls=[
                    _tool_span(c) for c in child_cases if c["input"].get("kind") == "tool_call"
                ],
            )
            for child, child_cases in children.items()
            if launch_case(child) in step_ids
        ]
        user_input = next((t for t in texts if t), None)
        built.append(
            TurnSpan(
                trace_id=steps[0]["case_id"],
                conversation_id=trajectory_id,
                started_at=min(stamps) if stamps else datetime.now(timezone.utc),
                ended_at=max(stamps) if stamps else None,
                model=first.get("model"),
                input_tokens=_int(totals.get("turn_input_tokens")),
                output_tokens=_int(totals.get("turn_output_tokens")),
                cache_read_tokens=_int(totals.get("turn_cache_read_tokens")),
                status_code=_STATUS.get(totals.get("turn_status"), "UNSET"),
                config_version=first.get("config_version"),
                git_branch=first.get("git_branch"),
                effort_level=first.get("effort"),
                session_id=first.get("session_id"),
                steering_count=_int(totals.get("turn_steering_count")),
                denial_count=_int(totals.get("turn_denial_count")),
                tool_error_count=_int(totals.get("turn_tool_error_count")),
                events=[
                    SpanEvent("compaction", _ts(c["input"].get("started_at")))
                    for c in steps
                    if c["input"]["kind"] == "compaction"
                ],
                tool_calls=[_tool_span(c) for c in steps if c["input"]["kind"] == "tool_call"],
                chat_spans=[
                    ChatSpan(
                        span_id=c["case_id"],
                        model=first.get("model") or "",
                        input_tokens=_int((c.get("output") or {}).get("input_tokens")),
                        output_tokens=_int((c.get("output") or {}).get("output_tokens")),
                        cache_read_tokens=_int((c.get("output") or {}).get("cache_read_tokens")),
                        cache_creation_tokens=_int(
                            (c.get("output") or {}).get("cache_write_tokens")
                        ),
                        finish_reason=(c.get("output") or {}).get("finish_reason"),
                    )
                    for c in steps
                    if c["input"]["kind"] == "assistant_message"
                ],
                subagents=subagents,
                user_input=user_input,
                assistant_output=next((a for a in reversed(answers) if a), None),
                trace_role=resolve_trace_role(first.get("trace_role"), user_input),
            )
        )
    first_turn = built[0] if built else None
    return SessionView(
        conversation_id=trajectory_id,
        turns=built,
        config_version=first_turn.config_version if first_turn else None,
        git_branch=first_turn.git_branch if first_turn else None,
    )


class CesClient:
    def __init__(
        self,
        base_url: str,
        scope_ref: str,
        *,
        api_key: str | None = None,
        namespace: str = "wandb",
        http: httpx.Client | None = None,
        sleep=time.sleep,
    ):
        self.base = f"/v1/namespaces/{namespace}/scopes/{quote(scope_ref, safe='')}"
        self._http = http or httpx.Client(
            base_url=base_url.rstrip("/"), auth=("api", api_key or _get_api_key()), timeout=60.0
        )
        self._sleep = sleep

    def close(self) -> None:
        self._http.close()

    def _request(
        self, method: str, path: str, *, params=None, body=None, key: str | None = None
    ) -> dict:
        """Retries network errors, 429, and 5xx, honoring Retry-After. Raises the last error."""
        headers = {"Idempotency-Key": key} if key else {}
        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = self._http.request(
                    method, self.base + path, params=params, json=body, headers=headers
                )
            except httpx.TransportError:
                if attempt == ATTEMPTS:
                    raise
                self._sleep(2 ** (attempt - 1))
                continue
            if response.status_code != 429 and response.status_code < 500 or attempt == ATTEMPTS:
                response.raise_for_status()
                return response.json()
            retry_after = response.headers.get("Retry-After", "")
            delay = (
                float(retry_after)
                if retry_after.replace(".", "", 1).isdigit()
                else 2 ** (attempt - 1)
            )
            self._sleep(min(delay, MAX_RETRY_AFTER_S))
        raise AssertionError("unreachable")

    def trajectory_cases(self, trajectory_id: str) -> list[dict]:
        cases, cursor = [], None
        while True:
            params = {"limit": PAGE_LIMIT, **({"cursor": cursor} if cursor else {})}
            page = self._request("GET", f"/trajectories/{trajectory_id}/cases", params=params)
            cases.extend(page.get("cases") or [])
            cursor = page.get("next_cursor")
            if not cursor:
                return cases

    def read_session(self, trajectory_id: str) -> SessionView:
        """S's Cases plus every subagent Trajectory below it, as Monopole turns."""
        cases = self.trajectory_cases(trajectory_id)
        children: dict[str, list[dict]] = {}
        pending = [child for case in cases for child in case.get("child_trajectory_ids") or []]
        while pending:
            child = pending.pop(0)
            if child in children:
                continue
            children[child] = self.trajectory_cases(child)
            pending.extend(
                c for case in children[child] for c in case.get("child_trajectory_ids") or []
            )
        return session_view(trajectory_id, cases, children)
