from __future__ import annotations

import base64
import json
import logging
import netrc as netrc_mod
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from weave_agent_signals.models import (
    ChatSpan,
    Score,
    SessionView,
    SpanEvent,
    SubagentSpan,
    ToolSpan,
    TurnSpan,
)

log = logging.getLogger(__name__)

TRACE_BASE = "https://trace.wandb.ai"

_CUSTOM_ATTR_COLUMNS = [
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.config_version"},
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.git_branch"},
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.effort_level"},
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.session_id"},
    {"source": "custom_attrs_int", "key": "weave_agent_adapter.steering_count"},
    {"source": "custom_attrs_int", "key": "weave_agent_adapter.denial_count"},
    {"source": "custom_attrs_int", "key": "weave_agent_adapter.tool_error_count"},
]


def _get_api_key() -> str:
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    netrc_path = os.path.expanduser("~/.netrc")
    if os.path.exists(netrc_path):
        try:
            auth = netrc_mod.netrc(netrc_path).authenticators("api.wandb.ai")
            if auth:
                return auth[2]
        except Exception:
            pass
    raise RuntimeError("No W&B API key found. Set WANDB_API_KEY or add api.wandb.ai to ~/.netrc")


class WeaveClient:
    def __init__(
        self,
        api_key: str | None = None,
        entity: str = "mliu-wandb-weights-biases",
        project: str = "agent-sessions",
    ):
        self.api_key = api_key or _get_api_key()
        self.entity = entity
        self.project = project
        self.project_id = f"{entity}/{project}"
        token = base64.b64encode(f"api:{self.api_key}".encode()).decode()
        self._http = httpx.Client(
            base_url=TRACE_BASE,
            headers={
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # --- Query building ---

    def _build_turn_query(
        self,
        limit: int = 100,
        since: datetime | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        conditions = [
            {"$eq": [{"$getField": "operation_name"}, {"$literal": "invoke_agent"}]},
            {"$eq": [{"$getField": "parent_span_id"}, {"$literal": ""}]},
        ]
        if since:
            conditions.append(
                {"$gte": [{"$getField": "started_at"}, {"$literal": _format_ts(since)}]}
            )
        if conversation_id:
            conditions.append(
                {"$eq": [{"$getField": "conversation_id"}, {"$literal": conversation_id}]}
            )

        return {
            "project_id": self.project_id,
            "limit": limit,
            "sort_by": [{"field": "started_at", "direction": "desc"}],
            "query": {"$expr": {"$and": conditions}},
            "custom_attr_columns": _CUSTOM_ATTR_COLUMNS,
        }

    def _build_children_query(self, trace_id: str) -> dict[str, Any]:
        # include_details pulls the heavy detail-only columns — notably
        # tool_call_arguments and tool_call_result, which the outcome scorers
        # parse. Without it those columns come back NULL.
        return {
            "project_id": self.project_id,
            "limit": 1000,
            "include_details": True,
            "query": {
                "$expr": {
                    "$eq": [
                        {"$getField": "trace_id"},
                        {"$literal": trace_id},
                    ]
                }
            },
        }

    # --- Hydration ---

    def _hydrate_turn(self, raw: dict[str, Any]) -> TurnSpan:
        attrs = _custom_attrs(raw)
        events = []
        events_dump = raw.get("events_dump", "")
        if events_dump:
            try:
                for ev in json.loads(events_dump):
                    events.append(SpanEvent(
                        name=ev.get("name", ""),
                        timestamp=_parse_ts(ev.get("timestamp")),
                        attributes=ev.get("attributes", {}),
                    ))
            except (json.JSONDecodeError, TypeError):
                pass

        return TurnSpan(
            trace_id=raw["trace_id"],
            conversation_id=raw.get("conversation_id", ""),
            started_at=_parse_ts(raw.get("started_at")) or datetime.now(timezone.utc),
            ended_at=_parse_ts(raw.get("ended_at")),
            # tokens live on chat spans, not the invoke_agent root; these are
            # backfilled from the child chat spans in _attach_children.
            model=raw.get("response_model") or raw.get("request_model") or None,
            input_tokens=raw.get("input_tokens", 0) or 0,
            output_tokens=raw.get("output_tokens", 0) or 0,
            cache_read_tokens=raw.get("cache_read_input_tokens", 0) or 0,
            status_code=raw.get("status_code", "UNSET"),
            config_version=attrs.get("weave_agent_adapter.config_version"),
            git_branch=attrs.get("weave_agent_adapter.git_branch"),
            effort_level=attrs.get("weave_agent_adapter.effort_level"),
            session_id=attrs.get("weave_agent_adapter.session_id"),
            steering_count=attrs.get("weave_agent_adapter.steering_count", 0) or 0,
            denial_count=attrs.get("weave_agent_adapter.denial_count", 0) or 0,
            tool_error_count=attrs.get("weave_agent_adapter.tool_error_count", 0) or 0,
            events=events,
            tool_calls=[],
            chat_spans=[],
            subagents=[],
        )

    def _hydrate_tool_span(self, raw: dict[str, Any]) -> ToolSpan:
        # arguments/result are detail-only columns: they populate only when the
        # spans query is issued with include_details=True (see _build_children_query).
        return ToolSpan(
            span_id=raw.get("span_id", ""),
            tool_name=raw.get("tool_name", ""),
            arguments=raw.get("tool_call_arguments") or "",
            result=raw.get("tool_call_result") or "",
            status_code=raw.get("status_code", "UNSET"),
            started_at=_parse_ts(raw.get("started_at")) or datetime.now(timezone.utc),
            ended_at=_parse_ts(raw.get("ended_at")),
        )

    def _attach_children(self, turn: TurnSpan, children: list[dict]) -> None:
        subagent_ids: set[str] = set()
        subagent_map: dict[str, SubagentSpan] = {}

        # First pass: identify subagent spans
        for child in children:
            if child.get("trace_id") != turn.trace_id:
                continue
            if child.get("span_id") == turn.trace_id:
                continue
            op = child.get("operation_name", "")
            parent = child.get("parent_span_id")
            if op == "invoke_agent" and parent:
                sid = child.get("span_id", "")
                subagent_ids.add(sid)
                # the subagent type is the agent identity on the nested
                # invoke_agent span (gen_ai.agent.name -> agent_name column)
                subagent_map[sid] = SubagentSpan(
                    span_id=sid,
                    agent_type=child.get("agent_name") or None,
                    tool_calls=[],
                )

        # Second pass: attach tool/chat spans to correct parent
        for child in children:
            if child.get("trace_id") != turn.trace_id:
                continue
            if child.get("span_id") == turn.trace_id:
                continue
            op = child.get("operation_name", "")
            parent = child.get("parent_span_id")
            if op == "invoke_agent":
                continue
            if op == "execute_tool":
                tool = self._hydrate_tool_span(child)
                if parent in subagent_ids:
                    subagent_map[parent].tool_calls.append(tool)
                else:
                    turn.tool_calls.append(tool)
            elif op == "chat":
                finish_reasons = child.get("finish_reasons") or []
                chat = ChatSpan(
                    span_id=child.get("span_id", ""),
                    model=child.get("response_model") or child.get("request_model") or "",
                    input_tokens=child.get("input_tokens", 0) or 0,
                    output_tokens=child.get("output_tokens", 0) or 0,
                    cache_read_tokens=child.get("cache_read_input_tokens", 0) or 0,
                    cache_creation_tokens=child.get("cache_creation_input_tokens", 0) or 0,
                    finish_reason=finish_reasons[0] if finish_reasons else None,
                )
                if parent not in subagent_ids:
                    turn.chat_spans.append(chat)

        turn.subagents = list(subagent_map.values())

        # Tokens are recorded on chat spans, not the invoke_agent root (which is
        # all-zero). Sum every chat span in the trace — main agent and subagents —
        # for the true turn total, and take the model from the first chat span.
        chat_children = [
            c for c in children
            if c.get("operation_name") == "chat" and c.get("trace_id") == turn.trace_id
        ]
        if chat_children:
            turn.input_tokens = sum((c.get("input_tokens") or 0) for c in chat_children)
            turn.output_tokens = sum((c.get("output_tokens") or 0) for c in chat_children)
            turn.cache_read_tokens = sum(
                (c.get("cache_read_input_tokens") or 0) for c in chat_children
            )
            if not turn.model:
                first = chat_children[0]
                turn.model = first.get("response_model") or first.get("request_model") or None

    # --- API calls ---

    def query_turns(
        self,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[TurnSpan]:
        body = self._build_turn_query(limit=limit, since=since)
        resp = self._http.post("/agents/spans/query", json=body)
        resp.raise_for_status()
        data = resp.json()
        spans = data.get("spans") or data.get("results") or []
        return [self._hydrate_turn(s) for s in spans]

    def _build_paginated_query(
        self,
        limit: int,
        since: datetime | None = None,
        after: datetime | None = None,
    ) -> dict[str, Any]:
        conditions = [
            {"$eq": [{"$getField": "operation_name"}, {"$literal": "invoke_agent"}]},
            {"$eq": [{"$getField": "parent_span_id"}, {"$literal": ""}]},
        ]
        if since:
            conditions.append(
                {"$gte": [{"$getField": "started_at"}, {"$literal": _format_ts(since)}]}
            )
        if after:
            # $gte (not $gt): started_at is not unique, so a strict cursor would
            # skip any turns sharing the previous page's boundary timestamp.
            # The boundary rows are re-fetched and dropped by dedup in the caller.
            conditions.append(
                {"$gte": [{"$getField": "started_at"}, {"$literal": _format_ts(after)}]}
            )
        return {
            "project_id": self.project_id,
            "limit": limit,
            "sort_by": [{"field": "started_at", "direction": "asc"}],
            "query": {"$expr": {"$and": conditions}},
            "custom_attr_columns": _CUSTOM_ATTR_COLUMNS,
        }

    def query_turns_paginated(
        self,
        page_size: int = 500,
        since: datetime | None = None,
    ) -> list[TurnSpan]:
        all_turns: list[TurnSpan] = []
        seen: set[str] = set()
        cursor: datetime | None = None
        while True:
            body = self._build_paginated_query(limit=page_size, since=since, after=cursor)
            resp = self._http.post("/agents/spans/query", json=body)
            resp.raise_for_status()
            data = resp.json()
            spans = data.get("spans") or data.get("results") or []
            if not spans:
                break
            page = [self._hydrate_turn(s) for s in spans]
            # dedup: the $gte cursor re-fetches the previous boundary timestamp
            new = [t for t in page if t.trace_id not in seen]
            seen.update(t.trace_id for t in page)
            all_turns.extend(new)
            if len(spans) < page_size:
                break
            if not new:
                # whole page already seen — a single timestamp exceeds page_size;
                # a timestamp cursor can't advance further without losing rows
                break
            cursor = max(t.started_at for t in page)
        return all_turns

    def hydrate_turn_children(self, turn: TurnSpan) -> None:
        body = self._build_children_query(turn.trace_id)
        resp = self._http.post("/agents/spans/query", json=body)
        resp.raise_for_status()
        data = resp.json()
        children = data.get("spans") or data.get("results") or []
        self._attach_children(turn, children)

    def query_session(self, conversation_id: str) -> SessionView:
        body = self._build_turn_query(limit=500, conversation_id=conversation_id)
        resp = self._http.post("/agents/spans/query", json=body)
        resp.raise_for_status()
        data = resp.json()
        spans = data.get("spans") or data.get("results") or []
        turns = sorted(
            [self._hydrate_turn(s) for s in spans],
            key=lambda t: t.started_at,
        )
        return SessionView(
            conversation_id=conversation_id,
            turns=turns,
            config_version=turns[0].config_version if turns else None,
            git_branch=turns[0].git_branch if turns else None,
        )

    # --- Score write-back ---

    def _build_feedback_body(self, score: Score, ref: str) -> dict[str, Any]:
        return score.to_feedback_payload(ref=ref, project_id=self.project_id)

    def write_score(self, score: Score, ref: str) -> dict | None:
        body = self._build_feedback_body(score, ref)
        resp = self._http.post("/feedback/create", json=body)
        resp.raise_for_status()
        return resp.json()

    def query_existing_feedback(
        self, ref: str, feedback_type: str
    ) -> list[dict]:
        body = {
            "project_id": self.project_id,
            "query": {
                "$expr": {
                    "$and": [
                        {"$eq": [{"$getField": "weave_ref"}, {"$literal": ref}]},
                        {"$eq": [{"$getField": "feedback_type"}, {"$literal": feedback_type}]},
                    ]
                }
            },
        }
        resp = self._http.post("/feedback/query", json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("feedback") or data.get("results") or []

    def delete_feedback(self, feedback_id: str) -> None:
        body = {
            "project_id": self.project_id,
            "query": {
                "$expr": {
                    "$eq": [{"$getField": "id"}, {"$literal": feedback_id}]
                }
            },
        }
        resp = self._http.post("/feedback/purge", json=body)
        resp.raise_for_status()

    def delete_feedback_ids(self, feedback_list: list[dict]) -> None:
        for fb in feedback_list:
            self.delete_feedback(fb["id"])

    def query_all_feedback(self, ref: str) -> list[dict]:
        body = {
            "project_id": self.project_id,
            "query": {
                "$expr": {
                    "$eq": [{"$getField": "weave_ref"}, {"$literal": ref}]
                }
            },
        }
        resp = self._http.post("/feedback/query", json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("feedback") or data.get("results") or []


def _custom_attrs(raw: dict[str, Any]) -> dict[str, Any]:
    # The spans API returns custom attributes split across typed maps
    # (custom_attrs_string / _int / _float / _bool), never as a single merged
    # dict. Flatten them so callers can look up by fully-qualified attr name.
    merged: dict[str, Any] = {}
    for key in ("custom_attrs_string", "custom_attrs_int",
                "custom_attrs_float", "custom_attrs_bool"):
        m = raw.get(key)
        if isinstance(m, dict):
            merged.update(m)
    return merged


def _format_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


def _parse_ts(val: str | None) -> datetime | None:
    if not val:
        return None
    try:
        if val.endswith("Z"):
            val = val[:-1] + "+00:00"
        return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return None
