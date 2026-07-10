from __future__ import annotations

import base64
import json
import logging
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
        import netrc as netrc_mod
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

    # --- Query building ---

    def _build_turn_query(
        self,
        limit: int = 100,
        since: datetime | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        conditions = [
            {"$eq": [{"$getField": "operation_name"}, {"$literal": "invoke_agent"}]},
            {"$not": [{"$getField": "parent_span_id"}]},
        ]
        if since:
            conditions.append(
                {"$gte": [{"$getField": "started_at"}, {"$literal": since.isoformat()}]}
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

    def _build_children_query(self, conversation_id: str) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "limit": 1000,
            "query": {
                "$expr": {
                    "$eq": [
                        {"$getField": "conversation_id"},
                        {"$literal": conversation_id},
                    ]
                }
            },
        }

    # --- Hydration ---

    def _hydrate_turn(self, raw: dict[str, Any]) -> TurnSpan:
        attrs = raw.get("custom_attrs") or {}
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
            model=raw.get("model"),
            input_tokens=raw.get("input_tokens", 0) or 0,
            output_tokens=raw.get("output_tokens", 0) or 0,
            cache_read_tokens=raw.get("cache_read_tokens", 0) or 0,
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
        attrs = raw.get("custom_attrs") or {}
        return ToolSpan(
            span_id=raw.get("span_id", ""),
            tool_name=attrs.get("gen_ai.tool.call.name", ""),
            arguments=attrs.get("gen_ai.tool.call.arguments", ""),
            result=attrs.get("gen_ai.tool.call.result", ""),
            status_code=raw.get("status_code", "UNSET"),
            started_at=_parse_ts(raw.get("started_at")) or datetime.now(timezone.utc),
            ended_at=_parse_ts(raw.get("ended_at")),
        )

    def _attach_children(self, turn: TurnSpan, children: list[dict]) -> None:
        for child in children:
            if child.get("span_id") == turn.trace_id:
                continue
            op = child.get("operation_name", "")
            parent = child.get("parent_span_id")
            if op == "execute_tool":
                turn.tool_calls.append(self._hydrate_tool_span(child))
            elif op == "chat":
                attrs = child.get("custom_attrs") or {}
                turn.chat_spans.append(ChatSpan(
                    span_id=child.get("span_id", ""),
                    model=child.get("model", ""),
                    input_tokens=child.get("input_tokens", 0) or 0,
                    output_tokens=child.get("output_tokens", 0) or 0,
                    cache_read_tokens=child.get("cache_read_tokens", 0) or 0,
                    cache_creation_tokens=attrs.get("gen_ai.usage.cache_creation_input_tokens", 0) or 0,
                    finish_reason=attrs.get("gen_ai.completion.finish_reason"),
                ))

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

    def hydrate_turn_children(self, turn: TurnSpan) -> None:
        body = self._build_children_query(turn.conversation_id)
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

    def write_scores_batch(self, items: list[tuple[Score, str]]) -> dict | None:
        if not items:
            return None
        batch = [self._build_feedback_body(s, ref) for s, ref in items]
        resp = self._http.post("/feedback/batch/create", json={"batch": batch})
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
        return data.get("feedback") or data.get("results") or []


def _parse_ts(val: str | None) -> datetime | None:
    if not val:
        return None
    try:
        if val.endswith("Z"):
            val = val[:-1] + "+00:00"
        return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return None
