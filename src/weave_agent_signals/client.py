from __future__ import annotations

import base64
import json
import netrc as netrc_mod
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import httpx

from weave_agent_signals.models import (
    FEEDBACK_PREFIX,
    ChatSpan,
    Score,
    SessionView,
    SpanEvent,
    SubagentSpan,
    ToolSpan,
    TurnSpan,
)

TRACE_BASE = "https://trace.wandb.ai"
_DETAIL_BATCH_LIMIT = 10_000
_FEEDBACK_BATCH_LIMIT = 10_000

_CUSTOM_ATTR_COLUMNS = [
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.config_version"},
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.git_branch"},
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.effort_level"},
    {"source": "custom_attrs_string", "key": "weave_agent_adapter.session_id"},
    {"source": "custom_attrs_int", "key": "weave_agent_adapter.steering_count"},
    {"source": "custom_attrs_int", "key": "weave_agent_adapter.denial_count"},
    {"source": "custom_attrs_int", "key": "weave_agent_adapter.tool_error_count"},
]


def _root_trace_counts(spans: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for span in spans:
        trace_id = span.get("trace_id")
        if (
            span.get("operation_name") == "invoke_agent"
            and span.get("parent_span_id") == ""
            and isinstance(trace_id, str)
            and trace_id
        ):
            counts[trace_id] += 1
    return counts


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
        entity: str = "weave-team",
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
            timeout=60.0,
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
        include_details: bool = False,
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
                {
                    "$eq": [
                        {"$getField": "conversation_id"},
                        {"$literal": conversation_id},
                    ]
                }
            )

        body: dict[str, Any] = {
            "project_id": self.project_id,
            "limit": limit,
            "sort_by": [{"field": "started_at", "direction": "desc"}],
            "query": {"$expr": {"$and": conditions}},
            "custom_attr_columns": _CUSTOM_ATTR_COLUMNS,
        }
        if include_details:
            body["include_details"] = True
        return body

    # --- Hydration ---

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str | None:
        content = message.get("content")
        if content is None:
            content = message.get("parts")
        if isinstance(content, str):
            return content.strip() or None
        if not isinstance(content, list):
            return None

        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict) or part.get("type") not in (None, "text"):
                continue
            text = part.get("text")
            if not isinstance(text, str):
                text = part.get("content")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        return " ".join(text_parts).strip() or None

    @classmethod
    def _extract_message(cls, raw: dict[str, Any], column: str, role: str) -> str | None:
        encoded = raw.get(column)
        if not encoded:
            return None
        try:
            msgs = json.loads(encoded) if isinstance(encoded, str) else encoded
            for msg in reversed(msgs) if isinstance(msgs, list) else []:
                if isinstance(msg, dict) and msg.get("role") == role:
                    text = cls._message_text(msg)
                    if text:
                        return text
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    @classmethod
    def _extract_user_input(cls, raw: dict[str, Any]) -> str | None:
        return cls._extract_message(raw, "input_messages", "user")

    @classmethod
    def _extract_assistant_output(cls, raw: dict[str, Any]) -> str | None:
        return cls._extract_message(raw, "output_messages", "assistant")

    def _hydrate_turn(self, raw: dict[str, Any]) -> TurnSpan:
        attrs = _custom_attrs(raw)
        events = []
        events_dump = raw.get("events_dump", "")
        if events_dump:
            try:
                for ev in json.loads(events_dump):
                    events.append(
                        SpanEvent(
                            name=ev.get("name", ""),
                            timestamp=_parse_ts(ev.get("timestamp")),
                        )
                    )
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
            user_input=self._extract_user_input(raw),
            assistant_output=self._extract_assistant_output(raw),
            events=events,
            tool_calls=[],
            chat_spans=[],
            subagents=[],
        )

    def _hydrate_tool_span(self, raw: dict[str, Any]) -> ToolSpan:
        # arguments/result are detail-only columns: they populate only when the
        # spans query is issued with include_details=True by detail hydration.
        return ToolSpan(
            span_id=raw.get("span_id", ""),
            tool_name=raw.get("tool_name", ""),
            arguments=raw.get("tool_call_arguments") or "",
            result=raw.get("tool_call_result") or "",
            status_code=raw.get("status_code", "UNSET"),
            started_at=_parse_ts(raw.get("started_at")) or datetime.now(timezone.utc),
            ended_at=_parse_ts(raw.get("ended_at")),
        )

    def _backfill_root_messages(self, turn: TurnSpan, spans: list[dict]) -> None:
        if turn.user_input and turn.assistant_output:
            return
        for span in spans:
            if (
                span.get("trace_id") == turn.trace_id
                and span.get("operation_name") == "invoke_agent"
                and span.get("parent_span_id") == ""
            ):
                if not turn.user_input:
                    turn.user_input = self._extract_user_input(span)
                if not turn.assistant_output:
                    turn.assistant_output = self._extract_assistant_output(span)
                return

    def _attach_children(self, turn: TurnSpan, children: list[dict]) -> None:
        subagent_ids: set[str] = set()
        subagent_map: dict[str, SubagentSpan] = {}

        # First pass: identify subagent spans
        for child in children:
            if child.get("trace_id") != turn.trace_id:
                continue
            if child.get("parent_span_id") == "" and child.get("operation_name") == "invoke_agent":
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
            if child.get("parent_span_id") == "" and child.get("operation_name") == "invoke_agent":
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
        # for the true turn total. Derive the generating model deterministically
        # from the earliest main-agent chat, never from a subagent.
        chat_children = [
            c
            for c in children
            if c.get("operation_name") == "chat" and c.get("trace_id") == turn.trace_id
        ]
        if chat_children:
            turn.input_tokens = sum((c.get("input_tokens") or 0) for c in chat_children)
            turn.output_tokens = sum((c.get("output_tokens") or 0) for c in chat_children)
            turn.cache_read_tokens = sum(
                (c.get("cache_read_input_tokens") or 0) for c in chat_children
            )
            if not turn.model:
                main_chats = [
                    child
                    for child in chat_children
                    if child.get("parent_span_id") not in subagent_ids
                ]
                if main_chats:
                    first = min(
                        main_chats,
                        key=lambda child: (
                            _parse_ts(child.get("started_at"))
                            or datetime.max.replace(tzinfo=timezone.utc),
                            child.get("span_id", ""),
                        ),
                    )
                    turn.model = first.get("response_model") or first.get("request_model") or None

    # --- API calls ---

    def query_turns(
        self,
        limit: int = 100,
        since: datetime | None = None,
        include_details: bool = False,
    ) -> list[TurnSpan]:
        body = self._build_turn_query(limit=limit, since=since, include_details=include_details)
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
        *,
        include_details: bool = False,
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
        if after:
            # $gte (not $gt): started_at is not unique, so a strict cursor would
            # skip any turns sharing the previous page's boundary timestamp.
            # The boundary rows are re-fetched and dropped by dedup in the caller.
            conditions.append(
                {"$gte": [{"$getField": "started_at"}, {"$literal": _format_ts(after)}]}
            )
        if conversation_id:
            conditions.append(
                {
                    "$eq": [
                        {"$getField": "conversation_id"},
                        {"$literal": conversation_id},
                    ]
                }
            )
        body: dict[str, Any] = {
            "project_id": self.project_id,
            "limit": limit,
            "sort_by": [{"field": "started_at", "direction": "asc"}],
            "query": {"$expr": {"$and": conditions}},
            "custom_attr_columns": _CUSTOM_ATTR_COLUMNS,
        }
        if include_details:
            body["include_details"] = True
        return body

    def query_turns_paginated(
        self,
        page_size: int = 500,
        since: datetime | None = None,
        *,
        include_details: bool = False,
        conversation_id: str | None = None,
    ) -> list[TurnSpan]:
        all_turns: list[TurnSpan] = []
        seen: set[str] = set()
        cursor: datetime | None = None
        while True:
            body = self._build_paginated_query(
                limit=page_size,
                since=since,
                after=cursor,
                include_details=include_details,
                conversation_id=conversation_id,
            )
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
            next_cursor = max(t.started_at for t in page)
            if cursor is not None and next_cursor <= cursor:
                raise RuntimeError(
                    "root turn pagination cannot advance past saturated timestamp "
                    f"{_format_ts(cursor)}"
                )
            cursor = next_cursor
        return all_turns

    def query_turns_by_trace_ids(
        self,
        trace_ids: list[str],
        *,
        batch_size: int = 100,
    ) -> list[TurnSpan]:
        """Fetch only the root turns in an already-pinned cohort."""
        requested = list(dict.fromkeys(trace_id for trace_id in trace_ids if trace_id))
        if not requested:
            return []

        hydrated: list[TurnSpan] = []
        for start in range(0, len(requested), batch_size):
            batch = requested[start : start + batch_size]
            row_limit = len(batch) + 1
            conditions = [
                {"$eq": [{"$getField": "operation_name"}, {"$literal": "invoke_agent"}]},
                {"$eq": [{"$getField": "parent_span_id"}, {"$literal": ""}]},
                {
                    "$in": [
                        {"$getField": "trace_id"},
                        [{"$literal": trace_id} for trace_id in batch],
                    ]
                },
            ]
            body = {
                "project_id": self.project_id,
                "limit": row_limit,
                "query": {"$expr": {"$and": conditions}},
                "custom_attr_columns": _CUSTOM_ATTR_COLUMNS,
            }
            resp = self._http.post("/agents/spans/query", json=body)
            resp.raise_for_status()
            data = resp.json()
            spans = data.get("spans") or data.get("results") or []
            if len(spans) >= row_limit:
                raise RuntimeError(
                    "root turn hydration reached the "
                    f"{row_limit}-row limit for traces: {', '.join(batch)}"
                )

            batch_set = set(batch)
            root_counts = _root_trace_counts(spans)
            unexpected = sorted(set(root_counts) - batch_set)
            if unexpected:
                label = "trace" if len(unexpected) == 1 else "traces"
                raise RuntimeError(
                    f"root turn hydration returned unexpected root {label}: "
                    + ", ".join(unexpected)
                )
            duplicates = sorted(trace_id for trace_id, count in root_counts.items() if count > 1)
            if duplicates:
                label = "trace" if len(duplicates) == 1 else "traces"
                raise RuntimeError(
                    f"root turn hydration returned duplicate root {label}: " + ", ".join(duplicates)
                )
            missing = [trace_id for trace_id in batch if root_counts[trace_id] == 0]
            if missing:
                label = "trace" if len(missing) == 1 else "traces"
                raise RuntimeError(
                    f"root turn hydration missing root {label}: " + ", ".join(missing)
                )

            roots_by_trace = {
                raw["trace_id"]: raw
                for raw in spans
                if raw.get("operation_name") == "invoke_agent" and raw.get("parent_span_id") == ""
            }
            hydrated.extend(self._hydrate_turn(roots_by_trace[trace_id]) for trace_id in batch)
        return hydrated

    def hydrate_turn_children(self, turn: TurnSpan) -> None:
        self.hydrate_turns_batch([turn], batch_size=1)

    def hydrate_turns_batch(self, turns: list[TurnSpan], batch_size: int = 40) -> None:
        for start in range(0, len(turns), batch_size):
            batch = turns[start : start + batch_size]
            trace_ids = [t.trace_id for t in batch]
            body = {
                "project_id": self.project_id,
                "limit": _DETAIL_BATCH_LIMIT,
                "sort_by": [{"field": "started_at", "direction": "desc"}],
                "include_details": True,
                "query": {
                    "$expr": {
                        "$in": [
                            {"$getField": "trace_id"},
                            [{"$literal": tid} for tid in trace_ids],
                        ]
                    }
                },
            }
            resp = self._http.post("/agents/spans/query", json=body)
            resp.raise_for_status()
            data = resp.json()
            all_spans = data.get("spans") or data.get("results") or []
            if len(all_spans) >= _DETAIL_BATCH_LIMIT:
                raise RuntimeError(
                    "detail hydration reached the "
                    f"{_DETAIL_BATCH_LIMIT}-span limit for traces: {', '.join(trace_ids)}"
                )

            by_trace: dict[str, list[dict]] = {}
            for span in all_spans:
                tid = span.get("trace_id", "")
                by_trace.setdefault(tid, []).append(span)

            requested = set(trace_ids)
            unexpected = sorted(set(by_trace) - requested)
            if unexpected:
                label = "trace" if len(unexpected) == 1 else "traces"
                raise RuntimeError(
                    f"detail hydration returned unexpected {label}: " + ", ".join(unexpected)
                )

            missing = [trace_id for trace_id in trace_ids if trace_id not in by_trace]
            if missing:
                label = "trace" if len(missing) == 1 else "traces"
                raise RuntimeError(
                    f"detail hydration missing requested {label}: {', '.join(missing)}"
                )

            detailed_root_counts = _root_trace_counts(all_spans)
            duplicate_roots = sorted(
                trace_id
                for trace_id, count in detailed_root_counts.items()
                if trace_id in requested and count > 1
            )
            if duplicate_roots:
                label = "trace" if len(duplicate_roots) == 1 else "traces"
                raise RuntimeError(
                    f"detail hydration returned duplicate detailed root {label}: "
                    + ", ".join(duplicate_roots)
                )

            missing_roots = [
                trace_id for trace_id in trace_ids if detailed_root_counts[trace_id] == 0
            ]
            if missing_roots:
                label = "trace" if len(missing_roots) == 1 else "traces"
                raise RuntimeError(
                    f"detail hydration missing detailed root {label}: " + ", ".join(missing_roots)
                )

            for turn in batch:
                children = by_trace.get(turn.trace_id, [])
                self._attach_children(turn, children)
                self._backfill_root_messages(turn, children)

    def query_session(self, conversation_id: str) -> SessionView:
        turns = sorted(
            self.query_turns_paginated(
                page_size=500,
                conversation_id=conversation_id,
                include_details=False,
            ),
            key=lambda turn: (turn.started_at, turn.trace_id),
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

    def query_existing_feedback_batch(
        self, refs: list[str], batch_size: int = 100
    ) -> dict[tuple[str, str], list[dict]]:
        result: dict[tuple[str, str], list[dict]] = {}
        for start in range(0, len(refs), batch_size):
            batch = refs[start : start + batch_size]
            body = {
                "project_id": self.project_id,
                "limit": _FEEDBACK_BATCH_LIMIT,
                "query": {
                    "$expr": {
                        "$in": [
                            {"$getField": "weave_ref"},
                            [{"$literal": r} for r in batch],
                        ]
                    }
                },
            }
            resp = self._http.post("/feedback/query", json=body)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("result") or data.get("feedback") or data.get("results") or []
            if len(items) >= _FEEDBACK_BATCH_LIMIT:
                raise RuntimeError(
                    "feedback hydration reached the "
                    f"{_FEEDBACK_BATCH_LIMIT}-row limit for refs: {', '.join(batch)}"
                )
            for fb in items:
                key = (fb.get("weave_ref", ""), fb.get("feedback_type", ""))
                result.setdefault(key, []).append(fb)
        return result

    def query_existing_feedback(self, ref: str, feedback_type: str) -> list[dict]:
        body = {
            "project_id": self.project_id,
            "query": {
                "$expr": {
                    "$and": [
                        {"$eq": [{"$getField": "weave_ref"}, {"$literal": ref}]},
                        {
                            "$eq": [
                                {"$getField": "feedback_type"},
                                {"$literal": feedback_type},
                            ]
                        },
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
            "query": {"$expr": {"$eq": [{"$getField": "id"}, {"$literal": feedback_id}]}},
        }
        resp = self._http.post("/feedback/purge", json=body)
        resp.raise_for_status()

    def delete_feedback_ids(self, feedback_list: list[dict]) -> None:
        for fb in feedback_list:
            self.delete_feedback(fb["id"])

    def query_all_feedback(self, ref: str) -> list[dict]:
        body = {
            "project_id": self.project_id,
            "query": {"$expr": {"$eq": [{"$getField": "weave_ref"}, {"$literal": ref}]}},
        }
        resp = self._http.post("/feedback/query", json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("feedback") or data.get("results") or []

    def query_all_feedback_batch(
        self,
        refs: list[str],
        *,
        batch_size: int = 100,
    ) -> dict[str, list[dict]]:
        """Fetch all feedback types for exact refs in bounded requests."""

        requested = list(dict.fromkeys(ref for ref in refs if ref))
        grouped = {ref: [] for ref in requested}
        for (ref, _feedback_type), items in self.query_existing_feedback_batch(
            requested,
            batch_size=batch_size,
        ).items():
            if ref in grouped:
                grouped[ref].extend(items)
        return grouped

    def query_feedback_for_refs(self, refs: list[str]) -> list[dict]:
        """Fetch complete feedback per exact ref, without a project-wide page cap."""
        feedback: list[dict] = []
        for ref in dict.fromkeys(ref for ref in refs if ref):
            feedback.extend(self.query_all_feedback(ref))
        return feedback

    def query_project_feedback(
        self,
        feedback_type_prefix: str = FEEDBACK_PREFIX,
        limit: int = 1000,
    ) -> list[dict]:
        """Query all signals feedback for the project."""
        body = {
            "project_id": self.project_id,
            "query": {
                "$expr": {
                    "$contains": {
                        "input": {"$getField": "feedback_type"},
                        "substr": {"$literal": feedback_type_prefix},
                    }
                }
            },
            "limit": limit,
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
    for key in (
        "custom_attrs_string",
        "custom_attrs_int",
        "custom_attrs_float",
        "custom_attrs_bool",
    ):
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
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None
