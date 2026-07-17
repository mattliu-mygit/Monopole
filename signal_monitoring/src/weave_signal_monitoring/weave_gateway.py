import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

import weave
from pydantic import ValidationError
from weave import Monitor
from weave.flow.scorer import Scorer
from weave.trace.objectify import register_object
from weave.trace.op import op
from weave.trace_server.agents.types import AgentSpansQueryReq
from weave.trace_server.interface.builtin_object_classes.llm_structured_model import (
    LLMStructuredCompletionModel,
)
from weave.trace_server.trace_server_interface import (
    FeedbackQueryReq,
    ObjectVersionFilter,
    ObjQueryReq,
)

from weave_signal_monitoring.catalog import (
    DEFAULT_MODEL,
    SAMPLING_RATE,
    TURN_OP_NAME,
    SignalDefinition,
)
from weave_signal_monitoring.models import (
    FeedbackRecord,
    ScoreOutput,
    SignalIdentity,
    TurnRecord,
)

_FINGERPRINT_RE = re.compile(r"^catalog_sha256=([0-9a-f]{64})$")
TRACE_ROLE_ATTRIBUTE = "weave_agent_signals.trace_role"
SIGNAL_TRACE_ROLE = "signal_evaluation"


class DefinitionConflict(RuntimeError):
    pass


class WeaveReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class InstallReport:
    created: tuple[str, ...]
    reused: tuple[str, ...]


class MonitorStore(Protocol):
    def get_definition(self, name: str) -> str | None: ...

    def create(self, definition: SignalDefinition, fingerprint: str) -> None: ...


@register_object
class LLMAsAJudgeScorer(Scorer):
    """Wire-compatible agent scorer without importing the full scorer extra."""

    model: LLMStructuredCompletionModel
    scoring_prompt: str

    @op(attributes={TRACE_ROLE_ATTRIBUTE: SIGNAL_TRACE_ROLE})
    def score(self, *, output: Any, **kwargs: Any) -> Any:
        prompt = self.scoring_prompt.format(output=output, **kwargs)
        return self.model.predict([{"role": "user", "content": prompt}])


def definition_fingerprint(definition: SignalDefinition) -> str:
    payload = {
        **asdict(definition),
        "monitor_name": definition.monitor_name,
        "scorer_name": definition.scorer_name,
        "model_name": definition.model_name,
        "scoring_prompt": definition.scoring_prompt,
        "model": DEFAULT_MODEL,
        "op_name": TURN_OP_NAME,
        "sampling_rate": SAMPLING_RATE,
        "active": True,
        "trace_role_attribute": TRACE_ROLE_ATTRIBUTE,
        "trace_role": SIGNAL_TRACE_ROLE,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_monitor(definition: SignalDefinition, fingerprint: str) -> Monitor:
    model = LLMStructuredCompletionModel(
        name=definition.model_name,
        llm_model_id=DEFAULT_MODEL,
        default_params={"temperature": 0, "response_format": "json_object"},
    )
    scorer = LLMAsAJudgeScorer(
        name=definition.scorer_name,
        model=model,
        scoring_prompt=definition.scoring_prompt,
    )
    return Monitor(
        name=definition.monitor_name,
        description=f"{definition.description}\ncatalog_sha256={fingerprint}",
        sampling_rate=SAMPLING_RATE,
        scorers=[scorer],
        op_names=[TURN_OP_NAME],
        active=False,
    )


def install_catalog(
    store: MonitorStore,
    catalog: tuple[SignalDefinition, ...],
) -> InstallReport:
    created: list[str] = []
    reused: list[str] = []
    for definition in catalog:
        fingerprint = definition_fingerprint(definition)
        existing = store.get_definition(definition.monitor_name)
        if existing == fingerprint:
            reused.append(definition.monitor_name)
            continue
        if existing is not None:
            raise DefinitionConflict(
                f"monitor {definition.monitor_name!r} exists with a conflicting definition"
            )
        store.create(definition, fingerprint)
        created.append(definition.monitor_name)
    return InstallReport(tuple(created), tuple(reused))


class WeaveGateway:
    def __init__(self, entity: str, project: str, *, client: Any | None = None):
        self.entity = entity
        self.project = project
        self.project_id = f"{entity}/{project}"
        self.client = client if client is not None else weave.init(self.project_id)

    def _read_monitor(self, name: str) -> Any | None:
        response = self.client.server.objs_query(
            ObjQueryReq(
                project_id=self.project_id,
                filter=ObjectVersionFilter(
                    object_ids=[name],
                    leaf_object_classes=["Monitor"],
                    latest_only=True,
                ),
            )
        )
        if not response.objs:
            return None
        if len(response.objs) != 1:
            raise DefinitionConflict(f"monitor {name!r} has multiple latest objects")
        return response.objs[0]

    @staticmethod
    def _fingerprint(monitor: Any, name: str) -> str:
        val = monitor.val
        if not isinstance(val, dict):
            raise DefinitionConflict(f"monitor {name!r} has a malformed definition")
        description = val.get("description")
        if not isinstance(description, str):
            raise DefinitionConflict(f"monitor {name!r} has no catalog fingerprint")
        matches = [
            match.group(1)
            for line in description.splitlines()
            if (match := _FINGERPRINT_RE.fullmatch(line))
        ]
        if len(matches) != 1:
            raise DefinitionConflict(f"monitor {name!r} has an invalid catalog fingerprint")
        return matches[0]

    def get_definition(self, name: str) -> str | None:
        monitor = self._read_monitor(name)
        if monitor is None:
            return None
        return self._fingerprint(monitor, name)

    def create(self, definition: SignalDefinition, fingerprint: str) -> None:
        monitor = build_monitor(definition, fingerprint)
        monitor.activate()
        if self.get_definition(definition.monitor_name) != fingerprint:
            raise DefinitionConflict(
                f"monitor {definition.monitor_name!r} could not be verified after activation"
            )

    def resolve_identities(
        self,
        catalog: tuple[SignalDefinition, ...],
    ) -> tuple[SignalIdentity, ...]:
        identities: list[SignalIdentity] = []
        for definition in catalog:
            monitor = self._read_monitor(definition.monitor_name)
            if monitor is None:
                raise DefinitionConflict(f"monitor {definition.monitor_name!r} is not installed")
            actual = self._fingerprint(monitor, definition.monitor_name)
            expected = definition_fingerprint(definition)
            if actual != expected:
                raise DefinitionConflict(
                    f"monitor {definition.monitor_name!r} has a conflicting definition"
                )
            scorers = monitor.val.get("scorers")
            if not isinstance(scorers, list) or len(scorers) != 1:
                raise DefinitionConflict(
                    f"monitor {definition.monitor_name!r} must contain exactly one scorer"
                )
            scorer_ref = scorers[0]
            if not isinstance(scorer_ref, str) or not scorer_ref.startswith("weave:///"):
                raise DefinitionConflict(
                    f"monitor {definition.monitor_name!r} contains a non-reference scorer"
                )
            identities.append(
                SignalIdentity(
                    slug=definition.slug,
                    version=definition.version,
                    monitor_name=definition.monitor_name,
                    scorer_ref=scorer_ref,
                )
            )
        return tuple(identities)

    def read_feedback(
        self,
        identities: tuple[SignalIdentity, ...],
        since: datetime,
        until: datetime,
    ) -> tuple[FeedbackRecord, ...]:
        if since >= until:
            raise WeaveReadError("feedback query requires since before until")
        refs = [identity.scorer_ref for identity in identities]
        if not refs:
            return ()
        known_refs = set(refs)
        page_size = 100
        offset = 0
        seen_ids: set[str] = set()
        records: list[FeedbackRecord] = []
        while True:
            request = FeedbackQueryReq(
                project_id=self.project_id,
                query={
                    "$expr": {
                        "$and": [
                            {
                                "$in": [
                                    {"$getField": "runnable_ref"},
                                    [{"$literal": ref} for ref in refs],
                                ]
                            },
                            {
                                "$gte": [
                                    {"$getField": "created_at"},
                                    {"$literal": since.isoformat()},
                                ]
                            },
                            {
                                "$not": [
                                    {
                                        "$gt": [
                                            {"$getField": "created_at"},
                                            {"$literal": until.isoformat()},
                                        ]
                                    }
                                ]
                            },
                        ]
                    }
                },
                sort_by=[{"field": "created_at", "direction": "asc"}],
                limit=page_size,
                offset=offset,
            )
            try:
                rows = self.client.server.feedback_query(request).result
            except Exception as error:
                raise WeaveReadError("feedback query failed") from error
            if len(rows) > page_size:
                raise WeaveReadError("feedback query returned an oversized page")
            for raw in rows:
                feedback_id = raw.get("id") if isinstance(raw, dict) else None
                if not isinstance(feedback_id, str) or not feedback_id:
                    raise WeaveReadError("feedback query returned a row without an id")
                if feedback_id in seen_ids:
                    raise WeaveReadError(f"feedback query repeated feedback id: {feedback_id}")
                seen_ids.add(feedback_id)
                runnable_ref = raw.get("runnable_ref")
                if runnable_ref not in known_refs:
                    continue
                feedback_type = raw.get("feedback_type")
                if feedback_type != "wandb.agent_monitor":
                    continue
                payload = raw.get("payload")
                if not isinstance(payload, dict) or "output" not in payload:
                    continue
                try:
                    output = ScoreOutput.model_validate(payload["output"])
                    ratings = raw.get("scorer_ratings")
                    if not isinstance(ratings, dict) or ratings.get("_rating_") != output.rating:
                        raise ValueError("typed rating does not match scorer output")
                    records.append(
                        FeedbackRecord(
                            id=feedback_id,
                            weave_ref=raw["weave_ref"],
                            runnable_ref=runnable_ref,
                            created_at=raw["created_at"],
                            output=output,
                        )
                    )
                except (KeyError, TypeError, ValueError, ValidationError) as error:
                    raise WeaveReadError(f"malformed eligible feedback: {feedback_id}") from error
            if len(rows) < page_size:
                break
            offset += len(rows)
        return tuple(records)

    @staticmethod
    def _turn_ref(ref: str) -> tuple[str, str]:
        parts = [unquote(part) for part in urlparse(ref).path.split("/") if part]
        if len(parts) < 2 or parts[-2] not in {"call", "agent_turn"} or not parts[-1]:
            raise WeaveReadError(f"unsupported turn ref: {ref!r}")
        return parts[-2], parts[-1]

    @staticmethod
    def _conversation_id(call: Any) -> str | None:
        thread_id = getattr(call, "thread_id", None)
        if isinstance(thread_id, str) and thread_id:
            return thread_id
        attributes = getattr(call, "attributes", None)
        if isinstance(attributes, dict):
            value = attributes.get("conversation_id")
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _is_turn(call: Any) -> bool:
        call_id = getattr(call, "id", None)
        turn_id = getattr(call, "turn_id", None)
        if turn_id is not None:
            return turn_id == call_id
        return getattr(call, "parent_id", None) in {None, ""}

    @staticmethod
    def _message_text(message: Any) -> str | None:
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, str):
            stripped = content.strip()
            if not stripped:
                return None
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            if isinstance(decoded, list):
                return WeaveGateway._message_text({"content": decoded})
            return stripped
        if not isinstance(content, list):
            return None
        parts: list[str] = []
        for part in content:
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return " ".join(parts) or None

    @classmethod
    def _user_request(cls, call: Any) -> str | None:
        inputs = getattr(call, "inputs", None)
        if not isinstance(inputs, dict):
            return None
        messages = inputs.get("input_messages")
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                return None
        if not isinstance(messages, list):
            return None
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                if text := cls._message_text(message):
                    return text
        return None

    @classmethod
    def _agent_span_record(cls, span: Any, *, turn_id: str | None = None) -> TurnRecord:
        conversation_id = getattr(span, "conversation_id", None)
        if not isinstance(conversation_id, str) or not conversation_id:
            raise WeaveReadError(
                f"turn {getattr(span, 'trace_id', '')!r} has no conversation identity"
            )
        messages = getattr(span, "input_messages", None)
        user_request = None
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    user_request = cls._message_text(message)
                    if user_request:
                        break
        display_name = getattr(span, "conversation_name", None)
        started_at = span.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        else:
            started_at = started_at.astimezone(timezone.utc)
        ended_at = getattr(span, "ended_at", None)
        if ended_at is not None:
            if ended_at.tzinfo is None:
                ended_at = ended_at.replace(tzinfo=timezone.utc)
            else:
                ended_at = ended_at.astimezone(timezone.utc)
        return TurnRecord(
            turn_id=turn_id or span.trace_id,
            conversation_id=conversation_id,
            started_at=started_at,
            ended_at=ended_at,
            display_name=display_name or None,
            user_request=user_request,
        )

    @classmethod
    def _turn_record(cls, call: Any, *, turn_id: str | None = None) -> TurnRecord:
        conversation_id = cls._conversation_id(call)
        if conversation_id is None:
            raise WeaveReadError(f"turn {getattr(call, 'id', '')!r} has no conversation identity")
        return TurnRecord(
            turn_id=turn_id or call.id,
            conversation_id=conversation_id,
            started_at=call.started_at,
            ended_at=getattr(call, "ended_at", None),
            display_name=getattr(call, "display_name", None),
            user_request=cls._user_request(call),
        )

    def _get_calls(self, **kwargs: Any) -> list[Any]:
        try:
            return list(self.client.get_calls(**kwargs))
        except Exception as error:
            raise WeaveReadError("turn query failed") from error

    def _get_agent_spans(self, query: dict[str, Any]) -> list[Any]:
        try:
            response = self.client.server.agent_spans_query(
                AgentSpansQueryReq(
                    project_id=self.project_id,
                    query=query,
                    include_details=True,
                    limit=10_000,
                )
            )
        except Exception as error:
            raise WeaveReadError("agent span query failed") from error
        if len(response.spans) == 10_000:
            raise WeaveReadError("agent span query reached its completeness limit")
        return response.spans

    def read_turns(self, refs: tuple[str, ...]) -> tuple[TurnRecord, ...]:
        requested = list(dict.fromkeys(self._turn_ref(ref) for ref in refs))
        call_ids = [value for kind, value in requested if kind == "call"]
        trace_ids = [value for kind, value in requested if kind == "agent_turn"]
        by_request: dict[tuple[str, str], list[Any]] = defaultdict(list)
        if call_ids:
            for call in self._get_calls(filter={"call_ids": call_ids}):
                if self._is_turn(call) and call.id in call_ids:
                    by_request[("call", call.id)].append(call)
        if trace_ids:
            query = {
                "$expr": {
                    "$and": [
                        {
                            "$in": [
                                {"$getField": "trace_id"},
                                [{"$literal": value} for value in trace_ids],
                            ]
                        },
                        {
                            "$eq": [
                                {"$getField": "parent_span_id"},
                                {"$literal": ""},
                            ]
                        },
                    ]
                }
            }
            for span in self._get_agent_spans(query):
                if span.parent_span_id == "" and span.trace_id in trace_ids:
                    by_request[("agent_turn", span.trace_id)].append(span)

        result: list[TurnRecord] = []
        for request in requested:
            matches = by_request.get(request, [])
            if not matches:
                raise WeaveReadError(f"missing requested turn: {request[1]}")
            if len(matches) != 1:
                raise WeaveReadError(f"ambiguous requested turn: {request[1]}")
            if request[0] == "agent_turn":
                result.append(self._agent_span_record(matches[0], turn_id=request[1]))
            else:
                result.append(self._turn_record(matches[0]))
        return tuple(result)

    def read_conversation_turns(
        self,
        conversation_ids: tuple[str, ...],
    ) -> tuple[TurnRecord, ...]:
        requested = tuple(dict.fromkeys(value for value in conversation_ids if value))
        if not requested:
            return ()
        calls = self._get_calls(filter={"thread_ids": list(requested)})
        calls.extend(
            self._get_calls(
                query={
                    "$expr": {
                        "$in": [
                            {"$getField": "attributes.conversation_id"},
                            [{"$literal": value} for value in requested],
                        ]
                    }
                }
            )
        )
        agent_query = {
            "$expr": {
                "$and": [
                    {
                        "$in": [
                            {"$getField": "conversation_id"},
                            [{"$literal": value} for value in requested],
                        ]
                    },
                    {
                        "$eq": [
                            {"$getField": "parent_span_id"},
                            {"$literal": ""},
                        ]
                    },
                ]
            }
        }
        agent_spans = self._get_agent_spans(agent_query)
        turns_by_id: dict[str, TurnRecord] = {}
        found_conversations: set[str] = set()
        for call in calls:
            if not self._is_turn(call):
                continue
            conversation_id = self._conversation_id(call)
            if conversation_id not in requested:
                continue
            record_id = call.id if getattr(call, "thread_id", None) else call.trace_id
            record = self._turn_record(call, turn_id=record_id)
            existing = turns_by_id.get(record.turn_id)
            if existing is not None and existing != record:
                raise WeaveReadError(f"inconsistent duplicate turn: {record.turn_id}")
            turns_by_id[record.turn_id] = record
            found_conversations.add(conversation_id)
        for span in agent_spans:
            if span.parent_span_id != "" or span.conversation_id not in requested:
                continue
            record = self._agent_span_record(span)
            existing = turns_by_id.get(record.turn_id)
            if existing is not None and existing != record:
                raise WeaveReadError(f"inconsistent duplicate turn: {record.turn_id}")
            turns_by_id[record.turn_id] = record
            found_conversations.add(record.conversation_id)
        missing = [value for value in requested if value not in found_conversations]
        if missing:
            raise WeaveReadError(f"missing conversation hydration: {', '.join(missing)}")
        return tuple(sorted(turns_by_id.values(), key=lambda turn: (turn.started_at, turn.turn_id)))
