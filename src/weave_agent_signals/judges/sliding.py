"""One reviewer's resumable digest-window-merge judging pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import ValidationError

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.inference import (
    ChatClient,
    InferenceCancelled,
    InferenceContextExceeded,
    JsonSchemaSpec,
)
from weave_agent_signals.judges.records import (
    JudgeCallAudit,
    JudgeCallRecord,
    judge_request_id,
)
from weave_agent_signals.judges.review import AttemptObservation, InferenceStepAudit
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS, Rubric
from weave_agent_signals.judges.sliding_contracts import (
    CHUNK_DIGEST_SCHEMA,
    MERGED_VERDICT_SCHEMA,
    WINDOW_FINDINGS_SCHEMA,
    ChunkDigest,
    MergedVerdict,
    WindowFinding,
    WindowFindings,
    bind_chunk_digest_schema,
    bind_merged_verdict_schema,
    bind_window_findings_schema,
    parse_chunk_digest,
    parse_merged_verdict,
    parse_window_findings,
)
from weave_agent_signals.judges.tokens import count_tokens
from weave_agent_signals.judges.windowing import (
    _turn_evidence_ids,
    build_window_plan,
    render_raw_turn,
    render_raw_window,
)
from weave_agent_signals.models import SessionView
from weave_agent_signals.run_config import JudgingContextPolicy, PositionedJudge, RubricDescriptor

CallLoader = Callable[[str], JudgeCallRecord | None]
CallRecorder = Callable[[JudgeCallRecord], object]
ActivityRecorder = Callable[[Mapping[str, object]], None]

log = logging.getLogger("weave_agent_signals.judges")

_ERROR_TEXT_LIMIT = 500
SLIDING_PROTOCOL_VERSION = "3"

_DIGEST_SYSTEM_TEMPLATE = (
    "PHASE: digest\nCreate a rubric-neutral factual digest of the supplied raw chunk. "
    "Preserve important actions, results, omissions, corrections, and constraints. Every digest "
    "must cite at least one ID from ALLOWED_EVIDENCE_IDS. Do not cite any other ID; the response "
    "schema enforces the exact chunk and evidence scope. Return the requested JSON."
)
_DIGEST_USER_TEMPLATE = (
    "EXPECTED_CHUNK_ID: {chunk_id}\n"
    "EXAMPLE_EVIDENCE_ID: {example_evidence_id}\n"
    "ALLOWED_EVIDENCE_IDS: {allowed_evidence_ids}\n"
    "RAW_CHUNK:\n{raw_text}"
)
_WINDOW_SYSTEM_TEMPLATE = (
    "PHASE: window\n{rubric_system}\nReturn bounded findings, not a score. Every finding must "
    "cite at least one ID from ALLOWED_FINDING_EVIDENCE_IDS. If no supported finding exists, "
    'return "findings": [] instead of an uncited finding. Do not cite any other ID; the response '
    "schema enforces the active raw window's exact evidence scope. Finding IDs need only be "
    "unique within this response; aggregation scopes them to EXPECTED_WINDOW_ID."
)
_WINDOW_USER_TEMPLATE = (
    "EXPECTED_WINDOW_ID: {window_id}\n"
    "EXAMPLE_EVIDENCE_ID: {example_evidence_id}\n"
    "ALLOWED_FINDING_EVIDENCE_IDS: {allowed_evidence_ids}\n"
    "RUBRIC_CRITERIA:\n{rubric_criteria}\n{sections}"
)
_MERGE_SYSTEM_TEMPLATE = (
    "PHASE: merge\n{rubric_system}\nReturn one anchored session verdict with evidence-cited "
    "behavioral feedback. Behavioral feedback describes what the agent did or should do; "
    "reflection separately decides whether and how to edit managed instructions. The response "
    "schema limits citations to this session. A scored verdict must cite at least one ID from "
    "ALLOWED_EVIDENCE_IDS; an insufficient_evidence verdict must cite none."
)
_MERGE_USER_TEMPLATE = (
    "COVERAGE_MANIFEST: {coverage_manifest}\n"
    "ORDERED_CHUNK_DIGESTS: {digest_context}\n"
    "ORDERED_DEDUPLICATED_FINDINGS: {finding_context}\n"
    "ALLOWED_EVIDENCE_IDS: {allowed_evidence_ids}\n"
    "RUBRIC_CRITERIA:\n{rubric_criteria}"
)


class _JudgeInvocationFailure(RuntimeError):
    def __init__(self, error_type: str) -> None:
        super().__init__("judge invocation failed")
        self.error_type = error_type


class _ReviewInfrastructureFailure(RuntimeError):
    def __init__(self, error_type: str) -> None:
        super().__init__("review infrastructure failed")
        self.error_type = error_type


@dataclass(frozen=True)
class _PendingCall:
    request_id: str
    phase: Literal["digest", "window", "merge"]
    rubric_id: str | None
    audit: JudgeCallAudit


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def sliding_protocol_contract_manifest() -> dict[str, object]:
    """Return the content-derived prompt and schema contract for artifact identity."""

    return {
        "protocol_version": SLIDING_PROTOCOL_VERSION,
        "prompt_templates": {
            "digest_system": _DIGEST_SYSTEM_TEMPLATE,
            "digest_user": _DIGEST_USER_TEMPLATE,
            "window_system": _WINDOW_SYSTEM_TEMPLATE,
            "window_user": _WINDOW_USER_TEMPLATE,
            "merge_system": _MERGE_SYSTEM_TEMPLATE,
            "merge_user": _MERGE_USER_TEMPLATE,
        },
        "schemas": {
            "digest": {
                "name": CHUNK_DIGEST_SCHEMA.name,
                "schema": dict(CHUNK_DIGEST_SCHEMA.schema),
            },
            "window": {
                "name": WINDOW_FINDINGS_SCHEMA.name,
                "schema": dict(WINDOW_FINDINGS_SCHEMA.schema),
            },
            "merge": {
                "name": MERGED_VERDICT_SCHEMA.name,
                "schema": dict(MERGED_VERDICT_SCHEMA.schema),
            },
        },
    }


def sliding_protocol_contract_digest() -> str:
    """Return the content digest bound into every sliding artifact ID."""

    return _sha256(sliding_protocol_contract_manifest())


def _bounded_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        parts = []
        for issue in error.errors(include_url=False, include_input=False):
            location = ".".join(str(part) for part in issue.get("loc", ()))
            message = " ".join(str(issue.get("msg", "validation failed")).split())
            parts.append(f"{location}: {message}" if location else message)
        return "; ".join(parts)[:_ERROR_TEXT_LIMIT] or "structured output validation failed"
    if isinstance(error, (_JudgeInvocationFailure, _ReviewInfrastructureFailure, ValueError)):
        return " ".join(str(error).split())[:_ERROR_TEXT_LIMIT] or type(error).__name__
    return "sliding review failed"


def _review_callback(callback: Callable[..., Any], *args: object) -> Any:
    try:
        return callback(*args)
    except InferenceCancelled:
        raise
    except Exception as error:
        raise _ReviewInfrastructureFailure(type(error).__name__) from None


def _normalized_usage(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and key.strip() and type(item) is int and item >= 0
    }


def _add_usage(total: dict[str, int], addition: Mapping[str, int]) -> None:
    for key, value in addition.items():
        total[key] = total.get(key, 0) + value


class SlidingReviewer:
    """Execute every raw window and one final merge for a positioned reviewer."""

    def __init__(
        self,
        *,
        session: SessionView,
        judge: PositionedJudge,
        plan_id: str,
        window_plan: Mapping[str, object],
        context_policy: JudgingContextPolicy,
        client: ChatClient,
        load_call: CallLoader,
        record_call: CallRecorder,
        is_cancelled: Callable[[], bool] = lambda: False,
        activity: ActivityRecorder | None = None,
    ) -> None:
        if not isinstance(session, SessionView):
            raise TypeError("session must be a SessionView")
        if not isinstance(judge, PositionedJudge):
            raise TypeError("judge must be a PositionedJudge")
        if not isinstance(context_policy, JudgingContextPolicy):
            raise TypeError("context_policy must be a JudgingContextPolicy")
        if not isinstance(plan_id, str) or not plan_id.startswith("sha256:"):
            raise ValueError("pinned judging plan ID is invalid")
        expected_plan = build_window_plan(
            session,
            context_policy,
            judge.max_input_tokens,
            judge.token_counter,
        )
        expected_slim = {
            key: expected_plan[key]
            for key in (
                "input_cap_tokens",
                "raw_budget_tokens",
                "target_raw_tokens",
                "token_counter",
                "capacity_reserve_tokens",
                "merge_input_tokens",
            )
        }
        expected_slim["windows"] = [
            {key: value[key] for key in ("index", "core_trace_ids", "raw_trace_ids", "raw_tokens")}
            for value in expected_plan["windows"]
        ]
        if dict(window_plan) != expected_slim:
            raise ValueError(
                "pinned window plan does not match current session evidence and policy"
            )

        self.session = session
        self.judge = judge
        self.plan_id = plan_id
        self.window_plan = expected_plan
        self.context_policy = context_policy
        self.client = client
        self._load_call = load_call
        self._record_call = record_call
        self._is_cancelled = is_cancelled
        self._activity = activity
        self._digest_lock = threading.Lock()
        self._digests: tuple[ChunkDigest, ...] | None = None
        self._digest_steps: tuple[InferenceStepAudit, ...] | None = None
        self._positions = {
            turn.trace_id: position for position, turn in enumerate(session.turns, start=1)
        }
        self._turns = {turn.trace_id: turn for turn in session.turns}
        self._all_evidence_ids = tuple(
            evidence_id for turn in session.turns for evidence_id in _turn_evidence_ids(turn)
        )

    @property
    def _windows(self) -> tuple[Mapping[str, object], ...]:
        windows = self.window_plan["windows"]
        if not isinstance(windows, list) or any(
            not isinstance(window, Mapping) for window in windows
        ):
            raise ValueError("pinned window plan contains invalid windows")
        return tuple(windows)

    def _chunk_id(self, window: Mapping[str, object]) -> str:
        return _sha256(
            {
                "plan_id": self.plan_id,
                "index": window["index"],
                "core_trace_ids": window["core_trace_ids"],
            }
        )

    def _rubric(self, descriptor: RubricDescriptor) -> Rubric:
        if not isinstance(descriptor, RubricDescriptor):
            raise TypeError("rubric must be a RubricDescriptor")
        try:
            current = build_rubric_catalog().rubric(descriptor.id)
            rubric = SESSION_RUBRICS[descriptor.id]
        except KeyError as error:
            raise ValueError(f"unknown pinned rubric: {descriptor.id}") from error
        if descriptor != current:
            raise ValueError(f"pinned rubric {descriptor.id} does not match current prompt content")
        return rubric

    def _check_cancelled(self) -> None:
        if _review_callback(self._is_cancelled):
            raise InferenceCancelled("sliding reviewer inference cancelled")

    def _emit_activity(self, event: Mapping[str, object]) -> None:
        if self._activity is None:
            return
        try:
            self._activity(dict(event))
        except Exception as error:
            log.warning(
                "Sliding judge activity callback failed: error_type=%s",
                type(error).__name__,
            )

    def _messages_fit(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        schema: JsonSchemaSpec,
    ) -> int:
        input_tokens = count_tokens(
            _canonical_json(
                {
                    "messages": messages,
                    "response_schema": schema.schema,
                }
            ),
            self.judge.token_counter,
        )
        if (
            input_tokens + max_tokens + self.context_policy.safety_reserve_tokens
            > self.judge.max_input_tokens
        ):
            raise InferenceContextExceeded()
        return input_tokens

    def _infer(
        self,
        *,
        phase: Literal["digest", "window", "merge"],
        messages: list[dict[str, str]],
        max_tokens: int,
        schema: JsonSchemaSpec,
        steps: list[InferenceStepAudit],
        usage: dict[str, int],
        rubric_id: str | None = None,
        rubric_label: str | None = None,
        item_index: int | None = None,
        item_total: int | None = None,
    ) -> tuple[Mapping[str, Any], _PendingCall | None]:
        estimated_input_tokens = self._messages_fit(messages, max_tokens, schema)
        request_id = judge_request_id(
            requested_model_id=self.judge.id,
            provider_model=self.judge.provider_model,
            messages=messages,
            response_schema=schema,
            temperature=0.0,
            max_tokens=max_tokens,
            protocol_version=SLIDING_PROTOCOL_VERSION,
        )
        stored = _review_callback(self._load_call, request_id)
        if stored is not None:
            if stored.result is None:
                raise ValueError("reusable judge call has no result")
            steps.append(
                InferenceStepAudit(
                    phase=phase,
                    artifact_id=request_id,
                    requested_model=self.judge.id,
                    resolved_model=stored.audit.resolved_model,
                    usage=stored.audit.usage,
                    output_mode=stored.audit.output_mode,
                    schema_name=stored.audit.schema_name,
                    schema_fallback_reason=stored.audit.schema_fallback_reason,
                    transport_request_count=stored.audit.transport_request_count,
                    raw_output_digest=stored.audit.raw_output_digest,
                    reused=True,
                )
            )
            return stored.result, None
        self._check_cancelled()
        if phase == "digest":
            message = f"{self.judge.label} is digesting chunk {item_index} of {item_total}"
        elif phase == "window":
            message = (
                f"{self.judge.label} is reviewing window {item_index} of {item_total} for "
                f"{rubric_label}"
            )
        else:
            message = f"{self.judge.label} is merging {rubric_label}"
        event: dict[str, object] = {
            "phase": f"{phase}_started",
            "message": message,
            "model": self.judge.id,
            "conversation_id": self.session.conversation_id,
            "artifact_id": request_id,
            "estimated_input_tokens": estimated_input_tokens,
            "max_output_tokens": max_tokens,
            "model_context_tokens": self.judge.max_input_tokens,
        }
        if rubric_id is not None:
            event["rubric"] = rubric_id
        if item_index is not None:
            event["item_index"] = item_index
        if item_total is not None:
            event["item_total"] = item_total
        self._emit_activity(event)
        try:
            parsed, response = self.client.chat_json(
                model=self.judge.provider_model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_schema=schema,
            )
        except InferenceCancelled:
            raise
        except Exception as error:
            request_count = getattr(error, "_transport_request_count", 1)
            count = request_count if type(request_count) is int and request_count > 0 else 1
            audit = JudgeCallAudit(
                schema_name=schema.name,
                transport_request_count=count,
                error_type=type(error).__name__,
                message=_bounded_error(error),
            )
            steps.append(
                InferenceStepAudit(
                    phase=phase,
                    artifact_id=request_id,
                    requested_model=self.judge.id,
                    resolved_model=None,
                    usage={},
                    output_mode=None,
                    schema_name=schema.name,
                    schema_fallback_reason=None,
                    transport_request_count=count,
                    raw_output_digest=None,
                )
            )
            _review_callback(
                self._record_call,
                JudgeCallRecord(
                    request_id=request_id,
                    phase=phase,
                    conversation_id=self.session.conversation_id,
                    reviewer_position=self.judge.position,
                    requested_model_id=self.judge.id,
                    rubric_id=rubric_id,
                    status="failed",
                    reusable=False,
                    result=None,
                    audit=audit,
                    created_at=datetime.now(timezone.utc),
                ),
            )
            if isinstance(error, InferenceContextExceeded):
                raise
            raise _JudgeInvocationFailure(type(error).__name__) from None

        normalized_usage = _normalized_usage(response.usage)
        _add_usage(usage, normalized_usage)
        audit = JudgeCallAudit(
            resolved_model=response.model,
            usage=normalized_usage,
            output_mode=response.output_mode,
            schema_name=response.schema_name or schema.name,
            schema_fallback_reason=response.schema_fallback_reason,
            transport_request_count=response.transport_request_count,
            raw_output_digest=response.raw_output_digest,
        )
        steps.append(
            InferenceStepAudit(
                phase=phase,
                artifact_id=request_id,
                requested_model=self.judge.id,
                resolved_model=audit.resolved_model,
                usage=audit.usage,
                output_mode=audit.output_mode,
                schema_name=audit.schema_name,
                schema_fallback_reason=audit.schema_fallback_reason,
                transport_request_count=audit.transport_request_count,
                raw_output_digest=audit.raw_output_digest,
            )
        )
        return parsed, _PendingCall(request_id, phase, rubric_id, audit)

    def _record_success(self, pending: _PendingCall | None, result: Mapping[str, Any]) -> None:
        if pending is None:
            return
        _review_callback(
            self._record_call,
            JudgeCallRecord(
                request_id=pending.request_id,
                phase=pending.phase,
                conversation_id=self.session.conversation_id,
                reviewer_position=self.judge.position,
                requested_model_id=self.judge.id,
                rubric_id=pending.rubric_id,
                status="succeeded",
                reusable=True,
                result=dict(result),
                audit=pending.audit,
                created_at=datetime.now(timezone.utc),
            ),
        )

    def _record_validation_failure(
        self,
        pending: _PendingCall | None,
        error: Exception,
    ) -> None:
        if pending is None:
            return
        audit = pending.audit.model_copy(
            update={"error_type": type(error).__name__, "message": _bounded_error(error)}
        )
        _review_callback(
            self._record_call,
            JudgeCallRecord(
                request_id=pending.request_id,
                phase=pending.phase,
                conversation_id=self.session.conversation_id,
                reviewer_position=self.judge.position,
                requested_model_id=self.judge.id,
                rubric_id=pending.rubric_id,
                status="failed",
                reusable=False,
                result=None,
                audit=audit,
                created_at=datetime.now(timezone.utc),
            ),
        )

    def _core_evidence(self, window: Mapping[str, object]) -> tuple[str, tuple[str, ...]]:
        trace_ids = window["core_trace_ids"]
        if not isinstance(trace_ids, list):
            raise ValueError("pinned window core_trace_ids are invalid")
        selected = [self._turns[trace_id] for trace_id in trace_ids]
        text = "\n\n".join(
            render_raw_turn(turn, self._positions[turn.trace_id]) for turn in selected
        )
        evidence_ids = tuple(
            evidence_id for turn in selected for evidence_id in _turn_evidence_ids(turn)
        )
        return text, evidence_ids

    def _digest_messages(
        self,
        *,
        chunk_id: str,
        raw_text: str,
        evidence_ids: Sequence[str],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": _DIGEST_SYSTEM_TEMPLATE,
            },
            {
                "role": "user",
                "content": _DIGEST_USER_TEMPLATE.format(
                    chunk_id=chunk_id,
                    example_evidence_id=evidence_ids[0],
                    allowed_evidence_ids=_canonical_json(list(evidence_ids)),
                    raw_text=raw_text,
                ),
            },
        ]

    def _load_or_create_digest(
        self,
        window: Mapping[str, object],
        *,
        steps: list[InferenceStepAudit],
        usage: dict[str, int],
    ) -> ChunkDigest:
        chunk_id = self._chunk_id(window)
        raw_text, evidence_ids = self._core_evidence(window)
        response_schema = bind_chunk_digest_schema(chunk_id, evidence_ids)
        payload, pending = self._infer(
            phase="digest",
            messages=self._digest_messages(
                chunk_id=chunk_id,
                raw_text=raw_text,
                evidence_ids=evidence_ids,
            ),
            max_tokens=self.context_policy.output_reserve_tokens,
            schema=response_schema,
            steps=steps,
            usage=usage,
            item_index=int(window["index"]),
            item_total=len(self._windows),
        )
        try:
            digest = parse_chunk_digest(
                payload,
                allowed_evidence_ids=evidence_ids,
                max_tokens=self.context_policy.digest_max_tokens,
                expected_chunk_id=chunk_id,
            )
        except Exception as error:
            self._record_validation_failure(pending, error)
            raise
        self._record_success(pending, digest.model_dump(mode="json"))
        return digest

    def _ensure_digests(
        self,
        *,
        steps: list[InferenceStepAudit],
        usage: dict[str, int],
    ) -> tuple[ChunkDigest, ...]:
        if self._digests is not None:
            if self._digest_steps is None:
                raise AssertionError("digest provenance must accompany cached digests")
            steps.extend(replace(audit, reused=True) for audit in self._digest_steps)
            return self._digests
        with self._digest_lock:
            if self._digests is None:
                first_digest_step = len(steps)
                self._digests = tuple(
                    self._load_or_create_digest(window, steps=steps, usage=usage)
                    for window in self._windows
                )
                self._digest_steps = tuple(steps[first_digest_step:])
            else:
                if self._digest_steps is None:
                    raise AssertionError("digest provenance must accompany cached digests")
                steps.extend(replace(audit, reused=True) for audit in self._digest_steps)
            return self._digests

    def _window_messages(
        self,
        *,
        rubric: Rubric,
        window_index: int,
        window: Mapping[str, object],
        digests: Sequence[ChunkDigest],
    ) -> tuple[list[dict[str, str]], tuple[str, ...]]:
        raw = render_raw_window(self.session, window, self.judge.token_counter)
        sections: list[str] = []
        for index, digest in enumerate(digests):
            sections.append(f"CHUNK_INDEX: {index + 1}")
            if index == window_index:
                sections.extend(("CONTEXT_KIND: raw_window", raw.text))
            else:
                sections.extend(
                    (
                        "CONTEXT_KIND: chunk_digest",
                        f"CHUNK_ID: {digest.chunk_id}",
                        f"DIGEST_EVIDENCE_IDS: {_canonical_json(list(digest.evidence_ids))}",
                        digest.text,
                    )
                )
        return (
            [
                {
                    "role": "system",
                    "content": _WINDOW_SYSTEM_TEMPLATE.format(rubric_system=rubric.system_prompt),
                },
                {
                    "role": "user",
                    "content": _WINDOW_USER_TEMPLATE.format(
                        window_id=window["window_id"],
                        example_evidence_id=raw.evidence_ids[0],
                        allowed_evidence_ids=_canonical_json(list(raw.evidence_ids)),
                        rubric_criteria=rubric.criteria_text,
                        sections="\n\n".join(sections),
                    ),
                },
            ],
            raw.evidence_ids,
        )

    def _load_or_create_window_findings(
        self,
        *,
        descriptor: RubricDescriptor,
        rubric: Rubric,
        window_index: int,
        window: Mapping[str, object],
        digests: Sequence[ChunkDigest],
        steps: list[InferenceStepAudit],
        usage: dict[str, int],
    ) -> WindowFindings:
        messages, allowed_evidence_ids = self._window_messages(
            rubric=rubric,
            window_index=window_index,
            window=window,
            digests=digests,
        )
        response_schema = bind_window_findings_schema(
            str(window["window_id"]),
            allowed_evidence_ids,
        )
        payload, pending = self._infer(
            phase="window",
            messages=messages,
            max_tokens=self.context_policy.finding_max_tokens,
            schema=response_schema,
            steps=steps,
            usage=usage,
            rubric_id=descriptor.id,
            rubric_label=descriptor.label,
            item_index=window_index + 1,
            item_total=len(self._windows),
        )
        try:
            findings = parse_window_findings(
                payload,
                allowed_evidence_ids=allowed_evidence_ids,
                max_tokens=self.context_policy.finding_max_tokens,
                expected_window_id=str(window["window_id"]),
            )
        except Exception as error:
            self._record_validation_failure(pending, error)
            raise
        self._record_success(pending, findings.model_dump(mode="json"))
        return findings

    def _deduplicated_findings(
        self,
        values: Sequence[WindowFindings],
    ) -> tuple[WindowFinding, ...]:
        ordered: list[WindowFinding] = []
        semantic_seen: set[tuple[object, ...]] = set()
        for window in values:
            for finding in window.findings:
                semantic = (
                    finding.polarity,
                    finding.observation,
                    tuple(sorted(set(finding.evidence_ids))),
                )
                if semantic in semantic_seen:
                    continue
                semantic_seen.add(semantic)
                ordered.append(
                    finding.model_copy(
                        update={"finding_id": f"{window.window_id}:{finding.finding_id}"}
                    )
                )
        return tuple(ordered)

    def _merge_messages(
        self,
        *,
        rubric: Rubric,
        digests: Sequence[ChunkDigest],
        findings: Sequence[WindowFinding],
    ) -> list[dict[str, str]]:
        digest_context = [digest.model_dump(mode="json") for digest in digests]
        finding_context = [finding.model_dump(mode="json") for finding in findings]
        manifest = {
            "plan_id": self.plan_id,
            "window_ids": [window["window_id"] for window in self._windows],
            "raw_coverage_trace_ids": self.window_plan["raw_coverage_trace_ids"],
        }
        return [
            {
                "role": "system",
                "content": _MERGE_SYSTEM_TEMPLATE.format(rubric_system=rubric.system_prompt),
            },
            {
                "role": "user",
                "content": _MERGE_USER_TEMPLATE.format(
                    coverage_manifest=_canonical_json(manifest),
                    digest_context=_canonical_json(digest_context),
                    finding_context=_canonical_json(finding_context),
                    allowed_evidence_ids=_canonical_json(list(self._all_evidence_ids)),
                    rubric_criteria=rubric.criteria_text,
                ),
            },
        ]

    def _load_or_create_merge(
        self,
        *,
        descriptor: RubricDescriptor,
        rubric: Rubric,
        digests: Sequence[ChunkDigest],
        findings: Sequence[WindowFinding],
        steps: list[InferenceStepAudit],
        usage: dict[str, int],
    ) -> MergedVerdict:
        response_schema = bind_merged_verdict_schema(self._all_evidence_ids)
        payload, pending = self._infer(
            phase="merge",
            messages=self._merge_messages(rubric=rubric, digests=digests, findings=findings),
            max_tokens=self.context_policy.output_reserve_tokens,
            schema=response_schema,
            steps=steps,
            usage=usage,
            rubric_id=descriptor.id,
            rubric_label=descriptor.label,
        )
        try:
            verdict = parse_merged_verdict(payload, allowed_evidence_ids=self._all_evidence_ids)
        except Exception as error:
            self._record_validation_failure(pending, error)
            raise
        self._record_success(pending, verdict.model_dump(mode="json"))
        return verdict

    def review(self, rubric: RubricDescriptor) -> AttemptObservation:
        """Review one pinned rubric, returning a fail-closed final observation."""

        steps: list[InferenceStepAudit] = []
        usage: dict[str, int] = {}
        try:
            resolved_rubric = self._rubric(rubric)
            digests = self._ensure_digests(steps=steps, usage=usage)
            windows = tuple(
                self._load_or_create_window_findings(
                    descriptor=rubric,
                    rubric=resolved_rubric,
                    window_index=index,
                    window=window,
                    digests=digests,
                    steps=steps,
                    usage=usage,
                )
                for index, window in enumerate(self._windows)
            )
            findings = self._deduplicated_findings(windows)
            verdict = self._load_or_create_merge(
                descriptor=rubric,
                rubric=resolved_rubric,
                digests=digests,
                findings=findings,
                steps=steps,
                usage=usage,
            )
        except InferenceCancelled:
            raise
        except InferenceContextExceeded:
            last_step = steps[-1] if steps else None
            self._emit_activity(
                {
                    "phase": "context_capacity_skipped",
                    "message": f"{self.judge.label} exceeded its context capacity",
                    "model": self.judge.id,
                    "conversation_id": self.session.conversation_id,
                    "rubric": rubric.id,
                }
            )
            return AttemptObservation(
                status="skipped",
                skip_reason="insufficient_context_capacity",
                resolved_model=last_step.resolved_model if last_step is not None else None,
                score=None,
                rationale=None,
                usage=usage,
                error_type=None,
                message=None,
                output_mode=last_step.output_mode if last_step is not None else None,
                schema_name=last_step.schema_name if last_step is not None else None,
                schema_fallback_reason=(
                    last_step.schema_fallback_reason if last_step is not None else None
                ),
                transport_request_count=sum(
                    step.transport_request_count for step in steps if not step.reused
                ),
                raw_output_digest=(last_step.raw_output_digest if last_step is not None else None),
                steps=tuple(steps),
            )
        except Exception as error:
            last_step = steps[-1] if steps else None
            error_type = (
                error.error_type
                if isinstance(error, (_JudgeInvocationFailure, _ReviewInfrastructureFailure))
                else type(error).__name__
            )
            message = _bounded_error(error)
            if last_step is not None and not isinstance(
                error,
                (_JudgeInvocationFailure, _ReviewInfrastructureFailure),
            ):
                self._emit_activity(
                    {
                        "phase": "validation_failed",
                        "message": (
                            f"{self.judge.label} returned invalid {last_step.phase} output for "
                            f"{rubric.label}"
                        ),
                        "model": self.judge.id,
                        "conversation_id": self.session.conversation_id,
                        "rubric": rubric.id,
                        "artifact_id": last_step.artifact_id,
                        "error_category": error_type,
                        "provider_error_message": message,
                        "output_mode": last_step.output_mode,
                        "output_sha256": last_step.raw_output_digest,
                    }
                )
            return AttemptObservation(
                status="failed",
                resolved_model=last_step.resolved_model if last_step is not None else None,
                score=None,
                rationale=None,
                usage=usage,
                error_type=error_type,
                message=message,
                output_mode=last_step.output_mode if last_step is not None else None,
                schema_name=last_step.schema_name if last_step is not None else None,
                schema_fallback_reason=(
                    last_step.schema_fallback_reason if last_step is not None else None
                ),
                transport_request_count=sum(
                    step.transport_request_count for step in steps if not step.reused
                ),
                raw_output_digest=(last_step.raw_output_digest if last_step is not None else None),
                steps=tuple(steps),
            )

        common = {
            "resolved_model": steps[-1].resolved_model if steps else self.judge.id,
            "usage": usage,
            "error_type": None,
            "message": None,
            "evidence_ids": tuple(verdict.evidence_ids),
            "output_mode": steps[-1].output_mode if steps else None,
            "schema_name": MERGED_VERDICT_SCHEMA.name,
            "schema_fallback_reason": steps[-1].schema_fallback_reason if steps else None,
            "transport_request_count": sum(
                step.transport_request_count for step in steps if not step.reused
            ),
            "verdict_schema_version": verdict.schema_version,
            "raw_output_digest": steps[-1].raw_output_digest if steps else None,
            "steps": tuple(steps),
        }
        if verdict.status == "insufficient_evidence":
            return AttemptObservation(
                status="abstained",
                score=None,
                rationale=verdict.rationale,
                behavioral_feedback=None,
                **common,
            )
        if verdict.feedback is None:  # Contract validation makes this unreachable.
            raise AssertionError("scored merged verdict requires feedback")
        return AttemptObservation(
            status="succeeded",
            score=verdict.score,
            rationale=verdict.rationale,
            behavioral_feedback=verdict.feedback.model_dump(mode="json"),
            **common,
        )
