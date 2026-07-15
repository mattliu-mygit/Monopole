"""One reviewer's resumable digest-window-merge judging pipeline."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

from pydantic import ValidationError

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.inference import ChatClient, InferenceCancelled, JsonSchemaSpec
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
    parse_chunk_digest,
    parse_merged_verdict,
    parse_window_findings,
)
from weave_agent_signals.judges.windowing import (
    _turn_evidence_ids,
    build_window_plan,
    estimate_tokens,
    render_raw_turn,
    render_raw_window,
)
from weave_agent_signals.models import SessionView
from weave_agent_signals.run_config import JudgingContextPolicy, PositionedJudge, RubricDescriptor

ArtifactKind = Literal["chunk_digest", "window_findings", "merged_verdict"]
ArtifactLoader = Callable[[str], Mapping[str, Any] | None]
ArtifactRecorder = Callable[[str, Mapping[str, Any]], object]

_ARTIFACT_FIELDS = frozenset({"schema_version", "kind", "content_digest", "payload"})
_ARTIFACT_PAYLOAD_FIELDS = frozenset({"schema_version", "result", "audit"})
_AUDIT_FIELDS = frozenset(
    {
        "phase",
        "artifact_id",
        "requested_model",
        "resolved_model",
        "usage",
        "output_mode",
        "schema_name",
        "schema_fallback_reason",
        "transport_request_count",
        "raw_output_digest",
        "reused",
    }
)
_ERROR_TEXT_LIMIT = 500
SLIDING_PROTOCOL_VERSION = "2"

_DIGEST_SYSTEM_TEMPLATE = (
    "PHASE: digest\nCreate a rubric-neutral factual digest of the supplied raw chunk. "
    "Preserve important actions, results, omissions, corrections, and constraints. Cite only "
    "allowed evidence IDs. Return the requested JSON."
)
_DIGEST_USER_TEMPLATE = (
    "EXPECTED_CHUNK_ID: {chunk_id}\n"
    "EXAMPLE_EVIDENCE_ID: {example_evidence_id}\n"
    "ALLOWED_EVIDENCE_IDS: {allowed_evidence_ids}\n"
    "RAW_CHUNK:\n{raw_text}"
)
_WINDOW_SYSTEM_TEMPLATE = (
    "PHASE: window\n{rubric_system}\nReturn bounded findings, not a score. Cite only evidence "
    "IDs visible in the active raw window."
)
_WINDOW_USER_TEMPLATE = (
    "EXPECTED_WINDOW_ID: {window_id}\n"
    "EXAMPLE_EVIDENCE_ID: {example_evidence_id}\n"
    "ALLOWED_FINDING_EVIDENCE_IDS: {allowed_evidence_ids}\n"
    "RUBRIC_CRITERIA:\n{rubric_criteria}\n{sections}"
)
_MERGE_SYSTEM_TEMPLATE = (
    "PHASE: merge\n{rubric_system}\nReturn one anchored session verdict with evidence-cited "
    "behavioral feedback."
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


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return _sha256(dict(payload))


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


def _audit_payload(audit: InferenceStepAudit) -> dict[str, object]:
    return {
        "phase": audit.phase,
        "artifact_id": audit.artifact_id,
        "requested_model": audit.requested_model,
        "resolved_model": audit.resolved_model,
        "usage": dict(audit.usage),
        "output_mode": audit.output_mode,
        "schema_name": audit.schema_name,
        "schema_fallback_reason": audit.schema_fallback_reason,
        "transport_request_count": audit.transport_request_count,
        "raw_output_digest": audit.raw_output_digest,
        "reused": audit.reused,
    }


def _artifact_envelope(
    kind: ArtifactKind,
    result: Mapping[str, Any],
    audit: InferenceStepAudit,
) -> dict[str, Any]:
    normalized = json.loads(
        _canonical_json(
            {
                "schema_version": 1,
                "result": dict(result),
                "audit": _audit_payload(audit),
            }
        )
    )
    return {
        "schema_version": "1",
        "kind": kind,
        "content_digest": _payload_digest(normalized),
        "payload": normalized,
    }


def _artifact_payload(
    artifact: Mapping[str, Any],
    *,
    artifact_id: str,
    expected_kind: ArtifactKind,
    expected_phase: Literal["digest", "window", "merge"],
    expected_schema: JsonSchemaSpec,
    requested_model: str,
) -> tuple[Mapping[str, Any], InferenceStepAudit]:
    value = dict(artifact)
    if set(value) != _ARTIFACT_FIELDS:
        raise ValueError(f"artifact {artifact_id} has invalid envelope fields")
    if value["schema_version"] != "1" or value["kind"] != expected_kind:
        raise ValueError(f"artifact {artifact_id} has invalid identity")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError(f"artifact {artifact_id} payload must be an object")
    if value["content_digest"] != _payload_digest(payload):
        raise ValueError(f"artifact {artifact_id} content digest is invalid")
    payload_value = dict(payload)
    if set(payload_value) != _ARTIFACT_PAYLOAD_FIELDS or payload_value["schema_version"] != 1:
        raise ValueError(f"artifact {artifact_id} payload contract is invalid")
    result = payload_value["result"]
    audit_value = payload_value["audit"]
    if not isinstance(result, Mapping) or not isinstance(audit_value, Mapping):
        raise ValueError(f"artifact {artifact_id} result and audit must be objects")
    audit_data = dict(audit_value)
    if set(audit_data) != _AUDIT_FIELDS:
        raise ValueError(f"artifact {artifact_id} audit fields are invalid")
    audit = InferenceStepAudit(**audit_data)  # type: ignore[arg-type]
    if (
        audit.phase != expected_phase
        or audit.artifact_id != artifact_id
        or audit.requested_model != requested_model
        or audit.schema_name != expected_schema.name
        or audit.resolved_model is None
        or audit.output_mode is None
        or audit.transport_request_count < 1
        or audit.raw_output_digest is None
        or audit.reused
    ):
        raise ValueError(f"artifact {artifact_id} inference audit identity is invalid")
    return result, replace(audit, reused=True)


class SlidingReviewer:
    """Execute every raw window and one final merge for a positioned reviewer."""

    def __init__(
        self,
        *,
        session: SessionView,
        judge: PositionedJudge,
        judging_plan: Mapping[str, object],
        context_policy: JudgingContextPolicy,
        client: ChatClient,
        load_artifact: ArtifactLoader,
        record_artifact: ArtifactRecorder,
        is_cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        if not isinstance(session, SessionView):
            raise TypeError("session must be a SessionView")
        if not isinstance(judge, PositionedJudge):
            raise TypeError("judge must be a PositionedJudge")
        if not isinstance(context_policy, JudgingContextPolicy):
            raise TypeError("context_policy must be a JudgingContextPolicy")
        plan = dict(judging_plan)
        plan_id = plan.get("plan_id")
        body = {key: value for key, value in plan.items() if key != "plan_id"}
        if plan_id != _sha256(body):
            raise ValueError("pinned judging plan ID does not match its full content")
        if plan.get("schema_version") != "2":
            raise ValueError("pinned judging plan schema is unsupported")
        if plan.get("input_policy") != context_policy.model_dump(mode="json"):
            raise ValueError("pinned judging context policy does not match")
        if plan.get("protocol") != sliding_protocol_contract_manifest():
            raise ValueError("pinned sliding protocol does not match")
        sessions = plan.get("sessions")
        if not isinstance(sessions, list):
            raise ValueError("pinned judging plan sessions are invalid")
        matching_sessions = [
            value
            for value in sessions
            if isinstance(value, Mapping)
            and value.get("conversation_id") == session.conversation_id
        ]
        if len(matching_sessions) != 1:
            raise ValueError("session is not an exact member of the pinned judging plan")
        reviewer_rows = matching_sessions[0].get("reviewers")
        if not isinstance(reviewer_rows, list):
            raise ValueError("pinned reviewer manifest is invalid")
        if any(not isinstance(value, Mapping) for value in reviewer_rows) or [
            value.get("ordinal") for value in reviewer_rows
        ] != list(range(1, len(reviewer_rows) + 1)):
            raise ValueError("pinned reviewer ordinals are invalid")
        if judge.position > len(reviewer_rows):
            raise ValueError("reviewer ordinal is absent from the pinned judging plan")
        reviewer_row = reviewer_rows[judge.position - 1]
        if reviewer_row.get("judge") != judge.model_dump(mode="json"):
            raise ValueError("reviewer is not the exact pinned plan member at this ordinal")
        window_plan = reviewer_row.get("window_plan")
        if not isinstance(window_plan, Mapping):
            raise ValueError("pinned reviewer window plan is invalid")
        expected_plan = build_window_plan(session, context_policy, judge.max_input_tokens)
        if dict(window_plan) != expected_plan:
            raise ValueError(
                "pinned window plan does not match current session evidence and policy"
            )

        self.session = session
        self.judge = judge
        self.window_plan = expected_plan
        self.context_policy = context_policy
        self.client = client
        self._load_artifact = load_artifact
        self._record_artifact = record_artifact
        self._is_cancelled = is_cancelled
        self._digest_lock = threading.Lock()
        self._digests: tuple[ChunkDigest, ...] | None = None
        self._digest_steps: tuple[InferenceStepAudit, ...] | None = None
        self._reviewer_key = hashlib.sha256(
            _canonical_json(judge.model_dump(mode="json")).encode()
        ).hexdigest()
        self._protocol_digest = sliding_protocol_contract_digest().removeprefix("sha256:")
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
                "plan_id": self.window_plan["plan_id"],
                "index": window["index"],
                "core_trace_ids": window["core_trace_ids"],
            }
        )

    def _artifact_id(
        self,
        phase: Literal["digest", "window", "merge"],
        identity: str,
    ) -> str:
        plan_hash = str(self.window_plan["plan_id"]).removeprefix("sha256:")
        identity_hash = hashlib.sha256(identity.encode()).hexdigest()
        return f"{phase}/{plan_hash}/{self._reviewer_key}/{self._protocol_digest}/{identity_hash}"

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

    def _messages_fit(self, messages: list[dict[str, str]], max_tokens: int) -> None:
        input_tokens = estimate_tokens(_canonical_json(messages))
        input_cap = min(
            self.context_policy.target_input_tokens,
            self.judge.max_input_tokens,
        )
        if input_tokens + max_tokens + self.context_policy.safety_reserve_tokens > input_cap:
            raise ValueError("rendered inference request exceeds the configured context budget")

    def _infer(
        self,
        *,
        phase: Literal["digest", "window", "merge"],
        artifact_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        schema: JsonSchemaSpec,
        steps: list[InferenceStepAudit],
        usage: dict[str, int],
    ) -> Mapping[str, Any]:
        self._messages_fit(messages, max_tokens)
        self._check_cancelled()
        try:
            parsed, response = self.client.chat_json(
                model=self.judge.id,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                response_schema=schema,
            )
        except InferenceCancelled:
            raise
        except Exception as error:
            request_count = getattr(error, "_transport_request_count", 1)
            steps.append(
                InferenceStepAudit(
                    phase=phase,
                    artifact_id=artifact_id,
                    requested_model=self.judge.id,
                    resolved_model=None,
                    usage={},
                    output_mode=None,
                    schema_name=schema.name,
                    schema_fallback_reason=None,
                    transport_request_count=(
                        request_count if type(request_count) is int and request_count > 0 else 1
                    ),
                    raw_output_digest=None,
                )
            )
            raise _JudgeInvocationFailure(type(error).__name__) from None

        normalized_usage = _normalized_usage(response.usage)
        _add_usage(usage, normalized_usage)
        steps.append(
            InferenceStepAudit(
                phase=phase,
                artifact_id=artifact_id,
                requested_model=self.judge.id,
                resolved_model=response.model,
                usage=normalized_usage,
                output_mode=response.output_mode,
                schema_name=response.schema_name or schema.name,
                schema_fallback_reason=response.schema_fallback_reason,
                transport_request_count=response.transport_request_count,
                raw_output_digest=response.raw_output_digest,
            )
        )
        return parsed

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
        artifact_id = self._artifact_id("digest", chunk_id)
        raw_text, evidence_ids = self._core_evidence(window)
        artifact = _review_callback(self._load_artifact, artifact_id)
        if artifact is not None:
            payload, audit = _artifact_payload(
                artifact,
                artifact_id=artifact_id,
                expected_kind="chunk_digest",
                expected_phase="digest",
                expected_schema=CHUNK_DIGEST_SCHEMA,
                requested_model=self.judge.id,
            )
            steps.append(audit)
        else:
            payload = self._infer(
                phase="digest",
                artifact_id=artifact_id,
                messages=self._digest_messages(
                    chunk_id=chunk_id,
                    raw_text=raw_text,
                    evidence_ids=evidence_ids,
                ),
                max_tokens=self.context_policy.digest_max_tokens,
                schema=CHUNK_DIGEST_SCHEMA,
                steps=steps,
                usage=usage,
            )
        digest = parse_chunk_digest(
            payload,
            allowed_evidence_ids=evidence_ids,
            max_tokens=self.context_policy.digest_max_tokens,
            expected_chunk_id=chunk_id,
        )
        if artifact is None:
            _review_callback(
                self._record_artifact,
                artifact_id,
                _artifact_envelope(
                    "chunk_digest",
                    digest.model_dump(mode="json"),
                    steps[-1],
                ),
            )
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
        raw = render_raw_window(self.session, window)
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
        artifact_id = self._artifact_id(
            "window",
            f"{descriptor.id}:{descriptor.content_digest}:{window['window_id']}",
        )
        messages, allowed_evidence_ids = self._window_messages(
            rubric=rubric,
            window_index=window_index,
            window=window,
            digests=digests,
        )
        artifact = _review_callback(self._load_artifact, artifact_id)
        if artifact is not None:
            payload, audit = _artifact_payload(
                artifact,
                artifact_id=artifact_id,
                expected_kind="window_findings",
                expected_phase="window",
                expected_schema=WINDOW_FINDINGS_SCHEMA,
                requested_model=self.judge.id,
            )
            steps.append(audit)
        else:
            payload = self._infer(
                phase="window",
                artifact_id=artifact_id,
                messages=messages,
                max_tokens=self.context_policy.finding_max_tokens,
                schema=WINDOW_FINDINGS_SCHEMA,
                steps=steps,
                usage=usage,
            )
        findings = parse_window_findings(
            payload,
            allowed_evidence_ids=allowed_evidence_ids,
            max_tokens=self.context_policy.finding_max_tokens,
            expected_window_id=str(window["window_id"]),
        )
        if artifact is None:
            _review_callback(
                self._record_artifact,
                artifact_id,
                _artifact_envelope(
                    "window_findings",
                    findings.model_dump(mode="json"),
                    steps[-1],
                ),
            )
        return findings

    def _deduplicated_findings(
        self,
        values: Sequence[WindowFindings],
    ) -> tuple[WindowFinding, ...]:
        ordered: list[WindowFinding] = []
        identities: dict[str, tuple[object, ...]] = {}
        semantic_seen: set[tuple[object, ...]] = set()
        for window in values:
            for finding in window.findings:
                semantic = (
                    finding.polarity,
                    finding.observation,
                    tuple(sorted(finding.evidence_ids)),
                )
                prior = identities.get(finding.finding_id)
                if prior is not None and prior != semantic:
                    raise ValueError("finding ID maps to conflicting content across windows")
                identities[finding.finding_id] = semantic
                if semantic in semantic_seen:
                    continue
                semantic_seen.add(semantic)
                ordered.append(finding)
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
            "plan_id": self.window_plan["plan_id"],
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
        artifact_id = self._artifact_id(
            "merge",
            f"{descriptor.id}:{descriptor.content_digest}",
        )
        artifact = _review_callback(self._load_artifact, artifact_id)
        if artifact is not None:
            payload, audit = _artifact_payload(
                artifact,
                artifact_id=artifact_id,
                expected_kind="merged_verdict",
                expected_phase="merge",
                expected_schema=MERGED_VERDICT_SCHEMA,
                requested_model=self.judge.id,
            )
            steps.append(audit)
        else:
            payload = self._infer(
                phase="merge",
                artifact_id=artifact_id,
                messages=self._merge_messages(rubric=rubric, digests=digests, findings=findings),
                max_tokens=self.context_policy.output_reserve_tokens,
                schema=MERGED_VERDICT_SCHEMA,
                steps=steps,
                usage=usage,
            )
        verdict = parse_merged_verdict(payload, allowed_evidence_ids=self._all_evidence_ids)
        if artifact is None:
            _review_callback(
                self._record_artifact,
                artifact_id,
                _artifact_envelope(
                    "merged_verdict",
                    verdict.model_dump(mode="json"),
                    steps[-1],
                ),
            )
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
        except Exception as error:
            last_step = steps[-1] if steps else None
            return AttemptObservation(
                status="failed",
                resolved_model=last_step.resolved_model if last_step is not None else None,
                score=None,
                rationale=None,
                usage=usage,
                error_type=(
                    error.error_type
                    if isinstance(error, (_JudgeInvocationFailure, _ReviewInfrastructureFailure))
                    else type(error).__name__
                ),
                message=_bounded_error(error),
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
