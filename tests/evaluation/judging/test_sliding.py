from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges import inference, sliding
from weave_agent_signals.judges.inference import (
    InferenceCancelled,
    InferenceContextExceeded,
    InferenceResponseDiagnostic,
    JsonSchemaSpec,
    JudgeResponse,
)
from weave_agent_signals.judges.plan import build_canonical_judging_plan
from weave_agent_signals.judges.records import JudgeCallRecord
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.judges.sliding import (
    SlidingReviewer,
    sliding_protocol_contract_digest,
    sliding_protocol_contract_manifest,
)
from weave_agent_signals.judges.sliding_contracts import WindowFinding, WindowFindings
from weave_agent_signals.judges.windowing import (
    build_window_plan,
    render_raw_turn,
    render_raw_window,
)
from weave_agent_signals.models import SessionView, TurnSpan
from weave_agent_signals.run_config import JudgingContextPolicy, PositionedJudge, RubricDescriptor
from weave_agent_signals.runs.events import normalize_judging_details


def _turn(trace_id: str, position: int, size: int = 9_000) -> TurnSpan:
    started = datetime(2026, 7, 15, 12, tzinfo=timezone.utc) + timedelta(minutes=position)
    turn = TurnSpan(
        trace_id=trace_id,
        conversation_id="session-1",
        started_at=started,
        ended_at=started + timedelta(seconds=30),
        model="hidden-generating-model",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        status_code="OK",
        config_version="config-1",
        git_branch="main",
        effort_level=None,
        session_id="session-id",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=[],
        chat_spans=[],
        subagents=[],
        user_input="",
        assistant_output=f"assistant-{trace_id}",
    )
    overhead = len(render_raw_turn(turn, position).encode())
    return replace(turn, user_input="x" * (size - overhead))


def _policy(**updates: int) -> JudgingContextPolicy:
    values = {
        "large_model_reserve_tokens": 100_000,
        "small_model_reserve_tokens": 50_000,
        "prompt_reserve_tokens": 3_000,
        "output_reserve_tokens": 3_000,
        "safety_reserve_tokens": 3_000,
        "digest_max_tokens": 2_000,
        "finding_max_tokens": 1_000,
        "max_chunks": 10,
    }
    values.update(updates)
    return JudgingContextPolicy(**values)


def _judge(**updates: object) -> PositionedJudge:
    values = {
        "id": "judge-1",
        "label": "Judge One",
        "provider": "openai",
        "provider_model": "judge-1",
        "family": "family-1",
        "supported_roles": ("judge",),
        "max_input_tokens": 60_000,
        "position": 1,
    }
    values.update(updates)
    return PositionedJudge(
        **values,  # type: ignore[arg-type]
    )


def _rubric(rubric_id: str) -> RubricDescriptor:
    return build_rubric_catalog().rubric(rubric_id)


def _phase(messages: list[dict[str, str]]) -> str:
    marker = "PHASE:"
    return messages[0]["content"].split(marker, 1)[1].splitlines()[0].strip()


def _marker(messages: list[dict[str, str]], name: str) -> str:
    marker = f"{name}:"
    content = next(
        message["content"] for message in reversed(messages) if marker in message["content"]
    )
    return content.split(marker, 1)[1].splitlines()[0].strip()


def _json_marker(messages: list[dict[str, str]], name: str) -> object:
    return json.loads(_marker(messages, name))


class _ScriptedClient:
    backend = "openai"

    def __init__(
        self,
        *,
        fail_merge: bool = False,
        invalid_window_citation: bool = False,
        invalid_window_citation_once: bool = False,
        large_digest: bool = False,
        invocation_error: Exception | None = None,
        invocation_error_phase: str | None = None,
        schema_fallback_reason: str | None = None,
        transport_request_count: int = 1,
    ):
        self.calls: list[dict[str, Any]] = []
        self.fail_merge = fail_merge
        self.invalid_window_citation = invalid_window_citation
        self.invalid_window_citation_once = invalid_window_citation_once
        self.large_digest = large_digest
        self.invocation_error = invocation_error
        self.invocation_error_phase = invocation_error_phase
        self.schema_fallback_reason = schema_fallback_reason
        self.transport_request_count = transport_request_count
        self._lock = threading.Lock()
        self._invalid_window_citation_returned = False

    def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: object | None = None,
        reasoning: str = "default",
    ) -> tuple[dict[str, Any], JudgeResponse]:
        phase = _phase(messages)
        with self._lock:
            call_index = len(self.calls) + 1
            self.calls.append(
                {
                    "phase": phase,
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "schema_name": getattr(response_schema, "name", None),
                    "schema": getattr(response_schema, "schema", None),
                    "reasoning": reasoning,
                }
            )
        if self.invocation_error is not None and (
            self.invocation_error_phase is None or self.invocation_error_phase == phase
        ):
            raise self.invocation_error
        if phase == "digest":
            payload = {
                "text": (
                    "x" * 7_000
                    if self.large_digest
                    else "The agent handled the cited turn evidence."
                ),
            }
        elif phase == "window":
            invalid_citation = self.invalid_window_citation or (
                self.invalid_window_citation_once and not self._invalid_window_citation_returned
            )
            self._invalid_window_citation_returned = (
                self._invalid_window_citation_returned or invalid_citation
            )
            evidence_id = (
                "unknown-evidence" if invalid_citation else _marker(messages, "EXAMPLE_EVIDENCE_ID")
            )
            payload = {
                "findings": [
                    {
                        "finding_id": "shared-finding",
                        "polarity": "positive",
                        "observation": "The agent used relevant checks.",
                        "evidence_ids": [evidence_id],
                        "quote": None,
                    }
                ],
            }
        else:
            payload = (
                {"not": "a verdict"}
                if self.fail_merge
                else {
                    "status": "scored",
                    "score": 0.75,
                    "rationale": "The session was mostly effective.",
                    "finding_ids": _json_marker(messages, "ALLOWED_FINDING_IDS")[:1],
                    "feedback": {
                        "success": "The agent used relevant checks.",
                        "problem": "One check was not repeated after the final change.",
                        "desired_behavior": "Repeat relevant checks after the final change.",
                    },
                }
            )
        raw = json.dumps(payload, sort_keys=True)
        return payload, JudgeResponse(
            content=raw,
            model="resolved-judge-1",
            usage={"input_tokens": call_index, "total_tokens": call_index + 1},
            output_mode=(
                "json_object_fallback" if self.schema_fallback_reason is not None else "json_schema"
            ),
            schema_name=getattr(response_schema, "name", None),
            schema_fallback_reason=self.schema_fallback_reason,
            transport_request_count=self.transport_request_count,
            raw_output_digest=hashlib.sha256(raw.encode()).hexdigest(),
        )


class _TransportActivityClient(_ScriptedClient):
    def __init__(self):
        super().__init__()
        self.activity = None

    def set_activity(self, callback):
        self.activity = callback

    def chat_json(self, **kwargs):
        assert self.activity is not None
        self.activity(
            {
                "phase": "transport_retry",
                "message": "retrying provider request",
                "request_attempt": 2,
                "max_attempts": 2,
                "retry_reason": "transport_error",
            }
        )
        return super().chat_json(**kwargs)


def _reviewer(
    *,
    client: _ScriptedClient | None = None,
    calls: dict[str, JudgeCallRecord] | None = None,
    policy: JudgingContextPolicy | None = None,
    cancelled=lambda: False,
    judge: PositionedJudge | None = None,
    activity=None,
    call_gate: threading.Semaphore | None = None,
    expected_windows: int = 2,
) -> tuple[SlidingReviewer, _ScriptedClient, dict[str, JudgeCallRecord], dict[str, object]]:
    session = SessionView(
        "session-1",
        [_turn(f"trace-{index}", index) for index in range(1, 5)],
        "config-1",
        "main",
    )
    active_policy = policy or _policy()
    active_judge = judge or _judge()
    plan = build_window_plan(
        session,
        active_policy,
        active_judge.max_input_tokens,
        active_judge.token_counter,
    )
    judging_plan = build_canonical_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=build_rubric_catalog().rubrics,
        judge_models=(active_judge,),
        context_policy=active_policy,
    )
    assert len(plan["windows"]) == expected_windows
    stored = {} if calls is None else calls
    scripted = client or _ScriptedClient()

    def record(call: JudgeCallRecord) -> None:
        existing = stored.setdefault(call.request_id, call)
        if existing != call:
            raise ValueError("judge call conflict")

    return (
        SlidingReviewer(
            session=session,
            judge=active_judge,
            plan_id=judging_plan.plan_id,
            window_plan=judging_plan.sessions[0].reviewers[0].window_plan.model_dump(mode="json"),
            context_policy=active_policy,
            client=scripted,
            load_call=stored.get,
            record_call=record,
            is_cancelled=cancelled,
            activity=activity,
            call_gate=call_gate,
        ),
        scripted,
        stored,
        plan,
    )


def test_live_provider_calls_wait_for_the_call_gate() -> None:
    class ObservedSemaphore(threading.Semaphore):
        def __init__(self) -> None:
            super().__init__(0)
            self.entered = threading.Event()

        def acquire(self, *args, **kwargs):
            self.entered.set()
            return super().acquire(*args, **kwargs)

    gate = ObservedSemaphore()
    reviewer, client, _, _ = _reviewer(call_gate=gate)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(reviewer.review, _rubric("judge.session_outcome"))
        assert gate.entered.wait(timeout=1)
        assert client.calls == []
        gate.release()
        result = future.result(timeout=2)

    assert result.status == "succeeded"


def test_cancellation_interrupts_waiting_for_the_call_gate() -> None:
    class ObservedSemaphore(threading.Semaphore):
        def __init__(self) -> None:
            super().__init__(0)
            self.entered = threading.Event()

        def acquire(self, *args, **kwargs):
            self.entered.set()
            return super().acquire(*args, **kwargs)

    cancelled = threading.Event()
    gate = ObservedSemaphore()
    reviewer, client, _, _ = _reviewer(call_gate=gate, cancelled=cancelled.is_set)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(reviewer.review, _rubric("judge.session_outcome"))
        assert gate.entered.wait(timeout=1)
        cancelled.set()
        try:
            with pytest.raises(InferenceCancelled):
                future.result(timeout=1)
        finally:
            gate.release()

    assert client.calls == []


def test_cancellation_after_acquiring_call_gate_releases_it() -> None:
    class CancellingSemaphore(threading.Semaphore):
        def __init__(self, cancelled: threading.Event) -> None:
            super().__init__(1)
            self.cancelled = cancelled

        def acquire(self, *args, **kwargs):
            acquired = super().acquire(*args, **kwargs)
            if acquired:
                self.cancelled.set()
            return acquired

    cancelled = threading.Event()
    gate = CancellingSemaphore(cancelled)
    reviewer, client, _, _ = _reviewer(call_gate=gate, cancelled=cancelled.is_set)

    with pytest.raises(InferenceCancelled):
        reviewer.review(_rubric("judge.session_outcome"))

    assert client.calls == []
    assert gate.acquire(blocking=False)


def test_cached_digests_wait_for_digest_publication_lock() -> None:
    reviewer, _, _, _ = _reviewer()
    reviewer._digests = ()
    reviewer._digest_steps = ()

    reviewer._digest_lock.acquire()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(reviewer._ensure_digests, steps=[], usage={})
            with pytest.raises(FutureTimeoutError):
                future.result(timeout=0.05)
            reviewer._digest_lock.release()
            assert future.result(timeout=1) == ()
    finally:
        if reviewer._digest_lock.locked():
            reviewer._digest_lock.release()


def test_reviewer_digests_once_then_reads_every_window_and_merges() -> None:
    reviewer, client, _, _ = _reviewer()

    first = reviewer.review(_rubric("judge.session_outcome"))
    second = reviewer.review(_rubric("judge.session_autonomy"))

    assert [call["phase"] for call in client.calls] == [
        "digest",
        "digest",
        "window",
        "window",
        "merge",
        "window",
        "window",
        "merge",
    ]
    assert first.status == "succeeded"
    assert second.status == "succeeded"
    assert first.evidence_ids == ("trace-1",)
    assert first.behavioral_feedback is not None
    assert first.usage == {"input_tokens": 15, "total_tokens": 20}
    assert len(first.steps) == 5
    assert not any(step.reused for step in first.steps)
    assert second.usage == {"input_tokens": 21, "total_tokens": 24}
    assert second.transport_request_count == 3
    assert [step.reused for step in second.steps] == [True, True, False, False, False]
    assert all(step.requested_model == "judge-1" for step in first.steps)
    assert all(call["temperature"] == 0.2 for call in client.calls)
    assert all(
        call["max_tokens"]
        == reviewer.context_policy.generation_budget(reviewer.judge.max_input_tokens)
        for call in client.calls
    )
    assert [call["schema_name"] for call in client.calls[:5]] == [
        "chunk_digest",
        "chunk_digest",
        "window_findings",
        "window_findings",
        "merged_verdict",
    ]


def test_large_reviewer_uses_10k_generation_budget_for_every_phase() -> None:
    reviewer, client, _, _ = _reviewer(
        policy=_policy(large_model_raw_target_tokens=20_000),
        judge=_judge(max_input_tokens=262_000),
        expected_windows=1,
    )

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "succeeded"
    assert {call["phase"] for call in client.calls} == {"digest", "window", "merge"}
    assert all(call["max_tokens"] == 10_000 for call in client.calls)


def test_wandb_reviewer_disables_reasoning_for_digest_and_merge() -> None:
    reviewer, client, _, _ = _reviewer(
        judge=_judge(provider="wandb"),
    )

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "succeeded"
    assert [call["reasoning"] for call in client.calls] == [
        "disabled",
        "disabled",
        "default",
        "default",
        "disabled",
    ]


def test_inference_failure_emits_contextual_provider_failure_activity() -> None:
    error = RuntimeError("private provider detail")
    error._transport_request_count = 2  # type: ignore[attr-defined]
    activity: list[dict[str, object]] = []
    reviewer, _, _, _ = _reviewer(
        client=_ScriptedClient(invocation_error=error),
        activity=activity.append,
    )

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    failed = [event for event in activity if event["phase"] == "provider_failed"]
    assert len(failed) == 1
    event = failed[0]
    assert event["request_attempt"] == 2
    assert event["max_attempts"] == 2
    assert event["retry_reason"] == "digest"
    assert event["error_category"] == "RuntimeError"
    assert event["item_index"] == 1
    assert event["item_total"] == 2
    assert "private provider detail" not in str(event)
    normalize_judging_details(
        {key: value for key, value in event.items() if key not in {"phase", "message"}}
    )


def test_transport_activity_includes_active_inference_context() -> None:
    activity: list[dict[str, object]] = []
    reviewer, _, _, _ = _reviewer(
        client=_TransportActivityClient(),
        activity=activity.append,
    )

    reviewer.review(_rubric("judge.session_outcome"))

    retry = next(event for event in activity if event["phase"] == "transport_retry")
    assert retry["conversation_id"] == "session-1"
    assert retry["model"] == "judge-1"
    assert retry["artifact_id"]
    assert retry["item_index"] == 1
    assert retry["item_total"] == 2
    assert retry["estimated_input_tokens"] > 0


def test_reviewer_binds_each_inference_schema_to_its_exact_evidence_scope() -> None:
    reviewer, client, _, plan = _reviewer()

    reviewer.review(_rubric("judge.session_outcome"))

    digest_calls = [call for call in client.calls if call["phase"] == "digest"]
    for call, window in zip(digest_calls, plan["windows"], strict=True):
        properties = call["schema"]["properties"]
        assert "chunk_id" not in properties
        assert "schema_version" not in properties
        assert "evidence_ids" not in properties

    window_calls = [call for call in client.calls if call["phase"] == "window"]
    for call, window in zip(window_calls, plan["windows"], strict=True):
        assert (
            "Reason carefully internally, then return only concise JSON"
            in call["messages"][0]["content"]
        )
        assert "otherwise set quote to null" in call["messages"][0]["content"]
        assert "window_id" not in call["schema"]["properties"]
        assert "schema_version" not in call["schema"]["properties"]
        evidence = call["schema"]["$defs"]["WindowFinding"]["properties"]["evidence_ids"]
        assert evidence["items"]["enum"] == list(
            reviewer._window_messages(
                rubric=SESSION_RUBRICS["judge.session_outcome"],
                window_index=int(window["index"]) - 1,
                window=window,
                digests=reviewer._digests or (),
            )[1]
        )

    merge_call = next(call for call in client.calls if call["phase"] == "merge")
    assert "evidence_ids" not in merge_call["schema"]["properties"]
    assert merge_call["schema"]["properties"]["finding_ids"]["items"]["enum"]
    assert (
        "Behavioral feedback describes what the agent did or should do; reflection separately "
        "decides whether and how to edit managed instructions."
        in merge_call["messages"][0]["content"]
    )


def test_window_prompt_exposes_only_active_raw_evidence_ids() -> None:
    reviewer, _, _, plan = _reviewer()

    reviewer.review(_rubric("judge.session_outcome"))

    digests = reviewer._digests or ()
    surrounding_evidence = render_raw_window(
        reviewer.session,
        plan["windows"][0],
        reviewer.judge.token_counter,
    ).evidence_ids
    cited_surrounding_id = surrounding_evidence[0]
    uncited_surrounding_id = next(
        evidence_id for evidence_id in surrounding_evidence if evidence_id != cited_surrounding_id
    )
    reviewer._all_evidence_ids = (*reviewer._all_evidence_ids, "prefix-1", "prefix-10")
    digest_sentinel = "surrounding digest content remains visible"
    tainted_digests = (
        digests[0].model_copy(
            update={
                "text": (
                    f"{digest_sentinel}: {cited_surrounding_id} {uncited_surrounding_id}; "
                    "overlap prefix-10 prefix-1"
                )
            }
        ),
        digests[1],
    )
    messages, active_aliases, _raw_text = reviewer._window_messages(
        rubric=SESSION_RUBRICS["judge.session_outcome"],
        window_index=1,
        window=plan["windows"][1],
        digests=tainted_digests,
    )
    prompt = messages[1]["content"]
    surrounding_ids = {cited_surrounding_id, uncited_surrounding_id}

    assert all(alias in prompt for alias in active_aliases)
    assert all(
        f"[evidence_id={evidence_id}]" not in prompt and f"trace_id: {evidence_id}\n" not in prompt
        for evidence_id in active_aliases.values()
    )
    assert digest_sentinel in prompt
    assert "overlap [citation omitted] [citation omitted]" in prompt
    assert surrounding_ids
    assert all(evidence_id not in prompt for evidence_id in surrounding_ids)


def test_evidence_id_redaction_is_longest_first() -> None:
    assert (
        sliding._redact_evidence_ids(
            "overlap prefix-10 prefix-1",
            ("prefix-1", "prefix-10"),
        )
        == "overlap [citation omitted] [citation omitted]"
    )


def test_evidence_aliasing_changes_labels_without_mutating_captured_content() -> None:
    rendered, aliases = sliding._alias_evidence(
        "## Turn 1 [evidence_id=trace-1]\n"
        "trace_id: trace-1\n"
        "user_input: preserve incidental trace-1 text",
        ("trace-1",),
    )

    assert aliases == {"e1": "trace-1"}
    assert "[evidence_id=e1]" in rendered
    assert "trace_id: e1\n" in rendered
    assert "preserve incidental trace-1 text" in rendered


def test_reviewer_prompts_distinguish_cited_findings_from_no_findings() -> None:
    reviewer, client, _, _ = _reviewer()

    reviewer.review(_rubric("judge.session_outcome"))

    digest_call = next(call for call in client.calls if call["phase"] == "digest")
    window_call = next(call for call in client.calls if call["phase"] == "window")
    merge_call = next(call for call in client.calls if call["phase"] == "merge")

    assert (
        "The host records which chunk this digest summarizes."
        in digest_call["messages"][0]["content"]
    )
    assert (
        "Every finding must cite at least one ID from ALLOWED_FINDING_EVIDENCE_IDS."
        in window_call["messages"][0]["content"]
    )
    assert (
        'If no supported finding exists, return "findings": [] instead of an uncited finding.'
        in window_call["messages"][0]["content"]
    )
    assert (
        "A scored verdict must cite at least one ID from ALLOWED_FINDING_IDS; an "
        "insufficient_evidence verdict must cite none." in merge_call["messages"][0]["content"]
    )


def test_reviewer_narrates_each_live_inference_phase_with_context() -> None:
    activity = []
    reviewer, _, _, _ = _reviewer(activity=activity.append)

    reviewer.review(_rubric("judge.session_outcome"))

    assert [event["phase"] for event in activity] == [
        "digest_started",
        "digest_started",
        "window_started",
        "window_started",
        "merge_started",
    ]
    assert activity[0] == {
        "phase": "digest_started",
        "message": "Judge One is digesting chunk 1 of 2",
        "model": "judge-1",
        "conversation_id": "session-1",
        "artifact_id": activity[0]["artifact_id"],
        "item_index": 1,
        "item_total": 2,
        "estimated_input_tokens": activity[0]["estimated_input_tokens"],
        "max_output_tokens": 3_000,
        "model_context_tokens": 60_000,
    }
    assert activity[0]["estimated_input_tokens"] > 0
    assert activity[2]["rubric"] == "judge.session_outcome"
    assert activity[2]["item_index"] == 1
    assert activity[2]["item_total"] == 2
    assert activity[-1]["message"] == "Judge One is merging Session Outcome Quality"
    assert activity[-1]["rubric"] == "judge.session_outcome"


def test_cross_window_finding_identity_ignores_duplicate_citation_multiplicity() -> None:
    reviewer, _, _, _ = _reviewer()
    first_finding = WindowFinding(
        finding_id="shared-finding",
        polarity="positive",
        observation="The agent used relevant checks.",
        evidence_ids=("trace-1",),
        quote=None,
    )
    repeated_citation = first_finding.model_copy(update={"evidence_ids": ("trace-1", "trace-1")})

    findings = reviewer._deduplicated_findings(
        (
            WindowFindings(schema_version=1, window_id="window-1", findings=(first_finding,)),
            WindowFindings(
                schema_version=1,
                window_id="window-2",
                findings=(repeated_citation,),
            ),
        )
    )

    assert findings == (first_finding.model_copy(update={"finding_id": "window-1:shared-finding"}),)


def test_cross_window_finding_ids_are_scoped_to_their_window() -> None:
    reviewer, _, _, _ = _reviewer()

    findings = reviewer._deduplicated_findings(
        (
            WindowFindings(
                schema_version=1,
                window_id="window-1",
                findings=(
                    WindowFinding(
                        finding_id="tool-match-positive",
                        polarity="positive",
                        observation="The agent chose appropriate repository tools.",
                        evidence_ids=("trace-1",),
                        quote=None,
                    ),
                ),
            ),
            WindowFindings(
                schema_version=1,
                window_id="window-2",
                findings=(
                    WindowFinding(
                        finding_id="tool-match-positive",
                        polarity="positive",
                        observation="The agent chose appropriate verification tools.",
                        evidence_ids=("trace-2",),
                        quote=None,
                    ),
                ),
            ),
        )
    )

    assert [finding.finding_id for finding in findings] == [
        "window-1:tool-match-positive",
        "window-2:tool-match-positive",
    ]


def test_window_replaces_own_digest_and_keeps_surrounding_digests_chronological() -> None:
    reviewer, client, _, plan = _reviewer()

    reviewer.review(_rubric("judge.session_outcome"))

    windows = [call for call in client.calls if call["phase"] == "window"]
    for index, call in enumerate(windows):
        text = call["messages"][-1]["content"]
        assert text.count("CONTEXT_KIND: raw_window") == 1
        assert text.count("CONTEXT_KIND: chunk_digest") == 1
        assert text.index("CHUNK_INDEX: 1") < text.index("CHUNK_INDEX: 2")
        assert "EXPECTED_WINDOW_ID:" not in text

    merge_text = [call for call in client.calls if call["phase"] == "merge"][0]["messages"][-1][
        "content"
    ]
    assert merge_text.count('"finding_id":') == 2
    assert '"raw_coverage_trace_ids":["trace-1","trace-2","trace-3","trace-4"]' in merge_text


def test_exact_call_replay_makes_zero_model_calls() -> None:
    first, _, calls, _ = _reviewer()
    first_result = first.review(_rubric("judge.session_outcome"))
    replay_client = _ScriptedClient()
    replay, _, _, _ = _reviewer(client=replay_client, calls=calls)

    replay_result = replay.review(_rubric("judge.session_outcome"))

    assert replay_client.calls == []
    assert replay_result.score == first_result.score
    assert replay_result.resolved_model == first_result.resolved_model
    assert replay_result.usage == {}
    assert all(step.reused for step in replay_result.steps)
    assert [replace(step, reused=False) for step in replay_result.steps] == list(first_result.steps)
    assert replay_result.transport_request_count == 0
    assert replay_result.output_mode == first_result.output_mode
    assert replay_result.raw_output_digest == first_result.raw_output_digest


def test_replay_provenance_matches_a_rubric_that_reused_cached_digests() -> None:
    first, _, calls, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    original = first.review(_rubric("judge.session_autonomy"))
    replay_client = _ScriptedClient()
    replay, _, _, _ = _reviewer(client=replay_client, calls=calls)

    restored = replay.review(_rubric("judge.session_autonomy"))

    assert replay_client.calls == []
    assert all(step.reused for step in restored.steps)
    assert [replace(step, reused=False) for step in restored.steps] == [
        replace(step, reused=False) for step in original.steps
    ]
    assert restored.usage == {}
    assert restored.transport_request_count == 0


def test_partial_replay_charges_only_new_window_and_merge_calls() -> None:
    first, _, calls, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    digest_calls = {
        request_id: call for request_id, call in calls.items() if call.phase == "digest"
    }
    partial_client = _ScriptedClient()
    partial, _, _, _ = _reviewer(client=partial_client, calls=digest_calls)

    result = partial.review(_rubric("judge.session_autonomy"))

    assert [call["phase"] for call in partial_client.calls] == ["window", "window", "merge"]
    assert [step.reused for step in result.steps] == [True, True, False, False, False]
    assert result.usage == {"input_tokens": 6, "total_tokens": 9}
    assert result.transport_request_count == 3


@pytest.mark.parametrize(
    "judge",
    [
        _judge(id="judge-2"),
        _judge(provider_model="judge-2"),
    ],
)
def test_request_identity_binds_requested_and_provider_models(judge: PositionedJudge) -> None:
    first, _, first_calls, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    second, _, second_calls, _ = _reviewer(judge=judge)

    second.review(_rubric("judge.session_outcome"))

    assert set(first_calls).isdisjoint(second_calls)


def test_protocol_contract_digest_binds_version_prompts_and_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weave_agent_signals.judges.sliding as sliding

    original = sliding_protocol_contract_digest()
    original_template = sliding._DIGEST_SYSTEM_TEMPLATE
    monkeypatch.setattr(sliding, "SLIDING_PROTOCOL_VERSION", "test-version")
    assert sliding_protocol_contract_digest() != original
    monkeypatch.setattr(sliding, "SLIDING_PROTOCOL_VERSION", "3")
    monkeypatch.setattr(
        sliding,
        "_DIGEST_SYSTEM_TEMPLATE",
        original_template + " changed",
    )
    prompt_changed = sliding_protocol_contract_digest()
    assert prompt_changed != original
    monkeypatch.setattr(sliding, "_DIGEST_SYSTEM_TEMPLATE", original_template)
    monkeypatch.setattr(
        sliding,
        "CHUNK_DIGEST_SCHEMA",
        JsonSchemaSpec(
            name="chunk_digest",
            schema={**dict(sliding.CHUNK_DIGEST_SCHEMA.schema), "description": "changed"},
        ),
    )
    assert sliding_protocol_contract_digest() not in {original, prompt_changed}

    monkeypatch.setattr(
        sliding,
        "CHUNK_DIGEST_SCHEMA",
        JsonSchemaSpec(
            name="chunk_digest",
            schema=dict(sliding.CHUNK_DIGEST_SCHEMA.schema),
            examples=({"text": "changed example"},),
        ),
    )
    assert sliding_protocol_contract_digest() not in {original, prompt_changed}


def test_protocol_contract_includes_examples_and_current_version():
    contract = sliding_protocol_contract_manifest()

    assert contract["protocol_version"] == "19"
    assert contract["schemas"]["digest"]["examples"] == [
        {"text": "The agent changed the target and verified the focused check passed."}
    ]


def test_digest_prompt_requests_a_concise_soft_token_target():
    prompt = sliding_protocol_contract_manifest()["prompt_templates"]["digest_system"]

    assert "about 1,000 tokens" in prompt


def test_request_ids_bind_protocol_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import weave_agent_signals.judges.sliding as sliding

    first, _, first_calls, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    monkeypatch.setattr(sliding, "SLIDING_PROTOCOL_VERSION", "future-version")
    second, _, second_calls, _ = _reviewer()

    second.review(_rubric("judge.session_outcome"))

    assert set(first_calls).isdisjoint(second_calls)


def test_concurrent_rubrics_share_one_digest_generation() -> None:
    reviewer, client, _, _ = _reviewer()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                reviewer.review,
                (_rubric("judge.session_outcome"), _rubric("judge.session_autonomy")),
            )
        )

    assert all(result.status == "succeeded" for result in results)
    assert [call["phase"] for call in client.calls].count("digest") == 2
    digest_steps = [step for result in results for step in result.steps if step.phase == "digest"]
    assert sum(not step.reused for step in digest_steps) == 2
    assert sum(step.reused for step in digest_steps) == 2
    assert sorted(result.transport_request_count for result in results) == [3, 5]


def test_cancellation_is_checked_between_model_calls() -> None:
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    reviewer, client, _, _ = _reviewer(cancelled=cancelled)

    with pytest.raises(InferenceCancelled):
        reviewer.review(_rubric("judge.session_outcome"))
    assert [call["phase"] for call in client.calls] == ["digest"]


def test_invalid_window_citation_fails_closed() -> None:
    activity = []
    reviewer, _, _, _ = _reviewer(
        client=_ScriptedClient(invalid_window_citation=True),
        activity=activity.append,
    )

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.evidence_ids == ()
    assert result.behavioral_feedback is None
    assert result.error_type == "ValueError"
    assert result.message == 'unknown evidence IDs: ["unknown-evidence"]'
    assert activity[-1] == {
        "phase": "validation_failed",
        "message": "Judge One returned invalid window output for Session Outcome Quality",
        "model": "judge-1",
        "conversation_id": "session-1",
        "rubric": "judge.session_outcome",
        "artifact_id": result.steps[-1].artifact_id,
        "error_category": "ValueError",
        "provider_error_message": 'unknown evidence IDs: ["unknown-evidence"]',
        "output_mode": "json_schema",
        "output_sha256": result.raw_output_digest,
    }


@pytest.mark.parametrize(
    ("client", "expected_output_mode"),
    [
        (
            _ScriptedClient(
                invalid_window_citation_once=True,
                schema_fallback_reason="retry_recovery",
            ),
            "json_object_fallback",
        ),
        (
            _ScriptedClient(invalid_window_citation_once=True),
            "json_schema",
        ),
    ],
)
def test_contract_failure_gets_one_correction_attempt_regardless_of_output_mode(
    client: _ScriptedClient,
    expected_output_mode: str,
) -> None:
    activity = []
    reviewer, client, calls, _ = _reviewer(
        client=client,
        judge=_judge(provider="wandb"),
        activity=activity.append,
    )

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "succeeded"
    window_calls = [call for call in client.calls if call["phase"] == "window"]
    assert len(window_calls) == 3
    assert "failed validation" in window_calls[1]["messages"][-1]["content"]
    assert "unknown evidence ID" in window_calls[1]["messages"][-1]["content"]
    assert window_calls[1]["reasoning"] == "disabled"
    assert [call.status for call in calls.values()].count("failed") == 1
    assert any(
        event["phase"] == "validation_retry"
        and event["request_attempt"] == 2
        and event["max_attempts"] == 3
        and event["output_mode"] == expected_output_mode
        for event in activity
    )


def test_fallback_contract_failure_stops_after_three_outputs() -> None:
    activity = []
    reviewer, client, _, _ = _reviewer(
        client=_ScriptedClient(
            invalid_window_citation=True,
            schema_fallback_reason="retry_recovery",
        ),
        judge=_judge(provider="wandb"),
        activity=activity.append,
    )

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.message == 'unknown evidence IDs: ["unknown-evidence"]'
    assert [call["phase"] for call in client.calls].count("window") == 3
    assert [
        (event["request_attempt"], event["max_attempts"])
        for event in activity
        if event["phase"] == "validation_retry"
    ] == [(2, 3), (3, 3)]


def test_custom_client_schema_fallback_detail_is_rejected_before_persistence() -> None:
    secret = "SENTINEL_PRIVATE_FALLBACK_DETAIL"
    reviewer, _, artifacts, _ = _reviewer(client=_ScriptedClient(schema_fallback_reason=secret))

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.message == "schema fallback metadata is invalid"
    assert secret not in result.message
    assert result.steps == ()
    assert artifacts == {}


def test_failed_merge_returns_failed_observation() -> None:
    reviewer, _, _, _ = _reviewer(client=_ScriptedClient(fail_merge=True))

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.score is None
    assert result.rationale is None
    assert result.evidence_ids == ()
    assert result.behavioral_feedback is None
    assert result.steps[-1].phase == "merge"
    assert result.usage == {"input_tokens": 28, "total_tokens": 35}
    assert result.transport_request_count == 7
    assert result.output_mode == "json_schema"
    assert result.schema_name == "merged_verdict"
    assert "status" in (result.message or "")


def test_inference_exception_message_is_sanitized_before_observation() -> None:
    error = RuntimeError("codex judge exited 1: sensitive raw CLI output")
    error._transport_request_count = 3  # type: ignore[attr-defined]
    reviewer, _, _, _ = _reviewer(client=_ScriptedClient(invocation_error=error))

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert result.message == "judge invocation failed"
    assert result.transport_request_count == 3
    assert len(result.steps) == 1
    assert result.steps[0].phase == "digest"


def test_safe_provider_message_is_preserved_in_failed_observation() -> None:
    error = RuntimeError("raw provider output")
    error._provider_error_code = "invalid_json_schema"  # type: ignore[attr-defined]
    error._provider_error_message = "Invalid schema: missing required field quote"  # type: ignore[attr-defined]
    reviewer, _, _, _ = _reviewer(client=_ScriptedClient(invocation_error=error))

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.error_type == "invalid_json_schema"
    assert result.message == "Invalid schema: missing required field quote"


def test_output_exhaustion_preserves_combined_failure_audit() -> None:
    diagnostics = (
        InferenceResponseDiagnostic(
            finish_reason="length",
            usage={"completion_tokens": 10_000},
            completion_details={"reasoning_tokens": 9_700},
            content_characters=0,
        ),
        InferenceResponseDiagnostic(
            finish_reason="length",
            usage={"completion_tokens": 10_000},
            completion_details={"reasoning_tokens": 9_500},
            content_characters=0,
        ),
    )
    response = JudgeResponse(
        content="unfinished reasoning",
        model="resolved-judge-1",
        usage={"prompt_tokens": 200, "completion_tokens": 20_000, "total_tokens": 20_200},
        output_mode="json_schema",
        schema_name="chunk_digest",
        transport_request_count=2,
        raw_output_digest="a" * 64,
        response_diagnostics=diagnostics,
    )
    error = inference.InferenceOutputExceeded(response)
    activity: list[dict[str, object]] = []
    reviewer, _, artifacts, _ = _reviewer(
        client=_ScriptedClient(invocation_error=error),
        activity=activity.append,
    )

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.error_type == "InferenceOutputExceeded"
    assert result.message == "provider output limit exhausted"
    assert result.usage == response.usage
    assert result.transport_request_count == 2
    assert result.resolved_model == "resolved-judge-1"
    assert result.output_mode == "json_schema"
    assert result.schema_name == "chunk_digest"
    assert result.raw_output_digest == "a" * 64
    assert result.steps[0].response_diagnostics == diagnostics
    assert len(artifacts) == 1
    stored = next(iter(artifacts.values()))
    assert stored.status == "failed"
    assert stored.audit.usage == response.usage
    assert stored.audit.error_type == "InferenceOutputExceeded"
    assert stored.audit.response_diagnostics == diagnostics
    assert activity[-1]["finish_reason"] == "length"
    assert activity[-1]["completion_tokens"] == 10_000
    assert activity[-1]["reasoning_tokens"] == 9_500
    assert activity[-2]["phase"] == "provider_failed"
    assert activity[-2]["request_attempt"] == 2


def test_exhaustion_retry_failure_preserves_first_response_audit() -> None:
    response = JudgeResponse(
        content="unfinished reasoning",
        model="resolved-judge-1",
        usage={"prompt_tokens": 100, "completion_tokens": 10_000, "total_tokens": 10_100},
        output_mode="json_schema",
        schema_name="chunk_digest",
        transport_request_count=1,
        raw_output_digest="b" * 64,
    )
    error = RuntimeError("retry failed")
    error._transport_request_count = 2  # type: ignore[attr-defined]
    error._inference_response = response  # type: ignore[attr-defined]
    reviewer, _, artifacts, _ = _reviewer(client=_ScriptedClient(invocation_error=error))

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert result.usage == response.usage
    assert result.transport_request_count == 2
    assert result.resolved_model == "resolved-judge-1"
    assert result.raw_output_digest == "b" * 64
    stored = next(iter(artifacts.values()))
    assert stored.audit.usage == response.usage
    assert stored.audit.transport_request_count == 2


def test_provider_context_rejection_skips_reviewer_instead_of_failing() -> None:
    error = InferenceContextExceeded()
    error._transport_request_count = 1
    reviewer, _, _, _ = _reviewer(
        client=_ScriptedClient(
            invocation_error=error,
            invocation_error_phase="window",
        )
    )

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "skipped"
    assert result.skip_reason == "insufficient_context_capacity"
    assert result.message is None
    assert len(result.steps) == 3
    rejected = result.steps[-1]
    assert rejected.phase == "window"
    assert rejected.resolved_model is None
    assert rejected.usage == {}
    assert rejected.transport_request_count == 1
    assert result.usage == {"input_tokens": 3, "total_tokens": 5}
    assert result.transport_request_count == 3


@pytest.mark.parametrize("callback", ["load", "record", "cancel"])
def test_callback_exception_message_is_sanitized_before_observation(callback: str) -> None:
    secret = "SENTINEL_PRIVATE_CALLBACK_DETAIL"
    reviewer, client, calls, _ = _reviewer()

    def fail(*_args: object) -> None:
        raise RuntimeError(secret)

    if callback == "load":
        reviewer._load_call = fail
    elif callback == "record":
        reviewer._record_call = fail
    else:
        reviewer._is_cancelled = fail

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert result.message == "review infrastructure failed"
    assert secret not in result.message
    if callback == "load":
        assert client.calls == []
        assert calls == {}


def test_each_phase_prompt_has_one_output_owner_and_unambiguous_task() -> None:
    reviewer, client, _, _ = _reviewer()
    reviewer.review(_rubric("judge.session_outcome"))
    for call in client.calls:
        system = call["messages"][0]["content"]
        assert system.count("PHASE:") == 1
        assert "schema_version" not in system
        if call["phase"] == "window":
            assert "findings, not a score" in system
        if call["phase"] == "merge":
            assert "session verdict" in system


def test_over_budget_request_skips_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    rubric = SESSION_RUBRICS["judge.session_outcome"]
    monkeypatch.setitem(
        SESSION_RUBRICS,
        rubric.scorer_name,
        replace(rubric, system_prompt=rubric.system_prompt + "y" * 200_000),
    )
    reviewer, client, _, _ = _reviewer()

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "skipped"
    assert result.skip_reason == "insufficient_context_capacity"
    assert result.message is None
    assert [call["phase"] for call in client.calls] == ["digest", "digest"]
