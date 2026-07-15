from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.inference import (
    InferenceCancelled,
    JsonSchemaSpec,
    JudgeResponse,
)
from weave_agent_signals.judges.plan import build_judging_plan
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.judges.sliding import (
    SlidingReviewer,
    sliding_protocol_contract_digest,
)
from weave_agent_signals.judges.windowing import build_window_plan, render_raw_turn
from weave_agent_signals.models import SessionView, TurnSpan
from weave_agent_signals.run_config import JudgingContextPolicy, PositionedJudge, RubricDescriptor


def _turn(trace_id: str, position: int, size: int = 18_000) -> TurnSpan:
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
        "target_input_tokens": 34_000,
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
        "family": "family-1",
        "backend": "openai",
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
    return messages[-1]["content"].split(marker, 1)[1].splitlines()[0].strip()


class _ScriptedClient:
    backend = "openai"

    def __init__(
        self,
        *,
        fail_merge: bool = False,
        invalid_window_citation: bool = False,
        invalid_chunk_id: bool = False,
        invalid_window_id: bool = False,
        large_digest: bool = False,
        invocation_error: Exception | None = None,
        schema_fallback_reason: str | None = None,
    ):
        self.calls: list[dict[str, Any]] = []
        self.fail_merge = fail_merge
        self.invalid_window_citation = invalid_window_citation
        self.invalid_chunk_id = invalid_chunk_id
        self.invalid_window_id = invalid_window_id
        self.large_digest = large_digest
        self.invocation_error = invocation_error
        self.schema_fallback_reason = schema_fallback_reason
        self._lock = threading.Lock()

    def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: object | None = None,
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
                }
            )
        if self.invocation_error is not None:
            raise self.invocation_error
        if phase == "digest":
            payload = {
                "schema_version": 1,
                "chunk_id": (
                    "SENTINEL_PRIVATE_CHUNK_ID"
                    if self.invalid_chunk_id
                    else _marker(messages, "EXPECTED_CHUNK_ID")
                ),
                "text": (
                    "x" * 5_900
                    if self.large_digest
                    else "The agent handled the cited turn evidence."
                ),
                "evidence_ids": [_marker(messages, "EXAMPLE_EVIDENCE_ID")],
            }
        elif phase == "window":
            evidence_id = "unknown-evidence" if self.invalid_window_citation else "trace-2"
            payload = {
                "schema_version": 1,
                "window_id": (
                    "SENTINEL_PRIVATE_WINDOW_ID"
                    if self.invalid_window_id
                    else _marker(messages, "EXPECTED_WINDOW_ID")
                ),
                "findings": [
                    {
                        "finding_id": "shared-finding",
                        "polarity": "positive",
                        "observation": "The agent used relevant checks.",
                        "evidence_ids": [evidence_id],
                    }
                ],
            }
        else:
            payload = (
                {"not": "a verdict"}
                if self.fail_merge
                else {
                    "schema_version": 1,
                    "status": "scored",
                    "score": 0.75,
                    "rationale": "The session was mostly effective.",
                    "evidence_ids": ["trace-1", "trace-2"],
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
            transport_request_count=1,
            raw_output_digest=hashlib.sha256(raw.encode()).hexdigest(),
        )


def _reviewer(
    *,
    client: _ScriptedClient | None = None,
    artifacts: dict[str, Mapping[str, Any]] | None = None,
    policy: JudgingContextPolicy | None = None,
    cancelled=lambda: False,
    judge: PositionedJudge | None = None,
) -> tuple[SlidingReviewer, _ScriptedClient, dict[str, Mapping[str, Any]], dict[str, object]]:
    session = SessionView(
        "session-1",
        [_turn(f"trace-{index}", index) for index in range(1, 5)],
        "config-1",
        "main",
    )
    active_policy = policy or _policy()
    active_judge = judge or _judge()
    plan = build_window_plan(session, active_policy, active_judge.max_input_tokens)
    judging_plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=build_rubric_catalog().rubrics,
        review_depth="primary",
        second_opinion_margin=None,
        judge_models=(active_judge,),
        context_policy=active_policy,
    )
    assert len(plan["windows"]) == 2
    stored = {} if artifacts is None else artifacts
    scripted = client or _ScriptedClient()

    def record(artifact_id: str, artifact: Mapping[str, Any]) -> None:
        existing = stored.setdefault(artifact_id, dict(artifact))
        if existing != artifact:
            raise ValueError("artifact conflict")

    return (
        SlidingReviewer(
            session=session,
            judge=active_judge,
            judging_plan=judging_plan,
            context_policy=active_policy,
            client=scripted,
            load_artifact=stored.get,
            record_artifact=record,
            is_cancelled=cancelled,
        ),
        scripted,
        stored,
        plan,
    )


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
    assert first.evidence_ids == ("trace-1", "trace-2")
    assert first.behavioral_feedback is not None
    assert first.usage == {"input_tokens": 15, "total_tokens": 20}
    assert len(first.steps) == 5
    assert not any(step.reused for step in first.steps)
    assert second.usage == {"input_tokens": 21, "total_tokens": 24}
    assert second.transport_request_count == 3
    assert [step.reused for step in second.steps] == [True, True, False, False, False]
    assert all(step.requested_model == "judge-1" for step in first.steps)
    assert all(call["temperature"] == 0.0 for call in client.calls)
    assert [call["schema_name"] for call in client.calls[:5]] == [
        "chunk_digest",
        "chunk_digest",
        "window_findings",
        "window_findings",
        "merged_verdict",
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
        own_window_id = plan["windows"][index]["window_id"]
        assert f"EXPECTED_WINDOW_ID: {own_window_id}" in text

    merge_text = [call for call in client.calls if call["phase"] == "merge"][0]["messages"][-1][
        "content"
    ]
    assert merge_text.count("shared-finding") == 1
    assert '"raw_coverage_trace_ids":["trace-1","trace-2","trace-3","trace-4"]' in merge_text


def test_artifact_replay_makes_zero_model_calls() -> None:
    first, _, artifacts, _ = _reviewer()
    first_result = first.review(_rubric("judge.session_outcome"))
    replay_client = _ScriptedClient()
    replay, _, _, _ = _reviewer(client=replay_client, artifacts=artifacts)

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
    first, _, artifacts, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    original = first.review(_rubric("judge.session_autonomy"))
    replay_client = _ScriptedClient()
    replay, _, _, _ = _reviewer(client=replay_client, artifacts=artifacts)

    restored = replay.review(_rubric("judge.session_autonomy"))

    assert replay_client.calls == []
    assert all(step.reused for step in restored.steps)
    assert [replace(step, reused=False) for step in restored.steps] == [
        replace(step, reused=False) for step in original.steps
    ]
    assert restored.usage == {}
    assert restored.transport_request_count == 0


def test_partial_replay_charges_only_new_window_and_merge_calls() -> None:
    first, _, artifacts, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    digest_artifacts = {
        artifact_id: artifact
        for artifact_id, artifact in artifacts.items()
        if artifact_id.startswith("digest/")
    }
    partial_client = _ScriptedClient()
    partial, _, _, _ = _reviewer(client=partial_client, artifacts=digest_artifacts)

    result = partial.review(_rubric("judge.session_autonomy"))

    assert [call["phase"] for call in partial_client.calls] == ["window", "window", "merge"]
    assert [step.reused for step in result.steps] == [True, True, False, False, False]
    assert result.usage == {"input_tokens": 6, "total_tokens": 9}
    assert result.transport_request_count == 3


def test_replay_rejects_tampered_inference_audit_without_model_call() -> None:
    first, _, artifacts, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    artifact_id = next(iter(artifacts))
    artifact = dict(artifacts[artifact_id])
    payload = dict(artifact["payload"])
    audit = dict(payload["audit"])
    audit["usage"] = {"total_tokens": "not-an-integer"}
    payload["audit"] = audit
    artifact["payload"] = payload
    artifact["content_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    artifacts[artifact_id] = artifact
    replay_client = _ScriptedClient()
    replay, _, _, _ = _reviewer(client=replay_client, artifacts=artifacts)

    result = replay.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.steps == ()
    assert result.usage == {}
    assert replay_client.calls == []


def test_replay_rejects_an_artifact_persisted_as_already_reused() -> None:
    first, _, artifacts, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    artifact_id = next(iter(artifacts))
    artifact = dict(artifacts[artifact_id])
    payload = dict(artifact["payload"])
    audit = dict(payload["audit"])
    audit["reused"] = True
    payload["audit"] = audit
    artifact["payload"] = payload
    artifact["content_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    artifacts[artifact_id] = artifact
    replay_client = _ScriptedClient()
    replay, _, _, _ = _reviewer(client=replay_client, artifacts=artifacts)

    result = replay.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.steps == ()
    assert replay_client.calls == []


def test_replay_rejects_arbitrary_schema_fallback_detail_without_persisting_it() -> None:
    first, _, artifacts, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    artifact_id = next(iter(artifacts))
    artifact = dict(artifacts[artifact_id])
    payload = dict(artifact["payload"])
    audit = dict(payload["audit"])
    secret = "SENTINEL_PRIVATE_REPLAY_DETAIL"
    audit["output_mode"] = "json_object_fallback"
    audit["schema_fallback_reason"] = secret
    payload["audit"] = audit
    artifact["payload"] = payload
    artifact["content_digest"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    artifacts[artifact_id] = artifact
    replay_client = _ScriptedClient()
    replay, _, _, _ = _reviewer(client=replay_client, artifacts=artifacts)

    result = replay.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.message == "schema fallback metadata is invalid"
    assert secret not in result.message
    assert result.steps == ()
    assert replay_client.calls == []


@pytest.mark.parametrize(
    "judge",
    [
        _judge(backend="wandb"),
        _judge(family="different-family"),
        _judge(max_input_tokens=61_000),
    ],
)
def test_artifact_identity_binds_complete_positioned_judge(judge: PositionedJudge) -> None:
    first, _, first_artifacts, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    second, _, second_artifacts, _ = _reviewer(judge=judge)

    second.review(_rubric("judge.session_outcome"))

    assert set(first_artifacts).isdisjoint(second_artifacts)


def test_protocol_contract_digest_binds_version_prompts_and_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weave_agent_signals.judges.sliding as sliding

    original = sliding_protocol_contract_digest()
    original_template = sliding._DIGEST_SYSTEM_TEMPLATE
    monkeypatch.setattr(sliding, "SLIDING_PROTOCOL_VERSION", "test-version")
    assert sliding_protocol_contract_digest() != original
    monkeypatch.setattr(sliding, "SLIDING_PROTOCOL_VERSION", "2")
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


def test_artifact_ids_bind_protocol_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import weave_agent_signals.judges.sliding as sliding

    first, _, first_artifacts, _ = _reviewer()
    first.review(_rubric("judge.session_outcome"))
    monkeypatch.setattr(sliding, "SLIDING_PROTOCOL_VERSION", "future-version")
    second, _, second_artifacts, _ = _reviewer()

    second.review(_rubric("judge.session_outcome"))

    assert set(first_artifacts).isdisjoint(second_artifacts)


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
    reviewer, _, _, _ = _reviewer(client=_ScriptedClient(invalid_window_citation=True))

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.evidence_ids == ()
    assert result.behavioral_feedback is None
    assert result.error_type == "ValueError"
    assert result.message == "unknown evidence ID"
    assert "unknown-evidence" not in result.message


@pytest.mark.parametrize(
    ("client", "expected_message", "secret"),
    [
        (
            _ScriptedClient(invalid_chunk_id=True),
            "unexpected chunk ID",
            "SENTINEL_PRIVATE_CHUNK_ID",
        ),
        (
            _ScriptedClient(invalid_window_id=True),
            "unexpected window ID",
            "SENTINEL_PRIVATE_WINDOW_ID",
        ),
    ],
)
def test_rejected_model_identifiers_are_not_persisted(
    client: _ScriptedClient,
    expected_message: str,
    secret: str,
) -> None:
    reviewer, _, _, _ = _reviewer(client=client)

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.message == expected_message
    assert secret not in result.message


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
    assert result.usage == {"input_tokens": 15, "total_tokens": 20}
    assert result.transport_request_count == 5
    assert result.output_mode == "json_schema"
    assert result.schema_name == "merged_verdict"
    assert "schema_version" in (result.message or "")


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


@pytest.mark.parametrize("callback", ["load", "record", "cancel"])
def test_callback_exception_message_is_sanitized_before_observation(callback: str) -> None:
    secret = "SENTINEL_PRIVATE_CALLBACK_DETAIL"
    reviewer, client, artifacts, _ = _reviewer()

    def fail(*_args: object) -> None:
        raise RuntimeError(secret)

    if callback == "load":
        reviewer._load_artifact = fail
    elif callback == "record":
        reviewer._record_artifact = fail
    else:
        reviewer._is_cancelled = fail

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert result.message == "review infrastructure failed"
    assert secret not in result.message
    if callback == "load":
        assert client.calls == []
        assert artifacts == {}


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


def test_reviewer_rejects_tampered_plan_before_inference() -> None:
    reviewer, client, artifacts, plan = _reviewer()
    del reviewer
    session = SessionView(
        "session-1",
        [_turn(f"trace-{index}", index) for index in range(1, 5)],
        "config-1",
        "main",
    )
    judging_plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=build_rubric_catalog().rubrics,
        review_depth="primary",
        second_opinion_margin=None,
        judge_models=(_judge(),),
        context_policy=_policy(),
    )
    tampered = dict(judging_plan)
    tampered["plan_id"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="full content"):
        SlidingReviewer(
            session=session,
            judge=_judge(),
            judging_plan=tampered,
            context_policy=_policy(),
            client=client,
            load_artifact=artifacts.get,
            record_artifact=lambda _id, _artifact: None,
        )


def test_reviewer_authenticates_exact_ordinal_against_the_full_plan() -> None:
    session = SessionView(
        "session-1",
        [_turn(f"trace-{index}", index) for index in range(1, 5)],
        "config-1",
        "main",
    )
    plan = build_judging_plan(
        [session],
        cohort_id="cohort",
        rubrics=build_rubric_catalog().rubrics,
        review_depth="primary",
        second_opinion_margin=None,
        judge_models=(_judge(),),
        context_policy=_policy(),
    )
    tampered = json.loads(json.dumps(plan))
    tampered["sessions"][0]["reviewers"][0]["ordinal"] = 2
    body = {key: value for key, value in tampered.items() if key != "plan_id"}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    tampered["plan_id"] = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    with pytest.raises(ValueError, match="ordinals"):
        SlidingReviewer(
            session=session,
            judge=_judge(),
            judging_plan=tampered,
            context_policy=_policy(),
            client=_ScriptedClient(),
            load_artifact=lambda _id: None,
            record_artifact=lambda _id, _artifact: None,
        )


def test_malformed_replayed_artifact_fails_without_model_call() -> None:
    reviewer, _, artifacts, _ = _reviewer()
    reviewer.review(_rubric("judge.session_outcome"))
    artifact_id = next(iter(artifacts))
    artifacts[artifact_id] = {**artifacts[artifact_id], "kind": "window_findings"}
    replay_client = _ScriptedClient()
    replay, _, _, _ = _reviewer(client=replay_client, artifacts=artifacts)

    result = replay.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert replay_client.calls == []


def test_over_budget_request_fails_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    rubric = SESSION_RUBRICS["judge.session_outcome"]
    monkeypatch.setitem(
        SESSION_RUBRICS,
        rubric.scorer_name,
        replace(rubric, system_prompt=rubric.system_prompt + "y" * 100_000),
    )
    reviewer, client, _, _ = _reviewer()

    result = reviewer.review(_rubric("judge.session_outcome"))

    assert result.status == "failed"
    assert result.error_type == "ValueError"
    assert "budget" in (result.message or "")
    assert [call["phase"] for call in client.calls] == ["digest", "digest"]
