"""Tests for judge runner with mocked inference."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch
import os

import respx

from weave_agent_signals.judges.inference import InferenceClient, INFERENCE_BASE, DEFAULT_ENTITY, DEFAULT_PROJECT
from weave_agent_signals.judges.rubrics import VERIFICATION_DISCIPLINE, RUBRICS
from weave_agent_signals.judges.runner import judge_turn, _run_rubric
from weave_agent_signals.judges.digest import build_turn_digest
from weave_agent_signals.models import ToolSpan, TurnSpan


def _ts(h=12, m=0):
    return datetime(2026, 7, 9, h, m, tzinfo=timezone.utc)


def _turn(tool_calls=None, model="claude-opus-4"):
    return TurnSpan(
        trace_id="t1",
        conversation_id="c1",
        started_at=_ts(),
        ended_at=_ts(12, 5),
        model=model,
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        status_code="OK",
        config_version="abc",
        git_branch="main",
        effort_level="high",
        session_id="s1",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=tool_calls or [],
        chat_spans=[],
        subagents=[],
    )


def _bash(cmd, result="", status="OK"):
    return ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments=f'{{"command": "{cmd}"}}',
        result=result,
        status_code=status,
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )


def _mock_judge_response(score: float, rationale: str = "test rationale"):
    return {
        "choices": [{
            "message": {
                "content": json.dumps({"score": score, "rationale": rationale}),
            }
        }],
        "model": "gpt-oss-20b",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_turn_returns_scores():
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json=_mock_judge_response(0.8, "Agent ran tests after changes"),
    )
    turn = _turn(tool_calls=[_bash("pytest tests/", "5 passed")])
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_turn(
        turn, client,
        rubrics=[VERIFICATION_DISCIPLINE],
        judge_candidates=["gpt-oss-20b"],
    )
    assert len(scores) == 1
    s = scores[0]
    assert s.scorer == "judge.verification"
    assert s.value == 0.8
    assert s.confidence == 0.7
    assert "verified" in s.tags
    assert s.metadata["judge_model"] == "gpt-oss-20b"
    assert s.metadata["agent_model"] == "claude-opus-4"
    assert s.metadata["rationale"] == "Agent ran tests after changes"


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_turn_low_score_tags():
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json=_mock_judge_response(0.25, "No verification"),
    )
    turn = _turn()
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_turn(
        turn, client,
        rubrics=[VERIFICATION_DISCIPLINE],
        judge_candidates=["gpt-oss-20b"],
    )
    assert len(scores) == 1
    assert "no_verification" in scores[0].tags
    assert scores[0].value == 0.25


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_turn_excludes_agent_family():
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json=_mock_judge_response(0.9),
    )
    turn = _turn(model="claude-opus-4")
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_turn(
        turn, client,
        rubrics=[VERIFICATION_DISCIPLINE],
        judge_candidates=["Llama-3.1-8B", "gpt-oss-20b"],
    )
    assert len(scores) == 1
    assert scores[0].metadata["judge_family"] != "anthropic"


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_turn_multiple_rubrics():
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json=_mock_judge_response(0.7),
    )
    turn = _turn(tool_calls=[_bash("pytest", "3 passed")])
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_turn(
        turn, client,
        rubrics=list(RUBRICS.values()),
        judge_candidates=["gpt-oss-20b"],
    )
    assert len(scores) == len(RUBRICS)
    scorer_names = {s.scorer for s in scores}
    assert "judge.verification" in scorer_names
    assert "judge.error_recovery" in scorer_names
    assert "judge.tool_choice" in scorer_names
    assert "judge.completion" in scorer_names


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_handles_bad_json():
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json={
            "choices": [{"message": {"content": "not json at all"}}],
            "model": "gpt-oss-20b",
            "usage": {},
        },
    )
    turn = _turn()
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_turn(
        turn, client,
        rubrics=[VERIFICATION_DISCIPLINE],
        judge_candidates=["gpt-oss-20b"],
    )
    assert len(scores) == 0


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_clamps_score():
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json=_mock_judge_response(1.5, "over-scored"),
    )
    turn = _turn()
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_turn(
        turn, client,
        rubrics=[VERIFICATION_DISCIPLINE],
        judge_candidates=["gpt-oss-20b"],
    )
    assert scores[0].value == 1.0


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_inference_client_chat_json():
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json={
            "choices": [{"message": {"content": '{"score": 0.5}'}}],
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )
    client = InferenceClient(base_url=INFERENCE_BASE)
    parsed, resp = client.chat_json(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
    )
    assert parsed["score"] == 0.5
    assert resp.model == "test-model"


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_metadata_includes_families():
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json=_mock_judge_response(0.8),
    )
    turn = _turn(model="claude-opus-4")
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_turn(
        turn, client,
        rubrics=[VERIFICATION_DISCIPLINE],
        judge_candidates=["gpt-oss-20b"],
    )
    meta = scores[0].metadata
    assert meta["agent_family"] == "anthropic"
    assert meta["judge_family"] == "openai"
