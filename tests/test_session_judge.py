"""Tests for session-level judging: digest, rubrics, PoLL panel."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import respx

from weave_agent_signals.judges.digest import build_judge_messages, build_session_digest
from weave_agent_signals.judges.inference import InferenceClient, INFERENCE_BASE
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS, SESSION_OUTCOME
from weave_agent_signals.judges.runner import (
    POLL_JUDGE_CANDIDATES_OPENAI,
    POLL_JUDGE_CANDIDATES_WANDB,
    _roster,
    judge_default_model,
    judge_session,
)
from weave_agent_signals.models import SessionView, ToolSpan, TurnSpan


class _StubClient:
    def __init__(self, backend):
        self.backend = backend


def _ts(h=12, m=0):
    return datetime(2026, 7, 9, h, m, tzinfo=timezone.utc)


def _turn(trace_id="t1", tool_calls=None, model="claude-opus-4",
          h=12, m=0, steering=0, denials=0, errors=0):
    return TurnSpan(
        trace_id=trace_id,
        conversation_id="c1",
        started_at=_ts(h, m),
        ended_at=_ts(h, m + 5),
        model=model,
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        status_code="OK",
        config_version="abc",
        git_branch="main",
        effort_level="high",
        session_id="s1",
        steering_count=steering,
        denial_count=denials,
        tool_error_count=errors,
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


def _session(turns=None):
    turns = turns or [_turn()]
    return SessionView(
        conversation_id="c1",
        turns=turns,
        config_version="abc",
        git_branch="main",
    )


# --- Session digest tests ---

def test_session_digest_includes_summary():
    session = _session([
        _turn("t1", h=12, m=0),
        _turn("t2", h=12, m=10),
    ])
    digest = build_session_digest(session)
    assert "Session:" in digest
    assert "Turns: 2" in digest
    assert "Tokens:" in digest


def test_session_digest_aggregates_events():
    session = _session([
        _turn("t1", steering=2, denials=1, errors=0),
        _turn("t2", steering=1, denials=0, errors=3),
    ])
    digest = build_session_digest(session)
    assert "steering=3" in digest
    assert "denials=1" in digest
    assert "tool_errors=3" in digest


def test_session_digest_includes_turn_summaries():
    session = _session([
        _turn("t1", tool_calls=[_bash("pytest", "5 passed")]),
        _turn("t2", tool_calls=[_bash("git commit -m fix", "")]),
    ])
    digest = build_session_digest(session)
    assert "Turn 1" in digest
    assert "Turn 2" in digest
    assert "pytest" in digest
    assert "git commit" in digest


def test_session_digest_limits_turn_detail():
    turns = [_turn(f"t{i}", h=12, m=i) for i in range(20)]
    session = _session(turns)
    digest = build_session_digest(session)
    assert len(digest) < 15000


# --- Session rubric tests ---

def test_session_rubrics_exist():
    assert "judge.session_outcome" in SESSION_RUBRICS
    assert SESSION_OUTCOME.scorer_name == "judge.session_outcome"


# --- Judge message framing ---

def test_build_judge_messages_session_framing():
    msgs = build_judge_messages("sys", "criteria", "digest", granularity="session")
    user = msgs[1]["content"].lower()
    assert "this session" in user
    assert "this turn" not in user


def test_build_judge_messages_turn_framing_default():
    msgs = build_judge_messages("sys", "criteria", "digest")
    assert "this turn" in msgs[1]["content"].lower()


# --- Backend-aware model selection ---

def test_roster_selects_wandb_models():
    roster = _roster(_StubClient("wandb"))
    assert roster["poll"] == POLL_JUDGE_CANDIDATES_WANDB
    assert judge_default_model(_StubClient("wandb")) == "gpt-oss-120b"


def test_roster_selects_openai_models():
    roster = _roster(_StubClient("openai"))
    assert roster["poll"] == POLL_JUDGE_CANDIDATES_OPENAI
    assert judge_default_model(_StubClient("openai")) == "gpt-4o"


def test_roster_unknown_backend_defaults_to_openai():
    roster = _roster(_StubClient("custom"))
    assert roster["poll"] == POLL_JUDGE_CANDIDATES_OPENAI


# --- PoLL panel tests ---

def _mock_judge_response(score, rationale="test"):
    return {
        "choices": [{"message": {"content": json.dumps({"score": score, "rationale": rationale})}}],
        "model": "gpt-oss-20b",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_session_returns_scores():
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json=_mock_judge_response(0.8, "Good session outcome"),
    )
    session = _session([
        _turn("t1", tool_calls=[_bash("pytest", "5 passed")]),
    ])
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_session(
        session, client,
        rubrics=[SESSION_OUTCOME],
        judge_candidates=["gpt-oss-20b"],
    )
    assert len(scores) == 1
    s = scores[0]
    assert s.scorer == "judge.session_outcome"
    assert s.granularity == "session"
    assert s.metadata["judge_model"] == "gpt-oss-20b"


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_session_poll_panel():
    """PoLL panel uses multiple judges and mean-pools scores."""
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        model = body["model"]
        if "gpt" in model:
            score = 0.8
        elif "Llama" in model.lower() or "llama" in model.lower():
            score = 0.6
        else:
            score = 0.7
        return respx.MockResponse(
            200,
            json=_mock_judge_response(score, f"judge {model}"),
        )

    respx.post(f"{INFERENCE_BASE}/chat/completions").mock(side_effect=side_effect)

    session = _session([_turn("t1")])
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_session(
        session, client,
        rubrics=[SESSION_OUTCOME],
        judge_candidates=["gpt-oss-20b", "Llama-3.1-8B", "granite-4.1-8b"],
        panel_size=3,
    )
    assert len(scores) == 1
    s = scores[0]
    assert s.scorer == "judge.session_outcome"
    assert 0.6 <= s.value <= 0.8
    assert s.metadata["panel_size"] == 3
    assert s.confidence > 0.7


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_session_poll_handles_failures():
    """PoLL panel handles individual judge failures gracefully."""
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return respx.MockResponse(500, json={"error": "internal"})
        return respx.MockResponse(
            200,
            json=_mock_judge_response(0.7, "ok"),
        )

    respx.post(f"{INFERENCE_BASE}/chat/completions").mock(side_effect=side_effect)

    session = _session([_turn("t1")])
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_session(
        session, client,
        rubrics=[SESSION_OUTCOME],
        judge_candidates=["gpt-oss-20b", "Llama-3.1-8B", "granite-4.1-8b"],
        panel_size=3,
    )
    assert len(scores) == 1
    assert scores[0].metadata["panel_size"] == 2


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_session_escalation_on_disagreement():
    """High panel spread triggers escalation to a larger model."""
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        model = body["model"]
        if "gpt-oss-120b" in model:
            return respx.MockResponse(
                200, json=_mock_judge_response(0.7, "escalation tiebreaker"),
            )
        if call_count == 1:
            return respx.MockResponse(
                200, json=_mock_judge_response(0.9, "high"),
            )
        return respx.MockResponse(
            200, json=_mock_judge_response(0.4, "low"),
        )

    respx.post(f"{INFERENCE_BASE}/chat/completions").mock(side_effect=side_effect)

    session = _session([_turn("t1")])
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_session(
        session, client,
        rubrics=[SESSION_OUTCOME],
        judge_candidates=["Llama-3.1-8B", "granite-4.1-8b"],
        panel_size=2,
    )
    assert len(scores) == 1
    s = scores[0]
    assert s.metadata["escalated"] is True
    assert s.metadata["panel_size"] == 3
    assert s.metadata["panel_spread"] > 0.3


@respx.mock
@patch.dict("os.environ", {"JUDGE_API_KEY": "test-key"})
def test_judge_session_no_escalation_when_agreed():
    """No escalation when panel agrees (low spread)."""
    respx.post(f"{INFERENCE_BASE}/chat/completions").respond(
        json=_mock_judge_response(0.75, "consensus"),
    )

    session = _session([_turn("t1")])
    client = InferenceClient(base_url=INFERENCE_BASE)
    scores = judge_session(
        session, client,
        rubrics=[SESSION_OUTCOME],
        judge_candidates=["Llama-3.1-8B", "granite-4.1-8b"],
        panel_size=2,
    )
    assert len(scores) == 1
    assert scores[0].metadata["escalated"] is False
    assert scores[0].metadata["panel_spread"] == 0.0
