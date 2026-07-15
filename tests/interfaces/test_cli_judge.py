"""Focused standalone judging policy and orchestration checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from weave_agent_signals import cli
from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog


def _catalog():
    return build_model_catalog(which=lambda _executable: "/bin/fake")


def _args(**changes):
    values = {
        "entity": "entity",
        "project": "project",
        "since": None,
        "limit": 10,
        "rubric": None,
        "judge_backend": "cli",
        "judge_models": ["claude-sonnet-5", "gpt-5.6-sol"],
        "review_depth": "selective",
        "second_opinion_margin": 0.1,
        "force": False,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def _turn(trace_id="turn-1"):
    turn = MagicMock()
    turn.trace_id = trace_id
    turn.conversation_id = "session-1"
    turn.started_at = datetime(2026, 7, 14, tzinfo=timezone.utc)
    turn.model = "evaluated-agent"
    turn.config_version = "config-1"
    turn.git_branch = "main"
    turn.ref_for.return_value = "weave:///turn"
    return turn


def _plan():
    return {
        "plan_id": "plan-1",
        "totals": {"episodes_selected": 1},
        "sessions": [
            {
                "conversation_id": "session-1",
                "selected_episodes": [
                    {
                        "trace_id": "turn-1",
                        "selection_kind": "signal",
                        "selection_reasons": ["tool evidence"],
                        "evidence_trace_ids": ["turn-1"],
                        "rubrics": [
                            {
                                "id": "judge.tool_choice",
                                "applicability": "applicable",
                            }
                        ],
                    }
                ],
                "session_rubrics": [
                    {
                        "id": "judge.session_outcome",
                        "applicability": "applicable",
                    }
                ],
            }
        ],
    }


def test_default_policy_resolves_catalog_recommendations(monkeypatch):
    catalog = _catalog()
    monkeypatch.setattr(cli, "build_model_catalog", lambda: catalog)

    policy = cli._resolve_judge_policy(
        _args(
            judge_backend=None,
            judge_models=None,
            review_depth=None,
            second_opinion_margin=None,
        )
    )

    backend = catalog.backend(catalog.recommended_judge_backend)
    assert [judge.id for judge in policy.judges] == list(backend.recommended_judges)
    assert policy.depth == backend.recommended_review_depth
    assert policy.second_opinion_margin == catalog.review_defaults["second_opinion_margin"]


def test_explicit_policy_preserves_order_and_rejects_invalid_cardinality(monkeypatch):
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)

    policy = cli._resolve_judge_policy(_args())

    assert [judge.id for judge in policy.judges] == [
        "claude-sonnet-5",
        "gpt-5.6-sol",
    ]
    with pytest.raises(RuntimeError, match="primary review requires exactly 1 judge"):
        cli._resolve_judge_policy(_args(review_depth="primary", second_opinion_margin=None))


def test_turn_and_session_judging_share_the_exact_policy(monkeypatch):
    model_catalog = _catalog()
    rubric_catalog = build_rubric_catalog()
    turn = _turn()
    weave = MagicMock()
    weave.query_turns.return_value = [turn]
    weave.__enter__.return_value = weave
    inference = MagicMock()
    inference.__enter__.return_value = inference
    judge_turn = MagicMock(return_value=[])
    judge_session = MagicMock(return_value=[])

    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(cli, "build_model_catalog", lambda: model_catalog)
    monkeypatch.setattr(cli, "build_rubric_catalog", lambda: rubric_catalog)
    monkeypatch.setattr(cli, "build_judging_plan", lambda *_args, **_kwargs: _plan())
    monkeypatch.setattr(
        cli,
        "_make_model_client",
        lambda _args, _model: inference,
    )
    monkeypatch.setattr(cli, "judge_turn", judge_turn)
    monkeypatch.setattr(cli, "judge_session", judge_session)

    rc = cli.cmd_judge(_args(rubric="judge.tool_choice,judge.session_outcome"))

    assert rc == 0
    weave.hydrate_turns_batch.assert_called_once_with([turn])
    turn_policy = judge_turn.call_args.kwargs
    session_policy = judge_session.call_args.kwargs
    assert [judge.id for judge in turn_policy["judges"]] == [
        "claude-sonnet-5",
        "gpt-5.6-sol",
    ]
    assert session_policy["judges"] == turn_policy["judges"]
    assert session_policy["review_depth"] == turn_policy["review_depth"] == "selective"
    assert session_policy["second_opinion_margin"] == 0.1
    assert turn_policy["second_opinion_margin"] == 0.1
    assert [rubric.id for rubric in turn_policy["rubrics"]] == ["judge.tool_choice"]
    assert [rubric.id for rubric in session_policy["rubrics"]] == ["judge.session_outcome"]


def test_session_judging_uses_the_ordered_union_of_episode_evidence(monkeypatch):
    model_catalog = _catalog()
    rubric_catalog = build_rubric_catalog()
    turns = [_turn(f"turn-{index}") for index in range(1, 5)]
    weave = MagicMock()
    weave.query_turns.return_value = turns
    weave.__enter__.return_value = weave
    inference = MagicMock()
    inference.__enter__.return_value = inference
    judge_session = MagicMock(return_value=[])
    plan = {
        "plan_id": "plan-1",
        "totals": {"episodes_selected": 2},
        "sessions": [
            {
                "conversation_id": "session-1",
                "selected_episodes": [
                    {
                        "trace_id": "turn-2",
                        "evidence_trace_ids": ["turn-1", "turn-2"],
                        "rubrics": [],
                    },
                    {
                        "trace_id": "turn-4",
                        "evidence_trace_ids": ["turn-2", "turn-3", "turn-4"],
                        "rubrics": [],
                    },
                ],
                "session_rubrics": [
                    {
                        "id": "judge.session_outcome",
                        "applicability": "applicable",
                    }
                ],
            }
        ],
    }

    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(cli, "build_model_catalog", lambda: model_catalog)
    monkeypatch.setattr(cli, "build_rubric_catalog", lambda: rubric_catalog)
    monkeypatch.setattr(cli, "build_judging_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(cli, "_make_model_client", lambda _args, _model: inference)
    monkeypatch.setattr(cli, "judge_session", judge_session)

    rc = cli.cmd_judge(_args(rubric="judge.session_outcome"))

    assert rc == 0
    assert judge_session.call_args.kwargs["evidence_trace_ids"] == [
        "turn-1",
        "turn-2",
        "turn-3",
        "turn-4",
    ]


def test_hydration_failure_aborts_before_inference(monkeypatch):
    weave = MagicMock()
    weave.query_turns.return_value = [_turn()]
    weave.hydrate_turns_batch.side_effect = RuntimeError("detail hydration truncated")
    weave.__enter__.return_value = weave
    make_judge = MagicMock()
    monkeypatch.setattr(cli, "WeaveClient", lambda **_kwargs: weave)
    monkeypatch.setattr(cli, "build_model_catalog", _catalog)
    monkeypatch.setattr(cli, "build_rubric_catalog", build_rubric_catalog)
    monkeypatch.setattr(cli, "_make_model_client", make_judge)

    with pytest.raises(RuntimeError, match="detail hydration truncated"):
        cli.cmd_judge(_args())

    make_judge.assert_not_called()
