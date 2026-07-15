"""Tests for pattern analysis: A/B leaderboards, trends, coaching digest."""

from __future__ import annotations

import pytest

from weave_agent_signals.patterns import (
    ScoreSummary,
    ab_leaderboard,
    aggregate_scores,
    coaching_digest,
    detect_config_regressions,
    detect_regressions,
)


def _fb(
    scorer,
    rating,
    config_version="v1",
    tags=None,
    conversation_id="c1",
    scored_at="2026-07-09T12:00:00",
):
    return {
        "feedback_type": f"weave_agent_signals.{scorer}",
        "payload": {
            "rating": rating,
            "confidence": 0.8,
            "tags": tags or [],
            "reason": "test",
            "details": {
                "config_version": config_version,
                "scored_at": scored_at,
            },
            "granularity": "turn",
        },
        "weave_ref": f"weave:///e/p/agent_turn/{conversation_id}",
    }


def _session_judge_fb(
    scorer,
    rating,
    *,
    review_status,
    context_overrides=None,
    behavioral_feedback=None,
    evidence_ids=None,
    **kwargs,
):
    feedback = _fb(scorer, rating, **kwargs)
    feedback["payload"]["granularity"] = "session"
    feedback["payload"]["details"].update(
        {
            "evaluation_unit": "session",
            "review_status": review_status,
            "rubric_version": "v2",
            "rubric_threshold": 0.5,
            "review_depth": "selective",
            "review_policy_version": "2",
            "second_opinion_margin": 0.1,
            "requested_judge_models": ["judge-a", "judge-b", "judge-c"],
            "behavioral_feedback": behavioral_feedback or [],
            "evidence_trace_ids": evidence_ids or [],
        }
    )
    feedback["payload"]["details"].update(context_overrides or {})
    return feedback


def _legacy_episode_judge_fb(scorer, rating, **kwargs):
    feedback = _fb(scorer, rating, **kwargs)
    feedback["payload"]["details"].update(
        {
            "evaluation_unit": "episode",
            "selection_kind": "deterministic_trigger",
            "selection_reasons": ["tool_error"],
        }
    )
    return feedback


# --- aggregate_scores tests ---


def test_aggregate_scores_basic():
    feedback = [
        _fb("outcome.test", 1.0),
        _fb("outcome.test", 0.0),
        _fb("outcome.test", 1.0),
        _fb("outcome.build", 1.0),
    ]
    result = aggregate_scores(feedback)
    assert "outcome.test" in result
    assert result["outcome.test"].count == 3
    assert result["outcome.test"].mean == pytest.approx(2 / 3, abs=0.01)
    assert result["outcome.build"].count == 1


def test_aggregate_scores_empty():
    assert aggregate_scores([]) == {}


def test_aggregate_scores_strips_prefix():
    feedback = [_fb("outcome.test", 0.5)]
    result = aggregate_scores(feedback)
    assert "outcome.test" in result


@pytest.mark.parametrize(
    "rating",
    [True, False, "0.5", float("nan"), float("inf"), -0.01, 1.01],
)
def test_aggregate_scores_ignores_invalid_ratings(rating):
    assert aggregate_scores([_fb("outcome.test", rating)]) == {}


def test_coaching_digest_counts_only_strict_numeric_ratings():
    digest = coaching_digest(
        [
            _fb("outcome.test", 0.75),
            _fb("outcome.test", "1"),
            _fb("outcome.test", True),
            _fb("outcome.test", float("nan")),
        ]
    )

    summary_line = next(line for line in digest.splitlines() if "outcome.test" in line)
    assert "mean=0.75" in summary_line
    assert "n=1" in summary_line


def test_population_summary_excludes_noncomplete_session_judgments():
    result = aggregate_scores(
        [
            _session_judge_fb("judge.session_outcome", 0.8, review_status="complete"),
            _session_judge_fb("judge.session_outcome", 0.2, review_status="degraded"),
            _session_judge_fb("judge.session_outcome", 0.5, review_status="unresolved"),
        ]
    )

    assert len(result) == 1
    summary = next(iter(result.values()))
    assert summary.count == 1
    assert summary.mean == 0.8


@pytest.mark.parametrize(
    "context_overrides",
    [
        {"rubric_version": "v3"},
        {"rubric_threshold": 0.6},
        {
            "review_depth": "full_panel",
            "second_opinion_margin": None,
        },
        {"review_policy_version": "3"},
        {"second_opinion_margin": 0.2},
        {"requested_judge_models": ["judge-b", "judge-a", "judge-c"]},
    ],
)
def test_session_judge_summary_keeps_evaluation_contexts_separate(context_overrides):
    result = aggregate_scores(
        [
            _session_judge_fb(
                "judge.session_outcome",
                1.0,
                review_status="complete",
            ),
            _session_judge_fb(
                "judge.session_outcome",
                0.0,
                review_status="complete",
                context_overrides=context_overrides,
            ),
        ]
    )

    assert len(result) == 2
    assert sorted(summary.mean for summary in result.values()) == [0.0, 1.0]


def test_session_judge_context_ignores_runtime_attempt_count():
    first = _session_judge_fb(
        "judge.session_outcome",
        1.0,
        review_status="complete",
    )
    second = _session_judge_fb(
        "judge.session_outcome",
        0.0,
        review_status="complete",
    )
    first["payload"]["details"].update({"attempt_count": 1, "attempts": [{"position": 1}]})
    second["payload"]["details"].update(
        {"attempt_count": 3, "attempts": [{"position": 1}, {"position": 2}]}
    )

    result = aggregate_scores([first, second])

    assert len(result) == 1
    assert next(iter(result.values())).mean == 0.5


def test_session_judge_context_label_explains_what_was_compared():
    result = aggregate_scores(
        [
            _session_judge_fb(
                "judge.session_outcome",
                1.0,
                review_status="complete",
            )
        ]
    )

    assert list(result) == [
        "judge.session_outcome [rubric v2, threshold 0.5; selective, "
        "policy 2, margin 0.1; judges judge-a → judge-b → judge-c]"
    ]


def test_incomplete_session_judge_context_is_audit_only():
    incomplete = _session_judge_fb(
        "judge.session_outcome",
        1.0,
        review_status="complete",
    )
    del incomplete["payload"]["details"]["review_policy_version"]
    canonical = _session_judge_fb(
        "judge.session_outcome",
        0.0,
        review_status="complete",
    )

    result = aggregate_scores([incomplete, canonical])

    assert len(result) == 1
    assert next(iter(result.values())).mean == 0.0


def test_session_judge_without_review_status_is_audit_only():
    incomplete = _fb("judge.session_outcome", 1.0)
    incomplete["payload"]["granularity"] = "session"

    assert aggregate_scores([incomplete]) == {}


@pytest.mark.parametrize(
    ("granularity", "evaluation_unit"),
    [
        ("turn", "episode"),
        ("turn", "turn"),
        ("turn", None),
        (None, None),
    ],
)
def test_non_session_judge_records_are_audit_only(granularity, evaluation_unit):
    feedback = _fb("judge.verification", 1.0)
    if granularity is None:
        del feedback["payload"]["granularity"]
    else:
        feedback["payload"]["granularity"] = granularity
    if evaluation_unit is not None:
        feedback["payload"]["details"]["evaluation_unit"] = evaluation_unit

    assert aggregate_scores([feedback]) == {}


def test_legacy_episode_judgments_do_not_enter_any_ordinary_analytics():
    current = _session_judge_fb(
        "judge.verification",
        0.8,
        review_status="complete",
        config_version="current",
        tags=["current-only"],
        scored_at="2026-07-01T00:00:00",
    )
    legacy = [
        _legacy_episode_judge_fb(
            "judge.verification",
            rating,
            config_version=config,
            tags=["legacy-only"],
            conversation_id=f"legacy-{index}",
            scored_at=scored_at,
        )
        for index, (rating, config, scored_at) in enumerate(
            (
                (1.0, "legacy-old", "2026-07-02T00:00:00"),
                (1.0, "legacy-old", "2026-07-03T00:00:00"),
                (0.0, "legacy-new", "2026-07-10T00:00:00"),
                (0.0, "legacy-new", "2026-07-11T00:00:00"),
            )
        )
    ]
    feedback = [current, *legacy]

    summaries = aggregate_scores(feedback)
    assert len(summaries) == 1
    assert next(iter(summaries.values())).mean == 0.8
    assert [result.config_version for result in ab_leaderboard(feedback)] == ["current"]
    assert detect_regressions(feedback) == []
    assert detect_config_regressions(feedback, min_samples=2) == []
    digest = coaching_digest(feedback)
    assert "current-only" in digest
    assert "legacy-only" not in digest


@pytest.mark.parametrize(
    "context_overrides",
    [
        {"rubric_threshold": -0.1},
        {"rubric_threshold": 1.1},
        {"review_depth": "selective", "second_opinion_margin": None},
        {"review_depth": "selective", "second_opinion_margin": 0.51},
        {
            "review_depth": "primary",
            "second_opinion_margin": 0.1,
            "requested_judge_models": ["judge-a"],
        },
        {
            "review_depth": "primary",
            "second_opinion_margin": None,
            "requested_judge_models": ["judge-a", "judge-b"],
        },
        {
            "review_depth": "full_panel",
            "second_opinion_margin": None,
            "requested_judge_models": ["judge-a", "judge-b"],
        },
        {"requested_judge_models": ["judge-a", "judge-a"]},
        {"requested_judge_models": ["judge-a", " "]},
        {"requested_judge_models": ("judge-a", "judge-b")},
    ],
)
def test_invalid_session_review_policy_context_is_audit_only(context_overrides):
    feedback = _session_judge_fb(
        "judge.session_outcome",
        0.2,
        review_status="complete",
        context_overrides=context_overrides,
    )

    assert aggregate_scores([feedback]) == {}


@pytest.mark.parametrize(
    ("granularity", "evaluation_unit"),
    [("turn", "session"), ("session", "turn")],
)
def test_conflicting_session_markers_are_audit_only(granularity, evaluation_unit):
    feedback = _session_judge_fb(
        "judge.session_outcome",
        0.2,
        review_status="complete",
    )
    feedback["payload"]["granularity"] = granularity
    feedback["payload"]["details"]["evaluation_unit"] = evaluation_unit

    assert aggregate_scores([feedback]) == {}


def test_noncomplete_session_judgments_do_not_enter_ab_or_trend_analysis():
    feedback = [
        _session_judge_fb(
            "judge.session_outcome",
            1.0,
            review_status="complete",
            config_version="v1",
            scored_at="2026-07-01T00:00:00",
        ),
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="unresolved",
            config_version="v2",
            scored_at="2026-07-09T00:00:00",
        ),
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="degraded",
            config_version="v2",
            scored_at="2026-07-10T00:00:00",
        ),
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="unresolved",
            config_version="v2",
            scored_at="2026-07-11T00:00:00",
        ),
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="degraded",
            config_version="v2",
            scored_at="2026-07-12T00:00:00",
        ),
    ]

    assert [item.config_version for item in ab_leaderboard(feedback)] == ["v1"]
    assert detect_regressions(feedback) == []
    assert detect_config_regressions(feedback) == []


# --- ab_leaderboard tests ---


def test_ab_leaderboard_two_versions():
    feedback = [
        _fb("outcome.test", 1.0, config_version="v1"),
        _fb("outcome.test", 1.0, config_version="v1"),
        _fb("outcome.test", 0.0, config_version="v2"),
        _fb("outcome.test", 0.5, config_version="v2"),
    ]
    results = ab_leaderboard(feedback)
    assert len(results) == 2
    by_version = {r.config_version: r for r in results}
    assert by_version["v1"].scores["outcome.test"].mean == 1.0
    assert by_version["v2"].scores["outcome.test"].mean == 0.25
    assert by_version["v1"].evaluated_target_count == 1
    assert by_version["v2"].evaluated_target_count == 1


def test_ab_leaderboard_empty():
    assert ab_leaderboard([]) == []


def test_ab_leaderboard_missing_config():
    feedback = [_fb("outcome.test", 1.0, config_version=None)]
    results = ab_leaderboard(feedback)
    assert len(results) == 1
    assert results[0].config_version == "unknown"


def test_ab_leaderboard_keeps_session_evaluation_contexts_separate():
    feedback = [
        _session_judge_fb(
            "judge.session_outcome",
            1.0,
            review_status="complete",
            config_version="v1",
        ),
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="complete",
            config_version="v1",
            context_overrides={"rubric_version": "3"},
        ),
    ]

    results = ab_leaderboard(feedback)

    assert len(results) == 1
    assert len(results[0].scores) == 2
    assert sorted(summary.mean for summary in results[0].scores.values()) == [0.0, 1.0]


# --- detect_regressions tests ---


def test_detect_regressions_flags_drop():
    feedback = [
        _fb("outcome.test", 1.0, scored_at="2026-07-01T00:00:00"),
        _fb("outcome.test", 1.0, scored_at="2026-07-02T00:00:00"),
        _fb("outcome.test", 0.0, scored_at="2026-07-08T00:00:00"),
        _fb("outcome.test", 0.0, scored_at="2026-07-09T00:00:00"),
    ]
    regressions = detect_regressions(feedback, window_days=7)
    assert len(regressions) >= 1
    assert regressions[0]["scorer"] == "outcome.test"
    assert regressions[0]["direction"] == "regression"


def test_detect_regressions_no_change():
    feedback = [
        _fb("outcome.test", 0.8, scored_at="2026-07-01T00:00:00"),
        _fb("outcome.test", 0.8, scored_at="2026-07-08T00:00:00"),
    ]
    assert detect_regressions(feedback) == []


def test_session_trends_do_not_compare_different_evaluation_contexts():
    older = [
        _session_judge_fb(
            "judge.session_outcome",
            1.0,
            review_status="complete",
            scored_at=f"2026-07-0{day}T00:00:00",
        )
        for day in (1, 2)
    ]
    recent = [
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="complete",
            scored_at=f"2026-07-0{day}T00:00:00",
            context_overrides={"requested_judge_models": ["judge-b", "judge-a", "judge-c"]},
        )
        for day in (8, 9)
    ]

    assert detect_regressions(older + recent) == []


def test_detect_regressions_orders_by_run_time_not_score_time():
    # All scored at the same instant (a single backfill), but the agent ran
    # them at different times. Ordering must follow run time, not score time.
    def _fb_run(scorer, rating, run_time):
        fb = _fb(scorer, rating, scored_at="2026-07-10T00:00:00")
        fb["payload"]["details"]["turn_started_at"] = run_time
        return fb

    # Deliberately out of order: only sorting by run time recovers the trend.
    feedback = [
        _fb_run("outcome.test", 0.0, "2026-07-09T00:00:00"),
        _fb_run("outcome.test", 1.0, "2026-07-01T00:00:00"),
        _fb_run("outcome.test", 0.0, "2026-07-08T00:00:00"),
        _fb_run("outcome.test", 1.0, "2026-07-02T00:00:00"),
    ]
    regressions = detect_regressions(feedback)
    assert len(regressions) == 1
    assert regressions[0]["direction"] == "regression"
    assert regressions[0]["older_mean"] == 1.0
    assert regressions[0]["recent_mean"] == 0.0


# --- detect_config_regressions (A/B safety net) ---


def test_detect_config_regressions_flags_worse_new_config():
    old = [
        _fb("outcome.test", 1.0, config_version="v_old", scored_at="2026-07-01T00:00:00")
        for _ in range(6)
    ]
    new = [
        _fb("outcome.test", 0.0, config_version="v_new", scored_at="2026-07-09T00:00:00")
        for _ in range(6)
    ]
    regs = detect_config_regressions(old + new)
    assert len(regs) == 1
    r = regs[0]
    assert r["scorer"] == "outcome.test"
    assert r["config"] == "v_new" and r["prev_config"] == "v_old"
    assert r["significant"] is True


def test_detect_config_regressions_needs_two_configs():
    fb = [_fb("outcome.test", 1.0, config_version="v1") for _ in range(6)]
    assert detect_config_regressions(fb) == []


def test_detect_config_regressions_skips_tie_in_run_time():
    # both configs stamped with the SAME time (e.g. one backfill) → can't tell
    # which is newer, so don't guess a direction.
    a = [
        _fb("outcome.test", 1.0, config_version="v_a", scored_at="2026-07-01T00:00:00")
        for _ in range(6)
    ]
    b = [
        _fb("outcome.test", 0.0, config_version="v_b", scored_at="2026-07-01T00:00:00")
        for _ in range(6)
    ]
    assert detect_config_regressions(a + b) == []


def test_detect_config_regressions_ignores_improvement():
    old = [
        _fb("outcome.test", 0.0, config_version="v_old", scored_at="2026-07-01T00:00:00")
        for _ in range(6)
    ]
    new = [
        _fb("outcome.test", 1.0, config_version="v_new", scored_at="2026-07-09T00:00:00")
        for _ in range(6)
    ]
    # newer config is better — not a regression
    assert detect_config_regressions(old + new) == []


def test_config_regression_does_not_compare_different_session_evaluation_contexts():
    old = [
        _session_judge_fb(
            "judge.session_outcome",
            1.0,
            review_status="complete",
            config_version="v_old",
            scored_at="2026-07-01T00:00:00",
        )
        for _ in range(6)
    ]
    new = [
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="complete",
            config_version="v_new",
            scored_at="2026-07-09T00:00:00",
            context_overrides={
                "review_depth": "full_panel",
                "second_opinion_margin": None,
            },
        )
        for _ in range(6)
    ]

    assert detect_config_regressions(old + new) == []


def test_config_regression_compares_latest_two_configs_within_evaluation_context():
    first_context_old = [
        _session_judge_fb(
            "judge.session_outcome",
            1.0,
            review_status="complete",
            config_version="v1",
            scored_at="2026-07-01T00:00:00",
        )
        for _ in range(6)
    ]
    other_context_middle = [
        _session_judge_fb(
            "judge.session_outcome",
            1.0,
            review_status="complete",
            config_version="v2",
            scored_at="2026-07-05T00:00:00",
            context_overrides={"rubric_version": "3"},
        )
        for _ in range(6)
    ]
    first_context_new = [
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="complete",
            config_version="v3",
            scored_at="2026-07-09T00:00:00",
        )
        for _ in range(6)
    ]

    regressions = detect_config_regressions(
        first_context_old + other_context_middle + first_context_new
    )

    assert len(regressions) == 1
    assert regressions[0]["config"] == "v3"
    assert regressions[0]["prev_config"] == "v1"


# --- coaching_digest tests ---


def test_coaching_digest_structure():
    feedback = [
        _fb("outcome.test", 1.0, tags=["pass"]),
        _fb("outcome.test", 0.0, tags=["fail", "timeout"]),
        _session_judge_fb("judge.verification", 0.8, review_status="complete", tags=["verified"]),
        _session_judge_fb(
            "judge.verification", 0.2, review_status="complete", tags=["no_verification"]
        ),
    ]
    digest = coaching_digest(feedback)
    assert "Summary" in digest
    assert "outcome.test" in digest
    assert "judge.verification" in digest


def test_coaching_digest_empty():
    digest = coaching_digest([])
    assert "No scores" in digest


def test_coaching_digest_includes_bounded_low_score_behavioral_feedback():
    digest = coaching_digest(
        [
            _session_judge_fb(
                "judge.verification",
                0.25,
                review_status="complete",
                conversation_id="session-1",
                behavioral_feedback=[
                    {
                        "success": "It changed approach after the failure.",
                        "problem": "Completion was claimed before the final check.",
                        "desired_behavior": "Run relevant checks after the final change.",
                    }
                ],
                evidence_ids=["trace-7"],
            ),
        ]
    )

    assert "## Behavioral feedback" in digest
    assert "Completion was claimed before the final check." in digest
    assert "Run relevant checks after the final change." in digest
    assert "It changed approach after the failure." in digest
    assert "session-1" in digest
    assert "trace-7" in digest


def test_behavioral_feedback_is_complete_low_scoring_bounded_and_deterministic():
    examples = []
    for index, rating in enumerate((0.4, 0.1, 0.1, 0.2, 0.0), start=1):
        examples.append(
            _session_judge_fb(
                "judge.verification",
                rating,
                review_status="complete",
                conversation_id=f"session-{index}",
                scored_at=f"2026-07-0{index}T00:00:00",
                behavioral_feedback=[
                    {
                        "success": None,
                        "problem": f"problem-{index} " + "x" * 1000,
                        "desired_behavior": f"desired-{index} " + "y" * 1000,
                    }
                ],
                evidence_ids=[f"trace-{index}", "z" * 1000],
            )
        )
    examples.extend(
        [
            _session_judge_fb(
                "judge.verification",
                0.0,
                review_status=status,
                conversation_id=f"excluded-{status}",
                behavioral_feedback=[
                    {
                        "success": None,
                        "problem": f"excluded {status}",
                        "desired_behavior": "excluded desired",
                    }
                ],
            )
            for status in ("unresolved", "degraded")
        ]
    )
    examples.append(
        _session_judge_fb(
            "judge.verification",
            0.6,
            review_status="complete",
            conversation_id="above-pinned-threshold",
            behavioral_feedback=[
                {
                    "success": None,
                    "problem": "high score problem",
                    "desired_behavior": "high score desired",
                }
            ],
        )
    )
    digest = coaching_digest(examples)

    assert digest.index("session-5") < digest.index("session-2") < digest.index("session-3")
    assert "session-4" not in digest
    assert "session-1" not in digest
    assert "excluded unresolved" not in digest
    assert "excluded degraded" not in digest
    assert "above-pinned-threshold" not in digest
    assert "x" * 500 not in digest
    assert "z" * 500 not in digest


def test_behavioral_feedback_uses_only_merged_fields_not_raw_or_findings():
    feedback = _session_judge_fb(
        "judge.verification",
        0.2,
        review_status="complete",
        behavioral_feedback=[
            {
                "success": "   ",
                "problem": "  missed   verification  ",
                "desired_behavior": " rerun   checks ",
            },
            {
                "success": None,
                "problem": "missed verification",
                "desired_behavior": "rerun checks",
            },
        ],
        evidence_ids=["trace-1"],
    )
    feedback["payload"]["reason"] = "RAW CONVERSATION SECRET"
    feedback["payload"]["details"]["attempts"] = [{"steps": [{"findings": "RAW WINDOW FINDING"}]}]

    digest = coaching_digest([feedback])

    assert "Problem: missed verification" in digest
    assert "Desired behavior: rerun checks" in digest
    assert digest.count("Problem: missed verification") == 1
    assert "Success:" not in digest
    assert "RAW CONVERSATION SECRET" not in digest
    assert "RAW WINDOW FINDING" not in digest


def test_behavioral_feedback_caps_and_authenticates_reviewer_items():
    huge_feedback = [
        {
            "success": None,
            "problem": "ignored malformed",
            "desired_behavior": "ignored malformed desired",
            "raw_window": "must not be accepted",
        },
        {
            "success": None,
            "problem": "reviewer two problem",
            "desired_behavior": "reviewer two desired",
        },
        {
            "success": None,
            "problem": "reviewer three problem",
            "desired_behavior": "reviewer three desired",
        },
    ] + [
        {
            "success": None,
            "problem": f"excess problem {index}",
            "desired_behavior": f"excess desired {index}",
        }
        for index in range(10_000)
    ]
    feedback = _session_judge_fb(
        "judge.verification",
        0.2,
        review_status="complete",
        behavioral_feedback=huge_feedback,
    )

    digest = coaching_digest([feedback])

    assert "ignored malformed" not in digest
    assert "reviewer two problem" in digest
    assert "reviewer three problem" in digest
    assert "excess problem" not in digest


def test_behavioral_feedback_orders_offset_timestamps_chronologically_with_typed_fallbacks():
    def example(session_id, started_at):
        feedback = _session_judge_fb(
            "judge.verification",
            0.1,
            review_status="complete",
            conversation_id=session_id,
            behavioral_feedback=[
                {
                    "success": None,
                    "problem": f"problem {session_id}",
                    "desired_behavior": f"desired {session_id}",
                }
            ],
        )
        feedback["payload"]["details"]["turn_started_at"] = started_at
        return feedback

    digest = coaching_digest(
        [
            example("invalid-dict", {"when": "later"}),
            example("later-offset", "2026-07-15T01:00:00+00:00"),
            example("earlier-offset", "2026-07-14T20:00:00-04:00"),
            example("invalid-int", 7),
        ]
    )

    assert digest.index("earlier-offset") < digest.index("later-offset")
    assert "invalid-dict" in digest
    assert "invalid-int" not in digest


def test_coaching_digest_does_not_merge_session_evaluation_contexts():
    digest = coaching_digest(
        [
            _session_judge_fb(
                "judge.session_outcome",
                1.0,
                review_status="complete",
            ),
            _session_judge_fb(
                "judge.session_outcome",
                0.0,
                review_status="complete",
                context_overrides={"rubric_threshold": 0.6},
            ),
        ]
    )

    session_lines = [line for line in digest.splitlines() if "judge.session_outcome" in line]
    assert len(session_lines) == 2
    assert all("n=1" in line for line in session_lines)
    assert all("mean=0.50" not in line for line in session_lines)


# --- pass_rate / robustness tests ---


def test_pass_rate_binary_scorer():
    s = ScoreSummary(
        scorer="outcome.test", count=4, mean=0.75, min_val=0.0, max_val=1.0, binary=True
    )
    assert s.pass_rate == 0.75


def test_pass_rate_continuous_scorer_with_extreme_values():
    # A continuous scorer whose observed values happen to span 0.0..1.0 must
    # NOT be reported as a pass rate.
    s = ScoreSummary(
        scorer="judge.verification",
        count=3,
        mean=0.5,
        min_val=0.0,
        max_val=1.0,
        binary=False,
    )
    assert s.pass_rate is None


def test_aggregate_marks_continuous_scorer_non_binary():
    feedback = [
        _session_judge_fb("judge.verification", 0.0, review_status="complete"),
        _session_judge_fb("judge.verification", 0.5, review_status="complete"),
        _session_judge_fb("judge.verification", 1.0, review_status="complete"),
    ]
    result = aggregate_scores(feedback)
    summary = next(iter(result.values()))
    assert summary.binary is False
    assert summary.pass_rate is None


def test_aggregate_marks_binary_scorer():
    feedback = [_fb("outcome.test", 1.0), _fb("outcome.test", 0.0)]
    result = aggregate_scores(feedback)
    assert result["outcome.test"].binary is True


def test_aggregate_handles_null_payload():
    feedback = [
        {"feedback_type": "weave_agent_signals.outcome.test", "payload": None},
        _fb("outcome.test", 1.0),
    ]
    result = aggregate_scores(feedback)
    assert result["outcome.test"].count == 1


# --- statistical significance ---


def test_single_sample_pass_rate_has_wide_ci():
    # "100% pass (n=1)" must not read as certain — the Wilson CI should be wide.
    feedback = [_fb("outcome.test", 1.0)]
    s = aggregate_scores(feedback)["outcome.test"]
    lo, hi = s.ci
    assert lo < 0.3  # huge uncertainty on the low side
    assert hi == 1.0
    assert s.confident is False


def test_large_sample_is_confident_with_tight_ci():
    feedback = [_fb("outcome.test", 1.0) for _ in range(40)]
    feedback += [_fb("outcome.test", 0.0) for _ in range(40)]
    s = aggregate_scores(feedback)["outcome.test"]
    lo, hi = s.ci
    assert s.confident is True
    assert (hi - lo) < 0.25  # 80 samples → reasonably tight


def test_continuous_scorer_ci_from_spread():
    feedback = [_fb("efficiency", v) for v in [0.9, 0.9, 0.91, 0.89, 0.9, 0.9]]
    s = aggregate_scores(feedback)["efficiency"]
    lo, hi = s.ci
    assert lo <= s.mean <= hi
    assert hi - lo < 0.1  # low variance → tight interval


def test_regression_marked_significant_only_with_evidence():
    # Clean, large drop → significant.
    big = (
        [_fb("outcome.test", 1.0, scored_at=f"2026-07-0{i}T00:00:00") for i in range(1, 5)]
        + [_fb("outcome.test", 0.0, scored_at=f"2026-07-0{i}T00:00:00") for i in range(5, 9)]
        + [_fb("outcome.test", 1.0, scored_at=f"2026-07-0{i}T00:00:00") for i in range(1, 5)]
        + [_fb("outcome.test", 0.0, scored_at=f"2026-07-0{i}T00:00:00") for i in range(5, 9)]
    )
    regs = detect_regressions(big, min_samples=4)
    assert len(regs) == 1
    assert regs[0]["significant"] is True


def test_tiny_regression_flagged_but_not_significant():
    feedback = [
        _fb("outcome.test", 1.0, scored_at="2026-07-01T00:00:00"),
        _fb("outcome.test", 1.0, scored_at="2026-07-02T00:00:00"),
        _fb("outcome.test", 0.0, scored_at="2026-07-08T00:00:00"),
        _fb("outcome.test", 0.0, scored_at="2026-07-09T00:00:00"),
    ]
    regs = detect_regressions(feedback)
    assert len(regs) == 1  # still surfaced for the human
    assert regs[0]["significant"] is False  # but flagged as low-confidence
