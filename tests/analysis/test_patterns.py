"""Tests for pattern analysis: summaries, A/B leaderboards, config regressions, trends, coaching."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.patterns import (
    ConfigStats,
    ScoreSummary,
    Stat,
    _ci,
    _stat,
    _summarize,
    _wilson_ci,
    ab_leaderboard,
    aggregate_scores,
    agreement,
    coaching_digest,
    config_latest,
    config_regressions,
    detect_regressions,
    difference_ci,
    is_evaluation_feedback_eligible,
    pass_threshold,
    segment_gaps,
    stat_leaderboard,
    stat_summaries,
    stats_from_feedback,
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
        "target_id": f"weave:///e/p/agent_turn/{conversation_id}",
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
            "panel_contract_version": "1",
            "requested_judge_models": ["judge-a", "judge-b", "judge-c"],
            "panel_size": 3,
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


def test_evaluation_feedback_eligibility_is_shared_by_deterministic_and_judge_scores():
    deterministic = _fb("outcome.test", 0.75)
    current_judge = _session_judge_fb(
        "judge.session_outcome",
        0.25,
        review_status="complete",
    )
    invalid_deterministic = _fb("outcome.test", float("nan"))
    legacy_judge = _legacy_episode_judge_fb("judge.session_outcome", 1.0)

    assert is_evaluation_feedback_eligible(deterministic)
    assert is_evaluation_feedback_eligible(current_judge)
    assert not is_evaluation_feedback_eligible(invalid_deterministic)
    assert not is_evaluation_feedback_eligible(legacy_judge)


@pytest.mark.parametrize(
    "feedback",
    [
        {"feedback_type": 7, "payload": {"rating": 1.0}},
        {"feedback_type": "weave_agent_signals.outcome.test", "payload": "invalid"},
        {
            "feedback_type": "weave_agent_signals.judge.session_outcome",
            "payload": {"rating": 1.0, "details": "invalid"},
        },
    ],
)
def test_evaluation_feedback_eligibility_rejects_malformed_records(feedback):
    assert not is_evaluation_feedback_eligible(feedback)


def test_population_summary_excludes_noncomplete_session_judgments():
    result = aggregate_scores(
        [
            _session_judge_fb("judge.session_outcome", 0.8, review_status="complete"),
            _session_judge_fb("judge.session_outcome", 0.2, review_status="failed"),
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
        "judge.session_outcome [rubric v2, threshold 0.5; panel 1; "
        "judges judge-a → judge-b → judge-c]"
    ]


def test_incomplete_session_judge_context_is_audit_only():
    incomplete = _session_judge_fb(
        "judge.session_outcome",
        1.0,
        review_status="complete",
    )
    del incomplete["payload"]["details"]["panel_contract_version"]
    canonical = _session_judge_fb(
        "judge.session_outcome",
        0.0,
        review_status="complete",
    )

    result = aggregate_scores([incomplete, canonical])

    assert len(result) == 1
    assert next(iter(result.values())).mean == 0.0


def test_degraded_session_judgment_is_audit_only():
    degraded = _session_judge_fb(
        "judge.session_outcome",
        0.75,
        review_status="degraded",
    )

    assert aggregate_scores([degraded]) == {}


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
    assert set(config_latest(feedback)) == {"current"}
    digest = coaching_digest(feedback)
    assert "current-only" in digest
    assert "legacy-only" not in digest


@pytest.mark.parametrize(
    "context_overrides",
    [
        {"rubric_threshold": -0.1},
        {"rubric_threshold": 1.1},
        {"panel_contract_version": ""},
        {"panel_contract_version": "2"},
        {"panel_size": 2},
        {"panel_size": True},
        {"requested_judge_models": ["judge-a", "judge-b", "judge-c", "judge-d"], "panel_size": 4},
        {"requested_judge_models": ["judge-a", "judge-a"]},
        {"requested_judge_models": ["judge-a", " "]},
        {"requested_judge_models": ("judge-a", "judge-b")},
    ],
)
def test_invalid_session_evaluation_context_is_audit_only(context_overrides):
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
            review_status="failed",
            config_version="v2",
            scored_at="2026-07-09T00:00:00",
        ),
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="failed",
            config_version="v2",
            scored_at="2026-07-10T00:00:00",
        ),
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="failed",
            config_version="v2",
            scored_at="2026-07-11T00:00:00",
        ),
        _session_judge_fb(
            "judge.session_outcome",
            0.0,
            review_status="failed",
            config_version="v2",
            scored_at="2026-07-12T00:00:00",
        ),
    ]

    assert [item.config_version for item in ab_leaderboard(feedback)] == ["v1"]
    assert detect_regressions(feedback) == []
    assert set(config_latest(feedback)) == {"v1"}


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


# --- config_regressions (A/B safety net) ---

_DAYS = {f"v{day}": datetime(2026, 7, day, tzinfo=timezone.utc) for day in range(1, 10)}


def _stats(by_config, targets=None):
    return ConfigStats(by_config, targets or {})


def test_config_regressions_flags_worse_new_config():
    stats = _stats(
        {
            "v1": {"outcome.test": Stat(6, 1.0, True)},
            "v9": {"outcome.test": Stat(6, 0.0, True)},
        }
    )

    regs = config_regressions(stats, _DAYS)

    assert len(regs) == 1
    r = regs[0]
    assert r["scorer"] == "outcome.test"
    assert r["config"] == "v9" and r["prev_config"] == "v1"
    assert (r["sample_count"], r["prev_sample_count"]) == (6, 6)
    assert r["difference_ci"][1] < 0
    assert r["significant"] is True


def test_config_regressions_flag_a_drop_whose_own_intervals_overlap():
    """Behavior change: 40/50 then 30/50 passing. The two Wilson intervals overlap
    (0.67-0.89 and 0.46-0.72), which the old test called not significant, but the interval
    on the difference, -0.36 to -0.02, excludes 0."""
    prev, new = Stat(50, 0.8, True), Stat(50, 0.6, True)
    stats = _stats({"v1": {"outcome.test": prev}, "v9": {"outcome.test": new}})

    (reg,) = config_regressions(stats, _DAYS)

    assert _ci(new)[1] > _ci(prev)[0]
    assert reg["difference_ci"] == (-0.364, -0.02)
    assert reg["significant"] is True


def test_config_regressions_keep_a_small_sample_drop_tentative():
    stats = _stats(
        {"v1": {"outcome.test": Stat(2, 1.0, True)}, "v9": {"outcome.test": Stat(2, 0.5, True)}}
    )

    (reg,) = config_regressions(stats, _DAYS)

    assert reg["difference_ci"][1] > 0
    assert reg["significant"] is False


@pytest.mark.parametrize("stddev", [None, 0.0])
def test_config_regressions_need_a_spread_for_a_continuous_scorer(stddev):
    # One rating has no spread, even when CES reports 0 for it.
    stats = _stats(
        {
            "v1": {"efficiency": Stat(1, 0.9, False, stddev)},
            "v9": {"efficiency": Stat(9, 0.1, False, 0.1)},
        }
    )

    (reg,) = config_regressions(stats, _DAYS)

    assert (reg["difference_ci"], reg["significant"]) == (None, False)


def test_config_regressions_need_two_ordered_configs():
    stats = _stats(
        {"v1": {"outcome.test": Stat(6, 1.0, True)}, "v9": {"outcome.test": Stat(6, 0.0, True)}}
    )

    assert config_regressions(_stats({"v1": stats.by_config["v1"]}), _DAYS) == []
    # A config with no run time among the newest turns can't be ordered.
    assert config_regressions(stats, {"v9": _DAYS["v9"]}) == []
    unknown = _stats({**stats.by_config, "unknown": {"outcome.test": Stat(6, 1.0, True)}})
    assert config_regressions(unknown, {"unknown": _DAYS["v1"], "v9": _DAYS["v9"]}) == []


def test_config_regressions_skip_a_tie_in_run_time():
    # both configs stamped with the SAME time (e.g. one backfill) → can't tell
    # which is newer, so don't guess a direction.
    stats = _stats(
        {"v_a": {"outcome.test": Stat(6, 1.0, True)}, "v_b": {"outcome.test": Stat(6, 0.0, True)}}
    )
    tie = datetime(2026, 7, 1, tzinfo=timezone.utc)

    assert config_regressions(stats, {"v_a": tie, "v_b": tie}) == []


def test_config_regressions_ignore_improvement():
    stats = _stats(
        {"v1": {"outcome.test": Stat(6, 0.0, True)}, "v9": {"outcome.test": Stat(6, 1.0, True)}}
    )

    assert config_regressions(stats, _DAYS) == []


def test_config_regressions_compare_each_scorers_latest_two_configs():
    # A panel Scorer that only v1 and v3 have skips v2, which scored a different panel.
    panel_a, panel_b = "judge.session_outcome.panel-a", "judge.session_outcome.panel-b"
    stats = _stats(
        {
            "v1": {panel_a: Stat(6, 0.9, False, 0.05)},
            "v2": {panel_b: Stat(6, 0.9, False, 0.05)},
            "v3": {panel_a: Stat(6, 0.1, False, 0.05)},
        }
    )

    (reg,) = config_regressions(stats, _DAYS)

    assert (reg["scorer"], reg["config"], reg["prev_config"]) == (panel_a, "v3", "v1")
    assert reg["significant"] is True


def test_config_latest_orders_configs_by_agent_run_time():
    feedback = [
        _fb("outcome.test", 1.0, config_version="v_old", scored_at="2026-07-01T00:00:00"),
        _fb("outcome.test", 1.0, config_version="v_old", scored_at="2026-07-03T00:00:00"),
        _fb("outcome.test", 0.0, config_version="v_new", scored_at="2026-07-09T00:00:00"),
        _fb("outcome.test", 0.0, config_version="v_bad", scored_at="not a time"),
    ]

    assert config_latest(feedback) == {
        "v_old": datetime(2026, 7, 3, tzinfo=timezone.utc),
        "v_new": datetime(2026, 7, 9, tzinfo=timezone.utc),
    }


# --- difference CI and query stats ---


@pytest.mark.parametrize(
    ("new", "prev", "want"),
    [
        # Newcombe (1998), Statistics in Medicine 17:873, Table II, method 10.
        ((56, 70), (48, 80), (0.0524, 0.3339)),
        ((9, 10), (3, 10), (0.1705, 0.8090)),
    ],
)
def test_difference_ci_matches_newcombes_published_intervals(new, prev, want):
    got = difference_ci(Stat(new[1], new[0] / new[1], True), Stat(prev[1], prev[0] / prev[1], True))

    assert got == pytest.approx(want, abs=1e-4)


def test_difference_ci_uses_welchs_standard_error_for_continuous_scorers():
    # se = sqrt(0.2^2/50 + 0.3^2/40) = 0.05523; -0.1 +- 1.96 * se
    got = difference_ci(Stat(50, 0.5, False, 0.2), Stat(40, 0.6, False, 0.3))

    assert got == pytest.approx((-0.20824, 0.00824), abs=1e-5)


def test_stat_summaries_pool_configs_exactly():
    values = {"v1": [0.2, 0.4, 0.9], "v2": [0.5, 0.7]}
    stats = _stats(
        {config: {"efficiency": _stat(ratings, binary=False)} for config, ratings in values.items()}
    )

    pooled = stat_summaries(stats)["efficiency"]
    direct = _summarize("efficiency", values["v1"] + values["v2"], [])

    assert (pooled.count, pooled.binary) == (5, False)
    assert pooled.mean == pytest.approx(direct.mean)
    assert pooled.ci == pytest.approx(direct.ci)


def test_stat_summaries_give_binary_scorers_a_pass_rate_and_wilson_interval():
    stats = _stats(
        {"v1": {"outcome.test": Stat(3, 2 / 3, True)}, "v2": {"outcome.test": Stat(1, 1.0, True)}}
    )

    summary = stat_summaries(stats)["outcome.test"]

    assert (summary.count, summary.pass_rate) == (4, 0.75)
    assert summary.ci == pytest.approx(_wilson_ci(3, 4))


def test_stat_summaries_count_tags_from_the_raw_score_records():
    def record(scorer, ces_scorer, tags):
        feedback = _fb(scorer, 1.0, tags=tags)
        feedback["payload"]["details"]["ces_scorer"] = ces_scorer
        return feedback

    panel = "judge.verification.panel-abc"
    stats = _stats({"v1": {"outcome.test": Stat(40, 0.5, True), panel: Stat(3, 0.5, False)}})
    judged = _session_judge_fb("judge.verification", 0.2, review_status="complete", tags=["x"])
    judged["payload"]["details"]["ces_scorer"] = panel
    feedback = [
        record("outcome.test", "outcome.test", ["test_pass"]),
        record("outcome.test", "outcome.test", ["test_failure"]),
        record("outcome.test", "outcome.test", ["test_failure"]),
        judged,
    ]

    summaries = stat_summaries(stats, feedback)

    assert summaries["outcome.test"].count == 40  # from the query, not the 3 records
    assert summaries["outcome.test"].tag_counts == {"test_pass": 1, "test_failure": 2}
    assert summaries[panel].tag_counts == {"x": 1}
    assert stat_summaries(stats)["outcome.test"].tag_counts == {}
    assert "Tags: test_failure=2, test_pass=1" in coaching_digest(feedback, stats)


def test_stat_leaderboard_reports_each_config_and_its_scored_targets():
    stats = _stats(
        {
            "v2": {"outcome.test": Stat(4, 0.5, True)},
            "v1": {"efficiency": Stat(3, 0.4, False, 0.1)},
        },
        {"v1": 5, "v2": 4},
    )

    board = stat_leaderboard(stats)

    assert [(r.config_version, r.evaluated_target_count) for r in board] == [("v1", 5), ("v2", 4)]
    assert board[1].scores["outcome.test"].pass_rate == 0.5


def test_stats_from_feedback_keys_continuous_stats_by_ces_scorer_and_config():
    def record(rating, config):
        feedback = _fb("efficiency", rating, config_version=config)
        feedback["payload"]["details"]["ces_scorer"] = "efficiency"
        return feedback

    found = stats_from_feedback([record(0.0, "v1"), record(1.0, "v1"), record(0.5, "v2")])

    assert found["v1"]["efficiency"] == Stat(2, 0.5, False, pytest.approx(0.7071, abs=1e-4))
    assert found["v2"]["efficiency"] == Stat(1, 0.5, False)


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
    s = ScoreSummary(scorer="outcome.test", count=4, mean=0.75, binary=True)
    assert s.pass_rate == 0.75


def test_pass_rate_continuous_scorer_with_extreme_values():
    # A continuous scorer whose observed values happen to span 0.0..1.0 must
    # NOT be reported as a pass rate.
    s = ScoreSummary(
        scorer="judge.verification",
        count=3,
        mean=0.5,
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
    # 1 of 2 passing, then 0 of 2: Newcombe's interval on -0.5 is about -0.91 to 0.27.
    feedback = [
        _fb("outcome.test", 1.0, scored_at="2026-07-01T00:00:00"),
        _fb("outcome.test", 0.0, scored_at="2026-07-02T00:00:00"),
        _fb("outcome.test", 0.0, scored_at="2026-07-08T00:00:00"),
        _fb("outcome.test", 0.0, scored_at="2026-07-09T00:00:00"),
    ]
    regs = detect_regressions(feedback)
    assert len(regs) == 1  # still surfaced for the human
    assert regs[0]["significant"] is False  # but flagged as low-confidence


def _window(scorer, ratings, day):
    return [_fb(scorer, rating, scored_at=f"2026-{day}T00:00:00") for rating in ratings]


def test_trend_is_significant_when_the_difference_interval_excludes_0():
    """40 of 50 passing, then 30 of 50. The windows' own Wilson intervals overlap (0.67-0.89
    and 0.46-0.72), which the old rule called not significant; the difference interval,
    -0.36 to -0.02, excludes 0."""
    feedback = _window("outcome.test", [1.0] * 40 + [0.0] * 10, "06-01") + _window(
        "outcome.test", [1.0] * 30 + [0.0] * 20, "07-01"
    )

    (reg,) = detect_regressions(feedback, threshold=0.1)

    assert (reg["older_mean"], reg["recent_mean"], reg["delta"]) == (0.8, 0.6, -0.2)
    assert reg["significant"] is True


def test_continuous_trends_use_welchs_interval():
    # Means 0.85 then 0.6, standard deviations 0.053 and 0.105: -0.25 +- 0.073.
    steady = _window("efficiency", [0.8, 0.9] * 5, "06-01") + _window(
        "efficiency", [0.5, 0.7] * 5, "07-01"
    )
    # The same drop with a recent spread of 0.52: -0.25 +- 0.32 includes 0. Values of only 0
    # and 1 don't make a continuous Scorer binary.
    noisy = _window("efficiency", [0.8, 0.9] * 5, "06-01") + _window(
        "efficiency", [0.0, 1.0, 1.0] * 3 + [0.0], "07-01"
    )

    (steady_reg,) = detect_regressions(steady)
    (noisy_reg,) = detect_regressions(noisy)

    assert (steady_reg["delta"], steady_reg["significant"]) == (-0.25, True)
    assert noisy_reg["significant"] is False


def test_agreement_counts_matching_pass_fail_and_the_judge_gap_from_exact_pairs():
    joint = {(True, 0.75): 6, (True, 0.25): 2, (False, 0.25): 3, (False, 0.75): 1}

    got = agreement("judge.verification.panel-abc", "implicit.completion", joint)

    assert (got.count, got.agree) == (12, 9)
    # Wilson's interval for 9 of 12.
    assert got.ci == pytest.approx((0.4677, 0.9111), abs=1e-4)
    assert got.passing == Stat(8, 0.625, False, pytest.approx(0.2315, abs=1e-4))
    assert got.failing == Stat(4, 0.375, False, 0.25)
    # Welch: 0.25 ± 1.96 × sqrt(0.05357/8 + 0.0625/4).
    assert got.gap_ci == pytest.approx((-0.0428, 0.5428), abs=1e-4)


def test_agreement_passes_a_rating_at_the_rubric_threshold(monkeypatch):
    rubric = SESSION_RUBRICS["judge.verification"]
    monkeypatch.setitem(SESSION_RUBRICS, "judge.verification", replace(rubric, threshold=0.75))

    # 0.75 passes the 0.75 threshold; 0.5 fails it, though it passes the other panel's 0.5.
    got = agreement(
        "judge.verification.panel-a",
        "judge.retired.panel-b",
        {(0.5, 0.75): 2, (0.5, 0.5): 1, (0.25, 0.5): 4},
    )

    assert pass_threshold("judge.verification.panel-a") == 0.75
    assert pass_threshold("judge.retired.panel-b") == 0.5
    assert (got.count, got.agree) == (7, 6)


def test_agreement_without_a_failing_reference_has_no_gap():
    got = agreement("judge.verification.panel-abc", "implicit.completion", {(True, 1.0): 3})

    assert got.failing is None and got.gap_ci is None
    assert (got.count, got.agree) == (3, 3)


def test_segment_gaps_compare_each_segment_with_the_rest_of_its_config():
    completion = "implicit.completion"
    segments = {
        ("v1", "main"): {
            completion: Stat(70, 56 / 70, True),
            "efficiency.session": Stat(20, 0.5, False),
        },
        ("v1", "dev"): {
            completion: Stat(80, 48 / 80, True),
            "efficiency.session": Stat(20, 0.4, False),
        },
        # Too few sessions on one side or the other: not shown.
        ("v2", "main"): {completion: Stat(9, 1.0, True)},
        ("v2", None): {completion: Stat(50, 0.5, True)},
        ("v3", "only"): {completion: Stat(50, 0.5, True)},
    }

    gaps = {(g["config"], g["segment"], g["scorer"]): g for g in segment_gaps(segments)}

    assert set(gaps) == {
        ("v1", "main", completion),
        ("v1", "dev", completion),
        ("v1", "main", "efficiency.session"),
        ("v1", "dev", "efficiency.session"),
    }
    main = gaps[("v1", "main", completion)]
    # Newcombe's published interval for 56/70 − 48/80.
    assert main["difference_ci"] == (0.052, 0.334)
    assert main["significant"] and main["rest_sample_count"] == 80
    assert gaps[("v1", "dev", completion)]["difference_ci"] == (-0.334, -0.052)
    # Without a stddev from CES, a continuous Scorer has a gap but no interval.
    efficiency = gaps[("v1", "main", "efficiency.session")]
    assert efficiency["delta"] == 0.1 and efficiency["difference_ci"] is None
    assert not efficiency["significant"]


def test_segment_gaps_pool_the_other_segments_as_the_rest():
    completion = "implicit.completion"
    segments = {
        ("v1", "a"): {completion: Stat(10, 1.0, True)},
        ("v1", "b"): {completion: Stat(10, 0.5, True)},
        ("v1", "c"): {completion: Stat(30, 0.5, True)},
    }

    (a,) = [g for g in segment_gaps(segments) if g["segment"] == "a"]

    assert (a["rest_mean"], a["rest_sample_count"]) == (0.5, 40)
