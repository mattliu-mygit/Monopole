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


def test_ab_leaderboard_empty():
    assert ab_leaderboard([]) == []


def test_ab_leaderboard_missing_config():
    feedback = [_fb("outcome.test", 1.0, config_version=None)]
    results = ab_leaderboard(feedback)
    assert len(results) == 1
    assert results[0].config_version == "unknown"


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


# --- coaching_digest tests ---


def test_coaching_digest_structure():
    feedback = [
        _fb("outcome.test", 1.0, tags=["pass"]),
        _fb("outcome.test", 0.0, tags=["fail", "timeout"]),
        _fb("judge.verification", 0.8, tags=["verified"]),
        _fb("judge.verification", 0.2, tags=["no_verification"]),
    ]
    digest = coaching_digest(feedback)
    assert "Summary" in digest
    assert "outcome.test" in digest
    assert "judge.verification" in digest


def test_coaching_digest_empty():
    digest = coaching_digest([])
    assert "No scores" in digest


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
        _fb("judge.verification", 0.0),
        _fb("judge.verification", 0.5),
        _fb("judge.verification", 1.0),
    ]
    result = aggregate_scores(feedback)
    assert result["judge.verification"].binary is False
    assert result["judge.verification"].pass_rate is None


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
