"""Pattern analysis: A/B leaderboards, trend detection, coaching digest.

Operates on feedback records queried from Weave. All analysis is pure
computation — no API calls. The CLI orchestrates query → analyze → report.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from weave_agent_signals.models import FEEDBACK_PREFIX

# Below this many samples a scorer's mean/pass-rate is not trustworthy on its own.
MIN_CONFIDENT_SAMPLES = 5
_Z_95 = 1.96  # z for a 95% interval


def _wilson_ci(k: float, n: int, z: float = _Z_95) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials.

    Preferred over the normal approximation for small n and proportions near
    0 or 1 (where the normal interval breaks down or escapes [0, 1]).
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _mean_ci(values: list[float], z: float = _Z_95) -> tuple[float, float]:
    """95% CI for a mean via the standard error (normal approximation)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return (max(0.0, mean), min(1.0, mean))
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return (max(0.0, mean - z * se), min(1.0, mean + z * se))


@dataclass
class ScoreSummary:
    scorer: str
    count: int
    mean: float
    min_val: float
    max_val: float
    tag_counts: dict[str, int] = field(default_factory=dict)
    binary: bool = False
    ci: tuple[float, float] = (0.0, 1.0)

    @property
    def pass_rate(self) -> float | None:
        # Only binary scorers (every observed value is exactly 0.0 or 1.0) have a
        # meaningful pass rate. A continuous scorer whose values merely span the
        # 0..1 range must be reported as a mean, not a pass percentage.
        return self.mean if self.binary else None

    @property
    def confident(self) -> bool:
        return self.count >= MIN_CONFIDENT_SAMPLES


@dataclass
class ABResult:
    config_version: str
    turn_count: int
    scores: dict[str, ScoreSummary]


def _extract_scorer(feedback: dict) -> str | None:
    ftype = feedback.get("feedback_type", "")
    if ftype.startswith(FEEDBACK_PREFIX):
        return ftype[len(FEEDBACK_PREFIX) :]
    return None


def _payload(feedback: dict) -> dict:
    # `.get("payload", {})` is not enough: the API can return an explicit null.
    return feedback.get("payload") or {}


def _details(feedback: dict) -> dict:
    return _payload(feedback).get("details") or {}


def _extract_rating(feedback: dict) -> float | None:
    rating = _payload(feedback).get("rating")
    if rating is not None:
        try:
            return float(rating)
        except (TypeError, ValueError):
            return None
    return None


def _extract_config(feedback: dict) -> str:
    return _details(feedback).get("config_version") or "unknown"


def _extract_tags(feedback: dict) -> list[str]:
    return _payload(feedback).get("tags") or []


def _extract_run_time(feedback: dict) -> str:
    # Order trends by when the agent ran, not when we scored it — a single
    # backfill stamps every score with the same scored_at, which would make
    # chronological comparison meaningless.
    details = _details(feedback)
    return (
        details.get("turn_started_at")
        or details.get("scored_at")
        or _payload(feedback).get("scored_at")
        or ""
    )


def _summarize(scorer: str, ratings: list[float], tags: list[list[str]]) -> ScoreSummary:
    tag_counts: dict[str, int] = defaultdict(int)
    for tag_list in tags:
        for tag in tag_list:
            tag_counts[tag] += 1

    binary = all(r in (0.0, 1.0) for r in ratings)
    ci = _wilson_ci(sum(ratings), len(ratings)) if binary else _mean_ci(ratings)

    return ScoreSummary(
        scorer=scorer,
        count=len(ratings),
        mean=sum(ratings) / len(ratings),
        min_val=min(ratings),
        max_val=max(ratings),
        tag_counts=dict(tag_counts),
        binary=binary,
        ci=ci,
    )


def aggregate_scores(feedback: list[dict]) -> dict[str, ScoreSummary]:
    """Group feedback by scorer and compute summary stats."""
    by_scorer: dict[str, tuple[list[float], list[list[str]]]] = defaultdict(lambda: ([], []))

    for fb in feedback:
        scorer = _extract_scorer(fb)
        rating = _extract_rating(fb)
        if scorer is None or rating is None:
            continue
        ratings, tags = by_scorer[scorer]
        ratings.append(rating)
        tags.append(_extract_tags(fb))

    return {
        scorer: _summarize(scorer, ratings, tags)
        for scorer, (ratings, tags) in by_scorer.items()
        if ratings
    }


def ab_leaderboard(feedback: list[dict]) -> list[ABResult]:
    """Build A/B leaderboard grouped by config_version."""
    by_config: dict[str, list[dict]] = defaultdict(list)

    for fb in feedback:
        config = _extract_config(fb)
        by_config[config].append(fb)

    results = []
    for config, config_feedback in sorted(by_config.items()):
        scores = aggregate_scores(config_feedback)
        refs = {fb.get("weave_ref") for fb in config_feedback} - {None, ""}
        results.append(
            ABResult(
                config_version=config,
                turn_count=len(refs),
                scores=scores,
            )
        )

    return results


def _parse_dt(val: str) -> datetime | None:
    try:
        if val.endswith("Z"):
            val = val[:-1] + "+00:00"
        dt = datetime.fromisoformat(val)
    except (ValueError, TypeError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _split_windows(
    entries: list[tuple[datetime, float]],
    window_days: int,
    min_samples: int,
) -> tuple[list[float], list[float]]:
    """Split time-sorted (time, rating) entries into (older, recent) ratings.

    Prefers a ``window_days`` recent window measured back from the latest entry;
    falls back to a median split when the window leaves either side below
    ``min_samples`` (common with sparse data).
    """
    latest = entries[-1][0]
    cutoff = latest - timedelta(days=window_days)
    recent = [r for t, r in entries if t >= cutoff]
    older = [r for t, r in entries if t < cutoff]
    if len(recent) >= min_samples and len(older) >= min_samples:
        return older, recent

    mid = len(entries) // 2
    return [r for _, r in entries[:mid]], [r for _, r in entries[mid:]]


def detect_regressions(
    feedback: list[dict],
    window_days: int = 7,
    min_samples: int = 2,
    threshold: float = 0.2,
) -> list[dict[str, Any]]:
    """Detect score regressions by comparing a recent window against older data.

    Entries are ordered by agent run time (``turn_started_at``), not scoring
    time, so a single backfill does not collapse the timeline.
    """
    by_scorer: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

    for fb in feedback:
        scorer = _extract_scorer(fb)
        rating = _extract_rating(fb)
        run_time = _parse_dt(_extract_run_time(fb))
        if scorer and rating is not None and run_time is not None:
            by_scorer[scorer].append((run_time, rating))

    regressions = []
    for scorer, entries in by_scorer.items():
        entries.sort(key=lambda x: x[0])
        if len(entries) < min_samples * 2:
            continue

        older, recent = _split_windows(entries, window_days, min_samples)

        older_mean = sum(older) / len(older)
        recent_mean = sum(recent) / len(recent)
        delta = recent_mean - older_mean

        if abs(delta) >= threshold:
            # "significant" = the two windows' 95% CIs don't overlap and each has
            # enough samples. Small-sample swings are still surfaced (a human may
            # want them) but flagged non-significant so they aren't over-read.
            older_ci = _mean_ci(older)
            recent_ci = _mean_ci(recent)
            disjoint = older_ci[0] > recent_ci[1] or recent_ci[0] > older_ci[1]
            significant = (
                disjoint
                and len(older) >= MIN_CONFIDENT_SAMPLES
                and len(recent) >= MIN_CONFIDENT_SAMPLES
            )
            regressions.append(
                {
                    "scorer": scorer,
                    "direction": "regression" if delta < 0 else "improvement",
                    "older_mean": round(older_mean, 3),
                    "recent_mean": round(recent_mean, 3),
                    "delta": round(delta, 3),
                    "sample_count": len(entries),
                    "significant": significant,
                }
            )

    return regressions


def detect_config_regressions(
    feedback: list[dict],
    min_samples: int = MIN_CONFIDENT_SAMPLES,
    threshold: float = 0.2,
) -> list[dict[str, Any]]:
    """Compare the newest config_version cohort against the previous one.

    The RSI safety net: catch a config change (a CLAUDE.md/skill edit) that made
    scores worse. Only drops are reported (improvements are not regressions);
    ``significant`` marks drops where the two cohorts' 95% CIs don't overlap and
    both have enough samples. Configs are ordered by their latest agent run time.
    """
    by_config: dict[str, list[dict]] = defaultdict(list)
    latest: dict[str, datetime] = {}
    for fb in feedback:
        cfg = _extract_config(fb)
        if cfg == "unknown":
            continue
        by_config[cfg].append(fb)
        rt = _parse_dt(_extract_run_time(fb))
        if rt is not None and (cfg not in latest or rt > latest[cfg]):
            latest[cfg] = rt

    ordered = sorted((c for c in by_config if c in latest), key=lambda c: latest[c])
    if len(ordered) < 2:
        return []
    new_cfg, prev_cfg = ordered[-1], ordered[-2]
    if latest[new_cfg] == latest[prev_cfg]:
        # Equal latest run times (e.g. a single backfill with no per-turn
        # timestamps) — we can't tell which config is newer, so comparing them
        # as new-vs-prev would be arbitrary. Skip rather than guess a direction.
        return []
    new_scores = aggregate_scores(by_config[new_cfg])
    prev_scores = aggregate_scores(by_config[prev_cfg])

    regressions = []
    for scorer, ns in new_scores.items():
        ps = prev_scores.get(scorer)
        if ps is None:
            continue
        delta = ns.mean - ps.mean
        if delta > -threshold:  # only drops are regressions
            continue
        disjoint = ns.ci[1] < ps.ci[0]  # new cohort's upper bound below prev's lower
        significant = disjoint and ns.count >= min_samples and ps.count >= min_samples
        regressions.append(
            {
                "scorer": scorer,
                "config": new_cfg,
                "prev_config": prev_cfg,
                "new_mean": round(ns.mean, 3),
                "prev_mean": round(ps.mean, 3),
                "delta": round(delta, 3),
                "sample_count": ns.count,
                "significant": significant,
            }
        )
    return regressions


def coaching_digest(feedback: list[dict]) -> str:
    """Build a human-readable coaching digest from scored feedback."""
    if not feedback:
        return "No scores found."

    summaries = aggregate_scores(feedback)
    if not summaries:
        return "No scores found."

    lines = ["# Coaching Digest", "", "## Summary", ""]

    for scorer, s in sorted(summaries.items()):
        lo, hi = s.ci
        low_n = "" if s.confident else "  ⚠ low n"
        pass_rate = s.pass_rate
        if pass_rate is not None:
            lines.append(
                f"- **{scorer}**: {pass_rate:.0%} pass rate "
                f"(n={s.count}, 95% CI {lo:.0%}–{hi:.0%}){low_n}"
            )
        else:
            lines.append(
                f"- **{scorer}**: mean={s.mean:.2f} (n={s.count}, 95% CI {lo:.2f}–{hi:.2f}){low_n}"
            )

        top_tags = sorted(s.tag_counts.items(), key=lambda x: -x[1])[:5]
        if top_tags:
            tag_str = ", ".join(f"{t}={c}" for t, c in top_tags)
            lines.append(f"  Tags: {tag_str}")

    regressions = detect_regressions(feedback)
    if regressions:
        lines.extend(["", "## Trends", ""])
        for r in regressions:
            arrow = "↓" if r["direction"] == "regression" else "↑"
            mark = "" if r["significant"] else "  (tentative — small sample)"
            lines.append(
                f"- {arrow} **{r['scorer']}**: {r['older_mean']:.2f} → {r['recent_mean']:.2f} "
                f"({r['delta']:+.2f}, n={r['sample_count']}){mark}"
            )

    ab = ab_leaderboard(feedback)
    if len(ab) > 1:
        lines.extend(["", "## A/B Comparison", ""])
        for result in ab:
            lines.append(f"### Config: `{result.config_version}` ({result.turn_count} turns)")
            for scorer, s in sorted(result.scores.items()):
                lines.append(f"  - {scorer}: mean={s.mean:.2f} (n={s.count})")

    return "\n".join(lines)
