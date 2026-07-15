"""Pattern analysis: A/B leaderboards, trend detection, coaching digest.

Operates on feedback records queried from Weave. All analysis is pure
computation — no API calls. The CLI orchestrates query → analyze → report.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import islice
from typing import Any
from urllib.parse import unquote

from weave_agent_signals.models import FEEDBACK_PREFIX

# Below this many samples a scorer's mean/pass-rate is not trustworthy on its own.
MIN_CONFIDENT_SAMPLES = 5
_Z_95 = 1.96  # z for a 95% interval
_SESSION_EVALUATION_CONTEXT_FIELDS = (
    "rubric_version",
    "rubric_threshold",
    "review_depth",
    "review_policy_version",
    "second_opinion_margin",
    "requested_judge_models",
)
_MAX_BEHAVIORAL_EXAMPLES = 3
_MAX_EVIDENCE_IDS = 5
_MAX_ID_CHARS = 160
_MAX_SUCCESS_CHARS = 300
_MAX_BEHAVIOR_CHARS = 500


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
    evaluated_target_count: int
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


def _is_judgment(feedback: dict) -> bool:
    scorer = _extract_scorer(feedback)
    return scorer is not None and scorer.startswith("judge.")


def _evaluation_context_value(details: dict, field: str) -> object:
    value = details[field]
    if field in {"rubric_threshold", "second_opinion_margin"}:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return value


def _has_complete_session_evaluation_context(feedback: dict) -> bool:
    if not _is_judgment(feedback):
        return False
    details = _details(feedback)
    if any(field not in details for field in _SESSION_EVALUATION_CONTEXT_FIELDS):
        return False

    rubric_version = details["rubric_version"]
    rubric_threshold = details["rubric_threshold"]
    review_depth = details["review_depth"]
    policy_version = details["review_policy_version"]
    margin = details["second_opinion_margin"]
    judges = details["requested_judge_models"]
    markers_match = (
        _payload(feedback).get("granularity") == "session"
        and details.get("evaluation_unit") == "session"
    )
    numeric_threshold = (
        isinstance(rubric_threshold, (int, float))
        and not isinstance(rubric_threshold, bool)
        and math.isfinite(float(rubric_threshold))
        and 0.0 <= float(rubric_threshold) <= 1.0
    )
    valid_judges = (
        isinstance(judges, list)
        and all(isinstance(judge, str) and judge.strip() for judge in judges)
        and len(judges) == len(set(judges))
    )
    valid_policy = False
    if valid_judges:
        if review_depth == "primary":
            valid_policy = len(judges) == 1 and margin is None
        elif review_depth == "selective":
            valid_policy = (
                len(judges) in {2, 3}
                and isinstance(margin, (int, float))
                and not isinstance(margin, bool)
                and math.isfinite(float(margin))
                and 0.0 <= float(margin) <= 0.5
            )
        elif review_depth == "full_panel":
            valid_policy = len(judges) == 3 and margin is None
    return (
        markers_match
        and isinstance(rubric_version, str)
        and bool(rubric_version.strip())
        and numeric_threshold
        and isinstance(policy_version, str)
        and bool(policy_version.strip())
        and valid_policy
    )


def _is_complete_session_judgment(feedback: dict) -> bool:
    return (
        _is_judgment(feedback)
        and _details(feedback).get("review_status") == "complete"
        and _has_complete_session_evaluation_context(feedback)
    )


def _evaluation_context_identity(feedback: dict) -> str | None:
    """Readable session-judge configuration identity used by every analysis."""

    if not _is_complete_session_judgment(feedback):
        return None
    details = _details(feedback)
    context = {
        field: _evaluation_context_value(details, field)
        for field in _SESSION_EVALUATION_CONTEXT_FIELDS
    }

    def text(value: object) -> str:
        if value is None:
            return "none"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    raw_judges = context["requested_judge_models"]
    judges = (
        " → ".join(text(model) for model in raw_judges)
        if isinstance(raw_judges, list)
        else text(raw_judges)
    )
    return (
        f"rubric {text(context['rubric_version'])}, "
        f"threshold {text(context['rubric_threshold'])}; "
        f"{text(context['review_depth'])}, "
        f"policy {text(context['review_policy_version'])}, "
        f"margin {text(context['second_opinion_margin'])}; "
        f"judges {judges}"
    )


def _analytics_scorer(feedback: dict) -> str | None:
    scorer = _extract_scorer(feedback)
    if scorer is None:
        return None
    if _is_judgment(feedback) and not _is_complete_session_judgment(feedback):
        return None
    context = _evaluation_context_identity(feedback)
    return scorer if context is None else f"{scorer} [{context}]"


def _is_ordinary_analytics_score(feedback: dict) -> bool:
    return not _is_judgment(feedback) or _is_complete_session_judgment(feedback)


def _extract_rating(feedback: dict) -> float | None:
    rating = _payload(feedback).get("rating")
    if isinstance(rating, bool) or not isinstance(rating, (int, float)):
        return None
    value = float(rating)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def _extract_config(feedback: dict) -> str:
    return _details(feedback).get("config_version") or "unknown"


def _extract_tags(feedback: dict) -> list[str]:
    return _payload(feedback).get("tags") or []


def _extract_run_time(feedback: dict) -> object:
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
    """Group representative feedback by scorer and compute summary stats.

    Only complete current session judgments are comparable judge scores.
    """
    by_scorer: dict[str, tuple[list[float], list[list[str]]]] = defaultdict(lambda: ([], []))

    for fb in feedback:
        if not _is_ordinary_analytics_score(fb):
            continue
        scorer = _analytics_scorer(fb)
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
        if not _is_ordinary_analytics_score(fb):
            continue
        config = _extract_config(fb)
        by_config[config].append(fb)

    results = []
    for config, config_feedback in sorted(by_config.items()):
        scores = aggregate_scores(config_feedback)
        refs = {fb.get("weave_ref") for fb in config_feedback} - {None, ""}
        results.append(
            ABResult(
                config_version=config,
                evaluated_target_count=len(refs),
                scores=scores,
            )
        )

    return results


def _parse_dt(val: object) -> datetime | None:
    try:
        if not isinstance(val, str):
            return None
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
        if not _is_ordinary_analytics_score(fb):
            continue
        scorer = _analytics_scorer(fb)
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
    """Compare the newest two configs within each comparable score cohort.

    The instruction-change safety net: catch a config change (a CLAUDE.md/skill edit) that made
    scores worse. Only drops are reported (improvements are not regressions);
    ``significant`` marks drops where the two cohorts' 95% CIs don't overlap and
    both have enough samples. Configs are ordered by their latest agent run time
    separately for each scorer and session-judge evaluation context.
    """
    by_scorer: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    latest: dict[str, dict[str, datetime]] = defaultdict(dict)
    for fb in feedback:
        if not _is_ordinary_analytics_score(fb):
            continue
        scorer = _analytics_scorer(fb)
        cfg = _extract_config(fb)
        if scorer is None or _extract_rating(fb) is None or cfg == "unknown":
            continue
        by_scorer[scorer][cfg].append(fb)
        rt = _parse_dt(_extract_run_time(fb))
        if rt is not None and (cfg not in latest[scorer] or rt > latest[scorer][cfg]):
            latest[scorer][cfg] = rt

    regressions = []
    for scorer in sorted(by_scorer):
        scorer_latest = latest[scorer]
        ordered = sorted(
            (cfg for cfg in by_scorer[scorer] if cfg in scorer_latest),
            key=lambda cfg: scorer_latest[cfg],
        )
        if len(ordered) < 2:
            continue
        new_cfg, prev_cfg = ordered[-1], ordered[-2]
        if scorer_latest[new_cfg] == scorer_latest[prev_cfg]:
            # Equal latest run times (e.g. a single backfill with no per-turn
            # timestamps) leave no defensible new-vs-previous direction.
            continue
        ns = aggregate_scores(by_scorer[scorer][new_cfg]).get(scorer)
        ps = aggregate_scores(by_scorer[scorer][prev_cfg]).get(scorer)
        if ns is None or ps is None:
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


def _bounded_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    if not normalized:
        return None
    return normalized[:limit]


def _session_id(feedback: dict) -> str:
    weave_ref = feedback.get("weave_ref")
    if not isinstance(weave_ref, str):
        return "unknown"
    identifier = unquote(weave_ref.rstrip("/").rsplit("/", 1)[-1])
    return _bounded_text(identifier.replace("`", "'"), _MAX_ID_CHARS) or "unknown"


def _bounded_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return _bounded_text(value.replace("`", "'"), _MAX_ID_CHARS)


def _reviewer_feedback(details: dict) -> list[dict[str, object]]:
    judges = details.get("requested_judge_models")
    raw_feedback = details.get("behavioral_feedback")
    if not isinstance(judges, list) or not isinstance(raw_feedback, list):
        return []
    limit = min(len(judges), 3)
    values: list[dict[str, object]] = []
    expected = {"success", "problem", "desired_behavior"}
    for item in islice(raw_feedback, limit):
        if (
            not isinstance(item, dict)
            or set(item) != expected
            or any(value is not None and not isinstance(value, str) for value in item.values())
        ):
            continue
        values.append(item)
    return values


def _behavior_values(
    reviewer_feedback: list[dict[str, object]], field: str, limit: int
) -> list[str]:
    values: list[str] = []
    for item in reviewer_feedback:
        value = _bounded_text(item.get(field), limit)
        if value is not None and value not in values:
            values.append(value)
    return values


def _execution_sort_key(feedback: dict) -> tuple[int, datetime, str, str]:
    raw = _extract_run_time(feedback)
    parsed = _parse_dt(raw)
    if parsed is not None:
        return (0, parsed, "", "")
    try:
        fallback = json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        fallback = repr(raw)
    return (
        1,
        datetime.max.replace(tzinfo=timezone.utc),
        type(raw).__name__,
        fallback[:_MAX_ID_CHARS],
    )


def _behavioral_examples(feedback: list[dict]) -> dict[str, list[dict[str, object]]]:
    by_rubric: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in feedback:
        if not _is_complete_session_judgment(item):
            continue
        scorer = _extract_scorer(item)
        rating = _extract_rating(item)
        details = _details(item)
        threshold = details.get("rubric_threshold")
        if (
            scorer is None
            or rating is None
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or rating >= float(threshold)
        ):
            continue
        reviewer_feedback = _reviewer_feedback(details)
        problems = _behavior_values(reviewer_feedback, "problem", _MAX_BEHAVIOR_CHARS)
        desired = _behavior_values(reviewer_feedback, "desired_behavior", _MAX_BEHAVIOR_CHARS)
        if not problems and not desired:
            continue
        successes = [
            value
            for value in _behavior_values(reviewer_feedback, "success", _MAX_SUCCESS_CHARS)
            if value not in problems and value not in desired
        ]
        evidence: list[str] = []
        raw_evidence = details.get("evidence_trace_ids")
        if isinstance(raw_evidence, list):
            for raw_id in raw_evidence:
                value = _bounded_id(raw_id)
                if value is not None and value not in evidence:
                    evidence.append(value)
                if len(evidence) == _MAX_EVIDENCE_IDS:
                    break
        by_rubric[scorer].append(
            {
                "rating": rating,
                "execution_sort_key": _execution_sort_key(item),
                "session_id": _session_id(item),
                "evidence_ids": evidence,
                "successes": successes,
                "problems": problems,
                "desired": desired,
            }
        )
    for examples in by_rubric.values():
        examples.sort(
            key=lambda value: (
                value["rating"],
                value["execution_sort_key"],
                value["session_id"],
            )
        )
        del examples[_MAX_BEHAVIORAL_EXAMPLES:]
    return dict(by_rubric)


def coaching_digest(feedback: list[dict]) -> str:
    """Build a human-readable coaching digest from scored feedback."""
    if not feedback:
        return "No scores found."

    representative = [fb for fb in feedback if _is_ordinary_analytics_score(fb)]
    summaries = aggregate_scores(representative)
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

    examples = _behavioral_examples(representative)
    if examples:
        lines.extend(["", "## Behavioral feedback", ""])
        for scorer, rubric_examples in sorted(examples.items()):
            lines.append(f"### {scorer}")
            for example in rubric_examples:
                evidence = example["evidence_ids"]
                evidence_text = (
                    ", ".join(f"`{value}`" for value in evidence)
                    if isinstance(evidence, list) and evidence
                    else "none"
                )
                lines.append(
                    f"- Session `{example['session_id']}` · score {example['rating']:.2f} "
                    f"· evidence {evidence_text}"
                )
                for label, field in (
                    ("Success", "successes"),
                    ("Problem", "problems"),
                    ("Desired behavior", "desired"),
                ):
                    values = example[field]
                    if isinstance(values, list):
                        for value in values:
                            lines.append(f"  - {label}: {value}")

    regressions = detect_regressions(representative)
    if regressions:
        lines.extend(["", "## Trends", ""])
        for r in regressions:
            arrow = "↓" if r["direction"] == "regression" else "↑"
            mark = "" if r["significant"] else "  (tentative — small sample)"
            lines.append(
                f"- {arrow} **{r['scorer']}**: {r['older_mean']:.2f} → {r['recent_mean']:.2f} "
                f"({r['delta']:+.2f}, n={r['sample_count']}){mark}"
            )

    ab = ab_leaderboard(representative)
    if len(ab) > 1:
        lines.extend(["", "## A/B Comparison", ""])
        for result in ab:
            lines.append(
                f"### Config: `{result.config_version}` "
                f"({result.evaluated_target_count} evaluated targets)"
            )
            for scorer, s in sorted(result.scores.items()):
                lines.append(f"  - {scorer}: mean={s.mean:.2f} (n={s.count})")

    return "\n".join(lines)
