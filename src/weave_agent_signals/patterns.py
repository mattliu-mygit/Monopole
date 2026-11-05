"""Pattern analysis: summaries, A/B leaderboards, config regressions, trends, coaching digest.

Summaries, the A/B leaderboard, and config regressions take `ConfigStats`, the per-config
count, mean, and spread that CES EvaluationQueries compute over every Score (`ces_queries`).
Trends, coaching examples, and reflection's digest take feedback-shaped records built from raw
CES Scores. All analysis is pure computation — no API calls.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import islice
from typing import Any, Mapping

from weave_agent_signals.judges.review import PANEL_CONTRACT_VERSION
from weave_agent_signals.models import FEEDBACK_PREFIX

# Below this many samples a scorer's mean/pass-rate is not trustworthy on its own.
MIN_CONFIDENT_SAMPLES = 5
_Z_95 = 1.96  # z for a 95% interval
_SESSION_EVALUATION_CONTEXT_FIELDS = (
    "rubric_version",
    "rubric_threshold",
    "panel_contract_version",
    "requested_judge_models",
    "panel_size",
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


@dataclass(frozen=True)
class Stat:
    """One Scorer's ratings in one group: how many, their mean, and their sample standard
    deviation. A binary Scorer's spread follows from its mean, so it has no stddev; a
    continuous one has none below two ratings."""

    count: int
    mean: float
    binary: bool
    stddev: float | None = None


@dataclass(frozen=True)
class ConfigStats:
    """Each config version's Stat per Scorer, and how many turns and sessions it scored.

    A config the adapter didn't record is "unknown"."""

    by_config: dict[str, dict[str, Stat]]
    targets: dict[str, int]


def _stat(ratings: list[float], binary: bool | None = None) -> Stat:
    """The Stat of observed ratings. Without `binary`, ratings that are all 0 or 1 are binary."""
    n = len(ratings)
    mean = sum(ratings) / n
    if binary is None:
        binary = all(r in (0.0, 1.0) for r in ratings)
    if binary or n < 2:
        return Stat(n, mean, binary)
    return Stat(n, mean, False, math.sqrt(sum((r - mean) ** 2 for r in ratings) / (n - 1)))


def _ci(stat: Stat, z: float = _Z_95) -> tuple[float, float]:
    """A 95% interval for the mean: Wilson for a binary Scorer, the normal approximation from
    the standard error otherwise, which collapses to the mean without a stddev."""
    if stat.binary:
        return _wilson_ci(stat.mean * stat.count, stat.count, z)
    margin = 0.0 if stat.stddev is None else z * stat.stddev / math.sqrt(stat.count)
    return (max(0.0, stat.mean - margin), min(1.0, stat.mean + margin))


def _pool(stats: list[Stat]) -> Stat:
    """The Stat of several groups' ratings taken together, exact for the sample variance:
    within-group squares plus each group's squared distance from the pooled mean."""
    n = sum(s.count for s in stats)
    mean = sum(s.count * s.mean for s in stats) / n
    binary = all(s.binary for s in stats)
    if binary or n < 2 or any(s.stddev is None and s.count > 1 for s in stats):
        return Stat(n, mean, binary)
    squares = sum(
        (s.count - 1) * (s.stddev or 0.0) ** 2 + s.count * (s.mean - mean) ** 2 for s in stats
    )
    return Stat(n, mean, False, math.sqrt(squares / (n - 1)))


def difference_ci(new: Stat, prev: Stat, z: float = _Z_95) -> tuple[float, float] | None:
    """A 95% interval for `new.mean - prev.mean`, or None when a continuous side has no stddev
    or fewer than two ratings.

    Binary Scorers use Newcombe's hybrid score interval, built from each side's Wilson interval.
    Continuous ones use Welch's standard error with the normal approximation. Unlike asking
    whether the two groups' own intervals overlap, which is stricter than a 95% test, this
    flags a difference whenever the interval excludes 0."""
    delta = new.mean - prev.mean
    if new.binary and prev.binary:
        new_lo, new_hi = _ci(new, z)
        prev_lo, prev_hi = _ci(prev, z)
        return (
            delta - math.hypot(new.mean - new_lo, prev_hi - prev.mean),
            delta + math.hypot(new_hi - new.mean, prev.mean - prev_lo),
        )
    # One rating has no spread, whatever stddev CES reports for it.
    if new.stddev is None or prev.stddev is None or min(new.count, prev.count) < 2:
        return None
    se = math.sqrt(new.stddev**2 / new.count + prev.stddev**2 / prev.count)
    return (delta - z * se, delta + z * se)


@dataclass
class ScoreSummary:
    scorer: str
    count: int
    mean: float
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
    if isinstance(ftype, str) and ftype.startswith(FEEDBACK_PREFIX):
        return ftype[len(FEEDBACK_PREFIX) :]
    return None


def _payload(feedback: dict) -> dict:
    # `.get("payload", {})` is not enough: the API can return an explicit null.
    payload = feedback.get("payload")
    return payload if isinstance(payload, dict) else {}


def _details(feedback: dict) -> dict:
    details = _payload(feedback).get("details")
    return details if isinstance(details, dict) else {}


def _is_judgment(feedback: dict) -> bool:
    scorer = _extract_scorer(feedback)
    return scorer is not None and scorer.startswith("judge.")


def _evaluation_context_value(details: dict, field: str) -> object:
    value = details[field]
    if field == "rubric_threshold":
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
    panel_version = details["panel_contract_version"]
    judges = details["requested_judge_models"]
    panel_size = details["panel_size"]
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
        and 1 <= len(judges) <= 3
        and all(isinstance(judge, str) and judge.strip() for judge in judges)
        and len(judges) == len(set(judges))
    )
    return (
        markers_match
        and isinstance(rubric_version, str)
        and bool(rubric_version.strip())
        and numeric_threshold
        and panel_version == PANEL_CONTRACT_VERSION
        and valid_judges
        and type(panel_size) is int
        and panel_size == len(judges)
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
        f"panel {text(context['panel_contract_version'])}; "
        f"judges {judges}"
    )


def _analytics_scorer(feedback: dict) -> str | None:
    if not is_evaluation_feedback_eligible(feedback):
        return None
    scorer = _extract_scorer(feedback)
    if scorer is None:
        return None
    context = _evaluation_context_identity(feedback)
    return scorer if context is None else f"{scorer} [{context}]"


def _extract_rating(feedback: dict) -> float | None:
    rating = _payload(feedback).get("rating")
    if isinstance(rating, bool) or not isinstance(rating, (int, float)):
        return None
    value = float(rating)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def is_evaluation_feedback_eligible(feedback: Mapping[str, Any]) -> bool:
    """Whether a score is valid evidence for analytics and reflection.

    Deterministic scores need a project feedback type and a valid rating.
    Model judgments additionally need the exact complete, comparable current
    session contract; legacy and audit-only judgment rows remain inspectable
    but cannot influence analysis or instruction reflection.
    """

    scorer = _extract_scorer(feedback)
    if scorer is None or _extract_rating(feedback) is None:
        return False
    return not scorer.startswith("judge.") or _is_complete_session_judgment(feedback)


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


def _summary(scorer: str, stat: Stat, tag_counts: dict[str, int] | None = None) -> ScoreSummary:
    return ScoreSummary(
        scorer=scorer,
        count=stat.count,
        mean=stat.mean,
        tag_counts=tag_counts or {},
        binary=stat.binary,
        ci=_ci(stat),
    )


def _summarize(scorer: str, ratings: list[float], tags: list[list[str]]) -> ScoreSummary:
    tag_counts: dict[str, int] = defaultdict(int)
    for tag_list in tags:
        for tag in tag_list:
            tag_counts[tag] += 1
    return _summary(scorer, _stat(ratings), dict(tag_counts))


def aggregate_scores(feedback: list[dict]) -> dict[str, ScoreSummary]:
    """Group representative feedback by scorer and compute summary stats.

    Only complete current session judgments are comparable judge scores.
    """
    by_scorer: dict[str, tuple[list[float], list[list[str]]]] = defaultdict(lambda: ([], []))

    for fb in feedback:
        if not is_evaluation_feedback_eligible(fb):
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
        if not is_evaluation_feedback_eligible(fb):
            continue
        config = _extract_config(fb)
        by_config[config].append(fb)

    results = []
    for config, config_feedback in sorted(by_config.items()):
        scores = aggregate_scores(config_feedback)
        refs = {fb.get("target_id") for fb in config_feedback} - {None, ""}
        results.append(
            ABResult(
                config_version=config,
                evaluated_target_count=len(refs),
                scores=scores,
            )
        )

    return results


def stat_summaries(stats: ConfigStats) -> dict[str, ScoreSummary]:
    """Each Scorer's summary across every config. Query results carry no tags."""
    by_scorer: dict[str, list[Stat]] = defaultdict(list)
    for scorers in stats.by_config.values():
        for scorer, stat in scorers.items():
            by_scorer[scorer].append(stat)
    return {scorer: _summary(scorer, _pool(group)) for scorer, group in by_scorer.items()}


def stat_leaderboard(stats: ConfigStats) -> list[ABResult]:
    """The A/B leaderboard: each config's summaries, configs in name order."""
    return [
        ABResult(
            config_version=config,
            evaluated_target_count=stats.targets.get(config, 0),
            scores={scorer: _summary(scorer, stat) for scorer, stat in scorers.items()},
        )
        for config, scorers in sorted(stats.by_config.items())
    ]


def stats_from_feedback(feedback: list[dict]) -> dict[str, dict[str, Stat]]:
    """Continuous Stats by config and CES Scorer name from raw Score records, for a CES
    without a spread aggregation. Every Scorer this serves is continuous."""
    ratings: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fb in feedback:
        name = _details(fb).get("ces_scorer")
        rating = _extract_rating(fb)
        if is_evaluation_feedback_eligible(fb) and isinstance(name, str) and rating is not None:
            ratings[_extract_config(fb)][name].append(rating)
    return {
        config: {name: _stat(values, binary=False) for name, values in scorers.items()}
        for config, scorers in ratings.items()
    }


def config_latest(feedback: list[dict]) -> dict[str, datetime]:
    """Each config's latest agent run time among these Score records. CES queries have no
    time, so this orders configs for config regressions."""
    latest: dict[str, datetime] = {}
    for fb in feedback:
        if not is_evaluation_feedback_eligible(fb):
            continue
        config, run_time = _extract_config(fb), _parse_dt(_extract_run_time(fb))
        if run_time is not None and (config not in latest or run_time > latest[config]):
            latest[config] = run_time
    return latest


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
        if not is_evaluation_feedback_eligible(fb):
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
            older_ci = _ci(_stat(older, binary=False))
            recent_ci = _ci(_stat(recent, binary=False))
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


def config_regressions(
    stats: ConfigStats,
    latest: dict[str, datetime],
    threshold: float = 0.2,
) -> list[dict[str, Any]]:
    """Compare each Scorer's newest two configs.

    The instruction-change safety net: catch a config change (a CLAUDE.md/skill edit) that made
    scores worse. Configs are ordered by their latest agent run time in `latest`; a config
    missing from it, or "unknown", isn't compared. Only drops of at least `threshold` are
    reported. A drop is ``significant`` when the 95% interval on the difference
    (`difference_ci`) lies wholly below 0.
    """
    ordered = sorted(
        (config for config in stats.by_config if config in latest and config != "unknown"),
        key=latest.__getitem__,
    )
    scorers = sorted({scorer for config in ordered for scorer in stats.by_config[config]})
    regressions = []
    for scorer in scorers:
        having = [config for config in ordered if scorer in stats.by_config[config]]
        if len(having) < 2:
            continue
        new_cfg, prev_cfg = having[-1], having[-2]
        if latest[new_cfg] == latest[prev_cfg]:
            # Equal latest run times (e.g. a single backfill with no per-turn
            # timestamps) leave no defensible new-vs-previous direction.
            continue
        new, prev = stats.by_config[new_cfg][scorer], stats.by_config[prev_cfg][scorer]
        delta = new.mean - prev.mean
        if delta > -threshold:  # only drops are regressions
            continue
        ci = difference_ci(new, prev)
        regressions.append(
            {
                "scorer": scorer,
                "config": new_cfg,
                "prev_config": prev_cfg,
                "new_mean": round(new.mean, 3),
                "prev_mean": round(prev.mean, 3),
                "delta": round(delta, 3),
                "difference_ci": None if ci is None else (round(ci[0], 3), round(ci[1], 3)),
                "sample_count": new.count,
                "prev_sample_count": prev.count,
                "significant": ci is not None and ci[1] < 0,
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
    target_id = feedback.get("target_id")
    if not isinstance(target_id, str):
        return "unknown"
    return _bounded_text(target_id.replace("`", "'"), _MAX_ID_CHARS) or "unknown"


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
        if not is_evaluation_feedback_eligible(item) or not _is_judgment(item):
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


def coaching_digest(feedback: list[dict], stats: ConfigStats | None = None) -> str:
    """Build a human-readable coaching digest from scored feedback.

    With `stats`, the summary and A/B sections come from them, and `feedback` supplies only
    the behavioral examples and trends. Without, every section comes from `feedback`, as
    reflection needs: its digest must describe exactly the records it pinned."""
    representative = [fb for fb in feedback if is_evaluation_feedback_eligible(fb)]
    summaries = aggregate_scores(representative) if stats is None else stat_summaries(stats)
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

    ab = ab_leaderboard(representative) if stats is None else stat_leaderboard(stats)
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
