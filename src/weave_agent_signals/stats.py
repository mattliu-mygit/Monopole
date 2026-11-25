"""Interval statistics for Score summaries: a Scorer's count, mean, and spread (`Stat`), 95%
intervals for a mean and for a difference of means, and exact or bounded spreads from value
histograms and pooled groups. Pure computation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

_Z_95 = 1.96  # z for a 95% interval


def wilson_ci(k: float, n: int, z: float = _Z_95) -> tuple[float, float]:
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
    # The stddev is an upper bound (`conservative_stat`), so intervals built from it are wider
    # than the data's own.
    conservative: bool = False


def histogram_stat(counts: Mapping[float, int]) -> Stat:
    """The exact Stat of continuous ratings given as how many times each value occurs, such as
    {0.5: 2, 1.0: 1} for the ratings 0.5, 0.5, and 1.0."""
    n = sum(counts.values())
    mean = sum(value * count for value, count in counts.items()) / n
    if n < 2:
        return Stat(n, mean, False)
    squares = sum(count * (value - mean) ** 2 for value, count in counts.items())
    return Stat(n, mean, False, math.sqrt(squares / (n - 1)))


def conservative_stat(stat: Stat) -> Stat:
    """The Stat with the largest stddev ratings in [0, 1] with its mean can have. Their
    population variance is at most mean·(1−mean), reached when every rating is 0 or 1, so the
    sample variance is at most n/(n−1) times that. Example: 4 ratings with mean 0.5 give
    sqrt(4/3 · 0.25) ≈ 0.577."""
    if stat.count < 2:
        return Stat(stat.count, stat.mean, False)
    mean = min(1.0, max(0.0, stat.mean))
    variance = stat.count / (stat.count - 1) * mean * (1 - mean)
    return Stat(stat.count, stat.mean, False, math.sqrt(variance), conservative=True)


def ratings_stat(ratings: list[float], binary: bool | None = None) -> Stat:
    """The Stat of observed ratings. Without `binary`, ratings that are all 0 or 1 are binary."""
    n = len(ratings)
    mean = sum(ratings) / n
    if binary is None:
        binary = all(r in (0.0, 1.0) for r in ratings)
    if binary or n < 2:
        return Stat(n, mean, binary)
    return Stat(n, mean, False, math.sqrt(sum((r - mean) ** 2 for r in ratings) / (n - 1)))


def mean_ci(stat: Stat, z: float = _Z_95) -> tuple[float, float]:
    """A 95% interval for the mean: Wilson for a binary Scorer, the normal approximation from
    the standard error otherwise, which collapses to the mean without a stddev."""
    if stat.binary:
        return wilson_ci(stat.mean * stat.count, stat.count, z)
    margin = 0.0 if stat.stddev is None else z * stat.stddev / math.sqrt(stat.count)
    return (max(0.0, stat.mean - margin), min(1.0, stat.mean + margin))


def pool_stats(stats: list[Stat]) -> Stat:
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
    # A bound on any group's spread bounds the pooled spread too.
    conservative = any(s.conservative for s in stats)
    return Stat(n, mean, False, math.sqrt(squares / (n - 1)), conservative)


def difference_ci(new: Stat, prev: Stat, z: float = _Z_95) -> tuple[float, float] | None:
    """A 95% interval for `new.mean - prev.mean`, or None when a continuous side has no stddev
    or fewer than two ratings.

    Binary Scorers use Newcombe's hybrid score interval, built from each side's Wilson interval.
    Continuous ones use Welch's standard error with the normal approximation. Unlike asking
    whether the two groups' own intervals overlap, which is stricter than a 95% test, this
    flags a difference whenever the interval excludes 0."""
    delta = new.mean - prev.mean
    if new.binary and prev.binary:
        new_lo, new_hi = mean_ci(new, z)
        prev_lo, prev_hi = mean_ci(prev, z)
        return (
            delta - math.hypot(new.mean - new_lo, prev_hi - prev.mean),
            delta + math.hypot(new_hi - new.mean, prev.mean - prev_lo),
        )
    # One rating has no spread, whatever stddev CES reports for it.
    if new.stddev is None or prev.stddev is None or min(new.count, prev.count) < 2:
        return None
    se = math.sqrt(new.stddev**2 / new.count + prev.stddev**2 / prev.count)
    return (delta - z * se, delta + z * se)
