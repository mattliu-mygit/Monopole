"""Score stats by config version that CES computes with EvaluationQueries.

Two saved queries cover every Score the Evaluations hold, not a newest-N window:

- turns: a Case query grouped by `i.config_version`. Turn Scores target each turn's first Case,
  so the turn's other Cases have no Score and drop out of a mean and count. Every scored Case is
  a main-agent Case: Monopole reads turns only from the session Trajectory's own Cases, and the
  adapter writes subagent Cases to child Trajectories. The query doesn't group by `i.agent_id`,
  because CES rejects grouping by a field no Case has set, as on a Dataset with no subagents.
- sessions: a Trajectory query grouped by the root Case's `i.config_version` and `depth`.
  Session Scores target the depth-0 session Trajectory.

CES runs EvaluationQueries only on an Evaluation whose primary target kind is case, so both run
on the turns Evaluation, which therefore also selects every session Scorer: the deterministic
ones and each judge panel's `<rubric>.panel-<digest>`.

A query definition never changes. Its create key is a digest of its body, so a rerun replays to
the same query ID, and a new judge panel, which is a new Scorer, makes a new query.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from weave_agent_signals.ces_catalog import DETAILS_SUFFIX, Catalog
from weave_agent_signals.ces_client import CesClient, content_digest
from weave_agent_signals.patterns import ConfigStats, Stat, stats_from_feedback

# CES's spread aggregation, from a CES change that hasn't merged. Its name and its sample (n-1)
# denominator are assumptions until it does.
STDDEV = "stddev"
_MAX_METRICS = 100  # CES's limit per query
_CONFIG = {"kind": "dataset_field", "source": "input", "name": "config_version"}
_DEPTH = {"kind": "depth"}


@dataclass(frozen=True)
class QueryScorer:
    name: str
    scorer_id: str
    version_id: str
    binary: bool


@dataclass(frozen=True)
class Group:
    """One result group: its group-by values in order, and each Scorer's numbers in it."""

    values: tuple
    stats: dict[str, Stat]


@dataclass
class QueryState:
    """What one process learns from CES: query IDs by create key, and whether CES has the
    spread aggregation (None until a create with it answers)."""

    ids: dict[str, str] = field(default_factory=dict)
    stddev: bool | None = None


def _metrics(scorers: list[QueryScorer], stddev: bool) -> list[dict]:
    metrics = []
    for scorer in scorers:
        aggregations = ["mean", "count"] + ([STDDEV] if stddev and not scorer.binary else [])
        metrics += [
            {
                "name": f"{scorer.name}:{aggregation}",
                "scorer_version_id": scorer.version_id,
                "aggregation": aggregation,
            }
            for aggregation in aggregations
        ]
    return metrics


def _rejects_stddev(error: httpx.HTTPStatusError) -> bool:
    # CES's schema check answers an unknown enum value with 422; its own checks answer 400.
    return error.response.status_code in (400, 422) and "aggregation" in error.response.text


def _create(
    client: CesClient, state: QueryState, evaluation_id: str, name: str, definition: dict
) -> str:
    body = {"name": name, "definition": definition}
    key = f"monopole-query-v1:{evaluation_id}:{content_digest(body)}"
    if key not in state.ids:
        state.ids[key] = client.create_query(evaluation_id, body, key=key)
    return state.ids[key]


def run_query(
    client: CesClient,
    state: QueryState,
    evaluation_id: str,
    *,
    name: str,
    target_kind: str,
    group_by: list[dict],
    scorers: list[QueryScorer],
) -> list[Group]:
    """Each group's Stat per Scorer, from one query per 100 metrics. A Scorer with no Score in
    a group is left out of that group. A continuous Scorer has a stddev only when CES has the
    spread aggregation; the first create that names it finds out."""
    chunk = _MAX_METRICS // 3
    groups: dict[tuple, dict[str, Stat]] = defaultdict(dict)
    for start in range(0, len(scorers), chunk):
        part = scorers[start : start + chunk]
        title = f"{name} {start // chunk + 1}"
        definition = {"v": 2, "target_kind": target_kind, "group_by": group_by}
        wants_stddev = state.stddev is not False and any(not s.binary for s in part)
        metrics = _metrics(part, wants_stddev)
        try:
            query_id = _create(
                client, state, evaluation_id, title, {**definition, "metrics": metrics}
            )
        except httpx.HTTPStatusError as error:
            if not wants_stddev or not _rejects_stddev(error):
                raise
            state.stddev = False
            metrics = _metrics(part, False)
            query_id = _create(
                client, state, evaluation_id, title, {**definition, "metrics": metrics}
            )
        else:
            if wants_stddev:
                state.stddev = True
        for row in client.query_results(evaluation_id, query_id):
            cells = {cell["name"]: cell["value"] for cell in row["metrics"]}
            for scorer in part:
                count = cells.get(f"{scorer.name}:count") or 0
                mean = cells.get(f"{scorer.name}:mean")
                if count and mean is not None:
                    groups[tuple(row["group"])][scorer.name] = Stat(
                        int(count), float(mean), scorer.binary, cells.get(f"{scorer.name}:{STDDEV}")
                    )
    return [Group(values, stats) for values, stats in groups.items()]


def _session_scorers(client: CesClient, catalog: Catalog) -> list[QueryScorer]:
    """Every session value Scorer the sessions Evaluation selects, each now also selected by
    the turns Evaluation. Judge panel Scorers aren't in the catalog, so their names are read."""
    refs = [ref for ref in catalog.scorers.values() if ref.spec.target_kind == "trajectory"]
    scorers = [QueryScorer(r.spec.name, r.scorer_id, r.version_id, r.spec.binary) for r in refs]
    known = {r.scorer_id for r in refs} | {r.details_scorer_id for r in refs}
    selection = client.select_scorers(
        catalog.sessions_evaluation_id,
        [{"scorer_id": s.scorer_id, "version_id": s.version_id} for s in scorers],
    )
    for item in selection:
        if item["scorer_id"] in known:
            continue
        name = client.scorer_name(item["scorer_id"], item["version_id"])
        # Judge ratings are continuous. An unpaneled judge Scorer predates panel identity, so it
        # could mix panels in one mean; details Scorers hold JSON.
        if name and ".panel-" in name and not name.endswith(DETAILS_SUFFIX):
            scorers.append(QueryScorer(name, item["scorer_id"], item["version_id"], False))
    client.select_scorers(
        catalog.turns_evaluation_id,
        [{"scorer_id": s.scorer_id, "version_id": s.version_id} for s in scorers],
    )
    return sorted(scorers, key=lambda s: s.name)


def config_stats(
    client: CesClient,
    catalog: Catalog,
    state: QueryState,
    feedback: Callable[[], list[dict[str, Any]]],
) -> ConfigStats:
    """Every config's turn and session Stats, main agent only.

    `feedback` reads raw Score records. It's called only when CES has no spread aggregation:
    then continuous Scorers come from those records instead, binary ones still from queries."""
    turn_scorers = sorted(
        (
            QueryScorer(ref.spec.name, ref.scorer_id, ref.version_id, ref.spec.binary)
            for ref in catalog.scorers.values()
            if ref.spec.target_kind == "case"
        ),
        key=lambda s: s.name,
    )
    session_scorers = _session_scorers(client, catalog)
    evaluation = catalog.turns_evaluation_id
    turns = run_query(
        client,
        state,
        evaluation,
        name="monopole turns by config",
        target_kind="case",
        group_by=[_CONFIG],
        scorers=turn_scorers,
    )
    sessions = run_query(
        client,
        state,
        evaluation,
        name="monopole sessions by config",
        target_kind="trajectory",
        group_by=[_CONFIG, _DEPTH],
        scorers=session_scorers,
    )
    by_config: dict[str, dict[str, Stat]] = defaultdict(dict)
    targets: Counter[str] = Counter()
    main = turns + [g for g in sessions if g.values[1] == 0]
    for group in main:
        config = group.values[0] or "unknown"
        by_config[config].update(group.stats)
        # Every scored turn has an efficiency Score and every scored session its session
        # Scores, so the most-counted Scorer counts the group's scored targets.
        targets[config] += max((stat.count for stat in group.stats.values()), default=0)

    if state.stddev is False:
        # Fallback: delete when CES stddev ships. Continuous Scorers take today's raw-Score
        # path, which covers only the newest turns that `feedback` reads.
        continuous = {s.name for s in turn_scorers + session_scorers if not s.binary}
        for scorers in by_config.values():
            for name in continuous & scorers.keys():
                del scorers[name]
        for config, scorers in stats_from_feedback(feedback()).items():
            by_config[config].update(
                (name, stat) for name, stat in scorers.items() if name in continuous
            )
    return ConfigStats(
        {config: scorers for config, scorers in by_config.items() if scorers}, dict(targets)
    )
