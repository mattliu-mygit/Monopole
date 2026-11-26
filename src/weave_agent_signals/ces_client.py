"""The single place Monopole talks to CES (CoreWeave Evaluations): HTTP requests, retries,
paging, and query slots. It reads the Cases weave-agent-adapter writes, and `ces_sessions`
rebuilds them into sessions and turns.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx

from weave_agent_signals.auth import get_api_key
from weave_agent_signals.ces_catalog import (
    DETAILS_SUFFIX,
    DETERMINISTIC,
    NOT_RATINGS,
    Catalog,
    ScorerRef,
    ScorerSpec,
    judge_spec,
    rubric_of,
)
from weave_agent_signals.ces_sessions import (
    SubagentIndex,
    child_ids,
    session_tree,
    session_view,
)
from weave_agent_signals.models import Score, SessionView, SubagentSpan, parse_utc
from weave_agent_signals.scorers.outcome import OUTCOME_TAGS

# CES runs at most 2 queries at a time per instance: row reads, Trajectory-row reads, and
# EvaluationQuery results share them, and a request with none free gets 503 with Retry-After: 1.
# Monopole sends at most QUERY_SLOTS of those requests at once from the whole process, so its own
# concurrent reads can't fill every slot.
QUERY_SLOTS = 2
_query_slots = threading.BoundedSemaphore(QUERY_SLOTS)
# Every request retries a 429, 5xx, or network error with waits of 1, 2, 4, 8, and 16 s, about
# 31 s in all. Prod CES answered about 1 in 6 requests of every kind, single Trajectory Case
# lists included, with 503 storage_capacity_exceeded in November 2026, at 2 or 16 requests at
# once alike, sometimes for 30 s running. Creates are keyed by content, so retrying them is safe.
ATTEMPTS = 6
MAX_RETRY_AFTER_S = 30.0
PAGE_LIMIT = 1000
# The smallest page the pager shrinks to. Prod CES answers a page of large Cases that reads too
# much with 503 storage_capacity_exceeded, and resending it unchanged keeps failing. Measured
# 2026-11-05: Trajectory Case lists of 1000 and 100 Cases (about 450 KB) failed 3 tries in a row
# on 4 of 6 Trajectories, 20 Cases on none; turn row searches of 200 to 1000 rows failed 17 of 18
# single tries, 100 rows 4 of 5. Starting at PAGE_LIMIT and halving still read sessions ae268659
# and 8138dbc5 in 40 s and 60 s, against 75 s and 89 s at a fixed 20, because CES accepted about
# 31 Cases per page there.
CASE_PAGE_LIMIT = 20
# A shrunken page doubles after this many accepted pages in a row. Measured 2026-11-25 against
# prod CES: staying at the accepted size read session 8138dbc5's Cases in 62-66 s with 171-206
# requests, and doubling after 1 or 3 accepted pages in 7-18 s with 14-37; the Dataset's 1,075
# turn rows took 2-34 s (mean 18 s) staying and 3-21 s (mean 10-12 s) doubling. Three bounds
# the refusals a uniformly heavy listing pays to one per three accepted pages.
GROW_AFTER = 3
# Independent reads (one session's turn rows, one subagent's Cases) run this many at once. Reads
# that take a CES query slot still wait for one of QUERY_SLOTS.
CONCURRENT_READS = 16
# Bounds one search filter's length: about 70 bytes per session.
SESSIONS_PER_SEARCH = 50
# A turn's row is its first Case; the adapter numbers each turn's Cases from 0. Subagent turns
# carry an agent ID and belong to the session turn that launched them, so they're left out.
FIRST_STEP = "i.step_index = 0 AND i.agent_id = null"
# The turn Case input fields a Score record reads (`_score_record`), and the session that groups
# the turn. The adapter writes them on every Case, so every adapter Dataset has them: CES rejects
# a selected field that its Dataset lacks.
RECORD_INPUT = ["session_trajectory_id", "started_at", "config_version", "git_branch"]


def _error_code(response: httpx.Response) -> str | None:
    try:
        return (response.json().get("error") or {}).get("code")
    except (ValueError, AttributeError):
        return None


def content_digest(body: dict[str, Any]) -> str:
    """Part of a create's Idempotency-Key, so a retry with the same content replays it and a
    retry with changed content, such as a judge's new scored_at, writes a new one."""
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def plan_judge_writes(
    client: CesClient, catalog: Catalog, pending: list, *, force: bool
) -> tuple[Catalog, dict[str, list[Score]]]:
    """Judge Scores to write, by session Trajectory, and the catalog that holds their Scorers.

    Only finished reviews (review_status complete) write: a degraded or failed review leaves
    the session's last finished Scores in place. Each rubric and panel has its own Scorer.
    Without `force`, a panel Scorer that already has a Score on the session is skipped."""
    by_session: dict[str, list[Score]] = {}
    specs = {}
    for score, session in pending:
        if score.metadata.get("review_status") != "complete":
            continue
        spec = judge_spec(score.scorer, score.metadata)
        specs[spec.name] = spec
        by_session.setdefault(session.conversation_id, []).append(replace(score, scorer=spec.name))
    if not by_session:
        return catalog, {}
    catalog = client.add_scorers(catalog, tuple(specs.values()))
    existing = {} if force else client.session_scores(catalog, list(by_session))
    writes = {}
    for sid, scores in by_session.items():
        have = catalog.scored(existing.get(sid, []))
        todo = [score for score in scores if force or score.scorer not in have]
        if todo:
            writes[sid] = todo
    return catalog, writes


def _score_record(target_id: str, case_input: dict, scores: list[dict], names: dict) -> list[dict]:
    """One target's Scores as feedback-shaped records, the shape the analysis code takes.

    `names` maps (scorer_id, scorer_version_id) to the CES Scorer name. A `.details` Scorer
    supplies the tags, reason, and details of the value Scorer it shadows; without it, a turn
    outcome Scorer's tag follows from its value. The target Case fills in the config, branch,
    and start time when the details lack them. A judge Scorer named `<rubric>.panel-<digest>`
    reports as its rubric."""
    values, details, ids = {}, {}, {}
    for score in scores:
        name = names.get((score["scorer_id"], score["scorer_version_id"]))
        if name is None:
            continue
        if name.endswith(DETAILS_SUFFIX):
            details[name[: -len(DETAILS_SUFFIX)]] = score["value"]
        else:
            values[name] = score["value"]
            ids[name] = score["score_id"]
    records = []
    for name, value in values.items():
        if name in NOT_RATINGS:
            continue  # records feed rating analysis only
        if not isinstance(value, (bool, int, float)) or not math.isfinite(value):
            continue  # not a rating; one malformed Score must not break a whole read
        extra = details.get(name) if isinstance(details.get(name), dict) else {}
        if name not in details and name in OUTCOME_TAGS:
            tag = OUTCOME_TAGS[name][value == 1]
            extra = {"tags": [tag] if tag else [], "granularity": "turn"}
        detail = dict(extra.get("details") or {})
        detail.setdefault("config_version", case_input.get("config_version"))
        detail.setdefault("git_branch", case_input.get("git_branch"))
        detail.setdefault("turn_started_at", case_input.get("started_at"))
        detail["ces_scorer"] = name
        payload = {
            "scorer_version": extra.get("scorer_version", "v1"),
            "rating": float(value) if isinstance(value, bool) else value,
            "tags": extra.get("tags") or [],
            "reason": extra.get("reason") or "",
            "granularity": extra.get("granularity"),
            "details": detail,
        }
        if extra.get("confidence") is not None:
            payload["confidence"] = extra["confidence"]
        if detail.get("scored_at"):
            payload["scored_at"] = detail["scored_at"]
        rubric = rubric_of(name)
        records.append(
            {
                "id": ids[name],
                "feedback_type": f"weave_agent_signals.{rubric}",
                "target_id": target_id,
                "payload": payload,
            }
        )
    return records


def session_or_none(
    client: CesClient,
    trajectory_id: str,
    *,
    subagents: bool = True,
    counts: SubagentCounts | None = None,
) -> SessionView | None:
    """The session, or None when CES has no such session Trajectory."""
    try:
        return client.read_session(trajectory_id, subagents=subagents, counts=counts)
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 404:
            return None
        raise


def ces_http(base_url: str, api_key: str | None = None) -> httpx.Client:
    """An HTTP client for CES. Clients that share one also share its connections; a
    CesClient closes only an HTTP client it made."""
    return httpx.Client(
        base_url=base_url.rstrip("/"), auth=("api", api_key or get_api_key()), timeout=60.0
    )


def _turns_of(session_id: str) -> str:
    """A rows/search filter for one session's turns."""
    return f"{FIRST_STEP} AND i.session_trajectory_id = {json.dumps(session_id)}"


def _turns_of_any(session_ids: list[str]) -> str:
    """A rows/search filter for several sessions' turns. CES has no IN operator."""
    either = " OR ".join(f"i.session_trajectory_id = {json.dumps(sid)}" for sid in session_ids)
    return f"{FIRST_STEP} AND ({either})"


def _turn_scorers(catalog: Catalog, details: Callable[[str], bool]) -> list[str]:
    """The CES Scorers a turn row read selects: every turn scorer's value Scorer, and the details
    Scorer of each one `details` accepts. A selected Scorer the turns Evaluation lacks is
    rejected, and the catalog's turn scorers are the ones it selects."""
    names = [name for name, ref in catalog.scorers.items() if ref.spec.target_kind == "case"]
    return names + [name + DETAILS_SUFFIX for name in names if details(name)]


def _turn_targets(turn_rows: dict[str, list[dict]]) -> list[tuple[str, dict, list]]:
    """(turn Case ID, Case input, Score rows) for each turn row, by session."""
    return [
        (r["case_id"], r["input"], r.get("scores") or [])
        for rows in turn_rows.values()
        for r in rows
    ]


class SubagentCounts:
    """Each subagent's type and tool-call count, the whole of what a judge shows of it, for
    session reads that need nothing more (`CesClient.read_session`). It reads the Dataset's
    Trajectory listing and one EvaluationQuery counting Cases by session, agent ID, and kind on
    first use, and every read given it shares them.

    Measured 2026-11-26 against prod CES: the listing and the query were 1.5 MB in 10 requests,
    where reading the Cases of the 2,117 subagents of 10 sessions was 603 MB in 2,191. Both grow
    with the whole Dataset, about 300 bytes per Trajectory and 3 query rows per subagent, so a
    Dataset holding far more subagents than the sessions read would favor reading them."""

    def __init__(self, client: CesClient, catalog: Catalog):
        self._client, self._catalog = client, catalog
        self._index: SubagentIndex | None = None
        self._lock = threading.Lock()

    def span(self, session_id: str, child: str) -> SubagentSpan | None:
        """The subagent's span from the counts, or None when it must be read."""
        with self._lock:
            if self._index is None:
                self._index = self._client.subagent_index(self._catalog)
        return self._index.span(session_id, child)


@dataclass
class CesSession:
    """A CES client and Monopole's catalog for the adapter's Dataset. Exiting closes the client."""

    client: CesClient
    catalog: Catalog

    def __enter__(self) -> CesSession:
        return self

    def __exit__(self, *_exc) -> None:
        self.client.close()


class CesClient:
    def __init__(
        self,
        base_url: str,
        scope_ref: str,
        *,
        api_key: str | None = None,
        namespace: str = "wandb",
        http: httpx.Client | None = None,
        sleep=time.sleep,
        scorer_names: dict[tuple[str, str], str] | None = None,
    ):
        self.base = f"/v1/namespaces/{namespace}/scopes/{quote(scope_ref, safe='')}"
        self._owns_http = http is None
        self._http = http or ces_http(base_url, api_key)
        self._sleep = sleep
        # A Scorer version's name never changes, so callers may share one cache across clients.
        self._scorer_names = {} if scorer_names is None else scorer_names

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params=None,
        body=None,
        key: str | None = None,
        slot: bool = False,
        shrinkable: bool = False,
    ) -> dict:
        """Retries network errors, 429, and 5xx with doubling waits, waiting at least any
        Retry-After. Raises the last error. A `slot` request takes a CES query slot: it waits
        for one of QUERY_SLOTS, holding it only while the request is in flight. A `shrinkable`
        page raises 503 `storage_capacity_exceeded` at once, because the pager answers it with a
        smaller page instead."""
        headers = {"Idempotency-Key": key} if key else {}
        for attempt in range(1, ATTEMPTS + 1):
            backoff = 2 ** (attempt - 1)
            try:
                with _query_slots if slot else contextlib.nullcontext():
                    response = self._http.request(
                        method, self.base + path, params=params, json=body, headers=headers
                    )
            except httpx.TransportError:
                if attempt == ATTEMPTS:
                    raise
                self._sleep(backoff)
                continue
            code = _error_code(response) if response.status_code == 503 else None
            if (
                response.status_code != 429
                and response.status_code < 500
                or (shrinkable and code == "storage_capacity_exceeded")
                or attempt == ATTEMPTS
            ):
                response.raise_for_status()
                return response.json()
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.replace(".", "", 1).isdigit():
                backoff = max(float(retry_after), backoff)
            self._sleep(min(backoff, MAX_RETRY_AFTER_S))
        raise AssertionError("unreachable")

    def trajectory_cases(self, trajectory_id: str) -> list[dict]:
        return self._rows(f"/trajectories/{trajectory_id}/cases", None, items="cases", slot=False)

    def root_case_input(self, trajectory_id: str) -> dict:
        """The input of the Case the Trajectory starts from: for a session, the session Case.
        Listing a Trajectory's Cases leaves its root out, so this reads the Trajectory, then
        the Case."""
        root = self._request("GET", f"/trajectories/{trajectory_id}")["root_case_id"]
        return self._request("GET", f"/cases/{root}").get("input") or {}

    def read_session(
        self, trajectory_id: str, *, subagents: bool = True, counts: SubagentCounts | None = None
    ) -> SessionView:
        """S's Cases plus every subagent Trajectory below it, as Monopole turns. Without
        `subagents`, the subagents S launched are named but not read, so they have no type or
        tool calls; a session can have hundreds. With `counts`, each subagent they cover gets
        its type and tool-call count from them, and no tool calls or tokens, instead of being
        read."""
        cases = self.trajectory_cases(trajectory_id)
        launched = child_ids(cases)
        if not subagents:
            return session_view(trajectory_id, cases, {c: [] for c in launched})
        return session_view(
            trajectory_id, cases, *self._descendants(launched, trajectory_id, counts)
        )

    def read_session_tree(self, trajectory_id: str) -> tuple[SessionView, list[SessionView]]:
        """S read in full, and each subagent Trajectory below S as its own view, for scoring."""
        cases = self.trajectory_cases(trajectory_id)
        children, _spans = self._descendants(child_ids(cases), trajectory_id)
        return session_tree(trajectory_id, cases, children)

    def _descendants(
        self, roots: list[str], session_id: str, counts: SubagentCounts | None = None
    ) -> tuple[dict[str, list[dict]], dict[str, SubagentSpan]]:
        """The Cases of these subagent Trajectories of session S and every one below them, in
        launch order, each level read concurrently, and the spans of those `counts` covers,
        which aren't read and have no Cases."""
        children: dict[str, list[dict]] = {}
        spans: dict[str, SubagentSpan] = {}
        level = roots
        with ThreadPoolExecutor(max_workers=CONCURRENT_READS) as pool:
            while level := [c for c in dict.fromkeys(level) if c not in children]:
                if counts is not None:
                    spans.update((c, span) for c in level if (span := counts.span(session_id, c)))
                unread = [c for c in level if c not in spans]
                read = dict(zip(unread, pool.map(self.trajectory_cases, unread)))
                children.update((c, read.get(c, [])) for c in level)
                level = [c for child in level for c in child_ids(children[child])]
        return children, spans

    def subagent_index(self, catalog: Catalog) -> SubagentIndex:
        """The Dataset's Trajectory listing and its Case counts by session, agent ID, and kind,
        read concurrently (`SubagentCounts`)."""
        scorer = catalog.scorers[
            min(n for n, r in catalog.scorers.items() if r.spec.target_kind == "case")
        ]
        body = {
            "name": "monopole subagent cases by kind",
            "definition": {
                "v": 2,
                "target_kind": "case",
                # CES takes no filter on a saved query, so the main thread's Cases (agent ID null)
                # form a few groups per session too.
                "group_by": [
                    {"kind": "dataset_field", "source": "input", "name": name}
                    for name in ("session_trajectory_id", "agent_id", "kind")
                ],
                # CES requires a metric; a group's `targets` is its Case count.
                "metrics": [
                    {
                        "name": "cases",
                        "scorer_version_id": scorer.version_id,
                        "aggregation": "count",
                    }
                ],
            },
        }
        evaluation = catalog.turns_evaluation_id
        with ThreadPoolExecutor(max_workers=2) as pool:
            listing = pool.submit(
                self._rows,
                f"/datasets/{catalog.dataset_id}/trajectories",
                None,
                items="trajectories",
                slot=False,
            )
            query_id = self.create_query(
                evaluation, body, key=f"monopole-query-v1:{evaluation}:{content_digest(body)}"
            )
            rows = self.query_results(evaluation, query_id)
            trajectories = {t["trajectory_id"]: t for t in listing.result()}
        cases: dict[tuple[str, str], dict[str, int]] = {}
        for row in rows:
            session, agent, kind = row["group"]
            if agent is not None:
                cases.setdefault((session, agent), {})[kind] = row["targets"]
        return SubagentIndex(trajectories, cases)

    def turn_subagents(
        self, trajectory_id: str, turn_case_id: str, counts: SubagentCounts | None = None
    ) -> list[SubagentSpan] | None:
        """The subagents one turn of S launched, nested ones included, read in full or, with
        `counts`, from them. None when S doesn't exist or has no turn whose first Case is
        `turn_case_id`."""
        try:
            cases = self.trajectory_cases(trajectory_id)
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise
            return None
        turn_id = next(
            ((c.get("input") or {}).get("turn_id") for c in cases if c["case_id"] == turn_case_id),
            None,
        )
        steps = [c for c in cases if turn_id and (c.get("input") or {}).get("turn_id") == turn_id]
        view = session_view(
            trajectory_id, steps, *self._descendants(child_ids(steps), trajectory_id, counts)
        )
        turn = next(iter(view.turns), None)
        return turn.subagents if turn is not None and turn.trace_id == turn_case_id else None

    def session_turn_rows(
        self, catalog: Catalog, session_ids: list[str], *, columns: dict | None = None
    ) -> dict[str, list]:
        """Each session's own turn rows with their Scores, or only the CES `columns` asked for,
        which must include the input's `session_trajectory_id`.

        CES runs only a few searches at a time and rejects the rest with 503 and a 1 s
        Retry-After, so sessions are read in batches with one search each. A deployment whose
        filters lack OR gets one search per session, a few at a time."""
        path = f"/evaluations/{catalog.turns_evaluation_id}/rows/search"
        batches = [
            session_ids[i : i + SESSIONS_PER_SEARCH]
            for i in range(0, len(session_ids), SESSIONS_PER_SEARCH)
        ]
        found: dict[str, list] = {sid: [] for sid in session_ids}
        selected = {} if columns is None else {"columns": columns}
        try:
            with ThreadPoolExecutor(max_workers=QUERY_SLOTS) as pool:
                for rows in pool.map(
                    lambda batch: self._rows(path, {"filter": _turns_of_any(batch), **selected}),
                    batches,
                ):
                    for row in rows:
                        if (sid := row["input"].get("session_trajectory_id")) in found:
                            found[sid].append(row)
            return found
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 400:
                raise
        with ThreadPoolExecutor(max_workers=QUERY_SLOTS) as pool:
            rows = pool.map(
                lambda sid: self._rows(path, {"filter": _turns_of(sid), **selected}), session_ids
            )
            return dict(zip(session_ids, rows))

    def _rows(
        self,
        path: str,
        body: dict | None,
        limit: int | None = None,
        *,
        params: dict | None = None,
        items: str = "rows",
        slot: bool = True,
    ) -> list[dict]:
        """Every row of a list (GET, with `params` such as a filter) or search (POST), or the
        first `limit`. A `slot` read takes a CES query slot, as row searches, Trajectory rows,
        and EvaluationQuery results do; a Trajectory's Case list doesn't."""
        rows = []
        for page in self._pages(path, body, params=params, slot=slot):
            rows.extend(page.get(items) or [])
            if limit is not None and len(rows) >= limit:
                return rows[:limit]
        return rows

    def _pages(
        self, path: str, body: dict | None, *, params: dict | None = None, slot: bool = True
    ):
        """Each page of a list (GET) or search (POST), so a caller can stop early. A page of
        large Cases or Scores can exceed CES's per-query storage budget, which CES rejects
        with 503 storage_capacity_exceeded, so the page starts at PAGE_LIMIT and halves on that
        error until CES accepts it. After GROW_AFTER accepted pages in a row it doubles again,
        since the refusals come and go with CES load and with the rows in the page. At
        CASE_PAGE_LIMIT it stops shrinking and gets the usual retries, since CES also refuses
        small pages now and then."""
        cursor, page_limit, accepted = None, PAGE_LIMIT, 0
        while True:
            paging = {"limit": page_limit, **({"cursor": cursor} if cursor else {})}
            shrinkable = page_limit > CASE_PAGE_LIMIT
            try:
                page = (
                    self._request(
                        "GET",
                        path,
                        params={**(params or {}), **paging},
                        slot=slot,
                        shrinkable=shrinkable,
                    )
                    if body is None
                    else self._request(
                        "POST", path, body={**body, **paging}, slot=slot, shrinkable=shrinkable
                    )
                )
            except httpx.HTTPStatusError as error:
                if not shrinkable or _error_code(error.response) != "storage_capacity_exceeded":
                    raise
                page_limit, accepted = max(page_limit // 2, CASE_PAGE_LIMIT), 0
                continue
            yield page
            accepted += 1
            if accepted == GROW_AFTER and page_limit < PAGE_LIMIT:
                page_limit, accepted = min(page_limit * 2, PAGE_LIMIT), 0
            cursor = page.get("next_cursor")
            if not cursor:
                return

    def ensure_catalog(
        self, dataset_id: str, specs: tuple[ScorerSpec, ...] = DETERMINISTIC
    ) -> Catalog:
        """Create, or find again by exact replay, the two Evaluations and the Scorers."""
        evaluations = {}
        for kind, label in (("case", "turns"), ("trajectory", "sessions")):
            body = {
                "name": f"monopole {label}",
                "primary_target_kind": kind,
                "source": {"kind": "live", "dataset_id": dataset_id},
            }
            evaluations[kind] = self._request(
                "POST",
                "/evaluations",
                body=body,
                key=f"monopole-evaluation-v1:{dataset_id}:{label}",
            )["evaluation_id"]
        catalog = Catalog(evaluations["case"], evaluations["trajectory"], {}, dataset_id)
        return self.add_scorers(catalog, specs)

    def add_scorers(self, catalog: Catalog, specs: tuple[ScorerSpec, ...]) -> Catalog:
        """Create each spec's value and details Scorers and add them to the matching Evaluation.
        The creates are independent, so they run concurrently."""

        def create(name: str, value_type: str, spec: ScorerSpec) -> dict:
            body = {
                "name": name,
                "value_type": value_type,
                "allowed_target_kinds": [spec.target_kind],
            }
            if spec.description:
                body["description"] = spec.description
            made = self._request(
                "POST",
                "/scorers",
                body=body,
                key=f"monopole-scorer-v1:{name}:{value_type}:{spec.target_kind}",
            )
            # Reads name a Score by its Scorer version; this one needs no lookup.
            self._scorer_names[(made["scorer_id"], made["version_id"])] = name
            return made

        with ThreadPoolExecutor(max_workers=CONCURRENT_READS) as pool:
            created = [
                (
                    spec,
                    pool.submit(create, spec.name, spec.value_type, spec),
                    pool.submit(create, spec.name + DETAILS_SUFFIX, "json", spec),
                )
                for spec in specs
            ]
            pairs = [(spec, (value.result(), details.result())) for spec, value, details in created]
        refs, selected = dict(catalog.scorers), {"case": [], "trajectory": []}
        for spec, made in pairs:
            refs[spec.name] = ScorerRef(
                spec,
                made[0]["scorer_id"],
                made[0]["version_id"],
                made[1]["scorer_id"],
                made[1]["version_id"],
            )
            # Comparison Scores are read by ID, so no Evaluation selects their Scorers.
            if spec.target_kind in selected:
                selected[spec.target_kind] += [
                    {"scorer_id": m["scorer_id"], "version_id": m["version_id"]} for m in made
                ]
        for kind, scorers in selected.items():
            if scorers:
                self.select_scorers(catalog.evaluation_for(kind), scorers)
        return replace(catalog, scorers=refs)

    def select_scorers(self, evaluation_id: str, scorers: list[dict]) -> list[dict]:
        """Add Scorer versions, given as {scorer_id, version_id}, to the Evaluation, and return
        every version it now selects. Adding versions it already selects changes nothing."""
        return self._request(
            "PATCH", f"/evaluations/{evaluation_id}/scorers", body={"scorers": scorers}
        )["scorers"]

    def create_query(self, evaluation_id: str, body: dict, *, key: str) -> str:
        """Save an EvaluationQuery definition and return its ID. A replay returns the same ID."""
        return self._request(
            "POST",
            f"/evaluations/{evaluation_id}/queries",
            body=body,
            key=key,
        )["evaluation_query_id"]

    def query_results(
        self, evaluation_id: str, query_id: str, limit: int | None = None
    ) -> list[dict]:
        """Every result row of the query on the Evaluation's latest data, or its first `limit`."""
        return self._rows(f"/evaluations/{evaluation_id}/queries/{query_id}/results", None, limit)

    def create_comparison(
        self, members: list[tuple[str, str]], *, key: str, attributes: dict | None = None
    ) -> str:
        """Group published Cases, given as (Case ID, role) pairs, into one Comparison."""
        comparison: dict[str, Any] = {
            "members": [{"case_id": case_id, "role": role} for case_id, role in members]
        }
        if attributes:
            comparison["attributes"] = attributes
        result = self._request("POST", "/comparisons", body={"comparisons": [comparison]}, key=key)
        return result["comparison_ids"][0]

    def find_sessions(
        self,
        catalog: Catalog,
        *,
        since: datetime | None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Session Trajectories with a turn that started in the window, newest first. With no
        `since` the window has no start, so pass a `limit` to keep the read bounded."""
        columns = {"input": ["session_trajectory_id", "started_at"]}
        found, _ = self.recent_turn_rows(
            catalog, since=since, until=until, limit=limit, columns=columns
        )
        return list(found)

    def recent_turn_rows(
        self,
        catalog: Catalog,
        *,
        since: datetime | None,
        until: datetime | None = None,
        limit: int | None = None,
        columns: dict | None = None,
    ) -> tuple[dict[str, list[dict]], bool]:
        """The turn rows read while finding the sessions with a turn in the window, by session,
        newest activity first, and whether they are all of those sessions' turn rows. They are
        when the read reached the Dataset's oldest turn; a read that stops at `since` or at
        `limit` sessions leaves their older turns unread. With `columns`, the rows hold only
        those CES columns, which must include the input's `session_trajectory_id` and
        `started_at`.

        CES filters can't range-compare string fields, so this reads turns newest first and
        stops at the first one before `since`. The adapter's timestamps sort in time order."""
        seen: dict[str, list[dict]] = {}
        found: dict[str, None] = {}
        path = f"/evaluations/{catalog.turns_evaluation_id}/rows/search"
        body = {"filter": FIRST_STEP, "order_by": "i.started_at desc"}
        if columns is not None:
            body["columns"] = columns
        for page in self._pages(path, body):
            for row in page.get("rows") or []:
                started, session = (
                    parse_utc(row["input"].get("started_at")),
                    row["input"].get("session_trajectory_id"),
                )
                if session:
                    seen.setdefault(session, []).append(row)
                if started is None:
                    continue  # a turn without a start time can't be placed in the window
                if since is not None and started < since:
                    return {sid: seen[sid] for sid in found}, False
                if session and (until is None or started <= until):
                    found.setdefault(session, None)
                    if limit is not None and len(found) >= limit:
                        return {sid: seen[sid] for sid in found}, False
        return {sid: seen[sid] for sid in found}, True

    def turn_scores(
        self, catalog: Catalog, trajectory_id: str, *, details: bool = True
    ) -> dict[str, list[dict]]:
        """Scores on each turn of one session, keyed by the turn's first Case ID. Without
        `details`, only the value Scores, which tell which Scorers scored the turn."""
        body: dict[str, Any] = {"filter": _turns_of(trajectory_id)}
        if not details:
            body["columns"] = {"score": _turn_scorers(catalog, lambda _name: False)}
        rows = self._rows(f"/evaluations/{catalog.turns_evaluation_id}/rows/search", body)
        return {r["case_id"]: r.get("scores") or [] for r in rows}

    def _session_rows(self, catalog: Catalog, trajectory_id: str, subagents: bool) -> list[dict]:
        """One session Trajectory's row, and with `subagents` the row of every subagent
        Trajectory below it, at any depth. The adapter writes the session's Trajectory ID as
        `session_trajectory_id` on every Case of the session, so on the root Case of each
        subagent too, and one filtered read finds them all. Nothing when CES has no such
        session."""
        path = f"/evaluations/{catalog.sessions_evaluation_id}/trajectory-rows"
        try:
            rows = [self._request("GET", f"{path}/{trajectory_id}", slot=True)]
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise
            return []
        if subagents:
            below = f"i.session_trajectory_id = {json.dumps(trajectory_id)}"
            rows += self._rows(path, None, params={"filter": below})
        return rows

    def _session_targets(
        self, catalog: Catalog, wanted=None, *, subagents: bool = False
    ) -> list[tuple[str, dict, list]]:
        """(Trajectory ID, root Case input, Score rows) for every session Trajectory, or only
        the wanted ones, read concurrently, with every subagent Trajectory below them when
        `subagents`."""
        if wanted is None:
            rows = self._rows(
                f"/evaluations/{catalog.sessions_evaluation_id}/trajectory-rows",
                None,
                params={"filter": "depth = 0"},
            )
        else:
            with ThreadPoolExecutor(max_workers=CONCURRENT_READS) as pool:
                found = pool.map(lambda t: self._session_rows(catalog, t, subagents), wanted)
                rows = [row for part in found for row in part]
        return [
            (
                r["trajectory"]["trajectory_id"],
                (r.get("root_case") or {}).get("input") or {},
                r.get("scores") or [],
            )
            for r in rows
        ]

    def session_scores(
        self, catalog: Catalog, trajectory_ids, *, subagents: bool = False
    ) -> dict[str, list[dict]]:
        """Score rows on session Trajectories, by Trajectory ID, and with `subagents` on every
        subagent Trajectory below them too."""
        return {
            t: scores
            for t, _, scores in self._session_targets(
                catalog, set(trajectory_ids), subagents=subagents
            )
        }

    def _score_records(self, targets: list[tuple[str, dict, list]]) -> list[dict]:
        for _, _, scores in targets:
            for score in scores:
                self.scorer_name(score["scorer_id"], score["scorer_version_id"])
        names = self._scorer_names
        records = [record for target in targets for record in _score_record(*target, names)]
        records.sort(
            key=lambda r: r["payload"]["details"].get("turn_started_at") or "", reverse=True
        )
        return records

    def scorer_name(self, scorer_id: str, version_id: str) -> str | None:
        """The Scorer version's name, or None when CES has no such version, so a Score from a
        deleted or foreign Scorer is skipped instead of failing the whole read."""
        key = (scorer_id, version_id)
        if key in self._scorer_names:
            return self._scorer_names[key]
        try:
            name = self._request("GET", f"/scorers/{scorer_id}/versions/{version_id}")["name"]
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise
            # Not cached: a Scorer version created later must still resolve.
            return None
        self._scorer_names[key] = name
        return name

    def read_all_scores(
        self, catalog: Catalog, limit: int | None = None, *, outcome_details: bool = True
    ) -> list[dict]:
        """Turn and session Scores, newest turn first, as feedback-shaped records.

        With `limit`, only the newest `limit` turns are read, with the Scores of their sessions,
        so a bound never drops session Scores for lacking a turn time. Without
        `outcome_details`, outcome Scorers' records carry the tag their value implies but no
        reason or details: rating analysis needs no more, and those details, which list every
        classified command, are most of a turn row's bytes."""
        details = (lambda _name: True) if outcome_details else (lambda n: n not in OUTCOME_TAGS)
        turn_rows = self._rows(
            f"/evaluations/{catalog.turns_evaluation_id}/rows/search",
            {
                "filter": FIRST_STEP,
                "order_by": "i.started_at desc",
                "columns": {"input": RECORD_INPUT, "score": _turn_scorers(catalog, details)},
            },
            limit=limit,
        )
        targets = [(r["case_id"], r["input"], r.get("scores") or []) for r in turn_rows]
        sessions = {r["input"].get("session_trajectory_id") for r in turn_rows}
        # One paged read of every session row instead of one read per session: CES can't filter
        # Trajectory rows by ID. It reads sessions the turns don't reach too, which costs more
        # than per-session reads only once the Dataset holds far more sessions than those.
        targets += [
            target
            for target in self._session_targets(catalog)
            if limit is None or target[0] in sessions
        ]
        return self._score_records(targets)

    def cohort_scores(
        self, catalog: Catalog, session_ids: list[str], *, turn_rows: dict | None = None
    ) -> list[dict]:
        """The Scores on these sessions and their turns, as feedback-shaped records. Pass
        `turn_rows` when the sessions' turn rows were already read."""
        if turn_rows is None:
            columns = {"input": RECORD_INPUT, "score": _turn_scorers(catalog, lambda _name: True)}
            turn_rows = self.session_turn_rows(catalog, session_ids, columns=columns)
        return self._score_records(
            _turn_targets(turn_rows) + self._session_targets(catalog, set(session_ids))
        )

    def turn_row_scores(self, turn_rows: dict[str, list[dict]]) -> list[dict]:
        """The Scores that already-read turn rows carry, as feedback-shaped records."""
        return self._score_records(_turn_targets(turn_rows))

    def write_scores(
        self, catalog: Catalog, pass_id: str, target_kind: str, target_id: str, scores: list[Score]
    ) -> list[str]:
        """Each Score and its details in one request. The pass ID keeps a later pass's
        equal value from replaying an earlier Score and hiding a newer one."""
        if not scores:
            return []
        body = []
        for score in scores:
            ref = catalog.scorers.get(score.scorer)
            if ref is None or ref.spec.target_kind != target_kind:
                raise ValueError(
                    f"scorer {score.scorer!r} is not in the CES catalog for {target_kind} targets"
                )
            value_type = (
                "boolean"
                if isinstance(score.value, bool)
                else "string"
                if isinstance(score.value, str)
                else "number"
            )
            if value_type != ref.spec.value_type:
                raise ValueError(
                    f"scorer {score.scorer!r} is {ref.spec.value_type}, got {score.value!r}"
                )
            details = {
                "scorer_version": score.version,
                "tags": score.tags,
                "reason": score.reason,
                "confidence": score.confidence,
                "granularity": score.granularity,
                "details": score.metadata,
            }
            target = {"kind": target_kind, "id": target_id}
            body += [
                {
                    "scorer_id": ref.scorer_id,
                    "scorer_version_id": ref.version_id,
                    "target": target,
                    "value": float(score.value) if value_type == "number" else score.value,
                },
                {
                    "scorer_id": ref.details_scorer_id,
                    "scorer_version_id": ref.details_version_id,
                    "target": target,
                    "value": details,
                },
            ]
        result = self._request(
            "POST",
            "/scores",
            body={"scores": body},
            key=f"score-v2:{pass_id}:{target_kind}:{target_id}:{content_digest({'scores': body})}",
        )
        return result["score_ids"]
