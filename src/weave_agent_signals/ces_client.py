"""The single place Monopole talks to CES (CoreWeave Evaluations).

Sessions come from the Dataset that weave-agent-adapter writes: a session Case
roots the session Trajectory S, whose member Cases each hold one user message,
assistant message, tool call, or compaction, and each subagent run is a child
Trajectory. The adapter's `specs/EVALUATIONS_CASE_CONTRACT.md` defines the fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from weave_agent_signals.auth import get_api_key
from weave_agent_signals.ces_catalog import (
    DETAILS_SUFFIX,
    DETERMINISTIC,
    Catalog,
    ScorerRef,
    ScorerSpec,
    judge_spec,
)
from weave_agent_signals.ces_score_index import ScoreIndex
from weave_agent_signals.models import (
    ChatSpan,
    Score,
    SessionView,
    SpanEvent,
    SubagentSpan,
    ToolSpan,
    TurnSpan,
    resolve_trace_role,
)

ATTEMPTS = 3
MAX_RETRY_AFTER_S = 30.0
PAGE_LIMIT = 1000
# A turn's row is its first Case; the adapter numbers each turn's Cases from 0.
FIRST_STEP = "i.step_index = 0"
SESSIONS_PER_SEARCH = 50
_STATUS = {"completed": "OK", "aborted": "ERROR"}


def iso(moment: datetime) -> str:
    """The adapter's timestamp format, so string comparison in filters is time order."""
    moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _span_id(case_id: str, turn_id: str | None) -> str:
    """A turn with no user prompt starts with a reply or tool call, whose Case ID is also the
    turn's ID. Judges need every evidence ID to be unique, so that span gets a suffix."""
    return f"{case_id}/span" if case_id == turn_id else case_id


def _tool_span(case: dict[str, Any], turn_id: str | None = None) -> ToolSpan:
    case_in, out = case["input"], case.get("output") or {}
    arguments = case_in.get("arguments")
    return ToolSpan(
        span_id=_span_id(case["case_id"], turn_id),
        tool_name=case_in.get("tool_name") or "",
        arguments="" if arguments is None else json.dumps(arguments, ensure_ascii=False),
        result=out.get("output") or out.get("error") or "",
        status_code="ERROR"
        if out.get("error") is not None or out.get("status") == "failed"
        else "OK"
        if out.get("status") == "completed"
        else "UNSET",
        started_at=_ts(case_in.get("started_at")) or datetime.now(timezone.utc),
        ended_at=_ts(case_in.get("ended_at")),
    )


def _digest(body: dict[str, Any]) -> str:
    """Part of a write's Idempotency-Key, so a retry with the same Scores replays them and a
    retry with changed content, such as a judge's new scored_at, writes newer ones."""
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


def _score_row(score: dict[str, Any]) -> dict[str, Any]:
    """A get-score result in the shape Evaluation rows return Scores."""
    return {k: score.get(k) for k in ("score_id", "scorer_id", "scorer_version_id", "value")}


def _feedback(target_id: str, case_input: dict, scores: list[dict], names: dict) -> list[dict]:
    """One target's Scores as feedback-shaped records, the shape the analysis code takes.

    `names` maps (scorer_id, scorer_version_id) to the CES Scorer name. A `.details` Scorer
    supplies the tags, reason, and details of the value Scorer it shadows. The target Case
    fills in the config, branch, and start time when the details lack them. A judge Scorer
    named `<rubric>.panel-<digest>` reports as its rubric."""
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
        if not isinstance(value, (bool, int, float)) or not math.isfinite(value):
            continue  # not a rating; one malformed Score must not break a whole read
        extra = details.get(name) if isinstance(details.get(name), dict) else {}
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
        rubric = name.split(".panel-")[0]
        records.append(
            {
                "id": ids[name],
                "feedback_type": f"weave_agent_signals.{rubric}",
                "target_id": target_id,
                "payload": payload,
            }
        )
    return records


def session_view(
    trajectory_id: str, cases: list[dict], children: dict[str, list[dict]]
) -> SessionView:
    """Rebuild Monopole's turns from S's Cases and its subagents' Cases.

    A turn is the Cases sharing `turn_id`, ordered by `step_index`. Its first Case
    carries the turn totals, and its ID is the turn's identity, where Weave used a
    trace ID. A subagent's Cases attach to the turn whose tool call launched it.
    """
    members = [c for c in cases if c["input"].get("kind") != "session"]
    turns: dict[str, list[dict]] = {}
    for case in sorted(
        members,
        key=lambda c: (_int(c["input"].get("turn_index")), _int(c["input"].get("step_index"))),
    ):
        turns.setdefault(case["input"]["turn_id"], []).append(case)
    launched = {
        child: case["case_id"]
        for case in [*members, *(c for cs in children.values() for c in cs)]
        for child in case.get("child_trajectory_ids") or []
    }
    owner = {c["case_id"]: child for child, cs in children.items() for c in cs}

    def launch_case(child: str) -> str | None:
        """The S Case a subagent descends from, following nested subagents up to S."""
        case_id, seen = launched.get(child), {child}
        while case_id in owner and owner[case_id] not in seen:
            seen.add(owner[case_id])
            case_id = launched.get(owner[case_id])
        return case_id

    built = []
    for steps in turns.values():
        first, totals = steps[0]["input"], steps[0].get("output") or {}
        texts = [c["input"].get("text") for c in steps if c["input"]["kind"] == "user_message"]
        answers = [
            (c.get("output") or {}).get("text")
            for c in steps
            if c["input"]["kind"] == "assistant_message"
        ]
        stamps = [
            s
            for c in steps
            for s in (_ts(c["input"].get("started_at")), _ts(c["input"].get("ended_at")))
            if s
        ]
        step_ids = {c["case_id"] for c in steps}
        subagents = [
            SubagentSpan(
                span_id=child,
                agent_type=None,
                tool_calls=[
                    _tool_span(c) for c in child_cases if c["input"].get("kind") == "tool_call"
                ],
            )
            for child, child_cases in children.items()
            if launch_case(child) in step_ids
        ]
        user_input = next((t for t in texts if t), None)
        built.append(
            TurnSpan(
                trace_id=steps[0]["case_id"],
                conversation_id=trajectory_id,
                started_at=min(stamps) if stamps else datetime.now(timezone.utc),
                ended_at=max(stamps) if stamps else None,
                model=first.get("model"),
                input_tokens=_int(totals.get("turn_input_tokens")),
                output_tokens=_int(totals.get("turn_output_tokens")),
                cache_read_tokens=_int(totals.get("turn_cache_read_tokens")),
                status_code=_STATUS.get(totals.get("turn_status"), "UNSET"),
                config_version=first.get("config_version"),
                git_branch=first.get("git_branch"),
                effort_level=first.get("effort"),
                session_id=first.get("session_id"),
                steering_count=_int(totals.get("turn_steering_count")),
                denial_count=_int(totals.get("turn_denial_count")),
                tool_error_count=_int(totals.get("turn_tool_error_count")),
                events=[
                    SpanEvent("compaction", _ts(c["input"].get("started_at")))
                    for c in steps
                    if c["input"]["kind"] == "compaction"
                ],
                tool_calls=[
                    _tool_span(c, steps[0]["case_id"])
                    for c in steps
                    if c["input"]["kind"] == "tool_call"
                ],
                chat_spans=[
                    ChatSpan(
                        span_id=_span_id(c["case_id"], steps[0]["case_id"]),
                        model=first.get("model") or "",
                        input_tokens=_int((c.get("output") or {}).get("input_tokens")),
                        output_tokens=_int((c.get("output") or {}).get("output_tokens")),
                        cache_read_tokens=_int((c.get("output") or {}).get("cache_read_tokens")),
                        cache_creation_tokens=_int(
                            (c.get("output") or {}).get("cache_write_tokens")
                        ),
                        finish_reason=(c.get("output") or {}).get("finish_reason"),
                    )
                    for c in steps
                    if c["input"]["kind"] == "assistant_message"
                ],
                subagents=subagents,
                user_input=user_input,
                assistant_output=next((a for a in reversed(answers) if a), None),
                trace_role=resolve_trace_role(first.get("trace_role"), user_input),
                case_ids=[c["case_id"] for c in steps],
            )
        )
    first_turn = built[0] if built else None
    return SessionView(
        conversation_id=trajectory_id,
        turns=built,
        config_version=first_turn.config_version if first_turn else None,
        git_branch=first_turn.git_branch if first_turn else None,
    )


def _turns_of(session_ids: list[str]) -> str:
    """A rows/search filter for the turns of these sessions."""
    either = " OR ".join(f"i.session_trajectory_id = {json.dumps(sid)}" for sid in session_ids)
    return f"{FIRST_STEP} AND ({either})"


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
        score_index: ScoreIndex | None = None,
    ):
        self.base = f"/v1/namespaces/{namespace}/scopes/{quote(scope_ref, safe='')}"
        self._http = http or httpx.Client(
            base_url=base_url.rstrip("/"), auth=("api", api_key or get_api_key()), timeout=60.0
        )
        self._sleep = sleep
        self._index = score_index or ScoreIndex()
        self._scorer_names: dict[tuple[str, str], str] = {}  # a Scorer version's name never changes

    def close(self) -> None:
        self._http.close()

    def _request(
        self, method: str, path: str, *, params=None, body=None, key: str | None = None
    ) -> dict:
        """Retries network errors, 429, and 5xx, honoring Retry-After. Raises the last error."""
        headers = {"Idempotency-Key": key} if key else {}
        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = self._http.request(
                    method, self.base + path, params=params, json=body, headers=headers
                )
            except httpx.TransportError:
                if attempt == ATTEMPTS:
                    raise
                self._sleep(2 ** (attempt - 1))
                continue
            if response.status_code != 429 and response.status_code < 500 or attempt == ATTEMPTS:
                response.raise_for_status()
                return response.json()
            retry_after = response.headers.get("Retry-After", "")
            delay = (
                float(retry_after)
                if retry_after.replace(".", "", 1).isdigit()
                else 2 ** (attempt - 1)
            )
            self._sleep(min(delay, MAX_RETRY_AFTER_S))
        raise AssertionError("unreachable")

    def trajectory_cases(self, trajectory_id: str) -> list[dict]:
        cases, cursor = [], None
        while True:
            params = {"limit": PAGE_LIMIT, **({"cursor": cursor} if cursor else {})}
            page = self._request("GET", f"/trajectories/{trajectory_id}/cases", params=params)
            cases.extend(page.get("cases") or [])
            cursor = page.get("next_cursor")
            if not cursor:
                return cases

    def read_session(self, trajectory_id: str) -> SessionView:
        """S's Cases plus every subagent Trajectory below it, as Monopole turns."""
        cases = self.trajectory_cases(trajectory_id)
        children: dict[str, list[dict]] = {}
        pending = [child for case in cases for child in case.get("child_trajectory_ids") or []]
        while pending:
            child = pending.pop(0)
            if child in children:
                continue
            children[child] = self.trajectory_cases(child)
            pending.extend(
                c for case in children[child] for c in case.get("child_trajectory_ids") or []
            )
        return session_view(trajectory_id, cases, children)

    def _rows(self, path: str, body: dict, limit: int | None = None) -> list[dict]:
        rows, cursor = [], None
        while True:
            page = self._request(
                "POST",
                path,
                body={**body, "limit": PAGE_LIMIT, **({"cursor": cursor} if cursor else {})},
            )
            rows.extend(page.get("rows") or [])
            if limit is not None and len(rows) >= limit:
                return rows[:limit]
            cursor = page.get("next_cursor")
            if not cursor:
                return rows

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
        return self.add_scorers(Catalog(evaluations["case"], evaluations["trajectory"], {}), specs)

    def add_scorers(self, catalog: Catalog, specs: tuple[ScorerSpec, ...]) -> Catalog:
        """Create each spec's value and details Scorers and add them to the matching Evaluation."""
        refs, selected = dict(catalog.scorers), {"case": [], "trajectory": []}
        for spec in specs:
            made = []
            for name, value_type in (
                (spec.name, spec.value_type),
                (spec.name + DETAILS_SUFFIX, "json"),
            ):
                body = {
                    "name": name,
                    "value_type": value_type,
                    "allowed_target_kinds": [spec.target_kind],
                }
                if spec.description:
                    body["description"] = spec.description
                made.append(
                    self._request(
                        "POST",
                        "/scorers",
                        body=body,
                        key=f"monopole-scorer-v1:{name}:{value_type}:{spec.target_kind}",
                    )
                )
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
                # Adding Scorers an Evaluation already selects returns the same selection.
                self._request(
                    "PATCH",
                    f"/evaluations/{catalog.evaluation_for(kind)}/scorers",
                    body={"scorers": scorers},
                )
        return Catalog(catalog.turns_evaluation_id, catalog.sessions_evaluation_id, refs)

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
        since: datetime,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Session Trajectories with a turn that started in the window, newest first.

        CES filters can't range-compare string fields, so this reads turns newest first and
        stops at the first one before `since`. The adapter's timestamps sort in time order."""
        lo, hi = iso(since), iso(until) if until is not None else None
        found, cursor = {}, None
        while True:
            body = {
                "filter": FIRST_STEP,
                "order_by": "i.started_at desc",
                "limit": PAGE_LIMIT,
                **({"cursor": cursor} if cursor else {}),
            }
            page = self._request(
                "POST", f"/evaluations/{catalog.turns_evaluation_id}/rows/search", body=body
            )
            for row in page.get("rows") or []:
                started, session = (
                    row["input"].get("started_at"),
                    row["input"].get("session_trajectory_id"),
                )
                if not started:
                    continue  # a turn without a start time can't be placed in the window
                if started < lo:
                    return list(found)
                if session and (hi is None or started <= hi):
                    found.setdefault(session, None)
                    if limit is not None and len(found) >= limit:
                        return list(found)
            cursor = page.get("next_cursor")
            if not cursor:
                return list(found)

    def turn_scores(self, catalog: Catalog, trajectory_id: str) -> dict[str, list[dict]]:
        """Scores on each turn of one session, keyed by the turn's first Case ID."""
        rows = self._rows(
            f"/evaluations/{catalog.turns_evaluation_id}/rows/search",
            {"filter": _turns_of([trajectory_id])},
        )
        return {r["case_id"]: r.get("scores") or [] for r in rows}

    def _session_targets(self, catalog: Catalog, wanted=None) -> list[tuple[str, dict, list]]:
        """(Trajectory ID, session Case input, Score rows) for each session Trajectory, or only
        the wanted ones. The route has no filter, so this pages every row."""
        try:
            rows = self._rows(
                f"/evaluations/{catalog.sessions_evaluation_id}/trajectory-rows/search", {}
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in (404, 405):
                raise
            # Workaround until the Trajectory-row route deploys; see ces_score_index.
            return [
                (trajectory, {}, [_score_row(self.get_score(sid)) for sid in newest.values()])
                for trajectory, newest in self._index.newest(wanted).items()
            ]
        return [
            (
                r["trajectory"]["trajectory_id"],
                (r.get("root_case") or {}).get("input") or {},
                r.get("scores") or [],
            )
            for r in rows
            if not r["trajectory"].get("depth")
            and (wanted is None or r["trajectory"]["trajectory_id"] in wanted)
        ]

    def session_scores(self, catalog: Catalog, trajectory_ids) -> dict[str, list[dict]]:
        """Score rows on session Trajectories, by Trajectory ID."""
        return {t: scores for t, _, scores in self._session_targets(catalog, set(trajectory_ids))}

    def _feedback_records(self, targets: list[tuple[str, dict, list]]) -> list[dict]:
        names = self._scorer_names
        for _, _, scores in targets:
            for score in scores:
                key = (score["scorer_id"], score["scorer_version_id"])
                if key not in names:
                    names[key] = self._request("GET", f"/scorers/{key[0]}/versions/{key[1]}")[
                        "name"
                    ]
        records = [record for target in targets for record in _feedback(*target, names)]
        records.sort(
            key=lambda r: r["payload"]["details"].get("turn_started_at") or "", reverse=True
        )
        return records

    def read_all_scores(self, catalog: Catalog, limit: int | None = None) -> list[dict]:
        """Turn and session Scores, newest turn first, as feedback-shaped records.

        With `limit`, only the newest `limit` turns are read, with the Scores of their sessions,
        so a bound never drops session Scores for lacking a turn time."""
        turn_rows = self._rows(
            f"/evaluations/{catalog.turns_evaluation_id}/rows/search",
            {"filter": FIRST_STEP, "order_by": "i.started_at desc"},
            limit=limit,
        )
        targets = [(r["case_id"], r["input"], r.get("scores") or []) for r in turn_rows]
        sessions = (
            None if limit is None else {r["input"].get("session_trajectory_id") for r in turn_rows}
        )
        return self._feedback_records(targets + self._session_targets(catalog, sessions))

    def cohort_feedback(self, catalog: Catalog, session_ids: list[str]) -> list[dict]:
        """The Scores on these sessions and their turns, as feedback-shaped records."""
        targets = []
        for start in range(0, len(session_ids), SESSIONS_PER_SEARCH):
            rows = self._rows(
                f"/evaluations/{catalog.turns_evaluation_id}/rows/search",
                {"filter": _turns_of(session_ids[start : start + SESSIONS_PER_SEARCH])},
            )
            targets += [(r["case_id"], r["input"], r.get("scores") or []) for r in rows]
        return self._feedback_records(targets + self._session_targets(catalog, set(session_ids)))

    def get_score(self, score_id: str) -> dict:
        return self._request("GET", f"/scores/{score_id}")

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
            boolean = isinstance(score.value, bool)
            if boolean != (ref.spec.value_type == "boolean"):
                raise ValueError(
                    f"scorer {score.scorer!r} is {ref.spec.value_type}, got {score.value!r}"
                )
            details = {
                "scorer_version": "v1",
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
                    "value": score.value if boolean else float(score.value),
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
            key=f"score-v2:{pass_id}:{target_kind}:{target_id}:{_digest({'scores': body})}",
        )
        if target_kind == "trajectory":
            self._index.record(
                target_id, list(zip((b["scorer_id"] for b in body), result["score_ids"]))
            )
        return result["score_ids"]
