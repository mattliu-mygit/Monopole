"""FastAPI REST API wrapping the CLI operations (spec 07).

Thin layer: every endpoint delegates to the same functions the CLI calls.
Long-running ops run as background tasks with status polling.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from weave_agent_signals import alerts
from weave_agent_signals.client import WeaveClient
from weave_agent_signals.jobstore import JobStore
from weave_agent_signals.judges.cli_backend import CliJudgeClient
from weave_agent_signals.judges.inference import InferenceClient
from weave_agent_signals.judges.rubrics import RUBRICS, SESSION_RUBRICS
from weave_agent_signals.judges.runner import (
    _BACKEND_ROSTERS,
    judge_default_model,
    judge_session,
    judge_turn,
)
from weave_agent_signals.models import Score, SessionView, TurnSpan
from weave_agent_signals.patterns import (
    ab_leaderboard,
    aggregate_scores,
    coaching_digest,
    detect_config_regressions,
    detect_regressions,
)
from weave_agent_signals.reflector import (
    extract_artifacts,
    make_cli_lm,
    render_proposal_diff,
    run_reflection,
)
from weave_agent_signals.runs import DataSelection, Run, RunStatus, RunStore
from weave_agent_signals.scorers import score_session, score_turn

log = logging.getLogger("weave_agent_signals.api")

load_dotenv()

ENTITY = os.environ.get("WANDB_ENTITY", "mliu-wandb-weights-biases")
PROJECT = os.environ.get("WANDB_PROJECT", "agent-sessions")
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())

# ---------------------------------------------------------------------------
# Job store + Run store (SQLite-backed, persist across restarts)
# ---------------------------------------------------------------------------

_job_store = JobStore()
_run_store = RunStore()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> WeaveClient:
    return WeaveClient(entity=ENTITY, project=PROJECT)


def _make_judge_client(backend: str | None):
    if backend == "cli":
        return CliJudgeClient()
    return InferenceClient(entity=ENTITY, project=PROJECT, backend=backend)


def _turn_to_dict(turn: TurnSpan, *, include_children: bool = False) -> dict:
    d = {
        "trace_id": turn.trace_id,
        "conversation_id": turn.conversation_id,
        "started_at": turn.started_at.isoformat() if turn.started_at else None,
        "ended_at": turn.ended_at.isoformat() if turn.ended_at else None,
        "model": turn.model,
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "cache_read_tokens": turn.cache_read_tokens,
        "status_code": turn.status_code,
        "config_version": turn.config_version,
        "git_branch": turn.git_branch,
        "effort_level": turn.effort_level,
        "session_id": turn.session_id,
        "steering_count": turn.steering_count,
        "denial_count": turn.denial_count,
        "tool_error_count": turn.tool_error_count,
        "tool_call_count": len(turn.tool_calls),
        "chat_span_count": len(turn.chat_spans),
        "subagent_count": len(turn.subagents),
        "user_input": turn.user_input,
    }
    if include_children:
        d["tool_calls"] = [
            {
                "span_id": tc.span_id,
                "tool_name": tc.tool_name,
                "arguments": tc.arguments or "",
                "result": tc.result or "",
                "status_code": tc.status_code,
                "started_at": tc.started_at.isoformat() if tc.started_at else None,
                "ended_at": tc.ended_at.isoformat() if tc.ended_at else None,
            }
            for tc in turn.tool_calls
        ]
        d["chat_spans"] = [
            {
                "span_id": cs.span_id,
                "model": cs.model,
                "input_tokens": cs.input_tokens,
                "output_tokens": cs.output_tokens,
                "cache_read_tokens": cs.cache_read_tokens,
                "finish_reason": cs.finish_reason,
            }
            for cs in turn.chat_spans
        ]
        d["subagents"] = [
            {
                "span_id": sa.span_id,
                "agent_type": sa.agent_type,
                "tool_call_count": len(sa.tool_calls),
            }
            for sa in turn.subagents
        ]
    return d


def _session_summary(conv_id: str, turns: list[TurnSpan]) -> dict:
    ordered = sorted(turns, key=lambda t: t.started_at)
    first = ordered[0] if ordered else None
    last = ordered[-1] if ordered else None
    models = sorted({t.model for t in ordered if t.model})
    first_input = first.user_input if first else None
    return {
        "conversation_id": conv_id,
        "session_id": first.session_id if first else None,
        "turn_count": len(ordered),
        "started_at": first.started_at.isoformat() if first else None,
        "ended_at": last.ended_at.isoformat() if last and last.ended_at else None,
        "model": models[0] if len(models) == 1 else ", ".join(models) if models else None,
        "effort_level": first.effort_level if first else None,
        "config_version": first.config_version if first else None,
        "git_branch": first.git_branch if first else None,
        "total_tokens": sum(t.input_tokens + t.output_tokens for t in ordered),
        "total_tool_calls": sum(t.tool_error_count for t in ordered),
        "input_preview": first_input,
    }


def _group_turns_by_session(turns: list[TurnSpan]) -> dict[str, list[TurnSpan]]:
    by_conv: dict[str, list[TurnSpan]] = {}
    for t in turns:
        by_conv.setdefault(t.conversation_id, []).append(t)
    return by_conv


def _stamp_metadata(score: Score, *, config_version, git_branch, run_time) -> None:
    score.metadata.setdefault("config_version", config_version)
    score.metadata.setdefault("git_branch", git_branch)
    if run_time is not None:
        score.metadata.setdefault("turn_started_at", run_time.isoformat())


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    since: str | None = None
    limit: int = 100
    dry_run: bool = False
    force: bool = False


class BackfillRequest(BaseModel):
    start: str
    end: str | None = None
    page_size: int = 500
    dry_run: bool = False
    force: bool = False


class JudgeRequest(BaseModel):
    since: str | None = None
    limit: int = 10
    rubrics: list[str] | None = None
    judge_backend: str | None = "cli"
    panel_size: int = 1
    dry_run: bool = False
    force: bool = False


class ReflectRequest(BaseModel):
    limit: int = 500
    model: str = "gpt-4o"
    judge_backend: str | None = "cli"
    iterations: int = 3
    dry_run: bool = False


class ApplyRequest(BaseModel):
    job_id: str


class MonitorRequest(BaseModel):
    limit: int = 1000
    alert_webhook: str | None = None
    dry_run: bool = False


class SelectionRequest(BaseModel):
    since: str | None = None
    until: str | None = None
    session_ids: list[str] | None = None
    excluded_session_ids: list[str] | None = None


class AdvanceRequest(BaseModel):
    judge_backend: str = "cli"
    panel_size: int = 1
    rubrics: list[str] | None = None
    model: str = "gpt-4o"
    iterations: int = 3
    dry_run: bool = False
    force: bool = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="weave-agent-signals", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Read-only endpoints
# ---------------------------------------------------------------------------


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise HTTPException(400, f"Cannot parse datetime: {s}")


@app.get("/api/turns")
def get_turns(since: str | None = None, limit: int = Query(default=100, le=500)):
    since_dt = _parse_dt(since) or (datetime.now(timezone.utc) - timedelta(hours=72))
    with _make_client() as client:
        turns = client.query_turns(limit=limit, since=since_dt)
        for t in turns:
            client.hydrate_turn_children(t)
    return {"turns": [_turn_to_dict(t) for t in turns]}


@app.get("/api/turns/{trace_id}")
def get_turn(trace_id: str):
    with _make_client() as client:
        turns = client.query_turns(limit=500, include_details=True)
        match = [t for t in turns if t.trace_id == trace_id]
        if not match:
            raise HTTPException(404, f"Turn {trace_id} not found")
        turn = match[0]
        client.hydrate_turn_children(turn)
        feedback = client.query_all_feedback(turn.ref_for(ENTITY, PROJECT))
    result = _turn_to_dict(turn, include_children=True)
    result["feedback"] = feedback
    return result


@app.get("/api/sessions")
def get_sessions(since: str | None = None, limit: int = Query(default=20, le=100)):
    since_dt = _parse_dt(since) or (datetime.now(timezone.utc) - timedelta(hours=168))
    with _make_client() as client:
        turns = client.query_turns(limit=500, since=since_dt)
    by_conv = _group_turns_by_session(turns)
    sessions = [_session_summary(cid, ts) for cid, ts in by_conv.items()]
    sessions.sort(key=lambda s: s["started_at"] or "", reverse=True)
    return {"sessions": sessions[:limit]}


@app.get("/api/sessions/{conversation_id}")
def get_session(conversation_id: str):
    with _make_client() as client:
        session = client.query_session(conversation_id)
        for t in session.turns:
            client.hydrate_turn_children(t)
        sess_feedback = client.query_all_feedback(session.ref_for(ENTITY, PROJECT))
        turn_feedback = {}
        for t in session.turns:
            fb = client.query_all_feedback(t.ref_for(ENTITY, PROJECT))
            if fb:
                turn_feedback[t.trace_id] = fb
    return {
        "conversation_id": session.conversation_id,
        "config_version": session.config_version,
        "git_branch": session.git_branch,
        "total_tokens": session.total_tokens,
        "turn_count": len(session.turns),
        "turns": [_turn_to_dict(t, include_children=True) for t in session.turns],
        "session_feedback": sess_feedback,
        "turn_feedback": turn_feedback,
    }


@app.get("/api/feedback")
def get_feedback(
    trace_id: str | None = None,
    ref: str | None = None,
    type_prefix: str = "weave_agent_signals.",
):
    if not trace_id and not ref:
        raise HTTPException(400, "Provide trace_id or ref")
    if not ref and trace_id:
        ref = f"weave:///{ENTITY}/{PROJECT}/agent_turn/{trace_id}"
    with _make_client() as client:
        feedback = client.query_all_feedback(ref)
    if type_prefix:
        feedback = [f for f in feedback if f.get("feedback_type", "").startswith(type_prefix)]
    return {"feedback": feedback}


@app.get("/api/analyze")
def get_analysis(limit: int = Query(default=1000, le=5000)):
    with _make_client() as client:
        feedback = client.query_project_feedback(limit=limit)
    if not feedback:
        return {"summary": [], "ab_leaderboard": [], "trends": [], "coaching_markdown": ""}

    summaries = aggregate_scores(feedback)
    summary_list = []
    for scorer, s in sorted(summaries.items()):
        summary_list.append(
            {
                "scorer": scorer,
                "count": s.count,
                "mean": round(s.mean, 4),
                "ci": [round(s.ci[0], 4), round(s.ci[1], 4)],
                "binary": s.binary,
                "pass_rate": round(s.pass_rate, 4) if s.pass_rate is not None else None,
                "tag_counts": dict(s.tag_counts),
                "confident": s.confident,
            }
        )

    ab = ab_leaderboard(feedback)
    ab_list = []
    for r in ab:
        scores_dict = {}
        for sc_name, sc in r.scores.items():
            scores_dict[sc_name] = {"mean": round(sc.mean, 4), "count": sc.count}
        ab_list.append(
            {
                "config_version": r.config_version,
                "turn_count": r.turn_count,
                "scores": scores_dict,
            }
        )

    trends = detect_regressions(feedback)
    trend_list = [
        {
            "scorer": r["scorer"],
            "direction": r["direction"],
            "older_mean": round(r["older_mean"], 4),
            "recent_mean": round(r["recent_mean"], 4),
            "delta": round(r["delta"], 4),
            "sample_count": r["sample_count"],
            "significant": r["significant"],
        }
        for r in trends
    ]

    coaching = coaching_digest(feedback)

    return {
        "summary": summary_list,
        "ab_leaderboard": ab_list,
        "trends": trend_list,
        "coaching_markdown": coaching,
    }


@app.get("/api/artifacts")
def get_artifacts():
    arts = extract_artifacts(PROJECT_ROOT)
    return {"artifacts": [{"name": a.name, "path": a.path, "content": a.content} for a in arts]}


@app.get("/api/rubrics")
def get_rubrics():
    all_rubrics = {**RUBRICS, **SESSION_RUBRICS}
    return {
        "rubrics": [
            {
                "name": name,
                "scorer_name": r.scorer_name,
                "description": r.description,
                "criteria": dict(r.criteria),
                "granularity": "session" if name in SESSION_RUBRICS else "turn",
            }
            for name, r in all_rubrics.items()
        ]
    }


@app.get("/api/models")
def get_models():
    data = {
        backend: {
            "default": roster["default"],
            "poll": roster["poll"],
            "escalation": roster["escalation"],
        }
        for backend, roster in _BACKEND_ROSTERS.items()
    }
    return JSONResponse(
        content=data,
        headers={"Cache-Control": "no-store"},
    )


_MONITOR_STATE_FILE = str(Path.home() / ".weave-agent-signals" / "monitor_seen.json")


@app.get("/api/monitor/state")
def get_monitor_state():
    seen = alerts.load_seen(_MONITOR_STATE_FILE)
    return {"seen_keys": sorted(seen), "count": len(seen)}


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": _job_store.list(50)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _job_store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job


def _run_score_job(job_id: str, req: ScoreRequest, *, end_dt: datetime | None = None):
    try:
        since = _parse_dt(req.since) or (datetime.now(timezone.utc) - timedelta(hours=24))
        with _make_client() as client:
            turns = client.query_turns(limit=req.limit, since=since)
            if end_dt:
                turns = [t for t in turns if t.started_at <= end_dt]
            _job_store.update(job_id, progress={"total": len(turns), "scored": 0, "written": 0})
            if not turns:
                _job_store.update(job_id, status="complete", result={"turns_scored": 0, "scores_written": 0, "errors": 0})
                return

            written = 0
            errors = 0
            for i, turn in enumerate(turns):
                try:
                    client.hydrate_turn_children(turn)
                    turn_ref = turn.ref_for(ENTITY, PROJECT)
                    for s in score_turn(turn):
                        _stamp_metadata(
                            s,
                            config_version=turn.config_version,
                            git_branch=turn.git_branch,
                            run_time=turn.started_at,
                        )
                        if not req.dry_run:
                            ft = f"weave_agent_signals.{s.scorer}"
                            existing = client.query_existing_feedback(turn_ref, ft)
                            if existing and not req.force:
                                continue
                            if existing and req.force:
                                client.delete_feedback_ids(existing)
                            client.write_score(s, turn_ref)
                            written += 1
                except Exception as e:
                    errors += 1
                    log.warning("Score error on %s: %s", turn.trace_id[:12], e)
                _job_store.update(job_id, progress={"total": len(turns), "scored": i + 1, "written": written})

            by_conv: dict[str, list] = {}
            for t in turns:
                by_conv.setdefault(t.conversation_id, []).append(t)
            for conv_id, sess_turns in by_conv.items():
                ordered = sorted(sess_turns, key=lambda t: t.started_at)
                session = SessionView(
                    conversation_id=conv_id,
                    turns=ordered,
                    config_version=ordered[0].config_version if ordered else None,
                    git_branch=ordered[0].git_branch if ordered else None,
                )
                try:
                    sess_ref = session.ref_for(ENTITY, PROJECT)
                    run_time = ordered[0].started_at if ordered else None
                    for s in score_session(session):
                        _stamp_metadata(
                            s,
                            config_version=session.config_version,
                            git_branch=session.git_branch,
                            run_time=run_time,
                        )
                        if not req.dry_run:
                            ft = f"weave_agent_signals.{s.scorer}"
                            existing = client.query_existing_feedback(sess_ref, ft)
                            if existing and not req.force:
                                continue
                            if existing and req.force:
                                client.delete_feedback_ids(existing)
                            client.write_score(s, sess_ref)
                            written += 1
                except Exception:
                    errors += 1

        _job_store.update(job_id, status="complete", result={
            "turns_scored": len(turns),
            "sessions_scored": len(by_conv),
            "scores_written": written,
            "errors": errors,
            "dry_run": req.dry_run,
        })
    except Exception as e:
        _job_store.update(job_id, status="failed", error=str(e))


def _run_judge_job(job_id: str, req: JudgeRequest):
    try:
        since = _parse_dt(req.since) or (datetime.now(timezone.utc) - timedelta(hours=24))
        with _make_client() as client:
            turns = client.query_turns(limit=req.limit, since=since)
            _job_store.update(job_id, progress={"total": len(turns), "judged": 0, "written": 0})
            if not turns:
                _job_store.update(job_id, status="complete", result={"turns_judged": 0, "scores_written": 0, "errors": 0})
                return

            for t in turns:
                try:
                    client.hydrate_turn_children(t)
                except Exception:
                    pass

            turn_rubrics = None
            session_rubrics = None
            if req.rubrics:
                turn_rubrics = [RUBRICS[n] for n in req.rubrics if n in RUBRICS]
                session_rubrics = [SESSION_RUBRICS[n] for n in req.rubrics if n in SESSION_RUBRICS]

            written = 0
            errors = 0

            with _make_judge_client(req.judge_backend) as inference:
                for i, turn in enumerate(turns):
                    try:
                        turn_ref = turn.ref_for(ENTITY, PROJECT)
                        scores = judge_turn(turn, inference, rubrics=turn_rubrics or None)
                        for s in scores:
                            _stamp_metadata(
                                s,
                                config_version=turn.config_version,
                                git_branch=turn.git_branch,
                                run_time=turn.started_at,
                            )
                            if not req.dry_run:
                                ft = f"weave_agent_signals.{s.scorer}"
                                existing = client.query_existing_feedback(turn_ref, ft)
                                if existing and not req.force:
                                    continue
                                if existing and req.force:
                                    client.delete_feedback_ids(existing)
                                client.write_score(s, turn_ref)
                                written += 1
                    except Exception as e:
                        errors += 1
                        log.warning("Judge error on %s: %s", turn.trace_id[:12], e)
                    _job_store.update(job_id, progress={"total": len(turns), "judged": i + 1, "written": written})

                by_conv = _group_turns_by_session(turns)
                for conv_id, sess_turns in by_conv.items():
                    ordered = sorted(sess_turns, key=lambda t: t.started_at)
                    session = SessionView(
                        conversation_id=conv_id,
                        turns=ordered,
                        config_version=ordered[0].config_version if ordered else None,
                        git_branch=ordered[0].git_branch if ordered else None,
                    )
                    try:
                        scores = judge_session(
                            session,
                            inference,
                            rubrics=session_rubrics or None,
                            panel_size=req.panel_size,
                        )
                        sess_ref = session.ref_for(ENTITY, PROJECT)
                        run_time = ordered[0].started_at if ordered else None
                        for s in scores:
                            _stamp_metadata(
                                s,
                                config_version=session.config_version,
                                git_branch=session.git_branch,
                                run_time=run_time,
                            )
                            if not req.dry_run:
                                ft = f"weave_agent_signals.{s.scorer}"
                                existing = client.query_existing_feedback(sess_ref, ft)
                                if existing and not req.force:
                                    continue
                                if existing and req.force:
                                    client.delete_feedback_ids(existing)
                                client.write_score(s, sess_ref)
                                written += 1
                    except Exception:
                        errors += 1

        _job_store.update(job_id, status="complete", result={
            "turns_judged": len(turns),
            "scores_written": written,
            "errors": errors,
            "dry_run": req.dry_run,
        })
    except Exception as e:
        _job_store.update(job_id, status="failed", error=str(e))


def _run_reflect_job(job_id: str, req: ReflectRequest):
    try:
        with _make_client() as client:
            feedback = client.query_project_feedback(limit=req.limit)

        if not feedback:
            _job_store.update(job_id, status="complete", result={"proposal": None, "reason": "No feedback found"})
            return

        coaching = coaching_digest(feedback)
        originals = extract_artifacts(PROJECT_ROOT)
        if not originals:
            _job_store.update(job_id, status="complete", result={"proposal": None, "reason": "No artifacts found"})
            return

        if req.dry_run:
            _job_store.update(job_id, status="complete", result={
                "proposal": None,
                "reason": "Dry run — skipping reflection",
                "coaching_markdown": coaching,
                "artifact_count": len(originals),
                "feedback_count": len(feedback),
                "dry_run": True,
            })
            return

        _job_store.update(job_id, progress={"phase": "reflecting", "iterations": req.iterations})

        cli_lm = make_cli_lm() if req.judge_backend == "cli" else None
        with _make_judge_client(req.judge_backend) as judge:
            proposal = run_reflection(
                project_root=PROJECT_ROOT,
                feedback=feedback,
                coaching_text=coaching,
                judge_client=judge,
                judge_model=judge_default_model(judge),
                model=req.model,
                max_iterations=req.iterations,
                reflection_lm=cli_lm,
            )

        if proposal is None:
            _job_store.update(job_id, status="complete", result={"proposal": None, "reason": "No improvements proposed"})
            return

        diff = render_proposal_diff(originals, proposal)
        _job_store.update(job_id, status="complete", result={
            "diff": diff,
            "rationale": proposal.rationale,
            "score_delta": proposal.score_delta,
            "artifacts": [
                {"name": a.name, "path": a.path, "content": a.content} for a in proposal.artifacts
            ],
            "coaching_markdown": coaching,
        })
    except Exception as e:
        _job_store.update(job_id, status="failed", error=str(e))


def _run_monitor_job(job_id: str, req: MonitorRequest):
    try:
        with _make_client() as client:
            feedback = client.query_project_feedback(limit=req.limit)

        if not feedback:
            _job_store.update(job_id, status="complete", result={"alerts": [], "reason": "No feedback found"})
            return

        trend = detect_regressions(feedback)
        config = detect_config_regressions(feedback)
        sig_trend = [r for r in trend if r["significant"] and r["direction"] == "regression"]
        sig_config = [r for r in config if r["significant"]]

        found = [alerts.trend_alert(r) for r in sig_trend] + [
            alerts.config_alert(r) for r in sig_config
        ]

        seen = alerts.load_seen(_MONITOR_STATE_FILE)
        new_alerts = [a for a in found if a.key not in seen]
        alert_list = [{"key": a.key, "text": a.text} for a in new_alerts]

        if not req.dry_run and new_alerts:
            delivered = alerts.send(new_alerts, webhook=req.alert_webhook)
            alert_list = [{"key": a.key, "text": a.text} for a in delivered]
            seen.update(a.key for a in delivered)
            alerts.save_seen(_MONITOR_STATE_FILE, seen)

        _job_store.update(job_id, status="complete", result={
            "alerts": alert_list,
            "total_found": len(found),
            "new_alerts": len(new_alerts),
            "dry_run": req.dry_run,
        })
    except Exception as e:
        _job_store.update(job_id, status="failed", error=str(e))


@app.post("/api/jobs/score")
async def submit_score(req: ScoreRequest):
    job = _job_store.create("score")
    asyncio.get_running_loop().run_in_executor(None, _run_score_job, job["job_id"], req)
    return {"job_id": job["job_id"], "status": "running"}


@app.post("/api/jobs/backfill")
async def submit_backfill(req: BackfillRequest):
    job = _job_store.create("backfill")
    end_dt = _parse_dt(req.end) if req.end else None

    def run():
        _run_score_job(
            job["job_id"],
            ScoreRequest(
                since=req.start,
                limit=10000,
                dry_run=req.dry_run,
                force=req.force,
            ),
            end_dt=end_dt,
        )

    asyncio.get_running_loop().run_in_executor(None, run)
    return {"job_id": job["job_id"], "status": "running"}


@app.post("/api/jobs/judge")
async def submit_judge(req: JudgeRequest):
    job = _job_store.create("judge")
    asyncio.get_running_loop().run_in_executor(None, _run_judge_job, job["job_id"], req)
    return {"job_id": job["job_id"], "status": "running"}


@app.post("/api/jobs/reflect")
async def submit_reflect(req: ReflectRequest):
    job = _job_store.create("reflect")
    asyncio.get_running_loop().run_in_executor(None, _run_reflect_job, job["job_id"], req)
    return {"job_id": job["job_id"], "status": "running"}


@app.post("/api/jobs/reflect/apply")
async def apply_reflect(req: ApplyRequest):
    job = _job_store.get(req.job_id)
    if not job:
        raise HTTPException(404, f"Job {req.job_id} not found")
    if job["status"] != "complete" or not job.get("result", {}).get("artifacts"):
        raise HTTPException(400, "Job has no artifacts to apply")

    for artifact in job["result"]["artifacts"]:
        path = os.path.realpath(os.path.join(PROJECT_ROOT, artifact["path"]))
        if not path.startswith(os.path.realpath(PROJECT_ROOT) + os.sep):
            raise HTTPException(400, f"Artifact path escapes project root: {artifact['path']}")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(artifact["content"])

    return {"applied": [a["path"] for a in job["result"]["artifacts"]]}


@app.post("/api/jobs/monitor")
async def submit_monitor(req: MonitorRequest):
    job = _job_store.create("monitor")
    asyncio.get_running_loop().run_in_executor(None, _run_monitor_job, job["job_id"], req)
    return {"job_id": job["job_id"], "status": "running"}


# ---------------------------------------------------------------------------
# Run endpoints and pipeline execution (spec 09)
#
# A run moves through a fixed lifecycle: created -> scoring -> judging ->
# reflecting -> complete (see runs.py). Each step's work runs the same
# scorer/judge/reflector logic as the _run_score_job/_run_judge_job/
# _run_reflect_job job handlers above, but reads its data from the run's
# `data_selection` (date range + included/excluded session ids) instead of
# `since`/`limit` request params, and reports progress/results onto the run
# instead of a job.
# ---------------------------------------------------------------------------

# Linear pipeline order. Shared by the advance endpoint (compute the next
# step) and _execute_run_step's auto-run cascade (compute what comes after a
# step that just finished). COMPLETE/FAILED have no entry: both are terminal.
_NEXT_STATUS: dict[RunStatus, RunStatus] = {
    RunStatus.CREATED: RunStatus.SCORING,
    RunStatus.SCORING: RunStatus.JUDGING,
    RunStatus.JUDGING: RunStatus.REFLECTING,
    RunStatus.REFLECTING: RunStatus.COMPLETE,
}

# Statuses whose entry triggers background work, as opposed to COMPLETE, which
# is just a terminal marker set after reflecting has already produced a result.
_EXECUTABLE_STATUSES = (RunStatus.SCORING, RunStatus.JUDGING, RunStatus.REFLECTING)


def _serialize_run(run: Run) -> dict:
    d = asdict(run)  # recursively converts the nested DataSelection too
    d["status"] = run.status.value
    return d


@app.post("/api/runs")
def create_run():
    run = _run_store.create()
    return _serialize_run(run)


@app.get("/api/runs")
def list_runs():
    return {"runs": [_serialize_run(r) for r in _run_store.list()]}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = _run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _serialize_run(run)


@app.put("/api/runs/{run_id}/selection")
def set_run_selection(run_id: str, req: SelectionRequest):
    run = _run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    _run_store.set_selection(
        run_id,
        since=req.since,
        until=req.until,
        session_ids=req.session_ids,
        excluded_session_ids=req.excluded_session_ids,
    )
    return _serialize_run(_run_store.get(run_id))


@app.post("/api/runs/{run_id}/advance")
async def advance_run(run_id: str, req: AdvanceRequest | None = None):
    run = _run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if req is None:
        req = AdvanceRequest()

    next_status = _NEXT_STATUS.get(run.status)
    if next_status is None:
        raise HTTPException(status_code=400, detail=f"Cannot advance from {run.status.value}")

    if run.status == RunStatus.CREATED and run.data_selection is None:
        raise HTTPException(status_code=400, detail="Data selection required before advancing")

    try:
        _run_store.update(run_id, status=next_status)
    except ValueError as exc:
        # RunStore's transition check is atomic (see runs.py), so this only
        # fires when a concurrent advance on the same run already moved its
        # status between our `get()` above and this `update()` — a real race,
        # not a bug. Surface it as a conflict instead of an unhandled 500.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if next_status in _EXECUTABLE_STATUSES:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _execute_run_step, run_id, next_status, req)

    return _serialize_run(_run_store.get(run_id))


def _execute_run_step(run_id: str, step_status: RunStatus, req: AdvanceRequest) -> None:
    try:
        if step_status == RunStatus.SCORING:
            _run_scoring_step(run_id, req)
        elif step_status == RunStatus.JUDGING:
            _run_judging_step(run_id, req)
        elif step_status == RunStatus.REFLECTING:
            _run_reflecting_step(run_id, req)

        # Auto-advance: recurse into the next step until one fails or the
        # pipeline reaches a non-executable status (COMPLETE).
        run = _run_store.get(run_id)
        if run and run.auto_run and run.status != RunStatus.FAILED:
            next_next = _NEXT_STATUS.get(run.status)
            if next_next:
                _run_store.update(run_id, status=next_next)
                if next_next in _EXECUTABLE_STATUSES:
                    _execute_run_step(run_id, next_next, req)
    except Exception as exc:
        log.warning("Run %s step %s failed: %s", run_id, step_status.value, exc)
        _run_store.update(run_id, status=RunStatus.FAILED, error=str(exc))


def _select_turns_for_run(client: WeaveClient, selection: DataSelection) -> list[TurnSpan]:
    """Fetch + filter turns per a run's data selection.

    A run's scope is defined entirely by the selection (date range, explicit
    session id include/exclude) rather than a page `limit`, so this paginates
    to completion instead of capping at a single page. `until` and session id
    filtering both happen client-side: the spans query has no `until` bound
    and no multi-id filter (see client.py).
    """
    since_dt = _parse_dt(selection.since)
    until_dt = _parse_dt(selection.until)
    turns = client.query_turns_paginated(page_size=500, since=since_dt)
    if until_dt:
        turns = [t for t in turns if t.started_at <= until_dt]
    if selection.session_ids:
        wanted = set(selection.session_ids)
        turns = [t for t in turns if t.conversation_id in wanted]
    if selection.excluded_session_ids:
        excluded = set(selection.excluded_session_ids)
        turns = [t for t in turns if t.conversation_id not in excluded]
    return turns


def _sessions_for_run(turns: list[TurnSpan]) -> dict[str, SessionView]:
    """Group a run's selected turns into per-conversation SessionViews."""
    sessions: dict[str, SessionView] = {}
    for conv_id, sess_turns in _group_turns_by_session(turns).items():
        ordered = sorted(sess_turns, key=lambda t: t.started_at)
        sessions[conv_id] = SessionView(
            conversation_id=conv_id,
            turns=ordered,
            config_version=ordered[0].config_version if ordered else None,
            git_branch=ordered[0].git_branch if ordered else None,
        )
    return sessions


def _run_data_selection(run_id: str) -> DataSelection:
    run = _run_store.get(run_id)
    if run is None or run.data_selection is None:
        raise ValueError(f"Run {run_id} has no data selection")
    return run.data_selection


def _run_scoring_step(run_id: str, req: AdvanceRequest) -> None:
    selection = _run_data_selection(run_id)
    with _make_client() as client:
        turns = _select_turns_for_run(client, selection)
        sessions = _sessions_for_run(turns)
        _run_store.update(run_id, scoring_progress={"total": len(turns), "scored": 0, "written": 0})

        if not turns:
            _run_store.update(
                run_id,
                scoring_result={
                    "turns_scored": 0,
                    "sessions_scored": 0,
                    "scores_written": 0,
                    "errors": 0,
                    "dry_run": req.dry_run,
                },
            )
            return

        written = 0
        errors = 0
        for i, turn in enumerate(turns):
            try:
                client.hydrate_turn_children(turn)
                turn_ref = turn.ref_for(ENTITY, PROJECT)
                for s in score_turn(turn):
                    _stamp_metadata(
                        s,
                        config_version=turn.config_version,
                        git_branch=turn.git_branch,
                        run_time=turn.started_at,
                    )
                    if not req.dry_run:
                        ft = f"weave_agent_signals.{s.scorer}"
                        existing = client.query_existing_feedback(turn_ref, ft)
                        if existing and not req.force:
                            continue
                        if existing and req.force:
                            client.delete_feedback_ids(existing)
                        client.write_score(s, turn_ref)
                        written += 1
            except Exception as e:
                errors += 1
                log.warning("Run %s score error on %s: %s", run_id, turn.trace_id[:12], e)
            _run_store.update(
                run_id,
                scoring_progress={"total": len(turns), "scored": i + 1, "written": written},
            )

        for session in sessions.values():
            try:
                sess_ref = session.ref_for(ENTITY, PROJECT)
                run_time = session.turns[0].started_at if session.turns else None
                for s in score_session(session):
                    _stamp_metadata(
                        s,
                        config_version=session.config_version,
                        git_branch=session.git_branch,
                        run_time=run_time,
                    )
                    if not req.dry_run:
                        ft = f"weave_agent_signals.{s.scorer}"
                        existing = client.query_existing_feedback(sess_ref, ft)
                        if existing and not req.force:
                            continue
                        if existing and req.force:
                            client.delete_feedback_ids(existing)
                        client.write_score(s, sess_ref)
                        written += 1
            except Exception:
                errors += 1

    _run_store.update(
        run_id,
        scoring_result={
            "turns_scored": len(turns),
            "sessions_scored": len(sessions),
            "scores_written": written,
            "errors": errors,
            "dry_run": req.dry_run,
        },
    )


def _run_judging_step(run_id: str, req: AdvanceRequest) -> None:
    selection = _run_data_selection(run_id)
    with _make_client() as client:
        turns = _select_turns_for_run(client, selection)
        sessions = _sessions_for_run(turns)
        _run_store.update(run_id, judging_progress={"total": len(turns), "judged": 0, "written": 0})

        if not turns:
            _run_store.update(
                run_id,
                judging_result={
                    "turns_judged": 0,
                    "scores_written": 0,
                    "errors": 0,
                    "dry_run": req.dry_run,
                },
            )
            return

        for t in turns:
            try:
                client.hydrate_turn_children(t)
            except Exception:
                pass

        turn_rubrics = None
        session_rubrics = None
        if req.rubrics:
            turn_rubrics = [RUBRICS[n] for n in req.rubrics if n in RUBRICS]
            session_rubrics = [SESSION_RUBRICS[n] for n in req.rubrics if n in SESSION_RUBRICS]

        written = 0
        errors = 0

        with _make_judge_client(req.judge_backend) as inference:
            for i, turn in enumerate(turns):
                try:
                    turn_ref = turn.ref_for(ENTITY, PROJECT)
                    scores = judge_turn(turn, inference, rubrics=turn_rubrics or None)
                    for s in scores:
                        _stamp_metadata(
                            s,
                            config_version=turn.config_version,
                            git_branch=turn.git_branch,
                            run_time=turn.started_at,
                        )
                        if not req.dry_run:
                            ft = f"weave_agent_signals.{s.scorer}"
                            existing = client.query_existing_feedback(turn_ref, ft)
                            if existing and not req.force:
                                continue
                            if existing and req.force:
                                client.delete_feedback_ids(existing)
                            client.write_score(s, turn_ref)
                            written += 1
                except Exception as e:
                    errors += 1
                    log.warning("Run %s judge error on %s: %s", run_id, turn.trace_id[:12], e)
                _run_store.update(
                    run_id,
                    judging_progress={"total": len(turns), "judged": i + 1, "written": written},
                )

            for session in sessions.values():
                try:
                    scores = judge_session(
                        session,
                        inference,
                        rubrics=session_rubrics or None,
                        panel_size=req.panel_size,
                    )
                    sess_ref = session.ref_for(ENTITY, PROJECT)
                    run_time = session.turns[0].started_at if session.turns else None
                    for s in scores:
                        _stamp_metadata(
                            s,
                            config_version=session.config_version,
                            git_branch=session.git_branch,
                            run_time=run_time,
                        )
                        if not req.dry_run:
                            ft = f"weave_agent_signals.{s.scorer}"
                            existing = client.query_existing_feedback(sess_ref, ft)
                            if existing and not req.force:
                                continue
                            if existing and req.force:
                                client.delete_feedback_ids(existing)
                            client.write_score(s, sess_ref)
                            written += 1
                except Exception:
                    errors += 1

    _run_store.update(
        run_id,
        judging_result={
            "turns_judged": len(turns),
            "scores_written": written,
            "errors": errors,
            "dry_run": req.dry_run,
        },
    )


# AdvanceRequest has no `limit` field (unlike ReflectRequest) — a run's scope
# is already bounded by its data selection, not a feedback page size. Fetch
# generously and filter down to this run's turn/session refs client-side,
# since the feedback API has no way to scope a query to a set of refs.
_REFLECT_FEEDBACK_LIMIT = 5000


def _run_reflecting_step(run_id: str, req: AdvanceRequest) -> None:
    selection = _run_data_selection(run_id)
    with _make_client() as client:
        turns = _select_turns_for_run(client, selection)
        sessions = _sessions_for_run(turns)
        valid_refs = {t.ref_for(ENTITY, PROJECT) for t in turns}
        valid_refs |= {s.ref_for(ENTITY, PROJECT) for s in sessions.values()}
        all_feedback = client.query_project_feedback(limit=_REFLECT_FEEDBACK_LIMIT)

    feedback = [fb for fb in all_feedback if fb.get("weave_ref") in valid_refs]

    if not feedback:
        _run_store.update(
            run_id, reflecting_result={"proposal": None, "reason": "No feedback found"}
        )
        return

    coaching = coaching_digest(feedback)
    originals = extract_artifacts(PROJECT_ROOT)
    if not originals:
        _run_store.update(
            run_id, reflecting_result={"proposal": None, "reason": "No artifacts found"}
        )
        return

    if req.dry_run:
        _run_store.update(
            run_id,
            reflecting_result={
                "proposal": None,
                "reason": "Dry run — skipping reflection",
                "coaching_markdown": coaching,
                "artifact_count": len(originals),
                "feedback_count": len(feedback),
                "dry_run": True,
            },
        )
        return

    _run_store.update(
        run_id, reflecting_progress={"phase": "reflecting", "iterations": req.iterations}
    )

    cli_lm = make_cli_lm() if req.judge_backend == "cli" else None
    with _make_judge_client(req.judge_backend) as judge:
        proposal = run_reflection(
            project_root=PROJECT_ROOT,
            feedback=feedback,
            coaching_text=coaching,
            judge_client=judge,
            judge_model=judge_default_model(judge),
            model=req.model,
            max_iterations=req.iterations,
            reflection_lm=cli_lm,
        )

    if proposal is None:
        _run_store.update(
            run_id, reflecting_result={"proposal": None, "reason": "No improvements proposed"}
        )
        return

    diff = render_proposal_diff(originals, proposal)
    _run_store.update(
        run_id,
        reflecting_result={
            "diff": diff,
            "rationale": proposal.rationale,
            "score_delta": proposal.score_delta,
            "artifacts": [
                {"name": a.name, "path": a.path, "content": a.content} for a in proposal.artifacts
            ],
            "coaching_markdown": coaching,
        },
    )


# ---------------------------------------------------------------------------
# Static file serving (production: serves frontend/dist/)
# ---------------------------------------------------------------------------

_frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = (_frontend_dist / full_path).resolve()
        if not str(file_path).startswith(str(_frontend_dist.resolve())):
            return FileResponse(_frontend_dist / "index.html")
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_frontend_dist / "index.html")
