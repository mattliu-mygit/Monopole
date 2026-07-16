"""Shared serialization for FastAPI route modules."""

from dataclasses import asdict

from weave_agent_signals.runs.reflection import NO_VALID_PROPOSAL_REASON
from weave_agent_signals.runs.store import Run, RunSummarySource


def serialize_run(run: Run) -> dict:
    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "current_stage_succeeded": run.current_stage_succeeded,
        "created_at": run.created_at,
        "auto_run": run.auto_run,
        "data_selection": asdict(run.data_selection) if run.data_selection is not None else None,
        "run_config": (
            run.run_config.model_dump(mode="json") if run.run_config is not None else None
        ),
        "effective_config": (
            run.effective_config.model_dump(mode="json")
            if run.effective_config is not None
            else None
        ),
        "turn_cohort": run.turn_cohort,
        "judging_plan": run.judging_plan,
        "judging_artifacts": run.judging_artifacts,
        "reflection_input": run.reflection_input,
        "scoring_progress": run.scoring_progress,
        "scoring_result": run.scoring_result,
        "judging_progress": run.judging_progress,
        "judging_result": run.judging_result,
        "reflecting_progress": run.reflecting_progress,
        "reflecting_result": run.reflecting_result,
        "reflection_review": run.reflection_review,
        "reflection_review_revision": run.reflection_review_revision,
        "error": run.error,
    }


def _review_state(summary: RunSummarySource) -> str:
    if summary.review_status == "pending":
        return "review-needed"
    if summary.review_status in {"promoted", "partial", "dismissed"}:
        return summary.review_status
    if summary.status.value == "complete":
        if summary.reflection_reason == NO_VALID_PROPOSAL_REASON:
            return "no-valid-proposal"
        if summary.baseline_won is True:
            return "no-change"
    return "none"


def serialize_run_summary(summary: RunSummarySource) -> dict:
    """Serialize only the stable fields needed by the polling run list."""

    return {
        "run_id": summary.run_id,
        "status": summary.status.value,
        "current_stage_succeeded": summary.current_stage_succeeded,
        "created_at": summary.created_at,
        "selection": (
            {
                "session_count": summary.session_count,
                "since": summary.selection_since,
                "until": summary.selection_until,
                "timezone": summary.selection_timezone,
            }
            if summary.session_count is not None
            else None
        ),
        "review_state": _review_state(summary),
    }
