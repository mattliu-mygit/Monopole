"""Typed SQLite persistence for the evaluation-run lifecycle."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from weave_agent_signals.run_config import EffectiveRunConfig, RunConfig

_DEFAULT_DB_DIR = Path.home() / ".weave-agent-signals"
RUN_DB_SCHEMA_VERSION = 9


def _default_db_path() -> Path:
    _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_DB_DIR / "runs.db"


class RunStatus(str, Enum):
    CREATED = "created"
    SCORING = "scoring"
    JUDGING = "judging"
    REFLECTING = "reflecting"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset({RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.SCORING: frozenset({RunStatus.JUDGING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.JUDGING: frozenset({RunStatus.REFLECTING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.REFLECTING: frozenset({RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.COMPLETE: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}
_ACTIVE_STAGES = frozenset({RunStatus.SCORING, RunStatus.JUDGING, RunStatus.REFLECTING})
_STAGE_FIELDS = {
    RunStatus.SCORING: ("scoring_progress", "scoring_result"),
    RunStatus.JUDGING: ("judging_progress", "judging_result"),
    RunStatus.REFLECTING: ("reflecting_progress", "reflecting_result"),
}
_STAGE_SUCCESS_FIELDS = {
    RunStatus.SCORING: "scoring_succeeded",
    RunStatus.JUDGING: "judging_succeeded",
    RunStatus.REFLECTING: "reflecting_succeeded",
}
_STAGE_SUCCESSORS = {
    RunStatus.SCORING: RunStatus.JUDGING,
    RunStatus.JUDGING: RunStatus.REFLECTING,
    RunStatus.REFLECTING: RunStatus.COMPLETE,
}
_REFLECTION_REVIEW_STATUSES = frozenset({"pending", "promoted", "partial", "dismissed"})
_RESOLVED_REFLECTION_REVIEW_STATUSES = frozenset({"promoted", "partial", "dismissed"})


class RunStoreConflictError(ValueError):
    """Base class for optimistic-concurrency and lifecycle conflicts."""


class RunLifecycleConflictError(RunStoreConflictError):
    def __init__(self, run_id: str, current_status: RunStatus):
        self.run_id = run_id
        self.current_status = current_status
        super().__init__(
            f"Run {run_id} can only be configured while created; "
            f"current status is {current_status.value}"
        )


class RunCancellationConflictError(RunStoreConflictError):
    def __init__(
        self,
        run_id: str,
        current_status: RunStatus,
        *,
        reflection_finalizing: bool,
    ):
        self.run_id = run_id
        self.current_status = current_status
        self.reflection_finalizing = reflection_finalizing
        reason = (
            "reflection is finalizing"
            if reflection_finalizing
            else f"current status is {current_status.value}"
        )
        super().__init__(f"Run {run_id} cannot be cancelled because {reason}")


class ReflectionReviewRevisionConflictError(RunStoreConflictError):
    def __init__(
        self,
        run_id: str,
        expected_revision: int,
        current_revision: int,
        current_review: dict[str, Any] | None,
    ):
        self.run_id = run_id
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        self.current_review = current_review
        super().__init__(
            f"Reflection review revision conflict for {run_id}: "
            f"expected {expected_revision}, current {current_revision}"
        )


class ReflectionReviewLifecycleConflictError(RunStoreConflictError):
    def __init__(
        self,
        run_id: str,
        message: str,
        *,
        current_run_status: RunStatus,
        current_review_status: str | None,
        current_revision: int,
    ):
        self.run_id = run_id
        self.current_run_status = current_run_status
        self.current_review_status = current_review_status
        self.current_revision = current_revision
        super().__init__(f"Reflection review conflict for {run_id}: {message}")


@dataclass(frozen=True)
class DataSelection:
    since: str | None = None
    until: str | None = None
    timezone: str | None = None
    session_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_ids", tuple(self.session_ids))


def _current_stage_succeeded(
    status: RunStatus,
    *,
    scoring: bool,
    judging: bool,
    reflecting: bool,
) -> bool:
    if status is RunStatus.SCORING:
        return scoring
    if status is RunStatus.JUDGING:
        return judging
    if status in {RunStatus.REFLECTING, RunStatus.COMPLETE}:
        return reflecting
    return False


@dataclass(frozen=True)
class RunSummarySource:
    """Scalar run-list fields projected without hydrating full evidence."""

    run_id: str
    status: RunStatus
    created_at: str
    scoring_succeeded: bool
    judging_succeeded: bool
    reflecting_succeeded: bool
    session_count: int | None
    selection_since: str | None
    selection_until: str | None
    selection_timezone: str | None
    review_status: str | None
    baseline_won: bool | None
    reflection_reason: str | None

    @property
    def current_stage_succeeded(self) -> bool:
        return _current_stage_succeeded(
            self.status,
            scoring=self.scoring_succeeded,
            judging=self.judging_succeeded,
            reflecting=self.reflecting_succeeded,
        )


@dataclass(frozen=True)
class Run:
    run_id: str
    status: RunStatus
    created_at: str
    auto_run: bool = False
    data_selection: DataSelection | None = None
    run_config: RunConfig | None = None
    effective_config: EffectiveRunConfig | None = None
    turn_cohort: dict[str, Any] | None = None
    judging_plan: dict[str, Any] | None = None
    judging_artifacts: dict[str, Any] | None = None
    reflection_input: dict[str, Any] | None = None
    scoring_progress: dict[str, Any] | None = None
    scoring_result: dict[str, Any] | None = None
    scoring_succeeded: bool = False
    judging_progress: dict[str, Any] | None = None
    judging_result: dict[str, Any] | None = None
    judging_succeeded: bool = False
    reflecting_progress: dict[str, Any] | None = None
    reflecting_result: dict[str, Any] | None = None
    reflecting_succeeded: bool = False
    reflection_review: dict[str, Any] | None = None
    reflection_review_revision: int = 0
    error: str | None = None

    @property
    def current_stage_succeeded(self) -> bool:
        """Whether the active stage worker returned and finalized successfully."""

        return _current_stage_succeeded(
            self.status,
            scoring=self.scoring_succeeded,
            judging=self.judging_succeeded,
            reflecting=self.reflecting_succeeded,
        )


def _encode_data_selection(selection: DataSelection) -> str:
    if not isinstance(selection, DataSelection):
        raise ValueError("data selection must be a DataSelection")
    if not selection.session_ids:
        raise ValueError("selection must include at least one session")
    if any(not session_id.strip() for session_id in selection.session_ids):
        raise ValueError("selection session IDs must be nonblank")
    if len(selection.session_ids) != len(set(selection.session_ids)):
        raise ValueError("selection session IDs must be unique")
    return json.dumps(asdict(selection), sort_keys=True)


def _encode_run_config(config: RunConfig) -> str:
    if not isinstance(config, RunConfig):
        raise ValueError("run configuration must be a RunConfig")
    return config.model_dump_json()


def _encode_effective_config(config: EffectiveRunConfig) -> str:
    if not isinstance(config, EffectiveRunConfig):
        raise ValueError("effective configuration must be an EffectiveRunConfig")
    return config.model_dump_json()


def _encode_json_object(value: Mapping[str, Any], label: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    try:
        return json.dumps(dict(value), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc


def _encode_turn_cohort(turn_cohort: Mapping[str, Any]) -> str:
    if not isinstance(turn_cohort, Mapping):
        raise ValueError("turn cohort must be a JSON object")
    cohort = dict(turn_cohort)
    turns = cohort.get("turns")
    sessions = cohort.get("sessions")
    if not isinstance(turns, list) or not turns:
        raise ValueError("turn cohort must contain at least one turn")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("turn cohort must contain at least one session")
    if cohort.get("turn_count") != len(turns):
        raise ValueError("turn cohort turn_count does not match turns")
    if cohort.get("session_count") != len(sessions):
        raise ValueError("turn cohort session_count does not match sessions")
    if not isinstance(cohort.get("cohort_id"), str) or not cohort["cohort_id"]:
        raise ValueError("turn cohort must have a cohort_id")

    trace_ids: set[str] = set()
    refs: set[str] = set()
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("turn cohort entries must be JSON objects")
        trace_id = turn.get("trace_id")
        weave_ref = turn.get("weave_ref")
        conversation_id = turn.get("conversation_id")
        if not all(
            isinstance(value, str) and value for value in (trace_id, weave_ref, conversation_id)
        ):
            raise ValueError("turn cohort entries require trace_id, weave_ref, and conversation_id")
        if trace_id in trace_ids or weave_ref in refs:
            raise ValueError("turn cohort entries must be unique")
        evaluated_model = turn.get("model")
        evaluated_family = turn.get("model_family")
        if evaluated_model is not None and (
            not isinstance(evaluated_model, str) or not evaluated_model
        ):
            raise ValueError("turn cohort model must be null or a nonblank string")
        if not isinstance(evaluated_family, str) or not evaluated_family.strip():
            raise ValueError("turn cohort model family must be a nonblank string")
        trace_ids.add(trace_id)
        refs.add(weave_ref)
    return _encode_json_object(cohort, "turn cohort")


def _encode_reflection_input(reflection_input: Mapping[str, Any]) -> str:
    if not isinstance(reflection_input, Mapping):
        raise ValueError("reflection input must be a JSON object")
    value = dict(reflection_input)
    feedback = value.get("feedback")
    if not isinstance(feedback, list):
        raise ValueError("reflection input feedback must be a list")
    if value.get("feedback_count") != len(feedback):
        raise ValueError("reflection input feedback_count does not match feedback")
    return _encode_json_object(value, "reflection input")


def _encode_judging_plan(judging_plan: Mapping[str, Any]) -> str:
    if not isinstance(judging_plan, Mapping):
        raise ValueError("judging plan must be a JSON object")
    value = dict(judging_plan)
    required = {
        "plan_id",
        "schema_version",
        "cohort_id",
        "panel_size",
        "requested_rubrics",
        "input_policy",
        "protocol",
        "sessions",
        "totals",
    }
    if set(value) != required:
        raise ValueError("judging plan fields do not match schema version 3")
    if value["schema_version"] != "3":
        raise ValueError("judging plan schema_version must be '3'")
    if not isinstance(value["plan_id"], str) or not value["plan_id"]:
        raise ValueError("judging plan must have a plan_id")
    if not isinstance(value["cohort_id"], str) or not value["cohort_id"]:
        raise ValueError("judging plan must have a cohort_id")
    sessions = value["sessions"]
    totals = value["totals"]
    if not isinstance(sessions, list):
        raise ValueError("judging plan sessions must be a list")
    if not isinstance(totals, dict):
        raise ValueError("judging plan totals must be an object")
    _validate_judging_plan_structure(value)

    body = {key: item for key, item in value.items() if key != "plan_id"}
    try:
        canonical = json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("judging plan must be JSON serializable") from exc
    expected_plan_id = f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
    if value["plan_id"] != expected_plan_id:
        raise ValueError("judging plan ID does not match its content")
    return _encode_json_object(value, "judging plan")


def _exact_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"judging plan {label} fields are invalid")
    return value


def _nonnegative_int(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"judging plan {label} must be a {qualifier} integer")
    return value


def _canonical_digest(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _validate_judging_plan_structure(value: dict[str, Any]) -> None:
    rubric_keys = {"id", "label", "evaluation_unit", "version", "content_digest", "pass_threshold"}
    requested = value.get("requested_rubrics")
    if not isinstance(requested, list) or not requested:
        raise ValueError("judging plan requested_rubrics must be a non-empty list")
    for rubric in requested:
        row = _exact_keys(rubric, rubric_keys, "requested rubric")
        if row["evaluation_unit"] != "session" or any(
            not isinstance(row[key], str) or not row[key].strip()
            for key in ("id", "label", "version", "content_digest")
        ):
            raise ValueError("judging plan requested rubric values are invalid")
        threshold = row["pass_threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1
        ):
            raise ValueError("judging plan rubric threshold is invalid")
    if len({row["id"] for row in requested}) != len(requested):
        raise ValueError("judging plan requested rubric IDs must be unique")

    panel_size = _nonnegative_int(value.get("panel_size"), "panel_size", positive=True)
    if panel_size > 3:
        raise ValueError("judging plan panel_size must be between one and three")
    input_policy = _exact_keys(
        value.get("input_policy"),
        {
            "contract_version",
            "large_model_threshold_tokens",
            "large_model_reserve_tokens",
            "small_model_reserve_tokens",
            "large_model_raw_target_tokens",
            "small_model_raw_target_tokens",
            "prompt_reserve_tokens",
            "output_reserve_tokens",
            "safety_reserve_tokens",
            "digest_max_tokens",
            "finding_max_tokens",
            "overlap_turns",
            "max_chunks",
        },
        "input policy",
    )
    if input_policy["contract_version"] != "3" or input_policy["overlap_turns"] != 1:
        raise ValueError("judging plan input policy contract is invalid")
    for key in (
        "large_model_threshold_tokens",
        "large_model_reserve_tokens",
        "small_model_reserve_tokens",
        "large_model_raw_target_tokens",
        "small_model_raw_target_tokens",
        "prompt_reserve_tokens",
        "output_reserve_tokens",
        "safety_reserve_tokens",
        "digest_max_tokens",
        "finding_max_tokens",
        "max_chunks",
    ):
        _nonnegative_int(input_policy[key], f"input policy {key}", positive=True)
    if (
        input_policy["large_model_threshold_tokens"] != 200_000
        or input_policy["large_model_reserve_tokens"] < 100_000
        or input_policy["small_model_reserve_tokens"] < 50_000
        or input_policy["large_model_raw_target_tokens"] != 128_000
        or input_policy["small_model_raw_target_tokens"] != 50_000
    ):
        raise ValueError("judging plan input policy capacity tiers are invalid")
    protocol = _exact_keys(
        value.get("protocol"), {"protocol_version", "prompt_templates", "schemas"}, "protocol"
    )
    if protocol["protocol_version"] != "3":
        raise ValueError("judging plan protocol version is invalid")
    prompts = _exact_keys(
        protocol["prompt_templates"],
        {
            "digest_system",
            "digest_user",
            "window_system",
            "window_user",
            "merge_system",
            "merge_user",
        },
        "protocol prompts",
    )
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts.values()):
        raise ValueError("judging plan protocol prompt values are invalid")
    schemas = _exact_keys(protocol["schemas"], {"digest", "window", "merge"}, "protocol schemas")
    for phase, schema in schemas.items():
        schema_row = _exact_keys(schema, {"name", "schema"}, f"{phase} schema")
        if not isinstance(schema_row["name"], str) or not isinstance(schema_row["schema"], dict):
            raise ValueError(f"judging plan {phase} schema values are invalid")
    if {phase: schema["name"] for phase, schema in schemas.items()} != {
        "digest": "chunk_digest",
        "window": "window_findings",
        "merge": "merged_verdict",
    }:
        raise ValueError("judging plan protocol schema names are invalid")

    sessions = value["sessions"]
    totals_expected = {
        "sessions_planned": len(sessions),
        "turns_considered": 0,
        "windows_planned": 0,
        "planned_rubrics": len(sessions) * len(requested),
        "minimum_reviewer_attempts": 0,
        "maximum_reviewer_attempts": 0,
        "maximum_digest_calls": 0,
        "maximum_window_calls": 0,
        "maximum_merge_calls": 0,
    }
    session_ids: set[str] = set()
    for session in sessions:
        row = _exact_keys(
            session,
            {"conversation_id", "turn_count", "raw_coverage_trace_ids", "rubrics", "reviewers"},
            "session",
        )
        conversation_id = row["conversation_id"]
        if (
            not isinstance(conversation_id, str)
            or not conversation_id
            or conversation_id in session_ids
        ):
            raise ValueError("judging plan session conversation_id is invalid")
        session_ids.add(conversation_id)
        turn_count = _nonnegative_int(row["turn_count"], "session turn_count")
        coverage = row["raw_coverage_trace_ids"]
        if (
            not isinstance(coverage, list)
            or len(coverage) != turn_count
            or len(set(coverage)) != len(coverage)
            or any(not isinstance(item, str) or not item for item in coverage)
        ):
            raise ValueError("judging plan session raw coverage is invalid")
        reviewers = row["reviewers"]
        if not isinstance(reviewers, list) or len(reviewers) != panel_size:
            raise ValueError("judging plan reviewers must match panel_size")
        applicable_count = sum(
            isinstance(reviewer, dict) and reviewer.get("status") == "planned"
            for reviewer in reviewers
        )
        expected_rubrics = [
            {
                **rubric,
                "minimum_reviewer_attempts": applicable_count,
                "maximum_reviewer_attempts": applicable_count,
            }
            for rubric in requested
        ]
        if row["rubrics"] != expected_rubrics:
            raise ValueError("judging plan session rubrics or attempt bounds are invalid")
        totals_expected["turns_considered"] += turn_count
        totals_expected["minimum_reviewer_attempts"] += len(requested) * applicable_count
        totals_expected["maximum_reviewer_attempts"] += len(requested) * applicable_count
        for ordinal, reviewer in enumerate(reviewers, start=1):
            reviewer_row = _exact_keys(
                reviewer,
                {"ordinal", "judge", "status", "skip_reason", "window_plan", "work_bounds"},
                "reviewer",
            )
            judge = _exact_keys(
                reviewer_row["judge"],
                {
                    "id",
                    "label",
                    "provider",
                    "provider_model",
                    "family",
                    "supported_roles",
                    "max_input_tokens",
                    "token_counter",
                    "role",
                    "position",
                },
                "reviewer judge",
            )
            if (
                type(reviewer_row["ordinal"]) is not int
                or reviewer_row["ordinal"] != ordinal
                or type(judge["position"]) is not int
                or judge["position"] != ordinal
                or judge["role"] != "judge"
            ):
                raise ValueError("judging plan reviewer ordinals are invalid")
            roles = judge["supported_roles"]
            if (
                any(
                    not isinstance(judge[key], str) or not judge[key].strip()
                    for key in ("id", "label", "provider", "provider_model", "family")
                )
                or not isinstance(roles, list)
                or "judge" not in roles
                or any(not isinstance(role, str) or not role for role in roles)
                or len(roles) != len(set(roles))
            ):
                raise ValueError("judging plan reviewer judge values are invalid")
            _nonnegative_int(judge["max_input_tokens"], "judge max_input_tokens", positive=True)
            if judge["token_counter"] not in {
                "utf8_bytes_div_3",
                "o200k_base",
                "o200k_harmony",
            }:
                raise ValueError("judging plan reviewer token counter is invalid")
            bounds = _exact_keys(
                reviewer_row["work_bounds"],
                {"digest_calls", "window_calls_per_rubric", "merge_calls_per_rubric"},
                "reviewer work bounds",
            )
            for key in bounds:
                _nonnegative_int(bounds[key], f"reviewer work bound {key}")
            if reviewer_row["status"] == "skipped":
                if (
                    reviewer_row["skip_reason"] != "insufficient_context_capacity"
                    or reviewer_row["window_plan"] is not None
                    or any(bounds.values())
                ):
                    raise ValueError("judging plan skipped reviewer disposition is invalid")
                continue
            if reviewer_row["status"] != "planned" or reviewer_row["skip_reason"] is not None:
                raise ValueError("judging plan reviewer disposition is invalid")
            window_plan = _validate_window_plan(
                reviewer_row["window_plan"],
                conversation_id,
                coverage,
                input_policy,
                judge["max_input_tokens"],
                judge["token_counter"],
            )
            chunk_count = window_plan["chunk_count"]
            if bounds != {
                "digest_calls": chunk_count,
                "window_calls_per_rubric": chunk_count,
                "merge_calls_per_rubric": 1,
            }:
                raise ValueError("judging plan reviewer work bounds are invalid")
            totals_expected["windows_planned"] += chunk_count
            totals_expected["maximum_digest_calls"] += chunk_count
            totals_expected["maximum_window_calls"] += chunk_count * len(requested)
            totals_expected["maximum_merge_calls"] += len(requested)
    if any(type(value["totals"].get(key)) is not int for key in totals_expected):
        raise ValueError("judging plan totals types are invalid")
    if value["totals"] != totals_expected:
        raise ValueError("judging plan totals are inconsistent")


def _validate_window_plan(
    value: object,
    conversation_id: str,
    coverage: list[str],
    input_policy: dict[str, Any],
    model_limit: int,
    token_counter: str,
) -> dict[str, Any]:
    keys = {
        "plan_id",
        "contract_version",
        "conversation_id",
        "input_cap_tokens",
        "raw_budget_tokens",
        "target_raw_tokens",
        "chunk_count",
        "overlap_turns",
        "token_counter",
        "capacity_reserve_tokens",
        "merge_input_tokens",
        "raw_turns",
        "raw_coverage_trace_ids",
        "windows",
    }
    row = _exact_keys(value, keys, "window plan")
    if (
        row["contract_version"] != "3"
        or row["conversation_id"] != conversation_id
        or row["raw_coverage_trace_ids"] != coverage
    ):
        raise ValueError("judging plan window identity is invalid")
    for key in (
        "input_cap_tokens",
        "raw_budget_tokens",
        "target_raw_tokens",
        "capacity_reserve_tokens",
        "merge_input_tokens",
    ):
        _nonnegative_int(row[key], f"window {key}")
    if row["overlap_turns"] != 1 or row["token_counter"] != token_counter:
        raise ValueError("judging plan window policy is invalid")
    body = {key: item for key, item in row.items() if key != "plan_id"}
    if row["plan_id"] != _canonical_digest(body):
        raise ValueError("judging plan window plan ID is invalid")
    windows = row["windows"]
    raw_turns = row["raw_turns"]
    if not isinstance(windows, list) or not isinstance(raw_turns, list):
        raise ValueError("judging plan window collections are invalid")
    chunk_count = _nonnegative_int(row["chunk_count"], "window chunk_count")
    if chunk_count > input_policy["max_chunks"]:
        raise ValueError("judging plan window exceeds the maximum chunk count")
    input_cap = model_limit
    base_reserve = (
        input_policy["prompt_reserve_tokens"]
        + input_policy["output_reserve_tokens"]
        + input_policy["safety_reserve_tokens"]
    )
    tier_reserve = (
        input_policy["large_model_reserve_tokens"]
        if model_limit > input_policy["large_model_threshold_tokens"]
        else input_policy["small_model_reserve_tokens"]
    )
    expected_capacity_reserve = max(
        tier_reserve,
        base_reserve + max(0, chunk_count - 1) * input_policy["digest_max_tokens"],
    )
    expected_raw_budget = input_cap - expected_capacity_reserve
    expected_raw_target = min(
        input_policy["large_model_raw_target_tokens"]
        if model_limit > input_policy["large_model_threshold_tokens"]
        else input_policy["small_model_raw_target_tokens"],
        expected_raw_budget,
    )
    expected_merge = base_reserve + chunk_count * (
        input_policy["digest_max_tokens"] + input_policy["finding_max_tokens"]
    )
    if (
        row["input_cap_tokens"] != input_cap
        or row["capacity_reserve_tokens"] != expected_capacity_reserve
        or row["raw_budget_tokens"] != expected_raw_budget
        or row["target_raw_tokens"] != expected_raw_target
        or row["merge_input_tokens"] != expected_merge
    ):
        raise ValueError("judging plan window token bounds are inconsistent")
    if row["merge_input_tokens"] > row["input_cap_tokens"]:
        raise ValueError("judging plan merge input exceeds the input cap")
    if (
        len(windows) != chunk_count
        or [item.get("trace_id") if isinstance(item, dict) else None for item in raw_turns]
        != coverage
    ):
        raise ValueError("judging plan window coverage is invalid")
    for position, raw_turn in enumerate(raw_turns, start=1):
        raw = _exact_keys(
            raw_turn,
            {"trace_id", "position", "estimated_tokens", "raw_digest"},
            "raw turn",
        )
        if (
            raw["trace_id"] != coverage[position - 1]
            or raw["position"] != position
            or not _is_sha256_digest(raw["raw_digest"])
        ):
            raise ValueError("judging plan raw turn digest or identity is invalid")
        _nonnegative_int(raw["estimated_tokens"], "raw turn estimated_tokens", positive=True)
    covered: list[str] = []
    raw_digests = {raw["trace_id"]: raw["raw_digest"] for raw in raw_turns}
    for index, window in enumerate(windows, start=1):
        item = _exact_keys(
            window,
            {
                "window_id",
                "index",
                "core_trace_ids",
                "raw_trace_ids",
                "raw_turn_digests",
                "raw_tokens",
            },
            "window",
        )
        if (
            item["index"] != index
            or not isinstance(item["core_trace_ids"], list)
            or not isinstance(item["raw_trace_ids"], list)
            or not isinstance(item["raw_turn_digests"], list)
        ):
            raise ValueError("judging plan window values are invalid")
        if any(trace_id not in coverage for trace_id in item["raw_trace_ids"]):
            raise ValueError("judging plan raw window coverage is invalid")
        if len(item["raw_trace_ids"]) != len(item["raw_turn_digests"]):
            raise ValueError("judging plan raw window digests are invalid")
        _nonnegative_int(item["raw_tokens"], "window raw_tokens")
        if item["raw_tokens"] > row["raw_budget_tokens"]:
            raise ValueError("judging plan window raw_tokens exceed the raw budget")
        core_positions = [coverage.index(trace_id) for trace_id in item["core_trace_ids"]]
        if not core_positions or core_positions != list(
            range(core_positions[0], core_positions[-1] + 1)
        ):
            raise ValueError("judging plan core window geometry is invalid")
        raw_positions = [coverage.index(trace_id) for trace_id in item["raw_trace_ids"]]
        allowed_raw_starts = {core_positions[0], max(0, core_positions[0] - 1)}
        allowed_raw_ends = {
            core_positions[-1],
            min(len(coverage) - 1, core_positions[-1] + 1),
        }
        raw_geometry_valid = (
            raw_positions
            and raw_positions == list(range(raw_positions[0], raw_positions[-1] + 1))
            and raw_positions[0] in allowed_raw_starts
            and raw_positions[-1] in allowed_raw_ends
        )
        if not raw_geometry_valid or item["raw_turn_digests"] != [
            raw_digests[trace_id] for trace_id in item["raw_trace_ids"]
        ]:
            raise ValueError("judging plan raw window geometry is invalid")
        window_body = {key: part for key, part in item.items() if key != "window_id"}
        if item["window_id"] != _canonical_digest(window_body):
            raise ValueError("judging plan window ID is invalid")
        covered.extend(item["core_trace_ids"])
    if covered != coverage:
        raise ValueError("judging plan core window coverage is invalid")
    return row


_JUDGING_ARTIFACT_FIELDS = frozenset({"schema_version", "kind", "content_digest", "payload"})
_JUDGING_ARTIFACT_KINDS = frozenset({"chunk_digest", "window_findings", "merged_verdict"})


def _validate_artifact_id(artifact_id: str) -> None:
    if not isinstance(artifact_id, str):
        raise ValueError("judging artifact ID must be a string")
    parts = artifact_id.split("/")
    if (
        len(parts) < 2
        or artifact_id != artifact_id.strip()
        or any(not part.strip() or part != part.strip() for part in parts)
    ):
        raise ValueError("judging artifact ID must be a nonblank slash-delimited string")


def _canonical_json_object(
    value: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        normalized = json.loads(encoded)
        canonical = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc
    return normalized, canonical


def _canonical_judging_artifact_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    normalized, canonical = _canonical_json_object(payload, label="judging artifact payload")
    digest = f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
    return normalized, digest


def judging_artifact_payload_digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical content digest for one JSON-object artifact payload."""

    return _canonical_judging_artifact_payload(payload)[1]


def _canonical_judging_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise ValueError("judging artifact must be a JSON object")
    value = dict(artifact)
    if set(value) != _JUDGING_ARTIFACT_FIELDS:
        raise ValueError(
            "judging artifact body must contain exactly schema_version, kind, "
            "content_digest, and payload"
        )
    if value["schema_version"] != "1":
        raise ValueError("judging artifact schema_version must be '1'")
    if not isinstance(value["kind"], str) or value["kind"] not in _JUDGING_ARTIFACT_KINDS:
        raise ValueError(
            "judging artifact kind must be chunk_digest, window_findings, or merged_verdict"
        )
    if not isinstance(value["content_digest"], str):
        raise ValueError("judging artifact content_digest must be a string")
    normalized_payload, expected_digest = _canonical_judging_artifact_payload(value["payload"])
    if value["content_digest"] != expected_digest:
        raise ValueError("judging artifact content digest does not match its payload")
    value["payload"] = normalized_payload
    return value


def _decode_judging_artifacts(encoded: str | None) -> dict[str, Any] | None:
    if encoded is None:
        return None
    try:
        stored = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("stored judging artifacts must be a valid JSON object") from exc
    if not isinstance(stored, dict):
        raise ValueError("stored judging artifacts must be a JSON object map")
    canonical: dict[str, Any] = {}
    for artifact_id, artifact in stored.items():
        _validate_artifact_id(artifact_id)
        canonical[artifact_id] = _canonical_judging_artifact(artifact)
    return canonical


def _judging_plan_matches_cohort(
    judging_plan: Mapping[str, Any],
    turn_cohort: Mapping[str, Any],
) -> bool:
    cohort_sessions = turn_cohort.get("sessions")
    cohort_turns = turn_cohort.get("turns")
    if not isinstance(cohort_sessions, list) or not isinstance(cohort_turns, list):
        return False

    session_turn_counts: dict[str, int] = {}
    for session in cohort_sessions:
        if not isinstance(session, dict):
            return False
        conversation_id = session.get("conversation_id")
        turn_count = session.get("turn_count")
        if (
            not isinstance(conversation_id, str)
            or conversation_id in session_turn_counts
            or type(turn_count) is not int
        ):
            return False
        session_turn_counts[conversation_id] = turn_count

    turn_sessions: dict[str, str] = {}
    for turn in cohort_turns:
        if not isinstance(turn, dict):
            return False
        trace_id = turn.get("trace_id")
        conversation_id = turn.get("conversation_id")
        if (
            not isinstance(trace_id, str)
            or trace_id in turn_sessions
            or not isinstance(conversation_id, str)
        ):
            return False
        turn_sessions[trace_id] = conversation_id

    totals = judging_plan.get("totals")
    sessions = judging_plan.get("sessions")
    if not isinstance(totals, dict) or not isinstance(sessions, list):
        return False
    if totals.get("turns_considered") != len(cohort_turns):
        return False

    planned_session_ids: set[str] = set()
    for session in sessions:
        if not isinstance(session, dict):
            return False
        conversation_id = session.get("conversation_id")
        if (
            not isinstance(conversation_id, str)
            or conversation_id in planned_session_ids
            or session.get("turn_count") != session_turn_counts.get(conversation_id)
        ):
            return False
        planned_session_ids.add(conversation_id)
        coverage = session.get("raw_coverage_trace_ids")
        reviewers = session.get("reviewers")
        if not isinstance(coverage, list) or not isinstance(reviewers, list):
            return False
        expected_coverage = [
            trace_id for trace_id, owner in turn_sessions.items() if owner == conversation_id
        ]
        if coverage != expected_coverage or len(coverage) != session["turn_count"]:
            return False
        ordinals: list[int] = []
        for reviewer in reviewers:
            if not isinstance(reviewer, dict):
                return False
            ordinal = reviewer.get("ordinal")
            judge = reviewer.get("judge")
            status = reviewer.get("status")
            window_plan = reviewer.get("window_plan")
            if type(ordinal) is not int or not isinstance(judge, dict):
                return False
            if judge.get("position") != ordinal:
                return False
            if status == "skipped":
                if window_plan is not None:
                    return False
            elif status != "planned" or not isinstance(window_plan, dict):
                return False
            elif (
                window_plan.get("conversation_id") != conversation_id
                or window_plan.get("raw_coverage_trace_ids") != coverage
            ):
                return False
            ordinals.append(ordinal)
        if ordinals != list(range(1, len(reviewers) + 1)):
            return False
    return planned_session_ids == set(session_turn_counts)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'created',
    created_at TEXT NOT NULL,
    auto_run INTEGER NOT NULL DEFAULT 0,
    data_selection TEXT,
    run_config TEXT,
    effective_config TEXT,
    turn_cohort TEXT,
    judging_plan TEXT,
    judging_artifacts TEXT,
    reflection_input TEXT,
    scoring_progress TEXT,
    scoring_result TEXT,
    scoring_succeeded INTEGER NOT NULL DEFAULT 0,
    judging_progress TEXT,
    judging_result TEXT,
    judging_succeeded INTEGER NOT NULL DEFAULT 0,
    reflecting_progress TEXT,
    reflecting_result TEXT,
    reflecting_succeeded INTEGER NOT NULL DEFAULT 0,
    reflection_review TEXT,
    reflection_review_revision INTEGER NOT NULL DEFAULT 0,
    error TEXT
)
"""

_JSON_FIELDS = {
    "turn_cohort",
    "judging_plan",
    "reflection_input",
    "scoring_progress",
    "scoring_result",
    "judging_progress",
    "judging_result",
    "reflecting_progress",
    "reflecting_result",
    "reflection_review",
}


class RunStore:
    """SQLite store whose writes enforce the evaluation-run lifecycle."""

    def __init__(self, db_path: str | Path | None = None):
        self._conn = sqlite3.connect(
            str(db_path or _default_db_path()),
            check_same_thread=False,
        )
        self._lock = threading.Lock()
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("BEGIN IMMEDIATE")
            stored_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if stored_version != RUN_DB_SCHEMA_VERSION:
                self._conn.execute("DROP TABLE IF EXISTS runs")
            self._conn.execute(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {RUN_DB_SCHEMA_VERSION}")
            self._conn.commit()
        except BaseException:
            try:
                self._conn.rollback()
            finally:
                self._conn.close()
            raise

    def create(self) -> Run:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, status, created_at) VALUES (?, ?, ?)",
                (run_id, RunStatus.CREATED.value, created_at),
            )
            self._conn.commit()
            row = self._get_row_locked(run_id)
        return self._row_to_run(row)

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._row_to_run(row) if row is not None else None

    def list_summaries(self, limit: int = 50) -> list[RunSummarySource]:
        """Project scalar list fields without loading full run payloads."""

        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    run_id,
                    status,
                    created_at,
                    scoring_succeeded,
                    judging_succeeded,
                    reflecting_succeeded,
                    json_array_length(data_selection, '$.session_ids') AS session_count,
                    json_extract(data_selection, '$.since') AS selection_since,
                    json_extract(data_selection, '$.until') AS selection_until,
                    json_extract(data_selection, '$.timezone') AS selection_timezone,
                    json_extract(reflection_review, '$.status') AS review_status,
                    json_extract(reflecting_result, '$.baseline_won') AS baseline_won,
                    json_extract(reflecting_result, '$.reason') AS reflection_reason
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            RunSummarySource(
                run_id=row["run_id"],
                status=RunStatus(row["status"]),
                created_at=row["created_at"],
                scoring_succeeded=bool(row["scoring_succeeded"]),
                judging_succeeded=bool(row["judging_succeeded"]),
                reflecting_succeeded=bool(row["reflecting_succeeded"]),
                session_count=row["session_count"],
                selection_since=row["selection_since"],
                selection_until=row["selection_until"],
                selection_timezone=row["selection_timezone"],
                review_status=row["review_status"],
                baseline_won=(
                    bool(row["baseline_won"]) if row["baseline_won"] is not None else None
                ),
                reflection_reason=row["reflection_reason"],
            )
            for row in rows
        ]

    def list_active(self) -> list[Run]:
        """Return every run whose pipeline stage has not reached a terminal state."""

        statuses = tuple(stage.value for stage in _ACTIVE_STAGES)
        placeholders = ", ".join("?" for _ in statuses)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM runs WHERE status IN ({placeholders}) ORDER BY created_at ASC",
                statuses,
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def save_selection(self, run_id: str, selection: DataSelection) -> Run:
        encoded = _encode_data_selection(selection)
        with self._lock:
            return self._write_created_only_locked(
                run_id,
                "UPDATE runs SET data_selection = ? WHERE run_id = ? AND status = ?",
                (encoded,),
            )

    def save_config(self, run_id: str, config: RunConfig) -> Run:
        encoded = _encode_run_config(config)
        with self._lock:
            return self._write_created_only_locked(
                run_id,
                "UPDATE runs SET run_config = ? WHERE run_id = ? AND status = ?",
                (encoded,),
            )

    def save_auto_run(self, run_id: str, auto_run: bool) -> Run:
        if type(auto_run) is not bool:
            raise ValueError("auto_run must be a boolean")
        with self._lock:
            return self._write_created_only_locked(
                run_id,
                "UPDATE runs SET auto_run = ? WHERE run_id = ? AND status = ?",
                (int(auto_run),),
            )

    def start(
        self,
        run_id: str,
        *,
        expected_selection: DataSelection,
        expected_config: RunConfig,
        turn_cohort: Mapping[str, Any],
        effective_config: EffectiveRunConfig,
    ) -> Run:
        """Atomically pin immutable inputs and transition created to scoring."""

        encoded_selection = _encode_data_selection(expected_selection)
        encoded_config = _encode_run_config(expected_config)
        encoded_cohort = _encode_turn_cohort(turn_cohort)
        encoded_effective = _encode_effective_config(effective_config)
        if (
            effective_config.model_catalog_version != expected_config.model_catalog_version
            or effective_config.rubric_catalog_version != expected_config.rubric_catalog_version
        ):
            raise ValueError("effective configuration does not match requested catalogs")

        with self._lock:
            row = self._get_row_locked(run_id)
            current = RunStatus(row["status"])
            if current is not RunStatus.CREATED:
                raise RunLifecycleConflictError(run_id, current)
            if row["data_selection"] != encoded_selection:
                raise RunStoreConflictError(
                    f"Run {run_id} data selection changed while pinning inputs"
                )
            if row["run_config"] is None:
                raise ValueError("run configuration required before start")
            if row["run_config"] != encoded_config:
                raise RunStoreConflictError(
                    f"Run {run_id} configuration changed while pinning inputs"
                )
            if row["turn_cohort"] is not None or row["effective_config"] is not None:
                raise RunStoreConflictError(f"Run {run_id} immutable inputs are already pinned")

            cursor = self._conn.execute(
                "UPDATE runs SET status = ?, turn_cohort = ?, effective_config = ? "
                "WHERE run_id = ? AND status = ? AND data_selection = ? "
                "AND run_config = ? AND turn_cohort IS NULL AND effective_config IS NULL",
                (
                    RunStatus.SCORING.value,
                    encoded_cohort,
                    encoded_effective,
                    run_id,
                    RunStatus.CREATED.value,
                    encoded_selection,
                    encoded_config,
                ),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                latest = self._get_row_locked(run_id)
                latest_status = RunStatus(latest["status"])
                if latest_status is not RunStatus.CREATED:
                    raise RunLifecycleConflictError(run_id, latest_status)
                raise RunStoreConflictError(f"Run {run_id} inputs changed while starting")
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def record_stage_progress(
        self,
        run_id: str,
        *,
        stage: RunStatus,
        progress: Mapping[str, Any],
    ) -> Run:
        return self._record_stage_value(
            run_id,
            stage=stage,
            value=progress,
            result=False,
        )

    def record_stage_result(
        self,
        run_id: str,
        *,
        stage: RunStatus,
        result: Mapping[str, Any],
    ) -> Run:
        return self._record_stage_value(
            run_id,
            stage=stage,
            value=result,
            result=True,
        )

    def finalize_stage_success(
        self,
        run_id: str,
        *,
        stage: RunStatus,
        advance: bool,
    ) -> Run:
        """Atomically mark a returned worker successful and optionally advance it."""

        if stage not in _ACTIVE_STAGES:
            raise ValueError("stage must be scoring, judging, or reflecting")
        if type(advance) is not bool:
            raise ValueError("advance must be a boolean")
        success_field = _STAGE_SUCCESS_FIELDS[stage]
        result_field = _STAGE_FIELDS[stage][1]
        new_status = _STAGE_SUCCESSORS[stage] if advance else stage
        with self._lock:
            row = self._get_row_locked(run_id)
            current = RunStatus(row["status"])
            if current is not stage:
                raise RunStoreConflictError(
                    f"Run {run_id} expected {stage.value}; current status is {current.value}"
                )
            if row[result_field] is None:
                raise RunStoreConflictError(
                    f"Run {run_id} {stage.value} cannot finalize without a result"
                )
            cursor = self._conn.execute(
                f"UPDATE runs SET {success_field} = 1, status = ? "
                f"WHERE run_id = ? AND status = ? AND {result_field} IS NOT NULL",
                (new_status.value, run_id, stage.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise RunStoreConflictError(f"Run {run_id} changed while finalizing {stage.value}")
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def transition(
        self,
        run_id: str,
        *,
        expected_status: RunStatus,
        new_status: RunStatus,
    ) -> Run:
        if not isinstance(expected_status, RunStatus) or not isinstance(new_status, RunStatus):
            raise ValueError("transition statuses must be RunStatus values")
        if expected_status is RunStatus.CREATED and new_status is RunStatus.SCORING:
            raise RunStoreConflictError(
                f"Run {run_id} must use start() to transition from created to scoring"
            )
        if new_status not in _TRANSITIONS[expected_status]:
            raise ValueError(f"Invalid transition: {expected_status.value} -> {new_status.value}")
        success_field = (
            _STAGE_SUCCESS_FIELDS[expected_status]
            if _STAGE_SUCCESSORS.get(expected_status) is new_status
            else None
        )
        success_guard = f" AND {success_field} = 1" if success_field is not None else ""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ? AND status = ?" + success_guard,
                (new_status.value, run_id, expected_status.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                latest = self._get_row_locked(run_id)
                current = RunStatus(latest["status"])
                if (
                    current is expected_status
                    and success_field is not None
                    and not bool(latest[success_field])
                ):
                    raise RunStoreConflictError(
                        f"Run {run_id} {expected_status.value} has not finalized successfully"
                    )
                raise RunStoreConflictError(
                    f"Run {run_id} expected {expected_status.value}; current status is "
                    f"{current.value}"
                )
            self._conn.commit()
            row = self._get_row_locked(run_id)
        return self._row_to_run(row)

    def fail(
        self,
        run_id: str,
        *,
        expected_status: RunStatus,
        error: str,
    ) -> Run:
        if (
            expected_status not in _TRANSITIONS
            or RunStatus.FAILED not in _TRANSITIONS[expected_status]
        ):
            raise ValueError(f"Run status {expected_status.value} cannot fail")
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be a nonblank string")
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runs SET status = ?, error = ? WHERE run_id = ? AND status = ?",
                (
                    RunStatus.FAILED.value,
                    error,
                    run_id,
                    expected_status.value,
                ),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                current = RunStatus(self._get_row_locked(run_id)["status"])
                raise RunStoreConflictError(
                    f"Run {run_id} expected {expected_status.value}; current status is "
                    f"{current.value}"
                )
            self._conn.commit()
            row = self._get_row_locked(run_id)
        return self._row_to_run(row)

    def pin_reflection_input(
        self,
        run_id: str,
        reflection_input: Mapping[str, Any],
    ) -> Run:
        encoded = _encode_reflection_input(reflection_input)
        value = json.loads(encoded)
        with self._lock:
            row = self._get_row_locked(run_id)
            if RunStatus(row["status"]) is not RunStatus.REFLECTING:
                raise RunStoreConflictError(
                    f"Run {run_id} can only pin reflection input while reflecting; "
                    f"current status is {row['status']}"
                )
            current = json.loads(row["reflection_input"]) if row["reflection_input"] else None
            if current is not None:
                if current == value:
                    return self._row_to_run(row)
                raise RunStoreConflictError(f"Run {run_id} reflection input is already pinned")
            cursor = self._conn.execute(
                "UPDATE runs SET reflection_input = ? "
                "WHERE run_id = ? AND status = ? AND reflection_input IS NULL",
                (encoded, run_id, RunStatus.REFLECTING.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise RunStoreConflictError(f"Run {run_id} changed while pinning reflection input")
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def pin_judging_plan(
        self,
        run_id: str,
        judging_plan: Mapping[str, Any],
    ) -> Run:
        encoded = _encode_judging_plan(judging_plan)
        value = json.loads(encoded)
        with self._lock:
            row = self._get_row_locked(run_id)
            if RunStatus(row["status"]) is not RunStatus.JUDGING:
                raise RunStoreConflictError(
                    f"Run {run_id} can only pin a judging plan while judging; "
                    f"current status is {row['status']}"
                )
            cohort = json.loads(row["turn_cohort"]) if row["turn_cohort"] else None
            if not isinstance(cohort, dict) or cohort.get("cohort_id") != value["cohort_id"]:
                raise RunStoreConflictError(
                    f"Run {run_id} judging plan cohort does not match its pinned cohort"
                )
            if not _judging_plan_matches_cohort(value, cohort):
                raise RunStoreConflictError(
                    f"Run {run_id} judging plan contents do not match its pinned cohort"
                )
            current = json.loads(row["judging_plan"]) if row["judging_plan"] else None
            if current is not None:
                if current == value:
                    return self._row_to_run(row)
                raise RunStoreConflictError(f"Run {run_id} judging plan is already pinned")
            cursor = self._conn.execute(
                "UPDATE runs SET judging_plan = ? "
                "WHERE run_id = ? AND status = ? AND judging_plan IS NULL",
                (encoded, run_id, RunStatus.JUDGING.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise RunStoreConflictError(f"Run {run_id} changed while pinning judging plan")
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def record_judging_artifact(
        self,
        run_id: str,
        artifact_id: str,
        artifact: Mapping[str, Any],
    ) -> Run:
        """Persist one immutable, content-addressed artifact during judging."""

        _validate_artifact_id(artifact_id)
        value = _canonical_judging_artifact(artifact)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._get_row_locked(run_id)
                current_status = RunStatus(row["status"])
                if current_status is not RunStatus.JUDGING:
                    raise RunStoreConflictError(
                        f"Run {run_id} can only record judging artifacts while judging; "
                        f"current status is {current_status.value}"
                    )
                artifacts = _decode_judging_artifacts(row["judging_artifacts"]) or {}
                existing = artifacts.get(artifact_id)
                if existing is not None:
                    if existing != value:
                        raise RunStoreConflictError(
                            f"Run {run_id} judging artifact already exists with different content: "
                            f"{artifact_id}"
                        )
                    self._conn.commit()
                    return self._row_to_run(row)

                artifacts[artifact_id] = value
                encoded = _encode_json_object(artifacts, "judging artifacts")
                cursor = self._conn.execute(
                    "UPDATE runs SET judging_artifacts = ? WHERE run_id = ? AND status = ?",
                    (encoded, run_id, RunStatus.JUDGING.value),
                )
                if cursor.rowcount != 1:
                    raise RunStoreConflictError(
                        f"Run {run_id} changed while recording judging artifact {artifact_id}"
                    )
                updated = self._get_row_locked(run_id)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return self._row_to_run(updated)

    @contextmanager
    def external_write_barrier(
        self,
        run_id: str,
        expected_status: RunStatus,
    ) -> Iterator[None]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._get_row_locked(run_id)
                current = RunStatus(row["status"])
                if current is not expected_status:
                    raise RunStoreConflictError(
                        f"Run {run_id} is no longer {expected_status.value}; "
                        f"current status is {current.value}"
                    )
                yield
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def initialize_reflection_review(
        self,
        run_id: str,
        review: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> Run:
        return self._write_reflection_review(
            run_id,
            review,
            expected_revision=expected_revision,
            initialize=True,
        )

    def update_reflection_review(
        self,
        run_id: str,
        review: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> Run:
        return self._write_reflection_review(
            run_id,
            review,
            expected_revision=expected_revision,
            initialize=False,
        )

    def cancel_if_safe(self, run_id: str) -> Run:
        cancellable = tuple(status.value for status in _TRANSITIONS if _TRANSITIONS[status])
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ? "
                "AND status IN (?, ?, ?, ?) "
                "AND reflecting_result IS NULL AND reflection_review IS NULL",
                (RunStatus.CANCELLED.value, run_id, *cancellable),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                row = self._get_row_locked(run_id)
                raise RunCancellationConflictError(
                    run_id,
                    RunStatus(row["status"]),
                    reflection_finalizing=(
                        row["reflecting_result"] is not None or row["reflection_review"] is not None
                    ),
                )
            self._conn.commit()
            row = self._get_row_locked(run_id)
        return self._row_to_run(row)

    def _record_stage_value(
        self,
        run_id: str,
        *,
        stage: RunStatus,
        value: Mapping[str, Any],
        result: bool,
    ) -> Run:
        if stage not in _ACTIVE_STAGES:
            raise ValueError("stage must be scoring, judging, or reflecting")
        field_name = _STAGE_FIELDS[stage][1 if result else 0]
        encoded = _encode_json_object(value, field_name)
        with self._lock:
            row = self._get_row_locked(run_id)
            current = RunStatus(row["status"])
            if current is not stage:
                raise RunStoreConflictError(
                    f"Run {run_id} is no longer {stage.value}; current status is {current.value}"
                )
            if field_name == "reflecting_result" and row["reflection_review"] is not None:
                current_review = json.loads(row["reflection_review"])
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    "reflection evidence is immutable once review is initialized",
                    current_run_status=current,
                    current_review_status=current_review.get("status"),
                    current_revision=int(row["reflection_review_revision"]),
                )
            review_guard = (
                " AND reflection_review IS NULL" if field_name == "reflecting_result" else ""
            )
            cursor = self._conn.execute(
                f"UPDATE runs SET {field_name} = ? WHERE run_id = ? AND status = ?{review_guard}",
                (encoded, run_id, stage.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                latest = self._get_row_locked(run_id)
                latest_review = (
                    json.loads(latest["reflection_review"])
                    if latest["reflection_review"] is not None
                    else None
                )
                if field_name == "reflecting_result" and latest_review is not None:
                    raise ReflectionReviewLifecycleConflictError(
                        run_id,
                        "reflection evidence is immutable once review is initialized",
                        current_run_status=RunStatus(latest["status"]),
                        current_review_status=latest_review.get("status"),
                        current_revision=int(latest["reflection_review_revision"]),
                    )
                raise RunStoreConflictError(f"Run {run_id} changed while writing {field_name}")
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def _write_created_only_locked(
        self,
        run_id: str,
        statement: str,
        values: tuple[Any, ...],
    ) -> Run:
        row = self._get_row_locked(run_id)
        current = RunStatus(row["status"])
        if current is not RunStatus.CREATED:
            raise RunLifecycleConflictError(run_id, current)
        cursor = self._conn.execute(
            statement,
            (*values, run_id, RunStatus.CREATED.value),
        )
        if cursor.rowcount != 1:
            self._conn.rollback()
            latest = self._get_row_locked(run_id)
            raise RunLifecycleConflictError(run_id, RunStatus(latest["status"]))
        self._conn.commit()
        return self._row_to_run(self._get_row_locked(run_id))

    def _write_reflection_review(
        self,
        run_id: str,
        review: Mapping[str, Any],
        *,
        expected_revision: int,
        initialize: bool,
    ) -> Run:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        encoded = _encode_json_object(review, "reflection review")
        value = json.loads(encoded)
        new_status = value.get("status")
        if new_status not in _REFLECTION_REVIEW_STATUSES:
            allowed = ", ".join(sorted(_REFLECTION_REVIEW_STATUSES))
            raise ValueError(f"reflection review status must be one of: {allowed}")
        if initialize and new_status != "pending":
            raise ValueError("initial reflection review status must be pending")

        with self._lock:
            row = self._get_row_locked(run_id)
            run_status = RunStatus(row["status"])
            revision = int(row["reflection_review_revision"])
            current_review = (
                json.loads(row["reflection_review"])
                if row["reflection_review"] is not None
                else None
            )
            review_status = current_review.get("status") if current_review else None
            required_status = RunStatus.REFLECTING if initialize else RunStatus.COMPLETE
            if run_status is not required_status:
                message = (
                    f"run status {run_status.value} is not reviewable"
                    if initialize
                    else "review mutations require a completed run; "
                    f"current status is {run_status.value}"
                )
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    message,
                    current_run_status=run_status,
                    current_review_status=review_status,
                    current_revision=revision,
                )
            if revision != expected_revision:
                raise ReflectionReviewRevisionConflictError(
                    run_id,
                    expected_revision,
                    revision,
                    current_review,
                )
            if initialize and current_review is not None:
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    "review is already initialized",
                    current_run_status=run_status,
                    current_review_status=review_status,
                    current_revision=revision,
                )
            if not initialize and current_review is None:
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    "review is not initialized",
                    current_run_status=run_status,
                    current_review_status=None,
                    current_revision=revision,
                )
            if review_status in _RESOLVED_REFLECTION_REVIEW_STATUSES:
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    f"review is already resolved as {review_status}",
                    current_run_status=run_status,
                    current_review_status=review_status,
                    current_revision=revision,
                )
            if not initialize and review_status != "pending":
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    "stored review status is invalid",
                    current_run_status=run_status,
                    current_review_status=review_status,
                    current_revision=revision,
                )

            evidence_guard = ""
            expected_evidence: str | None = None
            if initialize:
                expected_evidence = row["reflecting_result"]
                if expected_evidence is None:
                    raise ReflectionReviewLifecycleConflictError(
                        run_id,
                        "reflection evidence is not finalized",
                        current_run_status=run_status,
                        current_review_status=review_status,
                        current_revision=revision,
                    )
                evidence = json.loads(expected_evidence)
                candidates = evidence.get("candidates") if isinstance(evidence, dict) else None
                selected_id = value.get("selected_candidate_id")
                if not (
                    isinstance(selected_id, str)
                    and selected_id
                    and isinstance(candidates, list)
                    and any(
                        isinstance(candidate, dict) and candidate.get("candidate_id") == selected_id
                        for candidate in candidates
                    )
                ):
                    raise ReflectionReviewLifecycleConflictError(
                        run_id,
                        "selected candidate is not present in finalized reflection evidence",
                        current_run_status=run_status,
                        current_review_status=review_status,
                        current_revision=revision,
                    )
                evidence_guard = " AND reflecting_result = ?"

            parameters: tuple[Any, ...] = (
                encoded,
                run_id,
                expected_revision,
                required_status.value,
            )
            if initialize:
                parameters = (*parameters, expected_evidence)
            cursor = self._conn.execute(
                "UPDATE runs SET reflection_review = ?, "
                "reflection_review_revision = reflection_review_revision + 1 "
                "WHERE run_id = ? AND reflection_review_revision = ? AND status = ?"
                f"{evidence_guard}",
                parameters,
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                latest = self._get_row_locked(run_id)
                latest_revision = int(latest["reflection_review_revision"])
                latest_review = (
                    json.loads(latest["reflection_review"])
                    if latest["reflection_review"] is not None
                    else None
                )
                if latest_revision != expected_revision:
                    raise ReflectionReviewRevisionConflictError(
                        run_id,
                        expected_revision,
                        latest_revision,
                        latest_review,
                    )
                raise ReflectionReviewLifecycleConflictError(
                    run_id,
                    "run or reflection evidence changed while writing review",
                    current_run_status=RunStatus(latest["status"]),
                    current_review_status=(latest_review.get("status") if latest_review else None),
                    current_revision=latest_revision,
                )
            self._conn.commit()
            updated = self._get_row_locked(run_id)
        return self._row_to_run(updated)

    def _get_row_locked(self, run_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Run {run_id} not found")
        return row

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        data = dict(row)
        data["status"] = RunStatus(data["status"])
        data["auto_run"] = bool(data["auto_run"])
        for field_name in _STAGE_SUCCESS_FIELDS.values():
            data[field_name] = bool(data[field_name])
        raw_selection = data.pop("data_selection")
        raw_config = data.pop("run_config")
        raw_effective = data.pop("effective_config")
        raw_judging_artifacts = data.pop("judging_artifacts")
        for field_name in _JSON_FIELDS:
            if data.get(field_name) is not None:
                data[field_name] = json.loads(data[field_name])
        data["data_selection"] = (
            DataSelection(**json.loads(raw_selection)) if raw_selection is not None else None
        )
        data["run_config"] = (
            RunConfig.model_validate_json(raw_config) if raw_config is not None else None
        )
        data["effective_config"] = (
            EffectiveRunConfig.model_validate_json(raw_effective)
            if raw_effective is not None
            else None
        )
        data["judging_artifacts"] = _decode_judging_artifacts(raw_judging_artifacts)
        return Run(**data)

    def close(self) -> None:
        self._conn.close()
