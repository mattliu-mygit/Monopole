"""Authoritative Pydantic response models for the HTTP boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel

from weave_agent_signals.judges.tokens import TokenCounterName
from weave_agent_signals.run_config import (
    EffectiveRunConfig,
    JudgingContextPolicy,
    ModelDescriptor,
    PositionedJudge,
    RubricDescriptor,
    RunConfig,
)
from weave_agent_signals.runs.challenges.contracts import ChallengeResult


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSelectionResponse(ResponseModel):
    since: str | None
    until: str | None
    timezone: str | None
    session_ids: list[str]


class ScopeDescriptorResponse(ResponseModel):
    kind: str
    target_id: str
    patterns: list[str]


class TargetSnapshotResponse(ResponseModel):
    kind: str
    locator: str
    display_name: str
    path: str | None
    exists: bool
    content: str | None
    revision: str


class BundleSnapshotResponse(ResponseModel):
    scope: ScopeDescriptorResponse | None
    targets: list[TargetSnapshotResponse]
    revision: str


class EvaluatorRecordResponse(ResponseModel):
    evaluation_id: str
    target_revision: str
    requested_model: str
    requested_family: str
    requested_backend: str
    resolved_model: str
    resolved_family: str
    resolved_backend: str
    score: float
    rationale: str
    usage: dict[str, int]


class GenerationAttemptResponse(ResponseModel):
    attempt_id: str
    number: int
    status: Literal["succeeded", "failed", "cancelled"]
    requested_writer: ModelDescriptor
    resolved_model: str | None
    resolved_family: str | None
    resolved_backend: str | None
    usage: dict[str, int]
    candidate_revision: str | None
    changed_paths: list[str]
    response_digest: str | None
    response_excerpt: str | None
    error_type: str | None
    error: str | None


class ReflectionCandidateResponse(ResponseModel):
    candidate_id: str
    bundle: BundleSnapshotResponse
    score: float
    score_delta: float
    rationale: str
    generation_attempt_id: str
    requested_writer: ModelDescriptor
    resolved_writer_model: str
    resolved_writer_family: str
    resolved_writer_backend: str
    evaluation: EvaluatorRecordResponse


class SuccessfulReflectionResultResponse(ResponseModel):
    baseline: BundleSnapshotResponse
    baseline_score: float
    baseline_evaluation: EvaluatorRecordResponse
    candidates: list[ReflectionCandidateResponse]
    generation_attempts: list[GenerationAttemptResponse]
    recommended_candidate_id: str | None
    provisional_candidate_id: str | None
    baseline_won: bool
    reason: str | None
    score_basis: Literal["predicted_evaluator"]
    challenge: ChallengeResult | None


class EmptyReflectionResultResponse(ResponseModel):
    candidates: list[ReflectionCandidateResponse]
    reason: str


class ReflectionResultResponse(
    RootModel[SuccessfulReflectionResultResponse | EmptyReflectionResultResponse]
):
    pass


class ReflectionDraftResponse(ResponseModel):
    candidate_id: str
    revision: str
    bundle: BundleSnapshotResponse


class PromotionTargetOutcomeResponse(ResponseModel):
    locator: str
    action: Literal["create", "update"]
    status: Literal["applied", "not_applied"]
    reason: Literal["source_drift", "write_failed"] | None
    message: str | None


class PromotionReceiptResponse(ResponseModel):
    promotion_id: str
    run_id: str
    candidate_id: str
    past: BundleSnapshotResponse
    evaluated_candidate: BundleSnapshotResponse
    promoted: BundleSnapshotResponse
    review_revision: int
    outcomes: list[PromotionTargetOutcomeResponse]
    decided_at: str
    promoted_was_evaluated: bool
    unevaluated_d_acknowledged: bool


class ReflectionReviewResponse(ResponseModel):
    status: Literal["pending", "promoted", "partial", "dismissed"]
    selected_candidate_id: str
    draft: ReflectionDraftResponse | None
    stale: bool | None = None
    changed_targets: list[str] | None = None
    current: BundleSnapshotResponse | None = None
    stale_reason: str | None = None
    receipt: PromotionReceiptResponse | None = None
    dismissed_at: str | None = None


class ScoringTurnDetailResponse(ResponseModel):
    trace_id: str
    conversation_id: str
    model: str | None
    user_input: str
    tokens: int
    tool_count: int
    tools_used: list[str]
    errors: int
    steering: int
    denials: int
    scores: dict[str, float]


class ScoringProgressResponse(ResponseModel):
    total: int
    scored: int
    written: int
    status_message: str
    turn_details: list[ScoringTurnDetailResponse]


class ScoringResultResponse(ResponseModel):
    turns_scored: int
    sessions_scored: int
    scores_written: int
    errors: int
    turn_details: list[ScoringTurnDetailResponse]


class JudgingPlanTotalsResponse(ResponseModel):
    sessions_planned: int
    turns_considered: int
    windows_planned: int
    planned_rubrics: int
    minimum_reviewer_attempts: int
    maximum_reviewer_attempts: int
    maximum_digest_calls: int
    maximum_window_calls: int
    maximum_merge_calls: int


class JudgingPlanRubricResponse(RubricDescriptor):
    minimum_reviewer_attempts: int
    maximum_reviewer_attempts: int


class JudgingRawTurnResponse(ResponseModel):
    trace_id: str
    position: int
    estimated_tokens: int
    raw_digest: str


class JudgingWindowResponse(ResponseModel):
    window_id: str
    index: int
    core_trace_ids: list[str]
    raw_trace_ids: list[str]
    raw_turn_digests: list[str]
    raw_tokens: int


class JudgingWindowPlanResponse(ResponseModel):
    plan_id: str
    contract_version: Literal["3"]
    conversation_id: str
    input_cap_tokens: int
    raw_budget_tokens: int
    target_raw_tokens: int
    chunk_count: int
    overlap_turns: int
    token_counter: TokenCounterName
    capacity_reserve_tokens: int
    merge_input_tokens: int
    raw_turns: list[JudgingRawTurnResponse]
    raw_coverage_trace_ids: list[str]
    windows: list[JudgingWindowResponse]


class JudgingWorkBoundsResponse(ResponseModel):
    digest_calls: int
    window_calls_per_rubric: int
    merge_calls_per_rubric: int


class JudgingReviewerPlanResponse(ResponseModel):
    ordinal: int
    judge: PositionedJudge
    status: Literal["planned", "skipped"]
    skip_reason: Literal["insufficient_context_capacity"] | None
    window_plan: JudgingWindowPlanResponse | None
    work_bounds: JudgingWorkBoundsResponse


class JudgingPlanSessionResponse(ResponseModel):
    conversation_id: str
    turn_count: int
    raw_coverage_trace_ids: list[str]
    rubrics: list[JudgingPlanRubricResponse]
    reviewers: list[JudgingReviewerPlanResponse]


class JudgingPlanResponse(ResponseModel):
    plan_id: str
    schema_version: str
    cohort_id: str
    panel_size: int
    requested_rubrics: list[RubricDescriptor]
    input_policy: JudgingContextPolicy
    protocol: dict[str, Any]
    totals: JudgingPlanTotalsResponse
    sessions: list[JudgingPlanSessionResponse]


class BehavioralFeedbackResponse(ResponseModel):
    success: str | None
    problem: str | None
    desired_behavior: str | None


class InferenceResponseDiagnosticResponse(ResponseModel):
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    completion_details: dict[str, int] = Field(default_factory=dict)
    content_characters: int = 0


class InferenceStepAuditResponse(ResponseModel):
    phase: Literal["digest", "window", "merge"]
    artifact_id: str
    requested_model: str
    resolved_model: str | None = None
    usage: dict[str, int]
    output_mode: str | None = None
    schema_name: str
    schema_fallback_reason: str | None = None
    transport_request_count: int
    raw_output_digest: str | None = None
    response_diagnostics: list[InferenceResponseDiagnosticResponse] = Field(default_factory=list)
    reused: bool


class ReviewAttemptResponse(ResponseModel):
    position: int
    role: str
    trigger: str
    requested_model: str
    requested_family: str
    requested_backend: str
    status: Literal["succeeded", "abstained", "failed", "skipped"]
    skip_reason: Literal["insufficient_context_capacity"] | None = None
    resolved_model: str | None
    resolved_family: str | None
    score: float | None
    rationale: str | None
    evidence_ids: list[str]
    usage: dict[str, Any]
    output_mode: str | None
    schema_name: str | None
    schema_fallback_reason: str | None
    transport_request_count: int
    verdict_schema_version: int | None
    raw_output_digest: str | None
    error_type: str | None
    message: str | None
    behavioral_feedback: BehavioralFeedbackResponse | None
    steps: list[InferenceStepAuditResponse]


class JudgingAttemptSummaryResponse(ResponseModel):
    scope: Literal["session"]
    rubric: str
    review_status: Literal["complete", "degraded", "not_evaluable", "failed"]
    rating: float | None
    attempt_count: int
    successful_reviewer_count: int | None
    attempts: list[ReviewAttemptResponse]
    conversation_id: str


class JudgingFailureDetailResponse(ResponseModel):
    scope: Literal["session"]
    rubric: str
    error_type: str
    message: str | None
    attempt_count: int
    attempts: list[ReviewAttemptResponse]
    conversation_id: str


class JudgingActivityEventResponse(ResponseModel):
    id: int
    at: str
    phase: str
    message: str
    model: str | None = None
    conversation_id: str | None = None
    rubric: str | None = None
    artifact_id: str | None = None
    item_index: int | None = None
    item_total: int | None = None
    request_attempt: int | None = None
    max_attempts: int | None = None
    elapsed_seconds: float | None = None
    error_category: str | None = None
    retry_reason: str | None = None
    provider_status: int | None = None
    provider_error_code: str | None = None
    provider_error_message: str | None = None
    finish_reason: str | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    output_sha256: str | None = None
    exit_code: int | None = None
    stdout_chars: int | None = None
    stderr_chars: int | None = None
    prompt_characters: int | None = None
    output_mode: str | None = None
    estimated_input_tokens: int | None = None
    max_output_tokens: int | None = None
    model_context_tokens: int | None = None
    status: str | None = None


class JudgingProgressResponse(ResponseModel):
    plan_id: str
    planned_rubrics: int
    rubrics_completed: int
    rated_rubrics: int
    not_evaluable_rubrics: int
    minimum_reviewer_attempts: int
    maximum_reviewer_attempts: int
    reviewer_attempts_completed: int
    digest_steps_completed: int
    maximum_digest_steps: int
    window_steps_completed: int
    maximum_window_steps: int
    merge_steps_completed: int
    maximum_merge_steps: int
    scores_written: int
    failure_count: int
    write_failure_count: int
    coverage_complete: bool
    phase: str
    status_message: str
    started_at: str
    events: list[JudgingActivityEventResponse]
    attempt_summary_count: int
    attempt_summaries_truncated: bool
    attempt_summaries: list[JudgingAttemptSummaryResponse]
    failure_detail_count: int
    failure_details_truncated: bool
    failure_details: list[JudgingFailureDetailResponse]


class ReflectionActivityEventResponse(ResponseModel):
    id: int
    at: str
    phase: str
    message: str
    candidate: int | None = None
    model: str | None = None
    acting_role: str | None = None
    attempt_id: str | None = None
    evaluation_id: str | None = None
    score: float | None = None
    status: str | None = None
    error_type: str | None = None
    error: str | None = None
    changed_paths: list[str] | None = None
    response_digest: str | None = None
    response_excerpt: str | None = None


class ReflectingProgressResponse(ResponseModel):
    phase: str
    status_message: str
    started_at: str
    proposal_writer: str
    proposal_evaluator: str
    no_improvement_patience: int
    attempted: int
    valid: int
    rejected: int
    scored: int
    total_attempts: int
    events: list[ReflectionActivityEventResponse]


class ModelCatalogResponse(ResponseModel):
    catalog_version: str
    available_models: list[ModelDescriptor]
    judging_context: JudgingContextPolicy
    recommended_proposal_model: str | None
    recommended_judges: list[str]
    recommended_challenge_judges: list[str]
    proposal_evaluator_preferences: list[str]


class RunResponse(ResponseModel):
    run_id: str
    status: Literal[
        "created",
        "scoring",
        "judging",
        "reflecting",
        "complete",
        "failed",
        "cancelled",
    ]
    current_stage_succeeded: bool
    created_at: str
    auto_run: bool
    data_selection: DataSelectionResponse | None
    run_config: RunConfig | None
    effective_config: EffectiveRunConfig | None
    turn_cohort: dict[str, Any] | None
    judging_plan: JudgingPlanResponse | None
    scoring_progress: ScoringProgressResponse | None
    scoring_result: ScoringResultResponse | None
    judging_progress: JudgingProgressResponse | None
    judging_result: JudgingProgressResponse | None
    reflecting_progress: ReflectingProgressResponse | None
    reflecting_result: ReflectionResultResponse | None
    reflection_review: ReflectionReviewResponse | None
    reflection_review_revision: int
    error: str | None


class RunSelectionSummary(ResponseModel):
    session_count: int
    since: str | None
    until: str | None
    timezone: str | None


class RunSummaryResponse(ResponseModel):
    run_id: str
    status: Literal[
        "created",
        "scoring",
        "judging",
        "reflecting",
        "complete",
        "failed",
        "cancelled",
    ]
    current_stage_succeeded: bool
    created_at: str
    selection: RunSelectionSummary | None
    review_state: Literal[
        "none",
        "review-needed",
        "promoted",
        "partial",
        "dismissed",
        "no-change",
        "no-valid-proposal",
    ]


class RunListResponse(ResponseModel):
    runs: list[RunSummaryResponse]


class SignalEvidenceResponse(ResponseModel):
    signal: str
    version: str
    rating: float
    reason: str
    turn_id: str
    turn_started_at: str


class JudgingTokenEstimateResponse(ResponseModel):
    total_tokens: int
    largest_turn_tokens: int
    turn_tokens: list[int]


class SessionSummaryResponse(ResponseModel):
    conversation_id: str
    session_id: str | None
    turn_count: int
    started_at: str | None
    ended_at: str | None
    last_activity: str | None
    model: str | None
    effort_level: str | None
    config_version: str | None
    git_branch: str | None
    total_tokens: int
    total_tool_calls: int
    input_preview: str | None
    signal_evidence: list[SignalEvidenceResponse]


class SessionListResponse(ResponseModel):
    sessions: list[SessionSummaryResponse]
    total: int
    truncated: bool
    limit: int


class ToolCallResponse(ResponseModel):
    span_id: str
    tool_name: str
    arguments: str
    result: str
    status_code: str
    started_at: str | None
    ended_at: str | None


class ChatSpanResponse(ResponseModel):
    span_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    finish_reason: str | None


class SubagentResponse(ResponseModel):
    span_id: str
    agent_type: str | None
    tool_call_count: int


class FeedbackPayloadResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    rating: float
    tags: list[str] = []
    details: dict[str, Any] = {}
    reason: str | None = None


class FeedbackItemResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    feedback_type: str
    payload: FeedbackPayloadResponse
    created_at: datetime


class TurnDetailResponse(ResponseModel):
    trace_id: str
    conversation_id: str
    started_at: str
    ended_at: str | None
    model: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    status_code: str
    config_version: str | None
    git_branch: str | None
    effort_level: str | None
    session_id: str | None
    steering_count: int
    denial_count: int
    tool_error_count: int
    tool_call_count: int
    chat_span_count: int
    subagent_count: int
    user_input: str | None
    tool_calls: list[ToolCallResponse]
    chat_spans: list[ChatSpanResponse]
    subagents: list[SubagentResponse]


class SessionDetailResponse(ResponseModel):
    conversation_id: str | None
    config_version: str | None
    git_branch: str | None
    total_tokens: int
    turn_count: int
    judging_token_estimates: dict[TokenCounterName, JudgingTokenEstimateResponse]
    turns: list[TurnDetailResponse]
    session_feedback: list[FeedbackItemResponse]
    turn_feedback: dict[str, list[FeedbackItemResponse]]
    signal_evidence: list[SignalEvidenceResponse]


class ScoreSummaryResponse(ResponseModel):
    scorer: str
    count: int
    mean: float
    ci: tuple[float, float]
    binary: bool
    pass_rate: float | None
    tag_counts: dict[str, int]
    confident: bool


class MeanCountResponse(ResponseModel):
    mean: float
    count: int


class ABEntryResponse(ResponseModel):
    config_version: str
    evaluated_target_count: int
    scores: dict[str, MeanCountResponse]


class TrendEntryResponse(ResponseModel):
    scorer: str
    direction: Literal["regression", "improvement"]
    older_mean: float
    recent_mean: float
    delta: float
    sample_count: int
    significant: bool


class AnalysisResponse(ResponseModel):
    summary: list[ScoreSummaryResponse]
    ab_leaderboard: list[ABEntryResponse]
    trends: list[TrendEntryResponse]
    coaching_markdown: str
