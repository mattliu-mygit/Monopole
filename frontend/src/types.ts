interface TurnSummary {
  trace_id: string
  conversation_id: string
  started_at: string
  ended_at: string | null
  model: string | null
  input_tokens: number
  output_tokens: number
  cache_read_tokens: number
  status_code: string
  config_version: string | null
  git_branch: string | null
  effort_level: string | null
  session_id: string | null
  steering_count: number
  denial_count: number
  tool_error_count: number
  tool_call_count: number
  chat_span_count: number
  subagent_count: number
  user_input: string | null
}

interface ToolCall {
  span_id: string
  tool_name: string
  arguments: string
  result: string
  status_code: string
  started_at: string | null
  ended_at: string | null
}

interface ChatSpan {
  span_id: string
  model: string
  input_tokens: number
  output_tokens: number
  cache_read_tokens: number
  finish_reason: string | null
}

interface Subagent {
  span_id: string
  agent_type: string | null
  tool_call_count: number
}

export interface FeedbackItem {
  id: string
  feedback_type: string
  payload: {
    rating: number
    tags?: string[]
    details?: Record<string, any>
    reason?: string
  }
  created_at: string
}

export interface TurnDetail extends TurnSummary {
  tool_calls: ToolCall[]
  chat_spans: ChatSpan[]
  subagents: Subagent[]
}

export interface SignalEvidence {
  signal: string
  version: string
  rating: number
  reason: string
  turn_id: string
  turn_started_at: string
}

export interface SessionSummary {
  conversation_id: string
  session_id: string | null
  turn_count: number
  started_at: string | null
  ended_at: string | null
  last_activity: string | null
  model: string | null
  effort_level: string | null
  config_version: string | null
  git_branch: string | null
  total_tokens: number
  total_tool_calls: number
  input_preview: string | null
  signal_evidence: SignalEvidence[]
}

export interface SessionDetail {
  conversation_id: string | null
  config_version: string | null
  git_branch: string | null
  total_tokens: number
  turn_count: number
  turns: TurnDetail[]
  session_feedback: FeedbackItem[]
  turn_feedback: Record<string, FeedbackItem[]>
}

export interface ScoreSummary {
  scorer: string
  count: number
  mean: number
  ci: [number, number]
  binary: boolean
  pass_rate: number | null
  tag_counts: Record<string, number>
  confident: boolean
}

export interface ABEntry {
  config_version: string
  evaluated_target_count: number
  scores: Record<string, { mean: number; count: number }>
}

export interface TrendEntry {
  scorer: string
  direction: 'regression' | 'improvement'
  older_mean: number
  recent_mean: number
  delta: number
  sample_count: number
  significant: boolean
}

export interface AnalysisResponse {
  summary: ScoreSummary[]
  ab_leaderboard: ABEntry[]
  trends: TrendEntry[]
  coaching_markdown: string
}

export type ReflectionReviewStatus = 'pending' | 'promoted' | 'dismissed'

export interface ReflectionScope {
  kind: string
  target_id: string
  patterns: string[]
}

export interface ReflectionTargetSnapshot {
  kind: string
  locator: string
  display_name: string
  path: string | null
  exists: boolean
  content: string | null
  revision: string
}

export interface ReflectionBundleSnapshot {
  scope: ReflectionScope | null
  targets: ReflectionTargetSnapshot[]
  revision: string
}

export interface EvaluatorRecord {
  evaluation_id: string
  target_revision: string
  requested_model: string
  requested_family: string
  requested_backend: string
  resolved_model: string
  resolved_family: string
  resolved_backend: string
  score: number
  rationale: string
  usage: Record<string, number>
}

export interface GenerationAttempt {
  attempt_id: string
  number: number
  status: 'succeeded' | 'failed' | 'cancelled'
  requested_writer: ModelDescriptor
  resolved_model: string | null
  resolved_family: string | null
  resolved_backend: string | null
  usage: Record<string, number>
  candidate_revision: string | null
  changed_paths: string[]
  response_digest: string | null
  response_excerpt: string | null
  error_type: string | null
  error: string | null
}

export interface ReflectionCandidate {
  candidate_id: string
  bundle: ReflectionBundleSnapshot
  score: number
  score_delta: number
  rationale: string
  generation_attempt_id: string
  requested_writer: ModelDescriptor
  resolved_writer_model: string
  resolved_writer_family: string
  resolved_writer_backend: string
  evaluation: EvaluatorRecord
}

export interface SuccessfulReflectionResult {
  baseline: ReflectionBundleSnapshot
  baseline_score: number
  baseline_evaluation: EvaluatorRecord
  candidates: ReflectionCandidate[]
  generation_attempts: GenerationAttempt[]
  recommended_candidate_id: string | null
  baseline_won: boolean
  reason: string | null
  score_basis: 'predicted_evaluator'
}

export interface EmptyReflectionResult {
  candidates: []
  reason: string
}

export type ReflectionResult = SuccessfulReflectionResult | EmptyReflectionResult

export interface ReflectionDraft {
  candidate_id: string
  revision: string
  bundle: ReflectionBundleSnapshot
}

export interface PromotionReceiptAction {
  action: 'create' | 'update' | 'delete'
  locator: string
  before: ReflectionTargetSnapshot
  after: ReflectionTargetSnapshot
}

export interface PromotionReceipt {
  promotion_id: string
  run_id: string
  candidate_id: string
  target_kind: string
  target_id: string
  past: ReflectionBundleSnapshot
  evaluated_candidate: ReflectionBundleSnapshot
  promoted: ReflectionBundleSnapshot
  review_revision: number
  actions: PromotionReceiptAction[]
  decided_at: string
  promoted_was_evaluated: boolean
  unevaluated_d_acknowledged: boolean
  git_metadata: Record<string, unknown> | null
}

export interface ReflectionReviewState {
  status: ReflectionReviewStatus
  selected_candidate_id: string
  draft: ReflectionDraft | null
  stale?: boolean
  changed_targets?: string[]
  current?: ReflectionBundleSnapshot | null
  stale_reason?: string | null
  receipt?: PromotionReceipt | null
  dismissed_at?: string | null
}

export interface DataSelection {
  since: string | null
  until: string | null
  timezone?: string | null
  session_ids: string[]
}

export type ModelRole = 'proposal_writer' | 'judge' | 'proposal_evaluator'
export type ReviewDepth = 'primary' | 'selective' | 'full_panel'
export type EvaluationUnit = 'episode' | 'session'

export interface ModelDescriptor {
  id: string
  label: string
  family: string
  backend: string
  supported_roles: ModelRole[] | readonly ModelRole[]
}

export interface RubricDescriptor {
  id: string
  label: string
  evaluation_unit: EvaluationUnit
  version: string
  content_digest: string
  pass_threshold: number
}

export interface SelectionWarning {
  code: string
  message: string
  affected_roles: string[]
  selected_model_ids: string[]
  compared_families: string[]
}

export interface ProposalCatalog {
  available_models: ModelDescriptor[]
  recommended_model: string | null
}

export interface JudgeBackendCatalog {
  available_models: ModelDescriptor[]
  recommended_judges: string[]
  proposal_evaluator_preferences: string[]
  recommended_review_depth: ReviewDepth | null
  supported_review_depths: ReviewDepth[]
}

export interface ModelCatalog {
  catalog_version: string
  proposal: ProposalCatalog
  review_defaults: { second_opinion_margin: number }
  recommended_judge_backend: string
  judge_backends: Record<string, JudgeBackendCatalog>
}

export interface RubricCatalog {
  catalog_version: string
  rubrics: RubricDescriptor[]
}

export interface RunConfig {
  model_catalog_version: string
  rubric_catalog_version: string
  judge_backend: string
  review_depth: ReviewDepth
  judge_models: string[]
  second_opinion_margin: number | null
  proposal_model: string
  proposal_evaluator_model: string
  rubrics: string[]
  candidate_budget: number
  force: boolean
}

export interface PositionedJudge extends ModelDescriptor {
  role: 'judge'
  position: number
}

export interface EffectiveModelSelection {
  proposal_writer: ModelDescriptor
  judges: PositionedJudge[]
  proposal_evaluator: ModelDescriptor
}

export interface EffectiveRunConfig {
  schema_version: '1'
  pipeline_version: string
  model_catalog_version: string
  rubric_catalog_version: string
  judge_backend: string
  review_depth: ReviewDepth
  second_opinion_margin: number | null
  models: EffectiveModelSelection
  rubrics: RubricDescriptor[]
  selection_warnings: SelectionWarning[]
  candidate_budget: number
  force: boolean
}

export interface RunTurnContext {
  user_input?: string
  tokens?: number
  tool_count?: number
  tools_used?: string[]
  errors?: number
  steering?: number
  denials?: number
}

export interface ScoringTurnDetail extends RunTurnContext {
  trace_id: string
  conversation_id: string
  model: string | null
  scores: Record<string, number>
}

export interface ScoringProgress {
  total: number
  scored: number
  written: number
  status_message?: string
  turn_details?: ScoringTurnDetail[]
}

export interface ScoringResult {
  turns_scored: number
  sessions_scored: number
  scores_written: number
  errors: number
  turn_details?: ScoringTurnDetail[]
}

export interface JudgingPlanTotals {
  turns_considered: number
  episodes_selected: number
  planned_episode_rubrics: number
  planned_session_rubrics: number
  planned_rubrics: number
  minimum_episode_reviewer_attempts: number
  maximum_episode_reviewer_attempts: number
  minimum_session_reviewer_attempts: number
  maximum_session_reviewer_attempts: number
  minimum_reviewer_attempts: number
  maximum_reviewer_attempts: number
}

export interface JudgingPlanRubric extends RubricDescriptor {
  applicability: 'applicable' | 'not_applicable'
  minimum_reviewer_attempts: number
  maximum_reviewer_attempts: number
  skip_reason?: string | null
}

export interface JudgingPlanEpisode {
  trace_id: string
  turn_index: number
  selection_kind: string
  selection_reasons: string[]
  evidence_trace_ids: string[]
  rubrics: JudgingPlanRubric[]
}

export interface JudgingPlanSession {
  conversation_id: string
  turn_count: number
  omitted_turn_count: number
  session_rubrics: JudgingPlanRubric[]
  selected_episodes: JudgingPlanEpisode[]
}

export interface JudgingPlan {
  plan_id: string
  schema_version: string
  cohort_id: string
  requested_rubrics: RubricDescriptor[]
  review_depth: ReviewDepth
  judge_count: number
  max_episodes_per_session: number
  totals: JudgingPlanTotals
  sessions: JudgingPlanSession[]
}

export interface ReviewAttempt {
  position: number
  role: string
  trigger: string
  requested_model: string
  requested_family: string
  requested_backend: string
  status: 'succeeded' | 'abstained' | 'failed'
  resolved_model: string | null
  resolved_family: string | null
  score: number | null
  rationale: string | null
  evidence_ids: string[]
  usage: Record<string, number>
  output_mode: 'json_object' | 'json_schema' | 'json_object_fallback' | null
  schema_name: string | null
  schema_fallback_reason: string | null
  transport_request_count: number
  verdict_schema_version: number | null
  raw_output_digest: string | null
  error_type: string | null
  message: string | null
}

export interface JudgingAttemptSummary {
  scope: 'episode' | 'session'
  rubric: string
  review_status: 'complete' | 'degraded' | 'unresolved' | 'failed'
  rating: number | null
  attempt_count: number
  successful_reviewer_count: number | null
  attempts: ReviewAttempt[]
  trace_id?: string
  conversation_id: string
}

export interface JudgingFailureDetail {
  scope: 'episode' | 'session'
  rubric: string
  error_type: string
  message: string | null
  attempt_count: number
  attempts: ReviewAttempt[]
  trace_id?: string
  conversation_id: string
}

export interface JudgingProgress {
  plan_id: string
  planned_rubrics: number
  rubrics_completed: number
  rated_rubrics: number
  minimum_reviewer_attempts: number
  maximum_reviewer_attempts: number
  reviewer_attempts_completed: number
  scores_written: number
  failure_count: number
  write_failure_count: number
  coverage_complete: boolean
  status_message: string
  attempt_summaries: JudgingAttemptSummary[]
  failure_details: JudgingFailureDetail[]
  attempt_summary_count?: number
  attempt_summaries_truncated?: boolean
  failure_detail_count?: number
  failure_details_truncated?: boolean
}

export type JudgingResult = JudgingProgress

export interface ReflectionActivityEvent {
  id: number
  at: string
  phase: string
  message: string
  candidate?: number
  model?: string
  acting_role?: string
  attempt_id?: string
  evaluation_id?: string
  score?: number
  status?: string
  error_type?: string
  error?: string
  changed_paths?: string[]
  response_digest?: string
  response_excerpt?: string
}

export interface ReflectingProgress {
  phase: string
  status_message: string
  started_at: string
  attempted: number
  valid: number
  rejected: number
  scored: number
  total_attempts: number
  events: ReflectionActivityEvent[]
}

export interface Run {
  run_id: string
  status: 'created' | 'scoring' | 'judging' | 'reflecting' | 'complete' | 'failed' | 'cancelled'
  current_stage_succeeded: boolean
  created_at: string
  auto_run: boolean
  data_selection: DataSelection | null
  run_config: RunConfig | null
  effective_config: EffectiveRunConfig | null
  turn_cohort: Record<string, unknown> | null
  reflection_input: Record<string, unknown> | null
  scoring_progress: ScoringProgress | null
  scoring_result: ScoringResult | null
  judging_plan?: JudgingPlan | null
  judging_progress: JudgingProgress | null
  judging_result: JudgingResult | null
  reflecting_progress: ReflectingProgress | null
  reflecting_result: ReflectionResult | null
  reflection_review: ReflectionReviewState | null
  reflection_review_revision: number
  error: string | null
}

export type RunReviewState =
  | 'none'
  | 'review-needed'
  | 'promoted'
  | 'dismissed'
  | 'no-change'
  | 'no-valid-proposal'

export interface RunSummary {
  run_id: string
  status: Run['status']
  current_stage_succeeded: boolean
  created_at: string
  selection: {
    session_count: number
    since: string | null
    until: string | null
    timezone: string | null
  } | null
  review_state: RunReviewState
}
