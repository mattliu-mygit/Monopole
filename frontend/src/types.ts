export interface TurnSummary {
  trace_id: string
  conversation_id: string
  started_at: string
  ended_at: string | null
  model: string
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

export interface ToolCall {
  span_id: string
  tool_name: string
  arguments: string
  result: string
  status_code: string
  started_at: string | null
  ended_at: string | null
}

export interface ChatSpan {
  span_id: string
  model: string
  input_tokens: number
  output_tokens: number
  cache_read_tokens: number
  finish_reason: string | null
}

export interface Subagent {
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
  feedback: FeedbackItem[]
}

export interface SessionSummary {
  conversation_id: string
  session_id: string | null
  turn_count: number
  started_at: string | null
  ended_at: string | null
  model: string | null
  effort_level: string | null
  config_version: string | null
  git_branch: string | null
  total_tokens: number
  total_tool_calls: number
  input_preview: string | null
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
  turn_count: number
  scores: Record<string, { mean: number; count: number }>
}

export interface TrendEntry {
  scorer: string
  direction: string
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

export interface Job {
  job_id: string
  type: string
  status: string
  started_at: string
  progress: Record<string, any>
  result: any
  error: string | null
}

export interface Artifact {
  name: string
  path: string
  content: string
}

export interface Rubric {
  name: string
  scorer_name: string
  description: string
  criteria: Record<string, string>
  granularity: string
}

export interface DataSelection {
  since: string | null
  until: string | null
  session_ids: string[]
  excluded_session_ids: string[]
}

export interface Run {
  run_id: string
  status: 'created' | 'scoring' | 'judging' | 'reflecting' | 'complete' | 'failed'
  created_at: string
  config_version: string | null
  auto_run: boolean
  data_selection: DataSelection | null
  scoring_progress: Record<string, unknown> | null
  scoring_result: Record<string, unknown> | null
  judging_progress: Record<string, unknown> | null
  judging_result: Record<string, unknown> | null
  reflecting_progress: Record<string, unknown> | null
  reflecting_result: Record<string, unknown> | null
  error: string | null
}
