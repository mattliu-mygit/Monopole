// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { JudgingPlan, JudgingProgress as Progress } from '../../src/types'
import JudgingProgress from '../../src/features/runs/JudgingProgress'

afterEach(cleanup)

const rubric = {
  id: 'judge.session_outcome',
  label: 'Session Outcome',
  evaluation_unit: 'session' as const,
  version: '4',
  content_digest: 'sha256:rubric',
  pass_threshold: 0.7,
}

const judge = {
  id: 'judge-a',
  label: 'Judge A',
  family: 'family-a',
  backend: 'cli',
  supported_roles: ['judge'] as const,
  max_input_tokens: 128_000,
  role: 'judge' as const,
  position: 1,
}

const secondJudge = {
  ...judge,
  id: 'judge-b',
  label: 'Judge B',
  family: 'family-b',
  position: 2,
}

const windowPlan = {
  plan_id: 'sha256:window-plan',
  contract_version: '1' as const,
  conversation_id: 'session-1',
  input_cap_tokens: 100_000,
  raw_budget_tokens: 80_000,
  chunk_count: 2,
  overlap_turns: 1 as const,
  token_estimator: 'utf8_bytes_div_3' as const,
  merge_input_tokens: 21_500,
  raw_turns: [
    { trace_id: 'trace-1', position: 1, estimated_tokens: 1200, raw_digest: 'sha256:raw-1' },
    { trace_id: 'trace-2', position: 2, estimated_tokens: 900, raw_digest: 'sha256:raw-2' },
  ],
  raw_coverage_trace_ids: ['trace-1', 'trace-2'],
  windows: [
    { window_id: 'sha256:window-1', index: 1, core_trace_ids: ['trace-1'], raw_trace_ids: ['trace-1', 'trace-2'], raw_turn_digests: ['sha256:raw-1', 'sha256:raw-2'], raw_tokens: 2100 },
    { window_id: 'sha256:window-2', index: 2, core_trace_ids: ['trace-2'], raw_trace_ids: ['trace-1', 'trace-2'], raw_turn_digests: ['sha256:raw-1', 'sha256:raw-2'], raw_tokens: 2100 },
  ],
}

const plan: JudgingPlan = {
  plan_id: 'sha256:plan',
  schema_version: '2',
  cohort_id: 'cohort-1',
  requested_rubrics: [rubric],
  review_depth: 'selective',
  second_opinion_margin: 0.1,
  input_policy: {
    contract_version: '1',
    target_input_tokens: 100_000,
    prompt_reserve_tokens: 6_000,
    output_reserve_tokens: 4_000,
    safety_reserve_tokens: 8_000,
    digest_max_tokens: 1_000,
    finding_max_tokens: 750,
    overlap_turns: 1,
    max_chunks: 40,
    token_estimator: 'utf8_bytes_div_3',
  },
  protocol: {
    protocol_version: '2',
    prompt_templates: {
      digest_system: 'digest system', digest_user: 'digest user',
      window_system: 'window system', window_user: 'window user',
      merge_system: 'merge system', merge_user: 'merge user',
    },
    schemas: {
      digest: { name: 'chunk_digest', schema: {} },
      window: { name: 'window_findings', schema: {} },
      merge: { name: 'merged_verdict', schema: {} },
    },
  },
  totals: {
    sessions_planned: 1,
    turns_considered: 2,
    windows_planned: 4,
    planned_rubrics: 1,
    minimum_reviewer_attempts: 1,
    maximum_reviewer_attempts: 2,
    maximum_digest_calls: 4,
    maximum_window_calls: 4,
    maximum_merge_calls: 2,
  },
  sessions: [{
    conversation_id: 'session-1',
    turn_count: 2,
    raw_coverage_trace_ids: ['trace-1', 'trace-2'],
    rubrics: [{ ...rubric, minimum_reviewer_attempts: 1, maximum_reviewer_attempts: 2 }],
    reviewers: [{
      ordinal: 1,
      judge,
      work_bounds: {
        digest_calls: 2,
        window_calls_per_rubric: 2,
        merge_calls_per_rubric: 1,
      },
      window_plan: windowPlan,
    }, {
      ordinal: 2,
      judge: secondJudge,
      work_bounds: {
        digest_calls: 2,
        window_calls_per_rubric: 2,
        merge_calls_per_rubric: 1,
      },
      window_plan: windowPlan,
    }],
  }],
}

const progress: Progress = {
  plan_id: 'sha256:plan',
  planned_rubrics: 1,
  rubrics_completed: 1,
  rated_rubrics: 1,
  minimum_reviewer_attempts: 1,
  maximum_reviewer_attempts: 2,
  reviewer_attempts_completed: 1,
  digest_steps_completed: 2,
  maximum_digest_steps: 4,
  window_steps_completed: 2,
  maximum_window_steps: 4,
  merge_steps_completed: 1,
  maximum_merge_steps: 2,
  scores_written: 0,
  failure_count: 0,
  write_failure_count: 0,
  coverage_complete: false,
  status_message: 'Judged 1 of 1 planned rubrics',
  attempt_summaries: [{
    scope: 'session',
    rubric: rubric.id,
    review_status: 'complete',
    rating: 0.72,
    attempt_count: 1,
    successful_reviewer_count: 1,
    conversation_id: 'session-1',
    attempts: [{
      position: 1,
      role: 'judge',
      trigger: 'initial',
      requested_model: 'judge-a',
      requested_family: 'family-a',
      requested_backend: 'cli',
      status: 'succeeded',
      resolved_model: 'judge-a-resolved',
      resolved_family: 'family-a',
      score: 0.72,
      rationale: 'The session reached the requested result.',
      evidence_ids: ['trace-1'],
      usage: { input_tokens: 10 },
      output_mode: 'json_schema',
      schema_name: 'merged_verdict',
      schema_fallback_reason: null,
      transport_request_count: 3,
      verdict_schema_version: 1,
      raw_output_digest: 'sha256:output',
      error_type: null,
      message: null,
      behavioral_feedback: {
        success: 'Kept the investigation tied to evidence.',
        problem: 'The final verification was incomplete.',
        desired_behavior: 'Run the focused check before claiming completion.',
      },
      steps: [
        { phase: 'digest', artifact_id: 'digest/1', requested_model: 'judge-a', resolved_model: 'judge-a-resolved', usage: { input_tokens: 4 }, output_mode: 'json_schema', schema_name: 'chunk_digest', transport_request_count: 1, raw_output_digest: 'sha256:digest', reused: true },
        { phase: 'window', artifact_id: 'window/1', requested_model: 'judge-a', resolved_model: 'judge-a-resolved', usage: { input_tokens: 3 }, output_mode: 'json_schema', schema_name: 'window_findings', transport_request_count: 1, raw_output_digest: 'sha256:window', reused: false },
        { phase: 'merge', artifact_id: 'merge/1', requested_model: 'judge-a', resolved_model: 'judge-a-resolved', usage: { input_tokens: 3 }, output_mode: 'json_object_fallback', schema_name: 'merged_verdict', schema_fallback_reason: 'Native schema unavailable', transport_request_count: 1, raw_output_digest: 'sha256:merge', reused: false },
      ],
    }],
  }],
  failure_details: [],
}

describe('JudgingProgress', () => {
  it('shows the sliding plan and phase work bounds', () => {
    render(<JudgingProgress plan={plan} progress={progress} result={null} />)

    expect(screen.getByText('1 session · 2 turns · 4 raw windows')).not.toBeNull()
    expect(screen.getByText('2 of 4 digests')).not.toBeNull()
    expect(screen.getByText('2 of 4 windows')).not.toBeNull()
    expect(screen.getByText('1 of 2 merges')).not.toBeNull()
    expect(screen.getByText('Judge 1 · Judge A')).not.toBeNull()
    expect(screen.getAllByText(/core trace-1/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/raw trace-1, trace-2/i).length).toBeGreaterThan(0)
  })

  it('shows final feedback and ordered step audits with reuse state', () => {
    render(<JudgingProgress plan={plan} progress={progress} result={null} />)

    expect(screen.getByText('The session reached the requested result.')).not.toBeNull()
    expect(screen.getByText('Kept the investigation tied to evidence.')).not.toBeNull()
    expect(screen.getByText('The final verification was incomplete.')).not.toBeNull()
    expect(screen.getByText('Run the focused check before claiming completion.')).not.toBeNull()
    expect(screen.getByText('digest · reused')).not.toBeNull()
    expect(screen.getByText('window · current')).not.toBeNull()
    expect(screen.getByText('merge · current')).not.toBeNull()
    expect(screen.getByText('Native schema unavailable')).not.toBeNull()
  })

  it('renders only the non-null categories in partial behavioral feedback', () => {
    const partial = structuredClone(progress)
    partial.attempt_summaries[0].attempts[0].behavioral_feedback = {
      success: null,
      problem: 'The final verification was incomplete.',
      desired_behavior: null,
    }

    render(<JudgingProgress plan={plan} progress={partial} result={null} />)

    expect(screen.queryByText('Success:')).toBeNull()
    expect(screen.getByText('Problem:')).not.toBeNull()
    expect(screen.queryByText('Desired behavior:')).toBeNull()
  })

  it('preserves failure details and truncation notices', () => {
    const failed: Progress = {
      ...progress,
      failure_count: 2,
      status_message: 'Judging coverage incomplete',
      failure_details: [{
        scope: 'session',
        rubric: 'judge.session_autonomy',
        error_type: 'JudgeExecutionError',
        message: 'Every reviewer failed',
        attempt_count: 1,
        attempts: progress.attempt_summaries[0].attempts,
        conversation_id: 'session-2',
      }],
      attempt_summary_count: 7,
      attempt_summaries_truncated: true,
      failure_detail_count: 2,
      failure_details_truncated: true,
    }

    render(<JudgingProgress plan={plan} progress={null} result={failed} />)
    expect(screen.getByText('JudgeExecutionError: Every reviewer failed')).not.toBeNull()
    expect(screen.getByText(/showing 1 of 7 review records/i)).not.toBeNull()
    expect(screen.getByText(/showing 1 of 2 failures/i)).not.toBeNull()
    expect(screen.getByText('Full failed-attempt audit')).not.toBeNull()
  })

  it('shows an honest empty state before a plan is available', () => {
    render(<JudgingProgress plan={null} progress={null} result={null} />)
    expect(screen.getByRole('status').textContent).toBe('Waiting for judging to start.')
  })
})
