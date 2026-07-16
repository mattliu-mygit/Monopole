// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type {
  JudgingPlan,
  JudgingProgress as Progress,
  ReviewAttempt,
} from '../../src/types'
import JudgingProgress from '../../src/features/runs/JudgingProgress'
import persistedPlan from '../fixtures/judging-plan.json'

afterEach(cleanup)

const plan = persistedPlan as unknown as JudgingPlan
const artifactIds = {
  digest: 'digest/3d4cad9a08c2864ea29e4681d40fcb70fa9e480b89fde612e7f12656c4603f3f/5ce4a7cd255dcf2c592dc9f13e32b5451be1fc6b0471531139802018fea6520b/402ac43615e38e86f8b55fc21f94dafa89f9250f2bd5d14066131efefa465d5c/5ea68ad0356c74ecde17f41dfbaedaaab6ae8e5b4d14e34edf379b84654424e3',
  window: 'window/3d4cad9a08c2864ea29e4681d40fcb70fa9e480b89fde612e7f12656c4603f3f/5ce4a7cd255dcf2c592dc9f13e32b5451be1fc6b0471531139802018fea6520b/402ac43615e38e86f8b55fc21f94dafa89f9250f2bd5d14066131efefa465d5c/c33641a1649d50f129bcd48bb0cdd542985616bf7a27250db7d3e2b538c925c2',
  merge: 'merge/3d4cad9a08c2864ea29e4681d40fcb70fa9e480b89fde612e7f12656c4603f3f/5ce4a7cd255dcf2c592dc9f13e32b5451be1fc6b0471531139802018fea6520b/402ac43615e38e86f8b55fc21f94dafa89f9250f2bd5d14066131efefa465d5c/3af044b7c3ac650ced5936bf3dcec316e7f620592eb7ab2c2046a206a6cfe1ae',
}

const attempt: ReviewAttempt = {
  position: 1,
  role: 'judge' as const,
  trigger: 'panel' as const,
  requested_model: 'judge-a',
  requested_family: 'family-a',
  requested_backend: 'cli',
  status: 'succeeded' as const,
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
  raw_output_digest: 'a'.repeat(64),
  error_type: null,
  message: null,
  behavioral_feedback: {
    success: 'Kept the investigation tied to evidence.',
    problem: 'The final verification was incomplete.',
    desired_behavior: 'Run the focused check before claiming completion.',
  },
  steps: [
    { phase: 'digest', artifact_id: artifactIds.digest, requested_model: 'judge-a', resolved_model: 'judge-a-resolved', usage: { input_tokens: 4 }, output_mode: 'json_schema', schema_name: 'chunk_digest', transport_request_count: 1, raw_output_digest: 'b'.repeat(64), reused: true },
    { phase: 'window', artifact_id: artifactIds.window, requested_model: 'judge-a', resolved_model: 'judge-a-resolved', usage: { input_tokens: 3 }, output_mode: 'json_schema', schema_name: 'window_findings', transport_request_count: 1, raw_output_digest: 'c'.repeat(64), reused: false },
    { phase: 'merge', artifact_id: artifactIds.merge, requested_model: 'judge-a', resolved_model: 'judge-a-resolved', usage: { input_tokens: 3 }, output_mode: 'json_object_fallback', schema_name: 'merged_verdict', schema_fallback_reason: 'Native schema unavailable', transport_request_count: 1, raw_output_digest: 'd'.repeat(64), reused: false },
  ],
}

const secondAttempt: ReviewAttempt = {
  ...attempt,
  position: 2,
  requested_model: 'judge-b',
  resolved_model: 'judge-b-resolved',
}

const progress: Progress = {
  plan_id: plan.plan_id,
  planned_rubrics: 1,
  rubrics_completed: 1,
  rated_rubrics: 1,
  not_evaluable_rubrics: 0,
  minimum_reviewer_attempts: 2,
  maximum_reviewer_attempts: 2,
  reviewer_attempts_completed: 2,
  digest_steps_completed: 1,
  maximum_digest_steps: 2,
  window_steps_completed: 1,
  maximum_window_steps: 2,
  merge_steps_completed: 1,
  maximum_merge_steps: 2,
  scores_written: 0,
  failure_count: 0,
  write_failure_count: 0,
  coverage_complete: false,
  phase: 'merge_started',
  status_message: 'Judge A is merging Session Outcome Quality',
  started_at: '2026-07-16T07:00:00+00:00',
  events: [
    { id: 1, at: '2026-07-16T07:00:00+00:00', phase: 'judging_started', message: 'Starting 1 planned rubric judgment' },
    { id: 2, at: '2026-07-16T07:00:01+00:00', phase: 'session_started', message: 'Reviewing session session-1', conversation_id: 'session-1' },
    { id: 3, at: '2026-07-16T07:00:02+00:00', phase: 'digest_started', message: 'Judge A is digesting chunk 1 of 2', model: 'judge-a', item_index: 1, item_total: 2 },
    { id: 4, at: '2026-07-16T07:03:14+00:00', phase: 'transport_retry', message: 'Judge A request failed after 191.8s; retrying attempt 2 of 3', model: 'judge-a', request_attempt: 2, max_attempts: 3, elapsed_seconds: 191.8, error_category: 'retryable_process_error', provider_status: 429, provider_error_code: 'rate_limit_exceeded', provider_error_message: 'Too many requests for this model.', output_sha256: 'e'.repeat(64) },
    { id: 5, at: '2026-07-16T07:03:25+00:00', phase: 'transport_recovered', message: 'Judge A recovered on request attempt 2 of 3', model: 'judge-a', request_attempt: 2, max_attempts: 3, elapsed_seconds: 202, output_sha256: 'f'.repeat(64) },
    { id: 6, at: '2026-07-16T07:03:26+00:00', phase: 'window_started', message: 'Judge A is reviewing window 1 of 2', model: 'judge-a', rubric: 'judge.session_outcome', item_index: 1, item_total: 2 },
    { id: 7, at: '2026-07-16T07:04:00+00:00', phase: 'merge_started', message: 'Judge A is merging Session Outcome Quality', model: 'judge-a', rubric: 'judge.session_outcome' },
  ],
  attempt_summaries: [{
    scope: 'session',
    rubric: plan.requested_rubrics[0].id,
    review_status: 'complete',
    rating: 0.72,
    attempt_count: 2,
    successful_reviewer_count: 2,
    conversation_id: 'session-1',
    attempts: [attempt, secondAttempt],
  }],
  attempt_summary_count: 1,
  attempt_summaries_truncated: false,
  failure_details: [],
  failure_detail_count: 0,
  failure_details_truncated: false,
}

describe('JudgingProgress', () => {
  it('shows current narration and a compact expandable activity timeline', () => {
    render(<JudgingProgress plan={plan} progress={progress} result={null} />)

    expect(screen.getAllByText('Judge A is merging Session Outcome Quality').length).toBeGreaterThan(0)
    expect(screen.getByLabelText('Judging activity')).not.toBeNull()
    expect(screen.getByText('Judge A request failed after 191.8s; retrying attempt 2 of 3')).not.toBeNull()
    expect(screen.getByText('Judge A recovered on request attempt 2 of 3')).not.toBeNull()
    expect(screen.getAllByText('attempt 2 of 3').length).toBeGreaterThan(0)
    expect(screen.getByText('191.8s')).not.toBeNull()
    expect(screen.getByText('retryable process error')).not.toBeNull()
    expect(screen.getByText('provider 429')).not.toBeNull()
    expect(screen.getByText('rate limit exceeded')).not.toBeNull()
    expect(screen.getByText('Too many requests for this model.')).not.toBeNull()
    expect(screen.queryByText('Starting 1 planned rubric judgment')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Show all activity' }))
    expect(screen.getByText('Starting 1 planned rubric judgment')).not.toBeNull()
  })

  it('shows sliding-window work instead of episode selection', () => {
    render(<JudgingProgress plan={plan} progress={progress} result={null} />)

    expect(screen.getByText('1 session · 2 turns · 2 raw windows')).not.toBeNull()
    expect(screen.getByText('1 of 2 digests')).not.toBeNull()
    expect(screen.getByText('1 of 2 windows')).not.toBeNull()
    expect(screen.getByText('1 of 2 merges')).not.toBeNull()
    expect(screen.getByText('Judge 1 · Judge A')).not.toBeNull()
    expect(screen.queryByText(/episode/i)).toBeNull()
  })

  it('shows capacity-skipped reviewers without trying to render windows', () => {
    const skippedPlan: JudgingPlan = {
      ...plan,
      sessions: [{
        ...plan.sessions[0],
        reviewers: [{
          ...plan.sessions[0].reviewers[0],
          status: 'skipped',
          skip_reason: 'insufficient_context_capacity',
          window_plan: null,
          work_bounds: {
            digest_calls: 0,
            window_calls_per_rubric: 0,
            merge_calls_per_rubric: 0,
          },
        }],
      }],
    }

    render(<JudgingProgress plan={skippedPlan} progress={null} result={null} />)

    expect(screen.getByText('Skipped · insufficient context capacity')).not.toBeNull()
    expect(screen.queryByText(/Window 1/)).toBeNull()
  })

  it('shows final behavioral feedback and the ordered inference audit', () => {
    render(<JudgingProgress plan={plan} progress={progress} result={null} />)

    expect(screen.getAllByText('The session reached the requested result.')).toHaveLength(2)
    expect(screen.getAllByText('Kept the investigation tied to evidence.')).toHaveLength(2)
    expect(screen.getAllByText('The final verification was incomplete.')).toHaveLength(2)
    expect(screen.getAllByText('Run the focused check before claiming completion.')).toHaveLength(2)
    expect(screen.getAllByText('digest · reused')).toHaveLength(2)
    expect(screen.getAllByText('window · current')).toHaveLength(2)
    expect(screen.getAllByText('merge · current')).toHaveLength(2)
  })

  it('shows skipped attempt status and reason without success styling', () => {
    const skippedAttempt: ReviewAttempt = {
      ...attempt,
      status: 'skipped',
      skip_reason: 'insufficient_context_capacity',
      resolved_model: null,
      resolved_family: null,
      score: null,
      rationale: null,
      evidence_ids: [],
      usage: {},
      output_mode: null,
      schema_name: null,
      schema_fallback_reason: null,
      transport_request_count: 0,
      verdict_schema_version: null,
      raw_output_digest: null,
      behavioral_feedback: null,
      steps: [],
    }
    const skippedProgress: Progress = {
      ...progress,
      attempt_summaries: [{
        ...progress.attempt_summaries[0],
        review_status: 'not_evaluable',
        rating: null,
        successful_reviewer_count: 0,
        attempts: [skippedAttempt],
      }],
    }

    render(<JudgingProgress plan={plan} progress={skippedProgress} result={null} />)

    const status = screen.getByText('panel · skipped')
    expect(status.className).toContain('text-amber-700')
    expect(status.className).not.toContain('text-green-700')
    expect(screen.getByText('Skipped: insufficient context capacity')).not.toBeNull()
  })

  it('keeps failure details and the pre-plan empty state visible', () => {
    const failed: Progress = {
      ...progress,
      failure_count: 1,
      failure_details: [{
        scope: 'session',
        rubric: 'judge.session_autonomy',
        error_type: 'JudgeExecutionError',
        message: 'Every reviewer failed',
        attempt_count: 1,
        attempts: [attempt],
        conversation_id: 'session-2',
      }],
    }
    const { rerender } = render(<JudgingProgress plan={plan} progress={null} result={failed} />)
    expect(screen.getByText('JudgeExecutionError: Every reviewer failed')).not.toBeNull()

    rerender(<JudgingProgress plan={null} progress={null} result={null} />)
    expect(screen.getByRole('status').textContent).toBe('Waiting for judging to start.')
  })
})
