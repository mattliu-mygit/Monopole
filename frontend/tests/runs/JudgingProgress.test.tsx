// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { JudgingPlan, JudgingProgress as Progress } from '../../src/types'
import JudgingProgress from '../../src/features/runs/JudgingProgress'
import persistedPlan from '../fixtures/judging-plan.json'

afterEach(cleanup)

const plan = persistedPlan as JudgingPlan
const rubric = plan.requested_rubrics[0]

const artifactIds = {
  digest: 'digest/3d4cad9a08c2864ea29e4681d40fcb70fa9e480b89fde612e7f12656c4603f3f/5ce4a7cd255dcf2c592dc9f13e32b5451be1fc6b0471531139802018fea6520b/402ac43615e38e86f8b55fc21f94dafa89f9250f2bd5d14066131efefa465d5c/5ea68ad0356c74ecde17f41dfbaedaaab6ae8e5b4d14e34edf379b84654424e3',
  window: 'window/3d4cad9a08c2864ea29e4681d40fcb70fa9e480b89fde612e7f12656c4603f3f/5ce4a7cd255dcf2c592dc9f13e32b5451be1fc6b0471531139802018fea6520b/402ac43615e38e86f8b55fc21f94dafa89f9250f2bd5d14066131efefa465d5c/c33641a1649d50f129bcd48bb0cdd542985616bf7a27250db7d3e2b538c925c2',
  merge: 'merge/3d4cad9a08c2864ea29e4681d40fcb70fa9e480b89fde612e7f12656c4603f3f/5ce4a7cd255dcf2c592dc9f13e32b5451be1fc6b0471531139802018fea6520b/402ac43615e38e86f8b55fc21f94dafa89f9250f2bd5d14066131efefa465d5c/3af044b7c3ac650ced5936bf3dcec316e7f620592eb7ab2c2046a206a6cfe1ae',
}

const progress: Progress = {
  plan_id: plan.plan_id,
  planned_rubrics: 1,
  rubrics_completed: 1,
  rated_rubrics: 1,
  minimum_reviewer_attempts: 1,
  maximum_reviewer_attempts: 2,
  reviewer_attempts_completed: 1,
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
      raw_output_digest: 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      error_type: null,
      message: null,
      behavioral_feedback: {
        success: 'Kept the investigation tied to evidence.',
        problem: 'The final verification was incomplete.',
        desired_behavior: 'Run the focused check before claiming completion.',
      },
      steps: [
        { phase: 'digest', artifact_id: artifactIds.digest, requested_model: 'judge-a', resolved_model: 'judge-a-resolved', usage: { input_tokens: 4 }, output_mode: 'json_schema', schema_name: 'chunk_digest', transport_request_count: 1, raw_output_digest: 'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', reused: true },
        { phase: 'window', artifact_id: artifactIds.window, requested_model: 'judge-a', resolved_model: 'judge-a-resolved', usage: { input_tokens: 3 }, output_mode: 'json_schema', schema_name: 'window_findings', transport_request_count: 1, raw_output_digest: 'sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc', reused: false },
        { phase: 'merge', artifact_id: artifactIds.merge, requested_model: 'judge-a', resolved_model: 'judge-a-resolved', usage: { input_tokens: 3 }, output_mode: 'json_object_fallback', schema_name: 'merged_verdict', schema_fallback_reason: 'Native schema unavailable', transport_request_count: 1, raw_output_digest: 'sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd', reused: false },
      ],
    }],
  }],
  failure_details: [],
}

describe('JudgingProgress', () => {
  it('shows the sliding plan and phase work bounds', () => {
    render(<JudgingProgress plan={plan} progress={progress} result={null} />)

    expect(screen.getByText('1 session · 2 turns · 2 raw windows')).not.toBeNull()
    expect(screen.getByText('1 of 2 digests')).not.toBeNull()
    expect(screen.getByText('1 of 2 windows')).not.toBeNull()
    expect(screen.getByText('1 of 2 merges')).not.toBeNull()
    expect(screen.getByText('Judge 1 · Judge A')).not.toBeNull()
    expect(screen.getAllByText(/core trace-1, trace-2/i).length).toBeGreaterThan(0)
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
