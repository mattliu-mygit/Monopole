// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { JudgingPlan, JudgingProgress as Progress } from '../../src/types'
import JudgingProgress from '../../src/features/runs/JudgingProgress'

afterEach(cleanup)

const plan: JudgingPlan = {
  plan_id: 'plan-1',
  schema_version: '1',
  cohort_id: 'cohort-1',
  requested_rubrics: [],
  review_depth: 'selective',
  judge_count: 3,
  max_episodes_per_session: 8,
  totals: {
    turns_considered: 9,
    episodes_selected: 3,
    planned_episode_rubrics: 4,
    planned_session_rubrics: 2,
    planned_rubrics: 6,
    minimum_episode_reviewer_attempts: 4,
    maximum_episode_reviewer_attempts: 12,
    minimum_session_reviewer_attempts: 2,
    maximum_session_reviewer_attempts: 6,
    minimum_reviewer_attempts: 6,
    maximum_reviewer_attempts: 18,
  },
  sessions: [],
}

const progress: Progress = {
  plan_id: 'plan-1',
  planned_rubrics: 6,
  rubrics_completed: 2,
  rated_rubrics: 2,
  minimum_reviewer_attempts: 6,
  maximum_reviewer_attempts: 18,
  reviewer_attempts_completed: 3,
  scores_written: 0,
  failure_count: 0,
  write_failure_count: 0,
  coverage_complete: false,
  status_message: 'Judged 2 of 6 planned rubrics',
  attempt_summaries: [
    {
      scope: 'episode',
      rubric: 'judge.verification',
      review_status: 'degraded',
      rating: 0.72,
      attempt_count: 3,
      successful_reviewer_count: 1,
      trace_id: 'trace-1',
      conversation_id: 'session-1',
      attempts: [
        {
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
          rationale: 'The run verified the changed behavior.',
          evidence_ids: ['trace-1'],
          usage: { input_tokens: 10 },
          output_mode: 'json_schema',
          schema_name: 'judge_verdict',
          schema_fallback_reason: null,
          transport_request_count: 1,
          verdict_schema_version: 2,
          raw_output_digest: 'a'.repeat(64),
          error_type: null,
          message: null,
        },
        {
          position: 2,
          role: 'judge',
          trigger: 'near_boundary',
          requested_model: 'judge-b',
          requested_family: 'family-b',
          requested_backend: 'cli',
          status: 'abstained',
          resolved_model: 'judge-b-resolved',
          resolved_family: 'family-b',
          score: null,
          rationale: 'The visible evidence is insufficient.',
          evidence_ids: [],
          usage: { input_tokens: 8 },
          output_mode: 'json_object_fallback',
          schema_name: 'judge_verdict',
          schema_fallback_reason: 'Native schema mode unavailable',
          transport_request_count: 2,
          verdict_schema_version: 2,
          raw_output_digest: 'b'.repeat(64),
          error_type: null,
          message: null,
        },
        {
          position: 3,
          role: 'judge',
          trigger: 'near_boundary',
          requested_model: 'judge-b',
          requested_family: 'family-b',
          requested_backend: 'cli',
          status: 'failed',
          resolved_model: null,
          resolved_family: null,
          score: null,
          rationale: null,
          evidence_ids: [],
          usage: {},
          output_mode: null,
          schema_name: 'judge_verdict',
          schema_fallback_reason: null,
          transport_request_count: 1,
          verdict_schema_version: null,
          raw_output_digest: null,
          error_type: 'RuntimeError',
          message: 'judge process exited',
        },
      ],
    },
  ],
  failure_details: [],
}

describe('JudgingProgress', () => {
  it('separates rubric completion from variable reviewer attempts', () => {
    render(<JudgingProgress plan={plan} progress={progress} result={null} />)

    expect(screen.getByRole('status').textContent).toContain('Judged 2 of 6 planned rubrics')
    expect(screen.getByRole('progressbar', { name: 'Rubrics reviewed' }).getAttribute('aria-valuenow')).toBe('2')
    expect(screen.getByText('2 of 6 rubrics reviewed')).not.toBeNull()
    expect(screen.getByText('3 reviewer attempts so far')).not.toBeNull()
    expect(screen.getByText('6 minimum · 18 maximum')).not.toBeNull()
    expect(screen.getByText('3 selected episodes from 9 turns')).not.toBeNull()
  })

  it('keeps model rationale and errors visible in the audit', () => {
    render(<JudgingProgress plan={plan} progress={progress} result={null} />)

    expect(screen.getByText('judge.verification')).not.toBeNull()
    expect(screen.getByText('The run verified the changed behavior.')).not.toBeNull()
    expect(screen.getByText('RuntimeError: judge process exited')).not.toBeNull()
    expect(screen.getAllByText(/near boundary/i).length).toBe(2)
    expect(screen.getByText(/abstained/i)).not.toBeNull()
    expect(screen.getByText('The visible evidence is insufficient.')).not.toBeNull()
    const audits = screen.getAllByText('Transport and schema audit')
    expect(audits.length).toBe(3)
    expect(screen.getByText('json object fallback')).not.toBeNull()
    expect(screen.getByText('Native schema mode unavailable')).not.toBeNull()
    expect(screen.getByText('trace-1')).not.toBeNull()
  })

  it('renders persisted failure details and truncation notices', () => {
    const failed: Progress = {
      ...progress,
      coverage_complete: false,
      failure_count: 2,
      status_message: 'Judging coverage incomplete',
      failure_details: [{
        scope: 'session',
        rubric: 'judge.session_autonomy',
        error_type: 'JudgeExecutionError',
        message: 'Every reviewer failed',
        attempt_count: 3,
        attempts: progress.attempt_summaries[0].attempts,
        conversation_id: 'session-2',
      }],
      attempt_summary_count: 7,
      attempt_summaries_truncated: true,
      failure_detail_count: 2,
      failure_details_truncated: true,
    }

    render(<JudgingProgress plan={plan} progress={null} result={failed} />)

    expect(screen.getByText('Judging coverage incomplete')).not.toBeNull()
    expect(screen.getByText('judge.session_autonomy')).not.toBeNull()
    expect(screen.getByText('JudgeExecutionError: Every reviewer failed')).not.toBeNull()
    expect(screen.getByText(/showing 1 of 7 review records/i)).not.toBeNull()
    expect(screen.getByText(/showing 1 of 2 failures/i)).not.toBeNull()
    expect(screen.getByText('Full failed-attempt audit')).not.toBeNull()
    expect(screen.getAllByText('The visible evidence is insufficient.').length).toBeGreaterThan(0)
  })

  it('shows rubrics skipped because captured evidence was too bare', () => {
    const planWithSkip: JudgingPlan = {
      ...plan,
      sessions: [{
        conversation_id: 'session-1',
        turn_count: 1,
        omitted_turn_count: 0,
        session_rubrics: [],
        selected_episodes: [{
          trace_id: 'trace-bare',
          turn_index: 0,
          selection_kind: 'deterministic_trigger',
          selection_reasons: ['verification'],
          evidence_trace_ids: ['trace-bare'],
          rubrics: [{
            id: 'judge.verification',
            label: 'Verification Discipline',
            evaluation_unit: 'episode',
            version: '1',
            content_digest: 'sha256:verification',
            pass_threshold: 0.5,
            applicability: 'not_applicable',
            minimum_reviewer_attempts: 0,
            maximum_reviewer_attempts: 0,
            skip_reason: 'Tool-execution turn: 2 tool calls were captured (Edit, Bash), but no assistant message was captured (assistant_output was null). The verification rubric requires an assistant completion or correctness claim to compare with the verification activity, so this rubric was skipped.',
          }],
        }],
      }],
    }

    render(<JudgingProgress plan={planWithSkip} progress={null} result={null} />)

    expect(screen.getByText('Skipped rubric checks')).not.toBeNull()
    expect(screen.getByText('judge.verification')).not.toBeNull()
    expect(screen.getByText(/tool-execution turn: 2 tool calls were captured \(edit, bash\)/i)).not.toBeNull()
    expect(screen.getByText(/trace trace-bare/i)).not.toBeNull()
  })

  it('shows an honest empty state before a plan is available', () => {
    render(<JudgingProgress plan={null} progress={null} result={null} />)
    expect(screen.getByRole('status').textContent).toBe('Waiting for judging to start.')
  })
})
