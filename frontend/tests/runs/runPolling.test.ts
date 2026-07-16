import { describe, expect, it } from 'vitest'
import type { Run, RunSummary } from '../../src/types'
import { shouldPollRun, shouldPollRuns } from '../../src/features/runs/runPolling'

function run(status: Run['status']): Run {
  return {
    run_id: 'run-polling',
    status,
    current_stage_succeeded: false,
    created_at: '2026-07-14T18:00:00Z',
    auto_run: false,
    data_selection: null,
    run_config: null,
    effective_config: null,
    turn_cohort: null,
    judging_plan: null,
    reflection_input: null,
    scoring_progress: null,
    scoring_result: null,
    judging_progress: null,
    judging_result: null,
    reflecting_progress: null,
    reflecting_result: null,
    reflection_review: null,
    reflection_review_revision: 0,
    error: null,
  }
}

describe('run polling', () => {
  it('polls active work and auto-run handoffs', () => {
    const scoring = run('scoring')
    const autoHandoff = run('scoring')
    autoHandoff.auto_run = true
    autoHandoff.scoring_result = {
      turns_scored: 2,
      sessions_scored: 1,
      scores_written: 4,
      errors: 0,
      turn_details: [],
    }

    expect(shouldPollRun(scoring)).toBe(true)
    expect(shouldPollRun(run('reflecting'))).toBe(true)
    expect(shouldPollRun(autoHandoff)).toBe(true)
    expect(shouldPollRun({ ...run('created'), auto_run: true })).toBe(false)
  })

  it('stops during manual handoffs after scoring or judging completes', () => {
    const scoring = run('scoring')
    scoring.scoring_result = {
      turns_scored: 2,
      sessions_scored: 1,
      scores_written: 4,
      errors: 0,
      turn_details: [],
    }
    scoring.current_stage_succeeded = true
    const judging = run('judging')
    judging.judging_result = {
      plan_id: 'plan-1',
      planned_rubrics: 2,
      rubrics_completed: 2,
      rated_rubrics: 2,
      not_evaluable_rubrics: 0,
      minimum_reviewer_attempts: 4,
      maximum_reviewer_attempts: 4,
      reviewer_attempts_completed: 3,
      digest_steps_completed: 0,
      maximum_digest_steps: 0,
      window_steps_completed: 0,
      maximum_window_steps: 0,
      merge_steps_completed: 0,
      maximum_merge_steps: 0,
      scores_written: 2,
      failure_count: 0,
      write_failure_count: 0,
      coverage_complete: true,
      phase: 'judging_complete',
      status_message: 'Judging complete',
      started_at: '2026-07-15T07:30:00Z',
      events: [],
      attempt_summaries: [],
      attempt_summary_count: 0,
      attempt_summaries_truncated: false,
      failure_details: [],
      failure_detail_count: 0,
      failure_details_truncated: false,
    }
    judging.current_stage_succeeded = true

    expect(shouldPollRun(scoring)).toBe(false)
    expect(shouldPollRun(judging)).toBe(false)
  })

  it('keeps polling when a result is saved but the worker has not finalized success', () => {
    const scoring = run('scoring')
    scoring.scoring_result = {
      turns_scored: 2,
      sessions_scored: 1,
      scores_written: 4,
      errors: 0,
      turn_details: [],
    }

    expect(shouldPollRun(scoring)).toBe(true)
  })

  it('stops for terminal and idle created runs, including list polling', () => {
    const summary = (status: RunSummary['status'], succeeded = false): RunSummary => ({
      run_id: `summary-${status}`,
      status,
      current_stage_succeeded: succeeded,
      created_at: '2026-07-14T18:00:00Z',
      selection: null,
      review_state: 'none',
    })
    expect(shouldPollRuns([summary('created'), summary('complete', true)])).toBe(false)
    expect(shouldPollRuns([summary('created'), summary('judging')])).toBe(true)
    for (const status of ['complete', 'failed', 'cancelled'] as const) {
      expect(shouldPollRun(run(status))).toBe(false)
    }
  })
})
