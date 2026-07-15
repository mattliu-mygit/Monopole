import type { Run, RunSummary } from '../../types'

const TERMINAL_STATUSES = new Set<Run['status']>(['complete', 'failed', 'cancelled'])

export function shouldPollRun(run: Run): boolean {
  if (TERMINAL_STATUSES.has(run.status)) return false
  if (run.status === 'reflecting') return true
  if (run.status === 'scoring' || run.status === 'judging') {
    return !run.current_stage_succeeded
  }
  return false
}

export function shouldPollRuns(runs: RunSummary[]): boolean {
  return runs.some((run) => {
    if (TERMINAL_STATUSES.has(run.status)) return false
    if (run.status === 'reflecting') return true
    if (run.status === 'scoring' || run.status === 'judging') {
      return !run.current_stage_succeeded
    }
    return false
  })
}
