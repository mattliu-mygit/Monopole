// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { ScoringProgress as Progress, ScoringResult } from '../../src/types'
import ScoringProgress from '../../src/features/runs/ScoringProgress'

afterEach(cleanup)

const turn = {
  trace_id: 'trace-123',
  conversation_id: 'session-456',
  model: 'gpt-5',
  user_input: 'Check the patch',
  tokens: 1200,
  tool_count: 3,
  errors: 1,
  scores: {
    'outcome.test': 1,
    'process.error_recovery': 0.75,
  },
}

describe('ScoringProgress', () => {
  it('renders persisted active progress and scored-turn evidence', () => {
    const progress: Progress = {
      total: 4,
      scored: 2,
      written: 6,
      status_message: 'Scoring turn 3 of 4',
      turn_details: [turn],
    }

    render(<ScoringProgress progress={progress} result={null} />)

    const progressbar = screen.getByRole('progressbar', { name: 'Turns scored' })
    expect(progressbar.getAttribute('aria-valuenow')).toBe('2')
    expect(progressbar.getAttribute('aria-valuemax')).toBe('4')
    expect(screen.getByRole('status').textContent).toContain('Scoring turn 3 of 4')
    expect(screen.getByText('2 of 4 turns scored · 6 scores written')).not.toBeNull()
    expect(screen.getByText('trace-123')).not.toBeNull()
    expect(screen.getByText('outcome.test')).not.toBeNull()
    expect(screen.getByText('1.00')).not.toBeNull()
  })

  it('renders the final persisted result', () => {
    const result: ScoringResult = {
      turns_scored: 4,
      sessions_scored: 2,
      scores_written: 12,
      errors: 1,
      turn_details: [turn],
    }

    render(<ScoringProgress progress={null} result={result} />)

    expect(screen.getByText('4 turns scored')).not.toBeNull()
    expect(screen.getByText('2 sessions scored')).not.toBeNull()
    expect(screen.getByText('12 scores written')).not.toBeNull()
    expect(screen.getByText('1 error', { selector: 'div' })).not.toBeNull()
  })

  it('shows an honest empty state before progress is persisted', () => {
    render(<ScoringProgress progress={null} result={null} />)

    expect(screen.getByRole('status').textContent).toBe('Waiting for scoring to start.')
  })
})
