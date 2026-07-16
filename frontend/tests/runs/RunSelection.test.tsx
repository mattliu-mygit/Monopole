// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { SessionSummary } from '../../src/types'
import RunSelection from '../../src/features/runs/RunSelection'

afterEach(cleanup)

const sessions: SessionSummary[] = [
  {
    conversation_id: 'session-one',
    session_id: 'one',
    turn_count: 3,
    started_at: '2026-07-12T12:00:00Z',
    ended_at: '2026-07-12T12:05:00Z',
    last_activity: '2026-07-12T12:05:00Z',
    model: 'gpt-5',
    effort_level: null,
    config_version: null,
    git_branch: null,
    total_tokens: 1200,
    total_tool_calls: 4,
    input_preview: 'Review the implementation',
    signal_evidence: [],
  },
  {
    conversation_id: 'session-two',
    session_id: 'two',
    turn_count: 1,
    started_at: '2026-07-13T15:00:00Z',
    ended_at: null,
    last_activity: '2026-07-13T15:00:00Z',
    model: 'claude',
    effort_level: null,
    config_version: null,
    git_branch: null,
    total_tokens: 400,
    total_tool_calls: 0,
    input_preview: null,
    signal_evidence: [],
  },
]

describe('RunSelection', () => {
  it('calls out sessions with low hydrated signal evidence', () => {
    const flaggedSessions = structuredClone(sessions)
    flaggedSessions[0].signal_evidence = [{
      signal: 'user-frustration',
      version: 'v1',
      rating: 0.5,
      reason: 'The user expressed frustration.',
      turn_id: 'turn-1',
      turn_started_at: '2026-07-12T12:00:00Z',
    }]

    render(
      <RunSelection
        since="2026-07-07"
        until="2026-07-14"
        sessions={flaggedSessions}
        selectedSessionIds={[]}
        totalSessions={2}
        onSinceChange={vi.fn()}
        onUntilChange={vi.fn()}
        onSessionIdsChange={vi.fn()}
      />,
    )

    expect(screen.getByText(/Needs review/)).not.toBeNull()
    expect(screen.getByText(/0\.50/)).not.toBeNull()
    expect(screen.getByText(/user frustration/)).not.toBeNull()
  })

  it('reports controlled date and session changes without owning run configuration', () => {
    const onSinceChange = vi.fn()
    const onUntilChange = vi.fn()
    const onSessionIdsChange = vi.fn()

    render(
      <RunSelection
        since="2026-07-07"
        until="2026-07-14"
        sessions={sessions}
        selectedSessionIds={['session-one']}
        totalSessions={2}
        onSinceChange={onSinceChange}
        onUntilChange={onUntilChange}
        onSessionIdsChange={onSessionIdsChange}
      />,
    )

    fireEvent.change(screen.getByLabelText('Since'), {
      target: { value: '2026-07-08' },
    })
    fireEvent.change(screen.getByLabelText('Until'), {
      target: { value: '2026-07-15' },
    })
    fireEvent.click(screen.getByRole('checkbox', { name: 'Select session session-one' }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Select session session-two' }))

    expect(onSinceChange).toHaveBeenCalledWith('2026-07-08')
    expect(onUntilChange).toHaveBeenCalledWith('2026-07-15')
    expect(onSessionIdsChange).toHaveBeenNthCalledWith(1, [])
    expect(onSessionIdsChange).toHaveBeenNthCalledWith(2, ['session-one', 'session-two'])
    expect(screen.queryByText('Pipeline Configuration')).toBeNull()
  })

  it('keeps loading, truncation, errors, and an empty range explicit', () => {
    const baseProps = {
      since: '2026-07-07',
      until: '2026-07-14',
      selectedSessionIds: [] as string[],
      totalSessions: 7,
      onSinceChange: vi.fn(),
      onUntilChange: vi.fn(),
      onSessionIdsChange: vi.fn(),
    }
    const { rerender } = render(
      <RunSelection {...baseProps} sessions={[]} loading />,
    )
    expect(screen.getByRole('status').textContent).toBe('Loading sessions...')

    rerender(
      <RunSelection
        {...baseProps}
        sessions={sessions}
        truncated
        error="Session lookup failed"
      />,
    )
    expect(screen.getByRole('alert').textContent).toContain('Session lookup failed')
    expect(screen.getByText(/Showing 2 of 7 sessions/)).not.toBeNull()

    rerender(<RunSelection {...baseProps} sessions={[]} totalSessions={0} />)
    expect(screen.getByText('No sessions found in this range.')).not.toBeNull()
  })

  it('offers an in-place retry without changing controlled selection state', () => {
    const onRetry = vi.fn()
    render(
      <RunSelection
        since="2026-07-07"
        until="2026-07-14"
        sessions={sessions}
        selectedSessionIds={['session-two']}
        totalSessions={2}
        error="Session lookup failed"
        onRetry={onRetry}
        onSinceChange={vi.fn()}
        onUntilChange={vi.fn()}
        onSessionIdsChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Retry session discovery' }))
    expect(onRetry).toHaveBeenCalledOnce()
    expect(screen.getByRole('checkbox', { name: 'Select session session-two' })).toHaveProperty('checked', true)
  })
})
