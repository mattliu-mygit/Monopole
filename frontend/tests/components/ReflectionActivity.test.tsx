// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ReflectionActivity from '../../src/features/reflection/ReflectionActivity'
import type { ReflectingProgress } from '../../src/types'

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

const progress: ReflectingProgress = {
  phase: 'candidate_rejected',
  status_message: 'Rejected proposal attempt 1',
  started_at: '2026-07-14T19:20:00+00:00',
  attempted: 1,
  valid: 0,
  rejected: 1,
  scored: 0,
  total_attempts: 3,
  events: Array.from({ length: 6 }, (_, index) => ({
    id: index + 1,
    at: `2026-07-14T19:20:0${index + 1}+00:00`,
    phase: index === 5 ? 'candidate_rejected' : 'working',
    message: `Event ${index + 1}`,
  })),
}

describe('ReflectionActivity', () => {
  it('shows five newest events and expands the complete activity history', () => {
    render(<ReflectionActivity progress={progress} active />)

    expect(screen.queryByText('Event 1')).toBeNull()
    expect(screen.getByText('Event 6')).not.toBeNull()

    const disclosure = screen.getByRole('button', { name: 'Show all activity' })
    expect(disclosure.getAttribute('aria-expanded')).toBe('false')
    expect(disclosure.getAttribute('aria-controls')).toBe('reflection-activity-events')
    fireEvent.click(disclosure)

    expect(screen.getByText('Event 1')).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Show latest 5' }).getAttribute('aria-expanded'))
      .toBe('true')
  })

  it('shows exact reflection counts, advances by attempts, and only marks a live event current', () => {
    const { container, rerender } = render(
      <ReflectionActivity progress={progress} active />,
    )

    const progressbar = screen.getByRole('progressbar')
    expect(progressbar.getAttribute('aria-valuenow')).toBe('1')
    expect(progressbar.getAttribute('aria-valuemax')).toBe('3')
    expect(
      screen.getByText('1 of 3 proposal attempts · 0 valid · 1 rejected · 0 scored'),
    ).not.toBeNull()
    expect(screen.queryByText(/candidates generated/i)).toBeNull()
    expect(container.querySelector('[aria-current="step"]')?.textContent).toContain('Event 6')

    rerender(<ReflectionActivity progress={progress} active={false} />)

    expect(container.querySelector('[aria-current="step"]')).toBeNull()
  })

  it('keeps elapsed time moving while activity props are unchanged', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-07-14T19:20:10+00:00'))

    render(<ReflectionActivity progress={progress} active />)
    expect(screen.getByText('10s')).not.toBeNull()

    act(() => vi.advanceTimersByTime(2_000))

    expect(screen.getByText('12s')).not.toBeNull()
  })

  it('renders candidate, model, and score metadata independently of event prose', () => {
    render(
      <ReflectionActivity
        progress={{
          ...progress,
          events: [
            {
              id: 1,
              at: '2026-07-14T19:20:01+00:00',
              phase: 'candidate_scored',
              message: 'Evaluation finished',
              candidate: 2,
              model: 'gpt-oss-20b',
              score: 0.875,
            },
          ],
        }}
        active={false}
      />,
    )

    expect(screen.getByText('Candidate 2')).not.toBeNull()
    expect(screen.getByText('gpt-oss-20b')).not.toBeNull()
    expect(screen.getByText('Predicted evaluator score 0.875')).not.toBeNull()
  })

  it('keeps stable activity audit identifiers visible as compact metadata', () => {
    render(
      <ReflectionActivity
        progress={{
          ...progress,
          events: [{
            id: 1,
            at: '2026-07-14T19:20:01+00:00',
            phase: 'candidate_scored',
            message: 'Evaluation finished',
            acting_role: 'proposal_evaluator',
            attempt_id: 'attempt-stable-1',
            evaluation_id: 'evaluation-stable-1',
            status: 'succeeded',
          }],
        }}
        active={false}
      />,
    )

    expect(screen.getByText('proposal evaluator')).not.toBeNull()
    expect(screen.getByText('attempt attempt-stable-1')).not.toBeNull()
    expect(screen.getByText('evaluation evaluation-stable-1')).not.toBeNull()
    expect(screen.getByText('succeeded')).not.toBeNull()
  })

  it('shows rejected paths and keeps bounded writer evidence behind a disclosure', () => {
    render(
      <ReflectionActivity
        progress={{
          ...progress,
          events: [{
            id: 1,
            at: '2026-07-14T19:20:01+00:00',
            phase: 'candidate_rejected',
            message: 'Proposal path is outside managed scope',
            candidate: 1,
            attempt_id: 'attempt-1',
            model: 'writer-model',
            acting_role: 'proposal_writer',
            error_type: 'BundleValidationError',
            changed_paths: ['../secret.md', 'CLAUDE.md'],
            response_digest: `sha256:${'a'.repeat(64)}`,
            response_excerpt: '{"changes":[{"path":"../secret.md"}]}',
          }],
        }}
        active={false}
      />,
    )

    expect(screen.getByText('../secret.md').tagName).toBe('CODE')
    expect(screen.getByText('CLAUDE.md').tagName).toBe('CODE')
    expect(screen.queryByText(/generated candidate/i)).toBeNull()

    const disclosure = screen.getByText('Show rejected output')
    expect(disclosure.closest('details')?.hasAttribute('open')).toBe(false)
    fireEvent.click(disclosure)

    expect(screen.getByText(`sha256:${'a'.repeat(64)}`)).not.toBeNull()
    expect(screen.getByText('{"changes":[{"path":"../secret.md"}]}')).not.toBeNull()
  })

  it('does not label paths from a valid proposal as rejected', () => {
    render(
      <ReflectionActivity
        progress={{
          ...progress,
          events: [{
            id: 1,
            at: '2026-07-14T19:20:01+00:00',
            phase: 'candidate_valid',
            message: 'Validated proposal attempt 1',
            candidate: 1,
            changed_paths: ['CLAUDE.md'],
          }],
        }}
        active={false}
      />,
    )

    expect(screen.getByLabelText('Changed paths')).not.toBeNull()
    expect(screen.queryByLabelText('Rejected paths')).toBeNull()
  })
})
