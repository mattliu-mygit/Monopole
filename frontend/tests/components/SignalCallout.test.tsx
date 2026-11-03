// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import SignalCallout from '../../src/components/SignalCallout'

afterEach(cleanup)

const evidence = [
  {
    signal: 'efficiency',
    version: 'v1',
    rating: 0.5,
    reason: 'The user sounds dissatisfied.',
    turn_id: 'turn-1',
    turn_started_at: '2026-07-15T12:00:00Z',
  },
  {
    signal: 'judge.verification',
    version: 'v1',
    rating: 0.25,
    reason: 'The response does not answer the request.',
    turn_id: 'turn-2',
    turn_started_at: '2026-07-15T12:01:00Z',
  },
  {
    signal: 'efficiency',
    version: 'v1',
    rating: 0.25,
    reason: 'The user explicitly says they are frustrated.',
    turn_id: 'turn-2',
    turn_started_at: '2026-07-15T12:01:00Z',
  },
]

describe('SignalCallout', () => {
  it('shows the lowest rating, each low scorer once, and the lowest Score\'s reason', () => {
    render(<SignalCallout evidence={evidence} />)

    const callout = screen.getByLabelText(/signal review recommendation/i)
    expect(callout.textContent).toContain('Needs review')
    expect(callout.textContent).toContain('0.25')
    expect(callout.textContent).toContain('efficiency')
    expect(callout.textContent).toContain('judge.verification')
    expect(callout.textContent?.match(/efficiency/g)).toHaveLength(1)
    expect(callout.getAttribute('title')).toBe('The response does not answer the request.')
  })

  it('renders nothing without low Signal evidence', () => {
    const { container } = render(<SignalCallout evidence={[]} />)

    expect(container.childElementCount).toBe(0)
  })
})
