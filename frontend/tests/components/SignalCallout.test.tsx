// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import SignalCallout from '../../src/components/SignalCallout'

afterEach(cleanup)

const evidence = [
  {
    signal: 'user-frustration',
    version: 'v1',
    rating: 0.5,
    reason: 'The user sounds dissatisfied.',
    turn_id: 'turn-1',
    turn_started_at: '2026-07-15T12:00:00Z',
  },
  {
    signal: 'low-quality-response',
    version: 'v1',
    rating: 0.25,
    reason: 'The response does not answer the request.',
    turn_id: 'turn-2',
    turn_started_at: '2026-07-15T12:01:00Z',
  },
  {
    signal: 'user-frustration',
    version: 'v1',
    rating: 0.25,
    reason: 'The user explicitly says they are frustrated.',
    turn_id: 'turn-2',
    turn_started_at: '2026-07-15T12:01:00Z',
  },
]

describe('SignalCallout', () => {
  it('shows one review recommendation with the lowest rating and distinct signals', () => {
    render(<SignalCallout evidence={evidence} />)

    const callout = screen.getByLabelText(/signal review recommendation/i)
    expect(callout.textContent).toContain('Needs review')
    expect(callout.textContent).toContain('0.25')
    expect(callout.textContent).toContain('user frustration')
    expect(callout.textContent).toContain('low quality response')
    expect(callout.textContent?.match(/user frustration/g)).toHaveLength(1)
  })

  it('renders nothing without low Signal evidence', () => {
    const { container } = render(<SignalCallout evidence={[]} />)

    expect(container.childElementCount).toBe(0)
  })
})
