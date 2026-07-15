// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { TrendEntry } from '../../src/types'
import ScoreCard from '../../src/components/ScoreCard'

afterEach(cleanup)

function card(scorer: string, direction: TrendEntry['direction'], delta: number) {
  return (
    <ScoreCard
      scorer={scorer}
      mean={0.6}
      count={20}
      ci={[0.5, 0.7]}
      passRate={null}
      trend={{ direction, delta }}
      confident
    />
  )
}

describe('ScoreCard trends', () => {
  it('renders canonical trend directions with meaningful labels and values', () => {
    render(
      <>
        {card('judge.verification', 'regression', -0.3)}
        {card('judge.tool_choice', 'improvement', 0.3)}
      </>,
    )

    const regression = screen.getByRole('img', { name: 'Regression' })
    expect(regression.textContent).toBe('↓ -30.0%')

    const improvement = screen.getByRole('img', { name: 'Improvement' })
    expect(improvement.textContent).toBe('↑ +30.0%')
  })
})
