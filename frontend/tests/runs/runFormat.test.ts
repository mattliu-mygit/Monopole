import { describe, expect, it } from 'vitest'
import { durationLabel, turnRanges, usageText, words } from '../../src/features/runs/runFormat'

describe('run activity formatters', () => {
  it('reads turn positions as ranges', () => {
    expect(turnRanges([41, 2, 1, 3])).toBe('turns 1–3, 41')
    expect(turnRanges([114])).toBe('turn 114')
  })

  it('names usage and durations', () => {
    expect(words('not_evaluable')).toBe('not evaluable')
    expect(usageText({ input_tokens: 1200, output_tokens: 300 })).toBe('input tokens 1200 · output tokens 300')
    expect(usageText({})).toBe('—')
    const start = '2026-11-05T22:00:00Z'
    expect(durationLabel(start, Date.parse(start) + 185_000)).toBe('3m 5s')
    expect(durationLabel(start, Date.parse(start) + 3_720_000)).toBe('1h 2m')
    expect(durationLabel('not a time', 0)).toBeNull()
  })
})
