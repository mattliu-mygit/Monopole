// @vitest-environment jsdom

import { cleanup, fireEvent, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Analyze from '../../src/pages/Analyze'
import { renderWithQueryClient } from '../support/render'

const api = vi.hoisted(() => ({ getAnalysis: vi.fn() }))
vi.mock('../../src/api', () => api)

afterEach(cleanup)

function renderPage() {
  return renderWithQueryClient(<Analyze />)
}

beforeEach(() => {
  vi.clearAllMocks()
  api.getAnalysis.mockResolvedValue({
    summary: [],
    ab_leaderboard: [],
    trends: [
      {
        scorer: 'judge.verification',
        direction: 'regression',
        older_mean: 0.8,
        recent_mean: 0.5,
        delta: -0.3,
        sample_count: 20,
        significant: true,
      },
      {
        scorer: 'judge.tool_choice',
        direction: 'improvement',
        older_mean: 0.4,
        recent_mean: 0.7,
        delta: 0.3,
        sample_count: 18,
        significant: false,
      },
    ],
    coaching_markdown: '',
  })
})

describe('Analyze trends', () => {
  it('renders regressions and improvements with accessible directional status', async () => {
    renderPage()
    fireEvent.click(screen.getByRole('tab', { name: 'Trends' }))

    const regression = await screen.findByRole('img', { name: 'Regression' })
    expect(regression.textContent).toBe('↓')

    const improvement = screen.getByRole('img', { name: 'Improvement' })
    expect(improvement.textContent).toBe('↑')

    expect(screen.getByText('judge.verification')).not.toBeNull()
    expect(screen.getByText('judge.tool_choice')).not.toBeNull()
  })
})

describe('Analyze empty states', () => {
  it('explains how to populate an empty score summary', async () => {
    renderPage()

    expect(await screen.findByText(/no score summaries yet/i)).not.toBeNull()
    expect(screen.getByText(/complete an evaluation run/i)).not.toBeNull()
  })

  it('explains what data is needed for an A/B comparison', async () => {
    renderPage()
    fireEvent.click(screen.getByRole('tab', { name: 'A/B Comparison' }))

    expect(await screen.findByText(/no a\/b comparisons yet/i)).not.toBeNull()
    expect(screen.getByText(/more than one configuration version/i)).not.toBeNull()
  })
})

describe('Analyze A/B evidence counts', () => {
  it('labels mixed turn and session score refs as evaluated targets', async () => {
    api.getAnalysis.mockResolvedValue({
      summary: [],
      ab_leaderboard: [{
        config_version: 'config-version-one',
        evaluated_target_count: 3,
        scores: { 'judge.verification': { mean: 0.75, count: 3 } },
      }],
      trends: [],
      coaching_markdown: '',
    })
    renderPage()
    fireEvent.click(screen.getByRole('tab', { name: 'A/B Comparison' }))

    expect(await screen.findByText('3 evaluated targets')).not.toBeNull()
    expect(screen.queryByText('3 turns')).toBeNull()
  })
})

describe('Analyze tabs', () => {
  it('associates tabs with panels and supports Arrow, Home, and End keys', () => {
    renderPage()

    const summary = screen.getByRole('tab', { name: 'Summary' })
    const comparison = screen.getByRole('tab', { name: 'A/B Comparison' })
    const coaching = screen.getByRole('tab', { name: 'Coaching' })
    const controlledId = summary.getAttribute('aria-controls')

    expect(screen.getByRole('tablist', { name: 'Analysis views' })).not.toBeNull()
    expect(summary.getAttribute('aria-selected')).toBe('true')
    expect(controlledId).not.toBeNull()
    expect(document.getElementById(controlledId!)?.getAttribute('aria-labelledby')).toBe(summary.id)

    fireEvent.keyDown(summary, { key: 'ArrowRight' })
    expect(comparison.getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(comparison)

    fireEvent.keyDown(comparison, { key: 'End' })
    expect(coaching.getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(coaching)

    fireEvent.keyDown(coaching, { key: 'Home' })
    expect(summary.getAttribute('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(summary)
  })
})
