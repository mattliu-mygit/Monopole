// @vitest-environment jsdom

import { cleanup, fireEvent, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Dashboard from '../../src/pages/Dashboard'
import { renderWithQueryClient } from '../support/render'

const api = vi.hoisted(() => ({ getAnalysis: vi.fn() }))
vi.mock('../../src/api', () => api)

afterEach(cleanup)
beforeEach(() => vi.clearAllMocks())

describe('Dashboard', () => {
  it('guides first-time users when no score summaries exist', async () => {
    api.getAnalysis.mockResolvedValue({
      summary: [],
      ab_leaderboard: [],
      trends: [],
      coaching_markdown: '',
    })
    renderWithQueryClient(<Dashboard />)

    expect(await screen.findByText(/no scores yet/i)).not.toBeNull()
    expect(screen.getByText(/complete an evaluation run/i)).not.toBeNull()
  })

  it('retries a failed analysis request without reloading the page', async () => {
    api.getAnalysis
      .mockRejectedValueOnce(new Error('Analysis unavailable'))
      .mockResolvedValueOnce({
        summary: [],
        ab_leaderboard: [],
        trends: [],
        coaching_markdown: '',
      })
    renderWithQueryClient(<Dashboard />)

    expect(await screen.findByText('Analysis unavailable')).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    expect(await screen.findByText(/no scores yet/i)).not.toBeNull()
    expect(api.getAnalysis).toHaveBeenCalledTimes(2)
  })
})
