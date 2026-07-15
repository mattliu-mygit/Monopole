// @vitest-environment jsdom

import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { RunSummary } from '../../src/types'
import Runs from '../../src/pages/Runs'
import { shouldPollRuns } from '../../src/features/runs/runPolling'
import { renderWithQueryClient } from '../support/render'

const api = vi.hoisted(() => ({ getRuns: vi.fn(), createRun: vi.fn() }))
vi.mock('../../src/api', () => api)

afterEach(cleanup)

function run(
  status: RunSummary['status'],
  reviewState: RunSummary['review_state'] = 'none',
): RunSummary {
  return {
    run_id: `run-${status}`,
    status,
    current_stage_succeeded: status === 'complete',
    created_at: '2026-07-14T18:00:00Z',
    selection: null,
    review_state: reviewState,
  }
}

function renderPage() {
  return renderWithQueryClient(
    <MemoryRouter>
      <Runs />
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  api.createRun.mockResolvedValue(run('created'))
})

describe('Runs review state', () => {
  it('announces a fetch error and retries in place', async () => {
    api.getRuns
      .mockRejectedValueOnce(new Error('Runs API unavailable'))
      .mockResolvedValueOnce({ runs: [] })
    renderPage()

    expect(await screen.findByRole('alert')).toHaveProperty(
      'textContent',
      'Runs API unavailable',
    )
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    await waitFor(() => expect(api.getRuns).toHaveBeenCalledTimes(2))
    expect(await screen.findByText(/No runs yet/)).not.toBeNull()
  })

  it('shows review lifecycle independently from pipeline completion', async () => {
    api.getRuns.mockResolvedValue({
      runs: [
        run('complete', 'review-needed'),
        {
          ...run('complete', 'promoted'),
          run_id: 'run-promoted',
        },
      ],
    })
    renderPage()

    expect(await screen.findByRole('columnheader', { name: 'Review' })).not.toBeNull()
    expect(screen.getByText('Review needed')).not.toBeNull()
    expect(screen.getByText('Promoted')).not.toBeNull()
    expect(screen.getByRole('link', { name: 'run-complete' }).getAttribute('href'))
      .toBe('/runs/run-complete')
  })

  it('labels a completed baseline winner as no change', async () => {
    const baselineWinner = run('complete', 'no-change')
    api.getRuns.mockResolvedValue({ runs: [baselineWinner] })
    renderPage()

    expect(await screen.findByText('No change')).not.toBeNull()
  })

  it('labels an all-invalid reflection separately from a baseline win', async () => {
    api.getRuns.mockResolvedValue({ runs: [run('complete', 'no-valid-proposal')] })
    renderPage()

    expect(await screen.findByText('No valid proposal')).not.toBeNull()
    expect(screen.queryByText('No change')).toBeNull()
  })

  it('treats cancelled as terminal for list polling and renders its badge', async () => {
    const cancelled = run('cancelled')
    api.getRuns.mockResolvedValue({ runs: [cancelled] })
    renderPage()

    expect(await screen.findByText('cancelled')).not.toBeNull()
    expect(shouldPollRuns([cancelled])).toBe(false)
    expect(shouldPollRuns([run('reflecting')])).toBe(true)
  })

  it('formats pinned date-only boundaries in the selected timezone and names it', async () => {
    api.getRuns.mockResolvedValue({
      runs: [{
        ...run('complete'),
        run_id: 'run-zoned',
        selection: {
          session_count: 2,
          since: '2026-01-01',
          until: '2026-01-02',
          timezone: 'America/Los_Angeles',
        },
      }],
    })
    renderPage()

    expect(await screen.findByText('Jan 1, 2026 – Jan 2, 2026')).not.toBeNull()
    expect(screen.getByText('America/Los_Angeles')).not.toBeNull()
  })
})
