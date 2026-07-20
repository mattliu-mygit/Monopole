// @vitest-environment jsdom

import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { DataSelection, RunConfig } from '../../src/types'
import NewRun from '../../src/pages/NewRun'
import { renderWithQueryClient } from '../support/render'

const api = vi.hoisted(() => ({
  getModels: vi.fn(),
  getRubrics: vi.fn(),
  createRun: vi.fn(),
  setRunSelection: vi.fn(),
  setRunConfig: vi.fn(),
  setAutoRun: vi.fn(),
  advanceRun: vi.fn(),
}))

vi.mock('../../src/api', () => api)
vi.mock('../../src/features/runs/RunSetup', () => ({
  default: ({
    onStart,
  }: {
    onStart: (selection: DataSelection, config: RunConfig, autoRun: boolean) => void
  }) => (
    <button
      type="button"
      onClick={() => onStart(
        { since: null, until: null, timezone: 'UTC', session_ids: ['session-1'] },
        { model_catalog_version: 'models', rubric_catalog_version: 'rubrics' } as RunConfig,
        true,
      )}
    >
      Start Scoring
    </button>
  ),
}))

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()
  api.getModels.mockResolvedValue({ catalog_version: 'models' })
  api.getRubrics.mockResolvedValue({ catalog_version: 'rubrics' })
  api.createRun.mockResolvedValue({ run_id: 'run-created' })
  api.setRunSelection.mockResolvedValue({ run_id: 'run-created' })
  api.setRunConfig.mockResolvedValue({ run_id: 'run-created' })
  api.setAutoRun.mockResolvedValue({ run_id: 'run-created' })
  api.advanceRun.mockResolvedValue({ run_id: 'run-created', status: 'scoring' })
})

function renderPage() {
  const router = createMemoryRouter([
    { path: '/runs/new', element: <NewRun /> },
    { path: '/runs/:runId', element: <div>Persisted run</div> },
  ], { initialEntries: ['/runs/new'] })
  return { router, ...renderWithQueryClient(<RouterProvider router={router} />) }
}

describe('NewRun', () => {
  it('does not create a run until setup is submitted', async () => {
    const { router } = renderPage()

    expect(api.createRun).not.toHaveBeenCalled()
    fireEvent.click(await screen.findByRole('button', { name: 'Start Scoring' }))

    await waitFor(() => expect(api.advanceRun).toHaveBeenCalledWith('run-created'))
    expect(api.createRun).toHaveBeenCalledTimes(1)
    expect(api.setRunSelection).toHaveBeenCalledWith(
      'run-created',
      expect.objectContaining({ session_ids: ['session-1'] }),
    )
    expect(api.setRunConfig).toHaveBeenCalledWith(
      'run-created',
      expect.objectContaining({ model_catalog_version: 'models' }),
    )
    expect(api.setAutoRun).toHaveBeenCalledWith('run-created', true)
    expect(api.createRun.mock.invocationCallOrder[0]).toBeLessThan(
      api.setRunSelection.mock.invocationCallOrder[0],
    )
    expect(api.setRunSelection.mock.invocationCallOrder[0]).toBeLessThan(
      api.advanceRun.mock.invocationCallOrder[0],
    )
    expect(router.state.location.pathname).toBe('/runs/run-created')
  })
})
