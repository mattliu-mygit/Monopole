// @vitest-environment jsdom

import { cleanup, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { SessionSummary } from '../../src/types'
import Sessions from '../../src/pages/Sessions'
import { renderWithQueryClient } from '../support/render'

const api = vi.hoisted(() => ({ getSessions: vi.fn() }))
vi.mock('../../src/api', () => api)

afterEach(cleanup)

function session(conversationId: string, preview: string): SessionSummary {
  return {
    conversation_id: conversationId,
    session_id: null,
    turn_count: 1,
    started_at: '2026-07-14T18:00:00Z',
    ended_at: '2026-07-14T18:01:00Z',
    last_activity: '2026-07-14T18:01:00Z',
    model: 'test-model',
    effort_level: null,
    config_version: null,
    git_branch: null,
    total_tokens: 100,
    total_tool_calls: 1,
    input_preview: preview,
  }
}

function renderPage() {
  return renderWithQueryClient(
    <MemoryRouter>
      <Sessions />
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  api.getSessions.mockResolvedValue({
    sessions: [
      session('plain-session', 'First session'),
      session('session/with space', 'Second session'),
    ],
    total: 2,
    truncated: false,
  })
})

describe('Sessions', () => {
  it('renders every navigable session row as a link with an encoded destination', async () => {
    renderPage()

    expect((await screen.findByRole('link', { name: /First session/ })).getAttribute('href'))
      .toBe('/sessions/plain-session')
    expect(screen.getByRole('link', { name: /Second session/ }).getAttribute('href'))
      .toBe('/sessions/session%2Fwith%20space')
  })

  it('shows the returned total and counts older sessions omitted by truncation', async () => {
    api.getSessions.mockResolvedValue({
      sessions: [
        session('newest-session', 'Newest session'),
        session('next-session', 'Next session'),
      ],
      total: 5,
      truncated: true,
      limit: 2,
    })
    renderPage()

    expect(await screen.findByText(/showing the newest 2 of 5 sessions/i)).not.toBeNull()
    expect(screen.getByText(/3 older sessions are omitted/i)).not.toBeNull()
  })
})
