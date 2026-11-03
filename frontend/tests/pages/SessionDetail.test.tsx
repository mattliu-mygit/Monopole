// @vitest-environment jsdom

import { cleanup, fireEvent, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { SessionDetail as SessionDetailData } from '../../src/types'
import SessionDetail from '../../src/pages/SessionDetail'
import { renderWithQueryClient } from '../support/render'

const api = vi.hoisted(() => ({
  getSession: vi.fn(),
  getRubrics: vi.fn(),
  getTurnSubagents: vi.fn(),
}))
vi.mock('../../src/api', () => api)

afterEach(cleanup)

function renderPage() {
  return renderWithQueryClient(
    <MemoryRouter initialEntries={['/sessions/conversation-1']}>
      <Routes>
        <Route path="/sessions/:id" element={<SessionDetail />} />
      </Routes>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  const session: SessionDetailData = {
    conversation_id: 'conversation-1',
    config_version: 'config-version',
    git_branch: 'main',
    total_tokens: 30,
    turn_count: 1,
    turns: [{
      trace_id: 'trace-1234567890',
      conversation_id: 'conversation-1',
      started_at: '2026-07-14T18:00:00Z',
      ended_at: '2026-07-14T18:01:00Z',
      model: 'test-model',
      input_tokens: 10,
      output_tokens: 20,
      cache_read_tokens: 0,
      status_code: 'OK',
      config_version: 'config-version',
      git_branch: 'main',
      effort_level: null,
      session_id: null,
      steering_count: 0,
      denial_count: 0,
      tool_error_count: 0,
      tool_call_count: 0,
      chat_span_count: 0,
      subagent_count: 0,
      user_input: 'Inspect the run',
      tool_calls: [],
      chat_spans: [],
      subagents: [],
    }],
    signal_evidence: [{
      signal: 'efficiency',
      version: 'v1',
      rating: 0.25,
      reason: 'The user explicitly says they are frustrated.',
      turn_id: 'trace-1234567890',
      turn_started_at: '2026-07-14T18:00:00Z',
    }],
    session_scores: [{
      id: 'feedback-1',
      feedback_type: 'weave_agent_signals.judge.verification',
      payload: {
        rating: 0.5,
        tags: [],
        reason: 'Some evidence was missing.',
        details: {
          attempts: [
            {
              position: 1,
              role: 'primary',
              trigger: 'panel',
              requested_model: 'gpt-5.6-sol',
              requested_family: 'openai',
              requested_backend: 'cli',
              resolved_model: 'gpt-5.6-sol',
              resolved_family: 'openai',
              status: 'succeeded',
              score: 0.45,
              rationale: 'Some evidence was missing.',
              usage: {},
              error_type: null,
              message: null,
            },
            {
              position: 2,
              role: 'review',
              trigger: 'near_boundary',
              requested_model: 'claude-sonnet-5',
              requested_family: 'anthropic',
              requested_backend: 'cli',
              resolved_model: null,
              resolved_family: null,
              status: 'failed',
              score: null,
              rationale: null,
              usage: {},
              error_type: 'RuntimeError',
              message: 'review offline',
            },
          ],
        },
      },
    }],
    turn_scores: {},
  }
  api.getSession.mockResolvedValue(session)
  api.getRubrics.mockResolvedValue({
    catalog_version: 'rubric-catalog-v1',
    rubrics: [{
      id: 'judge.verification',
      label: 'Verification',
      evaluation_unit: 'session',
      version: '1',
      content_digest: 'sha256:verification',
      pass_threshold: 0.7,
    }],
  })
})

describe('SessionDetail disclosures', () => {
  it('renders Agent Signals separately from score feedback', async () => {
    renderPage()

    expect(
      await screen.findByLabelText('Signal review recommendation: 0.25; efficiency'),
    ).not.toBeNull()
    expect(await screen.findByText('Session Scores')).not.toBeNull()
  })

  it('renders current review attempts including failures', async () => {
    renderPage()

    const attempts = await screen.findByRole('list', { name: 'Review attempts' })
    expect(attempts.textContent).toContain('Judge 1')
    expect(attempts.textContent).toContain('gpt-5.6-sol')
    expect(attempts.textContent).toContain('0.450')
    expect(attempts.textContent).toContain('Judge 2')
    expect(attempts.textContent).toContain('near boundary')
    expect(attempts.textContent).toContain('RuntimeError: review offline')
  })

  it('shows canonical rubric metadata and an accessible turn disclosure', async () => {
    renderPage()

    expect(await screen.findByText('Verification · whole session · pass threshold 0.70')).not.toBeNull()

    const turnButton = screen.getByRole('button', { name: /Turn 1/ })
    expect(turnButton.getAttribute('aria-expanded')).toBe('false')
    expect(turnButton.getAttribute('aria-controls')).toBeTruthy()

    fireEvent.click(turnButton)
    expect(turnButton.getAttribute('aria-expanded')).toBe('true')
    expect(document.getElementById(turnButton.getAttribute('aria-controls')!)).not.toBeNull()
  })

  it('marks a denied tool call and shows its exit code', async () => {
    const session: SessionDetailData = await api.getSession('conversation-1')
    api.getSession.mockResolvedValue({
      ...session,
      turns: [{
        ...session.turns[0],
        tool_call_count: 1,
        tool_calls: [{
          span_id: 'tool-1',
          tool_name: 'Bash',
          arguments: '{"command":"rm -rf build"}',
          result: '',
          status_code: 'ERROR',
          started_at: '2026-07-14T18:00:10Z',
          ended_at: '2026-07-14T18:00:11Z',
          exit_code: 1,
          permission_decision: 'not_approved',
        }],
      }],
    })
    renderPage()

    fireEvent.click(await screen.findByRole('button', { name: /Turn 1/ }))

    expect(screen.getByText('denied')).not.toBeNull()
    expect(screen.getByText('exit 1')).not.toBeNull()
  })

  it("loads a turn's subagents only when asked", async () => {
    const session: SessionDetailData = await api.getSession('conversation-1')
    api.getSession.mockResolvedValue({
      ...session,
      turns: [{
        ...session.turns[0],
        subagent_count: 1,
        subagents: [{ span_id: 'traj-sub', agent_type: null, tool_call_count: null }],
      }],
    })
    api.getTurnSubagents.mockResolvedValue({
      subagents: [{ span_id: 'traj-sub', agent_type: 'explorer', tool_call_count: 3 }],
    })
    renderPage()

    fireEvent.click(await screen.findByRole('button', { name: /Turn 1/ }))
    expect(api.getTurnSubagents).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Load details' }))

    expect(await screen.findByText('explorer')).not.toBeNull()
    expect(screen.getByText('3 tool calls')).not.toBeNull()
    expect(api.getTurnSubagents).toHaveBeenCalledWith('conversation-1', 'trace-1234567890')
  })
})
