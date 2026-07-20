// @vitest-environment jsdom

import { act, cleanup, fireEvent, screen, waitFor } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
  EffectiveRunConfig,
  JudgingPlan,
  ModelCatalog,
  ModelDescriptor,
  RubricCatalog,
  Run,
  RunConfig,
  SessionSummary,
} from '../../src/types'
import { ApiError } from '../../src/api'
import RunDetail from '../../src/pages/RunDetail'
import persistedPlan from '../fixtures/judging-plan-view.json'
import { renderWithQueryClient } from '../support/render'

const api = vi.hoisted(() => ({
  getRun: vi.fn(),
  getSessions: vi.fn(),
  getSession: vi.fn(),
  getModels: vi.fn(),
  getRubrics: vi.fn(),
  setRunSelection: vi.fn(),
  setRunConfig: vi.fn(),
  setAutoRun: vi.fn(),
  advanceRun: vi.fn(),
  cancelRun: vi.fn(),
  setReflectionSelection: vi.fn(),
  saveReflectionDraft: vi.fn(),
  resetReflectionDraft: vi.fn(),
  promoteRunReflection: vi.fn(),
  dismissRunReflection: vi.fn(),
}))

vi.mock('../../src/api', () => ({
  ...api,
  ApiError: class ApiError extends Error {
    status: number
    detail: unknown
    constructor(status: number, body: string, detail: unknown) {
      const message =
        detail && typeof detail === 'object' && 'message' in detail
          ? String(detail.message)
          : body
      super(`API ${status}: ${message}`)
      this.status = status
      this.detail = detail
    }
  },
}))

afterEach(cleanup)

const writer: ModelDescriptor = {
  id: 'writer-openai',
  label: 'Writer OpenAI',
  family: 'openai',
  provider: 'codex',
  provider_model: 'writer-openai',
  supported_roles: ['proposal_writer'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const judgeOne: ModelDescriptor = {
  id: 'judge-anthropic',
  label: 'Judge Anthropic',
  family: 'anthropic',
  provider: 'claude',
  provider_model: 'judge-anthropic',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const judgeTwo: ModelDescriptor = {
  id: 'judge-openai',
  label: 'Judge OpenAI',
  family: 'openai',
  provider: 'codex',
  provider_model: 'judge-openai',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const judgeThree: ModelDescriptor = {
  id: 'judge-meta',
  label: 'Judge Meta',
  family: 'meta',
  provider: 'agy',
  provider_model: 'judge-meta',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}

const models: ModelCatalog = {
  catalog_version: 'models-v1',
  judging_context: {
    contract_version: '3', large_model_threshold_tokens: 200_000,
    large_model_reserve_tokens: 100_000, small_model_reserve_tokens: 50_000,
    large_model_raw_target_tokens: 128_000, small_model_raw_target_tokens: 50_000,
    prompt_reserve_tokens: 6_000, output_reserve_tokens: 4_000,
    large_model_output_reserve_tokens: 10_000,
    safety_reserve_tokens: 8_000, digest_max_tokens: 1_000,
    finding_max_tokens: 4_000, overlap_turns: 1, max_chunks: 40,
  },
  available_models: [writer, judgeOne, judgeTwo, judgeThree],
  recommended_proposal_model: writer.id,
  recommended_judges: [judgeOne.id, judgeTwo.id, judgeThree.id],
  recommended_challenge_judges: [judgeOne.id],
  proposal_evaluator_preferences: [judgeOne.id, judgeTwo.id],
}

const rubrics: RubricCatalog = {
  catalog_version: 'rubrics-v1',
  rubrics: [
    {
      id: 'judge.verification',
      label: 'Verification discipline',
      evaluation_unit: 'session',
      version: 'v1',
      content_digest: 'digest-verification',
      pass_threshold: 0.5,
    },
    {
      id: 'judge.session_outcome',
      label: 'Session outcome',
      evaluation_unit: 'session',
      version: 'v1',
      content_digest: 'digest-outcome',
      pass_threshold: 0.6,
    },
  ],
}

const requestedConfig: RunConfig = {
  model_catalog_version: models.catalog_version,
  rubric_catalog_version: rubrics.catalog_version,
  judge_models: [judgeOne.id, judgeTwo.id, judgeThree.id],
  challenge_judge_models: [judgeOne.id],
  proposal_model: writer.id,
  proposal_evaluator_model: judgeOne.id,
  rubrics: rubrics.rubrics.map((rubric) => rubric.id),
  candidate_budget: 3,
  force: false,
}

const effectiveConfig: EffectiveRunConfig = {
  schema_version: '5',
  pipeline_version: '8',
  model_catalog_version: models.catalog_version,
  rubric_catalog_version: rubrics.catalog_version,
  models: {
    proposal_writer: writer,
    judges: [judgeOne, judgeTwo, judgeThree].map((judge, index) => ({
      ...judge,
      role: 'judge' as const,
      position: index + 1,
    })),
    challenge_judges: [{ ...judgeOne, role: 'judge' as const, position: 1 }],
    proposal_evaluator: judgeOne,
  },
  rubrics: rubrics.rubrics,
  selection_warnings: [],
  judging_context: {
    contract_version: '3', large_model_threshold_tokens: 200_000,
    large_model_reserve_tokens: 100_000, small_model_reserve_tokens: 50_000,
    large_model_raw_target_tokens: 128_000, small_model_raw_target_tokens: 50_000,
    prompt_reserve_tokens: 6_000,
    output_reserve_tokens: 4_000, large_model_output_reserve_tokens: 10_000,
    safety_reserve_tokens: 8_000, digest_max_tokens: 1_000,
    finding_max_tokens: 4_000, overlap_turns: 1, max_chunks: 40,
  },
  candidate_budget: 3,
  force: false,
}

function baseRun(overrides: Partial<Run> = {}): Run {
  return {
    run_id: 'run-ui',
    status: 'created',
    current_stage_succeeded: false,
    created_at: '2026-07-14T18:00:00Z',
    auto_run: false,
    data_selection: null,
    run_config: null,
    effective_config: null,
    turn_cohort: null,
    judging_plan: null,
    scoring_progress: null,
    scoring_result: null,
    judging_progress: null,
    judging_result: null,
    reflecting_progress: null,
    reflecting_result: null,
    reflection_review: null,
    reflection_review_revision: 0,
    error: null,
    ...overrides,
  }
}

const session: SessionSummary = {
  conversation_id: 'session-1',
  session_id: null,
  turn_count: 2,
  started_at: '2026-07-13T22:00:00Z',
  ended_at: null,
  last_activity: null,
  model: 'agent-model',
  effort_level: null,
  config_version: null,
  git_branch: null,
  total_tokens: 100,
  total_tool_calls: 2,
  input_preview: 'Evaluate this session',
  signal_evidence: [],
}

const sessionEstimates = {
  utf8_bytes_div_3: { total_tokens: 100, largest_turn_tokens: 60, turn_tokens: [60, 40] },
  o200k_base: { total_tokens: 90, largest_turn_tokens: 55, turn_tokens: [55, 35] },
  o200k_harmony: { total_tokens: 85, largest_turn_tokens: 50, turn_tokens: [50, 35] },
}

const plan = persistedPlan as unknown as JudgingPlan

function renderPage() {
  const router = createMemoryRouter([
    { path: '/runs/:runId', element: <RunDetail /> },
    { path: '/runs', element: <div>Run list</div> },
  ], { initialEntries: ['/runs/run-ui'] })
  return {
    router,
    ...renderWithQueryClient(<RouterProvider router={router} />),
  }
}

function completedReflectionRun(): Run {
  const past = {
    kind: 'file',
    locator: 'CLAUDE.md',
    display_name: 'CLAUDE.md',
    path: 'CLAUDE.md',
    exists: true,
    content: 'Past',
    revision: 'target:b',
  }
  const proposed = { ...past, content: 'Proposed', revision: 'target:c' }
  return baseRun({
    status: 'complete',
    current_stage_succeeded: true,
    run_config: requestedConfig,
    effective_config: effectiveConfig,
    scoring_result: {
      turns_scored: 2,
      sessions_scored: 1,
      scores_written: 4,
      errors: 0,
      turn_details: [],
    },
    judging_result: {
      plan_id: plan.plan_id,
      planned_rubrics: 1,
      rubrics_completed: 1,
      rated_rubrics: 1,
      not_evaluable_rubrics: 0,
      minimum_reviewer_attempts: 2,
      maximum_reviewer_attempts: 2,
      reviewer_attempts_completed: 2,
      digest_steps_completed: 2,
      maximum_digest_steps: 2,
      window_steps_completed: 1,
      maximum_window_steps: 2,
      merge_steps_completed: 1,
      maximum_merge_steps: 2,
      scores_written: 1,
      failure_count: 0,
      write_failure_count: 0,
      coverage_complete: true,
      phase: 'judging_complete',
      status_message: 'Judging complete',
      started_at: '2026-07-15T07:30:00Z',
      events: [],
      attempt_summaries: [],
      attempt_summary_count: 0,
      attempt_summaries_truncated: false,
      failure_details: [],
      failure_detail_count: 0,
      failure_details_truncated: false,
    },
    reflecting_result: {
      baseline: { scope: null, targets: [past], revision: 'bundle:b' },
      baseline_score: 0.4,
      baseline_evaluation: {
        evaluation_id: 'evaluation-b',
        target_revision: 'bundle:b',
        requested_model: 'evaluator',
        requested_family: 'family',
        requested_backend: 'cli',
        resolved_model: 'evaluator',
        resolved_family: 'family',
        resolved_backend: 'cli',
        score: 0.4,
        rationale: 'Baseline evidence.',
        usage: {},
      },
      candidates: [
        {
          candidate_id: 'candidate-1',
          bundle: { scope: null, targets: [proposed], revision: 'bundle:c' },
          score: 0.7,
          score_delta: 0.3,
          rationale: 'Improve verification.',
          generation_attempt_id: 'attempt-1',
          requested_writer: writer,
          resolved_writer_model: writer.id,
          resolved_writer_family: writer.family,
          resolved_writer_backend: writer.provider,
          evaluation: {
            evaluation_id: 'evaluation-c',
            target_revision: 'bundle:c',
            requested_model: 'evaluator',
            requested_family: 'family',
            requested_backend: 'cli',
            resolved_model: 'evaluator',
            resolved_family: 'family',
            resolved_backend: 'cli',
            score: 0.7,
            rationale: 'Improves verification.',
            usage: {},
          },
        },
      ],
      generation_attempts: [],
      recommended_candidate_id: 'candidate-1',
      baseline_won: false,
      reason: null,
      score_basis: 'predicted_evaluator',
      provisional_candidate_id: 'candidate-1',
      challenge: null,
    },
    reflection_review: {
      status: 'pending',
      selected_candidate_id: 'candidate-1',
      draft: null,
    },
    reflection_review_revision: 1,
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  const created = baseRun()
  api.getRun.mockResolvedValue(created)
  api.getSessions.mockResolvedValue({ sessions: [session], total: 1, truncated: false })
  api.getSession.mockResolvedValue({
    judging_token_estimates: sessionEstimates,
  })
  api.getModels.mockResolvedValue(models)
  api.getRubrics.mockResolvedValue(rubrics)
  api.setRunSelection.mockResolvedValue(created)
  api.setRunConfig.mockResolvedValue(created)
  api.setAutoRun.mockResolvedValue(created)
  api.advanceRun.mockResolvedValue({ ...created, status: 'scoring' })
  api.cancelRun.mockResolvedValue({ ...created, status: 'cancelled' })
})

describe('RunDetail wiring', () => {
  it('starts with discovered sessions unchecked and saves explicit selection before starting', async () => {
    renderPage()

    const sessionCheckbox = await screen.findByRole('checkbox', {
      name: 'Select session session-1',
    })
    const start = await screen.findByRole('button', { name: 'Start Scoring' })
    expect(sessionCheckbox).toHaveProperty('checked', false)
    expect(start).toHaveProperty('disabled', true)
    expect(await screen.findByRole('alert')).toHaveProperty(
      'textContent',
      expect.stringContaining('Select at least one session.'),
    )

    fireEvent.click(sessionCheckbox)
    await waitFor(() => expect(start).toHaveProperty('disabled', false))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Continue automatically' }))
    fireEvent.click(start)

    await waitFor(() => expect(api.advanceRun).toHaveBeenCalledWith('run-ui'))
    expect(api.setRunSelection).toHaveBeenCalledWith(
      'run-ui',
      expect.objectContaining({ session_ids: ['session-1'] }),
    )
    expect(api.setRunConfig).toHaveBeenCalledWith('run-ui', requestedConfig)
    expect(api.setAutoRun).toHaveBeenCalledWith('run-ui', true)
    expect(api.setRunSelection.mock.invocationCallOrder[0]).toBeLessThan(
      api.setRunConfig.mock.invocationCallOrder[0],
    )
    expect(api.setRunConfig.mock.invocationCallOrder[0]).toBeLessThan(
      api.advanceRun.mock.invocationCallOrder[0],
    )
    expect(screen.queryByRole('textbox', { name: /model/i })).toBeNull()
  })

  it('clears the unsaved session selection when the date range changes', async () => {
    renderPage()

    let sessionCheckbox = await screen.findByRole('checkbox', {
      name: 'Select session session-1',
    })
    fireEvent.click(sessionCheckbox)
    expect(screen.getByText('1 selected')).not.toBeNull()

    const sinceInput = screen.getByLabelText('Since') as HTMLInputElement
    fireEvent.change(sinceInput, {
      target: { value: sinceInput.value === '2026-07-10' ? '2026-07-11' : '2026-07-10' },
    })

    sessionCheckbox = await screen.findByRole('checkbox', {
      name: 'Select session session-1',
    })
    expect(sessionCheckbox).toHaveProperty('checked', false)
    expect(screen.getByText('0 selected')).not.toBeNull()

    fireEvent.click(sessionCheckbox)
    const untilInput = screen.getByLabelText('Until') as HTMLInputElement
    fireEvent.change(untilInput, {
      target: { value: untilInput.value === '2026-07-15' ? '2026-07-16' : '2026-07-15' },
    })

    expect(await screen.findByRole('checkbox', {
      name: 'Select session session-1',
    })).toHaveProperty('checked', false)
    expect(screen.getByText('0 selected')).not.toBeNull()
  })

  it('restores an explicit saved selection on a created run', async () => {
    api.getRun.mockResolvedValue(baseRun({
      data_selection: {
        since: '2026-07-10T07:00:00Z',
        until: null,
        timezone: 'America/Los_Angeles',
        session_ids: ['session-1'],
      },
    }))
    renderPage()

    expect(await screen.findByRole('checkbox', {
      name: 'Select session session-1',
    })).toHaveProperty('checked', true)
    await waitFor(() => expect(screen.getByRole('button', {
      name: 'Start Scoring',
    })).toHaveProperty('disabled', false))
  })

  it('restores saved form state when navigating to a cached created run', async () => {
    let cachedRun = baseRun({ run_id: 'run-ui' })
    api.getRun.mockImplementation(() => Promise.resolve(cachedRun))
    const rendered = renderPage()
    await screen.findByText('0 selected')
    cachedRun = baseRun({
      run_id: 'run-b',
      data_selection: {
        since: '2026-07-10T07:00:00Z',
        until: null,
        timezone: 'America/Los_Angeles',
        session_ids: ['session-1'],
      },
      run_config: {
        ...requestedConfig,
        judge_models: [judgeOne.id],
        rubrics: [rubrics.rubrics[0].id],
        candidate_budget: 1,
      },
    })
    rendered.client.setQueryData(['run', 'run-b'], cachedRun)

    await act(() => rendered.router.navigate('/runs/run-b'))

    expect(await screen.findByRole('checkbox', {
      name: 'Select session session-1',
    })).toHaveProperty('checked', true)
    expect(screen.getByLabelText('Proposal attempt limit')).toHaveProperty('value', '1')
    expect(screen.queryByLabelText('Judge 2')).toBeNull()
  })

  it('blocks a saved selection whose session summary is not currently loaded', async () => {
    api.getRun.mockResolvedValue(baseRun({
      data_selection: {
        since: '2026-07-10T07:00:00Z',
        until: null,
        timezone: 'America/Los_Angeles',
        session_ids: ['session-outside-page'],
      },
    }))
    api.getSessions.mockResolvedValue({ sessions: [session], total: 10, truncated: true })
    renderPage()

    expect(await screen.findByRole('button', {
      name: 'Start Scoring',
    })).toHaveProperty('disabled', true)
    expect(await screen.findByRole('alert')).toHaveProperty(
      'textContent',
      expect.stringContaining('Load every selected session before starting.'),
    )
  })

  it('allows an explicit visible selection when session discovery is truncated', async () => {
    api.getSessions.mockResolvedValue({ sessions: [session], total: 10, truncated: true })
    renderPage()

    fireEvent.click(await screen.findByRole('checkbox', {
      name: 'Select session session-1',
    }))

    await waitFor(() => expect(screen.getByRole('button', {
      name: 'Start Scoring',
    })).toHaveProperty('disabled', false))
    expect(screen.queryByText(/Narrow the date range before starting/)).toBeNull()
    expect(screen.getByText(/Only displayed sessions are available to select/)).not.toBeNull()
  })

  it('keeps start disabled when the selected session exceeds configured model capacity', async () => {
    api.getSession.mockResolvedValue({
      judging_token_estimates: {
        utf8_bytes_div_3: { total_tokens: 100_000, largest_turn_tokens: 90_000, turn_tokens: [90_000, 10_000] },
        o200k_base: { total_tokens: 90_000, largest_turn_tokens: 80_000, turn_tokens: [80_000, 10_000] },
        o200k_harmony: { total_tokens: 85_000, largest_turn_tokens: 75_000, turn_tokens: [75_000, 10_000] },
      },
    })
    renderPage()

    fireEvent.click(await screen.findByRole('checkbox', {
      name: 'Select session session-1',
    }))

    await waitFor(() => expect(screen.getByRole('button', {
      name: 'Start Scoring',
    })).toHaveProperty('disabled', true))
    expect(await screen.findByRole('alert')).toHaveProperty(
      'textContent',
      expect.stringContaining('Select models that fit the selected sessions.'),
    )
  })

  it('retries session discovery and model catalogs in place', async () => {
    api.getModels
      .mockRejectedValueOnce(new Error('Model catalog unavailable'))
      .mockResolvedValue(models)
    api.getSessions
      .mockRejectedValueOnce(new Error('Session discovery unavailable'))
      .mockResolvedValue({ sessions: [session], total: 1, truncated: false })
    renderPage()

    fireEvent.click(await screen.findByRole('button', { name: 'Retry configuration catalogs' }))
    await waitFor(() => expect(api.getModels).toHaveBeenCalledTimes(2))
    fireEvent.click(await screen.findByRole('button', { name: 'Retry session discovery' }))
    await waitFor(() => expect(api.getSessions).toHaveBeenCalledTimes(2))
    expect(await screen.findByRole('checkbox', { name: 'Select session session-1' })).not.toBeNull()
  })

  it('supports manual advance and confirmed cancellation', async () => {
    const scoring = baseRun({
      status: 'scoring',
      current_stage_succeeded: true,
      run_config: requestedConfig,
      effective_config: effectiveConfig,
      scoring_result: {
        turns_scored: 2,
        sessions_scored: 1,
        scores_written: 4,
        errors: 0,
        turn_details: [],
      },
    })
    api.getRun.mockResolvedValue(scoring)
    api.advanceRun.mockResolvedValue({ ...scoring, status: 'judging' })
    api.cancelRun.mockResolvedValue({ ...scoring, status: 'cancelled' })
    renderPage()

    fireEvent.click(await screen.findByRole('button', { name: 'Continue to judging' }))
    await waitFor(() => expect(api.advanceRun).toHaveBeenCalledWith('run-ui'))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel Run' }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel run' }))
    await waitFor(() => expect(api.cancelRun).toHaveBeenCalledWith('run-ui'))
  })

  it('composes pinned audit and active judging progress', async () => {
    api.getRun.mockResolvedValue(
      baseRun({
        status: 'judging',
        data_selection: {
          since: '2026-07-01T07:00:00Z',
          until: '2026-07-15T06:59:59.999999Z',
          timezone: 'America/Los_Angeles',
          session_ids: [
            'session-1', 'session-2', 'session-3', 'session-4',
            'session-5', 'session-6', 'session-7',
          ],
        },
        run_config: requestedConfig,
        effective_config: effectiveConfig,
        turn_cohort: {
          schema_version: '2',
          cohort_id: 'sha256:cohort-123',
          pinned_at: '2026-07-15T07:30:00Z',
          turn_count: 12,
          session_count: 7,
          turns: [],
          sessions: [],
        },
        judging_plan: plan,
        judging_progress: {
          plan_id: plan.plan_id,
          planned_rubrics: 1,
          rubrics_completed: 1,
          rated_rubrics: 1,
          not_evaluable_rubrics: 0,
          minimum_reviewer_attempts: 2,
          maximum_reviewer_attempts: 2,
          reviewer_attempts_completed: 2,
          digest_steps_completed: 2,
          maximum_digest_steps: 2,
          window_steps_completed: 1,
          maximum_window_steps: 2,
          merge_steps_completed: 1,
          maximum_merge_steps: 2,
          scores_written: 0,
          failure_count: 0,
          write_failure_count: 0,
          coverage_complete: false,
          phase: 'score_writes_started',
          status_message: 'Writing 1 judge score...',
          started_at: '2026-07-15T07:30:00Z',
          events: [],
          attempt_summaries: [],
          attempt_summary_count: 0,
          attempt_summaries_truncated: false,
          failure_details: [],
          failure_detail_count: 0,
          failure_details_truncated: false,
        },
      }),
    )
    renderPage()

    const selectionStage = await screen.findByRole('button', {
      name: 'Data selection & configuration',
    })
    expect(selectionStage.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(selectionStage)
    expect(screen.getByRole('region', { name: 'Pinned selection and cohort' })).not.toBeNull()
    expect(screen.getByText('Jul 1, 2026 – Jul 14, 2026')).not.toBeNull()
    expect(screen.getByText('America/Los_Angeles')).not.toBeNull()
    expect(screen.getByText('sha256:cohort-123')).not.toBeNull()
    expect(screen.getByText('12 turns across 7 sessions')).not.toBeNull()
    expect(screen.getByText('7 selected session IDs')).not.toBeNull()
    expect(screen.getByText('session-7')).not.toBeNull()
    expect(screen.getByRole('region', { name: 'Pinned run configuration' })).not.toBeNull()
    expect(screen.getByRole('region', { name: 'Judging progress' })).not.toBeNull()
    expect(screen.getByText('1 of 1 rubrics reviewed')).not.toBeNull()
    expect(screen.getByText('2 reviewer attempts so far')).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Judging' }).getAttribute('aria-expanded')).toBe('true')
  })

  it('collapses completed stages but keeps a pending reflection decision visible', async () => {
    api.getRun.mockResolvedValue(completedReflectionRun())
    renderPage()

    expect((await screen.findByRole('button', {
      name: 'Data selection & configuration',
    })).getAttribute('aria-expanded')).toBe('false')
    expect(screen.getByRole('button', { name: 'Scoring' }).getAttribute('aria-expanded')).toBe('false')
    expect(screen.getByRole('button', { name: 'Judging' }).getAttribute('aria-expanded')).toBe('false')
    expect(screen.getByRole('button', { name: 'Reflecting' }).getAttribute('aria-expanded')).toBe('true')
    expect(screen.getByRole('region', { name: 'Reflection decision' })).not.toBeNull()
  })

  it('protects unsaved D from route navigation and browser unload', async () => {
    api.getRun.mockResolvedValue(completedReflectionRun())
    const { router } = renderPage()

    fireEvent.click(await screen.findByRole('button', { name: 'Edit proposal inline' }))
    fireEvent.change(screen.getByLabelText('Edit CLAUDE.md'), {
      target: { value: 'Unsaved edited D' },
    })

    const unload = new Event('beforeunload', { cancelable: true })
    window.dispatchEvent(unload)
    expect(unload.defaultPrevented).toBe(true)

    void router.navigate('/runs')
    expect(await screen.findByRole('dialog', {
      name: 'Leave without saving edited D?',
    })).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Keep editing' }))
    expect(screen.getByDisplayValue('Unsaved edited D')).not.toBeNull()

    void router.navigate('/runs')
    fireEvent.click(await screen.findByRole('button', { name: 'Leave without saving' }))
    expect(await screen.findByText('Run list')).not.toBeNull()
  })

  it('keeps cached run evidence and unsaved D visible when a background refresh fails', async () => {
    api.getRun
      .mockResolvedValueOnce(completedReflectionRun())
      .mockRejectedValueOnce(new Error('Run refresh unavailable'))
    const { client } = renderPage()

    fireEvent.click(await screen.findByRole('button', { name: 'Edit proposal inline' }))
    fireEvent.change(screen.getByLabelText('Edit CLAUDE.md'), {
      target: { value: 'Cached unsaved D' },
    })
    await act(() => client.refetchQueries({ queryKey: ['run', 'run-ui'] }))

    expect(await screen.findByRole('alert')).toHaveProperty(
      'textContent',
      expect.stringContaining('Could not refresh run: Run refresh unavailable'),
    )
    expect(screen.getByDisplayValue('Cached unsaved D')).not.toBeNull()
    expect(screen.getByRole('region', { name: 'Reflection review' })).not.toBeNull()
  })

  it('composes completed reflection evidence and decision controls', async () => {
    const completed = completedReflectionRun()
    api.getRun.mockResolvedValue(completed)
    api.promoteRunReflection.mockResolvedValue(completed)
    renderPage()

    expect(await screen.findByRole('region', { name: 'Reflection review' })).not.toBeNull()
    expect(screen.getByText('Review needed')).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Promote evaluated C' }))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm promotion' }))

    await waitFor(() => expect(api.promoteRunReflection).toHaveBeenCalledWith('run-ui', {
      expectedRevision: 1,
      expectedDraftRevision: null,
      acknowledgeUnevaluated: false,
      idempotencyKey: 'run-ui:1:evaluated',
    }))
  })

  it('shows the exact actionable terminal reflection error without an empty logs panel', async () => {
    const message = 'Proposal evaluator failed: W&B credits are disabled'
    api.getRun.mockResolvedValue(baseRun({
      status: 'failed',
      error: message,
      reflecting_progress: {
        phase: 'reflection_failed',
        status_message: message,
        started_at: '2026-07-14T19:20:00+00:00',
        proposal_writer: 'gpt-5.6-sol',
        proposal_evaluator: 'claude-sonnet-5',
        no_improvement_patience: 2,
        attempted: 1,
        valid: 1,
        rejected: 0,
        scored: 0,
        total_attempts: 3,
        events: [],
      },
    }))
    renderPage()

    expect(await screen.findByText(message)).not.toBeNull()
    expect(screen.queryByText(`Last update before failure: ${message}`)).toBeNull()
    expect(screen.queryByText('Logs')).toBeNull()
  })

  it('routes an all-invalid finalized reflection through the full review evidence view', async () => {
    const completed = completedReflectionRun()
    const successful = completed.reflecting_result!
    if (!('baseline' in successful)) throw new Error('fixture must be successful')
    api.getRun.mockResolvedValue({
      ...completed,
      reflecting_result: {
        ...successful,
        candidates: [],
        recommended_candidate_id: null,
        baseline_won: false,
        reason: 'No valid proposal generated',
      },
      reflection_review: null,
    })
    renderPage()

    fireEvent.click(await screen.findByRole('button', { name: 'Reflecting' }))
    expect(await screen.findByRole('region', { name: 'Reflection review' })).not.toBeNull()
    expect(screen.getByText(/No proposed C was evaluated/i)).not.toBeNull()
    expect(screen.getAllByText('Baseline evidence.').length).toBeGreaterThan(0)
  })

  it('shows API conflicts and requeries durable run state', async () => {
    const scoring = baseRun({
      status: 'scoring',
      current_stage_succeeded: true,
      run_config: requestedConfig,
      effective_config: effectiveConfig,
      scoring_result: {
        turns_scored: 2,
        sessions_scored: 1,
        scores_written: 4,
        errors: 0,
        turn_details: [],
      },
    })
    api.getRun
      .mockResolvedValueOnce(scoring)
      .mockResolvedValue({ ...scoring, status: 'judging' })
    api.advanceRun.mockRejectedValue(
      new ApiError(409, '', { message: 'Run advanced in another tab.' }),
    )
    renderPage()

    fireEvent.click(await screen.findByRole('button', { name: 'Continue to judging' }))

    expect(await screen.findByRole('alert')).toHaveProperty(
      'textContent',
      expect.stringContaining('Run advanced in another tab.'),
    )
    await waitFor(() => expect(api.getRun).toHaveBeenCalledTimes(2))
  })
})
