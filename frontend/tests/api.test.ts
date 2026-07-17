import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  advanceRun,
  createRun,
  getModels,
  getRuns,
  getRubrics,
  promoteRunReflection,
  saveReflectionDraft,
  setRunConfig,
  setReflectionSelection,
} from '../src/api'
import type {
  EffectiveRunConfig,
  ModelDescriptor,
  ModelCatalog,
  RubricCatalog,
  Run,
  RunSummary,
  RunConfig,
} from '../src/types'

afterEach(() => {
  vi.unstubAllGlobals()
})

function mockFetch(payload: unknown = {}) {
  const fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: vi.fn().mockResolvedValue(payload),
  })
  vi.stubGlobal('fetch', fetch)
  return fetch
}

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

const judge: ModelDescriptor = {
  id: 'judge-anthropic',
  label: 'Judge Anthropic',
  family: 'anthropic',
  provider: 'claude',
  provider_model: 'judge-anthropic',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}

const rubric = {
  id: 'judge.verification',
  label: 'Verification discipline',
  evaluation_unit: 'session' as const,
  version: 'v1',
  content_digest: 'sha256:rubric',
  pass_threshold: 0.5,
}

const modelCatalog: ModelCatalog = {
  catalog_version: 'sha256:models',
  available_models: [writer, judge],
  recommended_proposal_model: writer.id,
  recommended_judges: [judge.id],
  recommended_challenge_judges: [judge.id],
  proposal_evaluator_preferences: [judge.id],
}

const rubricCatalog: RubricCatalog = {
  catalog_version: 'sha256:rubrics',
  rubrics: [rubric],
}

const runConfig: RunConfig = {
  model_catalog_version: modelCatalog.catalog_version,
  rubric_catalog_version: rubricCatalog.catalog_version,
  judge_models: [judge.id],
  challenge_judge_models: [judge.id],
  proposal_model: writer.id,
  proposal_evaluator_model: judge.id,
  rubrics: [],
  candidate_budget: 3,
  force: false,
}

const effectiveConfig: EffectiveRunConfig = {
  schema_version: '5',
  pipeline_version: '8',
  model_catalog_version: modelCatalog.catalog_version,
  rubric_catalog_version: rubricCatalog.catalog_version,
  models: {
    proposal_writer: writer,
    judges: [{ ...judge, role: 'judge', position: 1 }],
    challenge_judges: [{ ...judge, role: 'judge', position: 1 }],
    proposal_evaluator: judge,
  },
  rubrics: [rubric],
  selection_warnings: [
    {
      code: 'proposal_evaluator_writer_family_overlap',
      message: 'Writer and evaluator share a family.',
      affected_roles: ['proposal_writer', 'proposal_evaluator'],
      selected_model_ids: [writer.id, judge.id],
      compared_families: ['openai'],
    },
  ],
  judging_context: {
    contract_version: '3', large_model_threshold_tokens: 200_000,
    large_model_reserve_tokens: 100_000, small_model_reserve_tokens: 50_000,
    large_model_raw_target_tokens: 128_000, small_model_raw_target_tokens: 50_000,
    prompt_reserve_tokens: 6_000,
    output_reserve_tokens: 4_000, safety_reserve_tokens: 8_000, digest_max_tokens: 1_000,
    finding_max_tokens: 4_000, overlap_turns: 1, max_chunks: 40,
  },
  candidate_budget: 3,
  force: false,
}

const effectiveRunFixture: Run = {
  run_id: 'run-effective',
  status: 'scoring',
  current_stage_succeeded: false,
  created_at: '2026-07-14T18:00:00Z',
  auto_run: false,
  data_selection: null,
  run_config: runConfig,
  effective_config: effectiveConfig,
  turn_cohort: null,
  judging_plan: null,
  reflection_input: null,
  scoring_progress: null,
  scoring_result: null,
  judging_progress: null,
  judging_result: null,
  reflecting_progress: null,
  reflecting_result: null,
  reflection_review: null,
  reflection_review_revision: 0,
  error: null,
}

describe('run API request contracts', () => {
  it('preserves structured API error details for conflict handling', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      text: vi.fn().mockResolvedValue(JSON.stringify({
        detail: { message: 'The reviewed bundle changed', revision: 7 },
      })),
    }))

    await expect(getRuns()).rejects.toEqual(expect.objectContaining({
      name: 'ApiError',
      status: 409,
      message: 'API 409: The reviewed bundle changed',
      detail: { message: 'The reviewed bundle changed', revision: 7 },
    }))
  })

  it('reads the compact run-list contract without requiring detail fields', async () => {
    const summary: RunSummary = {
      run_id: 'run-summary',
      status: 'complete',
      current_stage_succeeded: true,
      created_at: '2026-07-14T18:00:00Z',
      selection: {
        session_count: 4,
        since: '2026-07-01T00:00:00Z',
        until: '2026-07-14T23:59:59Z',
        timezone: 'America/Los_Angeles',
      },
      review_state: 'no-valid-proposal',
    }
    mockFetch({ runs: [summary] })

    await expect(getRuns()).resolves.toEqual({ runs: [summary] })
  })
  it('returns the exact versioned model and rubric catalogs', async () => {
    const fetch = mockFetch(modelCatalog)

    await expect(getModels()).resolves.toEqual(modelCatalog)
    expect(fetch).toHaveBeenLastCalledWith('/api/models', undefined)

    fetch.mockResolvedValueOnce({
      ok: true,
      json: vi.fn().mockResolvedValue(rubricCatalog),
    })
    await expect(getRubrics()).resolves.toEqual(rubricCatalog)
    expect(fetch).toHaveBeenLastCalledWith('/api/rubrics', undefined)
  })

  it('sends one complete explicit run configuration', async () => {
    const fetch = mockFetch(effectiveRunFixture)

    await setRunConfig('run-configured', runConfig)

    expect(fetch).toHaveBeenCalledWith('/api/runs/run-configured/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(runConfig),
    })
  })

  it('creates a run without a request body', async () => {
    const fetch = mockFetch()

    await createRun()

    expect(fetch).toHaveBeenCalledWith('/api/runs', {
      method: 'POST',
    })
  })

  it('omits the advance body so the backend uses the persisted run configuration', async () => {
    const fetch = mockFetch()

    await advanceRun('run-configured')

    expect(fetch).toHaveBeenCalledWith('/api/runs/run-configured/advance', {
      method: 'POST',
    })
  })

  it('selects reflection candidates only by stable candidate ID', async () => {
    const fetch = mockFetch()

    await setReflectionSelection('run-review', 'candidate-stable', 4, true)

    expect(fetch).toHaveBeenCalledWith('/api/runs/run-review/reflection_selection', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        candidate_id: 'candidate-stable',
        expected_revision: 4,
        discard_draft: true,
      }),
    })
  })

  it('sends the unevaluated-D acknowledgement to the promotion endpoint', async () => {
    const fetch = mockFetch()

    await promoteRunReflection('run-review', {
      expectedRevision: 5,
      expectedDraftRevision: 'draft:d',
      idempotencyKey: 'promotion-1',
      acknowledgeUnevaluated: true,
    })

    expect(fetch).toHaveBeenCalledWith('/api/runs/run-review/promote', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        expected_revision: 5,
        expected_draft_revision: 'draft:d',
        idempotency_key: 'promotion-1',
        acknowledge_unevaluated: true,
      }),
    })
  })

  it('sends draft content and leaves canonical revision hashing to the server', async () => {
    const fetch = mockFetch()
    const contents = {
      'CLAUDE.md': '# Edited D',
      '.claude/skills/removed.md': null,
    }

    await saveReflectionDraft('run-review', contents, 3, 'sha256:old-draft')

    expect(fetch).toHaveBeenCalledWith('/api/runs/run-review/reflection_draft', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        expected_revision: 3,
        expected_draft_revision: 'sha256:old-draft',
        contents,
      }),
    })
  })
})
