import { describe, expect, it } from 'vitest'
import type { ModelCatalog, ModelDescriptor, RubricCatalog, RunConfig } from '../../src/types'
import {
  assessRunConfig,
  initializeRunConfigState,
  toRunConfig,
  transitionRunConfig,
} from '../../src/features/runs/runConfigState'

const writer: ModelDescriptor = {
  id: 'writer', label: 'Writer', family: 'openai', backend: 'cli',
  supported_roles: ['proposal_writer'], max_input_tokens: 128_000,
}
const anthropic: ModelDescriptor = {
  id: 'anthropic', label: 'Anthropic', family: 'anthropic', backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'], max_input_tokens: 128_000,
}
const openai: ModelDescriptor = {
  id: 'openai', label: 'OpenAI', family: 'openai', backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'], max_input_tokens: 128_000,
}
const meta: ModelDescriptor = {
  id: 'meta', label: 'Meta', family: 'meta', backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'], max_input_tokens: 128_000,
}
const metaAlt = { ...meta, id: 'meta-alt', label: 'Meta alternate' }

const models: ModelCatalog = {
  catalog_version: 'sha256:models',
  proposal: { available_models: [writer], recommended_model: writer.id },
  recommended_judge_backend: 'cli',
  judge_backends: {
    cli: {
      available_models: [anthropic, openai, meta, metaAlt],
      recommended_judges: [anthropic.id, openai.id, meta.id],
      proposal_evaluator_preferences: [anthropic.id, openai.id, meta.id],
    },
  },
}

const rubrics: RubricCatalog = {
  catalog_version: 'sha256:rubrics',
  rubrics: [{
    id: 'judge.verification', label: 'Verification', evaluation_unit: 'session',
    version: 'v1', content_digest: 'sha256:verification', pass_threshold: 0.5,
  }],
}

describe('judge panel configuration state', () => {
  it('expands the recommended three-judge panel into the saved contract', () => {
    const state = initializeRunConfigState(models, rubrics)

    expect(state.judgeModels).toEqual({
      value: [anthropic.id, openai.id, meta.id],
      source: 'recommended',
    })
    expect(toRunConfig(state, models, rubrics)).toEqual({
      model_catalog_version: models.catalog_version,
      rubric_catalog_version: rubrics.catalog_version,
      judge_backend: 'cli',
      judge_models: [anthropic.id, openai.id, meta.id],
      proposal_model: writer.id,
      proposal_evaluator_model: anthropic.id,
      rubrics: ['judge.verification'],
      candidate_budget: 3,
      force: false,
    })
  })

  it('allows one through three ordered unique judges', () => {
    let state = initializeRunConfigState(models, rubrics)
    state = transitionRunConfig(state, { type: 'remove-last-judge' }, models)
    state = transitionRunConfig(state, { type: 'remove-last-judge' }, models)
    expect(state.judgeModels.value).toEqual([anthropic.id])
    expect(assessRunConfig(state, models, rubrics).errors).toEqual([])

    state = transitionRunConfig(
      state,
      { type: 'select-judge', position: 2, modelId: openai.id },
      models,
    )
    expect(state.judgeModels.value).toEqual([anthropic.id, openai.id])
  })

  it('rejects duplicate or out-of-range panels and keeps family overlap as a warning', () => {
    let duplicate = initializeRunConfigState(models, rubrics)
    duplicate = transitionRunConfig(
      duplicate,
      { type: 'select-judge', position: 2, modelId: anthropic.id },
      models,
    )
    expect(assessRunConfig(duplicate, models, rubrics).errors).toContain(
      'Judge selections must be unique.',
    )

    let overlap = initializeRunConfigState(models, rubrics)
    overlap = transitionRunConfig(
      overlap,
      { type: 'select-judge', position: 2, modelId: metaAlt.id },
      models,
    )
    expect(assessRunConfig(overlap, models, rubrics).warnings).toContain(
      'Multiple selected judges share the meta model family.',
    )
  })

  it('round trips a saved explicit panel without runtime defaults', () => {
    const saved: RunConfig = {
      model_catalog_version: models.catalog_version,
      rubric_catalog_version: rubrics.catalog_version,
      judge_backend: 'cli',
      judge_models: [openai.id],
      proposal_model: writer.id,
      proposal_evaluator_model: anthropic.id,
      rubrics: ['judge.verification'],
      candidate_budget: 4,
      force: true,
    }

    expect(toRunConfig(initializeRunConfigState(models, rubrics, saved), models, rubrics))
      .toEqual(saved)
  })
})
