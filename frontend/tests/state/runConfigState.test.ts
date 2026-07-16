import { describe, expect, it } from 'vitest'
import type { ModelCatalog, RubricCatalog, RunConfig } from '../../src/types'
import {
  assessRunConfig,
  initializeRunConfigState,
  toRunConfig,
  transitionRunConfig,
} from '../../src/features/runs/runConfigState'

const writerOpenAI = {
  id: 'writer-openai',
  label: 'Writer OpenAI',
  family: 'openai',
  backend: 'cli',
  supported_roles: ['proposal_writer'] as const,
  max_input_tokens: 128_000,
}

const writerAnthropic = {
  id: 'writer-anthropic',
  label: 'Writer Anthropic',
  family: 'anthropic',
  backend: 'cli',
  supported_roles: ['proposal_writer'] as const,
  max_input_tokens: 128_000,
}

const cliAnthropic = {
  id: 'cli-anthropic',
  label: 'CLI Anthropic',
  family: 'anthropic',
  backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
}

const cliOpenAI = {
  id: 'cli-openai',
  label: 'CLI OpenAI',
  family: 'openai',
  backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
}

const cliMeta = {
  id: 'cli-meta',
  label: 'CLI Meta',
  family: 'meta',
  backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
}

const cliMetaAlt = {
  id: 'cli-meta-alt',
  label: 'CLI Meta Alternate',
  family: 'meta',
  backend: 'cli',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
}

const wandbMeta = {
  id: 'wandb-meta',
  label: 'W&B Meta',
  family: 'meta',
  backend: 'wandb',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
}

const wandbOpenAI = {
  id: 'wandb-openai',
  label: 'W&B OpenAI',
  family: 'openai',
  backend: 'wandb',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
}

const wandbIbm = {
  id: 'wandb-ibm',
  label: 'W&B IBM',
  family: 'ibm',
  backend: 'wandb',
  supported_roles: ['judge', 'proposal_evaluator'] as const,
  max_input_tokens: 128_000,
}

const models: ModelCatalog = {
  catalog_version: 'sha256:models',
  proposal: {
    available_models: [writerOpenAI, writerAnthropic],
    recommended_model: writerOpenAI.id,
  },
  review_defaults: { second_opinion_margin: 0.1 },
  recommended_judge_backend: 'cli',
  judge_backends: {
    cli: {
      available_models: [cliAnthropic, cliOpenAI, cliMeta, cliMetaAlt],
      recommended_judges: [cliAnthropic.id, cliOpenAI.id, cliMeta.id],
      proposal_evaluator_preferences: [cliOpenAI.id, cliAnthropic.id, cliMeta.id],
      recommended_review_depth: 'selective',
      supported_review_depths: ['primary', 'selective', 'full_panel'],
    },
    wandb: {
      available_models: [wandbMeta, wandbOpenAI, wandbIbm],
      recommended_judges: [wandbMeta.id, wandbOpenAI.id, wandbIbm.id],
      proposal_evaluator_preferences: [wandbOpenAI.id, wandbMeta.id, wandbIbm.id],
      recommended_review_depth: 'selective',
      supported_review_depths: ['primary', 'selective', 'full_panel'],
    },
  },
}

const rubrics: RubricCatalog = {
  catalog_version: 'sha256:rubrics',
  rubrics: [
    {
      id: 'judge.verification',
      label: 'Verification discipline',
      evaluation_unit: 'session',
      version: 'v1',
      content_digest: 'sha256:verification',
      pass_threshold: 0.5,
    },
    {
      id: 'judge.session_outcome',
      label: 'Session outcome',
      evaluation_unit: 'session',
      version: 'v1',
      content_digest: 'sha256:session',
      pass_threshold: 0.5,
    },
  ],
}

function recommendedState() {
  return initializeRunConfigState(models, rubrics)
}

describe('guided run configuration state', () => {
  it('expands every initial recommendation into visible choices', () => {
    const state = recommendedState()

    expect(state).toEqual({
      proposalModel: { value: writerOpenAI.id, source: 'recommended' },
      judgeBackend: { value: 'cli', source: 'recommended' },
      reviewDepth: { value: 'selective', source: 'recommended' },
      judgeModels: {
        value: [cliAnthropic.id, cliOpenAI.id, cliMeta.id],
        source: 'recommended',
      },
      proposalEvaluatorModel: { value: cliAnthropic.id, source: 'automatic' },
      secondOpinionMargin: 0.1,
      rubricIds: ['judge.verification', 'judge.session_outcome'],
      candidateBudget: 3,
      force: false,
    })
    expect(toRunConfig(state, models, rubrics)).toEqual({
      model_catalog_version: models.catalog_version,
      rubric_catalog_version: rubrics.catalog_version,
      judge_backend: 'cli',
      review_depth: 'selective',
      judge_models: [cliAnthropic.id, cliOpenAI.id, cliMeta.id],
      second_opinion_margin: 0.1,
      proposal_model: writerOpenAI.id,
      proposal_evaluator_model: cliAnthropic.id,
      rubrics: ['judge.verification', 'judge.session_outcome'],
      candidate_budget: 3,
      force: false,
    })
  })

  it('labels explicit choices and restores all recommendations', () => {
    let state = recommendedState()
    state = transitionRunConfig(state, { type: 'select-judge', position: 1, modelId: cliMeta.id }, models)
    state = transitionRunConfig(state, { type: 'select-evaluator', modelId: cliMeta.id }, models)
    state = transitionRunConfig(state, { type: 'set-rubrics', rubricIds: ['judge.verification'] }, models)
    state = transitionRunConfig(state, { type: 'set-budget', value: 5 }, models)
    state = transitionRunConfig(state, { type: 'set-force', value: true }, models)

    expect(state.judgeModels.source).toBe('overridden')
    expect(state.proposalEvaluatorModel).toEqual({ value: cliMeta.id, source: 'overridden' })
    expect(state.rubricIds).toEqual(['judge.verification'])
    expect(state.candidateBudget).toBe(5)
    expect(state.force).toBe(true)

    const restored = transitionRunConfig(state, { type: 'use-recommended' }, models)
    expect(restored.proposalModel).toEqual(recommendedState().proposalModel)
    expect(restored.judgeBackend).toEqual(recommendedState().judgeBackend)
    expect(restored.reviewDepth).toEqual(recommendedState().reviewDepth)
    expect(restored.judgeModels).toEqual(recommendedState().judgeModels)
    expect(restored.proposalEvaluatorModel).toEqual(recommendedState().proposalEvaluatorModel)
    expect(restored.secondOpinionMargin).toBe(recommendedState().secondOpinionMargin)
    expect(restored.rubricIds).toEqual(['judge.verification'])
    expect(restored.candidateBudget).toBe(5)
    expect(restored.force).toBe(true)
  })

  it('changes backend without changing the independent writer or compatible policy fields', () => {
    let state = transitionRunConfig(
      recommendedState(),
      { type: 'select-writer', modelId: writerAnthropic.id },
      models,
    )
    state = transitionRunConfig(state, { type: 'set-margin', value: 0.2 }, models)
    state = transitionRunConfig(state, { type: 'select-backend', backend: 'wandb' }, models)

    expect(state.proposalModel).toEqual({ value: writerAnthropic.id, source: 'overridden' })
    expect(state.judgeBackend).toEqual({ value: 'wandb', source: 'overridden' })
    expect(state.reviewDepth).toEqual({ value: 'selective', source: 'recommended' })
    expect(state.secondOpinionMargin).toBe(0.2)
    expect(state.judgeModels).toEqual({
      value: [wandbMeta.id, wandbOpenAI.id, wandbIbm.id],
      source: 'recommended',
    })
    expect(state.proposalEvaluatorModel).toEqual({
      value: wandbOpenAI.id,
      source: 'automatic',
    })
  })

  it('uses the new backend recommendation unless review depth was explicitly overridden', () => {
    const primaryWandb: ModelCatalog = {
      ...models,
      judge_backends: {
        ...models.judge_backends,
        wandb: {
          ...models.judge_backends.wandb,
          recommended_judges: [wandbMeta.id],
          recommended_review_depth: 'primary',
        },
      },
    }
    const recommended = transitionRunConfig(
      initializeRunConfigState(primaryWandb, rubrics),
      { type: 'select-backend', backend: 'wandb' },
      primaryWandb,
    )
    expect(recommended.reviewDepth).toEqual({ value: 'primary', source: 'recommended' })
    expect(recommended.judgeModels).toEqual({ value: [wandbMeta.id], source: 'recommended' })

    let overridden = transitionRunConfig(
      recommendedState(),
      { type: 'select-depth', depth: 'primary' },
      models,
    )
    overridden = transitionRunConfig(
      overridden,
      { type: 'select-backend', backend: 'wandb' },
      models,
    )
    expect(overridden.reviewDepth).toEqual({ value: 'primary', source: 'overridden' })
    expect(overridden.judgeModels.value).toEqual([wandbMeta.id])
  })

  it('recomputes an automatic evaluator for a writer change but preserves an explicit one', () => {
    const automatic = transitionRunConfig(
      recommendedState(),
      { type: 'select-writer', modelId: writerAnthropic.id },
      models,
    )
    expect(automatic.proposalEvaluatorModel).toEqual({
      value: cliOpenAI.id,
      source: 'automatic',
    })

    let overridden = transitionRunConfig(
      recommendedState(),
      { type: 'select-evaluator', modelId: cliMeta.id },
      models,
    )
    overridden = transitionRunConfig(
      overridden,
      { type: 'select-writer', modelId: writerAnthropic.id },
      models,
    )
    expect(overridden.proposalEvaluatorModel).toEqual({
      value: cliMeta.id,
      source: 'overridden',
    })
  })

  it('normalizes judge cardinality and margin across depth transitions', () => {
    let state = transitionRunConfig(
      recommendedState(),
      { type: 'select-depth', depth: 'primary' },
      models,
    )
    expect(state.reviewDepth).toEqual({ value: 'primary', source: 'overridden' })
    expect(state.judgeModels).toEqual({ value: [cliAnthropic.id], source: 'automatic' })
    expect(state.secondOpinionMargin).toBeNull()

    state = transitionRunConfig(state, { type: 'select-depth', depth: 'full_panel' }, models)
    expect(state.judgeModels).toEqual({
      value: [cliAnthropic.id, cliOpenAI.id, cliMeta.id],
      source: 'automatic',
    })
    expect(state.secondOpinionMargin).toBeNull()

    state = transitionRunConfig(state, { type: 'select-depth', depth: 'selective' }, models)
    expect(state.secondOpinionMargin).toBe(0.1)
    state = transitionRunConfig(state, { type: 'remove-third-judge' }, models)
    expect(state.judgeModels).toEqual({
      value: [cliAnthropic.id, cliOpenAI.id],
      source: 'overridden',
    })
  })

  it('initializes saved choices without inventing runtime defaults', () => {
    const saved: RunConfig = {
      model_catalog_version: models.catalog_version,
      rubric_catalog_version: rubrics.catalog_version,
      judge_backend: 'cli',
      review_depth: 'selective',
      judge_models: [cliAnthropic.id, cliMeta.id],
      second_opinion_margin: 0.25,
      proposal_model: writerAnthropic.id,
      proposal_evaluator_model: cliMeta.id,
      rubrics: ['judge.session_outcome'],
      candidate_budget: 4,
      force: true,
    }

    const state = initializeRunConfigState(models, rubrics, saved)

    expect(state.proposalModel.source).toBe('overridden')
    expect(state.judgeModels.source).toBe('overridden')
    expect(state.proposalEvaluatorModel.source).toBe('overridden')
    expect(toRunConfig(state, models, rubrics)).toEqual(saved)
  })

  it('blocks invalid selections but keeps family overlap as warnings only', () => {
    let state = recommendedState()
    state = transitionRunConfig(
      state,
      { type: 'select-judge', position: 2, modelId: cliAnthropic.id },
      models,
    )
    state = transitionRunConfig(
      state,
      { type: 'select-evaluator', modelId: cliOpenAI.id },
      models,
    )

    const duplicate = assessRunConfig(state, models, rubrics)
    expect(duplicate.errors).toContain('Judge selections must be unique.')
    expect(() => toRunConfig(state, models, rubrics)).toThrow('Judge selections must be unique.')

    state = transitionRunConfig(
      recommendedState(),
      { type: 'select-judge', position: 2, modelId: cliMetaAlt.id },
      models,
    )
    const overlap = assessRunConfig(state, models, rubrics)
    expect(overlap.errors).toEqual([])
    expect(overlap.warnings).toContain('Multiple selected judges share the meta model family.')
  })

  it('validates unsupported model IDs, rubrics, margins, and budgets at the boundary', () => {
    const state = {
      ...recommendedState(),
      proposalModel: { value: 'raw-unlisted-model', source: 'overridden' as const },
      secondOpinionMargin: 0.75,
      rubricIds: ['judge.unknown'],
      candidateBudget: 0,
    }

    expect(assessRunConfig(state, models, rubrics).errors).toEqual([
      'Select an available proposal writer.',
      'Second-opinion margin must be between 0 and 0.5.',
      'Select only available rubrics.',
      'Proposal attempt limit must be at least 1.',
    ])
  })

  it('requires the form to show at least one explicit rubric selection', () => {
    const state = {
      ...recommendedState(),
      rubricIds: [],
    }

    expect(assessRunConfig(state, models, rubrics).errors).toContain(
      'Select at least one rubric.',
    )
  })

  it('rejects proposal attempt limits above the server cap', () => {
    const state = { ...recommendedState(), candidateBudget: 11 }

    expect(assessRunConfig(state, models, rubrics).errors).toContain(
      'Proposal attempt limit must be between 1 and 10.',
    )
    expect(() => toRunConfig(state, models, rubrics)).toThrow(
      'Proposal attempt limit must be between 1 and 10.',
    )
  })
})
