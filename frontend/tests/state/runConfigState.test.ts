import { describe, expect, it } from 'vitest'
import type { ModelCatalog, ModelDescriptor, RubricCatalog, RunConfig } from '../../src/types'
import {
  assessRunConfig,
  initializeRunConfigState,
  toRunConfig,
  transitionRunConfig,
} from '../../src/features/runs/runConfigState'

const writer: ModelDescriptor = {
  id: 'codex:writer', label: 'Writer', family: 'openai', provider: 'codex', provider_model: 'writer',
  supported_roles: ['proposal_writer'], max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const anthropic: ModelDescriptor = {
  id: 'claude:anthropic', label: 'Anthropic', family: 'anthropic', provider: 'claude', provider_model: 'anthropic',
  supported_roles: ['judge', 'proposal_evaluator'], max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const openai: ModelDescriptor = {
  id: 'codex:openai', label: 'OpenAI', family: 'openai', provider: 'codex', provider_model: 'openai',
  supported_roles: ['judge', 'proposal_evaluator'], max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const meta: ModelDescriptor = {
  id: 'agy:meta', label: 'Meta', family: 'meta', provider: 'agy', provider_model: 'meta',
  supported_roles: ['judge', 'proposal_evaluator'], max_input_tokens: 128_000,
  token_counter: 'utf8_bytes_div_3',
}
const metaAlt = { ...meta, id: 'meta-alt', label: 'Meta alternate' }

const models: ModelCatalog = {
  catalog_version: 'sha256:models',
  judging_context: {
    contract_version: '3', large_model_threshold_tokens: 200_000,
    large_model_reserve_tokens: 100_000, small_model_reserve_tokens: 50_000,
    large_model_raw_target_tokens: 128_000, small_model_raw_target_tokens: 50_000,
    prompt_reserve_tokens: 6_000, output_reserve_tokens: 4_000,
    large_model_output_reserve_tokens: 10_000,
    safety_reserve_tokens: 8_000, digest_max_tokens: 1_000,
    finding_max_tokens: 4_000, overlap_turns: 1, max_chunks: 40,
  },
  available_models: [writer, anthropic, openai, meta, metaAlt],
  recommended_proposal_model: writer.id,
  recommended_judges: [anthropic.id, openai.id, meta.id],
  recommended_challenge_judges: [openai.id],
  proposal_evaluator_preferences: [anthropic.id, openai.id, meta.id],
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
    expect(state.challengeJudgeModels).toEqual({
      value: [openai.id],
      source: 'recommended',
    })
    expect(toRunConfig(state, models, rubrics)).toEqual({
      model_catalog_version: models.catalog_version,
      rubric_catalog_version: rubrics.catalog_version,
      judge_models: [anthropic.id, openai.id, meta.id],
      challenge_judge_models: [openai.id],
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

  it('configures the B/C verification panel independently', () => {
    let state = initializeRunConfigState(models, rubrics)
    state = transitionRunConfig(
      state,
      { type: 'select-challenge-judge', position: 1, modelId: meta.id },
      models,
    )
    state = transitionRunConfig(
      state,
      { type: 'select-challenge-judge', position: 2, modelId: openai.id },
      models,
    )

    expect(state.judgeModels.value).toEqual([anthropic.id, openai.id, meta.id])
    expect(state.challengeJudgeModels.value).toEqual([meta.id, openai.id])
    expect(assessRunConfig(state, models, rubrics).errors).toEqual([])

    state = transitionRunConfig(state, { type: 'remove-last-challenge-judge' }, models)
    expect(state.challengeJudgeModels.value).toEqual([meta.id])
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
      judge_models: [openai.id],
      challenge_judge_models: [meta.id],
      proposal_model: writer.id,
      proposal_evaluator_model: anthropic.id,
      rubrics: ['judge.verification'],
      candidate_budget: 4,
      force: true,
    }

    expect(toRunConfig(initializeRunConfigState(models, rubrics, saved), models, rubrics))
      .toEqual(saved)
  })

  it('blocks selected models that cannot fit the largest planned request', () => {
    const state = initializeRunConfigState(models, rubrics)
    const sessions = [{
      judging_token_estimates: {
        utf8_bytes_div_3: { total_tokens: 100_000, largest_turn_tokens: 90_000, turn_tokens: [90_000, 10_000] },
        o200k_base: { total_tokens: 100_000, largest_turn_tokens: 90_000, turn_tokens: [90_000, 10_000] },
        o200k_harmony: { total_tokens: 100_000, largest_turn_tokens: 90_000, turn_tokens: [90_000, 10_000] },
      },
    }]

    expect(assessRunConfig(state, models, rubrics, sessions).errors).toEqual([
      'Select models that fit the selected sessions.',
    ])
    expect(() => toRunConfig(state, models, rubrics, sessions)).toThrow(
      'Select models that fit the selected sessions.',
    )
  })

  it('uses fitting alternatives when recommendations exceed selected-session capacity', () => {
    const roomyWriter = { ...writer, id: 'agy:roomy-writer', max_input_tokens: 1_048_576 }
    const roomyJudge: ModelDescriptor = {
      ...anthropic,
      id: 'agy:roomy-judge',
      max_input_tokens: 1_048_576,
      supported_roles: ['judge', 'proposal_evaluator'],
    }
    const capacityModels: ModelCatalog = {
      ...models,
      available_models: [writer, roomyWriter, anthropic, roomyJudge],
      recommended_judges: [anthropic.id],
      recommended_challenge_judges: [anthropic.id],
      proposal_evaluator_preferences: [anthropic.id, roomyJudge.id],
    }
    const sessions = [{
      judging_token_estimates: {
        utf8_bytes_div_3: { total_tokens: 100_000, largest_turn_tokens: 90_000, turn_tokens: [90_000, 10_000] },
        o200k_base: { total_tokens: 100_000, largest_turn_tokens: 90_000, turn_tokens: [90_000, 10_000] },
        o200k_harmony: { total_tokens: 100_000, largest_turn_tokens: 90_000, turn_tokens: [90_000, 10_000] },
      },
    }]
    const initial = initializeRunConfigState(capacityModels, rubrics)
    const next = transitionRunConfig(
      initial,
      { type: 'use-recommended' },
      capacityModels,
      sessions,
    )

    expect(next.proposalModel.value).toBe(roomyWriter.id)
    expect(next.judgeModels.value).toEqual([roomyJudge.id])
    expect(next.challengeJudgeModels.value).toEqual([roomyJudge.id])
    expect(next.proposalEvaluatorModel.value).toBe(roomyJudge.id)
  })
})
