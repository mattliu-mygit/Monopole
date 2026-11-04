import { describe, expect, it } from 'vitest'
import type { JudgingTokenEstimates, ModelDescriptor } from '../../src/types'
import { capacityReserve, estimateModelCapacity } from '../../src/features/runs/modelCapacity'

describe('capacityReserve', () => {
  it.each([[8_192, 4_000], [32_800, 9_840], [131_072, 16_554], [1_000_000, 24_000]])(
    'reserve for %i tokens is %i', (limit, reserve) => {
      expect(capacityReserve(limit)).toBe(reserve)
    })

  it.each([[13_350, 4_337], [13_370, 4_343]])(
    'rounds the exact-half prompt reserve for %i tokens to even like Python, reserving %i',
    (limit, reserve) => {
      expect(capacityReserve(limit)).toBe(reserve)
    })
})

const policy = {
  contract_version: '4' as const,
  large_model_threshold_tokens: 200_000 as const,
  digest_max_tokens: 1_000,
  finding_max_tokens: 4_000,
  overlap_turns: 1 as const,
  max_chunks: 40,
}

function model(
  maxInputTokens: number,
  tokenCounter: ModelDescriptor['token_counter'] = 'utf8_bytes_div_3',
  windowBudgetTokens: number | null = 32_000,
): ModelDescriptor {
  return {
    id: `test:${maxInputTokens}`,
    label: 'Test model',
    provider: 'test',
    provider_model: 'test-model',
    family: 'test',
    supported_roles: ['judge'],
    max_input_tokens: maxInputTokens,
    token_counter: tokenCounter,
    question_budget_tokens: 4_000,
    window_budget_tokens: windowBudgetTokens,
    whole_budget_tokens: windowBudgetTokens,
    merge_budget_tokens: 32_000,
  }
}

function session(
  totalTokens: number,
  largestTurnTokens: number,
  turnTokens: number[] = (() => {
    const values: number[] = []
    for (let remaining = totalTokens; remaining > 0; remaining -= largestTurnTokens) {
      values.push(Math.min(remaining, largestTurnTokens))
    }
    return values
  })(),
): { judging_token_estimates: JudgingTokenEstimates } {
  return {
    judging_token_estimates: {
      utf8_bytes_div_3: { total_tokens: totalTokens, largest_turn_tokens: largestTurnTokens, turn_tokens: turnTokens },
      o200k_base: { total_tokens: totalTokens, largest_turn_tokens: largestTurnTokens, turn_tokens: turnTokens },
      o200k_harmony: { total_tokens: totalTokens, largest_turn_tokens: largestTurnTokens, turn_tokens: turnTokens },
    },
  }
}

describe('estimateModelCapacity', () => {
  it('returns unknown fit when no sessions are selected', () => {
    expect(estimateModelCapacity(model(131_072), [], policy, 'judge')).toEqual({
      fits: null,
      estimatedRequestTokens: null,
      estimatedChunks: null,
    })
  })

  it('packs turns into windows no larger than the model window budget', () => {
    const sessions = [session(100_000, 30_000, [30_000, 30_000, 30_000, 10_000])]

    expect(estimateModelCapacity(
      model(131_072, 'utf8_bytes_div_3', 32_000),
      sessions,
      policy,
      'judge',
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 51_554,
      estimatedChunks: 4,
    })
    expect(estimateModelCapacity(
      model(131_072, 'utf8_bytes_div_3', 64_000),
      sessions,
      policy,
      'judge',
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 81_554,
      estimatedChunks: 2,
    })
  })

  it('limits only ordered judges to the window budget; writers and evaluators pack up to the usable context', () => {
    const noWindow = model(131_072, 'utf8_bytes_div_3', null)
    const sessions = [session(100_000, 30_000, [30_000, 30_000, 30_000, 10_000])]
    const oneWindow = { fits: true, estimatedRequestTokens: 116_554, estimatedChunks: 1 }

    expect(estimateModelCapacity(noWindow, sessions, policy, 'judge')).toEqual({
      fits: false,
      estimatedRequestTokens: null,
      estimatedChunks: null,
    })
    expect(estimateModelCapacity(noWindow, sessions, policy, 'proposal_writer')).toEqual(oneWindow)
    expect(estimateModelCapacity(noWindow, sessions, policy, 'proposal_evaluator')).toEqual(oneWindow)
  })

  it('fits an A/B judge when its limit exceeds 2,048 output plus 1,024 safety tokens, whatever the session', () => {
    const sessions = [session(100_000, 30_000, [30_000, 30_000, 30_000, 10_000])]
    const noRequest = { estimatedRequestTokens: null, estimatedChunks: null }

    expect(estimateModelCapacity(
      model(131_072, 'utf8_bytes_div_3', null),
      sessions,
      policy,
      'challenge_judge',
    )).toEqual({ fits: true, ...noRequest })
    expect(estimateModelCapacity(model(3_073), sessions, policy, 'challenge_judge'))
      .toEqual({ fits: true, ...noRequest })
    expect(estimateModelCapacity(model(3_072), sessions, policy, 'challenge_judge'))
      .toEqual({ fits: false, ...noRequest })
  })

  it('uses the selected model token counter', () => {
    const selected = session(130_000, 120_000)
    selected.judging_token_estimates.o200k_harmony = {
      total_tokens: 70_000,
      largest_turn_tokens: 50_000,
      turn_tokens: [50_000, 20_000],
    }

    expect(estimateModelCapacity(model(131_072), [selected], policy, 'judge').fits).toBe(false)
    expect(estimateModelCapacity(
      model(131_072, 'o200k_harmony'),
      [selected],
      policy,
      'judge',
    )).toMatchObject({ fits: true, estimatedChunks: 2 })
  })

  it('gives one turn larger than the window budget a window of its own', () => {
    expect(estimateModelCapacity(
      model(131_072),
      [session(112_000, 112_000)],
      policy,
      'judge',
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 128_554,
      estimatedChunks: 1,
    })
  })

  it('counts multiple indivisible large turns as separate fitting chunks', () => {
    expect(estimateModelCapacity(
      model(131_072),
      [session(224_000, 112_000, [112_000, 112_000])],
      policy,
      'judge',
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 129_554,
      estimatedChunks: 2,
    })
  })

  it('fits a large turn only in the context left after digests for the other windows', () => {
    expect(estimateModelCapacity(
      model(131_072),
      [session(114_000, 114_000)],
      policy,
      'judge',
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 130_554,
      estimatedChunks: 1,
    })
    expect(estimateModelCapacity(
      model(131_072),
      [session(134_000, 114_000, [10_000, 114_000, 10_000])],
      policy,
      'judge',
    )).toEqual({
      fits: false,
      estimatedRequestTokens: 132_554,
      estimatedChunks: 3,
    })
  })

  it('reserves the larger generation allowance for large-model requests', () => {
    expect(estimateModelCapacity(
      model(262_000),
      [session(128_000, 128_000)],
      policy,
      'judge',
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 152_000,
      estimatedChunks: 1,
    })
  })

  it('fails sessions that exceed the pinned chunk limit', () => {
    const large = model(1_048_576, 'utf8_bytes_div_3', 128_000)

    expect(estimateModelCapacity(large, [session(4_000_000, 100_000)], policy, 'judge')).toEqual({
      fits: true,
      estimatedRequestTokens: 224_000,
      estimatedChunks: 40,
    })
    expect(estimateModelCapacity(large, [session(4_100_000, 100_000)], policy, 'judge')).toEqual({
      fits: false,
      estimatedRequestTokens: 229_000,
      estimatedChunks: 41,
    })
  })

  it('fails sessions whose merge request, a digest and a finding per window, exceeds the model', () => {
    expect(estimateModelCapacity(
      model(131_072),
      [session(660_000, 30_000)],
      policy,
      'judge',
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 126_554,
      estimatedChunks: 22,
    })
    expect(estimateModelCapacity(
      model(131_072),
      [session(690_000, 30_000)],
      policy,
      'judge',
    )).toEqual({
      fits: false,
      estimatedRequestTokens: 131_554,
      estimatedChunks: 23,
    })
  })

  it('reports the worst request and largest chunk count across selected sessions', () => {
    expect(estimateModelCapacity(
      model(131_072),
      [session(100_000, 30_000), session(300_000, 70_000)],
      policy,
      'judge',
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 90_554,
      estimatedChunks: 5,
    })
  })
})
