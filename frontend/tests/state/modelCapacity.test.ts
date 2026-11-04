import { describe, expect, it } from 'vitest'
import type { JudgingTokenEstimates, ModelDescriptor } from '../../src/types'
import { capacityReserve, estimateModelCapacity } from '../../src/features/runs/modelCapacity'

describe('capacityReserve', () => {
  it.each([[8_192, 4_000], [32_800, 9_840], [131_072, 16_554], [1_000_000, 24_000]])(
    'reserve for %i tokens is %i', (limit, reserve) => {
      expect(capacityReserve(limit)).toBe(reserve)
    })

  // 0.15 × 13,350 is exactly 2,002.5 and 0.15 × 13,370 is exactly 2,005.5. Python's round()
  // takes both prompt reserves to the even neighbor, 2,002 and 2,006, where Math.round gives
  // 2,003 for the first. The 0.05 × 10,010 = 500.5 half is hidden by the 1,000 safety floor.
  it.each([[13_350, 4_337], [13_370, 4_343]])(
    'rounds an exact half to even like Python for %i tokens', (limit, reserve) => {
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
    expect(estimateModelCapacity(model(131_072), [], policy)).toEqual({
      fits: null,
      estimatedRequestTokens: null,
      estimatedChunks: null,
    })
  })

  it('packs turns into windows no larger than the model window budget', () => {
    // Turns of 30k, 30k, 30k and 10k: a 32k budget takes one turn per window, and a 64k
    // budget takes 30k + 30k and then 30k + 10k.
    expect(estimateModelCapacity(
      model(131_072, 'utf8_bytes_div_3', 32_000),
      [session(100_000, 30_000)],
      policy,
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 51_554,
      estimatedChunks: 4,
    })
    expect(estimateModelCapacity(
      model(131_072, 'utf8_bytes_div_3', 64_000),
      [session(100_000, 30_000)],
      policy,
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 81_554,
      estimatedChunks: 2,
    })
  })

  it('uses the selected model token counter', () => {
    const selected = session(130_000, 120_000)
    selected.judging_token_estimates.o200k_harmony = {
      total_tokens: 70_000,
      largest_turn_tokens: 50_000,
      turn_tokens: [50_000, 20_000],
    }

    expect(estimateModelCapacity(model(131_072), [selected], policy).fits).toBe(false)
    expect(estimateModelCapacity(
      model(131_072, 'o200k_harmony'),
      [selected],
      policy,
    )).toMatchObject({ fits: true, estimatedChunks: 2 })
  })

  it('gives one turn larger than the window budget a window of its own', () => {
    expect(estimateModelCapacity(model(131_072), [session(112_000, 112_000)], policy)).toEqual({
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
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 129_554,
      estimatedChunks: 2,
    })
  })

  it('fits a large turn only in the context left after digests for the other windows', () => {
    // A 131,072-token model keeps 114,518 tokens after its 16,554 reserve. Beside two other
    // windows, 2,000 more go to their digests, so a 114,000-token turn no longer fits.
    expect(estimateModelCapacity(model(131_072), [session(114_000, 114_000)], policy)).toEqual({
      fits: true,
      estimatedRequestTokens: 130_554,
      estimatedChunks: 1,
    })
    expect(estimateModelCapacity(
      model(131_072),
      [session(134_000, 114_000, [10_000, 114_000, 10_000])],
      policy,
    )).toEqual({
      fits: false,
      estimatedRequestTokens: 132_554,
      estimatedChunks: 3,
    })
  })

  it('does not fit a model without a window budget', () => {
    expect(estimateModelCapacity(
      model(131_072, 'utf8_bytes_div_3', null),
      [session(10_000, 10_000)],
      policy,
    )).toEqual({
      fits: false,
      estimatedRequestTokens: null,
      estimatedChunks: null,
    })
  })

  it('reserves the larger generation allowance for large-model requests', () => {
    expect(estimateModelCapacity(model(262_000), [session(128_000, 128_000)], policy)).toEqual({
      fits: true,
      estimatedRequestTokens: 152_000,
      estimatedChunks: 1,
    })
  })

  it('fails sessions that exceed the pinned chunk limit', () => {
    const large = model(1_048_576, 'utf8_bytes_div_3', 128_000)

    expect(estimateModelCapacity(large, [session(4_000_000, 100_000)], policy)).toEqual({
      fits: true,
      estimatedRequestTokens: 224_000,
      estimatedChunks: 40,
    })
    expect(estimateModelCapacity(large, [session(4_100_000, 100_000)], policy)).toEqual({
      fits: false,
      estimatedRequestTokens: 229_000,
      estimatedChunks: 41,
    })
  })

  it('fails sessions whose merge request exceeds the model', () => {
    // Each window adds a 1,000-token digest and a 4,000-token finding to the merge request.
    expect(estimateModelCapacity(model(131_072), [session(660_000, 30_000)], policy)).toEqual({
      fits: true,
      estimatedRequestTokens: 126_554,
      estimatedChunks: 22,
    })
    expect(estimateModelCapacity(model(131_072), [session(690_000, 30_000)], policy)).toEqual({
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
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 90_554,
      estimatedChunks: 5,
    })
  })
})
