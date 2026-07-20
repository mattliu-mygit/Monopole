import { describe, expect, it } from 'vitest'
import type { ModelDescriptor, SessionDetail } from '../../src/types'
import { estimateModelCapacity } from '../../src/features/runs/modelCapacity'

const policy = {
  contract_version: '3' as const,
  large_model_threshold_tokens: 200_000 as const,
  large_model_reserve_tokens: 100_000,
  small_model_reserve_tokens: 50_000,
  large_model_raw_target_tokens: 128_000,
  small_model_raw_target_tokens: 50_000,
  prompt_reserve_tokens: 6_000,
  output_reserve_tokens: 4_000,
  large_model_output_reserve_tokens: 10_000,
  safety_reserve_tokens: 8_000,
  digest_max_tokens: 1_000,
  finding_max_tokens: 4_000,
  overlap_turns: 1 as const,
  max_chunks: 40,
}

function model(
  maxInputTokens: number,
  tokenCounter: ModelDescriptor['token_counter'] = 'utf8_bytes_div_3',
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
): Pick<SessionDetail, 'judging_token_estimates'> {
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

  it('uses total tokens for chunk overhead and the largest turn for raw-window fit', () => {
    expect(estimateModelCapacity(model(131_072), [session(100_000, 30_000)], policy)).toEqual({
      fits: true,
      estimatedRequestTokens: 110_000,
      estimatedChunks: 2,
    })
    expect(estimateModelCapacity(model(131_072), [session(100_000, 90_000)], policy)).toEqual({
      fits: false,
      estimatedRequestTokens: 140_000,
      estimatedChunks: 1,
    })
  })

  it('uses the selected model token counter', () => {
    const selected = session(100_000, 90_000)
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

  it('does not split one indivisible turn when estimating chunk overhead', () => {
    const protocolReserve = {
      ...policy,
      large_model_reserve_tokens: 18_000,
      small_model_reserve_tokens: 18_000,
    }

    expect(estimateModelCapacity(
      model(131_072),
      [session(112_000, 112_000)],
      protocolReserve,
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 130_000,
      estimatedChunks: 1,
    })
  })

  it('counts multiple indivisible large turns as separate fitting chunks', () => {
    const protocolReserve = {
      ...policy,
      large_model_reserve_tokens: 18_000,
      small_model_reserve_tokens: 18_000,
    }

    expect(estimateModelCapacity(
      model(131_072),
      [session(224_000, 112_000, [112_000, 112_000])],
      protocolReserve,
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 131_000,
      estimatedChunks: 2,
    })
  })

  it('uses the large-model target and reserve above the threshold', () => {
    expect(estimateModelCapacity(model(1_048_576), [session(2_100_000, 100_000)], policy)).toEqual({
      fits: true,
      estimatedRequestTokens: 228_000,
      estimatedChunks: 21,
    })
  })

  it('reserves the larger generation allowance for large-model requests', () => {
    const protocolReserve = {
      ...policy,
      large_model_reserve_tokens: 18_000,
      small_model_reserve_tokens: 18_000,
    }

    expect(estimateModelCapacity(
      model(262_000),
      [session(128_000, 128_000)],
      protocolReserve,
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 152_000,
      estimatedChunks: 1,
    })
  })

  it('fails sessions that exceed the pinned chunk limit', () => {
    expect(estimateModelCapacity(model(131_072), [session(2_500_000, 30_000)], policy)).toEqual({
      fits: false,
      estimatedRequestTokens: 228_000,
      estimatedChunks: 42,
    })
  })

  it('reports the worst request and largest chunk count across selected sessions', () => {
    expect(estimateModelCapacity(
      model(131_072),
      [session(100_000, 30_000), session(300_000, 70_000)],
      policy,
    )).toEqual({
      fits: true,
      estimatedRequestTokens: 120_000,
      estimatedChunks: 5,
    })
  })
})
