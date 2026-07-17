import { describe, expect, it } from 'vitest'
import type { ModelDescriptor, SessionSummary } from '../../src/types'
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
  safety_reserve_tokens: 8_000,
  digest_max_tokens: 1_000,
  finding_max_tokens: 4_000,
  overlap_turns: 1 as const,
  max_chunks: 40,
}

function model(maxInputTokens: number): ModelDescriptor {
  return {
    id: `test:${maxInputTokens}`,
    label: 'Test model',
    provider: 'test',
    provider_model: 'test-model',
    family: 'test',
    supported_roles: ['judge'],
    max_input_tokens: maxInputTokens,
    token_counter: 'utf8_bytes_div_3',
  }
}

function session(totalTokens: number, largestTurnTokens: number): SessionSummary {
  return {
    conversation_id: `session-${totalTokens}-${largestTurnTokens}`,
    session_id: null,
    turn_count: 2,
    started_at: null,
    ended_at: null,
    last_activity: null,
    model: null,
    effort_level: null,
    config_version: null,
    git_branch: null,
    total_tokens: totalTokens,
    largest_turn_tokens: largestTurnTokens,
    total_tool_calls: 0,
    input_preview: null,
    signal_evidence: [],
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
      estimatedRequestTokens: 100_000,
      estimatedChunks: 2,
    })
    expect(estimateModelCapacity(model(131_072), [session(100_000, 90_000)], policy)).toEqual({
      fits: false,
      estimatedRequestTokens: 140_000,
      estimatedChunks: 2,
    })
  })

  it('uses the large-model target and reserve above the threshold', () => {
    expect(estimateModelCapacity(model(1_048_576), [session(2_100_000, 100_000)], policy)).toEqual({
      fits: true,
      estimatedRequestTokens: 228_000,
      estimatedChunks: 17,
    })
  })

  it('fails sessions that exceed the pinned chunk limit', () => {
    expect(estimateModelCapacity(model(131_072), [session(2_100_000, 30_000)], policy)).toEqual({
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
      estimatedChunks: 6,
    })
  })
})
