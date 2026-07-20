import type { ModelCatalog, ModelDescriptor, SessionDetail } from '../../types'

export interface ModelCapacityEstimate {
  fits: boolean | null
  estimatedRequestTokens: number | null
  estimatedChunks: number | null
}

type SessionTokens = Pick<SessionDetail, 'judging_token_estimates'>
type ContextPolicy = ModelCatalog['judging_context']

function partitionCoreTokens(
  turnTokens: readonly number[],
  rawBudget: number,
  target: number,
): number[] | null {
  if (turnTokens.some((tokens) => tokens > rawBudget)) return null
  const cores: number[] = []
  for (let start = 0; start < turnTokens.length;) {
    let current = turnTokens[start]
    let end = start
    while (end + 1 < turnTokens.length) {
      const candidate = current + turnTokens[end + 1]
      if (candidate > rawBudget || current >= target) break
      if (Math.abs(target - candidate) > Math.abs(target - current)) break
      current = candidate
      end += 1
    }
    cores.push(current)
    start = end + 1
  }
  return cores
}

function estimateSession(
  model: ModelDescriptor,
  session: SessionTokens,
  policy: ContextPolicy,
): Required<ModelCapacityEstimate> {
  const tokens = session.judging_token_estimates[model.token_counter]
  const large = model.max_input_tokens > policy.large_model_threshold_tokens
  const rawTarget = large
    ? policy.large_model_raw_target_tokens
    : policy.small_model_raw_target_tokens
  const tierReserve = large
    ? policy.large_model_reserve_tokens
    : policy.small_model_reserve_tokens
  const outputReserve = large
    ? policy.large_model_output_reserve_tokens
    : policy.output_reserve_tokens
  const baseReserve = policy.prompt_reserve_tokens
    + outputReserve
    + policy.safety_reserve_tokens
  let chunks = 1
  let coreTokens: number[] | null = null
  const seen = new Set<number>()
  while (!seen.has(chunks)) {
    seen.add(chunks)
    const reserve = Math.max(
      tierReserve,
      baseReserve + Math.max(0, chunks - 1) * policy.digest_max_tokens,
    )
    coreTokens = partitionCoreTokens(
      tokens.turn_tokens,
      model.max_input_tokens - reserve,
      Math.min(rawTarget, model.max_input_tokens - reserve),
    )
    if (coreTokens === null || coreTokens.length === chunks) break
    chunks = coreTokens.length
  }
  const windowReserve = Math.max(
    tierReserve,
    baseReserve + Math.max(0, chunks - 1) * policy.digest_max_tokens,
  )
  const largestCore = coreTokens === null
    ? tokens.largest_turn_tokens
    : Math.max(...coreTokens, 0)
  const estimatedRawChunk = Math.min(tokens.total_tokens, Math.max(rawTarget, largestCore))
  const windowRequest = estimatedRawChunk + windowReserve
  const mergeRequest = baseReserve
    + chunks * (policy.digest_max_tokens + policy.finding_max_tokens)
  const estimatedRequestTokens = Math.max(windowRequest, mergeRequest)
  return {
    fits: coreTokens !== null
      && chunks <= policy.max_chunks
      && estimatedRequestTokens <= model.max_input_tokens,
    estimatedRequestTokens,
    estimatedChunks: chunks,
  }
}

export function estimateModelCapacity(
  model: ModelDescriptor,
  sessions: readonly SessionTokens[],
  policy: ContextPolicy,
): ModelCapacityEstimate {
  if (sessions.length === 0) {
    return { fits: null, estimatedRequestTokens: null, estimatedChunks: null }
  }
  const estimates = sessions.map((session) => estimateSession(model, session, policy))
  return {
    fits: estimates.every((estimate) => estimate.fits),
    estimatedRequestTokens: Math.max(
      ...estimates.map((estimate) => estimate.estimatedRequestTokens),
    ),
    estimatedChunks: Math.max(...estimates.map((estimate) => estimate.estimatedChunks)),
  }
}
