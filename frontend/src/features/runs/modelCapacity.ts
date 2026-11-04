import type { JudgingTokenEstimates, ModelCatalog, ModelDescriptor } from '../../types'

export interface ModelCapacityEstimate {
  fits: boolean | null
  estimatedRequestTokens: number | null
  estimatedChunks: number | null
}

type SessionTokens = { judging_token_estimates: JudgingTokenEstimates }
type ContextPolicy = ModelCatalog['judging_context']
type ModelRole = ModelDescriptor['supported_roles'][number]

function roundHalfToEven(value: number): number {
  const floor = Math.floor(value)
  if (value - floor !== 0.5) return Math.round(value)
  return floor % 2 === 0 ? floor : floor + 1
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(value, high))
}

export function capacityReserve(modelLimit: number): number {
  const prompt = clamp(roundHalfToEven(0.15 * modelLimit), 2_000, 6_000)
  const generationCap = modelLimit > 200_000 ? 10_000 : 4_000
  const generation = clamp(roundHalfToEven(0.10 * modelLimit), 1_000, generationCap)
  const safety = clamp(roundHalfToEven(0.05 * modelLimit), 1_000, 8_000)
  return prompt + generation + safety
}

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
    while (end + 1 < turnTokens.length && current + turnTokens[end + 1] <= target) {
      end += 1
      current += turnTokens[end]
    }
    cores.push(current)
    start = end + 1
  }
  return cores
}

function estimateSession(
  model: ModelDescriptor,
  windowBudget: number,
  session: SessionTokens,
  policy: ContextPolicy,
): Required<ModelCapacityEstimate> {
  const tokens = session.judging_token_estimates[model.token_counter]
  const baseReserve = capacityReserve(model.max_input_tokens)
  let chunks = 1
  let coreTokens: number[] | null = null
  const seen = new Set<number>()
  while (!seen.has(chunks)) {
    seen.add(chunks)
    const reserve = baseReserve + Math.max(0, chunks - 1) * policy.digest_max_tokens
    coreTokens = partitionCoreTokens(
      tokens.turn_tokens,
      model.max_input_tokens - reserve,
      Math.min(windowBudget, model.max_input_tokens - reserve),
    )
    if (coreTokens === null || coreTokens.length === chunks) break
    chunks = coreTokens.length
  }
  const windowReserve = baseReserve + Math.max(0, chunks - 1) * policy.digest_max_tokens
  const target = Math.min(windowBudget, model.max_input_tokens - windowReserve)
  const largestCore = coreTokens === null
    ? tokens.largest_turn_tokens
    : Math.max(...coreTokens, 0)
  const estimatedRawChunk = Math.min(tokens.total_tokens, Math.max(target, largestCore))
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
  role: ModelRole,
): ModelCapacityEstimate {
  if (sessions.length === 0) {
    return { fits: null, estimatedRequestTokens: null, estimatedChunks: null }
  }
  const windowBudget = role === 'judge'
    ? model.window_budget_tokens
    : model.max_input_tokens - capacityReserve(model.max_input_tokens)
  if (windowBudget === null) {
    return { fits: false, estimatedRequestTokens: null, estimatedChunks: null }
  }
  const estimates = sessions.map((session) => estimateSession(model, windowBudget, session, policy))
  return {
    fits: estimates.every((estimate) => estimate.fits),
    estimatedRequestTokens: Math.max(
      ...estimates.map((estimate) => estimate.estimatedRequestTokens),
    ),
    estimatedChunks: Math.max(...estimates.map((estimate) => estimate.estimatedChunks)),
  }
}
