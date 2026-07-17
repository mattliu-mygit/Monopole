import type { ModelCatalog, ModelDescriptor, SessionSummary } from '../../types'

export interface ModelCapacityEstimate {
  fits: boolean | null
  estimatedRequestTokens: number | null
  estimatedChunks: number | null
}

type SessionTokens = Pick<SessionSummary, 'total_tokens' | 'largest_turn_tokens'>
type ContextPolicy = ModelCatalog['judging_context']

function estimateSession(
  model: ModelDescriptor,
  session: SessionTokens,
  policy: ContextPolicy,
): Required<ModelCapacityEstimate> {
  const large = model.max_input_tokens > policy.large_model_threshold_tokens
  const rawTarget = large
    ? policy.large_model_raw_target_tokens
    : policy.small_model_raw_target_tokens
  const tierReserve = large
    ? policy.large_model_reserve_tokens
    : policy.small_model_reserve_tokens
  const chunks = Math.max(1, Math.ceil(session.total_tokens / rawTarget))
  const baseReserve = policy.prompt_reserve_tokens
    + policy.output_reserve_tokens
    + policy.safety_reserve_tokens
  const windowReserve = Math.max(
    tierReserve,
    baseReserve + Math.max(0, chunks - 1) * policy.digest_max_tokens,
  )
  const estimatedRawChunk = Math.min(
    session.total_tokens,
    Math.max(rawTarget, session.largest_turn_tokens),
  )
  const windowRequest = estimatedRawChunk + windowReserve
  const mergeRequest = baseReserve
    + chunks * (policy.digest_max_tokens + policy.finding_max_tokens)
  const estimatedRequestTokens = Math.max(windowRequest, mergeRequest)
  return {
    fits: chunks <= policy.max_chunks
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
