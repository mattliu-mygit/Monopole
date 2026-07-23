import type {
  PromotionReceipt,
  PromotionReceiptAction,
  ReflectionBundleSnapshot,
  ReflectionCandidate,
  ReflectionResult,
  ReflectionTargetSnapshot,
  Run,
} from '../../types'

export type ReviewState =
  | 'none'
  | 'review-needed'
  | 'promoted'
  | 'partial'
  | 'dismissed'
  | 'stale'
  | 'no-change'

export type PromotionAvailability = {
  enabled: boolean
  target: 'evaluated-candidate' | 'edited-candidate' | null
  requiresUnevaluatedAcknowledgement: boolean
  requiresUnverifiedAcknowledgement: boolean
  reason: string | null
}

export function reflectionCandidates(
  result: ReflectionResult | null | undefined,
): ReflectionCandidate[] {
  return result?.candidates ?? []
}

export function reflectionCandidate(
  result: ReflectionResult | null | undefined,
  candidateId: string | null | undefined,
): ReflectionCandidate | null {
  if (!candidateId) return null
  return reflectionCandidates(result).find((candidate) => candidate.candidate_id === candidateId) ?? null
}

export function selectedReflectionCandidate(run: Run): ReflectionCandidate | null {
  const result = run.reflecting_result
  if (!result || !('baseline' in result)) return null
  if (run.reflection_review) {
    return reflectionCandidate(result, run.reflection_review.selected_candidate_id)
  }
  const recommendation = result?.recommended_candidate_id ?? null
  return (
    reflectionCandidate(result, recommendation) ??
    reflectionCandidates(result)[0] ??
    null
  )
}

export function reviewState(run: Run): ReviewState {
  if (run.reflection_review?.stale) return 'stale'
  if (run.reflection_review?.status === 'pending') return 'review-needed'
  if (run.reflection_review?.status === 'promoted') return 'promoted'
  if (run.reflection_review?.status === 'partial') return 'partial'
  if (run.reflection_review?.status === 'dismissed') return 'dismissed'
  if (
    run.reflecting_result &&
    'baseline_won' in run.reflecting_result &&
    run.reflecting_result.baseline_won
  ) {
    return 'no-change'
  }
  return 'none'
}

export function bundlesEqual(
  left: ReflectionBundleSnapshot | null | undefined,
  right: ReflectionBundleSnapshot | null | undefined,
): boolean {
  return left != null && right != null && left.revision === right.revision
}

function targetMap(bundle: ReflectionBundleSnapshot): Map<string, ReflectionTargetSnapshot> {
  return new Map(bundle.targets.map((target) => [target.locator, target]))
}

function missingLike(target: ReflectionTargetSnapshot): ReflectionTargetSnapshot {
  return {
    ...target,
    exists: false,
    content: null,
    revision: `missing:${target.kind}:${target.locator}`,
  }
}

export function bundleActions(
  past: ReflectionBundleSnapshot,
  next: ReflectionBundleSnapshot,
): PromotionReceiptAction[] {
  const before = targetMap(past)
  const after = targetMap(next)
  const locators = [...new Set([...before.keys(), ...after.keys()])].sort()
  const actions: PromotionReceiptAction[] = []
  for (const locator of locators) {
    const beforeTarget = before.get(locator)
    const afterTarget = after.get(locator)
    const left = beforeTarget ?? missingLike(afterTarget!)
    const right = afterTarget ?? missingLike(beforeTarget!)
    let action: PromotionReceiptAction['action'] | null = null
    if (!left.exists && right.exists) action = 'create'
    else if (left.exists && right.exists && left.content !== right.content) action = 'update'
    if (action) actions.push({ action, locator, before: left, after: right })
  }
  return actions
}

export function changedLocators(
  past: ReflectionBundleSnapshot,
  next: ReflectionBundleSnapshot,
): string[] {
  return bundleActions(past, next).map((action) => action.locator)
}

export function promotionAvailability(run: Run, mutating = false): PromotionAvailability {
  const candidate = selectedReflectionCandidate(run)
  const review = run.reflection_review
  const draft = review?.draft
  const edited = candidate != null && draft?.candidate_id === candidate.candidate_id &&
    !bundlesEqual(draft.bundle, candidate.bundle)
  const target: PromotionAvailability['target'] = candidate == null
    ? null
    : edited
      ? 'edited-candidate'
      : 'evaluated-candidate'
  const recommendedCandidateId = run.reflecting_result &&
    'recommended_candidate_id' in run.reflecting_result
    ? run.reflecting_result.recommended_candidate_id
    : null
  const base: Pick<
    PromotionAvailability,
    'target' | 'requiresUnevaluatedAcknowledgement' | 'requiresUnverifiedAcknowledgement'
  > = {
    target,
    requiresUnevaluatedAcknowledgement: target === 'edited-candidate',
    requiresUnverifiedAcknowledgement: candidate != null &&
      candidate.candidate_id !== recommendedCandidateId,
  }
  if (mutating) return { ...base, enabled: false, reason: 'Another review action is in progress.' }
  if (run.status === 'reflecting') {
    return { ...base, enabled: false, reason: 'Reflection is still finalizing.' }
  }
  if (run.status !== 'complete') {
    return { ...base, enabled: false, reason: 'Promotion requires a completed run.' }
  }
  if (review?.stale) {
    return { ...base, enabled: false, reason: 'Managed instructions changed after evaluation.' }
  }
  if (review?.status === 'promoted' || review?.status === 'partial' || review?.status === 'dismissed') {
    return { ...base, enabled: false, reason: `This review is already resolved as ${review.status}.` }
  }
  if (review?.status !== 'pending') {
    return { ...base, enabled: false, reason: 'No pending review is available.' }
  }
  if (!candidate) return { ...base, enabled: false, reason: 'No evaluated candidate is selected.' }
  return { ...base, enabled: true, reason: null }
}

export function receiptBundles(receipt: PromotionReceipt) {
  return {
    past: receipt.past,
    evaluated: receipt.evaluated_candidate,
    promoted: receipt.promoted,
  }
}
