/** Public UI names backed by the generated HTTP contract. */
import type { components } from './generated/api'

type Schemas = components['schemas']

export type TurnDetail = Schemas['TurnDetailResponse']
export type FeedbackItem = Schemas['FeedbackItemResponse']
export type SignalEvidence = Schemas['SignalEvidenceResponse']
export type SessionSummary = Schemas['SessionSummaryResponse']
export type SessionDetail = Schemas['SessionDetailResponse']

export type ScoreSummary = Schemas['ScoreSummaryResponse']
export type ABEntry = Schemas['ABEntryResponse']
export type TrendEntry = Schemas['TrendEntryResponse']
export type AnalysisResponse = Schemas['AnalysisResponse']

export type ReflectionScope = Schemas['ScopeDescriptorResponse']
export type ReflectionTargetSnapshot = Schemas['TargetSnapshotResponse']
export type ReflectionBundleSnapshot = Schemas['BundleSnapshotResponse']
export type EvaluatorRecord = Schemas['EvaluatorRecordResponse']
export type GenerationAttempt = Schemas['GenerationAttemptResponse']
export type ReflectionCandidate = Schemas['ReflectionCandidateResponse']
export type SuccessfulReflectionResult = Schemas['SuccessfulReflectionResultResponse']
export type EmptyReflectionResult = Schemas['EmptyReflectionResultResponse']
export type ReflectionResult = Schemas['ReflectionResultResponse']
export type ReflectionDraft = Schemas['ReflectionDraftResponse']
export type PromotionTargetOutcome = Schemas['PromotionTargetOutcomeResponse']
export type PromotionReceipt = Schemas['PromotionReceiptResponse']
export type ReflectionReviewState = Schemas['ReflectionReviewResponse']
export type ReflectionReviewStatus = ReflectionReviewState['status']

export type DataSelection = Schemas['DataSelectionResponse']
export type ModelDescriptor = Schemas['ModelDescriptor']
export type RubricDescriptor = Schemas['RubricDescriptor']
export type ProposalCatalog = Schemas['ProposalCatalog']
export type JudgeBackendCatalog = Schemas['JudgeBackendCatalog']
export type ModelCatalog = Schemas['ModelCatalogResponse']
export type RubricCatalog = Schemas['RubricCatalog']
export type RunConfig = Schemas['RunConfig']
export type EffectiveRunConfig = Schemas['EffectiveRunConfig']

export type ScoringTurnDetail = Schemas['ScoringTurnDetailResponse']
export type ScoringProgress = Schemas['ScoringProgressResponse']
export type ScoringResult = Schemas['ScoringResultResponse']
export type JudgingPlan = Schemas['JudgingPlanResponse']
export type ReviewAttempt = Schemas['ReviewAttemptResponse']
export type InferenceStepAudit = Schemas['InferenceStepAuditResponse']
export type JudgingAttemptSummary = Schemas['JudgingAttemptSummaryResponse']
export type JudgingFailureDetail = Schemas['JudgingFailureDetailResponse']
export type JudgingProgress = Schemas['JudgingProgressResponse']
export type JudgingResult = Schemas['JudgingProgressResponse']
export type ReflectingProgress = Schemas['ReflectingProgressResponse']

export type Run = Schemas['RunResponse']
export type RunSummary = Schemas['RunSummaryResponse']
export type RunReviewState = RunSummary['review_state']

/** A rendered before/after action derived locally from two bundle snapshots. */
export interface PromotionReceiptAction {
  action: 'create' | 'update'
  locator: string
  before: ReflectionTargetSnapshot
  after: ReflectionTargetSnapshot
}
