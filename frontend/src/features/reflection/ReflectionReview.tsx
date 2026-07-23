import { useEffect, useMemo, useRef, useState } from 'react'
import type {
  EmptyReflectionResult,
  EvaluatorRecord,
  GenerationAttempt,
  ReflectionCandidate,
  ReflectionBundleSnapshot,
  ReflectionResult,
  ReflectionTargetSnapshot,
  Run,
  SuccessfulReflectionResult,
} from '../../types'
import {
  promotionAvailability,
  bundleActions,
  reflectionCandidates,
  reviewState,
  selectedReflectionCandidate,
} from './reflectionState'
import Dialog from '../../components/Dialog'
import BundleComparison from './BundleComparison'
import PromotionReceiptView from './PromotionReceiptView'
import ReflectionDecisionControls from './ReflectionDecisionControls'
import ReflectionScoreComparison from './ReflectionScoreComparison'
import EvaluatorAudit from './EvaluatorAudit'
import StaleEvidence from './StaleEvidence'

const NO_VALID_PROPOSAL_REASON = 'No valid proposal generated'
const SNAPSHOT_PAGE_SIZE = 25
const MAX_RESPONSE_EXCERPT_LENGTH = 2_000

type MaybePromise = void | Promise<void>

export interface ReflectionReviewProps {
  run: Run
  onSelect?: (candidateId: string, discardDraft: boolean) => MaybePromise
  onSaveDraft?: (contents: Record<string, string | null>) => MaybePromise
  onResetDraft?: () => MaybePromise
  onPromote?: (
    acknowledgeUnevaluated: boolean,
    acknowledgeUnverified: boolean,
  ) => MaybePromise
  onDismiss?: () => MaybePromise
  onDirtyChange?: (dirty: boolean) => void
}

type DialogState = 'promote' | 'dismiss' | { selectCandidate: string } | null

function sameTargets(
  left: readonly ReflectionTargetSnapshot[],
  right: readonly ReflectionTargetSnapshot[],
): boolean {
  if (left.length !== right.length) return false
  const other = new Map(right.map((target) => [target.locator, target]))
  return left.every((target) => {
    const candidate = other.get(target.locator)
    return candidate?.exists === target.exists && candidate.content === target.content
  })
}

function cloneTargets(targets: readonly ReflectionTargetSnapshot[]): ReflectionTargetSnapshot[] {
  return targets.map((target) => ({ ...target }))
}

function contents(targets: readonly ReflectionTargetSnapshot[]): Record<string, string | null> {
  return Object.fromEntries(targets.map((target) => [
    target.locator,
    target.exists ? target.content ?? '' : null,
  ]))
}

function actionKey(action: { locator: string; action: string }): string {
  return `${action.locator}:${action.action}`
}

function actionSetMismatches(
  past: ReflectionBundleSnapshot,
  proposed: ReflectionBundleSnapshot,
  editedTargets: ReflectionTargetSnapshot[],
): string[] {
  const expected = new Set(bundleActions(past, proposed).map(actionKey))
  const edited = { ...proposed, revision: 'unsaved:d', targets: editedTargets }
  const actual = new Set(bundleActions(past, edited).map(actionKey))
  const mismatches = new Set<string>()
  for (const key of expected) if (!actual.has(key)) mismatches.add(key.slice(0, key.lastIndexOf(':')))
  for (const key of actual) if (!expected.has(key)) mismatches.add(key.slice(0, key.lastIndexOf(':')))
  return [...mismatches].sort()
}

function SnapshotTargets({ bundle }: { bundle: ReflectionBundleSnapshot }) {
  const [visibleCount, setVisibleCount] = useState(SNAPSHOT_PAGE_SIZE)
  const [openTargets, setOpenTargets] = useState<Set<string>>(() => new Set())

  useEffect(() => {
    setVisibleCount(SNAPSHOT_PAGE_SIZE)
    setOpenTargets(new Set())
  }, [bundle.revision])

  const visibleTargets = bundle.targets.slice(0, visibleCount)
  const remaining = bundle.targets.length - visibleTargets.length

  function toggleTarget(locator: string) {
    setOpenTargets((current) => {
      const next = new Set(current)
      if (next.has(locator)) next.delete(locator)
      else next.add(locator)
      return next
    })
  }

  return (
    <div className="space-y-2">
      {visibleTargets.map((target) => {
        const open = openTargets.has(target.locator)
        return (
          <details key={target.locator} open={open} className="rounded-lg border border-gray-200 p-3">
            <summary
              className="cursor-pointer font-mono text-xs font-semibold text-gray-800"
              onClick={(event) => {
                event.preventDefault()
                toggleTarget(target.locator)
              }}
            >
              {target.locator}
            </summary>
            {open && (
              <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-3 font-mono text-xs text-gray-100">
                {target.exists ? target.content : 'This file did not exist.'}
              </pre>
            )}
          </details>
        )
      })}
      {remaining > 0 && (
        <button
          type="button"
          className="rounded-lg border border-gray-300 px-3 py-2 text-xs font-medium text-gray-700 hover:bg-gray-50"
          onClick={() => setVisibleCount((count) => count + SNAPSHOT_PAGE_SIZE)}
        >
          Show {Math.min(SNAPSHOT_PAGE_SIZE, remaining)} more files
        </button>
      )}
    </div>
  )
}

function isSuccessfulResult(result: ReflectionResult): result is SuccessfulReflectionResult {
  return 'baseline' in result
}

function EmptyResult({ result }: { result: EmptyReflectionResult }) {
  return (
    <section aria-label="Reflection review" className="rounded-lg border border-gray-200 bg-gray-50 p-4">
      <h4 className="text-sm font-semibold text-gray-900">{result.reason}</h4>
      <p className="mt-1 text-sm text-gray-600">
        Reflection ended before a baseline could be evaluated. There is no proposal to review.
      </p>
    </section>
  )
}

function boundedExcerpt(value: string): string {
  if (value.length <= MAX_RESPONSE_EXCERPT_LENGTH) return value
  return `${value.slice(0, MAX_RESPONSE_EXCERPT_LENGTH)}…`
}

function FailedAttemptAudit({ attempts }: { attempts: GenerationAttempt[] }) {
  const failed = attempts.filter((attempt) => attempt.status === 'failed')
  if (failed.length === 0) return null
  return (
    <details className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs">
      <summary className="cursor-pointer font-medium text-gray-800">
        Rejected proposal attempts
      </summary>
      <div className="mt-3 space-y-3">
        {failed.map((attempt) => (
          <section key={attempt.attempt_id} className="rounded-lg border border-gray-200 bg-white p-3">
            <h5 className="font-semibold text-gray-900">Attempt {attempt.number}</h5>
            <dl className="mt-2 grid gap-2 sm:grid-cols-2">
              <div><dt className="text-gray-500">Attempt ID</dt><dd className="break-all font-mono">{attempt.attempt_id}</dd></div>
              <div><dt className="text-gray-500">Changed paths</dt><dd className="font-mono">{attempt.changed_paths.join(', ') || 'None reported'}</dd></div>
              <div><dt className="text-gray-500">Error type</dt><dd>{attempt.error_type ?? 'Unknown'}</dd></div>
              <div><dt className="text-gray-500">Error</dt><dd>{attempt.error ?? 'No error message recorded'}</dd></div>
              <div className="sm:col-span-2"><dt className="text-gray-500">Response digest</dt><dd className="break-all font-mono">{attempt.response_digest ?? 'Not recorded'}</dd></div>
            </dl>
            {attempt.response_excerpt && (
              <pre
                aria-label={`Attempt ${attempt.number} response excerpt`}
                className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-3 font-mono text-xs text-gray-100"
              >
                {boundedExcerpt(attempt.response_excerpt)}
              </pre>
            )}
          </section>
        ))}
      </div>
    </details>
  )
}

function EvaluatorAssessment({
  baseline,
  candidate,
}: {
  baseline: EvaluatorRecord
  candidate: ReflectionCandidate
}) {
  return (
    <section className="space-y-3 rounded-lg border border-gray-200 p-4">
      <h4 className="text-sm font-semibold text-gray-900">Evaluator assessment</h4>
      <p className="text-sm text-gray-600">{candidate.evaluation.rationale}</p>
      <p className="text-xs text-gray-500">
        Score basis: predicted evaluator score, not a verification run.
      </p>
      <div className="grid gap-2 lg:grid-cols-2">
        <EvaluatorAudit label="A" record={baseline} />
        <EvaluatorAudit label="B" record={candidate.evaluation} />
      </div>
    </section>
  )
}

function statusLabel(run: Run): string {
  const state = reviewState(run)
  if (state === 'review-needed') return 'Review needed'
  if (state === 'promoted') return 'Promoted'
  if (state === 'partial') return 'Partially applied'
  if (state === 'dismissed') return 'Dismissed'
  if (state === 'stale') return 'Stale'
  if (state === 'no-change') return 'No change recommended'
  return 'Read only'
}

function BaselineResult({ result }: { result: SuccessfulReflectionResult }) {
  return (
    <section aria-label="Reflection review" className="space-y-4">
      <div className="rounded-lg border border-green-200 bg-green-50 p-4">
        <span className="rounded-full bg-green-100 px-2.5 py-1 text-xs font-semibold text-green-800">
          No change recommended
        </span>
        <p className="mt-3 text-sm text-green-950">
          Evaluated past A scored best. No instruction change is recommended.
        </p>
        <div className="mt-3 font-mono text-2xl font-semibold text-green-950">
          {result.baseline_score.toFixed(3)}
        </div>
      </div>
      <SnapshotTargets bundle={result.baseline} />
      <EvaluatorAudit label="A" record={result.baseline_evaluation} />
    </section>
  )
}

type Challenge = NonNullable<SuccessfulReflectionResult['challenge']>
type ChallengeArm = NonNullable<Challenge['baseline']>

function challengeTitle(challenge: Challenge): string {
  if (challenge.status !== 'complete') return 'Paired sandbox verification incomplete'
  if (challenge.winner === 'candidate') return 'B won paired sandbox verification'
  if (challenge.winner === 'baseline') return 'A won paired sandbox verification'
  return 'Paired sandbox verification tied'
}

function PairedArmEvidence({ label, arm }: { label: 'A' | 'B', arm: ChallengeArm | null }) {
  if (!arm) return null
  return (
    <details className="rounded border border-sky-200 bg-white p-3 text-xs">
      <summary className="cursor-pointer font-medium text-sky-950">
        {label} arm: {arm.status} · exit {arm.exit_code ?? 'none'} · {arm.duration_seconds.toFixed(1)}s
      </summary>
      <div className="mt-3 space-y-3">
        {arm.infrastructure_error && <p className="text-red-700">{arm.infrastructure_error}</p>}
        <div>
          <p className="font-medium text-gray-800">Transcript</p>
          <pre className="mt-1 max-h-80 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-3 text-gray-100">
            {arm.transcript || 'No transcript output.'}
          </pre>
        </div>
        <div>
          <p className="font-medium text-gray-800">Changed artifacts ({arm.artifact_changes.length})</p>
          <div className="mt-1 space-y-2">
            {arm.artifact_changes.map((artifact) => (
              <details key={artifact.path} className="rounded border border-gray-200 p-2">
                <summary className="cursor-pointer font-mono">
                  {artifact.action} {artifact.path}
                </summary>
                <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-3 text-gray-100">
                  {artifact.diff}
                </pre>
              </details>
            ))}
          </div>
        </div>
      </div>
    </details>
  )
}

function PairedChallengeEvidence({ challenge }: { challenge: Challenge }) {
  const execution = challenge.execution
  return (
    <section
      aria-label="Paired sandbox verification"
      className="space-y-3 rounded-lg border border-sky-200 bg-sky-50/50 p-4"
    >
      <div>
        <h4 className="text-sm font-semibold text-sky-950">{challengeTitle(challenge)}</h4>
        {challenge.reason && <p className="mt-1 text-sm text-sky-900">{challenge.reason}</p>}
      </div>
      {challenge.task && (
        <div className="space-y-2 text-sm">
          <dl className="grid gap-2 sm:grid-cols-2">
            <div><dt className="font-medium text-sky-950">Prompt</dt><dd>{challenge.task.prompt}</dd></div>
            <div><dt className="font-medium text-sky-950">Goal</dt><dd>{challenge.task.goal}</dd></div>
            <div><dt className="font-medium text-sky-950">Setup mode</dt><dd>{challenge.task.setup_mode}</dd></div>
          </dl>
          {challenge.task.materials.length > 0 && (
            <div>
              <p className="font-medium text-sky-950">Pinned materials</p>
              <ul className="list-disc pl-5">
                {challenge.task.materials.map((material) => (
                  <li key={`${material.destination}:${material.revision}`}>
                    <span>{material.url}</span>{' '}
                    <span className="font-mono text-xs">@ {material.revision} → {material.destination}/</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="grid gap-2 sm:grid-cols-2">
            <div>
              <p className="font-medium text-sky-950">Task judging criteria</p>
              <ul className="list-disc pl-5">
                {challenge.task.judging_criteria.map((criterion) => <li key={criterion}>{criterion}</li>)}
              </ul>
            </div>
            <div>
              <p className="font-medium text-sky-950">Starting-state checks</p>
              <ul className="list-disc pl-5">
                {challenge.task.start_checks.map((check) => <li key={check}>{check}</li>)}
              </ul>
            </div>
          </div>
          <div>
            <p className="font-medium text-sky-950">Preflight requirements</p>
            <ul className="list-disc pl-5 font-mono text-xs">
              {challenge.task.required_files.map((path) => (
                <li key={`file:${path}`}>{path}</li>
              ))}
              {challenge.task.required_executables.map((executable) => (
                <li key={`executable:${executable}`}>{executable}</li>
              ))}
              <li>
                {challenge.task.requires_git_metadata
                  ? 'Git metadata required'
                  : 'Git metadata not required'}
              </li>
            </ul>
          </div>
        </div>
      )}
      <p className="text-xs text-sky-900">
        {execution.model} via {execution.harness_version} · {execution.effort ?? 'default effort'} ·{' '}
        {execution.timeout_seconds}s timeout · {execution.network_enabled ? 'network enabled' : 'network disabled'}
      </p>
      <div className="grid gap-2 sm:grid-cols-2">
        <PairedArmEvidence label="A" arm={challenge.baseline} />
        <PairedArmEvidence label="B" arm={challenge.candidate} />
      </div>
      <details className="rounded border border-sky-200 bg-white p-3 text-xs">
        <summary className="cursor-pointer font-medium text-sky-950">
          Blinded judge evidence ({challenge.judges.length})
        </summary>
        <div className="mt-3 space-y-3">
          {challenge.judges.map((judge) => (
            <div key={judge.position}>
              <p className="font-medium">Judge {judge.position}: {judge.winner} · {judge.resolved_model}</p>
              {!judge.task_valid && (
                <p className="mt-1 font-medium text-amber-700">
                  Invalid task: {judge.task_invalid_reason}
                </p>
              )}
              <p className="mt-1 text-gray-600">{judge.rationale}</p>
              <ul className="mt-2 space-y-1 font-mono">
                {judge.rubrics.map((rubric) => (
                  <li key={rubric.rubric_id}>
                    {rubric.rubric_id}: A {rubric.baseline_score.toFixed(2)} / B{' '}
                    {rubric.candidate_score.toFixed(2)} · Δ{' '}
                    {(rubric.delta ?? rubric.candidate_score - rubric.baseline_score) >= 0 ? '+' : ''}
                    {(rubric.delta ?? rubric.candidate_score - rubric.baseline_score).toFixed(2)}{' '}
                    ({rubric.winner})
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </details>
      <p className="break-all font-mono text-[11px] text-sky-800">
        {challenge.challenge_id} · {execution.execution_id}
      </p>
    </section>
  )
}

function NoValidProposalResult({ result }: { result: SuccessfulReflectionResult }) {
  return (
    <section aria-label="Reflection review" className="space-y-4">
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
        <h4 className="text-sm font-semibold text-amber-950">{result.reason}</h4>
        <p className="mt-1 text-sm text-amber-900">
          Every proposal attempt was rejected. No proposed B was evaluated, so editing and promotion are unavailable.
        </p>
        <div className="mt-3 text-xs font-semibold uppercase tracking-wide text-amber-800">Evaluated past A</div>
        <div className="mt-1 font-mono text-2xl font-semibold text-amber-950">
          {result.baseline_score.toFixed(3)}
        </div>
        <p className="mt-2 text-sm text-amber-900">{result.baseline_evaluation.rationale}</p>
      </div>
      <SnapshotTargets bundle={result.baseline} />
      <EvaluatorAudit label="A" record={result.baseline_evaluation} />
      <FailedAttemptAudit attempts={result.generation_attempts} />
    </section>
  )
}

export default function ReflectionReview({
  run,
  onSelect,
  onSaveDraft,
  onResetDraft,
  onPromote,
  onDismiss,
  onDirtyChange,
}: ReflectionReviewProps) {
  const result = run.reflecting_result
  const candidates = useMemo(() => reflectionCandidates(result), [result])
  const selected = selectedReflectionCandidate(run)
  const review = run.reflection_review
  const persistedDraft = review?.draft && review.draft.candidate_id === selected?.candidate_id
    ? review.draft
    : null
  const persistedTargets = persistedDraft?.bundle.targets
  const selectedTargets = selected?.bundle.targets
  const initialTargets = persistedTargets ?? selectedTargets ?? []
  const resetTargetsRef = useRef(initialTargets)
  resetTargetsRef.current = initialTargets
  const [draftTargets, setDraftTargets] = useState<ReflectionTargetSnapshot[]>(() => (
    cloneTargets(initialTargets)
  ))
  const [busy, setBusy] = useState(false)
  const [acknowledgedUnevaluated, setAcknowledgedUnevaluated] = useState(false)
  const [acknowledgedUnverified, setAcknowledgedUnverified] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dialog, setDialog] = useState<DialogState>(null)
  const [inspectedCandidateId, setInspectedCandidateId] = useState<string | null>(
    () => selected?.candidate_id ?? null,
  )
  const dirty = !sameTargets(
    draftTargets,
    persistedTargets ?? selectedTargets ?? [],
  )
  const finalizing = run.status === 'reflecting'
  const serverSelectionAuthoritative = (
    review?.status === 'pending' && !review.stale && !finalizing && Boolean(onSelect)
  )

  useEffect(() => {
    setDraftTargets(cloneTargets(resetTargetsRef.current))
    setAcknowledgedUnevaluated(false)
    setAcknowledgedUnverified(false)
    setDialog(null)
  }, [
    selected?.candidate_id,
    selected?.bundle.revision,
    persistedDraft?.revision,
    review?.status,
  ])

  useEffect(() => {
    setInspectedCandidateId((current) => {
      if (serverSelectionAuthoritative) return selected?.candidate_id ?? null
      return candidates.some((candidate) => candidate.candidate_id === current)
        ? current
        : selected?.candidate_id ?? null
    })
  }, [candidates, selected?.candidate_id, serverSelectionAuthoritative])

  useEffect(() => {
    if (finalizing) {
      setDialog(null)
      return
    }
    if (review?.stale) {
      setDialog((current) => current === 'dismiss' ? current : null)
      setAcknowledgedUnevaluated(false)
      setAcknowledgedUnverified(false)
    }
  }, [finalizing, review?.stale])

  useEffect(() => onDirtyChange?.(dirty), [dirty, onDirtyChange])
  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange])

  if (!result) return null
  if (!isSuccessfulResult(result)) return <EmptyResult result={result} />
  if (result.reason === NO_VALID_PROPOSAL_REASON) {
    return <NoValidProposalResult result={result} />
  }
  if (candidates.length === 0) return <BaselineResult result={result} />
  if (!selected) {
    return (
      <section aria-label="Reflection review" className="rounded-lg border border-red-200 bg-red-50 p-4">
        <h4 className="text-sm font-semibold text-red-900">Selected candidate evidence unavailable</h4>
        <p className="mt-1 text-sm text-red-700">
          The persisted candidate ID does not match this run's immutable reflection evidence. Review decisions are disabled.
        </p>
      </section>
    )
  }
  const inspectedCandidate = candidates.find(
    (candidate) => candidate.candidate_id === inspectedCandidateId,
  )
  const activeCandidate = serverSelectionAuthoritative
    ? selected
    : inspectedCandidate ?? selected
  const activeCandidateIsSelected = activeCandidate.candidate_id === selected.candidate_id
  const editable = activeCandidateIsSelected && review?.status === 'pending' &&
    !finalizing

  const activeDraft = persistedDraft?.candidate_id === activeCandidate.candidate_id
    ? persistedDraft
    : null
  const savedDraft = activeDraft != null && activeDraft.revision !== activeCandidate.bundle.revision
  const unsaved = dirty
  const showEditedDraft = activeCandidateIsSelected && (dirty || savedDraft)
  const actionMismatchPaths = editable && dirty
    ? actionSetMismatches(result.baseline, activeCandidate.bundle, draftTargets)
    : []
  const availability = promotionAvailability(run, busy)
  const receipt = review?.receipt

  async function perform(action: () => MaybePromise) {
    setBusy(true)
    setError(null)
    try {
      await action()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The review action failed.')
    } finally {
      setBusy(false)
    }
  }

  function choose(candidate: ReflectionCandidate) {
    if (candidate.candidate_id === activeCandidate.candidate_id || busy) return
    if (!serverSelectionAuthoritative) {
      setInspectedCandidateId(candidate.candidate_id)
      return
    }
    if (!onSelect || finalizing || review?.stale) return
    if (savedDraft || unsaved) {
      setDialog({ selectCandidate: candidate.candidate_id })
      return
    }
    void perform(() => onSelect(candidate.candidate_id, false))
  }

  function resetDraft() {
    if (persistedDraft && onResetDraft) {
      void perform(async () => {
        await onResetDraft()
      })
      return
    }
    setDraftTargets(cloneTargets(activeCandidate.bundle.targets))
  }

  return (
    <section aria-label="Reflection review" className="space-y-5">
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
        <span className="rounded-full bg-indigo-100 px-2.5 py-1 text-xs font-semibold text-indigo-800">
          {statusLabel(run)}
        </span>
        {finalizing && (
          <p role="status" className="mt-3 text-sm text-gray-600">
            Finalizing review evidence. Editing and promotion will unlock when the run completes.
          </p>
        )}
        {result.baseline_won && result.provisional_candidate_id && (
          <p className="mt-3 text-sm text-amber-800">
            B scored better in predicted evaluation, but paired sandbox verification did not confirm it. You can review, edit, and promote it with an explicit acknowledgement.
          </p>
        )}
        {result.baseline_won && !result.provisional_candidate_id && (
          <p className="mt-3 text-sm text-gray-600">
            A scored best. Generated alternatives remain visible below as read-only evaluation evidence.
          </p>
        )}
        {review?.stale && (
          <p className="mt-3 text-sm text-amber-900">Promotion is blocked because evaluated A no longer matches the managed files.</p>
        )}
      </div>

      {result.challenge && <PairedChallengeEvidence challenge={result.challenge} />}

      {review?.stale && (
        <StaleEvidence
          past={result.baseline}
          current={review.current}
          reason={review.stale_reason}
          changedTargets={review.changed_targets}
        />
      )}

      <nav aria-label="Reflection decision sequence" className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
        <span>1. A evaluated</span><span aria-hidden="true">→</span>
        <span>2. B evaluated</span><span aria-hidden="true">→</span>
        <span>3. Review or edit C</span><span aria-hidden="true">→</span>
        <a className="font-medium text-blue-700 hover:underline" href="#reflection-decision">4. Decide</a>
      </nav>

      {candidates.length > 1 && (
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
            Evaluated proposals
          </h4>
          <div className="flex flex-wrap gap-2" role="group" aria-label="Evaluated proposals">
            {candidates.map((candidate, index) => (
              <button
                key={candidate.candidate_id}
                type="button"
                aria-pressed={candidate.candidate_id === activeCandidate.candidate_id}
                disabled={busy}
                onClick={() => choose(candidate)}
                className={`rounded-lg border px-3 py-2 text-left text-xs disabled:opacity-50 ${
                  candidate.candidate_id === activeCandidate.candidate_id
                    ? 'border-indigo-400 bg-indigo-50 text-indigo-900'
                    : 'border-gray-200 text-gray-700 hover:bg-gray-50'
                }`}
              >
                <span className="font-medium">
                  Proposal {index + 1} · attempt {candidate.generation_attempt_id} · {candidate.resolved_writer_model}
                </span>
                <span className="ml-2 font-mono">
                  {candidate.score_delta >= 0 ? '+' : ''}{candidate.score_delta.toFixed(3)}
                </span>
                <span className="ml-2 font-mono text-[0.625rem] text-gray-500">{candidate.candidate_id}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <ReflectionScoreComparison
        result={result}
        candidate={activeCandidate}
        hasEditedDraft={showEditedDraft && (savedDraft || unsaved)}
      />

      <EvaluatorAssessment baseline={result.baseline_evaluation} candidate={activeCandidate} />

      <section className="rounded-lg border border-gray-200 p-4">
        <h4 className="text-sm font-semibold text-gray-900">Writer provenance</h4>
        <dl className="mt-2 grid gap-2 text-xs sm:grid-cols-2">
          <div><dt className="text-gray-500">Generation attempt</dt><dd className="font-mono">{activeCandidate.generation_attempt_id}</dd></div>
          <div><dt className="text-gray-500">Candidate ID</dt><dd className="break-all font-mono">{activeCandidate.candidate_id}</dd></div>
          <div><dt className="text-gray-500">Requested writer</dt><dd>{activeCandidate.requested_writer.id} · {activeCandidate.requested_writer.family} · {activeCandidate.requested_writer.provider}</dd></div>
          <div><dt className="text-gray-500">Resolved writer</dt><dd>{activeCandidate.resolved_writer_model} · {activeCandidate.resolved_writer_family} · {activeCandidate.resolved_writer_backend}</dd></div>
        </dl>
      </section>

      <BundleComparison
        past={result.baseline}
        proposed={activeCandidate.bundle}
        editedTargets={editable || savedDraft ? draftTargets : null}
        onEditedTargetsChange={editable ? setDraftTargets : undefined}
      />

      {receipt ? (
        <PromotionReceiptView receipt={receipt} />
      ) : review?.status === 'pending' && !finalizing && activeCandidateIsSelected && (
        <ReflectionDecisionControls
          availability={availability}
          hasSavedDraft={savedDraft}
          hasUnsavedChanges={unsaved}
          actionMismatchPaths={actionMismatchPaths}
          busy={busy}
          canDismiss={run.status === 'complete' && review?.status === 'pending'}
          acknowledgedUnevaluated={acknowledgedUnevaluated}
          acknowledgedUnverified={acknowledgedUnverified}
          onAcknowledgedUnevaluatedChange={setAcknowledgedUnevaluated}
          onAcknowledgedUnverifiedChange={setAcknowledgedUnverified}
          onSaveDraft={() => {
            if (onSaveDraft) void perform(() => onSaveDraft(contents(draftTargets)))
          }}
          onResetDraft={resetDraft}
          onPromote={() => setDialog('promote')}
          onDismiss={() => setDialog('dismiss')}
        />
      )}

      {review?.status === 'dismissed' && (
        <p className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-sm text-gray-600">
          This proposal was dismissed without changing managed instructions.
        </p>
      )}
      {error && <p role="alert" className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</p>}

      {dialog === 'promote' && (
        <Dialog
          title={savedDraft
            ? 'Promote unevaluated C?'
            : availability.requiresUnverifiedAcknowledgement
              ? 'Promote unverified B?'
              : 'Promote evaluated B?'}
          busy={busy}
          onClose={() => setDialog(null)}
        >
          <p className="mt-3 text-sm text-gray-600">
            The server will first confirm every managed instruction still matches evaluated A.
          </p>
          <button
            type="button"
            disabled={busy}
            onClick={() => void perform(async () => {
              if (onPromote) await onPromote(savedDraft, acknowledgedUnverified)
              setDialog(null)
            })}
            className="mt-4 rounded-lg bg-green-700 px-3 py-2 text-sm font-semibold text-white disabled:bg-gray-300"
          >
            Confirm promotion
          </button>
        </Dialog>
      )}
      {dialog === 'dismiss' && (
        <Dialog title="Dismiss this proposal?" busy={busy} onClose={() => setDialog(null)}>
          {unsaved && (
            <p className="mt-3 text-sm font-medium text-amber-900">
              Dismissal will discard your unsaved edited C.
            </p>
          )}
          <p className="mt-3 text-sm text-gray-600">No managed files will be changed.</p>
          <button
            type="button"
            disabled={busy}
            onClick={() => void perform(async () => {
              if (onDismiss) await onDismiss()
              setDraftTargets(cloneTargets(activeCandidate.bundle.targets))
              setAcknowledgedUnevaluated(false)
              setAcknowledgedUnverified(false)
              setDialog(null)
            })}
            className="mt-4 rounded-lg bg-gray-800 px-3 py-2 text-sm font-semibold text-white disabled:bg-gray-300"
          >
            Confirm dismissal
          </button>
        </Dialog>
      )}
      {dialog && typeof dialog === 'object' && 'selectCandidate' in dialog && (
        <Dialog title="Discard edited C?" busy={busy} onClose={() => setDialog(null)}>
          <p className="mt-3 text-sm text-gray-600">
            Selecting another evaluated proposal discards the current edited C.
          </p>
          <button
            type="button"
            disabled={busy}
            onClick={() => void perform(async () => {
              if (onSelect) await onSelect(dialog.selectCandidate, true)
              setDialog(null)
            })}
            className="mt-4 rounded-lg bg-red-700 px-3 py-2 text-sm font-semibold text-white disabled:bg-gray-300"
          >
            Discard C and select proposal
          </button>
        </Dialog>
      )}
    </section>
  )
}
