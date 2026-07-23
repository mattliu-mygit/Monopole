import type { ReflectionCandidate, SuccessfulReflectionResult } from '../../types'

function score(value: number): string {
  return value.toFixed(3)
}

function evaluatorNames(candidate: ReflectionCandidate): string {
  return candidate.evaluation.resolved_model
}

export default function ReflectionScoreComparison({
  result,
  candidate,
  hasEditedDraft,
}: {
  result: SuccessfulReflectionResult
  candidate: ReflectionCandidate
  hasEditedDraft: boolean
}) {
  return (
    <section aria-label="Reflection score comparison" className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-[1fr_auto_1fr] sm:items-stretch">
        <div className="rounded-lg border border-gray-200 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Past (A)</div>
          <div className="mt-2 font-mono text-2xl font-semibold text-gray-900">
            {score(result.baseline_score)}
          </div>
          <div className="mt-1 text-xs text-gray-500">Evaluated bundle {result.baseline.revision}</div>
        </div>
        <div className="flex items-center justify-center">
          <span className={`rounded-full px-3 py-1 font-mono text-sm font-semibold ${
            candidate.score_delta >= 0
              ? 'bg-green-100 text-green-800'
              : 'bg-red-100 text-red-800'
          }`}>
            {candidate.score_delta >= 0 ? '+' : ''}{score(candidate.score_delta)}
          </span>
        </div>
        <div className="rounded-lg border border-indigo-200 bg-indigo-50/40 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-indigo-700">
            Proposed (B)
          </div>
          <div className="mt-2 font-mono text-2xl font-semibold text-gray-900">
            {score(candidate.score)}
          </div>
          <div className="mt-1 text-xs text-gray-600">Evaluated by {evaluatorNames(candidate)}</div>
        </div>
      </div>

      {hasEditedDraft && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          Edited proposal C is not evaluated. The scores above still compare evaluated B with A.
        </div>
      )}
    </section>
  )
}
