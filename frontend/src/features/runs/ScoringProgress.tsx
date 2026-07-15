import type {
  ScoringProgress as ScoringProgressData,
  ScoringResult,
  ScoringTurnDetail,
} from '../../types'

export interface ScoringProgressProps {
  progress: ScoringProgressData | null
  result: ScoringResult | null
}

function plural(value: number, singular: string, pluralValue = `${singular}s`): string {
  return `${value} ${value === 1 ? singular : pluralValue}`
}

function TurnEvidence({ detail }: { detail: ScoringTurnDetail }) {
  return (
    <details className="rounded-lg border border-gray-200 bg-white text-xs">
      <summary className="cursor-pointer px-3 py-2 hover:bg-gray-50">
        <span className="font-mono text-gray-700">{detail.trace_id}</span>
        <span className="ml-3 text-gray-500">
          {Object.keys(detail.scores).length} score{Object.keys(detail.scores).length === 1 ? '' : 's'}
        </span>
      </summary>
      <div className="space-y-2 border-t bg-gray-50 px-3 py-2">
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-gray-500">
          <span>session: <code>{detail.conversation_id}</code></span>
          <span>model: {detail.model ?? '—'}</span>
          {detail.tokens !== undefined && <span>{detail.tokens.toLocaleString()} tokens</span>}
          {detail.tool_count !== undefined && <span>{plural(detail.tool_count, 'tool call')}</span>}
          {detail.errors !== undefined && detail.errors > 0 && (
            <span className="text-red-600">{plural(detail.errors, 'error')}</span>
          )}
        </div>
        {detail.user_input && <p className="text-gray-600">{detail.user_input}</p>}
        <dl className="space-y-1">
          {Object.entries(detail.scores).map(([scorer, score]) => (
            <div key={scorer} className="flex items-center justify-between gap-3">
              <dt className="font-medium text-gray-700">{scorer}</dt>
              <dd className="font-mono tabular-nums text-gray-800">{score.toFixed(2)}</dd>
            </div>
          ))}
        </dl>
      </div>
    </details>
  )
}

export default function ScoringProgress({
  progress,
  result,
}: ScoringProgressProps) {
  if (!progress && !result) {
    return (
      <p role="status" className="text-sm text-gray-500">
        Waiting for scoring to start.
      </p>
    )
  }

  const details = result?.turn_details ?? progress?.turn_details ?? []
  const percent = progress && progress.total > 0
    ? Math.min(100, Math.max(0, Math.round((progress.scored / progress.total) * 100)))
    : 0

  return (
    <section aria-label="Scoring progress" className="space-y-4">
      {progress && !result && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p role="status" aria-live="polite" className="text-sm font-medium text-gray-800">
              {progress.status_message ?? 'Scoring turns...'}
            </p>
            <span className="text-xs tabular-nums text-gray-500">
              {progress.scored} of {progress.total} turns scored · {progress.written} scores written
            </span>
          </div>
          <div
            role="progressbar"
            aria-label="Turns scored"
            aria-valuemin={0}
            aria-valuemax={progress.total}
            aria-valuenow={progress.scored}
            className="h-2 overflow-hidden rounded-full bg-gray-200"
          >
            <div
              className="h-full rounded-full bg-blue-500 transition-all"
              style={{ width: `${percent}%` }}
            />
          </div>
        </div>
      )}

      {result && (
        <div className="grid gap-2 text-sm text-gray-700 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-lg bg-gray-50 p-3 font-medium">
            {plural(result.turns_scored, 'turn')} scored
          </div>
          <div className="rounded-lg bg-gray-50 p-3 font-medium">
            {plural(result.sessions_scored, 'session')} scored
          </div>
          <div className="rounded-lg bg-gray-50 p-3 font-medium">
            {plural(result.scores_written, 'score')} written
          </div>
          <div className={`rounded-lg p-3 font-medium ${result.errors > 0 ? 'bg-red-50 text-red-700' : 'bg-gray-50'}`}>
            {plural(result.errors, 'error')}
          </div>
        </div>
      )}

      {details.length > 0 && (
        <div>
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-500">
            Scored turn evidence
          </h4>
          <div className="max-h-96 space-y-2 overflow-y-auto">
            {details.map((detail) => <TurnEvidence key={detail.trace_id} detail={detail} />)}
          </div>
        </div>
      )}
    </section>
  )
}
