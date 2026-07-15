import { useEffect, useState } from 'react'
import type { ReflectingProgress } from '../../types'

function durationLabel(start: string, endMs: number): string | null {
  const startMs = Date.parse(start)
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return null

  const totalSeconds = Math.max(0, Math.floor((endMs - startMs) / 1000))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  if (hours > 0) return `${hours}h ${minutes}m`
  if (minutes > 0) return `${minutes}m ${seconds}s`
  return `${seconds}s`
}

function eventTimeLabel(start: string, at: string): string {
  const label = durationLabel(start, Date.parse(at))
  return label ? `+${label}` : ''
}

export default function ReflectionActivity({
  progress,
  active,
}: {
  progress: ReflectingProgress
  active: boolean
}) {
  const [showAll, setShowAll] = useState(false)
  const [nowMs, setNowMs] = useState(() => Date.now())

  useEffect(() => {
    if (!active) return

    setNowMs(Date.now())
    const interval = window.setInterval(() => setNowMs(Date.now()), 1_000)
    return () => window.clearInterval(interval)
  }, [active])

  const events = progress.events
  const visibleEvents = showAll ? events : events.slice(-5)
  const lastEvent = events.at(-1)
  const currentMessage = progress.status_message
  const elapsedEnd = active ? nowMs : Date.parse(lastEvent?.at ?? '')
  const elapsed = durationLabel(progress.started_at, elapsedEnd)

  const total = progress.total_attempts
  const attempted = progress.attempted
  const valid = progress.valid
  const rejected = progress.rejected
  const scored = progress.scored
  const percent = total > 0 ? Math.min(100, Math.round((attempted / total) * 100)) : 0

  return (
    <section className="mb-4 space-y-3" aria-label="Reflection progress">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2" aria-live="polite">
          {active ? (
            <span
              className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-blue-200 border-t-blue-600 motion-reduce:animate-none"
              aria-hidden="true"
            />
          ) : (
            <span className="h-2.5 w-2.5 shrink-0 rounded-full bg-gray-400" aria-hidden="true" />
          )}
          <span className="text-sm font-medium text-gray-800">{currentMessage}</span>
        </div>
        {elapsed && <span className="text-xs tabular-nums text-gray-500">{elapsed}</span>}
      </div>

      {total > 0 && (
        <div>
          <div
            className="h-2 overflow-hidden rounded-full bg-gray-200"
            role="progressbar"
            aria-label="Proposal attempts"
            aria-valuemin={0}
            aria-valuemax={total}
            aria-valuenow={attempted}
          >
            <div
              className="h-full rounded-full bg-blue-500 transition-all"
              style={{ width: `${percent}%` }}
            />
          </div>
          <div className="mt-1 text-xs text-gray-500">
            {attempted} of {total} proposal attempts · {valid} valid · {rejected} rejected ·{' '}
            {scored} scored
          </div>
        </div>
      )}

      <ol
        id="reflection-activity-events"
        className="space-y-2"
        aria-label="Reflection activity"
      >
        {visibleEvents.map((event) => {
          const isCurrent = active && event.id === lastEvent?.id
          const isRejected = event.phase === 'candidate_rejected'
          const metadata = [
            event.candidate != null
              ? `${isRejected ? 'Proposal attempt' : 'Candidate'} ${event.candidate}`
              : null,
            event.model || null,
            event.acting_role ? event.acting_role.replaceAll('_', ' ') : null,
            event.attempt_id ? `attempt ${event.attempt_id}` : null,
            event.evaluation_id ? `evaluation ${event.evaluation_id}` : null,
            event.status || null,
            event.error_type || null,
            event.score != null ? `Predicted evaluator score ${event.score.toFixed(3)}` : null,
          ].filter((value): value is string => value != null)
          const rejectedExcerpt = event.response_excerpt?.slice(0, 1_000)
          return (
            <li
              key={event.id}
              className="grid grid-cols-[0.75rem_minmax(0,1fr)_auto] items-start gap-2 text-xs"
              aria-current={isCurrent ? 'step' : undefined}
            >
              <span
                className={`mt-1 h-2 w-2 rounded-full ${
                  isCurrent ? 'bg-blue-500 ring-4 ring-blue-50' : 'bg-gray-300'
                }`}
                aria-hidden="true"
              />
              <div className={isCurrent ? 'text-gray-800' : 'text-gray-600'}>
                <div>{event.message}</div>
                {metadata.length > 0 && (
                  <div className="mt-0.5 flex flex-wrap gap-1 text-[0.6875rem] text-gray-400">
                    {metadata.map((item) => (
                      <span key={item} className="rounded bg-gray-100 px-1.5 py-0.5">
                        {item}
                      </span>
                    ))}
                  </div>
                )}
                {event.changed_paths && event.changed_paths.length > 0 && (
                  <div
                    className="mt-1 flex flex-wrap gap-1"
                    aria-label={isRejected ? 'Rejected paths' : 'Changed paths'}
                  >
                    {event.changed_paths.map((path) => (
                      <code
                        key={path}
                        className={`rounded border px-1.5 py-0.5 text-[0.6875rem] ${
                          isRejected
                            ? 'border-red-100 bg-red-50 text-red-800'
                            : 'border-gray-200 bg-gray-50 text-gray-700'
                        }`}
                      >
                        {path}
                      </code>
                    ))}
                  </div>
                )}
                {isRejected && (event.response_digest || rejectedExcerpt) && (
                  <details className="mt-1 rounded border border-gray-200 bg-gray-50 p-2">
                    <summary className="cursor-pointer font-medium text-gray-700">
                      Show rejected output
                    </summary>
                    {event.response_digest && (
                      <div className="mt-2 break-all text-[0.6875rem] text-gray-500">
                        Response digest: <code>{event.response_digest}</code>
                      </div>
                    )}
                    {rejectedExcerpt && (
                      <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded bg-gray-950 p-2 text-[0.6875rem] text-gray-100">
                        {rejectedExcerpt}
                      </pre>
                    )}
                  </details>
                )}
              </div>
              <time className="font-mono text-gray-400" dateTime={event.at}>
                {eventTimeLabel(progress.started_at, event.at)}
              </time>
            </li>
          )
        })}
      </ol>

      {events.length > 5 && (
        <button
          type="button"
          className="text-xs font-medium text-blue-600 hover:text-blue-800"
          onClick={() => setShowAll((value) => !value)}
          aria-expanded={showAll}
          aria-controls="reflection-activity-events"
        >
          {showAll ? 'Show latest 5' : 'Show all activity'}
        </button>
      )}
    </section>
  )
}
